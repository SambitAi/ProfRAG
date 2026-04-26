from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
from pathlib import Path
from typing import Any

import chromadb

from core.llm import get_openai_client
from core.metadata import load_metadata, mark_summary_complete, update_summary_progress
from core.paths import ensure_directory
from core.storage import read_json, write_json

logger = logging.getLogger(__name__)

_MAX_CHARS_PER_CALL = 50_000


def _get_chroma_client(config: dict[str, Any]) -> chromadb.ClientAPI:
    chroma_host = os.getenv("CHROMA_HOST")
    if chroma_host:
        return chromadb.HttpClient(host=chroma_host, port=int(os.getenv("CHROMA_PORT", "8000")))
    persist_dir = config["vector_db"]["persist_directory"]
    ensure_directory(persist_dir)
    return chromadb.PersistentClient(path=str(persist_dir))


def _tree_summarize(text_chunks: list[str], query: str, client: Any, model: str) -> str:
    """Recursive bottom-up summarization. Mirrors TreeSummarize without LlamaIndex."""
    chunks = [c for c in text_chunks if c.strip()]
    if not chunks:
        return ""

    while True:
        batches: list[list[str]] = []
        current: list[str] = []
        current_len = 0
        for chunk in chunks:
            if current_len + len(chunk) > _MAX_CHARS_PER_CALL and current:
                batches.append(current)
                current = []
                current_len = 0
            current.append(chunk)
            current_len += len(chunk)
        if current:
            batches.append(current)

        if len(batches) == 1:
            break

        # Intermediate pass: compress each batch
        chunks = []
        for batch in batches:
            resp = client.chat.completions.create(
                model=model, temperature=0,
                messages=[
                    {"role": "system", "content": query},
                    {"role": "user", "content": "\n\n".join(batch)},
                ],
            )
            chunks.append(resp.choices[0].message.content or "")

    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[
            {"role": "system", "content": query},
            {"role": "user", "content": "\n\n".join(chunks)},
        ],
    )
    return resp.choices[0].message.content or ""


def _group_chunks_by_h1(document_folder: Path) -> dict[str, list[str]]:
    metadata = load_metadata(document_folder)
    chunk_paths = metadata.get("chunk_paths", [])
    sections: dict[str, list[str]] = {}
    for chunk_path in chunk_paths:
        chunk = read_json(chunk_path, default={})
        text = chunk.get("text", "").strip()
        if not text:
            continue
        h1 = chunk.get("h1") or "Document"
        sections.setdefault(h1, []).append(text)
    return sections


def _run_level3(document_folder: Path, config: dict[str, Any], progress: dict) -> None:
    sections = _group_chunks_by_h1(document_folder)
    if not sections:
        update_summary_progress(document_folder, "level3_complete", True)
        return

    summaries_dir = document_folder / "summaries"
    ensure_directory(summaries_dir)
    level3_path = summaries_dir / "level3_detailed.json"

    done_sections: set[str] = set(progress.get("level3_sections_done", []))
    pending = [(h1, texts) for h1, texts in sections.items() if h1 not in done_sections]

    if not pending:
        update_summary_progress(document_folder, "level3_complete", True)
        return

    client = get_openai_client()
    model = config["summarizer"]["model"]
    query = (
        "Summarize this section in detail. Preserve all numerical values, financial figures, "
        "dates, named entities, and key findings verbatim. Write in prose."
    )
    lock = threading.Lock()

    def _summarize_one(item: tuple[str, list[str]]) -> None:
        h1, texts = item
        try:
            summary = _tree_summarize(texts, query, client, model)
        except Exception:
            logger.exception("Level 3 summarization failed for section '%s'", h1)
            raise
        with lock:
            current = read_json(level3_path, default={"sections": []})
            current["sections"].append({"section": h1, "summary": summary})
            write_json(level3_path, current)
            done_sections.add(h1)
            update_summary_progress(document_folder, "level3_sections_done", list(done_sections))

    try:
        max_workers = min(4, len(pending))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_summarize_one, item): item[0] for item in pending}
            for future in concurrent.futures.as_completed(futures):
                future.result()
    except RuntimeError:
        # Interpreter shutting down (e.g. Streamlit hot-reload) — executor cannot
        # accept new tasks. Fall back to sequential in the current daemon thread.
        for item in pending:
            if item[0] not in done_sections:
                _summarize_one(item)

    update_summary_progress(document_folder, "level3_complete", True)


def _run_level2(document_folder: Path, config: dict[str, Any]) -> None:
    level3_path = document_folder / "summaries" / "level3_detailed.json"
    level3 = read_json(level3_path, default={"sections": []})
    section_texts = [s["summary"] for s in level3.get("sections", []) if s.get("summary")]
    if not section_texts:
        section_texts = ["No content available."]

    client = get_openai_client()
    model = config["summarizer"]["model"]
    query = (
        "Write a structured 4-6 paragraph summary of this document covering all major sections. "
        "Retain all key numbers, findings, and use section headers."
    )
    summary = _tree_summarize(section_texts, query, client, model)
    write_json(document_folder / "summaries" / "level2_medium.json", {"summary": summary})
    update_summary_progress(document_folder, "level2_complete", True)


def _run_level1(document_folder: Path, config: dict[str, Any]) -> None:
    level2 = read_json(document_folder / "summaries" / "level2_medium.json", default={})
    level2_text = level2.get("summary", "") or "No content available."

    client = get_openai_client()
    model = config["summarizer"]["model"]
    query = (
        "Write an executive summary in 250 words or fewer. "
        "State what this document is about, then list the 5 most important findings with specific numbers."
    )
    summary = _tree_summarize([level2_text], query, client, model)
    write_json(document_folder / "summaries" / "level1_onepager.json", {"summary": summary})
    update_summary_progress(document_folder, "level1_complete", True)


def _index_summaries(document_folder: Path, level: int, config: dict[str, Any]) -> None:
    openai_client = get_openai_client()
    embedding_model = config["embeddings"]["model"]
    collection_name = config["vector_db"].get("summary_collection_name", "pdf_rag_summaries")
    chroma_client = _get_chroma_client(config)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    metadata_doc = load_metadata(document_folder)
    doc_name = metadata_doc.get("document_name", document_folder.name)
    folder_str = str(document_folder)

    if level == 1:
        data = read_json(document_folder / "summaries" / "level1_onepager.json", default={})
        text = data.get("summary", "")
        if not text:
            return
        resp = openai_client.embeddings.create(model=embedding_model, input=[text])
        slug = document_folder.name
        collection.upsert(
            ids=[f"{slug}_level1"],
            embeddings=[resp.data[0].embedding],
            documents=[text],
            metadatas=[{
                "document_folder": folder_str,
                "document_name": doc_name,
                "section_name": "",
                "level": "1",
            }],
        )
        update_summary_progress(document_folder, "level1_indexed", True)

    elif level == 3:
        data = read_json(document_folder / "summaries" / "level3_detailed.json", default={"sections": []})
        slug = document_folder.name
        for entry in data.get("sections", []):
            section = entry.get("section", "")
            text = entry.get("summary", "")
            if not text:
                continue
            section_slug = section.lower().replace(" ", "_")[:40]
            resp = openai_client.embeddings.create(model=embedding_model, input=[text])
            collection.upsert(
                ids=[f"{slug}_{section_slug}_level3"],
                embeddings=[resp.data[0].embedding],
                documents=[text],
                metadatas=[{
                    "document_folder": folder_str,
                    "document_name": doc_name,
                    "section_name": section,
                    "level": "3",
                }],
            )
        update_summary_progress(document_folder, "level3_indexed", True)


def run(document_folder: str | Path, config: dict[str, Any]) -> None:
    document_folder = Path(document_folder)
    ensure_directory(document_folder / "summaries")

    metadata = load_metadata(document_folder)
    progress = metadata.get("summary_progress", {})

    update_summary_progress(document_folder, "started", True)

    try:
        if not progress.get("level3_complete"):
            logger.info("Summarizing Level 3 (per-section, parallel) for %s", document_folder.name)
            _run_level3(document_folder, config, progress)
            progress = load_metadata(document_folder).get("summary_progress", {})

        if not progress.get("level3_indexed"):
            logger.info("Indexing Level 3 summaries for %s", document_folder.name)
            _index_summaries(document_folder, 3, config)

        if not progress.get("level2_complete"):
            logger.info("Summarizing Level 2 (medium) for %s", document_folder.name)
            _run_level2(document_folder, config)

        if not progress.get("level1_complete"):
            logger.info("Summarizing Level 1 (1-pager) for %s", document_folder.name)
            _run_level1(document_folder, config)

        if not progress.get("level1_indexed"):
            logger.info("Indexing Level 1 summary for %s", document_folder.name)
            _index_summaries(document_folder, 1, config)

        summary_paths = {
            "level3": str(document_folder / "summaries" / "level3_detailed.json"),
            "level2": str(document_folder / "summaries" / "level2_medium.json"),
            "level1": str(document_folder / "summaries" / "level1_onepager.json"),
        }
        level1_data = read_json(document_folder / "summaries" / "level1_onepager.json", default={})
        mark_summary_complete(document_folder, summary_paths, summary_brief=level1_data.get("summary", ""))
        logger.info("Summarization complete for %s", document_folder.name)

    except Exception:
        meta = load_metadata(document_folder)
        meta["summary_status"] = "error"
        from core.metadata import save_metadata
        save_metadata(document_folder, meta)
        logger.exception("Summarization failed for %s", document_folder.name)
        raise
