from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from pathlib import Path
from typing import Any

import chromadb

from core.llm import get_openai_client
from core.global_index import write_global_index_entry
from core.metadata import load_metadata, mark_summary_complete, update_summary_progress
from core.paths import compute_document_id, ensure_directory
from core.storage import read_json, write_json

logger = logging.getLogger(__name__)

_MAX_CHARS_PER_CALL = 50_000
_DEFAULT_RETRY_ATTEMPTS = 5
_DEFAULT_RETRY_BASE_SECONDS = 2.0


def _is_rate_limited_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc)
    return status_code == 429 or "RESOURCE_EXHAUSTED" in text or "429" in text


def _get_chroma_client(config: dict[str, Any]) -> chromadb.ClientAPI:
    chroma_host = config["vector_db"].get("host")
    if chroma_host:
        return chromadb.HttpClient(host=chroma_host, port=int(config["vector_db"].get("port", 8000)))
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


def _tree_summarize_with_fallback(
    text_chunks: list[str],
    query: str,
    client: Any,
    primary_model: str,
    fallback_model: str,
    retry_attempts: int = _DEFAULT_RETRY_ATTEMPTS,
    retry_base_seconds: float = _DEFAULT_RETRY_BASE_SECONDS,
) -> str:
    models = [primary_model]
    if fallback_model and fallback_model != primary_model:
        models.append(fallback_model)

    last_exc: Exception | None = None
    for model in models:
        for attempt in range(1, max(1, retry_attempts) + 1):
            try:
                if model != primary_model and attempt == 1:
                    logger.warning(
                        "Summarizer model '%s' failed earlier. Retrying with fallback chat model '%s'.",
                        primary_model,
                        fallback_model,
                    )
                return _tree_summarize(text_chunks, query, client, model)
            except Exception as exc:
                last_exc = exc
                if _is_rate_limited_error(exc) and attempt < retry_attempts:
                    sleep_s = retry_base_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Summarizer hit rate limit (429). Backing off %.1fs (attempt %d/%d, model=%s).",
                        sleep_s,
                        attempt,
                        retry_attempts,
                        model,
                    )
                    time.sleep(sleep_s)
                    continue
                if not _is_rate_limited_error(exc):
                    break
    if last_exc:
        raise last_exc
    raise RuntimeError("Summarization failed without exception details.")


def _summary_model(config: dict[str, Any]) -> str:
    model = (config.get("summarizer", {}) or {}).get("model")
    if model:
        return model
    return config["chat"]["model"]


def _section_slug(section: str) -> str:
    return (section or "").lower().replace(" ", "_")[:40]


def _merge_summaries_into_cards(document_folder: Path, config: dict[str, Any]) -> None:
    """Merge generated summaries back into document/section card files.

    Embeddings remain in Chroma. We only store summary text and summary vector IDs.
    """
    meta = load_metadata(document_folder)
    document_id = meta.get("document_id") or compute_document_id(
        meta.get("document_name", document_folder.name),
        meta.get("source_url", ""),
    )

    level1_data = read_json(document_folder / "summaries" / "level1_onepager.json", default={})
    level3_data = read_json(document_folder / "summaries" / "level3_detailed.json", default={"sections": []})
    l1_summary = (level1_data.get("summary", "") or "").strip()

    # Merge L1 into document card in metadata.
    doc_card = dict(meta.get("document_card", {}) or {})
    if l1_summary:
        doc_card["l1_summary"] = l1_summary
        doc_card["summary_card_id"] = f"{document_id}_level1"
    if doc_card:
        meta["document_card"] = doc_card
        from core.metadata import save_metadata
        save_metadata(document_folder, meta)

    # Merge L3 summaries into section cards index.
    section_index_path = document_folder / "sections" / "index.json"
    section_index = read_json(section_index_path, default={"sections": []})
    sections = section_index.get("sections", [])
    if not isinstance(sections, list) or not sections:
        return

    l3_by_heading: dict[str, str] = {}
    for entry in level3_data.get("sections", []):
        heading = (entry.get("section", "") or "").strip()
        text = (entry.get("summary", "") or "").strip()
        if heading and text:
            l3_by_heading[heading] = text

    changed = False
    for section in sections:
        heading = (section.get("heading", "") or section.get("section_path", "") or "").strip()
        l3 = l3_by_heading.get(heading, "")
        if not l3:
            continue
        section["l3_summary"] = l3
        section["summary_card_id"] = f"{document_id}_{_section_slug(heading)}_level3"
        changed = True

    if changed:
        write_json(section_index_path, section_index)


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

    client = get_openai_client("summarizer")
    model = _summary_model(config)
    fallback_model = config["chat"]["model"]
    summarizer_cfg = config.get("summarizer", {}) or {}
    retry_attempts = int(summarizer_cfg.get("retry_attempts", _DEFAULT_RETRY_ATTEMPTS))
    retry_base_seconds = float(summarizer_cfg.get("retry_base_seconds", _DEFAULT_RETRY_BASE_SECONDS))
    query = (
        "Summarize this section in detail. Preserve all numerical values, financial figures, "
        "dates, named entities, and key findings verbatim. Write in prose."
    )
    lock = threading.Lock()

    def _summarize_one(item: tuple[str, list[str]]) -> None:
        h1, texts = item
        try:
            summary = _tree_summarize_with_fallback(
                texts,
                query,
                client,
                model,
                fallback_model,
                retry_attempts=retry_attempts,
                retry_base_seconds=retry_base_seconds,
            )
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
        max_parallel = int(summarizer_cfg.get("max_parallel_sections", 1))
        max_workers = max(1, min(max_parallel, len(pending)))
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


_LEVEL_QUERIES = {
    1: ("Write an executive summary in 250 words or fewer. "
        "State what this document is about, then list the 5 most important findings with specific numbers."),
    2: ("Write a structured 4-6 paragraph summary of this document covering all major sections. "
        "Retain all key numbers, findings, and use section headers."),
}

_LEVEL_OUTPUT_FILES = {1: "level1_onepager.json", 2: "level2_medium.json"}


def _run_lower_level(
    document_folder: Path, config: dict[str, Any], *,
    level: int, input_texts: list[str],
) -> None:
    """Generate L1 or L2 summary from already-prepared input texts."""
    client = get_openai_client("summarizer")
    primary_model = _summary_model(config)
    fallback_model = config["chat"]["model"]
    summarizer_cfg = config.get("summarizer", {}) or {}
    retry_attempts = int(summarizer_cfg.get("retry_attempts", _DEFAULT_RETRY_ATTEMPTS))
    retry_base_seconds = float(summarizer_cfg.get("retry_base_seconds", _DEFAULT_RETRY_BASE_SECONDS))
    summary = _tree_summarize_with_fallback(
        input_texts,
        _LEVEL_QUERIES[level],
        client,
        primary_model,
        fallback_model,
        retry_attempts=retry_attempts,
        retry_base_seconds=retry_base_seconds,
    )
    write_json(document_folder / "summaries" / _LEVEL_OUTPUT_FILES[level], {"summary": summary})
    update_summary_progress(document_folder, f"level{level}_complete", True)


def _run_level2(document_folder: Path, config: dict[str, Any]) -> None:
    level3 = read_json(document_folder / "summaries" / "level3_detailed.json", default={"sections": []})
    texts = [s["summary"] for s in level3.get("sections", []) if s.get("summary")] or ["No content available."]
    _run_lower_level(document_folder, config, level=2, input_texts=texts)


def _run_level1(document_folder: Path, config: dict[str, Any]) -> None:
    level2 = read_json(document_folder / "summaries" / "level2_medium.json", default={})
    text = level2.get("summary", "") or "No content available."
    _run_lower_level(document_folder, config, level=1, input_texts=[text])


def _index_summaries(document_folder: Path, level: int, config: dict[str, Any]) -> None:
    openai_client = get_openai_client("summarizer")
    embedding_model = config["embeddings"]["model"]
    collection_name = config["vector_db"].get("summary_collection_name", "pdf_rag_summaries")
    chroma_client = _get_chroma_client(config)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    metadata_doc = load_metadata(document_folder)
    doc_name = metadata_doc.get("document_name", document_folder.name)
    document_id = metadata_doc.get("document_id") or compute_document_id(doc_name, metadata_doc.get("source_url", ""))
    folder_str = str(document_folder)

    if level == 1:
        data = read_json(document_folder / "summaries" / "level1_onepager.json", default={})
        text = data.get("summary", "")
        if not text:
            return
        resp = openai_client.embeddings.create(model=embedding_model, input=[text])
        collection.upsert(
            ids=[f"{document_id}_level1"],
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
        for entry in data.get("sections", []):
            section = entry.get("section", "")
            text = entry.get("summary", "")
            if not text:
                continue
            section_slug = section.lower().replace(" ", "_")[:40]
            resp = openai_client.embeddings.create(model=embedding_model, input=[text])
            collection.upsert(
                ids=[f"{document_id}_{section_slug}_level3"],
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

        # Phase 4: merge L1/L3 summaries back into cards metadata/index.
        _merge_summaries_into_cards(document_folder, config)

        summary_paths = {
            "level3": str(document_folder / "summaries" / "level3_detailed.json"),
            "level2": str(document_folder / "summaries" / "level2_medium.json"),
            "level1": str(document_folder / "summaries" / "level1_onepager.json"),
        }
        level1_data = read_json(document_folder / "summaries" / "level1_onepager.json", default={})
        mark_summary_complete(document_folder, summary_paths, summary_brief=level1_data.get("summary", ""))
        # Keep global index in sync with summary progression (Phase 4).
        meta_after = load_metadata(document_folder)
        write_global_index_entry(
            Path(config["paths"]["artifacts_root"]) / "metadata.json",
            document_folder,
            {
                "summary_status": meta_after.get("summary_status", "pending"),
                "summary_ready": meta_after.get("summary_ready", False),
                "l1_summary": level1_data.get("summary", ""),
                "document_card": meta_after.get("document_card", {}),
                "section_cards_path": meta_after.get("section_cards_path", ""),
            },
        )
        logger.info("Summarization complete for %s", document_folder.name)

    except Exception:
        meta = load_metadata(document_folder)
        meta["summary_status"] = "error"
        from core.metadata import save_metadata
        save_metadata(document_folder, meta)
        logger.exception("Summarization failed for %s", document_folder.name)
        raise
