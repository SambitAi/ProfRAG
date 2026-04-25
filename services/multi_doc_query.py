from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import chromadb

from core.llm import get_openai_client
from core.metadata import load_metadata
from core.paths import ensure_directory
from core.storage import read_json, write_json
from services import retrieve_context

logger = logging.getLogger(__name__)


def _get_chroma_client(config: dict[str, Any]) -> chromadb.ClientAPI:
    chroma_host = os.getenv("CHROMA_HOST")
    if chroma_host:
        return chromadb.HttpClient(host=chroma_host, port=int(os.getenv("CHROMA_PORT", "8000")))
    persist_dir = config["vector_db"]["persist_directory"]
    ensure_directory(persist_dir)
    return chromadb.PersistentClient(path=str(persist_dir))


def find_relevant_documents(question: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """3-stage funnel: L1 semantic filter → L3 section confirmation (no LLM).

    Returns a ranked list of candidate dicts:
    [{folder, doc_name, top_section, score}]
    """
    client = get_openai_client()
    embedding_model = config["embeddings"]["model"]
    collection_name = config["vector_db"].get("summary_collection_name", "pdf_rag_summaries")

    chroma_client = _get_chroma_client(config)

    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        logger.warning("Summary collection '%s' not found — no summaries indexed yet.", collection_name)
        return []

    resp = client.embeddings.create(model=embedding_model, input=[question])
    query_embedding = resp.data[0].embedding

    # ── Stage 1: Level 1 filter — top-20 unique documents ───────────────────
    stage1_folders: list[str] = []
    try:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=20,
            where={"level": {"$eq": "1"}},
        )
        metas = (result.get("metadatas") or [[]])[0]
        seen: set[str] = set()
        for meta in metas:
            folder = meta.get("document_folder", "")
            if folder and folder not in seen:
                seen.add(folder)
                stage1_folders.append(folder)
    except Exception:
        logger.warning("Stage 1 summary query failed — returning empty candidates.")
        return []

    if not stage1_folders:
        return []

    # ── Stage 2: Level 3 confirmation — top section per document ────────────
    candidates: list[dict[str, Any]] = []
    try:
        where_filter: dict[str, Any]
        if len(stage1_folders) == 1:
            where_filter = {"$and": [
                {"level": {"$eq": "3"}},
                {"document_folder": {"$eq": stage1_folders[0]}},
            ]}
        else:
            where_filter = {"$and": [
                {"level": {"$eq": "3"}},
                {"document_folder": {"$in": stage1_folders}},
            ]}

        result2 = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(10, len(stage1_folders) * 3),
            where=where_filter,
        )
        metas2 = (result2.get("metadatas") or [[]])[0]
        distances2 = (result2.get("distances") or [[]])[0]

        best_per_doc: dict[str, dict[str, Any]] = {}
        for meta, dist in zip(metas2, distances2):
            folder = meta.get("document_folder", "")
            if not folder:
                continue
            score = 1.0 - float(dist)
            if folder not in best_per_doc or score > best_per_doc[folder]["score"]:
                best_per_doc[folder] = {
                    "folder": folder,
                    "doc_name": meta.get("document_name", Path(folder).name),
                    "top_section": meta.get("section_name", ""),
                    "score": score,
                }

        candidates = sorted(best_per_doc.values(), key=lambda x: x["score"], reverse=True)[:5]

    except Exception:
        logger.warning("Stage 2 summary query failed — falling back to Stage 1 results.")
        candidates = [
            {
                "folder": f,
                "doc_name": load_metadata(f).get("document_name", Path(f).name),
                "top_section": "",
                "score": 0.0,
            }
            for f in stage1_folders[:5]
        ]

    return candidates


def ask_across_documents(
    question: str,
    document_folders: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Stage 3: retrieve real chunks from confirmed documents, generate answer with LLM.

    The LLM context contains only real document chunks — no summaries.
    Sources include doc name, section, chunk number, and a text snippet for UI display.
    """
    client = get_openai_client()
    all_context_blocks: list[str] = []
    all_sources: list[dict[str, Any]] = []
    doc_names: list[str] = []

    for folder in document_folders:
        folder_path = Path(folder)
        metadata = load_metadata(folder_path)
        doc_name = metadata.get("document_name", folder_path.name)
        doc_names.append(doc_name)

        retrieval_dir = folder_path / "retrieval"
        ensure_directory(retrieval_dir)
        existing = list(retrieval_dir.glob("multidoc_*.json"))
        retrieval_path = retrieval_dir / f"multidoc_{len(existing) + 1:06d}.json"

        try:
            retrieve_context.run(
                question=question,
                document_folder=folder,
                chroma_persist_dir=config["vector_db"]["persist_directory"],
                collection_name=config["vector_db"]["collection_name"],
                embedding_model=config["embeddings"]["model"],
                retrieval_output_path=retrieval_path,
                top_k=config["retrieval"]["top_k"],
                media_top_k=config["retrieval"].get("media_top_k", 4),
                expand_parent=config["retrieval"].get("expand_parent", True),
            )
        except Exception:
            logger.warning("Retrieval failed for document %s", folder)
            continue

        payload = read_json(retrieval_path, default={})
        texts = payload.get("documents", [])
        metas = payload.get("metadatas", [])

        for i, (text, meta) in enumerate(zip(texts, metas), 1):
            section_path = meta.get("section_path", "")
            chunk_num = meta.get("chunk_number", "?")
            label = f"{doc_name} > {section_path}" if section_path else f"{doc_name}, Chunk {chunk_num}"
            all_context_blocks.append(f"[Source: {label}]\n{text}")
            all_sources.append({
                "document_name": doc_name,
                "document_folder": folder,
                "section_path": section_path,
                "chunk_number": chunk_num,
                "chunk_text_snippet": text[:150],
                "chunk_path": meta.get("chunk_path", ""),
            })

    if not all_context_blocks:
        return {
            "question": question,
            "answer": "I could not retrieve relevant context from the selected documents.",
            "sources": [],
            "multi_doc": True,
            "document_names": doc_names,
        }

    context = "\n\n".join(all_context_blocks)
    prompt = (
        "You are answering questions across multiple documents. "
        "Answer based only on the provided context. "
        "For each fact, cite the document name. "
        "If calculating totals or comparing values, show per-document values first, then the aggregate.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = client.chat.completions.create(
        model=config["chat"]["model"],
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Answer the user's question using only the retrieved document context. Cite document names.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    answer = response.choices[0].message.content or "I could not generate an answer."

    return {
        "question": question,
        "answer": answer,
        "sources": all_sources,
        "multi_doc": True,
        "document_names": doc_names,
    }
