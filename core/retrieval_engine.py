from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb

from core.llm import get_openai_client
from core.metadata import mark_step_success
from core.paths import build_images_dir, build_sections_dir, ensure_directory
from core.storage import read_json, write_json

logger = logging.getLogger(__name__)


def _load_doc_images(document_folder: Path) -> list[dict[str, Any]]:
    images_index = read_json(build_images_dir(document_folder) / "index.json", default={})
    return [
        {
            "abs_path": info.get("abs_path", ""),
            "caption": info.get("caption"),
            "ocr_text": info.get("ocr_text"),
            "title": info.get("title"),
            "prev_text": info.get("prev_text"),
            "next_text": info.get("next_text"),
        }
        for info in images_index.values()
    ]


def run_retrieval(
    question: str,
    document_folder: str | Path,
    chroma_persist_dir: str | Path,
    collection_name: str,
    embedding_model: str,
    retrieval_output_path: str | Path,
    top_k: int,
    media_top_k: int = 4,
    expand_parent: bool = True,
    chroma_host: str | None = None,
    chroma_port: int = 8000,
) -> str:
    client = get_openai_client()
    document_folder = Path(document_folder)
    ensure_directory(Path(retrieval_output_path).parent)
    if chroma_host:
        chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=int(chroma_port),
        )
    else:
        chroma_client = chromadb.PersistentClient(path=str(chroma_persist_dir))
    collection = chroma_client.get_collection(name=collection_name)

    if not question or not question.strip():
        raise ValueError("question cannot be empty")
    response = client.embeddings.create(model=embedding_model, input=[question])
    query_embedding = response.data[0].embedding

    text_docs: list[str] = []
    text_metas: list[dict] = []
    text_ids: list[str] = []

    try:
        text_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"$and": [
                {"document_folder": {"$eq": str(document_folder)}},
                {"item_type": {"$eq": "text"}},
            ]},
        )
        text_docs = (text_result.get("documents") or [[]])[0]
        text_metas = (text_result.get("metadatas") or [[]])[0]
        text_ids = (text_result.get("ids") or [[]])[0]
    except Exception:
        logger.warning(
            "Document indexed without item_type — media retrieval disabled. Re-index to enable."
        )
        fallback = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"document_folder": str(document_folder)},
        )
        text_docs = (fallback.get("documents") or [[]])[0]
        text_metas = (fallback.get("metadatas") or [[]])[0]
        text_ids = (fallback.get("ids") or [[]])[0]
        retrieval_payload = {
            "question": question,
            "documents": text_docs,
            "metadatas": text_metas,
            "ids": text_ids,
            "media_items": [],
            "section_media": {"image_paths": [], "table_paths": []},
            "images_meta": _load_doc_images(document_folder),
        }
        write_json(retrieval_output_path, retrieval_payload)
        mark_step_success(document_folder, "retrieve_context", {"retrieval_output_path": str(retrieval_output_path)})
        return str(retrieval_output_path)

    media_items: list[dict[str, Any]] = []
    try:
        media_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=media_top_k,
            where={"$and": [
                {"document_folder": {"$eq": str(document_folder)}},
                {"item_type": {"$in": ["image", "table"]}},
            ]},
        )
        media_docs = (media_result.get("documents") or [[]])[0]
        media_metas = (media_result.get("metadatas") or [[]])[0]
        for doc, meta in zip(media_docs, media_metas):
            media_items.append({
                "item_type": meta.get("item_type", ""),
                "abs_path": meta.get("abs_path", ""),
                "caption": meta.get("caption", ""),
                "document": doc,
            })
    except Exception:
        pass

    section_media: dict[str, list[str]] = {
        "image_paths": [],
        "table_paths": [],
    }
    if expand_parent:
        seen_parent_ids: set[str] = set()
        direct_image_paths: set[str] = set()
        direct_table_paths: set[str] = set()
        for meta in text_metas:
            chunk_path = meta.get("chunk_path", "")
            if chunk_path:
                chunk_data = read_json(chunk_path, default={})
                direct_image_paths.update(chunk_data.get("image_paths", []))
                direct_table_paths.update(chunk_data.get("table_paths", []))
                parent_id = chunk_data.get("parent_section_id")
                if parent_id and parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    section_file = build_sections_dir(document_folder) / f"{parent_id}.json"
                    sec = read_json(section_file, default={})
                    for ip in sec.get("image_paths", []):
                        if ip not in direct_image_paths:
                            section_media["image_paths"].append(ip)
                    for tp in sec.get("table_paths", []):
                        if tp not in direct_table_paths:
                            section_media["table_paths"].append(tp)

    retrieval_payload = {
        "question": question,
        "documents": text_docs,
        "metadatas": text_metas,
        "ids": text_ids,
        "media_items": media_items,
        "section_media": section_media,
        "images_meta": _load_doc_images(document_folder),
    }
    write_json(retrieval_output_path, retrieval_payload)
    mark_step_success(document_folder, "retrieve_context", {"retrieval_output_path": str(retrieval_output_path)})
    return str(retrieval_output_path)
