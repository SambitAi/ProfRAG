from __future__ import annotations

import os
from pathlib import Path

import chromadb

from core.llm import get_openai_client
from core.metadata import load_metadata, mark_step_success, save_metadata
from core.paths import build_images_dir, build_tables_dir, ensure_directory
from core.storage import read_json, write_json


def run(
    chunk_paths: list[str],
    document_folder: str | Path,
    chroma_persist_dir: str | Path,
    collection_name: str,
    embedding_model: str,
    index_result_path: str | Path,
) -> str:
    client = get_openai_client()
    document_folder = Path(document_folder)
    chroma_host = os.getenv("CHROMA_HOST")
    if chroma_host:
        chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=int(os.getenv("CHROMA_PORT", "8000")),
        )
    else:
        ensure_directory(chroma_persist_dir)
        chroma_client = chromadb.PersistentClient(path=str(chroma_persist_dir))
    collection = chroma_client.get_or_create_collection(name=collection_name)

    processed_chunk_ids: list[str] = []
    for chunk_path in chunk_paths:
        payload = read_json(chunk_path)
        chunk_id = payload["chunk_id"]

        # Prepend section path so cross-section chunks share domain context in vector space.
        section_path = payload.get("section_path", "")
        raw_text = payload.get("text") or ""
        embed_input = f"[{section_path}]\n{raw_text}" if section_path else raw_text
        if not embed_input.strip():
            continue
        response = client.embeddings.create(model=embedding_model, input=[embed_input])
        embedding = response.data[0].embedding

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[payload["text"]],
            metadatas=[
                {
                    "document_name": payload["document_name"],
                    "chunk_number": payload["chunk_number"],
                    "document_folder": str(document_folder),
                    "chunk_path": str(chunk_path),
                    "section_path": payload.get("section_path", ""),
                    "section_level": payload.get("section_level", 0),
                    "item_type": "text",
                }
            ],
        )
        processed_chunk_ids.append(chunk_id)

    # ── Media item indexing ──────────────────────────────────────────────────
    indexed_media = 0

    # Index images: embed title + caption + surrounding context
    images_index = read_json(build_images_dir(document_folder) / "index.json", default={})
    for filename, info in images_index.items():
        caption = info.get("caption") or ""
        title = info.get("title") or ""
        prev_text = info.get("prev_text") or ""
        next_text = info.get("next_text") or ""
        parts = [p for p in [title, caption, prev_text, next_text] if p]
        embed_text = "\n".join(parts) if parts else filename
        img_resp = client.embeddings.create(model=embedding_model, input=[embed_text])
        collection.upsert(
            ids=[f"img_{filename}"],
            embeddings=[img_resp.data[0].embedding],
            documents=[embed_text],
            metadatas=[{
                "document_folder": str(document_folder),
                "item_type": "image",
                "abs_path": info.get("abs_path", ""),
                "caption": caption,
                "page": info.get("page", 0),
            }],
        )
        indexed_media += 1

    # Index tables: embed CSV content + surrounding context
    tables_index = read_json(build_tables_dir(document_folder) / "index.json", default={})
    for _name, info in tables_index.items():
        table_path = info.get("csv_path", "")
        if not table_path:
            continue
        try:
            csv_text = Path(table_path).read_text(encoding="utf-8")[:800]
        except OSError:
            continue
        parts = [p for p in [info.get("caption"), info.get("title"), csv_text, info.get("prev_text"), info.get("next_text")] if p]
        tbl_embed = "\n".join(parts)
        if not tbl_embed.strip():
            continue
        tbl_resp = client.embeddings.create(model=embedding_model, input=[tbl_embed])
        collection.upsert(
            ids=[f"table_{Path(table_path).stem}"],
            embeddings=[tbl_resp.data[0].embedding],
            documents=[tbl_embed],
            metadatas=[{
                "document_folder": str(document_folder),
                "item_type": "table",
                "abs_path": table_path,
            }],
        )
        indexed_media += 1

    result = {
        "collection_name": collection_name,
        "indexed_chunk_count": len(processed_chunk_ids),
        "indexed_chunk_ids": processed_chunk_ids,
        "indexed_media_count": indexed_media,
        "persist_directory": str(chroma_persist_dir),
    }
    write_json(index_result_path, result)

    metadata = load_metadata(document_folder)
    metadata["vector_store"] = result
    metadata["ready_to_chat"] = True
    save_metadata(document_folder, metadata)
    mark_step_success(document_folder, "write_to_vector_db", result)
    return str(index_result_path)
