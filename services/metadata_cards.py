from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import chromadb

from core.llm import get_openai_client
from core.metadata import load_metadata, save_metadata
from core.paths import compute_document_id, ensure_directory
from core.storage import read_json, read_text, write_json


_HEADING_RE = re.compile(r"^(#{1,2})\s+(.*)$")
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("metadata_cards", {})
    return {
        "first_page_chars": int(block.get("first_page_chars", 1200)),
        "section_preview_chars": int(block.get("section_preview_chars", 600)),
        "include_asset_captions": bool(block.get("include_asset_captions", True)),
        "embed_on_ingest": bool(block.get("embed_on_ingest", True)),
        "summary_boost": float(block.get("summary_boost", 0.25)),
        "summary_required_for_routing": bool(block.get("summary_required_for_routing", False)),
    }


def _get_chroma_client(config: dict[str, Any]) -> chromadb.ClientAPI:
    vector_db = config.get("vector_db", {})
    chroma_host = vector_db.get("host")
    chroma_port = int(vector_db.get("port", 8000))
    if chroma_host:
        return chromadb.HttpClient(host=chroma_host, port=chroma_port)
    persist_dir = vector_db.get("persist_directory", "artifacts/chroma_db")
    ensure_directory(persist_dir)
    return chromadb.PersistentClient(path=str(persist_dir))


def _upsert_card_vectors(
    config: dict[str, Any],
    document_folder: Path,
    document_name: str,
    document_id: str,
    document_card: dict[str, Any],
    section_entries: list[dict[str, Any]],
) -> None:
    if not section_entries and not document_card:
        return
    client = get_openai_client()
    embedding_model = config["embeddings"]["model"]
    collection_name = config.get("vector_db", {}).get("card_collection_name", "pdf_rag_cards")
    chroma = _get_chroma_client(config)
    collection = chroma.get_or_create_collection(name=collection_name)

    doc_text = document_card.get("_embed_text", "")
    if doc_text:
        resp = client.embeddings.create(model=embedding_model, input=[doc_text])
        collection.upsert(
            ids=[document_card["card_id"]],
            embeddings=[resp.data[0].embedding],
            documents=[doc_text],
            metadatas=[{
                "document_id": document_id,
                "document_folder": str(document_folder),
                "document_name": document_name,
                "card_type": "document",
                "section_id": "",
                "section_path": "",
            }],
        )

    section_ids: list[str] = []
    section_texts: list[str] = []
    section_metas: list[dict[str, Any]] = []
    for section in section_entries:
        embed_text = section.get("_embed_text", "")
        if not embed_text:
            continue
        section_ids.append(section["card_id"])
        section_texts.append(embed_text)
        section_metas.append({
            "document_id": document_id,
            "document_folder": str(document_folder),
            "document_name": document_name,
            "card_type": "section",
            "section_id": section.get("section_id", ""),
            "section_path": section.get("section_path", ""),
        })

    if section_ids:
        section_embeddings: list[list[float]] = []
        for text in section_texts:
            resp = client.embeddings.create(model=embedding_model, input=[text])
            section_embeddings.append(resp.data[0].embedding)
        collection.upsert(
            ids=section_ids,
            embeddings=section_embeddings,
            documents=section_texts,
            metadatas=section_metas,
        )


def _markdown_path(document_folder: Path, metadata: dict[str, Any]) -> Path:
    mp = metadata.get("markdown_path")
    if mp:
        return Path(mp)
    return document_folder / "markdown" / "document.md"


def _first_heading(markdown: str) -> str | None:
    for line in markdown.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
    return None


def _headings_h1_h2(markdown: str) -> list[str]:
    result: list[str] = []
    for line in markdown.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            result.append(m.group(2).strip())
    return result


def _first_nonempty_paragraph(markdown: str) -> str:
    para_lines: list[str] = []
    for line in markdown.splitlines():
        s = line.strip()
        if not s:
            if para_lines:
                break
            continue
        if s.startswith("#") or _IMAGE_REF_RE.search(s) or s.startswith("|"):
            if para_lines:
                break
            continue
        para_lines.append(s)
    return " ".join(para_lines).strip()


def _first_page_text(markdown: str, max_chars: int) -> str:
    cleaned: list[str] = []
    for line in markdown.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or _IMAGE_REF_RE.search(s):
            continue
        cleaned.append(s)
    return "\n".join(cleaned)[:max_chars]


def build_document_card(document_folder: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    folder = Path(document_folder)
    metadata = load_metadata(folder)
    cfg = _cfg(config)
    markdown = read_text(_markdown_path(folder, metadata))
    document_name = metadata.get("document_name", folder.name)
    source_url = metadata.get("source_url", "")
    document_id = metadata.get("document_id") or compute_document_id(document_name, source_url)

    title = _first_heading(markdown) or document_name
    opening_text = _first_nonempty_paragraph(markdown)
    first_page_text = _first_page_text(markdown, cfg["first_page_chars"])
    headings = _headings_h1_h2(markdown)

    images_index = read_json(folder / "images" / "index.json", default={})
    tables_index = read_json(folder / "tables" / "index.json", default={})
    total_chunks = len(metadata.get("chunk_paths", []))
    total_images = len(images_index)
    total_tables = len(tables_index)

    embed_text = f"{title}\n{opening_text}\n{first_page_text}\n" + "\n".join(headings)

    return {
        "card_id": f"{document_id}_doc_card",
        "title": title,
        "opening_text": opening_text,
        "first_page_text": first_page_text,
        "headings": headings,
        "summary_card_id": "",
        "summary_boost": cfg["summary_boost"],
        "counts": {
            "total_chunks": total_chunks,
            "total_images": total_images,
            "total_tables": total_tables,
        },
        "_embed_text": embed_text,
    }


def build_section_index(document_folder: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    folder = Path(document_folder)
    metadata = load_metadata(folder)
    cfg = _cfg(config)
    sections_dir = folder / "sections"
    document_name = metadata.get("document_name", folder.name)
    source_url = metadata.get("source_url", "")
    document_id = metadata.get("document_id") or compute_document_id(document_name, source_url)

    chunk_by_id: dict[str, dict[str, Any]] = {}
    chunk_path_by_id: dict[str, str] = {}
    for chunk_path in metadata.get("chunk_paths", []):
        payload = read_json(chunk_path, default={})
        cid = payload.get("chunk_id")
        if cid:
            chunk_by_id[cid] = payload
            chunk_path_by_id[cid] = str(chunk_path)

    images_by_abs = {}
    for info in read_json(folder / "images" / "index.json", default={}).values():
        abs_path = info.get("abs_path")
        if abs_path:
            images_by_abs[abs_path] = info

    tables_by_abs = {}
    for info in read_json(folder / "tables" / "index.json", default={}).values():
        csv_path = info.get("csv_path")
        if csv_path:
            tables_by_abs[csv_path] = info

    def _build_entry(
        section_id: str,
        section_path: str,
        heading: str,
        chunk_ids: list[str],
        image_paths: list[str],
        table_paths: list[str],
    ) -> dict[str, Any]:
        chunk_paths = [chunk_path_by_id[cid] for cid in chunk_ids if cid in chunk_path_by_id]
        chunk_texts = [chunk_by_id[cid].get("text", "") for cid in chunk_ids if cid in chunk_by_id]
        preview_text = "\n".join(t for t in chunk_texts if t).strip()[: cfg["section_preview_chars"]]

        image_captions: list[str] = []
        for image_path in image_paths:
            cap = (images_by_abs.get(image_path, {}) or {}).get("caption")
            if cap:
                image_captions.append(cap)

        table_captions: list[str] = []
        for table_path in table_paths:
            cap = (tables_by_abs.get(table_path, {}) or {}).get("caption")
            if cap:
                table_captions.append(cap)

        embed_parts = [heading, preview_text]
        if cfg["include_asset_captions"]:
            if image_captions:
                embed_parts.append("\n".join(image_captions))
            if table_captions:
                embed_parts.append("\n".join(table_captions))
        embed_text = "\n".join(part for part in embed_parts if part).strip()

        return {
            "card_id": f"{document_id}_{section_id}_card",
            "section_id": section_id,
            "section_path": section_path,
            "heading": heading,
            "preview_text": preview_text,
            "chunk_ids": chunk_ids,
            "chunk_paths": chunk_paths,
            "image_captions": image_captions if cfg["include_asset_captions"] else [],
            "table_captions": table_captions if cfg["include_asset_captions"] else [],
            "l3_summary": "",
            "summary_card_id": "",
            "summary_boost": cfg["summary_boost"],
            "_embed_text": embed_text,
        }

    section_entries: list[dict[str, Any]] = []
    section_files = [p for p in sorted(sections_dir.glob("*.json")) if p.name != "index.json"]
    if section_files:
        for section_file in section_files:
            sec = read_json(section_file, default={})
            section_id = section_file.stem
            section_path = sec.get("section_path") or sec.get("h1") or section_id
            heading = sec.get("h1") or section_path
            chunk_ids: list[str] = sec.get("chunk_ids", [])
            section_entries.append(
                _build_entry(
                    section_id=section_id,
                    section_path=section_path,
                    heading=heading,
                    chunk_ids=chunk_ids,
                    image_paths=sec.get("image_paths", []),
                    table_paths=sec.get("table_paths", []),
                )
            )
    else:
        grouped: dict[str, dict[str, Any]] = {}
        for chunk_path in metadata.get("chunk_paths", []):
            chunk = read_json(chunk_path, default={})
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            section_path = chunk.get("section_path") or chunk.get("h1") or chunk.get("h2") or "Document"
            key = str(section_path)
            entry = grouped.setdefault(
                key,
                {"chunk_ids": [], "image_paths": [], "table_paths": []},
            )
            entry["chunk_ids"].append(chunk_id)
            entry["image_paths"].extend(chunk.get("image_paths", []))
            entry["table_paths"].extend(chunk.get("table_paths", []))

        for idx, (section_path, payload) in enumerate(grouped.items(), start=1):
            section_entries.append(
                _build_entry(
                    section_id=f"section_{idx}",
                    section_path=section_path,
                    heading=section_path,
                    chunk_ids=payload["chunk_ids"],
                    image_paths=list(dict.fromkeys(payload["image_paths"])),
                    table_paths=list(dict.fromkeys(payload["table_paths"])),
                )
            )

    payload = {
        "schema_version": 1,
        "sections": [
            {k: v for k, v in s.items() if k != "_embed_text"}
            for s in section_entries
        ],
    }
    write_json(sections_dir / "index.json", payload)
    return {"sections": section_entries, "persisted": payload}


def run(document_folder: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    folder = Path(document_folder)
    metadata = load_metadata(folder)
    document_name = metadata.get("document_name", folder.name)
    source_url = metadata.get("source_url", "")
    document_id = metadata.get("document_id") or compute_document_id(document_name, source_url)
    metadata["document_id"] = document_id

    document_card = build_document_card(folder, config)
    section_index_result = build_section_index(folder, config)
    section_entries = section_index_result["sections"]
    section_index_for_file = section_index_result["persisted"]
    section_cards_path = str(folder / "sections" / "index.json")

    if _cfg(config)["embed_on_ingest"]:
        _upsert_card_vectors(
            config=config,
            document_folder=folder,
            document_name=document_name,
            document_id=document_id,
            document_card=document_card,
            section_entries=section_entries,
        )

    metadata["document_card"] = {k: v for k, v in document_card.items() if k != "_embed_text"}
    metadata["section_cards_path"] = section_cards_path
    metadata["section_names"] = [s.get("heading", "") for s in section_entries]
    metadata["section_chunk_counts"] = {
        s.get("heading", s.get("section_id", "")): len(s.get("chunk_ids", []))
        for s in section_entries
    }
    save_metadata(folder, metadata)

    return {
        "document_card": metadata["document_card"],
        "section_cards_path": section_cards_path,
        "section_count": len(section_index_for_file.get("sections", [])),
    }
