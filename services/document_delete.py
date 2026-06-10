from __future__ import annotations

from pathlib import Path
from typing import Any
import inspect
import logging

import chromadb
from chromadb.errors import NotFoundError

from core.paths import ensure_directory

logger = logging.getLogger(__name__)
_SUPPORTS_WHERE_DELETE: bool | None = None


class DocumentVectorCleanupError(Exception):
    """Raised when vector cleanup fails for a target document."""


def _parse_version_tuple(version_text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in (version_text or "").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def supports_where_delete() -> bool:
    global _SUPPORTS_WHERE_DELETE
    if _SUPPORTS_WHERE_DELETE is not None:
        return _SUPPORTS_WHERE_DELETE

    version_text = str(getattr(chromadb, "__version__", "") or "")
    version_tuple = _parse_version_tuple(version_text)
    if version_tuple:
        _SUPPORTS_WHERE_DELETE = version_tuple >= (0, 4)
        return _SUPPORTS_WHERE_DELETE

    try:
        client = chromadb.PersistentClient(path=str(Path.cwd() / "artifacts" / "chroma_probe"))
        collection = client.get_or_create_collection(name="codexprobe")
        _SUPPORTS_WHERE_DELETE = "where" in inspect.signature(collection.delete).parameters
    except Exception:
        _SUPPORTS_WHERE_DELETE = False
    return _SUPPORTS_WHERE_DELETE


def _get_chroma_client(config: dict[str, Any]) -> chromadb.ClientAPI:
    vector_db = config.get("vector_db", {})
    chroma_host = vector_db.get("host")
    chroma_port = int(vector_db.get("port", 8000))
    if chroma_host:
        return chromadb.HttpClient(host=chroma_host, port=chroma_port)
    persist_dir = vector_db.get("persist_directory", "artifacts/chroma_db")
    ensure_directory(persist_dir)
    return chromadb.PersistentClient(path=str(persist_dir))


def _target_collection_names(config: dict[str, Any]) -> list[str]:
    vector_db = config.get("vector_db", {})
    names = [
        str(vector_db.get("collection_name", "pdf_rag_chunks")).strip(),
        str(vector_db.get("summary_collection_name", "pdf_rag_summaries")).strip(),
        str(vector_db.get("card_collection_name", "pdf_rag_cards")).strip(),
    ]
    return list(dict.fromkeys(name for name in names if name))


def _lookup_collection(chroma_client: chromadb.ClientAPI, collection_name: str):
    try:
        return chroma_client.get_collection(name=collection_name)
    except (NotFoundError, ValueError):
        return None


def _selector_for_folder(document_folder: str) -> dict[str, Any]:
    return {"document_folder": {"$eq": document_folder}}


def _selector_for_document_id(document_id: str) -> dict[str, Any]:
    return {"document_id": {"$eq": document_id}}


def _matching_ids(collection, selector: dict[str, Any]) -> list[str]:
    result = collection.get(where=selector, include=["metadatas"])
    ids = result.get("ids", []) or []
    if isinstance(ids, str):
        return [ids]
    return [str(item) for item in ids if item]


def _delete_by_ids(collection, ids: list[str]) -> None:
    if ids:
        collection.delete(ids=ids)


def _delete_with_selector(
    collection,
    selector: dict[str, Any],
    ids: list[str],
    allow_where_delete: bool,
    *,
    document_folder: str,
    document_id: str,
    collection_name: str,
) -> str:
    if not ids:
        return "skip_no_match"
    if allow_where_delete:
        try:
            collection.delete(where=selector)
            return "where"
        except Exception:
            logger.warning(
                "document_vector_cleanup_where_delete_failed",
                extra={
                    "document_folder": document_folder,
                    "document_id": document_id,
                    "collection_name": collection_name,
                    "selector": selector,
                },
                exc_info=True,
            )
            _delete_by_ids(collection, ids)
            return "ids_fallback"
    _delete_by_ids(collection, ids)
    return "ids"


def cleanup_document_vectors(
    config: dict[str, Any],
    *,
    document_folder: str,
    document_id: str = "",
) -> dict[str, Any]:
    chroma_client = _get_chroma_client(config)
    allow_where_delete = supports_where_delete()
    folder_selector = _selector_for_folder(document_folder)
    folder_path = str(document_folder)
    cleanup_details: list[dict[str, Any]] = []
    collections_cleaned: list[str] = []

    for collection_name in _target_collection_names(config):
        collection = _lookup_collection(chroma_client, collection_name)
        if collection is None:
            cleanup_details.append(
                {
                    "collection_name": collection_name,
                    "status": "skip_no_collection",
                    "selector": "document_folder",
                    "matched_ids": 0,
                    "delete_mode": "none",
                }
            )
            continue

        try:
            ids = _matching_ids(collection, folder_selector)
            selector_used = "document_folder"
            selector = folder_selector
            if not ids and document_id:
                selector = _selector_for_document_id(document_id)
                ids = _matching_ids(collection, selector)
                selector_used = "document_id"

            delete_mode = _delete_with_selector(
                collection,
                selector,
                ids,
                allow_where_delete,
                document_folder=folder_path,
                document_id=document_id,
                collection_name=collection_name,
            )
            matched_count = len(ids)
            if matched_count:
                collections_cleaned.append(collection_name)
            cleanup_details.append(
                {
                    "collection_name": collection_name,
                    "status": "cleaned" if matched_count else "skip_no_match",
                    "selector": selector_used,
                    "matched_ids": matched_count,
                    "delete_mode": delete_mode,
                }
            )
            logger.info(
                "document_vector_cleanup",
                extra={
                    "document_folder": folder_path,
                    "document_id": document_id,
                    "collection_name": collection_name,
                    "selector": selector_used,
                    "matched_ids": matched_count,
                    "delete_mode": delete_mode,
                },
            )
        except Exception as exc:
            logger.exception(
                "document_vector_cleanup_failed",
                extra={
                    "document_folder": folder_path,
                    "document_id": document_id,
                    "collection_name": collection_name,
                },
            )
            raise DocumentVectorCleanupError(
                f"Vector cleanup failed for collection '{collection_name}' and document '{folder_path}'."
            ) from exc

    return {
        "collections_cleaned": collections_cleaned,
        "details": cleanup_details,
        "supports_where_delete": allow_where_delete,
    }
