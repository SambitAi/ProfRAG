from __future__ import annotations

"""Global index utilities and document-folder discovery for artifacts metadata."""

from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any

from core.locks import global_index_lock
from core.paths import slugify_filename
from core.storage import read_json, write_json_atomic


SCHEMA_VERSION = 1
_CACHE: dict[str, Any] = {
    "path": None,
    "mtime": None,
    "loaded_at": 0.0,
    "data": None,
}
_INDEX_LOCK = threading.RLock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_index() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "last_updated": _now_iso(),
        "documents": {},
    }


def load_global_index(path: str | Path, cache_ttl_seconds: float = 5.0) -> dict[str, Any]:
    file_path = Path(path)
    with _INDEX_LOCK:
        now = datetime.now(timezone.utc).timestamp()
        mtime = file_path.stat().st_mtime if file_path.exists() else None

        if (
            _CACHE["data"] is not None
            and _CACHE["path"] == str(file_path)
            and _CACHE["mtime"] == mtime
            and (now - float(_CACHE["loaded_at"])) <= cache_ttl_seconds
        ):
            return _CACHE["data"]

        data = read_json(file_path, default=_default_index())
        if not isinstance(data, dict):
            data = _default_index()
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("last_updated", _now_iso())
        data.setdefault("documents", {})

        _CACHE["path"] = str(file_path)
        _CACHE["mtime"] = mtime
        _CACHE["loaded_at"] = now
        _CACHE["data"] = data
        return data


def write_global_index_entry(path: str | Path, document_folder: str | Path, entry: dict[str, Any]) -> dict[str, Any]:
    file_path = Path(path)
    with global_index_lock(file_path):
        with _INDEX_LOCK:
            current = read_json(file_path, default=_default_index())
            if not isinstance(current, dict):
                current = _default_index()

            current.setdefault("schema_version", SCHEMA_VERSION)
            current.setdefault("documents", {})
            current["last_updated"] = _now_iso()

            folder_key = Path(document_folder).name
            existing = current["documents"].get(folder_key, {})
            merged = dict(existing)
            merged.update(entry)
            current["documents"][folder_key] = merged

            write_json_atomic(file_path, current)

            # Invalidate cache so next read observes latest write.
            _CACHE["path"] = None
            _CACHE["mtime"] = None
            _CACHE["loaded_at"] = 0.0
            _CACHE["data"] = None
            return current


def delete_global_index_entry(path: str | Path, document_folder: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    with global_index_lock(file_path):
        with _INDEX_LOCK:
            current = read_json(file_path, default=_default_index())
            if not isinstance(current, dict):
                current = _default_index()
            current.setdefault("schema_version", SCHEMA_VERSION)
            current.setdefault("documents", {})
            current["last_updated"] = _now_iso()
            folder_key = Path(document_folder).name
            current["documents"].pop(folder_key, None)
            write_json_atomic(file_path, current)
            _CACHE["path"] = None
            _CACHE["mtime"] = None
            _CACHE["loaded_at"] = 0.0
            _CACHE["data"] = None
            return current


def global_index_path(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["artifacts_root"]) / "metadata.json"


def find_document_folders(artifacts_root: str | Path) -> list[Path]:
    root = Path(artifacts_root)
    if not root.exists():
        return []
    excluded_names = {"jobs"}
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".") and path.name not in excluded_names
        ],
        key=lambda item: item.name,
    )


def find_latest_same_name_document(artifacts_root: str | Path, file_name: str) -> Path | None:
    slug = slugify_filename(file_name)
    matches: list[Path] = []
    for folder in find_document_folders(artifacts_root):
        metadata_path = folder / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = read_json(metadata_path, default={})
        if slugify_filename(metadata.get("document_name", "")) == slug:
            matches.append(folder)
    return sorted(matches, key=lambda item: item.name)[-1] if matches else None


def build_global_entry(metadata: dict[str, Any], folder: Path) -> dict[str, Any]:
    return {
        "document_id": metadata.get("document_id", folder.name),
        "slug": metadata.get("document_slug", folder.name),
        "version": metadata.get("document_version", 1),
        "document_name": metadata.get("document_name", folder.name),
        "document_folder": str(folder),
        "last_successful_step": metadata.get("last_successful_step", "unknown"),
        "ready_to_chat": metadata.get("ready_to_chat", False),
        "total_chunks": metadata.get("total_chunks", 0),
        "document_card": metadata.get("document_card", {}),
        "section_cards_path": metadata.get("section_cards_path", ""),
        "summary_status": metadata.get("summary_status", "pending"),
        "summary_ready": metadata.get("summary_ready", False),
        "extracted_fields": metadata.get("extracted_fields", {}),
        "extracted_fields_profile": metadata.get("extracted_fields_profile", ""),
        "indexed_at": metadata.get("indexed_at", ""),
    }
