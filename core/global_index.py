from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.storage import read_json, write_json_atomic


SCHEMA_VERSION = 1
_CACHE: dict[str, Any] = {
    "path": None,
    "mtime": None,
    "loaded_at": 0.0,
    "data": None,
}


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

