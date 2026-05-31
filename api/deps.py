from __future__ import annotations

from pathlib import Path
from fastapi import HTTPException
import threading

from core.config import load_app_config
from api.settings import load_api_settings

_settings = load_api_settings()
_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: dict = {}
_CONFIG_MTIME: float = -1.0


def get_config_path() -> str:
    return _settings.config_path


def get_api_settings():
    return _settings


def get_app_config(force_reload: bool = False) -> dict:
    global _CONFIG_CACHE, _CONFIG_MTIME
    config_path = Path(_settings.config_path)
    try:
        mtime = config_path.stat().st_mtime
    except FileNotFoundError:
        mtime = -1.0
    with _CONFIG_LOCK:
        if force_reload or not _CONFIG_CACHE or mtime != _CONFIG_MTIME:
            _CONFIG_CACHE = load_app_config(_settings.config_path)
            _CONFIG_MTIME = mtime
        return _CONFIG_CACHE


def get_artifacts_root() -> str:
    config = get_app_config()
    return str(Path(config["paths"]["artifacts_root"]))


def resolve_document_folder(folder: str) -> Path:
    root = Path(get_artifacts_root()).resolve()
    resolved = (root / folder).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid folder")
    return resolved


def require_existing_document_folder(folder: str) -> Path:
    resolved = resolve_document_folder(folder)
    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"Document folder not found: {folder}")
    return resolved
