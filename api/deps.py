from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from fastapi import Depends, HTTPException, Request
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


@dataclass(frozen=True)
class UserContext:
    user_id: str
    email: str
    user_key: str


def get_current_user(request: Request) -> UserContext:
    user_id = str(getattr(request.state, "user_id", ""))
    user_key = str(getattr(request.state, "user_key", ""))
    email = str(getattr(request.state, "user_email", ""))
    if not user_id or not user_key:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return UserContext(user_id=user_id, email=email, user_key=user_key)


def get_user_artifacts_root(user: UserContext = Depends(get_current_user)) -> str:
    root = Path(get_artifacts_root()) / user.user_key
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def get_user_key(user: UserContext = Depends(get_current_user)) -> str:
    return user.user_key


def resolve_document_folder(folder: str, artifacts_root: str | None = None) -> Path:
    root = Path(artifacts_root or get_artifacts_root()).resolve()
    resolved = (root / folder).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid folder")
    return resolved


def require_existing_document_folder(folder: str, artifacts_root: str | None = None) -> Path:
    resolved = resolve_document_folder(folder, artifacts_root)
    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"Document folder not found: {folder}")
    return resolved
