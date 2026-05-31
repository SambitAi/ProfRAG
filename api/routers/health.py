from __future__ import annotations

from fastapi import APIRouter
from pathlib import Path
from fastapi.responses import JSONResponse
import socket

from api.deps import get_app_config, get_artifacts_root, get_config_path
from api.schemas.common import HealthResponse


router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health() -> HealthResponse | JSONResponse:
    config_path = Path(get_config_path())
    if not config_path.exists():
        payload = HealthResponse(status="error", service="profrag-api")
        return JSONResponse(status_code=503, content=payload.model_dump())
    artifacts_root = Path(get_artifacts_root())
    try:
        artifacts_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        payload = HealthResponse(status="error", service="profrag-api")
        return JSONResponse(status_code=503, content=payload.model_dump())

    config = get_app_config()
    vector_cfg = config.get("vector_db", {})
    host = vector_cfg.get("host")
    port = vector_cfg.get("port")
    if host and port:
        try:
            with socket.create_connection((str(host), int(port)), timeout=1.0):
                pass
        except OSError:
            payload = HealthResponse(status="error", service="profrag-api")
            return JSONResponse(status_code=503, content=payload.model_dump())
    return HealthResponse(status="ok", service="profrag-api")
