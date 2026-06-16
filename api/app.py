from __future__ import annotations

import asyncio
from fastapi import FastAPI
from pathlib import Path
from fastapi.responses import JSONResponse

from api.deps import get_artifacts_root, get_config_path
from api.middleware import AuthMiddleware, RateLimitMiddleware, RequestLoggingMiddleware, RequestSizeLimitMiddleware
from api.routers.auth import router as auth_router
from api.routers.chat import router as chat_router
from api.routers.documents import router as documents_router
from api.routers.health import router as health_router
from api.routers.jobs import router as jobs_router
from api.routers.summaries import router as summaries_router
from api.schemas.common import ErrorResponse
from api.security import validate_jwt_secret
from core.collections import iter_user_roots
from core.job_store import recover_inflight_jobs


def create_app() -> FastAPI:
    app = FastAPI(title="ProfRAG API", version="2.0")
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)

    @app.on_event("startup")
    async def _startup_validate() -> None:
        validate_jwt_secret()
        config_path = Path(get_config_path())
        if not config_path.exists():
            raise RuntimeError(f"Config file not found: {config_path}")
        base_root = Path(get_artifacts_root())
        base_root.mkdir(parents=True, exist_ok=True)
        # Recover jobs in every user root that already exists on disk, plus the base root
        # itself (backward-compat with single-tenant installs). iter_user_roots applies
        # strict detection so infra dirs (chroma_db, jobs) and legacy document folders
        # never get a junk jobs/ tree created under them.
        roots_to_recover = [base_root, *iter_user_roots(base_root)]
        await asyncio.gather(*[asyncio.to_thread(recover_inflight_jobs, r) for r in roots_to_recover])

    app.include_router(health_router, prefix="/v2")
    app.include_router(auth_router, prefix="/v2")
    app.include_router(documents_router, prefix="/v2")
    app.include_router(summaries_router, prefix="/v2")
    app.include_router(chat_router, prefix="/v2")
    app.include_router(jobs_router, prefix="/v2")

    @app.exception_handler(FileNotFoundError)
    async def _file_not_found_handler(_, exc: FileNotFoundError) -> JSONResponse:
        err = ErrorResponse(code="not_found", message=str(exc), retryable=False)
        return JSONResponse(status_code=404, content=err.model_dump())

    @app.exception_handler(TimeoutError)
    async def _timeout_handler(_, exc: TimeoutError) -> JSONResponse:
        err = ErrorResponse(code="timeout", message=str(exc), retryable=True)
        return JSONResponse(status_code=504, content=err.model_dump())

    @app.exception_handler(Exception)
    async def _unhandled_handler(_, exc: Exception) -> JSONResponse:
        # Keep client payload generic to avoid leaking internals.
        err = ErrorResponse(code="internal_error", message="Internal server error", retryable=True)
        return JSONResponse(status_code=500, content=err.model_dump())

    return app


app = create_app()
