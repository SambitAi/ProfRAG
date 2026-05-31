from __future__ import annotations

from collections import defaultdict, deque
import hmac
import logging
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.deps import get_api_settings

logger = logging.getLogger(__name__)

_REQUEST_WINDOWS: dict[str, deque[float]] = defaultdict(deque)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = get_api_settings().auth_token
        if not token:
            return await call_next(request)
        if request.url.path.startswith("/v2/health"):
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        expected = f"Bearer {token}"
        if not hmac.compare_digest(auth_header, expected):
            return JSONResponse(
                status_code=401,
                content={"code": "unauthorized", "message": "Unauthorized", "retryable": False, "context": {}},
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        limit = get_api_settings().rate_limit_per_minute
        if limit <= 0:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.time()
        window = _REQUEST_WINDOWS[client]
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if not window:
            _REQUEST_WINDOWS.pop(client, None)
            window = _REQUEST_WINDOWS[client]
        if len(window) >= limit:
            return JSONResponse(
                status_code=429,
                content={"code": "rate_limited", "message": "Rate limit exceeded", "retryable": True, "context": {}},
            )
        window.append(now)
        return await call_next(request)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH"}:
            max_bytes = get_api_settings().max_request_bytes
            content_length = request.headers.get("content-length")
            if content_length is None:
                return JSONResponse(
                    status_code=411,
                    content={"code": "length_required", "message": "Content-Length header is required", "retryable": False, "context": {}},
                )
            try:
                if int(content_length) > max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"code": "payload_too_large", "message": "Payload too large", "retryable": False, "context": {}},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"code": "invalid_content_length", "message": "Invalid Content-Length header", "retryable": False, "context": {}},
                )
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        response.headers["x-request-id"] = request_id
        logger.info(
            "api_request",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
