from __future__ import annotations

from collections import defaultdict, deque
import logging
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.deps import get_api_settings
from api.security import TokenError, decode_token

logger = logging.getLogger(__name__)

_REQUEST_WINDOWS: dict[str, deque[float]] = defaultdict(deque)
_WINDOW_SECONDS = 60.0
# Mutable single-element list so the async dispatch method can update it without
# a `global` declaration (avoids the implicit-global footgun in coroutines).
_last_purge: list[float] = [0.0]

# Routes that bootstrap or recover a session cannot require a token.
# /v2/auth/me and /v2/auth/logout are intentionally NOT listed: they need a valid token.
_AUTH_SKIP_PREFIXES = (
    "/v2/health",
    "/v2/auth/register",
    "/v2/auth/login",
    "/v2/auth/reset/",
)


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"code": "unauthorized", "message": "Unauthorized", "retryable": False, "context": {}},
    )


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(_AUTH_SKIP_PREFIXES):
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return _unauthorized()
        try:
            payload = decode_token(auth_header[len("Bearer "):])
        except TokenError:
            return _unauthorized()

        request.state.user_id = str(payload["sub"])
        request.state.user_email = str(payload["email"])
        request.state.user_key = str(payload["user_key"])
        request.state.token_jti = str(payload["jti"])
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        limit = get_api_settings().rate_limit_per_minute
        if limit <= 0:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - _WINDOW_SECONDS

        # Lazy eviction: once per window, sweep all keys and remove those with no
        # activity in the last 60 s.  Uses a snapshot (list()) so the iteration is
        # safe even if another coroutine races to add a key between checks.
        if now - _last_purge[0] >= _WINDOW_SECONDS:
            _last_purge[0] = now
            stale = [k for k, w in list(_REQUEST_WINDOWS.items()) if not w or w[-1] < cutoff]
            for k in stale:
                _REQUEST_WINDOWS.pop(k, None)

        window = _REQUEST_WINDOWS[client]
        while window and window[0] < cutoff:
            window.popleft()
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
