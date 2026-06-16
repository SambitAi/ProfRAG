from __future__ import annotations

from collections import defaultdict, deque
import logging
import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from api import security
from api.deps import UserContext, get_artifacts_root, get_current_user
from api.schemas.auth import (
    LoginRequest,
    MeResponse,
    RegisterRequest,
    RegisterResponse,
    ResetFinishRequest,
    ResetStartRequest,
    ResetStartResponse,
    TokenResponse,
)
from core import user_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 60.0
RESET_MAX_ATTEMPTS = 5
RESET_WINDOW_SECONDS = 3600.0

_RATE_LOCK = threading.Lock()
_LOGIN_ATTEMPTS: dict[str, deque[float]] = defaultdict(deque)
_RESET_ATTEMPTS: dict[str, deque[float]] = defaultdict(deque)
# Tracks last purge time per store object so LOGIN and RESET are evicted independently.
_PURGE_TIMESTAMPS: dict[int, float] = {}


def _purge_stale(store: dict[str, deque[float]], window_seconds: float) -> None:
    """Evict keys with no activity in the last window_seconds. Must be called under _RATE_LOCK."""
    store_id = id(store)
    now = time.time()
    if now - _PURGE_TIMESTAMPS.get(store_id, 0.0) < window_seconds:
        return
    _PURGE_TIMESTAMPS[store_id] = now
    cutoff = now - window_seconds
    stale = [k for k, w in list(store.items()) if not w or w[-1] < cutoff]
    for k in stale:
        store.pop(k, None)


def _too_many_attempts(store: dict[str, deque[float]], key: str, limit: int, window_seconds: float) -> bool:
    now = time.time()
    with _RATE_LOCK:
        _purge_stale(store, window_seconds)
        attempts = store[key]
        cutoff = now - window_seconds
        while attempts and attempts[0] < cutoff:
            attempts.popleft()
        if len(attempts) >= limit:
            return True
        attempts.append(now)
        return False


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("/register", response_model=RegisterResponse, status_code=201)
def register(req: RegisterRequest) -> RegisterResponse:
    try:
        record = user_store.create_user(
            get_artifacts_root(),
            req.email,
            req.password,
            req.security_question,
            req.security_answer,
        )
    except user_store.UserStoreError:
        # Generic message: do not reveal whether the email is taken or why input failed.
        raise HTTPException(status_code=400, detail="Could not create account.")
    return RegisterResponse(email=record.email_normalized, user_key=record.user_key)


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, request: Request) -> TokenResponse:
    if _too_many_attempts(_LOGIN_ATTEMPTS, f"ip:{_client_ip(request)}", LOGIN_MAX_ATTEMPTS, LOGIN_WINDOW_SECONDS):
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")
    record = user_store.verify_user(get_artifacts_root(), req.email, req.password)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = security.create_access_token(
        user_id=record.id,
        email=record.email_normalized,
        user_key=record.user_key,
    )
    return TokenResponse(access_token=token, expires_in=security.ACCESS_TOKEN_EXPIRE_MINUTES * 60)


@router.get("/me", response_model=MeResponse)
def me(user: UserContext = Depends(get_current_user)) -> MeResponse:
    return MeResponse(email=user.email, user_key=user.user_key)


@router.post("/logout", status_code=204)
def logout(request: Request) -> Response:
    security.revoke_jti(str(getattr(request.state, "token_jti", "")))
    return Response(status_code=204)


@router.post("/reset/start", response_model=ResetStartResponse)
def reset_start(req: ResetStartRequest) -> ResetStartResponse:
    question = user_store.get_security_question(get_artifacts_root(), req.email)
    if not question:
        # Unknown email gets a deterministic decoy so the flow never reveals
        # whether an account exists.
        question = security.decoy_question_for(req.email)
    return ResetStartResponse(security_question=question)


@router.post("/reset/finish", status_code=204)
def reset_finish(req: ResetFinishRequest, request: Request) -> Response:
    email_key = f"email:{user_store.normalize_email(req.email)}"
    ip_key = f"ip:{_client_ip(request)}"
    limited_by_email = _too_many_attempts(_RESET_ATTEMPTS, email_key, RESET_MAX_ATTEMPTS, RESET_WINDOW_SECONDS)
    limited_by_ip = _too_many_attempts(_RESET_ATTEMPTS, ip_key, RESET_MAX_ATTEMPTS, RESET_WINDOW_SECONDS)
    if limited_by_email or limited_by_ip:
        raise HTTPException(status_code=429, detail="Too many reset attempts. Try again later.")

    base = get_artifacts_root()
    if not user_store.verify_security_answer(base, req.email, req.security_answer):
        raise HTTPException(status_code=401, detail="Could not reset password.")
    try:
        updated = user_store.update_password(base, req.email, req.new_password)
    except user_store.UserValidationError:
        raise HTTPException(status_code=401, detail="Could not reset password.")
    if not updated:
        raise HTTPException(status_code=401, detail="Could not reset password.")
    record = user_store.get_user_by_email(base, req.email)
    if record is not None:
        security.revoke_user(record.id)
    logger.info("password_reset_completed", extra={"user_key": record.user_key if record else ""})
    return Response(status_code=204)
