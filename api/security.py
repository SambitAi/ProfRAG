from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid

import jwt

from core.user_store import SECURITY_QUESTIONS

logger = logging.getLogger(__name__)

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
JWT_SECRET_ENV_VAR = "PROFRAG_JWT_SECRET"
MIN_JWT_SECRET_LENGTH = 32


class TokenError(Exception):
    """Raised when a token is missing, malformed, expired, or revoked."""


_DENYLIST_LOCK = threading.Lock()
_DENYLISTED_JTIS: set[str] = set()
_USER_REVOKED_AT: dict[str, int] = {}


def get_jwt_secret() -> str:
    secret = os.environ.get(JWT_SECRET_ENV_VAR, "")
    if len(secret) < MIN_JWT_SECRET_LENGTH:
        raise RuntimeError(
            f"{JWT_SECRET_ENV_VAR} must be set to a value of at least {MIN_JWT_SECRET_LENGTH} characters."
        )
    return secret


def validate_jwt_secret() -> None:
    get_jwt_secret()


def create_access_token(*, user_id: int | str, email: str, user_key: str) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "email": email,
        "user_key": user_key,
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": now + ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "email", "user_key", "jti", "iat", "exp"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Invalid token.") from exc

    for claim in ("sub", "email", "user_key", "jti"):
        if not str(payload.get(claim, "")).strip():
            raise TokenError("Invalid token.")

    jti = str(payload["jti"])
    sub = str(payload["sub"])
    with _DENYLIST_LOCK:
        if jti in _DENYLISTED_JTIS:
            raise TokenError("Token has been revoked.")
        revoked_at = _USER_REVOKED_AT.get(sub)
    # Fail closed: <= means a token minted in the same second as a revocation
    # is rejected; the issuer just retries a second later.
    if revoked_at is not None and int(payload.get("iat", 0)) <= revoked_at:
        raise TokenError("Token has been revoked.")
    return payload


def revoke_jti(jti: str) -> None:
    if not jti:
        return
    with _DENYLIST_LOCK:
        _DENYLISTED_JTIS.add(jti)
    logger.info("token_revoked", extra={"jti": jti})


def revoke_user(user_id: int | str) -> None:
    sub = str(user_id)
    with _DENYLIST_LOCK:
        _USER_REVOKED_AT[sub] = int(time.time())
    logger.info("user_tokens_revoked", extra={"sub": sub})


def decoy_question_for(email: str) -> str:
    normalized = str(email).strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return SECURITY_QUESTIONS[int(digest, 16) % len(SECURITY_QUESTIONS)]
