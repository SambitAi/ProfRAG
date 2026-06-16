from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import logging
import re
import sqlite3

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

logger = logging.getLogger(__name__)

MIN_PASSWORD_LENGTH = 8

SECURITY_QUESTIONS = [
    "What was the name of your first pet?",
    "What city were you born in?",
    "What was the name of your first school?",
    "What was the make of your first car?",
    "What is the name of your favorite teacher?",
    "What is the title of your favorite book?",
]

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_HASHER = PasswordHasher()


class UserStoreError(Exception):
    """Base error for the user store."""


class UserAlreadyExistsError(UserStoreError):
    """Raised when registering an email that already has an account."""


class UserValidationError(UserStoreError):
    """Raised when signup or update input is invalid."""


@dataclass(frozen=True)
class UserRecord:
    id: int
    email: str
    email_normalized: str
    user_key: str
    security_question: str
    is_active: bool
    created_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_email(email: str) -> str:
    return str(email).strip().lower()


def normalize_answer(answer: str) -> str:
    return " ".join(str(answer).strip().lower().split())


def compute_user_key(email_normalized: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", email_normalized).strip("_")
    digest = hashlib.sha256(email_normalized.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{digest}"


def _db_path(base_artifacts_root: str | Path) -> Path:
    users_dir = Path(base_artifacts_root) / ".users"
    users_dir.mkdir(parents=True, exist_ok=True)
    return users_dir / "users.db"


def _connect(base_artifacts_root: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path(base_artifacts_root)), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            email_normalized TEXT NOT NULL UNIQUE,
            user_key TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            security_question TEXT NOT NULL,
            security_answer_hash TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def _record_from_row(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        id=int(row["id"]),
        email=str(row["email"]),
        email_normalized=str(row["email_normalized"]),
        user_key=str(row["user_key"]),
        security_question=str(row["security_question"]),
        is_active=bool(row["is_active"]),
        created_at=str(row["created_at"]),
    )


def _fetch_active_row(conn: sqlite3.Connection, email_normalized: str) -> sqlite3.Row | None:
    cursor = conn.execute(
        "SELECT * FROM users WHERE email_normalized = ? AND is_active = 1",
        (email_normalized,),
    )
    return cursor.fetchone()


def create_user(
    base_artifacts_root: str | Path,
    email: str,
    password: str,
    security_question: str,
    security_answer: str,
) -> UserRecord:
    email_normalized = normalize_email(email)
    if not _EMAIL_PATTERN.match(email_normalized):
        raise UserValidationError("Invalid email address.")
    if len(str(password)) < MIN_PASSWORD_LENGTH:
        raise UserValidationError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if security_question not in SECURITY_QUESTIONS:
        raise UserValidationError("Unknown security question.")
    answer_normalized = normalize_answer(security_answer)
    if not answer_normalized:
        raise UserValidationError("Security answer is required.")

    user_key = compute_user_key(email_normalized)
    created_at = _now_iso()
    with closing(_connect(base_artifacts_root)) as conn, conn:
        try:
            cursor = conn.execute(
                """
                INSERT INTO users (
                    email, email_normalized, user_key, password_hash,
                    security_question, security_answer_hash, is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    str(email).strip(),
                    email_normalized,
                    user_key,
                    _HASHER.hash(password),
                    security_question,
                    _HASHER.hash(answer_normalized),
                    created_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise UserAlreadyExistsError("An account with this email already exists.") from exc
        user_id = int(cursor.lastrowid)
    logger.info("user_created", extra={"user_key": user_key})
    return UserRecord(
        id=user_id,
        email=str(email).strip(),
        email_normalized=email_normalized,
        user_key=user_key,
        security_question=security_question,
        is_active=True,
        created_at=created_at,
    )


def get_user_by_email(base_artifacts_root: str | Path, email: str) -> UserRecord | None:
    with closing(_connect(base_artifacts_root)) as conn:
        row = _fetch_active_row(conn, normalize_email(email))
    return _record_from_row(row) if row else None


def verify_user(base_artifacts_root: str | Path, email: str, password: str) -> UserRecord | None:
    with closing(_connect(base_artifacts_root)) as conn:
        row = _fetch_active_row(conn, normalize_email(email))
    if row is None:
        return None
    try:
        _HASHER.verify(str(row["password_hash"]), password)
    except VerifyMismatchError:
        return None
    except Exception:
        logger.exception("user_verify_failed", extra={"user_key": str(row["user_key"])})
        return None
    return _record_from_row(row)


def get_security_question(base_artifacts_root: str | Path, email: str) -> str | None:
    record = get_user_by_email(base_artifacts_root, email)
    return record.security_question if record else None


def verify_security_answer(base_artifacts_root: str | Path, email: str, answer: str) -> bool:
    with closing(_connect(base_artifacts_root)) as conn:
        row = _fetch_active_row(conn, normalize_email(email))
    if row is None:
        return False
    try:
        _HASHER.verify(str(row["security_answer_hash"]), normalize_answer(answer))
    except VerifyMismatchError:
        return False
    except Exception:
        logger.exception("security_answer_verify_failed", extra={"user_key": str(row["user_key"])})
        return False
    return True


def update_password(base_artifacts_root: str | Path, email: str, new_password: str) -> bool:
    if len(str(new_password)) < MIN_PASSWORD_LENGTH:
        raise UserValidationError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    email_normalized = normalize_email(email)
    with closing(_connect(base_artifacts_root)) as conn, conn:
        cursor = conn.execute(
            "UPDATE users SET password_hash = ? WHERE email_normalized = ? AND is_active = 1",
            (_HASHER.hash(new_password), email_normalized),
        )
        updated = cursor.rowcount > 0
    if updated:
        logger.info("user_password_updated", extra={"user_key": compute_user_key(email_normalized)})
    return updated
