from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from core.paths import ensure_directory


class LockTimeoutError(TimeoutError):
    pass


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _clear_stale_lock(path: Path) -> bool:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        pid = int(raw) if raw else -1
    except Exception:
        pid = -1
    if _is_pid_alive(pid):
        return False
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


@contextmanager
def _file_lock(lock_path: str | Path, timeout_seconds: float = 30.0, poll_interval_seconds: float = 0.1) -> Iterator[None]:
    path = Path(lock_path)
    ensure_directory(path.parent)
    deadline = time.time() + timeout_seconds

    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            _clear_stale_lock(path)
            if time.time() >= deadline:
                raise LockTimeoutError(f"Timed out acquiring lock: {path}")
            time.sleep(poll_interval_seconds)

    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        fd = -1
        yield
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Best-effort cleanup; stale lock handling can be added later if needed.
            pass


def _locks_dir_for_artifacts(artifacts_root: str | Path) -> Path:
    return ensure_directory(Path(artifacts_root) / ".locks")


@contextmanager
def global_index_lock(global_index_path: str | Path, timeout_seconds: float = 30.0) -> Iterator[None]:
    index_path = Path(global_index_path)
    artifacts_root = index_path.parent
    lock_path = _locks_dir_for_artifacts(artifacts_root) / "global_index.lock"
    with _file_lock(lock_path, timeout_seconds=timeout_seconds):
        yield


@contextmanager
def document_lock(document_folder: str | Path, timeout_seconds: float = 30.0) -> Iterator[None]:
    folder = Path(document_folder)
    artifacts_root = folder.parent
    lock_name = f"{folder.name}.lock"
    lock_path = _locks_dir_for_artifacts(artifacts_root) / lock_name
    with _file_lock(lock_path, timeout_seconds=timeout_seconds):
        yield


@contextmanager
def job_store_lock(artifacts_root: str | Path, timeout_seconds: float = 30.0) -> Iterator[None]:
    lock_path = _locks_dir_for_artifacts(artifacts_root) / "job_store.lock"
    with _file_lock(lock_path, timeout_seconds=timeout_seconds):
        yield
