from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def read_json(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return {} if default is None else default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a unique temp file to avoid collisions across concurrent writers.
    tmp_path = file_path.with_suffix(f"{file_path.suffix}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.flush()
        os.fsync(handle.fileno())
    # Windows can transiently deny replace if another process has metadata.json open.
    # Retry briefly on known lock/permission races (WinError 5/32).
    delay = 0.02
    for attempt in range(10):
        try:
            os.replace(tmp_path, file_path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(delay)
            delay *= 1.7
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if winerror in (5, 32) and attempt < 9:
                time.sleep(delay)
                delay *= 1.7
                continue
            raise
    # Best-effort temp cleanup if replace never succeeded.
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except OSError:
        pass


def write_text(path: str | Path, content: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def read_text(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as handle:
        return handle.read()
