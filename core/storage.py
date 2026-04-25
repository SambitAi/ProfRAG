from __future__ import annotations

import json
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


def write_text(path: str | Path, content: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def read_text(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as handle:
        return handle.read()
