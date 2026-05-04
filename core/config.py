from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_app_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    vector_db = config.setdefault("vector_db", {})
    # Centralize env-derived Chroma connectivity so services consume config only.
    chroma_host = os.getenv("CHROMA_HOST")
    if chroma_host:
        vector_db["host"] = chroma_host
    if "port" not in vector_db:
        vector_db["port"] = int(os.getenv("CHROMA_PORT", "8000"))
    return config
