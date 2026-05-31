from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
import re


def _slug(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return token or "default"


def _with_embedding_suffix(base_name: str, embedding_model: str, enabled: bool) -> str:
    if not enabled:
        return base_name
    # Stable per-embedding-model collection routing avoids dimension mismatches
    # when users switch embedding models (e.g. 768 -> 1536).
    return f"{base_name}_{_slug(embedding_model)}"


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

    embeddings = config.setdefault("embeddings", {})
    embedding_model = embeddings.get("model", "")
    auto_suffix = bool(vector_db.get("auto_collection_suffix", False))
    # Apply model-aware names to all embedding-backed collections.
    if "collection_name" in vector_db:
        vector_db["collection_name"] = _with_embedding_suffix(
            vector_db["collection_name"], embedding_model, auto_suffix
        )
    if "summary_collection_name" in vector_db:
        vector_db["summary_collection_name"] = _with_embedding_suffix(
            vector_db["summary_collection_name"], embedding_model, auto_suffix
        )
    if "card_collection_name" in vector_db:
        vector_db["card_collection_name"] = _with_embedding_suffix(
            vector_db["card_collection_name"], embedding_model, auto_suffix
        )
    return config
