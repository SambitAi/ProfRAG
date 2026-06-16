from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

# Matches the user_key format produced by core.user_store.compute_user_key:
# "<slug>__<sha256[:8] hex>". Document folders end in "_vN" and never match.
_USER_KEY_RE = re.compile(r"^.+__[0-9a-f]{8}$")

# Top-level service directories under the base artifacts root that must never
# be treated as user roots (mirrors the exclusion in core.global_index).
_INFRA_DIR_NAMES = frozenset({"jobs", "chroma_db"})


def looks_like_user_key(name: str) -> bool:
    return bool(_USER_KEY_RE.match(name or ""))


def iter_user_roots(base_root: str | Path) -> Iterator[Path]:
    """Yield per-user root directories directly under the base artifacts root.

    A user root must match the user_key naming format, must not be a known
    infra directory, and must not itself be a document folder (those carry
    their own metadata.json).
    """
    base = Path(base_root)
    if not base.exists():
        return
    for sub in base.iterdir():
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        if sub.name in _INFRA_DIR_NAMES:
            continue
        if not looks_like_user_key(sub.name):
            continue
        if (sub / "metadata.json").exists():
            continue
        yield sub


def user_collection_names(config: dict[str, Any], user_key: str) -> dict[str, str]:
    """Return per-user Chroma collection names in the <base>__u__<user_key> format."""
    vdb = config.get("vector_db", {})
    base_chunks = str(vdb.get("collection_name", "pdf_rag_chunks"))
    base_summaries = str(vdb.get("summary_collection_name", "pdf_rag_summaries"))
    base_cards = str(vdb.get("card_collection_name", "pdf_rag_cards"))
    return {
        "chunks": f"{base_chunks}__u__{user_key}",
        "summaries": f"{base_summaries}__u__{user_key}",
        "cards": f"{base_cards}__u__{user_key}",
    }


def patch_config_for_user(config: dict[str, Any], user_key: str) -> dict[str, Any]:
    """Return a shallow copy of config with user-scoped collection names and persist dir.

    Call this AFTER the artifacts_root patch so config["paths"]["artifacts_root"] already
    points to the user subtree; the persist dir is derived from it.
    """
    if not user_key:
        return config
    names = user_collection_names(config, user_key)
    vdb = config.get("vector_db", {})
    # PersistentClient: each user gets their own chroma_db under their artifacts root.
    # HttpClient: shared server — isolation is purely at the collection-name level.
    if not vdb.get("host"):
        user_persist_dir = str(Path(config["paths"]["artifacts_root"]) / "chroma_db")
    else:
        user_persist_dir = str(vdb.get("persist_directory", "artifacts/chroma_db"))
    return {
        **config,
        "vector_db": {
            **vdb,
            "collection_name": names["chunks"],
            "summary_collection_name": names["summaries"],
            "card_collection_name": names["cards"],
            "persist_directory": user_persist_dir,
        },
    }
