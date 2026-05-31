from __future__ import annotations

from pathlib import Path
from typing import Any

from core.metadata import load_metadata
from core.storage import read_json


def read_chunk(chunk_path: str | Path) -> dict[str, Any]:
    return read_json(chunk_path, default={})


def read_prev_chunk(document_folder: str | Path, chunk_number: int) -> dict[str, Any]:
    """Return the chunk JSON for chunk N-1, or {} if at start or folder unknown."""
    if not document_folder or not chunk_number:
        return {}
    metadata = load_metadata(document_folder)
    chunk_paths = metadata.get("chunk_paths", [])
    prev_idx = chunk_number - 2  # chunk_number is 1-indexed; prev chunk is at index chunk_number-2
    if 0 <= prev_idx < len(chunk_paths):
        return read_json(chunk_paths[prev_idx], default={})
    return {}


def read_next_chunk(document_folder: str | Path, chunk_number: int) -> dict[str, Any]:
    """Return the chunk JSON for chunk N+1, or {} if at end or folder unknown."""
    if not document_folder or not chunk_number:
        return {}
    metadata = load_metadata(document_folder)
    chunk_paths = metadata.get("chunk_paths", [])
    next_idx = chunk_number  # chunk_number is 1-indexed; next chunk is at index chunk_number
    if next_idx < len(chunk_paths):
        return read_json(chunk_paths[next_idx], default={})
    return {}


def resolve_top_chunk_images(answer_payload: dict[str, Any], default_folder: str) -> list[str]:
    """Find images near the top source chunk; return absolute, deduplicated paths.

    Falls back from the chunk itself to next chunk to previous chunk, since the
    chunk that best answers the query may not be where the image appears.
    """
    sources = answer_payload.get("sources", [])
    if not sources:
        return []
    # In v1.1 tree retrieval, top source may not be the chunk nearest to a figure.
    # Try several top sources and return the first successful image match.
    raw: list[str] = []
    for chosen in sources[:8]:
        folder = chosen.get("document_folder") or default_folder
        chunk_num = int(chosen.get("chunk_number", 0) or 0)
        chunk_path = chosen.get("chunk_path", "")
        section_path = chosen.get("section_path", "")

        raw = read_chunk(chunk_path).get("image_paths", []) if chunk_path else []
        if not raw and chunk_num:
            raw = read_next_chunk(folder, chunk_num).get("image_paths", [])
        if not raw and chunk_num:
            raw = read_prev_chunk(folder, chunk_num).get("image_paths", [])
        # Section-level fallback: choose first chunk in same section that has images.
        if not raw and section_path:
            meta = load_metadata(folder)
            for cp in meta.get("chunk_paths", []):
                c = read_json(cp, default={})
                if (c.get("section_path") or c.get("h1") or "") == section_path and c.get("image_paths"):
                    raw = c.get("image_paths", [])
                    break
        if raw:
            break

    project_root = Path(__file__).resolve().parent.parent
    resolved: list[str] = []
    for p in dict.fromkeys(raw):
        if not p:
            continue
        ph = Path(p)
        abs_path = ph if ph.is_absolute() else project_root / p
        if abs_path.exists():
            resolved.append(str(abs_path))
    return resolved
