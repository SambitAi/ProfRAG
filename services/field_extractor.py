from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any

from core.metadata import load_metadata, save_metadata
from core.storage import read_text

logger = logging.getLogger(__name__)

def _to_float(value: str) -> float:
    cleaned = value.replace(",", "").replace("$", "").strip()
    return float(cleaned)


def _to_int(value: str) -> int:
    cleaned = value.replace(",", "").strip()
    return int(cleaned)


def _cast(value: str, field_type: str) -> Any:
    t = (field_type or "string").lower()
    if t == "float":
        return _to_float(value)
    if t == "int":
        return _to_int(value)
    # Keep date/string as string for now (typed but lossless).
    return value.strip()


def _select_profile(document_name: str, profiles: dict[str, Any]) -> dict[str, Any] | None:
    name = (document_name or "").lower()
    for profile_name, profile in (profiles or {}).items():
        if not isinstance(profile, dict):
            continue
        match_any = profile.get("match_any", [])
        if not match_any:
            # If no match rules, treat profile as opt-in template only.
            continue
        for token in match_any:
            if str(token).lower() in name:
                picked = dict(profile)
                picked["_profile_name"] = profile_name
                return picked
    return None


def run(document_folder: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    folder = Path(document_folder)
    meta = load_metadata(folder)
    fx = config.get("field_extraction", {}) or {}
    if not bool(fx.get("enabled", False)):
        return {}

    profile = _select_profile(meta.get("document_name", folder.name), fx.get("profiles", {}) or {})
    if not profile:
        meta["extracted_fields"] = {}
        save_metadata(folder, meta)
        return {}

    markdown_path = meta.get("markdown_path", str(folder / "markdown" / "document.md"))
    text = read_text(markdown_path)

    fields_cfg = profile.get("fields", {}) or {}
    extracted: dict[str, Any] = {}
    for field_name, rule in fields_cfg.items():
        if not isinstance(rule, dict):
            continue
        pattern = rule.get("regex", "")
        if not pattern:
            continue
        flags = re.IGNORECASE | re.MULTILINE
        m = re.search(pattern, text, flags)
        if not m:
            continue
        value = m.group(1) if m.groups() else m.group(0)
        try:
            extracted[field_name] = _cast(str(value), str(rule.get("type", "string")))
        except (ValueError, TypeError):
            # Keep pipeline resilient: skip invalid cast for this field.
            logger.warning("Failed to cast extracted field '%s' in %s", field_name, folder)
            continue

    meta["extracted_fields_profile"] = profile.get("_profile_name", "")
    meta["extracted_fields"] = extracted
    save_metadata(folder, meta)
    return extracted
