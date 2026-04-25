from __future__ import annotations

from pathlib import Path
from typing import Any

from core.storage import read_json, write_json


DEFAULT_LAST_SUCCESSFUL_STEP = "created"


def load_metadata(document_folder: str | Path) -> dict[str, Any]:
    return read_json(Path(document_folder) / "metadata.json", default={})


def save_metadata(document_folder: str | Path, metadata: dict[str, Any]) -> None:
    write_json(Path(document_folder) / "metadata.json", metadata)


def mark_step_success(
    document_folder: str | Path,
    step_name: str,
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = load_metadata(document_folder)
    steps = metadata.setdefault("steps", {})
    steps[step_name] = {
        "status": "success",
        "outputs": outputs or {},
    }
    metadata["last_successful_step"] = step_name
    save_metadata(document_folder, metadata)
    return metadata


def update_summary_progress(document_folder: str | Path, key: str, value: Any) -> None:
    metadata = load_metadata(document_folder)
    progress = metadata.setdefault("summary_progress", {})
    progress[key] = value
    metadata["summary_status"] = "in_progress"
    save_metadata(document_folder, metadata)


def mark_summary_complete(
    document_folder: str | Path,
    summary_paths: dict[str, str],
    summary_brief: str = "",
) -> None:
    metadata = load_metadata(document_folder)
    metadata["summary_ready"] = True
    metadata["summary_status"] = "ready"
    metadata["summary_paths"] = summary_paths
    if summary_brief:
        metadata["summary_brief"] = summary_brief
    save_metadata(document_folder, metadata)
