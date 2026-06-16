from __future__ import annotations

from pathlib import Path
from typing import Any

from services import url_ingest
import workflows as _workflows


# Compatibility shim during migration: keep UI imports stable while orchestration lives in workflows.py.
# Re-export via module import for UI compatibility: pipeline.url_ingest.url_to_document_name(...)
# All functions accept the user-scoped artifacts_root / user_key from the UI session context
# and forward them to the workflow layer (None preserves single-tenant behavior).


def list_documents(config_path: str | Path, artifacts_root: str | None = None) -> list[dict[str, Any]]:
    return _workflows.list_documents(config_path, artifacts_root)


def inspect_same_name_document(
    config_path: str | Path,
    file_name: str,
    artifacts_root: str | None = None,
) -> dict[str, Any] | None:
    return _workflows.inspect_same_name_document(config_path, file_name, artifacts_root)


def prepare_document(
    config_path: str | Path,
    file_name: str,
    file_bytes: bytes,
    user_choice: str,
    artifacts_root: str | None = None,
    user_key: str | None = None,
) -> dict[str, Any]:
    return _workflows.prepare_document(config_path, file_name, file_bytes, user_choice, artifacts_root, user_key=user_key)


def prepare_url_document(
    config_path: str | Path,
    url: str,
    user_choice: str,
    artifacts_root: str | None = None,
    user_key: str | None = None,
) -> dict[str, Any]:
    return _workflows.prepare_url_document(config_path, url, user_choice, artifacts_root, user_key=user_key)


def load_document(document_folder: str | Path, artifacts_root: str | None = None) -> dict[str, Any]:
    return _workflows.load_document(document_folder, artifacts_root)


def find_relevant_documents(
    config_path: str | Path,
    question: str,
    artifacts_root: str | None = None,
    user_key: str | None = None,
) -> list[dict[str, Any]]:
    return _workflows.find_relevant_documents(config_path, question, artifacts_root, user_key=user_key)


def ask_multi_document_question(
    config_path: str | Path,
    document_folders: list[str],
    question: str,
    artifacts_root: str | None = None,
    user_key: str | None = None,
) -> dict[str, Any]:
    return _workflows.ask_multi_document_question(
        config_path, document_folders, question, artifacts_root=artifacts_root, user_key=user_key
    )


def start_summarization_background(
    config_path: str | Path,
    document_folder: str | Path,
    artifacts_root: str | None = None,
    user_key: str | None = None,
) -> None:
    return _workflows.start_summarization_background(
        config_path, document_folder, artifacts_root=artifacts_root, user_key=user_key
    )


def start_summary_watcher(
    config_path: str | Path,
    artifacts_root: str | None = None,
    user_key: str | None = None,
) -> None:
    return _workflows.start_summary_watcher(config_path, artifacts_root, user_key=user_key)


def reset_summary_level(
    config_path: str | Path,
    document_folder: str | Path,
    level: str,
    artifacts_root: str | None = None,
    user_key: str | None = None,
) -> None:
    return _workflows.reset_summary_level(config_path, document_folder, level, artifacts_root=artifacts_root, user_key=user_key)


def ask_question(
    config_path: str | Path,
    document_folder: str | Path,
    question: str,
    artifacts_root: str | None = None,
    user_key: str | None = None,
) -> dict[str, Any]:
    return _workflows.ask_question(config_path, document_folder, question, artifacts_root=artifacts_root, user_key=user_key)


def delete_documents(
    config_path: str | Path,
    folders: list[str],
    artifacts_root: str | None = None,
    user_key: str | None = None,
) -> dict[str, Any]:
    return _workflows.delete_documents(config_path, folders, artifacts_root, user_key=user_key)
