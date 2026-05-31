from __future__ import annotations

from pathlib import Path
from typing import Any

from services import url_ingest
import workflows as _workflows


# Compatibility shim during migration: keep UI imports stable while orchestration lives in workflows.py.
# Re-export via module import for UI compatibility: pipeline.url_ingest.url_to_document_name(...)


def list_documents(config_path: str | Path) -> list[dict[str, Any]]:
    return _workflows.list_documents(config_path)


def inspect_same_name_document(config_path: str | Path, file_name: str) -> dict[str, Any] | None:
    return _workflows.inspect_same_name_document(config_path, file_name)




def prepare_document(
    config_path: str | Path,
    file_name: str,
    file_bytes: bytes,
    user_choice: str,
) -> dict[str, Any]:
    return _workflows.prepare_document(config_path, file_name, file_bytes, user_choice)


def prepare_url_document(
    config_path: str | Path,
    url: str,
    user_choice: str,
) -> dict[str, Any]:
    return _workflows.prepare_url_document(config_path, url, user_choice)


def load_document(document_folder: str | Path) -> dict[str, Any]:
    return _workflows.load_document(document_folder)


def find_relevant_documents(config_path: str | Path, question: str) -> list[dict[str, Any]]:
    return _workflows.find_relevant_documents(config_path, question)


def ask_multi_document_question(
    config_path: str | Path,
    document_folders: list[str],
    question: str,
) -> dict[str, Any]:
    return _workflows.ask_multi_document_question(config_path, document_folders, question)


def start_summarization_background(config_path: str | Path, document_folder: str | Path) -> None:
    return _workflows.start_summarization_background(config_path, document_folder)


def start_summary_watcher(config_path: str | Path) -> None:
    return _workflows.start_summary_watcher(config_path)


def reset_summary_level(config_path: str | Path, document_folder: str | Path, level: str) -> None:
    return _workflows.reset_summary_level(config_path, document_folder, level)


def ask_question(config_path: str | Path, document_folder: str | Path, question: str) -> dict[str, Any]:
    return _workflows.ask_question(config_path, document_folder, question)

