from __future__ import annotations

from typing import Any

from core.tree_retrieval import retrieve_tree


def retrieve(
    question: str,
    config: dict[str, Any],
    selected_document_folders: list[str] | None = None,
) -> dict[str, Any]:
    return retrieve_tree(question, config, selected_document_folders=selected_document_folders)
