from __future__ import annotations

from pathlib import Path

from core.metadata import save_metadata
from core.paths import (
    build_document_folder_name,
    compute_document_id,
    ensure_directory,
    next_document_version,
    slugify_filename,
)
from core.storage import write_text


def save_scraped_web_page(
    markdown_text: str,
    document_name: str,
    artifacts_root: str | Path,
    source_url: str,
    version: int | None = None,
) -> str:
    root = ensure_directory(artifacts_root)
    document_version = version or next_document_version(root, document_name)
    document_folder = ensure_directory(root / build_document_folder_name(document_name, document_version))
    markdown_dir = ensure_directory(document_folder / "markdown")
    markdown_path = markdown_dir / "document.md"
    write_text(markdown_path, markdown_text)

    metadata = {
        "document_name": document_name,
        "document_slug": slugify_filename(document_name),
        "document_id": compute_document_id(document_name, source_url),
        "document_version": document_version,
        "document_folder": str(document_folder),
        "source_url": source_url,
        "source_pdf_path": None,
        "last_successful_step": "pdf_to_markdown",
        "ready_to_chat": False,
        "total_chunks": 0,
        "last_chunk_processed": 0,
        "vector_store": {},
        "steps": {
            "upload": {
                "status": "success",
                "outputs": {"source_url": source_url},
            },
            "pdf_to_markdown": {
                "status": "success",
                "outputs": {
                    "markdown_path": str(markdown_path),
                    "scraper": "trafilatura",
                },
            },
        },
    }
    save_metadata(document_folder, metadata)
    return str(document_folder)
