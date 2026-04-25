from __future__ import annotations

from pathlib import Path

from core.metadata import save_metadata
from core.paths import build_document_folder_name, ensure_directory, next_document_version, slugify_filename


def save_uploaded_pdf(
    file_bytes: bytes,
    original_file_name: str,
    artifacts_root: str | Path,
    version: int | None = None,
) -> str:
    root = ensure_directory(artifacts_root)
    document_version = version or next_document_version(root, original_file_name)
    document_folder = ensure_directory(root / build_document_folder_name(original_file_name, document_version))
    source_folder = ensure_directory(document_folder / "source")
    source_path = source_folder / original_file_name
    source_path.write_bytes(file_bytes)

    metadata = {
        "document_name": original_file_name,
        "document_slug": slugify_filename(original_file_name),
        "document_version": document_version,
        "document_folder": str(document_folder),
        "source_pdf_path": str(source_path),
        "last_successful_step": "upload",
        "ready_to_chat": False,
        "total_chunks": 0,
        "last_chunk_processed": 0,
        "vector_store": {},
        "steps": {
            "upload": {
                "status": "success",
                "outputs": {
                    "source_pdf_path": str(source_path),
                },
            }
        },
    }
    save_metadata(document_folder, metadata)
    return str(document_folder)
