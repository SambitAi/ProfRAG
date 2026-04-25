from __future__ import annotations

import re
from pathlib import Path

from core.metadata import load_metadata, mark_step_success, save_metadata
from core.paths import ensure_directory, slugify_filename
from core.storage import read_text, write_json


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - chunk_overlap, 0)

    return chunks


def run(
    input_markdown_path: str | Path,
    output_chunks_dir: str | Path,
    document_folder: str | Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    markdown_text = read_text(input_markdown_path)
    chunks_dir = ensure_directory(output_chunks_dir)
    metadata = load_metadata(document_folder)
    document_name = metadata["document_name"]
    chunk_prefix = slugify_filename(document_name)
    chunks = split_text_into_chunks(markdown_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunk_paths: list[str] = []
    for chunk_number, chunk_text in enumerate(chunks, start=1):
        chunk_id = f"{chunk_prefix}_chunk_{chunk_number:06d}"
        chunk_path = chunks_dir / f"{chunk_id}.json"
        write_json(
            chunk_path,
            {
                "chunk_id": chunk_id,
                "document_name": document_name,
                "chunk_number": chunk_number,
                "text": chunk_text,
            },
        )
        chunk_paths.append(str(chunk_path))

    metadata["chunk_paths"] = chunk_paths
    metadata["total_chunks"] = len(chunk_paths)
    metadata["last_chunk_processed"] = len(chunk_paths)
    save_metadata(document_folder, metadata)
    mark_step_success(
        document_folder,
        "markdown_chunker",
        {
            "chunks_dir": str(chunks_dir),
            "total_chunks": len(chunk_paths),
        },
    )
    return chunk_paths
