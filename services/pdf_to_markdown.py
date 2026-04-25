from __future__ import annotations

from pathlib import Path

import fitz

from core.metadata import load_metadata, mark_step_success, save_metadata
from core.paths import ensure_directory
from core.storage import write_text


def convert_pdf_to_markdown(input_pdf_path: str | Path, output_markdown_path: str | Path) -> str:
    document = fitz.open(input_pdf_path)
    pages: list[str] = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        pages.append(f"# Page {page_number}\n\n{text}")

    markdown_text = "\n\n".join(pages).strip()
    if not markdown_text:
        raise RuntimeError("No extractable text was found in the PDF.")

    ensure_directory(Path(output_markdown_path).parent)
    write_text(output_markdown_path, markdown_text)
    return str(output_markdown_path)


def run(input_pdf_path: str | Path, output_markdown_path: str | Path, document_folder: str | Path) -> str:
    markdown_path = convert_pdf_to_markdown(input_pdf_path, output_markdown_path)
    metadata = load_metadata(document_folder)
    metadata["markdown_path"] = markdown_path
    save_metadata(document_folder, metadata)
    mark_step_success(document_folder, "pdf_to_markdown", {"markdown_path": markdown_path})
    return markdown_path
