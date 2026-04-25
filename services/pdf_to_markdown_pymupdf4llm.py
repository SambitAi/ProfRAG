from __future__ import annotations

import re
from pathlib import Path

import pymupdf4llm

from core.metadata import load_metadata, mark_step_success, save_metadata
from core.paths import build_images_dir, ensure_directory, slugify_filename
from core.storage import write_text

_IMAGE_REF_RE = re.compile(r'(!\[([^\]]*)\]\()([^)]+\.png)(\))', re.IGNORECASE)
_PAGE_IDX_RE = re.compile(r'.+-(\d+)-(\d+)\.png$', re.IGNORECASE)


def convert_pdf_to_markdown(
    input_pdf_path: str | Path,
    output_markdown_path: str | Path,
    images_dir: str | Path,
) -> tuple[str, list[str]]:
    input_pdf_path = Path(input_pdf_path)
    output_markdown_path = Path(output_markdown_path)
    images_dir = Path(images_dir)
    ensure_directory(images_dir)

    # pymupdf4llm writes images relative to image_path directory
    raw_md: str = pymupdf4llm.to_markdown(
        str(input_pdf_path),
        write_images=True,
        image_path=str(images_dir),
    )

    doc_slug = slugify_filename(input_pdf_path.name)

    # Rename images to normalised names and rewrite markdown refs
    existing_before = {f.name for f in images_dir.glob("*.png")}
    page_img_counter: dict[int, int] = {}
    image_paths: list[str] = []

    def _rewrite_ref(match: re.Match) -> str:
        orig_name = Path(match.group(3)).name
        m = _PAGE_IDX_RE.match(orig_name)
        page_n = int(m.group(1)) if m else 0
        img_idx = page_img_counter.get(page_n, 0)
        page_img_counter[page_n] = img_idx + 1
        new_name = f"{doc_slug}_page_{page_n}_img_{img_idx}.png"

        src = images_dir / orig_name
        dst = images_dir / new_name
        if src.exists() and src != dst:
            src.rename(dst)

        image_paths.append(str(dst))
        return f"{match.group(1)}images/{new_name}{match.group(4)}"

    modified_md = _IMAGE_REF_RE.sub(_rewrite_ref, raw_md)

    # Also pick up any images written by pymupdf4llm that weren't in a ref
    for f in images_dir.glob("*.png"):
        if f.name not in existing_before and str(f) not in image_paths:
            image_paths.append(str(f))

    ensure_directory(output_markdown_path.parent)
    write_text(output_markdown_path, modified_md)
    return str(output_markdown_path), image_paths


def run(
    input_pdf_path: str | Path,
    output_markdown_path: str | Path,
    document_folder: str | Path,
) -> str:
    document_folder = Path(document_folder)
    images_dir = build_images_dir(document_folder)

    markdown_path, image_paths = convert_pdf_to_markdown(
        input_pdf_path, output_markdown_path, images_dir
    )

    metadata = load_metadata(document_folder)
    metadata["markdown_path"] = markdown_path
    metadata["image_paths"] = image_paths
    save_metadata(document_folder, metadata)
    mark_step_success(document_folder, "pdf_to_markdown", {
        "markdown_path": markdown_path,
        "image_count": len(image_paths),
        "pdf_parser": "pymupdf4llm",
    })
    return markdown_path
