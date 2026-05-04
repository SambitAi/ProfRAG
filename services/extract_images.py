from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.metadata import mark_step_success
from core.paths import build_images_dir, ensure_directory
from core.storage import read_text, write_json

_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")
_IMAGE_REF_FULL_RE = re.compile(r"!\[[^\]]*\]\(images/[^)]+\)")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)")
_PAGE_IN_NAME_RE = re.compile(r"_page_(\d+)_img_\d+")
_OCR_BLOCK_START = "**----- Start of picture text -----**"
_OCR_BLOCK_END = "**----- End of picture text -----**"
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _parse_ocr_block(lines: list[str], j: int) -> tuple[str, int]:
    """Given j pointing at the line containing _OCR_BLOCK_START, return (caption, next_j).

    Handles three layouts produced by pymupdf4llm:
    1. Single line: **start**<br>content<br>**end**<br>
    2. Start + content on same line, end on later line
    3. Start on own line, content on subsequent lines (most common)
    """
    start_line = lines[j]
    after_start = start_line.split(_OCR_BLOCK_START, 1)[1]

    if _OCR_BLOCK_END in after_start:
        # Entire block on one line
        inner = after_start.split(_OCR_BLOCK_END, 1)[0]
        return _BR_RE.sub(" ", inner).strip(), j + 1

    # Multi-line: capture any inline content after the start marker, then read on
    ocr_lines: list[str] = []
    inline = _BR_RE.sub(" ", after_start).strip()
    if inline:
        ocr_lines.append(inline)
    j += 1
    while j < len(lines):
        if _OCR_BLOCK_END in lines[j]:
            j += 1
            break
        cleaned = _BR_RE.sub(" ", lines[j]).strip()
        if cleaned:
            ocr_lines.append(cleaned)
        j += 1
    return " ".join(ocr_lines).strip(), j


def _collect_ocr_captions(lines: list[str]) -> dict[str, str]:
    """Return {image_rel_path: ocr_caption} without modifying text."""
    ocr_captions: dict[str, str] = {}
    i = 0
    while i < len(lines):
        refs = _IMAGE_REF_RE.findall(lines[i])
        i += 1
        if refs:
            j = i
            while j < min(i + 5, len(lines)):
                peek = lines[j].strip()
                if _OCR_BLOCK_START in peek:
                    caption, j = _parse_ocr_block(lines, j)
                    for ref in refs:
                        ocr_captions[ref] = caption
                    i = j
                    break
                elif peek:
                    break
                else:
                    j += 1
    return ocr_captions


def _build_image_index(
    lines: list[str],
    document_folder: Path,
    ocr_captions: dict[str, str],
) -> dict[str, Any]:
    index: dict[str, Any] = {}

    def _section_path_at_line(idx: int) -> str:
        h1: str | None = None
        h2: str | None = None
        h3: str | None = None
        for k in range(0, min(idx + 1, len(lines))):
            m = _HEADING_RE.match(lines[k].strip())
            if not m:
                continue
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1:
                h1, h2, h3 = text, None, None
            elif level == 2:
                h2, h3 = text, None
            elif level == 3:
                h3 = text
        return " > ".join(part for part in (h1, h2, h3) if part)

    def _nearest_text_before(idx: int) -> str | None:
        for k in range(idx - 1, -1, -1):
            s = lines[k].strip()
            if s and not _IMAGE_REF_FULL_RE.search(s) and not s.startswith("#"):
                return s
        return None

    def _nearest_text_after(idx: int) -> str | None:
        for k in range(idx + 1, len(lines)):
            s = lines[k].strip()
            if s and not _IMAGE_REF_FULL_RE.search(s) and not s.startswith("#"):
                return s
        return None

    def _find_title(idx: int) -> str | None:
        # Scan forward, skipping over OCR block content
        in_ocr = False
        for k in range(idx + 1, min(idx + 15, len(lines))):
            s = lines[k].strip()
            if _OCR_BLOCK_START in s:
                in_ocr = _OCR_BLOCK_END not in s  # single-line block: in_ocr stays False
                continue
            if in_ocr:
                if _OCR_BLOCK_END in s:
                    in_ocr = False
                continue
            if s.startswith("#"):
                break
            if s and not _IMAGE_REF_FULL_RE.search(s):
                return s
        # Fall back to before the image ref
        for k in range(idx - 1, max(idx - 4, -1), -1):
            s = lines[k].strip()
            if s.startswith("#"):
                break
            if s and not _IMAGE_REF_FULL_RE.search(s):
                return s
        return None

    for i, line in enumerate(lines):
        for ref in _IMAGE_REF_RE.findall(line):
            abs_path = str((document_folder.resolve() / ref))
            ocr_text: str | None = ocr_captions.get(ref)
            caption: str | None = ocr_text
            caption_line_idx: int | None = None

            if caption is None:
                for j in range(i + 1, min(i + 4, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    if next_line.startswith("#") or next_line.startswith("|"):
                        break
                    if len(next_line) <= 200:
                        caption = next_line
                        caption_line_idx = j
                    break

            prev_text = _nearest_text_before(i)
            if ocr_text:
                next_text: str | None = ocr_text
            elif caption_line_idx is not None:
                next_text = _nearest_text_after(caption_line_idx)
            else:
                next_text = _nearest_text_after(i)
            title = _find_title(i)

            page_m = _PAGE_IN_NAME_RE.search(ref)
            page = int(page_m.group(1)) if page_m else 0
            img_filename = Path(ref).name

            index[img_filename] = {
                "abs_path": abs_path,
                "caption": caption,
                "ocr_text": ocr_text,
                "title": title,
                "prev_text": prev_text,
                "next_text": next_text,
                "page": page,
                "start_line": i,
                "section_path": _section_path_at_line(i),
                "isolated": not prev_text or not next_text,
                "ref": ref,
            }

    return index


def run(markdown_path: str | Path, document_folder: str | Path) -> str:
    markdown_path = Path(markdown_path)
    document_folder = Path(document_folder)
    lines = read_text(markdown_path).splitlines()

    ocr_captions = _collect_ocr_captions(lines)
    image_index = _build_image_index(lines, document_folder, ocr_captions)

    images_dir = build_images_dir(document_folder)
    index_path = images_dir / "index.json"
    if image_index:
        ensure_directory(images_dir)
        write_json(index_path, image_index)

    mark_step_success(document_folder, "extract_images", {"image_count": len(image_index)})
    return str(index_path)
