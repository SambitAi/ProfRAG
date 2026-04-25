from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

from core.metadata import load_metadata, mark_step_success, save_metadata
from core.paths import build_tables_dir, ensure_directory, slugify_filename
from core.storage import read_text, write_json

_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")
_PAGE_IN_NAME_RE = re.compile(r"_page_(\d+)_img_\d+")


def _is_separator(line: str) -> bool:
    inner = line.strip().strip("|")
    return bool(inner) and all(c in " -:" for c in inner)


def _extract_tables(lines: list[str]) -> list[tuple[list[str], int, int]]:
    tables: list[tuple[list[str], int, int]] = []
    current: list[str] = []
    start_idx = -1
    for i, line in enumerate(lines):
        if _TABLE_LINE_RE.match(line):
            if not current:
                start_idx = i
            current.append(line)
        else:
            if current:
                tables.append((current, start_idx, i - 1))
                current = []
                start_idx = -1
    if current:
        tables.append((current, start_idx, len(lines) - 1))
    return tables


def _pipe_table_to_csv(table_lines: list[str]) -> str:
    rows = []
    for line in table_lines:
        if _is_separator(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue()


def _extract_caption(table_lines: list[str]) -> tuple[str | None, str | None, str]:
    """Return (title, headers, caption) where caption = 'title — headers'."""
    title: str | None = None
    headers: str | None = None
    for line in table_lines:
        if _is_separator(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        non_empty = [c for c in cells if c]
        if not non_empty:
            continue
        if len(non_empty) == 1 and title is None:
            title = _BOLD_RE.sub(r"\1", non_empty[0]).strip()
        elif headers is None:
            headers = " | ".join(_BOLD_RE.sub(r"\1", c) for c in non_empty)
    parts = [p for p in [title, headers] if p]
    caption = " — ".join(parts) if parts else ""
    return title, headers, caption


def _context_around(
    lines: list[str], start: int, end: int, max_lines: int = 3
) -> tuple[str | None, str | None]:
    def is_prose(s: str) -> bool:
        return bool(s) and not s.startswith("#") and not _TABLE_LINE_RE.match(s)

    prev_parts: list[str] = []
    for k in range(start - 1, -1, -1):
        s = lines[k].strip()
        if s.startswith("#"):
            break
        if is_prose(s):
            prev_parts.append(s)
            if len(prev_parts) >= max_lines:
                break

    next_parts: list[str] = []
    for k in range(end + 1, len(lines)):
        s = lines[k].strip()
        if s.startswith("#"):
            break
        if is_prose(s):
            next_parts.append(s)
            if len(next_parts) >= max_lines:
                break

    return (
        " ".join(reversed(prev_parts)) if prev_parts else None,
        " ".join(next_parts) if next_parts else None,
    )


def _infer_page(lines: list[str], start: int, window: int = 30) -> int | None:
    search_start = max(0, start - window)
    search_end = min(len(lines), start + window)
    # Search nearest lines first (expanding outward from start)
    candidates = sorted(range(search_start, search_end), key=lambda i: abs(i - start))
    for i in candidates:
        for ref in _IMAGE_REF_RE.findall(lines[i]):
            m = _PAGE_IN_NAME_RE.search(ref)
            if m:
                return int(m.group(1))
    return None


def run(markdown_path: str | Path, document_folder: str | Path) -> list[str]:
    markdown_path = Path(markdown_path)
    document_folder = Path(document_folder)
    text = read_text(markdown_path)
    lines = text.split("\n")

    tables_dir = ensure_directory(build_tables_dir(document_folder)).resolve()
    doc_slug = slugify_filename(load_metadata(document_folder)["document_name"])

    csv_paths: list[str] = []
    table_index: dict[str, Any] = {}

    for t_idx, (table_lines, t_start, t_end) in enumerate(_extract_tables(lines), start=1):
        csv_str = _pipe_table_to_csv(table_lines)
        if not csv_str.strip():
            continue
        name = f"{doc_slug}_table_{t_idx}"
        tpath = (tables_dir / f"{name}.csv").resolve()
        tpath.write_text(csv_str, encoding="utf-8")
        csv_paths.append(str(tpath))

        title, headers, caption = _extract_caption(table_lines)
        prev_text, next_text = _context_around(lines, t_start, t_end)
        page = _infer_page(lines, t_start)

        table_index[name] = {
            "csv_path": str(tpath),
            "start_line": t_start,
            "end_line": t_end,
            "caption": caption,
            "title": title,
            "headers": headers,
            "prev_text": prev_text,
            "next_text": next_text,
            "page": page,
        }

    if table_index:
        write_json(tables_dir / "index.json", table_index)

    metadata = load_metadata(document_folder)
    metadata["table_paths"] = csv_paths
    metadata["tables_dir"] = str(tables_dir)
    save_metadata(document_folder, metadata)
    mark_step_success(document_folder, "extract_tables", {
        "total_tables": len(csv_paths),
        "tables_dir": str(tables_dir),
    })
    return csv_paths
