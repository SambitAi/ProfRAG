from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.metadata import load_metadata, mark_step_success, save_metadata
from core.paths import (
    build_chunk_name,
    build_images_dir,
    build_section_path,
    build_sections_dir,
    build_tables_dir,
    ensure_directory,
    slugify_filename,
    slugify_heading,
)
from core.storage import read_text, read_json, write_json

# ── Patterns ────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)")
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")
_OCR_BLOCK_START = "**----- Start of picture text -----**"
_OCR_BLOCK_END = "**----- End of picture text -----**"


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class _Section:
    level: int
    heading_text: str
    lines: list[str] = field(default_factory=list)
    line_nums: list[int] = field(default_factory=list)
    children: list["_Section"] = field(default_factory=list)
    heading_line: int = 0

    @property
    def start_line(self) -> int:
        return self.heading_line

    @property
    def end_line(self) -> int:
        nums = list(self.line_nums)
        for child in self.children:
            nums.append(child.end_line)
        return max(nums) if nums else self.heading_line


@dataclass
class _ChunkCandidate:
    h1: str | None
    h2: str | None
    h3: str | None
    text: str
    section_level: int
    start_line: int = 0
    end_line: int = 0


# ── OCR block cleanup (keeps text clean for chunk storage) ───────────────────

def _strip_ocr_blocks(text: str) -> str:
    """Strip OCR marker blocks from chunk text so the LLM doesn't see raw OCR markers."""
    lines = text.splitlines()
    output: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        refs = _IMAGE_REF_RE.findall(line)
        output.append(line)
        i += 1
        if refs:
            j = i
            while j < min(i + 5, len(lines)):
                peek = lines[j].strip()
                if _OCR_BLOCK_START in peek:
                    if _OCR_BLOCK_END in peek:
                        # Single-line block: skip just this one line
                        j += 1
                    else:
                        # Multi-line block: skip until end marker
                        j += 1
                        while j < len(lines):
                            if _OCR_BLOCK_END in lines[j]:
                                j += 1
                                break
                            j += 1
                    i = j
                    break
                elif peek:
                    break
                else:
                    j += 1
    return "\n".join(output)


# ── Isolated image removal using pre-built index ─────────────────────────────

def _remove_isolated_refs(text: str, image_index: dict[str, Any]) -> str:
    """Remove image ref lines whose index entry is marked isolated."""
    isolated_refs = {v.get("ref", "") for v in image_index.values() if v.get("isolated")}
    if not isolated_refs:
        return text
    output: list[str] = []
    for line in text.splitlines():
        refs_in_line = _IMAGE_REF_RE.findall(line)
        if refs_in_line and all(r in isolated_refs for r in refs_in_line):
            continue
        output.append(line)
    return "\n".join(output)


# ── Heading parser ───────────────────────────────────────────────────────────

def _parse_sections(
    markdown_text: str,
) -> tuple[list[str], list[_Section], int]:
    """Return (root_lines, top-level sections, total_line_count)."""
    root_lines: list[str] = []
    sections: list[_Section] = []
    current_h1: _Section | None = None
    current_h2: _Section | None = None
    current_h3: _Section | None = None
    all_lines = markdown_text.splitlines()

    for line_num, line in enumerate(all_lines):
        m = _HEADING_RE.match(line)
        if not m:
            if current_h3:
                current_h3.lines.append(line)
                current_h3.line_nums.append(line_num)
            elif current_h2:
                current_h2.lines.append(line)
                current_h2.line_nums.append(line_num)
            elif current_h1:
                current_h1.lines.append(line)
                current_h1.line_nums.append(line_num)
            else:
                root_lines.append(line)
            continue

        level = len(m.group(1))
        text = m.group(2).strip()

        if level == 1:
            current_h1 = _Section(level=1, heading_text=text, heading_line=line_num)
            current_h2 = None
            current_h3 = None
            sections.append(current_h1)
        elif level == 2:
            current_h2 = _Section(level=2, heading_text=text, heading_line=line_num)
            current_h3 = None
            if current_h1:
                current_h1.children.append(current_h2)
            else:
                sections.append(current_h2)
        else:
            current_h3 = _Section(level=3, heading_text=text, heading_line=line_num)
            if current_h2:
                current_h2.children.append(current_h3)
            elif current_h1:
                current_h1.children.append(current_h3)
            else:
                sections.append(current_h3)

    return root_lines, sections, len(all_lines)


# ── Tree flattener ───────────────────────────────────────────────────────────

def _flatten(
    section: _Section,
    parent_h1: str | None = None,
    parent_h2: str | None = None,
) -> list[_ChunkCandidate]:
    h1 = parent_h1
    h2 = parent_h2
    h3: str | None = None

    if section.level == 1:
        h1 = section.heading_text
    elif section.level == 2:
        h2 = section.heading_text
    elif section.level == 3:
        h3 = section.heading_text

    own_body = "\n".join(section.lines).strip()
    heading_line = "#" * section.level + " " + section.heading_text
    own_end = max(section.line_nums) if section.line_nums else section.heading_line
    results: list[_ChunkCandidate] = []

    if section.children:
        if own_body:
            preamble = heading_line + "\n\n" + own_body
            results.append(_ChunkCandidate(
                h1, h2, h3, preamble, section.level,
                start_line=section.start_line, end_line=own_end,
            ))
        for child in section.children:
            results.extend(_flatten(child, h1, h2))
    else:
        full_text = heading_line + ("\n\n" + own_body if own_body else "")
        results.append(_ChunkCandidate(
            h1, h2, h3, full_text, section.level,
            start_line=section.start_line, end_line=own_end,
        ))

    return results


# ── Part splitter ────────────────────────────────────────────────────────────

def _split_into_parts(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    parts: list[str] = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= chunk_size:
            current += "\n\n" + para
        else:
            if current:
                parts.append(current.strip())
            current = para
    if current.strip():
        parts.append(current.strip())

    final: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            final.append(part)
        else:
            start = 0
            while start < len(part):
                end = min(start + chunk_size, len(part))
                final.append(part[start:end].strip())
                if end >= len(part):
                    break
                start = max(end - chunk_overlap, 0)
    return [p for p in final if p]


# ── Asset lookup helper ──────────────────────────────────────────────────────

def _assets_in_range(index: dict[str, Any], path_key: str, start: int, end: int) -> list[str]:
    result: list[str] = []
    for entry in index.values():
        sl = entry.get("start_line", -1)
        if start <= sl <= end:
            p = entry.get(path_key, "")
            if p:
                result.append(p)
    return result


# ── Main run ─────────────────────────────────────────────────────────────────

def run(
    input_markdown_path: str | Path,
    output_chunks_dir: str | Path,
    document_folder: str | Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    document_folder = Path(document_folder)
    markdown_text = read_text(input_markdown_path)
    chunks_dir = ensure_directory(output_chunks_dir)
    sections_dir = ensure_directory(build_sections_dir(document_folder))

    # Load pre-built asset indexes (written by extract_* services)
    tables_dir = build_tables_dir(document_folder).resolve()
    images_dir = build_images_dir(document_folder)

    table_index: dict[str, Any] = read_json(tables_dir / "index.json", default={})
    image_index: dict[str, Any] = read_json(images_dir / "index.json", default={})

    metadata = load_metadata(document_folder)
    document_name = metadata["document_name"]
    doc_slug = slugify_filename(document_name)

    # Parse markdown into section tree with line-number tracking
    root_lines, sections, total_lines = _parse_sections(markdown_text)
    candidates: list[_ChunkCandidate] = []

    root_text = "\n".join(root_lines).strip()
    if root_text:
        root_end = sections[0].start_line - 1 if sections else total_lines - 1
        candidates.append(_ChunkCandidate(None, None, None, root_text, 0,
                                          start_line=0, end_line=max(root_end, 0)))

    for section in sections:
        candidates.extend(_flatten(section))

    # Process candidates → chunks
    chunk_paths: list[str] = []
    seen_ids: set[str] = set()
    global_chunk_num = 0
    h1_section_registry: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        parts = _split_into_parts(candidate.text, chunk_size, chunk_overlap)
        sl, el = candidate.start_line, candidate.end_line

        for part_idx, part_text in enumerate(parts, start=1):
            global_chunk_num += 1
            part_suffix = part_idx if len(parts) > 1 else None
            chunk_id = build_chunk_name(
                doc_slug,
                slugify_heading(candidate.h1) if candidate.h1 else None,
                slugify_heading(candidate.h2) if candidate.h2 else None,
                slugify_heading(candidate.h3) if candidate.h3 else None,
                part=part_suffix,
            )
            original_id = chunk_id
            suffix_num = 2
            while chunk_id in seen_ids:
                chunk_id = f"{original_id}_dup{suffix_num}"
                suffix_num += 1
            seen_ids.add(chunk_id)

            section_path = build_section_path(candidate.h1, candidate.h2, candidate.h3)

            # Text cleanup: strip OCR blocks
            part_text = _strip_ocr_blocks(part_text)
            # Remove isolated image refs from chunk text
            part_text = _remove_isolated_refs(part_text, image_index)

            # Assign pre-extracted assets by section line range
            table_paths = _assets_in_range(table_index, "csv_path", sl, el)
            image_paths = [
                e["abs_path"] for e in image_index.values()
                if not e.get("isolated") and sl <= e.get("start_line", -1) <= el
            ]

            parent_section_id: str | None = slugify_heading(candidate.h1) if candidate.h1 else None

            chunk_path = chunks_dir / f"{chunk_id}.json"
            write_json(chunk_path, {
                "chunk_id": chunk_id,
                "document_name": document_name,
                "chunk_number": global_chunk_num,
                "section_path": section_path,
                "section_level": candidate.section_level,
                "h1": candidate.h1,
                "h2": candidate.h2,
                "h3": candidate.h3,
                "parent_section_id": parent_section_id,
                "text": part_text,
                "image_paths": image_paths,
                "table_paths": table_paths,
            })
            chunk_paths.append(str(chunk_path))

            # Accumulate section registry (deduplicate paths)
            if parent_section_id:
                if parent_section_id not in h1_section_registry:
                    h1_section_registry[parent_section_id] = {
                        "h1": candidate.h1,
                        "section_path": candidate.h1,
                        "chunk_ids": [],
                        "image_paths": [],
                        "table_paths": [],
                    }
                reg = h1_section_registry[parent_section_id]
                reg["chunk_ids"].append(chunk_id)
                for p in image_paths:
                    if p not in reg["image_paths"]:
                        reg["image_paths"].append(p)
                for p in table_paths:
                    if p not in reg["table_paths"]:
                        reg["table_paths"].append(p)

    # Write sections/*.json
    for h1_slug, sec_data in h1_section_registry.items():
        write_json(sections_dir / f"{h1_slug}.json", sec_data)

    # Collect all asset paths from indexes (authoritative source)
    all_table_paths = [e["csv_path"] for e in table_index.values() if e.get("csv_path")]

    metadata["chunk_paths"] = chunk_paths
    metadata["total_chunks"] = len(chunk_paths)
    metadata["last_chunk_processed"] = len(chunk_paths)
    metadata["table_paths"] = all_table_paths
    metadata["tables_dir"] = str(tables_dir)
    metadata["sections_dir"] = str(sections_dir)
    metadata["extraction_strategy"] = "section_aware"
    save_metadata(document_folder, metadata)
    mark_step_success(document_folder, "markdown_chunker", {
        "chunks_dir": str(chunks_dir),
        "total_chunks": len(chunk_paths),
        "tables_dir": str(tables_dir),
        "sections_dir": str(sections_dir),
    })
    return chunk_paths
