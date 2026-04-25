from __future__ import annotations

import re
from pathlib import Path


def slugify_filename(file_name: str) -> str:
    stem = Path(file_name).stem.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "document"


def build_document_folder_name(file_name: str, version: int) -> str:
    return f"{slugify_filename(file_name)}_v{version}"


def next_document_version(artifacts_root: str | Path, file_name: str) -> int:
    root = Path(artifacts_root)
    slug = slugify_filename(file_name)
    versions: list[int] = []

    if root.exists():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            match = re.fullmatch(rf"{re.escape(slug)}_v(\d+)", child.name)
            if match:
                versions.append(int(match.group(1)))

    return (max(versions) + 1) if versions else 1


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def slugify_heading(heading_text: str, max_length: int = 40) -> str:
    text = re.sub(r"^#+\s*", "", heading_text).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", text)
    slug = re.sub(r"_+", "_", slug).strip("_")
    slug = slug or "section"
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_")
    return slug


def build_chunk_name(
    doc_slug: str,
    h1_slug: str | None,
    h2_slug: str | None,
    h3_slug: str | None,
    part: int | None,
) -> str:
    parts = [doc_slug]
    for segment in (h1_slug, h2_slug, h3_slug):
        if segment:
            parts.append(segment)
    if part is not None:
        parts.append(f"part{part}")
    name = "_".join(parts)
    if len(name) > 120:
        name = name[:120].rstrip("_")
    # Strip characters that are invalid in Windows filenames
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"_+", "_", name).strip("_") or "chunk"
    return name


def build_section_path(
    h1: str | None,
    h2: str | None,
    h3: str | None,
) -> str:
    return " > ".join(level for level in (h1, h2, h3) if level)


def build_tables_dir(document_folder: str | Path) -> Path:
    return Path(document_folder) / "tables"


def build_images_dir(document_folder: str | Path) -> Path:
    return Path(document_folder) / "images"


def build_sections_dir(document_folder: str | Path) -> Path:
    return Path(document_folder) / "sections"
