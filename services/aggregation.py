from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.metadata import load_metadata


def _pick_numeric_field(question: str, field_names: list[str]) -> str | None:
    q = (question or "").lower()
    q_tokens = set(re.findall(r"[a-z0-9_]+", q))
    best_name = None
    best_score = 0
    for name in field_names:
        parts = set(name.lower().split("_"))
        score = len(parts & q_tokens)
        if score > best_score:
            best_name = name
            best_score = score
    return best_name if best_score > 0 else None


def aggregate(question: str, document_folders: list[str], config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    numeric_candidates: set[str] = set()
    for folder in document_folders:
        meta = load_metadata(folder)
        fields = meta.get("extracted_fields", {}) or {}
        if not fields:
            continue
        doc_name = meta.get("document_name", Path(folder).name)
        row = {"document_name": doc_name, "document_folder": folder, "fields": fields}
        rows.append(row)
        for k, v in fields.items():
            if isinstance(v, (int, float)):
                numeric_candidates.add(k)

    if not rows:
        return {
            "question": question,
            "answer": "No extracted structured fields were found in the selected documents.",
            "sources": [],
            "multi_doc": True,
            "document_names": [],
            "mode": "aggregation",
        }

    target = _pick_numeric_field(question, sorted(numeric_candidates))
    if not target:
        # Fallback summary over available fields by document.
        lines = ["Structured fields found per document:"]
        sources: list[dict[str, Any]] = []
        for row in rows:
            fields = ", ".join(f"{k}={v}" for k, v in row["fields"].items())
            lines.append(f"- {row['document_name']}: {fields} [Source: {row['document_name']}, extracted_fields]")
            sources.append({
                "document_name": row["document_name"],
                "document_folder": row["document_folder"],
                "section_path": "extracted_fields",
                "chunk_number": "",
                "chunk_text_snippet": fields[:150],
                "chunk_path": "",
            })
        return {
            "question": question,
            "answer": "\n".join(lines),
            "sources": sources,
            "multi_doc": True,
            "document_names": [r["document_name"] for r in rows],
            "mode": "aggregation",
        }

    total = 0.0
    used = 0
    lines = [f"Aggregated `{target}` across selected documents:"]
    sources: list[dict[str, Any]] = []
    for row in rows:
        value = row["fields"].get(target)
        if not isinstance(value, (int, float)):
            continue
        used += 1
        total += float(value)
        lines.append(
            f"- {row['document_name']}: {value} [Source: {row['document_name']}, extracted_fields.{target}]"
        )
        sources.append({
            "document_name": row["document_name"],
            "document_folder": row["document_folder"],
            "section_path": f"extracted_fields.{target}",
            "chunk_number": "",
            "chunk_text_snippet": f"{target}={value}",
            "chunk_path": "",
        })

    if used == 0:
        return {
            "question": question,
            "answer": f"Field `{target}` was identified but no numeric values were found in selected documents.",
            "sources": [],
            "multi_doc": True,
            "document_names": [r["document_name"] for r in rows],
            "mode": "aggregation",
        }

    lines.append(f"Total: {total:g} [Source: aggregated from {used} document(s)]")
    return {
        "question": question,
        "answer": "\n".join(lines),
        "sources": sources,
        "multi_doc": True,
        "document_names": [r["document_name"] for r in rows],
        "mode": "aggregation",
    }

