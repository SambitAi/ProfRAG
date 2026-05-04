from __future__ import annotations

import re


_AGG_RE = re.compile(
    r"\b(total|sum|aggregate|overall|across all|compare all|combined|average|avg|min|max|count)\b",
    re.IGNORECASE,
)
_TOPICAL_RE = re.compile(
    r"\b(explain|overview|summarize|summary|broadly|high[- ]level|what is|concept)\b",
    re.IGNORECASE,
)


def classify_query(question: str) -> str:
    text = (question or "").strip()
    if not text:
        return "specific"
    if _AGG_RE.search(text):
        return "aggregation"
    if _TOPICAL_RE.search(text):
        return "topical"
    return "specific"

