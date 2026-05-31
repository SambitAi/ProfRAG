from __future__ import annotations

import re
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "who", "can", "does", "are",
    "was", "were", "into", "have", "has", "had", "about", "across", "over", "under", "all",
}
_FLAG_RERANKER = None


def _query_terms(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS}


def _lexical_score(query: str, document: str) -> float:
    q = _query_terms(query)
    if not q:
        return 0.0
    d = (document or "").lower()
    hits = sum(1 for t in q if re.search(r"\b" + re.escape(t) + r"\b", d))
    return hits / len(q)


def _get_flag_reranker():
    global _FLAG_RERANKER
    if _FLAG_RERANKER is None:
        from FlagEmbedding import FlagReranker  # type: ignore
        _FLAG_RERANKER = FlagReranker("BAAI/bge-reranker-base", use_fp16=False)
    return _FLAG_RERANKER


def rerank_items(
    *,
    query: str,
    documents: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
    top_k: int = 8,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    if not documents:
        return documents, metadatas, ids

    def _lexical_fallback() -> tuple[list[str], list[dict[str, Any]], list[str]]:
        ranked = sorted(
            list(zip(documents, metadatas, ids)),
            key=lambda item: _lexical_score(query, item[0]),
            reverse=True,
        )[:top_k]
        return [r[0] for r in ranked], [r[1] for r in ranked], [r[2] for r in ranked]

    try:
        reranker = _get_flag_reranker()
    except Exception:
        return _lexical_fallback()

    try:
        pairs = [[query, doc] for doc in documents]
        scores = reranker.compute_score(pairs)
    except Exception:
        return _lexical_fallback()
    ranked = sorted(
        list(zip(scores, documents, metadatas, ids)),
        key=lambda x: float(x[0]),
        reverse=True,
    )[:top_k]
    out_docs = [r[1] for r in ranked]
    out_metas = [r[2] for r in ranked]
    out_ids = [r[3] for r in ranked]
    return out_docs, out_metas, out_ids
