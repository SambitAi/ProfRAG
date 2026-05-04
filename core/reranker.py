from __future__ import annotations

from typing import Any


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
    try:
        from FlagEmbedding import FlagReranker  # type: ignore
    except Exception:
        return documents[:top_k], metadatas[:top_k], ids[:top_k]

    reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=False)
    pairs = [[query, doc] for doc in documents]
    scores = reranker.compute_score(pairs)
    ranked = sorted(
        list(zip(scores, documents, metadatas, ids)),
        key=lambda x: float(x[0]),
        reverse=True,
    )[:top_k]
    out_docs = [r[1] for r in ranked]
    out_metas = [r[2] for r in ranked]
    out_ids = [r[3] for r in ranked]
    return out_docs, out_metas, out_ids

