from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from core.global_index import load_global_index
from core.llm import get_openai_client
from core.query_classifier import classify_query
from core.reranker import rerank_items
from core.similarity import dynamic_k_from_scores, similarity_from_distance


def _auto_doc_k(scores: list[float], candidate_count: int) -> int:
    if candidate_count <= 1:
        return candidate_count
    default_k = max(2, int(candidate_count ** 0.5))
    return dynamic_k_from_scores(
        scores,
        min_k=1,
        max_k=min(12, candidate_count),
        default_k=default_k,
        elbow_drop_pct=0.2,
    )


def _auto_section_k(selected_doc_count: int) -> int:
    return max(4, min(24, selected_doc_count * 3))


def _auto_leaf_k(selected_doc_count: int) -> int:
    return max(8, selected_doc_count * 6)


def _should_abstain(scores: list[float], selection_mode: bool) -> bool:
    if selection_mode:
        return False
    if not scores:
        return True
    mx = max(scores)
    avg = sum(scores) / len(scores)
    spread = mx - avg
    return mx < 0.18 and spread < 0.03


def _get_chroma_client(config: dict[str, Any]) -> chromadb.ClientAPI:
    vdb = config.get("vector_db", {})
    if vdb.get("host"):
        return chromadb.HttpClient(host=vdb["host"], port=int(vdb.get("port", 8000)))
    return chromadb.PersistentClient(path=str(vdb["persist_directory"]))


def _summary_doc_rank(
    *,
    chroma: chromadb.ClientAPI,
    query_embedding: list[float],
    summary_collection_name: str,
    allowed_doc_ids: set[str],
    allowed_folders: set[str],
    n_results: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    try:
        summaries = chroma.get_collection(name=summary_collection_name)
    except Exception:
        return [], []
    try:
        res = summaries.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"level": {"$eq": "1"}},
        )
    except Exception:
        return [], []
    metas = (res.get("metadatas") or [[]])[0]
    dist = (res.get("distances") or [[]])[0]
    out_metas: list[dict[str, Any]] = []
    out_scores: list[float] = []
    for m, d in zip(metas, dist):
        did = m.get("document_id", "")
        folder = m.get("document_folder", "")
        if (did and did in allowed_doc_ids) or (folder and folder in allowed_folders):
            out_metas.append(m)
            out_scores.append(similarity_from_distance(d))
    return out_metas, out_scores


def _summary_section_rank(
    *,
    chroma: chromadb.ClientAPI,
    query_embedding: list[float],
    summary_collection_name: str,
    top_doc_ids: list[str],
    top_doc_folders: list[str],
    n_results: int,
) -> list[dict[str, Any]]:
    try:
        summaries = chroma.get_collection(name=summary_collection_name)
    except Exception:
        return []
    try:
        where_filter: dict[str, Any]
        if top_doc_ids:
            where_filter = {"$and": [{"level": {"$eq": "3"}}, {"document_id": {"$in": top_doc_ids}}]}
        else:
            where_filter = {"$and": [{"level": {"$eq": "3"}}, {"document_folder": {"$in": top_doc_folders}}]}
        res = summaries.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
        )
    except Exception:
        return []
    return (res.get("metadatas") or [[]])[0]


def _llm_should_continue(client: Any, chat_model: str, question: str, scoped_docs: dict[str, Any]) -> bool:
    # Lightweight ambiguity probe: only used when score-based abstention would trigger.
    previews: list[str] = []
    for key, d in list(scoped_docs.items())[:8]:
        card = (d or {}).get("document_card", {}) or {}
        title = card.get("title") or d.get("document_name") or key
        opening = (card.get("opening_text") or "")[:180]
        previews.append(f"- {title}: {opening}")
    prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Candidate documents:\n"
        + "\n".join(previews)
        + "\n\nRespond with ONLY YES or NO: Is at least one candidate potentially relevant?"
    )
    try:
        resp = client.chat.completions.create(
            model=chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a strict relevance router."},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        return text.startswith("YES")
    except Exception:
        return False


def retrieve_tree(
    question: str,
    config: dict[str, Any],
    selected_document_folders: list[str] | None = None,
) -> dict[str, Any]:
    selected_document_folders = selected_document_folders or []
    mode = classify_query(question)

    global_index = load_global_index(
        Path(config["paths"]["artifacts_root"]) / "metadata.json",
        cache_ttl_seconds=float(config.get("global_index", {}).get("cache_ttl_seconds", 5)),
    )
    docs_map = (global_index or {}).get("documents", {}) or {}
    if selected_document_folders:
        allowed = {Path(f).name for f in selected_document_folders}
        scoped_docs = {k: v for k, v in docs_map.items() if k in allowed}
    else:
        scoped_docs = dict(docs_map)

    if not scoped_docs:
        return {"mode": mode, "confidence": "low", "documents": [], "metadatas": [], "ids": [], "abstain": True}

    client = get_openai_client()
    q_resp = client.embeddings.create(model=config["embeddings"]["model"], input=[question])
    query_embedding = q_resp.data[0].embedding

    chroma = _get_chroma_client(config)
    cards = chroma.get_or_create_collection(name=config["vector_db"].get("card_collection_name", "pdf_rag_cards"))
    chunks = chroma.get_or_create_collection(name=config["vector_db"]["collection_name"])
    summary_collection = config["vector_db"].get("summary_collection_name", "pdf_rag_summaries")

    doc_ids = [v.get("document_id", "") for v in scoped_docs.values() if v.get("document_id")]
    doc_folders = [v.get("document_folder", "") for v in scoped_docs.values() if v.get("document_folder")]
    if not doc_ids:
        return {"mode": mode, "confidence": "low", "documents": [], "metadatas": [], "ids": [], "abstain": True}

    # Phase 1: Card-first (source of truth), optional summary augmentation.
    where_docs: dict[str, Any] = {"$and": [{"card_type": {"$eq": "document"}}, {"document_id": {"$in": doc_ids}}]}
    phase1 = cards.query(
        query_embeddings=[query_embedding],
        n_results=min(max(len(doc_ids), 5), 50),
        where=where_docs,
    )
    phase1_metas = (phase1.get("metadatas") or [[]])[0]
    phase1_dist = (phase1.get("distances") or [[]])[0]
    phase1_scores = [similarity_from_distance(d) for d in phase1_dist]

    # Optional compatibility/cache layer from summary vectors.
    summary_doc_metas, summary_doc_scores = _summary_doc_rank(
        chroma=chroma,
        query_embedding=query_embedding,
        summary_collection_name=summary_collection,
        allowed_doc_ids=set(doc_ids),
        allowed_folders=set(doc_folders),
        n_results=min(max(len(doc_ids) * 2, 8), 100),
    )
    if summary_doc_metas:
        seen = {m.get("document_id", "") for m in phase1_metas}
        for m, s in zip(summary_doc_metas, summary_doc_scores):
            did = m.get("document_id", "")
            if did and did not in seen:
                phase1_metas.append(m)
                phase1_scores.append(s)
                seen.add(did)

    max_score = max(phase1_scores) if phase1_scores else 0.0
    if _should_abstain(phase1_scores, selection_mode=bool(selected_document_folders)):
        # Ambiguity recovery: let LLM decide if traversal should continue.
        if not _llm_should_continue(client, config["chat"]["model"], question, scoped_docs):
            return {"mode": mode, "confidence": "low", "documents": [], "metadatas": [], "ids": [], "abstain": True}

    doc_k = _auto_doc_k(phase1_scores, len(doc_ids))
    top_doc_ids: list[str] = []
    top_doc_folders: list[str] = []
    if selected_document_folders:
        top_doc_ids = list(dict.fromkeys(doc_ids))
        top_doc_folders = list(dict.fromkeys(doc_folders))
    else:
        for meta in phase1_metas[:doc_k]:
            did = meta.get("document_id", "")
            folder = meta.get("document_folder", "")
            if did and did not in top_doc_ids:
                top_doc_ids.append(did)
            if folder and folder not in top_doc_folders:
                top_doc_folders.append(folder)
        if not top_doc_ids:
            top_doc_ids = doc_ids[:doc_k]
        if not top_doc_folders:
            top_doc_folders = doc_folders[:doc_k]

    # Phase 2: Card-first section ranking, optional summary augmentation.
    section_k = _auto_section_k(len(top_doc_ids))
    where_sections = {"$and": [{"card_type": {"$eq": "section"}}, {"document_id": {"$in": top_doc_ids}}]}
    phase2 = cards.query(
        query_embeddings=[query_embedding],
        n_results=max(len(top_doc_ids) * section_k, 8),
        where=where_sections,
    )
    phase2_metas = (phase2.get("metadatas") or [[]])[0]

    summary_section_metas = _summary_section_rank(
        chroma=chroma,
        query_embedding=query_embedding,
        summary_collection_name=summary_collection,
        top_doc_ids=top_doc_ids,
        top_doc_folders=top_doc_folders,
        n_results=max(len(top_doc_ids) * section_k, 8),
    )
    if summary_section_metas:
        seen_sp = {m.get("section_path", "") for m in phase2_metas}
        for m in summary_section_metas:
            sp = m.get("section_path", "")
            if sp and sp not in seen_sp:
                phase2_metas.append(m)
                seen_sp.add(sp)

    section_paths: list[str] = []
    for meta in phase2_metas:
        sp = meta.get("section_path", "")
        if sp and sp not in section_paths:
            section_paths.append(sp)
    section_paths = section_paths[: max(section_k, 1)]

    def _query_leaf(with_sections: bool, n_results: int) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        if with_sections and section_paths:
            where_chunks: dict[str, Any] = {
                "$and": [
                    {"document_id": {"$in": top_doc_ids}},
                    {"section_path": {"$in": section_paths}},
                ]
            }
        else:
            where_chunks = {"document_id": {"$in": top_doc_ids}}
        leaf = chunks.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_chunks,
        )
        return (
            (leaf.get("documents") or [[]])[0],
            (leaf.get("metadatas") or [[]])[0],
            (leaf.get("ids") or [[]])[0],
        )

    base_leaf_k = _auto_leaf_k(max(1, len(top_doc_ids)))
    docs, metas, ids = _query_leaf(with_sections=True, n_results=base_leaf_k)
    # Adaptive backtracking: if section-constrained retrieval is too sparse, broaden.
    if len(docs) < max(3, len(top_doc_ids)):
        broad_docs, broad_metas, broad_ids = _query_leaf(with_sections=False, n_results=base_leaf_k * 2)
        docs.extend(broad_docs)
        metas.extend(broad_metas)
        ids.extend(broad_ids)

    if selected_document_folders and len(top_doc_ids) > 1:
        seen_doc_ids = {m.get("document_id", "") for m in metas if m.get("document_id")}
        missing = [did for did in top_doc_ids if did not in seen_doc_ids]
        for did in missing:
            if section_paths:
                per_doc_where: dict[str, Any] = {
                    "$and": [
                        {"document_id": {"$eq": did}},
                        {"section_path": {"$in": section_paths}},
                    ]
                }
            else:
                per_doc_where = {"document_id": {"$eq": did}}
            extra = chunks.query(
                query_embeddings=[query_embedding],
                n_results=1,
                where=per_doc_where,
            )
            edocs = (extra.get("documents") or [[]])[0]
            emetas = (extra.get("metadatas") or [[]])[0]
            eids = (extra.get("ids") or [[]])[0]
            # Cross-doc safety: if section-filtered query yields nothing for this doc,
            # retry without section constraint so each selected doc can contribute.
            if not edocs:
                extra2 = chunks.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    where={"document_id": {"$eq": did}},
                )
                edocs = (extra2.get("documents") or [[]])[0]
                emetas = (extra2.get("metadatas") or [[]])[0]
                eids = (extra2.get("ids") or [[]])[0]
            if edocs and emetas and eids:
                docs.append(edocs[0])
                metas.append(emetas[0])
                ids.append(eids[0])

        dedup_docs: list[str] = []
        dedup_metas: list[dict[str, Any]] = []
        dedup_ids: list[str] = []
        seen_ids: set[str] = set()
        for d, m, i in zip(docs, metas, ids):
            if not i or i in seen_ids:
                continue
            seen_ids.add(i)
            dedup_docs.append(d)
            dedup_metas.append(m)
            dedup_ids.append(i)
        docs, metas, ids = dedup_docs, dedup_metas, dedup_ids

    docs, metas, ids = rerank_items(
        query=question,
        documents=docs,
        metadatas=metas,
        ids=ids,
        top_k=8,
    )

    return {
        "mode": mode,
        "confidence": "low" if _should_abstain(phase1_scores, selection_mode=False) else "normal",
        "abstain": False,
        "documents": docs,
        "metadatas": metas,
        "ids": ids,
        "candidate_document_ids": top_doc_ids,
        "candidate_section_paths": section_paths,
        "max_doc_score": max_score,
    }
