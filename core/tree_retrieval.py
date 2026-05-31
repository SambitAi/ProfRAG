from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from core.global_index import load_global_index
from core.llm import get_openai_client
from core.metadata import load_metadata
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
        summaries = chroma.get_or_create_collection(name=summary_collection_name)
    except Exception:
        return [], []
    try:
        count = int(summaries.count())
    except Exception:
        count = 0
    if count <= 0:
        return [], []
    try:
        res = summaries.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
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


def _llm_should_continue(client: Any, chat_model: str, question: str, scoped_docs: dict[str, Any]) -> bool:
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
        artifacts_root = Path(config["paths"]["artifacts_root"])
        for folder_name in allowed:
            if folder_name in scoped_docs:
                continue
            folder_path = artifacts_root / folder_name
            metadata_path = folder_path / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = load_metadata(folder_path)
            scoped_docs[folder_name] = {
                "document_folder": str(folder_path),
                "document_id": metadata.get("document_id", ""),
                "document_name": metadata.get("document_name", folder_name),
                "document_card": metadata.get("document_card", {}),
            }
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

    scoped_pairs: list[tuple[str, str]] = []
    for v in scoped_docs.values():
        did = str(v.get("document_id", "") or "")
        folder = str(v.get("document_folder", "") or "")
        if did or folder:
            scoped_pairs.append((did, folder))
    doc_ids = [did for did, _ in scoped_pairs if did]
    doc_folders = [folder for _, folder in scoped_pairs if folder]
    if not doc_ids and not doc_folders:
        return {"mode": mode, "confidence": "low", "documents": [], "metadatas": [], "ids": [], "abstain": True}

    try:
        cards_count = int(cards.count())
    except Exception:
        cards_count = 0
    try:
        chunks_count = int(chunks.count())
    except Exception:
        chunks_count = 0
    if cards_count <= 0 or chunks_count <= 0:
        return {"mode": mode, "confidence": "low", "documents": [], "metadatas": [], "ids": [], "abstain": True}

    phase1_scores: list[float] = []
    top_doc_ids: list[str] = []
    top_doc_folders: list[str] = []
    max_score = 0.0

    if selected_document_folders:
        # Selection mode: skip corpus-wide routing and walk selected docs directly.
        top_doc_ids = list(dict.fromkeys([did for did, _ in scoped_pairs if did]))
        top_doc_folders = list(dict.fromkeys([folder for _, folder in scoped_pairs if folder]))
    else:
        # Discovery mode: route docs first, then use the same per-doc walk as selection mode.
        if doc_ids and doc_folders:
            where_docs: dict[str, Any] = {
                "$and": [
                    {"card_type": {"$eq": "document"}},
                    {"$or": [
                        {"document_id": {"$in": doc_ids}},
                        {"document_folder": {"$in": doc_folders}},
                    ]},
                ]
            }
        elif doc_ids:
            where_docs = {"$and": [{"card_type": {"$eq": "document"}}, {"document_id": {"$in": doc_ids}}]}
        else:
            where_docs = {"$and": [{"card_type": {"$eq": "document"}}, {"document_folder": {"$in": doc_folders}}]}

        phase1_n_results = min(min(max(max(len(doc_ids), len(doc_folders)), 5), 50), cards_count)
        phase1 = cards.query(
            query_embeddings=[query_embedding],
            n_results=phase1_n_results,
            where=where_docs,
        )
        phase1_metas = (phase1.get("metadatas") or [[]])[0]
        phase1_dist = (phase1.get("distances") or [[]])[0]
        phase1_scores = [similarity_from_distance(d) for d in phase1_dist]

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
        if _should_abstain(phase1_scores, selection_mode=False):
            if not _llm_should_continue(client, config["chat"]["model"], question, scoped_docs):
                return {"mode": mode, "confidence": "low", "documents": [], "metadatas": [], "ids": [], "abstain": True}

        doc_k = _auto_doc_k(phase1_scores, max(len(doc_ids), len(doc_folders)))
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

    routed_pairs: list[tuple[str, str]] = []
    for did, folder in scoped_pairs:
        if (did and did in top_doc_ids) or (folder and folder in top_doc_folders):
            routed_pairs.append((did, folder))
    if not routed_pairs:
        routed_pairs = list(scoped_pairs)
    if not routed_pairs:
        return {"mode": mode, "confidence": "low", "documents": [], "metadatas": [], "ids": [], "abstain": True}

    doc_count = max(1, len(routed_pairs))
    section_k_total = _auto_section_k(doc_count)
    leaf_k_total = _auto_leaf_k(doc_count)
    per_doc_section_k = max(2, min(8, section_k_total // doc_count if doc_count else section_k_total))
    per_doc_leaf_k = max(2, min(leaf_k_total, max(4, leaf_k_total // doc_count)))

    docs: list[str] = []
    metas: list[dict[str, Any]] = []
    ids: list[str] = []

    for did, folder in routed_pairs:
        if did and folder:
            doc_filter: dict[str, Any] = {
                "$or": [
                    {"document_id": {"$eq": did}},
                    {"document_folder": {"$eq": folder}},
                ]
            }
        elif did:
            doc_filter = {"document_id": {"$eq": did}}
        else:
            doc_filter = {"document_folder": {"$eq": folder}}

        section_where = {"$and": [{"card_type": {"$eq": "section"}}, doc_filter]}
        section_res = cards.query(
            query_embeddings=[query_embedding],
            n_results=min(max(per_doc_section_k, 1), cards_count),
            where=section_where,
        )
        section_metas = (section_res.get("metadatas") or [[]])[0]

        local_sections: list[str] = []
        for meta in section_metas:
            sp = str(meta.get("section_path", "") or "")
            if sp and sp not in local_sections:
                local_sections.append(sp)
            if len(local_sections) >= per_doc_section_k:
                break

        if local_sections:
            chunk_where = {"$and": [doc_filter, {"section_path": {"$in": local_sections}}]}
        else:
            chunk_where = doc_filter
        leaf = chunks.query(
            query_embeddings=[query_embedding],
            n_results=min(max(per_doc_leaf_k, 1), chunks_count),
            where=chunk_where,
        )
        local_docs = (leaf.get("documents") or [[]])[0]
        local_metas = (leaf.get("metadatas") or [[]])[0]
        local_ids = (leaf.get("ids") or [[]])[0]

        if not local_docs and local_sections:
            leaf_fallback = chunks.query(
                query_embeddings=[query_embedding],
                n_results=min(max(per_doc_leaf_k, 1), chunks_count),
                where=doc_filter,
            )
            local_docs = (leaf_fallback.get("documents") or [[]])[0]
            local_metas = (leaf_fallback.get("metadatas") or [[]])[0]
            local_ids = (leaf_fallback.get("ids") or [[]])[0]

        docs.extend(local_docs)
        metas.extend(local_metas)
        ids.extend(local_ids)

    seen_ids: set[str] = set()
    dedup_docs: list[str] = []
    dedup_metas: list[dict[str, Any]] = []
    dedup_ids: list[str] = []
    for d, m, i in zip(docs, metas, ids):
        if not i or i in seen_ids:
            continue
        seen_ids.add(i)
        dedup_docs.append(d)
        dedup_metas.append(m)
        dedup_ids.append(i)
    docs, metas, ids = dedup_docs, dedup_metas, dedup_ids

    retrieval_cfg = config.get("retrieval", {})
    cap = int(retrieval_cfg.get("multi_doc_chunk_cap", 20))
    cap = max(8, cap)
    # Scale top-k with actual per-doc retrieval breadth to avoid starving selected docs.
    effective_top_k = min(max(8, len(routed_pairs) * per_doc_leaf_k), cap)
    reranked_docs, reranked_metas, reranked_ids = rerank_items(
        query=question,
        documents=docs,
        metadatas=metas,
        ids=ids,
        top_k=effective_top_k,
    )
    # Post-rerank floor: prevent semantic collapse to a single dominant document.
    # Each routed document should retain at least floor(effective_top_k / routed_doc_count) chunks
    # when candidates are available.
    def _route_key(did: str, folder: str) -> str:
        return f"folder:{folder}" if folder else f"id:{did}"

    expected_keys: list[str] = []
    for did, folder in routed_pairs:
        key = _route_key(did, folder)
        if key not in expected_keys:
            expected_keys.append(key)

    if expected_keys:
        floor_n = max(1, effective_top_k // len(expected_keys))

        all_by_key: dict[str, list[tuple[str, dict[str, Any], str]]] = {k: [] for k in expected_keys}
        for d, m, i in zip(docs, metas, ids):
            key = _route_key(str(m.get("document_id", "") or ""), str(m.get("document_folder", "") or ""))
            if key in all_by_key:
                all_by_key[key].append((d, m, i))

        selected: list[tuple[str, dict[str, Any], str]] = []
        selected_ids: set[str] = set()
        selected_count_by_key: dict[str, int] = {k: 0 for k in expected_keys}

        # Seed from reranked order first.
        for d, m, i in zip(reranked_docs, reranked_metas, reranked_ids):
            if i in selected_ids:
                continue
            key = _route_key(str(m.get("document_id", "") or ""), str(m.get("document_folder", "") or ""))
            if key not in selected_count_by_key:
                continue
            selected.append((d, m, i))
            selected_ids.add(i)
            selected_count_by_key[key] += 1

        # Enforce per-doc floor using best remaining candidates from each doc.
        for key in expected_keys:
            while selected_count_by_key[key] < floor_n:
                candidate = next((x for x in all_by_key.get(key, []) if x[2] not in selected_ids), None)
                if candidate is None:
                    break
                selected.append(candidate)
                selected_ids.add(candidate[2])
                selected_count_by_key[key] += 1

        # Fill to effective_top_k by reranked preference, then remaining pool.
        for d, m, i in zip(reranked_docs, reranked_metas, reranked_ids):
            if len(selected) >= effective_top_k:
                break
            if i in selected_ids:
                continue
            selected.append((d, m, i))
            selected_ids.add(i)
        if len(selected) < effective_top_k:
            for d, m, i in zip(docs, metas, ids):
                if len(selected) >= effective_top_k:
                    break
                if i in selected_ids:
                    continue
                selected.append((d, m, i))
                selected_ids.add(i)

        docs = [d for d, _, _ in selected[:effective_top_k]]
        metas = [m for _, m, _ in selected[:effective_top_k]]
        ids = [i for _, _, i in selected[:effective_top_k]]
    else:
        docs, metas, ids = reranked_docs, reranked_metas, reranked_ids

    if phase1_scores:
        confidence = "low" if _should_abstain(phase1_scores, selection_mode=bool(selected_document_folders)) else "normal"
    else:
        confidence = "normal"

    return {
        "mode": mode,
        "confidence": confidence,
        "abstain": False,
        "documents": docs,
        "metadatas": metas,
        "ids": ids,
        "candidate_document_ids": top_doc_ids,
        "max_doc_score": max_score,
    }
