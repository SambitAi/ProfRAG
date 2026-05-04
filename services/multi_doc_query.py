from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import chromadb

from core.global_index import load_global_index
from core.llm import get_openai_client
from core.metadata import load_metadata
from core.query_classifier import classify_query
from core.paths import ensure_directory
from core.retrieval_engine import run_retrieval
from core.storage import read_json, write_json
from core.tree_retrieval import retrieve_tree
from services import aggregation

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[Source:\s*[^\]]+\]")
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "who", "can", "does", "are",
    "was", "were", "into", "have", "has", "had", "about", "across", "over", "under", "all",
}


def _query_terms(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS}


def _lexical_overlap_score(query: str, *fields: str) -> float:
    q = _query_terms(query)
    if not q:
        return 0.0
    hay = " ".join(f or "" for f in fields).lower()
    if not hay.strip():
        return 0.0
    hits = sum(1 for t in q if t in hay)
    return hits / len(q)


def _is_citable_sentence(sentence: str) -> bool:
    s = (sentence or "").strip().lower()
    if not s:
        return False
    # Pure refusal/uncertainty or routing statements are not required to carry citations.
    non_citable_starts = (
        "i cannot",
        "i can't",
        "i do not",
        "i don't",
        "the provided documents do not",
        "this document does not",
        "there is no information",
        "insufficient information",
        "not enough information",
        "based on the provided context",
    )
    if s.startswith(non_citable_starts):
        return False
    # Very short fragments are usually connectors, not factual claims.
    if len(s.split()) <= 4:
        return False
    return True


def _citation_post_check(answer: str) -> tuple[bool, str]:
    text = (answer or "").strip()
    if not text:
        return False, "empty answer"
    lowered = text.lower()
    uncertainty_markers = (
        "do not contain information",
        "cannot determine",
        "cannot confirm",
        "not enough information",
        "insufficient information",
    )
    # If answer is explicitly uncertainty-aware and still cites at least one source,
    # do not surface citation weakness noise to the UI.
    if any(m in lowered for m in uncertainty_markers) and _CITATION_RE.search(text):
        return True, ""
    # Require at least one citation marker and good coverage on citable sentences.
    if not _CITATION_RE.search(text):
        return False, "missing citation markers"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    citable = [s for s in sentences if _is_citable_sentence(s)]
    if not citable:
        return True, ""
    cited = sum(1 for s in citable if _CITATION_RE.search(s))
    ratio = cited / len(citable)
    # Keep strictness, but avoid false warnings on short answers.
    required = 1.0 if len(citable) <= 2 else 0.7
    return (ratio >= required), f"low citation ratio ({ratio:.2f})"


def _repair_citations(client: Any, chat_model: str, answer: str, context: str, question: str) -> str:
    repair_prompt = (
        "Rewrite the answer by preserving meaning but adding citations.\n"
        "For each factual sentence append a citation in this exact format:\n"
        "[Source: document_name, section_path]\n"
        "Do not invent sources. Use only context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer to fix:\n{answer}"
    )
    resp = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You enforce citation formatting strictly."},
            {"role": "user", "content": repair_prompt},
        ],
    )
    return resp.choices[0].message.content or answer


def _attach_fallback_citations(answer: str, sources: list[dict[str, Any]]) -> str:
    text = (answer or "").strip()
    if not text or not sources:
        return text
    source_labels: list[str] = []
    for src in sources:
        doc = src.get("document_name", "unknown")
        section = src.get("section_path", "")
        if section:
            source_labels.append(f"[Source: {doc}, {section}]")
        else:
            source_labels.append(f"[Source: {doc}, chunk]")
    if not source_labels:
        return text

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return f"{text} {source_labels[0]}".strip()
    out: list[str] = []
    for i, s in enumerate(sentences):
        if _CITATION_RE.search(s):
            out.append(s)
        else:
            out.append(f"{s} {source_labels[i % len(source_labels)]}")
    return " ".join(out)


def _get_chroma_client(config: dict[str, Any]) -> chromadb.ClientAPI:
    chroma_host = config["vector_db"].get("host")
    if chroma_host:
        return chromadb.HttpClient(host=chroma_host, port=int(config["vector_db"].get("port", 8000)))
    persist_dir = config["vector_db"]["persist_directory"]
    ensure_directory(persist_dir)
    return chromadb.PersistentClient(path=str(persist_dir))


def _find_relevant_documents_card_first(
    question: str,
    config: dict[str, Any],
    chroma_client: chromadb.ClientAPI,
    query_embedding: list[float],
) -> list[dict[str, Any]]:
    card_collection_name = config["vector_db"].get("card_collection_name", "pdf_rag_cards")
    global_index = load_global_index(
        Path(config["paths"]["artifacts_root"]) / "metadata.json",
        cache_ttl_seconds=float(config.get("global_index", {}).get("cache_ttl_seconds", 5)),
    )
    docs_map = (global_index or {}).get("documents", {}) or {}
    doc_ids = [v.get("document_id", "") for v in docs_map.values() if v.get("document_id")]
    if not doc_ids:
        return []

    cards = chroma_client.get_collection(name=card_collection_name)
    doc_res = cards.query(
        query_embeddings=[query_embedding],
        n_results=min(max(len(doc_ids), 8), 50),
        where={"$and": [{"card_type": {"$eq": "document"}}, {"document_id": {"$in": doc_ids}}]},
    )
    doc_metas = (doc_res.get("metadatas") or [[]])[0]
    doc_distances = (doc_res.get("distances") or [[]])[0]

    best_docs: dict[str, dict[str, Any]] = {}
    for meta, dist in zip(doc_metas, doc_distances):
        folder = meta.get("document_folder", "")
        if not folder:
            continue
        doc_name = meta.get("document_name", Path(folder).name)
        semantic_score = max(0.0, min(1.0, 1.0 - float(dist)))
        try:
            doc_meta = load_metadata(folder)
        except Exception:
            doc_meta = {}
        doc_card = (doc_meta or {}).get("document_card", {}) or {}
        lexical_score = _lexical_overlap_score(
            question,
            doc_name,
            doc_card.get("title", ""),
            doc_card.get("opening_text", ""),
            doc_card.get("l1_summary", ""),
        )
        score = (0.8 * semantic_score) + (0.2 * lexical_score)
        prev = best_docs.get(folder)
        if prev is None or score > prev["score"]:
            best_docs[folder] = {
                "folder": folder,
                "doc_name": doc_name,
                "top_section": "",
                "score": score,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
            }

    if not best_docs:
        return []

    top_folders = sorted(best_docs, key=lambda f: best_docs[f]["score"], reverse=True)[:8]
    sec_res = cards.query(
        query_embeddings=[query_embedding],
        n_results=min(len(top_folders) * 3, 24),
        where={"$and": [{"card_type": {"$eq": "section"}}, {"document_folder": {"$in": top_folders}}]},
    )
    sec_metas = (sec_res.get("metadatas") or [[]])[0]
    sec_dist = (sec_res.get("distances") or [[]])[0]
    for meta, dist in zip(sec_metas, sec_dist):
        folder = meta.get("document_folder", "")
        if folder not in best_docs:
            continue
        sp = meta.get("section_path", "") or meta.get("section_id", "")
        if not sp:
            continue
        sec_score = max(0.0, min(1.0, 1.0 - float(dist)))
        if not best_docs[folder]["top_section"] or sec_score > best_docs[folder].get("section_score", -1.0):
            best_docs[folder]["top_section"] = sp
            best_docs[folder]["section_score"] = sec_score

    ranked = sorted(best_docs.values(), key=lambda x: x["score"], reverse=True)
    filtered = [c for c in ranked if c["score"] >= 0.18 or c.get("lexical_score", 0.0) >= 0.34]
    return (filtered or ranked)[:5]


def find_relevant_documents(question: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """3-stage funnel: L1 semantic filter → L3 section confirmation (no LLM).

    Returns a ranked list of candidate dicts:
    [{folder, doc_name, top_section, score}]
    """
    client = get_openai_client()
    embedding_model = config["embeddings"]["model"]
    collection_name = config["vector_db"].get("summary_collection_name", "pdf_rag_summaries")

    chroma_client = _get_chroma_client(config)
    try:
        resp = client.embeddings.create(model=embedding_model, input=[question])
        query_embedding = resp.data[0].embedding
        candidates = _find_relevant_documents_card_first(question, config, chroma_client, query_embedding)
        if candidates:
            return candidates
    except Exception:
        logger.warning("Card-first candidate routing failed; using summary fallback.")

    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        logger.warning("Summary collection '%s' not found — no summaries indexed yet.", collection_name)
        return []

    # ── Stage 1: Level 1 filter — top-20 unique documents ───────────────────
    stage1_folders: list[str] = []
    try:
        resp = client.embeddings.create(model=embedding_model, input=[question])
        query_embedding = resp.data[0].embedding
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=20,
            where={"level": {"$eq": "1"}},
        )
        metas = (result.get("metadatas") or [[]])[0]
        seen: set[str] = set()
        for meta in metas:
            folder = meta.get("document_folder", "")
            if folder and folder not in seen:
                seen.add(folder)
                stage1_folders.append(folder)
    except Exception:
        logger.warning("Stage 1 summary query failed — returning empty candidates.")
        return []

    if not stage1_folders:
        return []

    # ── Stage 2: Level 3 confirmation — top section per document ────────────
    candidates: list[dict[str, Any]] = []
    try:
        where_filter: dict[str, Any]
        if len(stage1_folders) == 1:
            where_filter = {"$and": [
                {"level": {"$eq": "3"}},
                {"document_folder": {"$eq": stage1_folders[0]}},
            ]}
        else:
            where_filter = {"$and": [
                {"level": {"$eq": "3"}},
                {"document_folder": {"$in": stage1_folders}},
            ]}

        result2 = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(10, len(stage1_folders) * 3),
            where=where_filter,
        )
        metas2 = (result2.get("metadatas") or [[]])[0]
        distances2 = (result2.get("distances") or [[]])[0]

        best_per_doc: dict[str, dict[str, Any]] = {}
        for meta, dist in zip(metas2, distances2):
            folder = meta.get("document_folder", "")
            if not folder:
                continue
            semantic_score = max(0.0, min(1.0, 1.0 - float(dist)))
            doc_name = meta.get("document_name", Path(folder).name)
            top_section = meta.get("section_name", "")
            try:
                doc_meta = load_metadata(folder)
            except Exception:
                doc_meta = {}
            doc_card = (doc_meta or {}).get("document_card", {}) or {}
            lexical_score = _lexical_overlap_score(
                question,
                doc_name,
                top_section,
                doc_card.get("title", ""),
                doc_card.get("opening_text", ""),
                doc_card.get("l1_summary", ""),
            )
            # Blend semantic + lexical to reduce noisy candidates.
            score = (0.8 * semantic_score) + (0.2 * lexical_score)
            if folder not in best_per_doc or score > best_per_doc[folder]["score"]:
                best_per_doc[folder] = {
                    "folder": folder,
                    "doc_name": doc_name,
                    "top_section": top_section,
                    "score": score,
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                }

        ranked = sorted(best_per_doc.values(), key=lambda x: x["score"], reverse=True)
        # Filter weak candidates unless there is clear lexical support.
        filtered = [
            c for c in ranked
            if c["score"] >= 0.18 or c.get("lexical_score", 0.0) >= 0.34
        ]
        candidates = (filtered or ranked)[:5]

    except Exception:
        logger.warning("Stage 2 summary query failed — falling back to Stage 1 results.")
        candidates = [
            {
                "folder": f,
                "doc_name": load_metadata(f).get("document_name", Path(f).name),
                "top_section": "",
                "score": 0.0,
            }
            for f in stage1_folders[:5]
        ]

    return candidates


def _score_folders_by_summary(
    question: str,
    document_folders: list[str],
    config: dict[str, Any],
) -> list[str]:
    """Return document_folders ranked by L1 summary relevance to the question.

    Folders whose L1 summary is not yet indexed are appended at the end unranked.
    On any error the original list is returned unchanged.
    """
    if len(document_folders) <= 1:
        return document_folders

    try:
        client = get_openai_client()
        embedding_model = config["embeddings"]["model"]
        collection_name = config["vector_db"].get("summary_collection_name", "pdf_rag_summaries")
        chroma_client = _get_chroma_client(config)
        collection = chroma_client.get_collection(name=collection_name)

        resp = client.embeddings.create(model=embedding_model, input=[question])
        query_embedding = resp.data[0].embedding

        where_filter: dict[str, Any] = {"$and": [
            {"level": {"$eq": "1"}},
            {"document_folder": {"$in": document_folders}},
        ]}

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(len(document_folders), 50),
            where=where_filter,
        )
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        scored: dict[str, float] = {}
        for meta, dist in zip(metas, distances):
            folder = meta.get("document_folder", "")
            if folder and folder not in scored:
                scored[folder] = 1.0 - float(dist)

        ranked = sorted(scored, key=lambda f: scored[f], reverse=True)
        unranked = [f for f in document_folders if f not in scored]
        return ranked + unranked

    except Exception:
        logger.warning("Summary scoring failed — using original folder order.")
        return document_folders


def ask_across_documents(
    question: str,
    document_folders: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Retrieve real chunks from the most relevant documents and generate an answer.

    In legacy mode, documents are pre-ranked by L1 summary similarity before chunk
    retrieval. In tree mode, tree traversal performs its own routing.
    Sources include doc name, section, chunk number, and a text snippet for UI display.
    """
    retrieval_cfg = config.get("retrieval", {})
    use_tree = bool(retrieval_cfg.get("tree_traversal", False))
    mode = classify_query(question)
    max_docs = retrieval_cfg.get("multi_doc_max_docs", 5)
    if use_tree:
        # Tree routing is card-first and should not depend on summary collection ranking.
        document_folders = document_folders[:max_docs]
    else:
        document_folders = _score_folders_by_summary(question, document_folders, config)[:max_docs]

    if mode == "aggregation":
        return aggregation.aggregate(question, document_folders, config)

    client = get_openai_client()
    all_context_blocks: list[str] = []
    all_sources: list[dict[str, Any]] = []
    doc_names: list[str] = []

    if use_tree:
        tree = retrieve_tree(question, config, selected_document_folders=document_folders)
        if tree.get("abstain"):
            return {
                "question": question,
                "answer": "I cannot answer this from the available documents.",
                "sources": [],
                "multi_doc": True,
                "document_names": [],
                "confidence": tree.get("confidence", "low"),
                "mode": tree.get("mode", "specific"),
            }
        docs = tree.get("documents", [])
        metas = tree.get("metadatas", [])
        for text, meta in zip(docs, metas):
            doc_name = meta.get("document_name", "")
            folder = meta.get("document_folder", "")
            section_path = meta.get("section_path", "")
            chunk_num = meta.get("chunk_number", "?")
            label = f"{doc_name} > {section_path}" if section_path else f"{doc_name}, Chunk {chunk_num}"
            all_context_blocks.append(f"[Source: {label}]\n{text}")
            all_sources.append({
                "document_name": doc_name,
                "document_folder": folder,
                "section_path": section_path,
                "chunk_number": chunk_num,
                "chunk_text_snippet": text[:150],
                "chunk_path": meta.get("chunk_path", ""),
            })
        doc_names = sorted({s.get("document_name", "") for s in all_sources if s.get("document_name")})

    if not use_tree:
        for folder in document_folders:
            folder_path = Path(folder)
            metadata = load_metadata(folder_path)
            doc_name = metadata.get("document_name", folder_path.name)
            doc_names.append(doc_name)

            retrieval_dir = folder_path / "retrieval"
            ensure_directory(retrieval_dir)
            existing = list(retrieval_dir.glob("multidoc_*.json"))
            retrieval_path = retrieval_dir / f"multidoc_{len(existing) + 1:06d}.json"

            try:
                run_retrieval(
                    question=question,
                    document_folder=folder,
                    chroma_persist_dir=config["vector_db"]["persist_directory"],
                    collection_name=config["vector_db"]["collection_name"],
                    embedding_model=config["embeddings"]["model"],
                    retrieval_output_path=retrieval_path,
                    top_k=config["retrieval"]["top_k"],
                    media_top_k=config["retrieval"].get("media_top_k", 4),
                    expand_parent=config["retrieval"].get("expand_parent", True),
                    chroma_host=config["vector_db"].get("host"),
                    chroma_port=int(config["vector_db"].get("port", 8000)),
                )
            except Exception:
                logger.warning("Retrieval failed for document %s", folder)
                continue

            payload = read_json(retrieval_path, default={})
            texts = payload.get("documents", [])
            metas = payload.get("metadatas", [])

            for text, meta in zip(texts, metas):
                section_path = meta.get("section_path", "")
                chunk_num = meta.get("chunk_number", "?")
                label = f"{doc_name} > {section_path}" if section_path else f"{doc_name}, Chunk {chunk_num}"
                all_context_blocks.append(f"[Source: {label}]\n{text}")
                all_sources.append({
                    "document_name": doc_name,
                    "document_folder": folder,
                    "section_path": section_path,
                    "chunk_number": chunk_num,
                    "chunk_text_snippet": text[:150],
                    "chunk_path": meta.get("chunk_path", ""),
                })

    if not all_context_blocks:
        return {
            "question": question,
            "answer": "I could not retrieve relevant context from the selected documents.",
            "sources": [],
            "multi_doc": True,
            "document_names": doc_names,
        }

    context = "\n\n".join(all_context_blocks)
    prompt = (
        "You are answering questions across multiple documents. "
        "Answer based only on the provided context. "
        "For each factual sentence append a citation in this exact format: "
        "[Source: document_name, section_path]. "
        "If calculating totals or comparing values, show per-document values first, then the aggregate.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = client.chat.completions.create(
        model=config["chat"]["model"],
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Answer the user's question using only the retrieved document context. Cite document names.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    answer = response.choices[0].message.content or "I could not generate an answer."
    ok, reason = _citation_post_check(answer)
    citation_warning = ""
    if not ok:
        repaired = _repair_citations(client, config["chat"]["model"], answer, context, question)
        ok2, reason2 = _citation_post_check(repaired)
        if ok2:
            answer = repaired
        else:
            fallback = _attach_fallback_citations(answer or repaired, all_sources)
            ok3, reason3 = _citation_post_check(fallback)
            if ok3:
                answer = fallback
                citation_warning = f"citation check repaired with fallback tags ({reason}; {reason2})"
            else:
                # Do not hard-abstain immediately; return answer with warning so user still gets utility.
                citation_warning = f"citation check weak: {reason}; repair pass: {reason2}; fallback: {reason3}"

    payload = {
        "question": question,
        "answer": answer,
        "sources": all_sources,
        "multi_doc": True,
        "document_names": doc_names,
    }
    if citation_warning:
        payload["citation_warning"] = citation_warning
    return payload
