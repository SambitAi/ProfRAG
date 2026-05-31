# Multi-doc retrieval: architectural rewrite

## Context

Symptom: chatting with 3 documents returns chunks from only 1. The user demonstrated this with a summarization question — which sits in the **first metadata layer** of the tree. If even that level can't span all selected docs, deeper needle-in-haystack retrieval is hopeless.

**This is a v2 regression. v1.1 had multi-doc working correctly.**

In v1.1 (commit `64d48ff` "V1 Stable Release"), [services/multi_doc_query.py](../../services/multi_doc_query.py) `ask_across_documents` walked **per-doc**:

```python
for folder in document_folders:
    retrieve_context.run(
        question=question,
        document_folder=folder,
        top_k=config["retrieval"]["top_k"],
        ...
    )
    # each doc gets its own retrieval, chunks are merged for context
```

Each selected doc got its own top-k pull against the chunks collection, scoped to that doc's folder. No global section search, no coverage hacks. Every selected doc was guaranteed to contribute chunks because the retrieval loop ran once per doc, independently.

That legacy path **still exists** in the current codebase at [services/multi_doc_query.py:535-584](../../services/multi_doc_query.py#L535-L584), gated by `retrieval.tree_traversal: false`. The current [config/app_config.yaml](../../config/app_config.yaml) has `tree_traversal: true`, which routes through the v2 tree path in [core/tree_retrieval.py](../../core/tree_retrieval.py) — which introduced the global Phase 2 section search and broke multi-doc.

The plan below restores the v1.1 per-doc principle while keeping the v2 section-card layer scoped per-doc (so we don't lose the hierarchical-retrieval improvement v2 brought).

---

## Two architectural defects

**Defect 1 — `document_id` is hashed.**
[core/paths.py:107-109](../../core/paths.py#L107-L109):
```python
return hashlib.sha256(f"{name}|{source_url}".encode()).hexdigest()[:12]
```
Same filename / same URL → same hash → v1 and v2 collide. `top_doc_ids = list(dict.fromkeys(...))` silently collapses them into one Chroma-queryable entity. Meanwhile the folder name is already `{slug}_v{N}` — version-unique, human-readable, on disk. The folder basename IS the correct ID.

**Defect 2 — Global Phase 1 + Phase 2 search runs even when docs are pre-selected.**
When `selected_document_folders` is set, the user has already told the system which docs to use. Yet `retrieve_tree` still:
- Runs Phase 1 card-first scoring across the whole corpus ([core/tree_retrieval.py](../../core/tree_retrieval.py) lines 200-244) → wasted compute, `top_doc_ids` gets overwritten from `scoped_pairs` anyway at lines 249-263.
- Runs Phase 2 as ONE global section search across all selected docs (lines 265-300) → semantically dominant doc fills `section_paths`, Phase 3 inherits the bias.

If documents are already selected, there is no doc-routing problem to solve — only a per-doc traversal problem. v1.1 understood this; v2 forgot it.

Both fixes are independent. Fix 1 must land first so per-doc filters in Fix 2 can distinguish versions.

---

## Fix 1 — Identity: `document_id = {slug}_v{N}`

`document_id` becomes literally the folder basename. No hashing, anywhere.

The folder name is already produced by [core/paths.py:15-16](../../core/paths.py#L15-L16) `doc_folder_name_from(name, version) → "{slug}_v{N}"`. We adopt that string as the ID.

Properties:
- **Filename-based**: `april_2025_v1`, `medium_ai_study_assistant_v2`
- **Reusable**: same input file → same ID, deterministic
- **Version-distinct**: v1 ≠ v2
- **Self-describing**: read the ID, know the doc + version
- **No maps required**: ID matches folder name matches global index key

### Files to change

| File | Change |
|---|---|
| [core/paths.py:107-109](../../core/paths.py#L107-L109) `compute_document_id` | Delete the SHA256 logic. New signature: `compute_document_id(slug: str, version: int) -> str` returning `f"{slug}_v{version}"`. Or delete the function entirely and replace callers with `Path(folder).name`. |
| [services/pdf_upload.py:31](../../services/pdf_upload.py#L31) | After computing `document_folder`, set `document_id = document_folder.name`. Write to `metadata.json`. |
| URL ingest path (search for remaining `compute_document_id` callers) | Same pattern: `document_id = Path(document_folder).name`. |
| [services/write_to_vector_db.py:34-71](../../services/write_to_vector_db.py#L34-L71) | Read `document_id` from `metadata.json` (it now equals folder basename). No schema change to chunk metadata — same `document_id` field, just correct values. |
| [services/metadata_cards.py](../../services/metadata_cards.py) lines 163, 202, 337 | Replace internal `compute_document_id(...)` recomputations with `Path(document_folder).name`. |
| [services/summarize_document.py](../../services/summarize_document.py) lines 155, 347 | Same: replace `compute_document_id(...)` with `Path(document_folder).name`. |
| [core/global_index.py:145-162](../../core/global_index.py#L145-L162) `build_global_entry` | `document_id` field set to `Path(folder).name`. The map key and `document_id` value become the same string. |

### What stays unchanged

- Folder naming (`core/paths.py:15-35`, `next_document_version`) — already produces the correct strings
- All Chroma `where` filters using `{"document_id": ...}` — they keep working, but now the values flowing in actually distinguish versions
- Global index key scheme

### Migration

**Re-ingest everything.** Delete `artifacts/` and `artifacts/chroma_db/`, then re-upload all docs through the UI. Old hash-based data is orphaned. Test data is small.

---

## Fix 2 — Traversal: restore v1.1 per-doc walk, with v2 section cards

When `selected_document_folders` is non-empty, the user has resolved the doc-routing question. Skip Phase 1 entirely. Skip the global Phase 2 section search. Walk each selected doc as its own independent tree, then merge & rerank.

This is the v1.1 design with one improvement: each per-doc walk uses the v2 section-card layer (filtered to *that one doc*) before pulling chunks, instead of just running a flat top-k chunk query.

### New flow

**Selection mode (multi-doc OR single-doc with explicit folder):**
```
[Skip Phase 1]    — no global doc routing needed
[Per selected doc, independently]:
    [Phase 2-doc]  Section query filtered to ONE doc → its section_paths
    [Phase 3-doc]  Leaf chunks filtered to this doc + its sections → doc's best K chunks
    [Backtrack-doc] If sparse, requery without section filter
[Merge]    all per-doc pools, dedup by chunk id
[Rerank]   global rerank over merged pool → effective_top_k
[Return]
```

**Discovery mode (no selection — e.g., "find relevant docs for me"):**
```
[Phase 1]  Card-first global document scoring → top docs
[then the same per-doc walk for those top docs]
```

The discovery branch keeps Phase 1 because there is a genuine routing question. The selection branch has no routing question — answering it is the user's job.

### Rewrite in [core/tree_retrieval.py](../../core/tree_retrieval.py)

Restructure `retrieve_tree`:

```python
def retrieve_tree(question, config, selected_document_folders=None):
    selected_document_folders = selected_document_folders or []
    mode = classify_query(question)

    global_index = load_global_index(...)
    docs_map = (global_index or {}).get("documents", {}) or {}

    if selected_document_folders:
        allowed = {Path(f).name for f in selected_document_folders}
        scoped_docs = {k: v for k, v in docs_map.items() if k in allowed}
    else:
        scoped_docs = dict(docs_map)

    if not scoped_docs:
        return _abstain(mode)

    client = get_openai_client()
    q_resp = client.embeddings.create(model=config["embeddings"]["model"], input=[question])
    query_embedding = q_resp.data[0].embedding

    chroma = _get_chroma_client(config)
    cards  = chroma.get_or_create_collection(name=config["vector_db"].get("card_collection_name", "pdf_rag_cards"))
    chunks = chroma.get_or_create_collection(name=config["vector_db"]["collection_name"])

    scoped_pairs = _build_scoped_pairs(scoped_docs)
    if not scoped_pairs:
        return _abstain(mode)

    if selected_document_folders:
        # Selection mode: skip Phase 1, walk per-doc directly (v1.1 principle)
        target_pairs = scoped_pairs
        confidence = "normal"
        max_score = 1.0
    else:
        # Discovery mode: Phase 1 card-first routing first
        target_pairs, confidence, max_score = _phase1_route(
            cards=cards, scoped_pairs=scoped_pairs, scoped_docs=scoped_docs,
            query_embedding=query_embedding, question=question, config=config,
            client=client,
        )
        if not target_pairs:
            return _abstain(mode)

    pool_docs, pool_metas, pool_ids = _walk_per_doc(
        cards=cards, chunks=chunks, target_pairs=target_pairs,
        query_embedding=query_embedding, config=config,
    )

    cap = max(8, int(config.get("retrieval", {}).get("multi_doc_chunk_cap", 20)))
    effective_top_k = min(max(8, len(target_pairs) * 2), cap)
    docs, metas, ids = rerank_items(
        query=question, documents=pool_docs, metadatas=pool_metas, ids=pool_ids,
        top_k=effective_top_k,
    )

    return {
        "mode": mode, "confidence": confidence, "abstain": False,
        "documents": docs, "metadatas": metas, "ids": ids,
        "candidate_document_ids": [did for did, _ in target_pairs],
        "candidate_section_paths": [],
        "max_doc_score": max_score,
    }
```

Two new helpers replace the deleted code:

```python
def _walk_per_doc(*, cards, chunks, target_pairs, query_embedding, config):
    retrieval_cfg = config.get("retrieval", {})
    per_doc_section_k = max(2, int(retrieval_cfg.get("per_doc_section_k", 4)))
    per_doc_leaf_k    = max(3, int(retrieval_cfg.get("per_doc_leaf_k", 6)))

    try:    cards_count  = int(cards.count())
    except Exception: cards_count = 0
    try:    chunks_count = int(chunks.count())
    except Exception: chunks_count = 0

    pool_docs:  list[str] = []
    pool_metas: list[dict[str, Any]] = []
    pool_ids:   list[str] = []
    seen_ids: set[str] = set()

    for did, folder in target_pairs:
        section_paths = _query_sections_for_doc(
            cards=cards, document_id=did, document_folder=folder,
            query_embedding=query_embedding,
            n_results=per_doc_section_k, cards_count=cards_count,
        )
        d_docs, d_metas, d_ids = _query_chunks_for_doc(
            chunks=chunks, document_id=did, document_folder=folder,
            section_paths=section_paths, query_embedding=query_embedding,
            n_results=per_doc_leaf_k, chunks_count=chunks_count,
        )
        if len(d_docs) < 2:  # backtrack for this doc only — drop section filter
            d_docs, d_metas, d_ids = _query_chunks_for_doc(
                chunks=chunks, document_id=did, document_folder=folder,
                section_paths=[], query_embedding=query_embedding,
                n_results=per_doc_leaf_k, chunks_count=chunks_count,
            )
        for d, m, i in zip(d_docs, d_metas, d_ids):
            if i and i not in seen_ids:
                seen_ids.add(i)
                pool_docs.append(d); pool_metas.append(m); pool_ids.append(i)

    return pool_docs, pool_metas, pool_ids


def _query_sections_for_doc(*, cards, document_id, document_folder,
                            query_embedding, n_results, cards_count) -> list[str]:
    if cards_count <= 0 or n_results <= 0:
        return []
    base = ({"document_id": {"$eq": document_id}} if document_id
            else {"document_folder": {"$eq": document_folder}})
    where = {"$and": [{"card_type": {"$eq": "section"}}, base]}
    try:
        res = cards.query(query_embeddings=[query_embedding],
                          n_results=min(n_results, cards_count), where=where)
    except Exception:
        return []
    paths: list[str] = []
    for m in (res.get("metadatas") or [[]])[0]:
        sp = m.get("section_path", "")
        if sp and sp not in paths:
            paths.append(sp)
    return paths


def _query_chunks_for_doc(*, chunks, document_id, document_folder, section_paths,
                          query_embedding, n_results, chunks_count
                          ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    if chunks_count <= 0 or n_results <= 0:
        return [], [], []
    base = ({"document_id": {"$eq": document_id}} if document_id
            else {"document_folder": {"$eq": document_folder}})
    where = {"$and": [base, {"section_path": {"$in": section_paths}}]} if section_paths else base
    try:
        res = chunks.query(query_embeddings=[query_embedding],
                           n_results=min(n_results, chunks_count), where=where)
    except Exception:
        return [], [], []
    return (
        (res.get("documents")  or [[]])[0],
        (res.get("metadatas")  or [[]])[0],
        (res.get("ids")        or [[]])[0],
    )
```

`_phase1_route` is just the existing Phase 1 card scoring + `_auto_doc_k` selection, lifted into its own helper, returning `(target_pairs, confidence, max_score)`. Only runs in discovery mode.

### What gets deleted

- `_summary_section_rank` (no longer used; per-doc cards-only is sufficient)
- `_auto_section_k`, `_auto_leaf_k` helpers (replaced by config constants)
- The Phase 2 global section query (lines 265-300)
- The entire pre-rerank coverage block (~lines 355-499)
- The entire post-rerank coverage block (~lines 501-547)
- Phase 1 work performed in selection mode (Phase 1 now only runs in discovery mode)
- The legacy non-tree path at [services/multi_doc_query.py:535-584](../../services/multi_doc_query.py#L535-L584) — no longer needed since the tree path now does the right thing
- The `tree_traversal` config flag — only one path remains

### What stays

- `_should_abstain`, `_llm_should_continue`, `_summary_doc_rank` — used by discovery mode's Phase 1
- [services/aggregation.py](../../services/aggregation.py) aggregation path (unchanged)
- Single-doc path: when `selected_document_folders` has 1 entry, `_walk_per_doc` runs once for that doc — no regression, simpler code path than before

### Config additions to [config/app_config.yaml](../../config/app_config.yaml)

```yaml
retrieval:
  # existing keys ...
  per_doc_section_k: 4   # sections to fetch per selected doc
  per_doc_leaf_k: 6      # chunks to fetch per selected doc
  multi_doc_chunk_cap: 20  # cap on final reranked output
  # tree_traversal: REMOVED — only one path exists now
```

---

## Order of execution

1. **Fix 1** — replace hash with `Path(folder).name` in the 6 files listed
2. Delete `artifacts/` and `artifacts/chroma_db/`, **re-ingest** all docs through the UI
3. **Fix 2** — rewrite `retrieve_tree` to split selection vs discovery, add `_walk_per_doc` + two query helpers, add config keys, remove legacy non-tree multi-doc path
4. **Restart the API server**

---

## Verification

1. **Identity check**: Re-ingest the same PDF twice. Inspect `artifacts/{slug}_v1/metadata.json` and `artifacts/{slug}_v2/metadata.json` — `document_id` values must be `"{slug}_v1"` and `"{slug}_v2"`, matching folder names. Direct Chroma probe:
   ```python
   col.get(where={"document_id": {"$eq": "<slug>_v1"}})  # only v1 chunks
   col.get(where={"document_id": {"$eq": "<slug>_v2"}})  # only v2 chunks
   ```

2. **Selection-mode skips Phase 1**: Add a temporary log line at the top of `_phase1_route`. Confirm it does NOT fire when selecting docs in the UI. Remove the log after verification.

3. **Traversal — multi-doc** (the v1.1 behavior restored): Select 3 genuinely different documents. Ask a factual cross-document question (not summarization):
   *"What methods or approaches are described across these documents?"*
   - The `sources` list must contain chunks from **all 3** distinct `document_id` values.
   - UI References must list 3 distinct document names.

4. **Needle-in-haystack**: Add a unique sentence to one of the 3 docs (e.g., "The capital of Mars is Olympus Mons"). Ask: "What is the capital of Mars?" The answer must cite that doc specifically — proves per-doc *depth*, not just breadth.

5. **Single-doc regression**: Ask a question against one doc in single-doc chat. Quality unchanged.

6. **Discovery mode regression**: Ask a question without selecting any docs (if the UI supports that flow). Phase 1 routing must still run and select reasonable candidates.

7. **Aggregation regression**: Ask an aggregation-mode question. [services/aggregation.py](../../services/aggregation.py) path runs (no tree retrieval), works as before.
