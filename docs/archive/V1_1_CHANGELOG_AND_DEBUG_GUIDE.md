# v1.1 Changelog And Debug Guide

Purpose: single reference for what changed in v1.1, where it changed, and why.  
Use this file first when triaging regressions.

## 1) Retrieval Architecture Shift

- What changed:
  - Retrieval became metadata/card-driven for narrowing, with chunk evidence from Chroma.
  - Tree traversal added for document -> section -> chunk flow.
- Where:
  - `core/tree_retrieval.py`
  - `services/multi_doc_query.py`
  - `main.py` (`ask_question`, `ask_multi_document_question`)
- Why:
  - Reduce token cost and improve “needle in haystack” recall by narrowing before full chunk retrieval.

## 2) Stable Document Identity

- What changed:
  - Added stable `document_id` and document-prefixed vector IDs.
- Where:
  - `core/paths.py` (`compute_document_id`)
  - `services/pdf_upload.py`, `services/web_upload.py`
  - `services/write_to_vector_db.py`, `services/summarize_document.py`
- Why:
  - Avoid vector collisions across v1/v2 re-ingests of same file name.

## 3) Metadata Model (Control Plane)

- What changed:
  - Global index (`artifacts/metadata.json`) and per-doc metadata responsibilities clarified.
  - Atomic writes for top-level metadata files.
- Where:
  - `core/global_index.py`
  - `core/metadata.py`
  - `core/storage.py` (`write_json_atomic`)
  - `main.py` (`_build_global_entry`, `list_documents`)
- Why:
  - Fast corpus catalog reads + crash-safe metadata updates.

## 4) Card Layer (Routing Layer)

- What changed:
  - Document cards + section cards created and indexed.
  - Card vectors stored in Chroma; metadata keeps references only.
- Where:
  - `services/metadata_cards.py`
  - `core/tree_retrieval.py`
- Why:
  - Improve routing precision before chunk retrieval and keep JSON lightweight.

## 5) Summary Runtime

- What changed:
  - Persistent summary watcher, resumable L3->L2->L1 pipeline.
  - Summary merge into cards (`l1_summary`, `l3_summary`, `summary_card_id`).
  - Retry/backoff + parallelism control for rate limits.
- Where:
  - `services/summary_watcher.py`
  - `services/summarize_document.py`
  - `config/app_config.yaml` (`summarizer.max_parallel_sections`, retry settings)
- Why:
  - Non-blocking enrichment and stable operation under provider throttling (429).

## 6) UI/Output Reliability Updates

- What changed:
  - Single-doc abstain short-circuit.
  - Low-confidence and citation warning handling.
  - Clickable citation tokens + unique anchor IDs.
  - Routing evidence panel (“Why this matched”).
  - Watcher status + retry failed summaries.
  - Restored image rendering fallback for adjacent/section-related chunks.
- Where:
  - `ui.py`
  - `main.py` (`_resolve_top_chunk_images`)
  - `services/multi_doc_query.py` (citation post-check/repair)
- Why:
  - Better explainability, fewer false negatives, and stable evidence rendering.

## 7) Phase 6 Baseline (Structured Fields + Aggregation)

- What changed:
  - Regex-based typed field extraction added.
  - Initial deterministic aggregation route (`mode == aggregation`).
- Where:
  - `services/field_extractor.py`
  - `services/aggregation.py`
  - `services/multi_doc_query.py`
  - `main.py` + `config/app_config.yaml`
- Why:
  - Low-cost, deterministic aggregation foundation without mandatory LLM extraction.

## 8) Common Regression Patterns And Fast Checks

1. Wrong/no routing candidates
- Check:
  - `artifacts/metadata.json` has document entries and `ready_to_chat=true`.
  - Card collection exists and contains `card_type=document/section`.
- Files:
  - `services/multi_doc_query.py`
  - `core/tree_retrieval.py`

2. No images shown in answer
- Check:
  - Retrieved sources include chunk references near image chunks.
  - Chunk JSON `image_paths` populated for nearby section/chunks.
- Files:
  - `main.py` (`_resolve_top_chunk_images`)
  - `artifacts/<doc>/chunks/*.json`

3. 429 / RESOURCE_EXHAUSTED during summarization
- Check:
  - `summarizer.max_parallel_sections`
  - `summarizer.retry_attempts`
  - `summarizer.retry_base_seconds`
- Files:
  - `services/summarize_document.py`
  - `config/app_config.yaml`

4. Cross-version vector collisions
- Check:
  - IDs in Chroma include `document_id` prefix.
  - Per-doc metadata has unique `document_id`.
- Files:
  - `services/write_to_vector_db.py`
  - `core/paths.py`

## 9) Non-Negotiable Invariants

- Chroma handles semantic similarity; metadata files do not.
- `artifacts/metadata.json` is global catalog only.
- `{doc}/metadata.json` is per-document lifecycle state.
- Vector IDs must remain `document_id`-prefixed.
- Top-level metadata writes must stay atomic.
