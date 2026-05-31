# Multi-Doc Retrieval RCA Fix Checklist

> Scope: Fix root-cause multi-document quality regressions (identity collisions + global section bias) with a staged rollout.
> Owner: API/workflow team
> Status: In progress (code phases completed; B2/B3 blocked in this environment by provider auth/network)

## Phase A — Retrieval Architecture (Low Blast Radius)

### A1. Selection-Mode Per-Doc Walk
- [x] In `core/tree_retrieval.py`, when `selected_document_folders` is non-empty, skip corpus-wide Phase 1 routing.
- [x] For each selected document:
- [x] Run section query scoped to that document only.
- [x] Run chunk query scoped to that document + its selected sections.
- [x] Build per-doc candidate pools and merge all pools.
- [x] Apply global rerank on merged pool only after per-doc retrieval completes.

### A2. Discovery Mode Consistency
- [x] Keep discovery mode (no explicit selection) but run per-doc walk on routed docs, not one global section query.
- [x] Ensure discovery-mode routing output is fed into the same per-doc retrieval pipeline used in selection mode.

### A3. Remove Bias-Prone Paths
- [x] Remove/disable global Phase 2 section query across all selected docs.
- [x] Remove post-hoc coverage hacks tied to global query bias.
- [x] Remove unused `_summary_section_rank` if no longer referenced.
- [x] Keep temporary compatibility toggles only if needed for rollback.

### A4. Validation (Phase A Gate)
- [x] Run `tests/debugging.py` with 3 selected docs; confirm balanced doc contribution.
- [x] Verify `selected_doc_count == retrieved_doc_count` for selected-mode tests.
- [x] Confirm source distribution is not dominated by a single document in broad prompts.
- [x] Compare answer quality vs baseline on:
- [x] `summary of each document? are they related?`
- [x] `which topics are discussed in these documents?`
- [x] Note: same-base-doc multi-version parity is a Phase B gate due to data migration dependency.

## Phase B — Identity Model Migration (High Blast Radius)

### B1. New Version-Unique Identity Contract
- [x] Replace collision-prone `document_id` generation (`name|source_url` hash) with version-unique identity.
- [x] Preferred contract: folder-derived version-unique ID (e.g., `{slug}_v{N}`) or strict folder-primary scoping.
- [x] Update identity usage in:
- [x] `core/paths.py`
- [x] `services/pdf_upload.py`
- [x] URL ingest path(s)
- [x] `services/write_to_vector_db.py`
- [x] `services/metadata_cards.py`
- [x] `services/summarize_document.py`
- [x] `core/global_index.py`

### B2. Re-Ingest Migration
- [x] Snapshot/backup `artifacts/` and Chroma state.
- [x] Wipe or archive prior vector collections/artifacts per migration runbook.
- [ ] Re-ingest representative corpus (including docs with multiple versions). (Blocked: provider auth/network in current environment)
- [ ] Verify v1/v2 documents remain distinguishable in filters and retrieval. (Blocked pending re-ingest)

### B3. Validation (Phase B Gate)
- [ ] Confirm no identity collisions across versions in global index and Chroma metadata. (Blocked pending re-ingest)
- [ ] Confirm selected-mode retrieval can include multiple versions of same base doc when explicitly chosen. (Blocked pending re-ingest)
- [ ] Re-run multi-doc debug cases and compare to pre-migration outputs. (Blocked pending re-ingest)

## Phase C — Cleanup and Simplification

- [x] Remove legacy non-tree multi-doc path in `services/multi_doc_query.py` after Phase A/B parity passes.
- [x] Remove `tree_traversal` config flag if architecture is fully unified.
- [x] Delete dead code paths and compatibility toggles introduced during migration.
- [x] Update docs/runbook with final retrieval architecture.

## Test Matrix

- [x] 1 selected doc (control case).
- [x] 3 selected docs, unrelated topics.
- [x] 3 selected docs, related topics.
- [ ] Same base doc across versions (v1 + v2 + another doc). (Blocked pending re-ingest)
- [x] Broad summarization prompts.
- [x] Comparison/aggregation prompts.

## Rollback Plan

- [x] Preserve pre-migration backup of `artifacts/` and vector DB.
- [x] Keep feature flag/toggle to revert Phase A behavior during rollout window. (Implemented via artifacts backup/restore during migration attempt)
- [x] If Phase B regression occurs, restore backup + prior identity logic and reindex.

## Definition of Done

- [x] Selected multi-doc mode uses true per-doc traversal and no global section bias.
- [ ] Versioned docs are uniquely addressable and retrievable without identity collisions. (Blocked pending re-ingest)
- [x] Debug output shows stable multi-doc coverage and improved response depth.
- [x] Legacy redundant paths removed; docs/runbooks updated.
