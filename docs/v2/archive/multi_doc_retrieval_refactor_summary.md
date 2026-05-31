# Multi-Doc Retrieval Refactor Summary

Date: 2026-05-31
Status: Code migration complete; data migration partially blocked by environment auth/network.

## What Was Done

### 1) Retrieval architecture unified to tree path
Why:
- Eliminate split behavior and legacy branch drift.
- Ensure selected-doc retrieval uses one predictable path.

Where:
- `services/multi_doc_query.py`
- `workflows.py`
- `config/app_config.yaml`

What changed:
- Removed legacy non-tree multi-doc retrieval path.
- Removed `retrieval.tree_traversal` toggle usage and config dependency.
- Single-doc and multi-doc retrieval now both run through tree retrieval.

### 2) Multi-doc retrieval bias reduced at core retrieval layer
Why:
- Per-doc retrieval breadth was being collapsed by reranker cutoff.

Where:
- `core/tree_retrieval.py`

What changed:
- Per-doc section + chunk walk retained.
- `effective_top_k` now scales with routed doc count and per-doc retrieval breadth.
- Post-rerank per-doc floor guarantee added to reduce single-doc dominance.
- Unused `candidate_section_paths` return plumbing removed.

### 3) Identity model migrated in code to version-unique folder IDs
Why:
- Avoid hash collisions across versions of same source.

Where:
- `core/paths.py`
- `services/pdf_upload.py`
- `services/web_upload.py`
- `services/write_to_vector_db.py`
- `services/metadata_cards.py`
- `services/summarize_document.py`
- `core/global_index.py`

What changed:
- Hash-based `document_id` generation removed from active call paths.
- Folder-derived version-unique identity is now the contract in code paths.

### 4) Cleanup/dead code reduction
Why:
- Remove maintenance overhead and stale logic.

Where:
- `services/multi_doc_query.py`
- `core/paths.py`

What changed:
- Removed unreachable legacy block after candidate return path.
- Removed dead `compute_document_id` helper.

## What Is Still Pending (Operational)

### B2/B3 re-ingest and validation
Why pending:
- Environment-level auth/network block while fetching provider token prevented clean re-ingest in this run.

Pending actions:
- Backup + wipe/rotate existing artifact/vector state.
- Re-ingest representative corpus.
- Validate versioned-doc separation in Chroma/global index.
- Re-run multi-doc validation matrix post re-ingest.

## User-visible outcome

- Multi-doc chat is more stable and less likely to collapse to one selected document.
- Retrieval behavior is more consistent between single-doc and multi-doc paths.
- API and workflow surfaces are cleaner due to legacy path removal.
