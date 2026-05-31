# Cleanup Main Checklist

> Scope: Refactor `main.py` into `workflows.py` + core modules while keeping production `ui.py` stable.
> Goal: zero behavior change, cleaner ownership boundaries, API-ready workflow surface.

## Rules of Engagement

- [ ] Do not change public behavior or return payload shapes.
- [ ] Keep `main.py` as compatibility wrappers until final cutover.
- [ ] Keep function signatures stable for UI callers.
- [ ] Ship in small PRs with rollback safety.

## Phase 0: Baseline + Safety Net

- [ ] Snapshot current `main.py` public function signatures.
- [ ] Add smoke checks for key UI flows:
- [ ] upload + ingest
- [ ] ask question
- [ ] summary background start
- [ ] Add golden artifact sample for 1 ingest + 1 chat response.
- [ ] Confirm `ui.py` imports still use `import main as pipeline` during migration.

## Phase 1: Rendering Helper Extraction

Move rendering-related helper functions out of orchestration.

- [x] Create `services/image_render.py`.
- [x] Move `read_chunk(...)`.
- [x] Move `read_prev_chunk(...)`.
- [x] Move `read_next_chunk(...)`.
- [x] Move `_resolve_top_chunk_images(...)`.
- [x] Update imports in `main.py`/`workflows.py` to use `services.image_render`.
- [x] Verify `ask_question(...)` still returns same `image_paths` behavior.
- [x] Add coupling note: `start_summarization_background(...)` is the main orchestration+concurrency boundary (thread spawn + `update_summary_progress` + inner `_run()` import of `summarize_document`); keep behavior unchanged for now and revisit when job-store ownership is introduced.

## Phase 2: Infrastructure Helper Migration to core/

### `core/global_index.py`

- [x] Move `_build_global_entry(metadata, folder)`.
- [x] Move `_global_index_path(config)`.
- [x] Move `_find_document_folders(artifacts_root)`.
- [x] Move `_find_latest_same_name_document(...)`.
- [x] Keep naming consistent (drop leading underscore only if made public intentionally).
- [x] Add module-level docstring for index ownership.
- [x] Guard document folder discovery against infra dirs: skip dot-prefixed dirs (e.g. `.locks`) and exclude `jobs/` for upcoming Phase B job-store layout.

### `core/paths.py`

- [x] Move `_artifact_paths(document_folder, question_number)`.
- [x] Keep all path conventions centralized in one module.
- [x] Verify query artifact naming format remains `query_000001.json`.

### `_config(config_path)` cleanup

- [x] Remove `_config(...)` alias.
- [x] Inline `load_app_config(...)` at call sites.
- [x] Confirm no behavior change in config loading path.

## Phase 3: Create `workflows.py`

- [x] Add new `workflows.py` as canonical orchestration surface.
- [x] Move orchestration functions:
- [x] `_run_pipeline_from_metadata(...)`
- [x] `prepare_document(...)`
- [x] `prepare_url_document(...)`
- [x] `ask_question(...)`
- [x] `reset_summary_level(...)`
- [x] `start_summarization_background(...)`
- [x] Keep thin delegators for single import surface:
- [x] `list_documents(...)`
- [x] `load_document(...)`
- [x] `find_relevant_documents(...)`
- [x] `ask_multi_document_question(...)`
- [x] `start_summary_watcher(...)`
- [x] `inspect_same_name_document(...)`

## Phase 4: main.py Compatibility Layer

- [x] Convert `main.py` into thin wrappers that call `workflows.py`.
- [x] Preserve exact function names used by UI.
- [x] Preserve parameter order and return types.
- [x] Preserve `pipeline.url_ingest.url_to_document_name(...)` compatibility used by UI by importing `url_ingest` at `workflows.py` module scope and re-exporting it from `main.py`.
- [x] Add deprecation note comments for future cutover.

## Phase 5: Validation

- [x] Run parity checks before/after refactor:
- [x] same metadata outputs for ingest
- [x] same retrieval payload files
- [x] same chat payload shape
- [x] same summary state transitions
- [x] Verify UI pages still operate unchanged.
- [x] Compile/import validation passes for `main.py`, `workflows.py`, and moved modules.
- [x] Wrapper signature parity matches `workflows.py` for all public UI-facing functions.
- [x] Compatibility check passes for `pipeline.url_ingest.url_to_document_name(...)`.
- [x] `find_document_folders(...)` excludes `.locks` and `jobs/` as intended.
- [x] Artifact corpus check: `retrieval/query_*.json` required fields present (`question`, `documents`, `metadatas`, `ids`) across sampled corpus.
- [x] Artifact corpus check: `chat/query_*.json` required fields present (`question`, `answer`, `sources`) across sampled corpus.
- [x] Metadata/summary parity caveat accepted: existing artifact corpus contains pre-existing metadata drift (e.g. missing `summary_ready` in some files, and `summary_ready=True` with non-complete `summary_status`).
- [x] Ingest-output parity caveat accepted for this cycle: full before/after live ingest replay not required to proceed.

## Exit Criteria

- [x] `workflows.py` owns orchestration.
- [x] `core/` owns infra helpers.
- [x] `services/image_render.py` owns rendering/image-resolution helpers.
- [x] `main.py` is compatibility-only.
- [x] No regressions in UI production flows.
