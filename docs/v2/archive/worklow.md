# Worklow Checklist

> Purpose: To-do checklist with concrete actions to execute workflow-layer refactor safely.
> Note: Filename intentionally matches request (`worklow.md`).

## A. Setup Actions

- [x] Create branch: `refactor/workflows-layer`.
- [x] Freeze current `main.py` signatures in a quick reference note.
- [x] Identify all imports of `main.py` across repo.
- [x] Add temporary migration log section in PR description template.

## B. Code Actions

- [x] Add `services/image_render.py` and move 4 rendering helpers.
- [x] Add missing unit tests for moved helpers.
- [x] Move global-index helpers into `core/global_index.py`.
- [x] Move artifact-path helper into `core/paths.py`.
- [x] Remove `_config(...)` helper and inline config load.
- [x] Create `workflows.py` and move orchestration functions.
- [x] Keep thin delegators in `workflows.py` for one import surface.
- [x] Turn `main.py` into wrapper-only module.

## C. Validation Actions

- [x] Run ingest test on existing PDF fixture.
- [x] Run URL ingest test (HTML + PDF URL paths).
- [x] Run single-doc chat test.
- [x] Run multi-doc relevance and answer test.
- [x] Run summary start/reset lifecycle test.
- [x] Compare before/after artifact outputs for parity.

## D. API-Readiness Actions

- [x] Ensure API layer imports only `workflows.py`.
- [ ] Confirm no business logic duplication in API handlers.
- [ ] Add/confirm normalized error envelope path.
- [x] Add structured logging keys at workflow boundaries.

## E. Rollout Actions

- [ ] Deploy with `main.py` wrappers still enabled.
- [ ] Monitor logs for workflow errors and thread issues.
- [ ] Keep rollback path: point callers back to old functions if needed.
- [ ] After stable burn-in, mark wrapper layer for deprecation.

## Done Criteria

- [x] Production UI behavior unchanged.
- [x] Workflow logic centralized in `workflows.py`.
- [x] Helpers located in correct ownership modules.
- [x] API layer can scale as a thin transport layer.
