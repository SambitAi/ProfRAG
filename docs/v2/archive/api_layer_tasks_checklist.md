# API Layer Tasks Checklist

> Scope: Build production-safe HTTP API over existing `workflows.py` without duplicating business logic.
> Status target: complete Sections A-H before external exposure.
> Hard dependency: complete Section E (locking/concurrency safety) before enabling Section C write routes and Section D async job execution.

## A. Foundation

- [x] Create `api/` package root.
- [x] Create `api/app.py` FastAPI app factory.
- [x] Create `api_server.py` uvicorn entry point.
- [x] Create `api/deps.py` for shared dependencies/config injection.
- [x] Create `api/settings.py` to map `config/app_config.yaml` API options.
- [x] Update `requirements.txt` with API runtime deps (`fastapi`, `uvicorn[standard]`, `python-multipart`).
- [x] Ensure all API handlers import and call `workflows.py` only.

## B. Contracts and DTOs

- [x] Create `api/schemas/common.py` (`ErrorResponse`, `JobStatus`, `HealthResponse`).
- [x] Create `api/schemas/documents.py` request/response models.
- [x] Create `api/schemas/chat.py` request/response models.
- [x] Create `api/schemas/jobs.py` request/response models.
- [x] Add strict response modeling for abstain/non-abstain chat payloads.
- [x] Add compatibility mapping from workflow dict payloads to DTOs.

## C. Routers and Endpoints

- [x] Add `api/routers/documents.py`.
- [x] Add `api/routers/chat.py`.
- [x] Add `api/routers/jobs.py`.
- [x] Add `api/routers/summaries.py` or explicitly document why summary routes stay in `documents.py`.
- [x] Add `api/routers/health.py`.
- [x] Register all routers in `api/app.py`.

### Document routes
- [x] `GET /v2/documents`
- [x] `GET /v2/documents/{folder}` (full metadata payload; `load_document` equivalent)
- [x] `GET /v2/documents/{folder}/status`
- [x] `GET /v2/documents/inspect?file_name=...` (same-name inspection surface)
- [x] `POST /v2/documents/upload` (job-based; returns `job_id`)
- [x] `POST /v2/documents/ingest-url` (job-based; returns `job_id`)
- [x] `DELETE /v2/documents/{folder}` (delete document folder + remove global index entry)
- [ ] Extend delete route to also purge related job records/artifacts outside the document folder (deferred hard-delete policy).
- [x] Add explicit upload mode contract: `reuse|reprocess|new_version|ignore` for same-name documents.
- [x] Add explicit ingest-url mode contract: `reuse|reprocess|new_version|ignore` for same-name documents.
- [x] Add explicit source-type support contract for ingestion: direct file upload and URL/link ingestion.
- [x] Routing safety: register static routes (`/inspect`, `/upload`, `/ingest-url`) before parameterized `/{folder}` routes to avoid path-capture conflicts.
- [x] Add code comment in `api/routers/documents.py` documenting the static-before-parameterized ordering rule.

### Summary routes
- [x] `POST /v2/documents/{folder}/summaries/start` (job-based)
- [x] `POST /v2/documents/{folder}/summaries/reset` (job-based)
- [x] `POST /v2/summaries/watcher/start`
- [x] `GET /v2/documents/{folder}/summaries/status`
- [x] `GET /v2/documents/{folder}/summaries` (all levels payload + readiness/status)
- [x] `GET /v2/documents/{folder}/summaries/{level}` (`level1|level2|level3`)
- [x] Define resume-from-failure behavior for summaries (`resume=true` default, optional force restart).

### Chat routes
- [x] `POST /v2/chat/single`
- [x] `POST /v2/chat/multi` (request includes explicit `document_folders: list[str]`)
- [x] `POST /v2/chat/find-relevant`

### Job routes
- [x] Remove generic `POST /v2/jobs/{type}` (do not model command creation as a top-level job resource).
- [x] `GET /v2/jobs/{job_id}`
- [x] `GET /v2/jobs` (supports status/type filters)

## D. Async Jobs and Execution Model

- [x] Implement `core/job_store.py` (durable file-backed store under `artifacts/jobs/`).
- [x] Define job states: `pending`, `running`, `success`, `error`.
- [x] Defer `cancelled` state until cooperative cancellation is implemented in workers.
- [x] Implement worker execution for long-running tasks.
- [x] Persist error payload + retryability metadata in failed jobs.
- [x] Add idempotency keys for write endpoints.
- [x] Add per-job correlation ID and propagate to logs.
- [x] Define restart/recovery behavior after process crash (resume pending/running jobs safely on startup).

## E. Concurrency and Locking (Hard Gate)

- [x] Implement `core/locks.py` cross-process locks.
- [x] Add per-document ingest lock.
- [x] Add global index lock usage where required.
- [x] Add job-store write/read lock.
- [x] Prevent duplicate summary runs for same document.
- [ ] Stress test concurrent upload/summary/chat scenarios.
- [ ] Gate: do not enable async write endpoints until all above items pass.

## F. Error Handling and Security

- [x] Standardize error envelope across all routes.
- [x] Map workflow exceptions to stable error codes.
- [x] Add auth middleware (required for non-local exposure).
- [x] Add rate-limiting middleware.
- [x] Add request-size limit for uploads.
- [x] Add safe input validation for `url`, `document_folder`, and query payloads.

## G. Observability

- [x] Add structured request logs: `request_id`, `path`, `method`, `status_code`, `duration_ms`.
- [x] Keep workflow logs consistent: `workflow`, `stage`, `document_id`, `status`, `duration_ms`.
- [x] Add job lifecycle logs: `job_id`, `job_type`, `state`, `retry_count`.
- [x] Add health endpoint readiness checks (config + storage + vector backend reachability).

## H. Testing and Release Gates

- [x] Unit tests for schema validation and response mapping.
- [x] Router tests for each endpoint (success + error paths).
- [ ] Concurrency tests (multi-thread + multi-process).
- [x] Contract tests for error envelope consistency.
- [x] Regression tests for UI compatibility while API layer is enabled.
- [x] Runbook: deployment, rollback, and failure triage.

## Definition of Done

- [x] API endpoints are stable and typed.
- [ ] No business logic duplication outside `workflows.py`.
- [x] Async long-running operations are job-based and trackable.
- [x] Security controls (auth + rate limit + size limits) are active.
- [ ] Concurrency stress tests pass without corruption.
- [ ] Observability supports root-cause analysis in production.
