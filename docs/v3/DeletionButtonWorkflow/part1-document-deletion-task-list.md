# V3 Part 1: Document Deletion Task List

## Delivery Strategy

Implement this feature in vertical slices, but keep deletion logic centralized from the start.

## Implementation Status

- `Phase 1`: Done
- `Phase 2`: Done
- `Phase 3`: Done
- `Phase 4`: Done
- `Phase 5`: Done
- `Phase 6`: Done
- `Phase 7`: Done

## Task List

### Phase 1: Shared deletion contract

Status: Done on 2026-06-09

Completed in this phase:

- added a shared workflow entry point in `workflows.delete_documents(...)`,
- added workflow-layer deletion exceptions so non-HTTP callers do not depend on `HTTPException`,
- defined delete response fields in `api/schemas/documents.py`,
- added `DeleteDocumentsRequest` and `DeleteDocumentsResponse` schema models,
- refactored `DELETE /documents/{folder}` to delegate to the workflow,
- locked in the caller model implicitly by implementing the workflow for direct reuse from Streamlit later,
- hardened folder validation so invalid folder paths now map to workflow validation errors rather than not-found errors,
- removed duplicate router-side folder validation so API and non-HTTP callers share one validation path,
- improved conflict messages to include the blocking folder name and job type,
- added a temporary active-job guard for overlapping `documents.*` jobs using current metadata and job payload matching,
- guarded metadata loading so missing or unreadable metadata does not block deletion,
- kept global-index removal ahead of filesystem deletion and added explicit logging around both steps.

- Add a deletion result shape in `workflows.py` or a dedicated helper module.
- Define the structured delete response fields up front:
  - `deleted`
  - `folders`
  - `collections_cleaned`
  - `global_index_removed`
- Decide the canonical workflow entry point for document deletion.
- Add request/response schemas in `api/schemas/documents.py` for batch deletion.
- Refactor the existing single-document route so it no longer deletes files directly.
- Lock in the caller model: Streamlit uses the shared workflow directly; the API remains for external clients.

Definition of done:

- there is one shared deletion workflow,
- router code delegates to it,
- deletion returns structured data instead of only a boolean,
- there is no ambiguity about whether the UI talks to the workflow or the API.

### Phase 2: Chroma cleanup helper

Status: Done on 2026-06-10

Completed in this phase:

- verified the installed Chroma version supports `collection.delete(where=...)`,
- implemented the helper in `services/document_delete.py`,
- added shared Chroma client creation for both persistent and HTTP-backed deployments,
- targeted `vector_db.collection_name`, `vector_db.summary_collection_name`, and `vector_db.card_collection_name`,
- implemented primary cleanup by `document_folder` with optional fallback lookup by `document_id`,
- implemented a fallback from where-delete to id-based delete if the server rejects `delete(where=...)`,
- fixed workflow handoff so vector cleanup uses the indexed `document_folder` value instead of the resolved absolute path,
- broadened missing-collection handling so older Chroma `ValueError` behavior does not block deletion,
- added warning logs when where-delete fails and cleanup falls back to ID-based deletion,
- cached where-delete capability detection so the probe runs at most once per process,
- returned per-collection cleanup details from the helper for logging and future tests,
- integrated vector cleanup into `workflows.delete_documents(...)` so `collections_cleaned` is now populated.

- Verify the installed Chroma version supports `collection.delete(where=...)` without ids.
- Implement the helper in `services/document_delete.py`.
- Add a helper that opens the configured Chroma client from app config.
- Target `vector_db.collection_name`, `vector_db.summary_collection_name`, and `vector_db.card_collection_name`.
- Delete records by `document_folder`, with `document_id` available as a compatibility fallback.
- If where-only delete is unavailable, implement a query-then-delete-by-ids fallback.
- Return per-collection cleanup details for debugging and tests.

Definition of done:

- all known collection families are covered,
- cleanup code is isolated from UI and router layers,
- failures produce clear exceptions,
- the helper does not assume unsupported Chroma delete semantics.

### Phase 3: Workflow orchestration

Status: Done on 2026-06-10

Completed in this phase:

- validated every requested folder inside the workflow layer against the artifacts root,
- kept folder resolution in workflow code rather than reusing `HTTPException`-raising API helpers,
- loaded metadata for each target before mutation begins,
- rejected delete requests when overlapping `summaries.*` jobs target the same folder,
- rejected delete requests when overlapping `documents.*` jobs match target metadata by file name or source URL,
- ran Chroma cleanup, global index cleanup, and filesystem deletion in a fixed order,
- reused `delete_global_index_entry()` for global index cleanup,
- emitted structured logs for each folder and cleanup stage before aborting on failure,
- returned a stable delete payload including deleted folders and per-folder cleanup details,
- aligned the API response schema so `details` is surfaced to API callers as well as direct workflow callers.

- Validate that every requested folder exists inside the artifacts root.
- Do not reuse `require_existing_document_folder()` from `api/deps.py` in the workflow layer because it raises `HTTPException`.
- Add or use a non-HTTP folder-resolution utility suitable for both Streamlit and API callers, then let the router translate workflow errors to HTTP responses.
- Load metadata for each target before mutation begins.
- Reject the request if any target has a pending or running overlapping job.
- Check `summaries.*` jobs by folder payload.
- Check `documents.*` jobs more broadly using job type and payload fields such as file name or source URL matched against target metadata.
- Run Chroma cleanup, global index cleanup, and filesystem deletion in a fixed order.
- Reuse `delete_global_index_entry()` from `core/global_index.py` for global index cleanup.
- Emit structured logs for each folder and collection cleanup attempt before aborting on failure.
- Return a stable result payload including deleted folders and cleanup details.

Definition of done:

- deletion order is explicit and deterministic,
- batch requests behave all-or-nothing,
- no direct router-owned deletion logic remains,
- active upload/ingest overlap is blocked safely.

### Phase 4: API surface

Status: Done on 2026-06-10

Completed in this phase:

- kept `DELETE /documents/{folder}` for single-delete compatibility,
- added `POST /documents/delete` for batch deletion,
- ensured both delete routes share the same workflow and HTTP error semantics through one router helper,
- kept active-job conflicts mapped to `409`,
- validated individual `folders` items at the request-schema layer so blank folder names fail consistently before reaching the workflow,
- updated the route-ordering note in `api/routers/documents.py` to include `/documents/delete`,
- did not add idempotency-key support for delete endpoints.

- Keep `DELETE /documents/{folder}` for single-delete compatibility.
- Add a batch delete endpoint such as `POST /documents/delete`.
- Ensure both routes share the same workflow and error semantics.
- Return `409` when active jobs block deletion.
- Do not add idempotency-key support for delete endpoints.

Definition of done:

- single and batch delete share one backend path,
- API semantics are consistent across both entry points,
- delete behavior is not modeled as an idempotent replay workflow.

### Phase 5: Streamlit UX

Status: Done on 2026-06-10

Completed in this phase:

- added a `Delete Selected` action in the document pane near the selection controls,
- enabled the delete action only when at least one document is selected,
- added a confirmation dialog using `@st.dialog`,
- showed the selected document names and folder names together with an irreversible-action warning,
- normalized selected document paths to folder basenames before calling the shared delete workflow,
- normalized post-delete UI cleanup so basename comparisons are applied consistently across session state,
- cleared deleted session state and reran on success,
- preserved selection and showed a clear error when deletion fails.

Definition of done:

- the feature is discoverable,
- confirmation is required,
- UI state stays consistent after success or failure.

### Phase 6: State cleanup in UI

Status: Done on 2026-06-10

Completed in this phase:

- removed deleted folders from `selected_document_folders`,
- cleared `doc_chk_*` keys for deleted documents,
- cleared summary-pane active document state when it referenced a deleted folder,
- exited deep-search state when deleted documents invalidated the active deep-search selection,
- cleared routed-chat candidate state and reset chat context when deleted documents invalidated the current selection,
- removed deleted documents from active summarization tracking.

Definition of done:

- no deleted document remains selected,
- no deleted document remains active in chat or summaries.

### Phase 7: Tests

Status: Done on 2026-06-10

Completed in this phase:

- added `tests/test_document_deletion.py` using the standard library `unittest` runner already available in the repo environment,
- added workflow-level coverage for successful multi-delete,
- added workflow-level coverage for active-job rejection,
- added assertions that global index entries are removed during successful delete,
- added coverage that delete cleanup uses the indexed `document_folder` value for Chroma cleanup handoff,
- added API-level coverage for `POST /documents/delete`, `DELETE /documents/{folder}`, and `409` conflict mapping,
- kept UI-adjacent delete verification manual because the repo does not already have a UI test harness pattern for Streamlit.

Definition of done:

- the core lifecycle is covered by automated tests,
- destructive behavior is verified at the workflow/API level.

## Test Scenarios

### Success cases

- delete one selected document,
- delete multiple selected documents,
- delete a document with chunk, image, table, card, and summary vectors,
- delete a document that is ready to chat and listed in the global index.

### Failure cases

- attempt delete with zero selected documents from the UI,
- attempt delete for a missing folder,
- attempt delete while a summary or ingest job is active,
- attempt delete when Chroma cleanup raises an exception,
- attempt delete when one requested folder is invalid inside a batch request.

## Manual Verification Checklist

- ingest two or more documents,
- select one document and confirm delete,
- select multiple documents and confirm delete,
- verify removed folders no longer exist under `artifacts`,
- verify deleted documents no longer appear in the Streamlit list,
- verify deleted documents no longer participate in retrieval,
- verify cancel in the confirmation dialog leaves everything unchanged.

## Open Questions To Resolve During Implementation

- whether Chroma delete result counts are available reliably enough to expose in the response,
- whether older indexed records exist that require `document_id` fallback cleanup beyond `document_folder`.

## Recommended File Touch Order

1. `api/schemas/documents.py`
2. new deletion helper module
3. `workflows.py`
4. `api/routers/documents.py`
5. `ui.py`
6. `tests/...`

## Out of Scope

- soft delete,
- restore,
- trash history,
- background deletion jobs,
- partial-success batch deletion,
- per-row inline delete icons,
- orphan cleanup for unrelated historic data.
