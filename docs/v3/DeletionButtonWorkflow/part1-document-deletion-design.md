# V3 Part 1: Document Deletion Design

## Goal

Add a user-facing delete flow for one or more selected documents that:

1. shows a confirmation dialog before deletion,
2. deletes the corresponding document folders from `artifacts`,
3. removes all related vectors from ChromaDB,
4. keeps the UI, API, global metadata, and background jobs in a consistent state,
5. avoids patchwork logic by introducing one reusable deletion workflow.

## Why This Needs Design First

The current codebase already has pieces of the problem, but they are spread across multiple layers:

- the Streamlit UI already supports multi-selection,
- the API already has a single-document delete route,
- Chroma indexing happens in multiple collections,
- global metadata is stored separately from per-document metadata,
- background summary jobs can still be active while a document exists.

If we only add a button and extend the existing route inline, we will create a fragile feature that deletes files but leaves stale vectors, stale UI state, or job conflicts behind.

## Current State Analysis

### 1. UI selection is already multi-document capable

`ui.py` already maintains `selected_document_folders` in session state and updates it inside `render_existing_documents()`.

Current strengths:

- selection state already supports 0, 1, or many documents,
- "Select All" and "Deselect All" already exist,
- the document list is the right place to surface bulk actions.

Current gap:

- there is no delete action tied to the selected set,
- there is no confirmation dialog for destructive actions,
- no local state cleanup exists after documents are deleted.

Design implication:

- the v3 UI should reuse `selected_document_folders` as the single source of truth for what gets deleted,
- deletion should be exposed as a bulk action in the document pane, not as per-row ad hoc logic.

### 2. The API already has a delete route, but it is incomplete

`api/routers/documents.py` defines `DELETE /documents/{folder}`.

Current strengths:

- it validates the folder path through `require_existing_document_folder()`,
- it blocks deletion when an active job payload references the folder,
- it removes the document from the global index,
- it removes the artifact folder.

Current gaps:

- it only supports one folder at a time,
- it does not remove Chroma records,
- the deletion logic lives directly in the router,
- it returns a very thin payload with no cleanup breakdown.

Design implication:

- router code should stop owning deletion behavior,
- deletion should move to a shared workflow/service so both UI and API use the same rules,
- batch delete should be first-class instead of implemented as UI loops over a single-delete endpoint.

### 3. Chroma writes happen in multiple collections

The system indexes data into more than one Chroma collection:

- `services/write_to_vector_db.py` writes chunk, image, and table vectors to `vector_db.collection_name`,
- `services/summarize_document.py` writes summary vectors to `vector_db.summary_collection_name`,
- `services/metadata_cards.py` writes document and section card vectors to `vector_db.card_collection_name`.

Important observation:

- all of these writers attach `document_folder` metadata,
- most also attach `document_id`,
- this gives us a stable deletion filter strategy.

Current gap:

- there is no central utility for deleting vectors from all affected collections.

Design implication:

- deletion must clean at least three collections,
- cleanup should prefer metadata-based deletion by `document_folder`,
- `document_id` can be used as a fallback if needed for older records or future compatibility.

### Chroma schema snapshot

This is the current effective record shape based on the indexing code.

Important implementation note:

- there is currently no existing `collection.delete(...)` usage anywhere in the codebase,
- Chroma delete-by-filter will be net-new code for this repo,
- the implementation must verify the installed Chroma version supports `delete(where=...)` without requiring explicit ids,
- if the installed version does not support where-only deletion, the helper must fall back to a query-then-delete-by-ids strategy instead of assuming filter deletion is available.

#### Chunk collection

Source:

- `services/write_to_vector_db.py`

Collection:

- `vector_db.collection_name`

Record id pattern:

- text chunks: `{document_id}_{chunk_slug}`
- images: `{document_id}_img_{filename}`
- tables: `{document_id}_table_{table_stem}`

Chunk `documents` payload:

- the stored Chroma document is the raw chunk text

Chunk embedding input:

- raw chunk text,
- or `[{section_path}]\n{raw_text}` when a section path exists

Chunk metadata fields:

- `document_name`
- `document_id`
- `document_folder`
- `chunk_number`
- `chunk_path`
- `section_path`
- `section_level`
- `item_type = "text"`

Image metadata fields:

- `document_folder`
- `document_id`
- `item_type = "image"`
- `abs_path`
- `caption`
- `page`
- `section_path`

Table metadata fields:

- `document_folder`
- `document_id`
- `item_type = "table"`
- `abs_path`
- `section_path`

Deletion relevance:

- `document_folder` is present across text, image, and table entries,
- `document_id` is also present and can serve as a fallback selector.

#### Summary collection

Source:

- `services/summarize_document.py`

Collection:

- `vector_db.summary_collection_name`

Record id pattern:

- level 1: `{document_id}_level1`
- level 3 section summaries: `{document_id}_{section_slug}_level3`

Summary `documents` payload:

- the generated summary text

Summary metadata fields:

- `document_folder`
- `document_name`
- `section_name`
- `level`

Deletion relevance:

- `document_folder` is present,
- summary records currently do not store `document_id`,
- summary cleanup should therefore primarily target `document_folder`.

#### Card collection

Source:

- `services/metadata_cards.py`

Collection:

- `vector_db.card_collection_name`

Record id pattern:

- document card: `{document_id}_doc_card`
- section card: `{document_id}_{section_id}_card`

Card `documents` payload:

- document card embed text, or
- section card embed text

Card metadata fields:

- `document_id`
- `document_folder`
- `document_name`
- `card_type`
- `section_id`
- `section_path`

Deletion relevance:

- both `document_folder` and `document_id` are present,
- this collection is safe to target by either selector.

### 4. Global metadata and artifact folders are separate consistency targets

The app stores control-plane metadata in:

- `artifacts/metadata.json` via `core/global_index.py`,
- `{document_folder}/metadata.json` via `core/metadata.py`.

Current strengths:

- global index writes are already centralized,
- global index delete already exists.

Current gap:

- there is no transactional deletion boundary across:
  - job checks,
  - Chroma cleanup,
  - global index cleanup,
  - filesystem removal,
  - UI session cleanup.

Design implication:

- we need a single ordered deletion workflow with explicit cleanup steps and deterministic failure handling.

### 5. Background jobs introduce a concurrency risk

Summary generation runs in background threads and jobs are tracked under `artifacts/jobs`.

Current strengths:

- the current delete route already refuses deletion if a pending/running job references the folder.

Current gaps:

- only one route enforces this today,
- the current router check only looks at `payload["folder"]`, which catches summary jobs but not upload/ingest jobs,
- UI state does not yet know how to surface partial deletion failures,
- a batch delete can contain a mix of deletable and non-deletable documents.

Design implication:

- batch deletion should validate the entire requested set first,
- active-job detection must cover both summary jobs and `documents.*` jobs,
- the new workflow must inspect job type and payload more broadly than the current route implementation.
- the design should choose between:
  - all-or-nothing deletion, or
  - partial-success deletion with detailed reporting.

Recommended choice:

- use all-or-nothing for v3 part 1.

Reason:

- it is simpler to reason about,
- it prevents half-deleted selections,
- it keeps the first implementation safer and easier to test.

## Architectural Recommendation

### Introduce one shared deletion workflow

Add a new workflow-level entry point, for example:

- `workflows.delete_documents(config_path, folder_names)`

This workflow should own the full deletion lifecycle and return a structured result.

Suggested internal breakdown:

1. resolve and validate requested folders,
2. load metadata for each target,
3. check for active jobs for all targets,
4. delete vectors from all Chroma collections,
5. remove entries from the global index,
6. remove document folders from `artifacts`,
7. return a result payload for the caller.

Why this is the right boundary:

- `ui.py` stays focused on interaction,
- API routers stay thin,
- deletion rules become reusable and testable,
- future CLI/admin tools can call the same workflow.

### Streamlit should call the workflow directly

This is no longer an open question.

Recommended choice:

- the Streamlit UI should call the shared workflow directly, not the API surface.

Reason:

- that matches the current architecture, where `ui.py` already calls `workflows.*` directly,
- it avoids introducing an unnecessary HTTP round trip,
- it avoids creating API-only complexity for an internal same-process caller,
- it keeps UI behavior aligned with the rest of the codebase.

### Introduce a dedicated Chroma cleanup helper

Create a focused helper module:

- `services/document_delete.py`

Responsibilities:

- open the Chroma client from config,
- target all collections used by this repo,
- delete records filtered by `document_folder`,
- optionally support fallback delete by `document_id`,
- return per-collection cleanup counts or attempted status.

This separation keeps Chroma-specific behavior out of `workflows.py`.

### Keep the router thin

Refactor `api/routers/documents.py` so routes call the shared workflow instead of deleting directly.

Recommended API additions:

- keep `DELETE /documents/{folder}` for single-delete compatibility,
- add a batch delete endpoint such as `POST /documents/delete`.

Why not only loop over the single route:

- router-level loops duplicate validation and error handling,
- they make all-or-nothing behavior awkward,
- they encourage UI-driven orchestration instead of server-side orchestration.

Naming note:

- `POST /documents/delete` is slightly less pure than `DELETE /documents` with a body,
- but it is the more compatible choice across clients and intermediaries,
- so it is the recommended batch-delete endpoint for this repo.

## UI Design Recommendation

### Where the delete action should live

Put the delete action in the document pane near the selection controls:

- alongside or below "Select All" and "Deselect All",
- enabled only when one or more documents are selected.

Why:

- the action operates on the selected set,
- this matches the existing mental model,
- it avoids hidden per-document destructive controls.

### Confirmation dialog behavior

Use a Streamlit dialog, similar to the existing `@st.dialog("Add Document")` pattern.

Recommended dialog content:

- title: `Delete Documents`
- summary text: count of selected documents
- explicit list of selected folder names
- warning that this removes:
  - artifact folders,
  - Chroma vectors,
  - document availability in chat/search
- actions:
  - `Cancel`
  - `Delete`

### UI state after successful deletion

On success, the UI should:

- clear deleted items from `selected_document_folders`,
- clear corresponding `doc_chk_*` checkbox state,
- reset active summary pane selection if the active doc was deleted,
- exit routing/deep-search state if the selected set became invalid,
- rerun to refresh the document list.

### UI state after failed deletion

On failure, the UI should:

- preserve current selection,
- show a precise error message,
- avoid partial local state mutation.

## Proposed Deletion Semantics

### Request contract

Deletion should operate on folder names, not absolute paths.

Reason:

- folder names are already the API identity,
- path resolution and sandboxing stay server-owned,
- UI should not send absolute filesystem paths around.

### Order of operations

Recommended order:

1. validate all folders exist,
2. reject if any target has active jobs,
3. delete Chroma records,
4. delete global index entries,
5. delete artifact folders,
6. return success.

Why Chroma first:

- once the artifact folder is gone, fallback metadata may be harder to inspect,
- stale vectors are more harmful than stale files because they silently affect retrieval,
- folder metadata is still available during Chroma cleanup.

### Failure handling

For v3 part 1:

- if validation fails, delete nothing,
- if Chroma cleanup fails for any target, delete nothing further,
- if filesystem deletion fails after vector cleanup starts, surface a hard error and log enough detail for manual recovery.

Note:

- fully atomic cross-system deletion is not realistic here because filesystem and Chroma do not share a transaction boundary,
- the design should instead aim for deterministic ordering, structured reporting, and easy recovery.

Required logging behavior:

- log per target folder and per collection whether cleanup started, succeeded, or failed,
- include the collection name and deletion selector used,
- if fallback delete-by-id is used, log that path explicitly,
- if a partial Chroma cleanup occurs, emit enough detail to identify which collections were already cleaned for which document folders.

## Data and API Shapes

### Suggested request model

Add a schema to `api/schemas/documents.py`, for example:

- `DeleteDocumentsRequest`
  - `folders: list[str]`

### Suggested response model

Return a structured response such as:

- `deleted: true`
- `folders: [...]`
- `collections_cleaned: [...]`
- `global_index_removed: [...]`

This is better than a boolean-only response because it helps the UI and tests verify behavior.

Idempotency note:

- do not add idempotency-key handling to delete endpoints,
- delete is not replay-safe in the same way as upload or ingest,
- a second delete against an already-deleted folder should remain a normal not-found path rather than an idempotent job replay.

## Code Areas To Touch During Implementation

### UI

- `ui.py`

Expected work:

- add delete action in the document pane,
- add confirmation dialog,
- call shared deletion pathway,
- clear session state on success.

### API

- `api/routers/documents.py`
- `api/schemas/documents.py`

Expected work:

- add batch delete request/response models,
- refactor existing single-delete route to call shared workflow,
- optionally add batch delete endpoint.

### Workflow / domain logic

- `workflows.py`
- new deletion helper module

Expected work:

- centralize validation and deletion orchestration,
- centralize Chroma cleanup,
- standardize returned result payload.

### Tests

- `tests/e2e_internal_api.py`
- add focused deletion tests

Expected work:

- verify folder removal,
- verify global index removal,
- verify Chroma cleanup,
- verify active-job rejection,
- verify multi-delete behavior.

## Recommended Non-Goals For Part 1

To keep the first version clean, do not include:

- undo/restore,
- soft delete,
- trash bin,
- partial-success deletion in one request,
- per-row inline delete icons,
- deletion of orphaned Chroma records unrelated to explicit user-selected documents.

## Acceptance Criteria

This feature is complete when:

1. selecting one or more documents reveals an enabled delete action,
2. clicking delete opens a confirmation dialog,
3. confirming deletion removes the selected document folders from `artifacts`,
4. confirming deletion removes matching vectors from chunk, summary, and card collections,
5. deleted documents disappear from the UI after rerun,
6. chat/search cannot route to deleted documents,
7. active jobs block deletion safely,
8. deletion logic is implemented once and reused across callers.

## Recommended Implementation Principle

The core rule for v3 part 1 should be:

"Deletion is a document-lifecycle operation, not a UI trick and not a router shortcut."

If we keep that boundary, the implementation will stay scalable, maintainable, and consistent with the rest of the architecture.
