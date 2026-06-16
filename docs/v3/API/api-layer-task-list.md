# V3 API Layer Alignment - Task List

## Goal

Align the V3 API layer with the implemented multi-user/auth changes and close the remaining gaps between the API contract, Streamlit behavior, and workflow execution.

## Current Snapshot

- Auth endpoints are implemented and protected by JWT middleware.
- Documents, jobs, and health routers exist.
- The summaries router is only partially top-level: watcher start exists, summary retrieval is document-scoped.
- Chat API exists only as `single`, `multi`, and `find-relevant`.
- Streamlit chat still calls workflows directly instead of using the API layer.
- No explicit API endpoints exist for chat "plan" / "respond" flows.

## Implementation Status

- `Phase 1`: Pending
- `Phase 2`: Pending
- `Phase 3`: Pending
- `Phase 4`: Pending
- `Phase 5`: Pending

## Task List

### Phase 1: Freeze the intended V3 API contract

Status: Pending

- Confirm the public chat API shape for V3:
  - decide whether the contract is `POST /v2/chat/query`,
  - or `POST /v2/chat/plan` plus `POST /v2/chat/respond`,
  - or keep `single` / `multi` / `find-relevant` as the final external API.
- Confirm whether Streamlit is supposed to remain direct-to-workflow or become API-backed.
- Confirm whether a top-level summaries list endpoint is required under `/v2/summaries/`.
- Confirm whether `POST /v2/auth/refresh` is intentionally out of scope.
- Write the approved contract in a short design note in `docs/v3/API/`.

Definition of done:

- the team has one explicit V3 API contract,
- endpoint names and responsibilities are no longer inferred from UI behavior or older plan notes.

### Phase 2: Fill missing chat API capabilities

Status: Pending

- If V3 requires plan/respond chat execution, add explicit endpoints and schemas:
  - `POST /v2/chat/plan`
  - `POST /v2/chat/respond`
- Ensure every chat endpoint accepts authenticated user context and passes both `artifacts_root` and `user_key`.
- Reuse the existing containment and per-user Chroma scoping patterns already used by current chat endpoints.
- Keep error behavior non-enumerating and tenant-safe.
- If plan/respond are not required, document that `single` / `multi` / `find-relevant` are the complete chat API and update docs/UI wording accordingly.

Definition of done:

- all required V3 chat actions exist in the API layer,
- there is no ambiguity about how "plan" and "respond" are executed.

### Phase 3: Decide and implement the UI-to-API boundary

Status: Pending

- Choose one architecture and apply it consistently:
  - keep Streamlit direct-to-workflow and document API as external-client-only,
  - or move Streamlit chat/document actions to call the FastAPI layer.
- If moving to API-backed UI:
  - route login/logout/session-aware actions through the API,
  - route chat, document, summary, and job polling through `/v2/...`,
  - remove duplicated behavior between `ui.py` and router/workflow glue.
- If keeping direct-to-workflow UI:
  - document that choice clearly,
  - avoid describing UI chat actions as API-backed in V3 docs.

Definition of done:

- UI behavior matches the documented V3 architecture,
- "API layer" and "UI workflow path" are no longer divergent by accident.

### Phase 4: Clean up API contract leaks and partial endpoints

Status: Pending

- Remove filesystem path leakage from document summary responses:
  - stop returning absolute or internal `summary_paths`,
  - stop returning raw file `path` values in `/documents/{folder}/summaries`.
- Replace leaked paths with API-safe metadata:
  - `exists`,
  - `status`,
  - `level`,
  - optional stable logical identifiers if needed.
- Decide whether to add a dedicated top-level summaries listing endpoint under `/v2/summaries/`.
- Review all response payloads for internal-path exposure and implementation-detail leakage.

Definition of done:

- API responses are stable and client-safe,
- internal filesystem layout is not exposed through the HTTP contract.

### Phase 5: Verification and regression coverage

Status: Pending

- Add API tests for the final chat contract:
  - success paths for every supported chat action,
  - cross-user isolation,
  - invalid folder/path rejection,
  - tenant-safe errors.
- Add tests for any new summaries endpoint or response-shape cleanup.
- Add tests for UI/API integration if Streamlit is moved to API-backed execution.
- Add auth-session tests if `refresh` is introduced.
- Update V3 docs to match the tested API surface exactly.

Definition of done:

- automated tests cover the final V3 API contract,
- docs, implementation, and tests describe the same behavior.

## Confirmed Findings Driving This List

- `api/routers/chat.py` currently exposes only:
  - `POST /v2/chat/single`
  - `POST /v2/chat/multi`
  - `POST /v2/chat/find-relevant`
- `api/routers/summaries.py` currently exposes only:
  - `POST /v2/summaries/watcher/start`
- `api/routers/documents.py` contains document-scoped summary retrieval/status/start/reset endpoints.
- `ui.py` still calls workflow functions directly for chat execution.
- `api/routers/auth.py` has register/login/me/logout/reset, but no refresh endpoint.

## Out of Scope

- OAuth / SSO,
- admin user management,
- roles and sharing,
- multi-worker or distributed coordination redesign,
- broader product redesign outside the V3 API alignment work.
