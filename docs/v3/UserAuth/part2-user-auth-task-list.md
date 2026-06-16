# V3 Part 2: User Auth & Per-User Isolation — Task List

## Delivery Strategy

Implement in vertical slices. Auth primitives first (standalone, unit-testable), then API surface, then storage partitioning, then Chroma, then UI. No phase depends on a later phase.

## Implementation Status

- `Phase 1`: Done
- `Phase 2`: Done
- `Phase 3`: Done
- `Phase 4`: Done
- `Phase 5`: Done
- `Phase 6`: Done
- `Phase 7`: Done

## Task List

### Phase 1: User store & security primitives

Status: Done on 2026-06-10

Completed in this phase:

- added `argon2-cffi` and `PyJWT` to `requirements.txt`,
- created `core/user_store.py` with a WAL-mode SQLite store at `artifacts/.users/users.db`,
- implemented `create_user`, `verify_user`, `get_user_by_email`, `get_security_question`, `verify_security_answer`, `update_password`,
- passwords and security answers stored only as argon2id hashes; answers normalized before hashing,
- `user_key` computed once at signup as `<slug>__<sha256(email_normalized)[:8]>`,
- connections explicitly closed per operation (`contextlib.closing`) to avoid file-handle leaks on Windows,
- created `api/security.py` with HS256 1-hour tokens, claims `sub`/`email`/`user_key`/`jti`/`iat`/`exp`,
- in-memory revocation: per-token (`revoke_jti`) and per-user issued-before-timestamp (`revoke_user`),
- JWT secret sourced from `PROFRAG_JWT_SECRET` via a `validate_jwt_secret()` helper (min 32 chars) — note: the app does not call it yet; wiring into `api/app.py` startup and replacing the legacy static-token middleware happen in Phase 2,
- deterministic decoy security question generator for unknown emails,
- verified with a 25-check smoke test (all passed), script removed after run.

Post-review fixes (2026-06-10):

- closed the one-second revocation hole: `decode_token` now rejects tokens with `iat <= revoked_at` (fail closed; a token minted in the same second as a reset/logout is rejected),
- `decode_token` now requires `sub`, `email`, `user_key`, `jti`, `iat`, `exp` to be present and non-empty — a signed token missing auth claims is rejected,
- corrected this log: `validate_jwt_secret()` exists but is not yet called at app startup; that wiring is Phase 2 scope.

- Add `argon2-cffi` and `PyJWT` to requirements.
- Create `core/user_store.py`:
  - SQLite store at `artifacts/.users/users.db`, opened with `PRAGMA journal_mode=WAL`.
  - `users` table: `id`, `email`, `email_normalized`, `user_key`, `password_hash`, `security_question`, `security_answer_hash`, `is_active`, `created_at`; unique index on `email_normalized`.
  - `create_user(...)` — normalize email, compute `user_key` as `<slug>__<sha256(email_normalized)[:8]>`, argon2id-hash password and normalized security answer.
  - `verify_user(email, password) -> UserRecord | None`.
  - `get_security_question(email) -> str | None`.
  - `verify_security_answer(email, answer) -> bool` (normalize answer before check).
  - `update_password(email, new_password)`.
- Create `api/security.py`:
  - `create_access_token(user) -> str` — HS256, 1-hour expiry, claims `sub`, `email`, `user_key`, `jti`, `iat`, `exp`.
  - `decode_token(token) -> dict` — raises on expired/invalid/denylisted.
  - `revoke_jti(jti)` and `revoke_user(sub)` — in-memory denylist.
  - JWT secret from `PROFRAG_JWT_SECRET`; module-level validation helper for startup.
  - Decoy security question generator: deterministic question seeded from `sha256(email)` for unknown emails.

Definition of done:

- user store and token module are importable and unit-testable with no FastAPI or Streamlit imports,
- passwords and answers are only ever stored as argon2id hashes,
- two different emails always produce distinct `user_key` values.

### Phase 2: Auth API surface

Status: Done on 2026-06-10

Completed in this phase:

- created `api/schemas/auth.py` with register/login/me/reset request and response models (password min length enforced at schema layer),
- created `api/routers/auth.py` with all six endpoints: register, login, me, logout, reset/start, reset/finish,
- generic non-enumerating errors: duplicate registration → "Could not create account.", login failure → "Invalid email or password", reset failure → "Could not reset password.",
- unknown emails in reset/start receive a deterministic decoy question,
- login rate limit 5/60 s per IP; reset/finish rate limit 5/hour per email AND per IP (both attempts recorded, no short-circuit),
- successful reset revokes all of the user's outstanding tokens via `revoke_user`,
- rewrote `AuthMiddleware`: JWT validation, populates `request.state.user_id`/`user_email`/`user_key`/`token_jti`; skips only health, register, login, and reset routes — `/me` and `/logout` require a token,
- removed the static `PROFRAG_API_TOKEN` check and the `auth_token` setting entirely,
- `api/app.py` registers the auth router and calls `validate_jwt_secret()` at startup (the gap flagged in the Phase 1 review is now closed),
- added `UserContext`, `get_current_user`, and `get_user_artifacts_root` (mkdir-on-first-use) to `api/deps.py`,
- updated `tests/test_document_deletion.py` to authenticate with a real JWT — all 5 existing tests still pass,
- verified with a 21-check smoke test (register/login/me/logout/reset lifecycle, decoy stability, both rate limits, public health, startup secret rejection) — all passed, script removed after run.

- Create `api/routers/auth.py`:
  - `POST /v2/auth/register` — generic 400 on duplicate/invalid; min 8-char password.
  - `POST /v2/auth/login` — `Invalid email or password` on any failure; 5 attempts / 60 s per IP → 429.
  - `GET /v2/auth/me` — returns `{email, user_key}` from the validated token.
  - `POST /v2/auth/logout` — denylist the token's `jti`, return 204.
  - `POST /v2/auth/reset/start` — always returns a question (real or decoy).
  - `POST /v2/auth/reset/finish` — verify answer, set new password, revoke user's outstanding tokens; 5 attempts / hour per email and per IP → 429.
- Rewrite `AuthMiddleware` in `api/middleware.py`:
  - validate JWT, set `request.state.user_id` / `user_key` / `email`,
  - skip `/v2/health`, `/v2/auth/register`, `/v2/auth/login`, `/v2/auth/reset/*`,
  - remove the static `PROFRAG_API_TOKEN` check and the `auth_token` setting.
- `api/app.py`: register auth router; startup check that `PROFRAG_JWT_SECRET` exists and is ≥ 32 chars.
- `api/deps.py`: add `UserContext` dataclass, `get_current_user(request)`, `get_user_artifacts_root(user)` (mkdir-on-first-use).

Definition of done:

- every non-auth, non-health route returns 401 without a valid token,
- login failures and registration failures are non-enumerating,
- reset flow never reveals whether an email exists,
- rate limits return 429 with no information leakage.

### Phase 3: Per-user storage partitioning

Status: Done on 2026-06-10

Completed in this phase:

- added optional `artifacts_root` override parameter to `workflows.list_documents`, `workflows.delete_documents`, `workflows.inspect_same_name_document`, `workflows.prepare_document`, `workflows.prepare_url_document`, and `workflows._run_pipeline_from_metadata`; when provided it patches `config["paths"]["artifacts_root"]` in-place so `global_index_path(config)`, `find_document_folders`, and all sub-helpers automatically use the user-scoped root,
- updated `api/deps.py`: `resolve_document_folder` and `require_existing_document_folder` now accept an explicit `artifacts_root` parameter (defaulting to the base root for backward compatibility),
- rewrote `api/routers/documents.py`: all 14 endpoints now carry `artifacts_root: str = Depends(get_user_artifacts_root)`; `_run_job` and `_run_summary_job` thread functions receive `artifacts_root` as a positional argument captured at spawn time — no call to `get_artifacts_root()` occurs inside any background thread,
- rewrote `api/routers/summaries.py`: `_run_watcher_job` receives `artifacts_root` as a positional argument; `start_watcher` route carries the dep,
- rewrote `api/routers/chat.py`: all three endpoints carry the dep; `resolve_document_folder` calls pass `artifacts_root`,
- rewrote `api/routers/jobs.py`: both routes carry the dep; `list_jobs` and `get_job` calls pass the user-scoped root,
- updated `tests/test_document_deletion.py`: route-level tests use `app.dependency_overrides[get_user_artifacts_root]` to pin the path to the temp dir; fake workflow functions updated to accept the 3-arg signature — all 5 existing tests pass.

- Change effective root to `artifacts/<user_key>/` everywhere a user-owned path is resolved:
  - `resolve_document_folder` / `require_existing_document_folder` accept explicit `artifacts_root`,
  - all router endpoints take `artifacts_root` via `Depends(get_user_artifacts_root)`,
  - background threads (`_run_job`, `_run_summary_job` in `api/routers/documents.py` and `summaries.py`) receive the user-scoped `artifacts_root` as a closure argument captured at spawn time — never resolved inside the thread.
- Thread user-scoped `artifacts_root` through workflow entry points: list/load/inspect, prepare document, URL ingest, summaries start/reset, delete, chat, multi-doc retrieval.
- Per-user global index: `artifacts/<user_key>/metadata.json` (no code change in `global_index_path` — it already derives from the root it is given).
- Per-user jobs: job create/list/get/recover and delete-overlap checks operate on `artifacts/<user_key>/jobs/`.
- Per-user locks under `artifacts/<user_key>/.locks/`.
- Same-name detection and versioning scoped to the user subtree.
- Startup job recovery (`recover_inflight_jobs`) iterates each user directory under the base root.

Definition of done:

- no workflow code path resolves the base artifacts root for user-owned data,
- two users can hold same-named documents with independent version counters,
- a folder name belonging to user A is a 404 for user B,
- job overlap checks never see another user's jobs.

### Phase 4: Chroma partitioning

Status: Done on 2026-06-11

Completed in this phase:

- created `core/collections.py` with `user_collection_names(config, user_key)` returning `{chunks, summaries, cards}` in the `<base>__u__<user_key>` format, and `patch_config_for_user(config, user_key)` which shallow-copies `config["vector_db"]` with scoped collection names and a per-user `persist_directory` (`artifacts/<user_key>/chroma_db/` in PersistentClient mode; HttpClient mode is unaffected),
- added `get_user_key` FastAPI dependency to `api/deps.py` (returns `user.user_key` from `get_current_user`),
- threaded `user_key: str | None = None` through nine `workflows.py` functions: `_run_pipeline_from_metadata`, `prepare_document`, `prepare_url_document`, `delete_documents`, `find_relevant_documents`, `ask_question`, `ask_multi_document_question`, `start_summary_watcher`, `start_summarization_background`, and `reset_summary_level`; each applies `patch_config_for_user` after the existing Phase 3 `artifacts_root` patch so all downstream service calls use the user-scoped collection names and persist dir,
- added `user_key: str = Depends(get_user_key)` to all Chroma-touching endpoints in `api/routers/documents.py` (upload, ingest-url, delete, summaries/start, summaries/reset, DELETE /{folder}), `api/routers/chat.py` (single, multi, find-relevant), and `api/routers/summaries.py` (watcher/start); background thread tuples capture `user_key` at spawn time,
- updated `tests/test_document_deletion.py`: both route-level fake functions accept `*, user_key: str | None = None`; all 5 existing tests pass.

- Add a single resolver (e.g. in `core/config.py` or a new `core/collections.py`):
  `user_collection_names(config, user_key) -> {"chunks": ..., "summaries": ..., "cards": ...}` using the `<family>__u__<user_key>` format.
- Update every collection-name consumer to use the resolver:
  - `services/write_to_vector_db.py`
  - `services/metadata_cards.py`
  - `services/summarize_document.py`
  - `core/tree_retrieval.py`
  - `services/document_delete.py` (`_target_collection_names` gains `user_key`)
- Per-user persist directory `artifacts/<user_key>/chroma_db/` in PersistentClient mode; shared server with per-user collection names in HttpClient mode.
- `workflows.delete_documents(...)` and other Chroma-touching workflows gain a `user_key` parameter.

Definition of done:

- no code path opens a collection by raw config name,
- writes for two users land in different collections,
- retrieval and deletion for one user can never read or modify another user's collections.

### Phase 5: Streamlit UI

Status: Done on 2026-06-11

Completed in this phase:

- added an auth gate to `ui.py`: `main()` renders login/register/reset screens and returns before any pane when `st.session_state["auth_user_key"]` is absent — nothing in the app is reachable without login,
- login form verifies against `core.user_store.verify_user` (direct in-process call against the base root's `.users/users.db`); on success stores `auth_email`, `auth_user_key`, and `auth_artifacts_root` (= `<base>/<user_key>`, mkdir-on-login) in session state,
- register form: email, password (with static min-length hint), security question dropdown from `user_store.SECURITY_QUESTIONS`, answer; all failures show generic "Could not create account.",
- reset form (two steps): email → question shown (real, or decoy via `api.security.decoy_question_for` so UI and API show the same decoy for the same unknown email) → answer + new password; all failures show generic "Could not reset password."; login failure shows generic "Invalid email or password",
- logout button (top row): `st.session_state.clear()` + rerun — a different user's login switches the visible corpus completely with no leaked chat/selection/summary state,
- extended the `main.py` shim: all 12 functions accept and forward `artifacts_root` / `user_key` to the workflow layer,
- threaded `_user_root()` / `_user_key()` from session context into every `pipeline.*` call in `ui.py` (list, inspect, upload ×2, URL ingest, delete, summarize start/reset, watcher snapshot, single/multi/routing chat),
- made `services/summary_watcher.py` multi-user aware: `_scan_targets` scans the base root AND every per-user subtree (subdir without its own `metadata.json`, name = `user_key`) with a per-user patched config (`artifacts_root` + `__u__` collections + per-user persist dir), so the single process-wide watcher serves all users — the UI starts it once with the base config; verified with a filesystem smoke test (legacy doc keeps base config, user doc gets scoped config, `.users`/`chroma_db` ignored),
- all 5 existing tests still pass; `ui`, `main`, and `services.summary_watcher` import cleanly.

Post-review fixes (2026-06-11):

- High — tenancy is now enforced at the workflow layer, not just the UI: `workflows._require_folder_in_root` rejects any `document_folder` outside the caller's `artifacts_root` (raises `FileNotFoundError` → 404, non-enumerating); applied in `load_document`, `ask_question`, `ask_multi_document_question`, `start_summarization_background`, and `reset_summary_level` whenever an `artifacts_root` is supplied; `core/tree_retrieval.retrieve_tree` additionally validates every selected folder against the configured (per-user-patched) root before the metadata-rehydration path runs, so forged or stale absolute paths fail even if a future caller bypasses the workflow checks; `main.py` and all eight `ui.py` `load_document` call sites pass the session root,
- Medium — watcher user-root detection is now strict: `core/collections.iter_user_roots` requires the `<slug>__<8-hex>` user_key naming format, excludes infra dirs (`jobs`, `chroma_db`), dot-dirs, and document folders; `summary_watcher._scan_targets` uses it instead of the permissive "any non-dot dir without metadata.json" rule.

- Login screen gating the entire app: email + password; on success store `UserContext` in `st.session_state`, rerun.
- Register screen: email, password, security question dropdown (fixed list), answer.
- Reset screen: email → question shown → answer + new password.
- All UI → workflow calls pass the user-scoped `artifacts_root` (and `user_key`) from session context.
- Document list, selection, chat, summaries, and delete operate only on the logged-in user's corpus.
- Logout button: clear auth keys from session state, rerun.
- Generic error messaging on all auth screens (mirror API non-enumeration behavior).

Definition of done:

- nothing in the app is reachable without login,
- a logged-in user sees only their own documents in every pane,
- logout followed by a different user's login switches the visible corpus completely.

### Phase 6: Concurrency hardening & migration

Status: Done on 2026-06-11

Completed in this phase:

- WAL mode verified on the user store — codified as `tests/test_auth.py::UserStoreTests::test_wal_mode_active` (asserts `PRAGMA journal_mode` is `wal` on the created db),
- thread-spawn audit: grep for `get_artifacts_root()` across the codebase confirms zero calls inside thread targets — remaining call sites are `api/app.py` startup, `api/deps.py` itself, the auth router (base root for the user store — correct by design), and the health router,
- documented the single-worker constraint (`uvicorn --workers 1`) and the scale-out path (Chroma HttpClient, Redis for denylist/rate limits, watcher as a separate single-instance process, shared job queue) in a new "v3 Multi-User Deployment" section of `README.md`,
- wrote `scripts/migrate_to_multiuser.py`: creates or reuses the owner account (interactive prompts or CLI flags), moves document folders / global index / `jobs/` / `chroma_db/` under `artifacts/<owner_key>/`, rewrites stored path strings inside each moved doc's `metadata.json` and the global index, renames Chroma collections to the `__u__` format, and is idempotent (re-run skips moved items, reuses the account, skips renamed collections; the index-merge branch rewrites only source entries so the mapping is never double-applied),
- verified end-to-end with a two-run smoke test on a synthetic single-tenant tree (doc folder, global index, jobs, seeded Chroma collection): run 1 migrated and renamed everything, run 2 was a clean no-op; Chroma record metadata (`document_folder` strings) is intentionally not rewritten — deletion/retrieval fall back to `document_id`.

Post-review fixes (2026-06-11):

- High — startup job recovery no longer creates junk `jobs/` trees under non-user directories: `api/app.py` startup uses `core/collections.iter_user_roots` (strict user_key-format detection, infra dirs and document folders excluded) instead of recovering every non-dot child of the base root; covered by `tests/test_api_isolation.py::test_startup_recovery_skips_infra_and_doc_dirs` (asserts no `jobs/` under `chroma_db/`, a legacy doc folder, or a non-user dir, while the base root and genuine user roots are recovered).

- Verify WAL mode is active on the user store connection.
- Verify all thread spawn sites capture `artifacts_root` at spawn time (grep for `get_artifacts_root()` inside thread targets — must be zero).
- Document the single-worker constraint (`--workers 1`) and the scale-out path (Chroma HttpClient + Redis) in the plan/README.
- Write `scripts/migrate_to_multiuser.py`:
  - create the owner account interactively,
  - move document folders, `metadata.json`, `jobs/`, `chroma_db/` under `artifacts/<owner_key>/`,
  - rename Chroma collections to the `__u__` format,
  - idempotent on re-run.

Definition of done:

- migration converts an existing single-tenant install in one run and is safe to re-run,
- no thread resolves user paths after request scope ends,
- concurrency constraints are documented.

### Phase 7: Tests

Status: Done on 2026-06-11

Completed in this phase (47 tests + 8 subtests, all passing):

- `tests/test_auth.py` (22): user store — create/distinct user_keys, duplicate email, short password, verify success/failure, normalized security answers, password update, argon2-only storage, WAL mode; tokens — create/decode roundtrip, expired, missing claims, denylisted jti, `revoke_user` iat fail-closed boundary, deterministic decoy; routes — full register→login→me→logout lifecycle with revoked-token 401, generic login/register/reset errors, decoy for unknown email, reset revokes outstanding tokens, login rate-limit 429 on the 6th attempt,
- `tests/test_user_isolation.py` (7): `list_documents` scoped per root; same filename → independent version counters per user; `inspect_same_name_document` scoped; `delete_documents` cannot reach another user's folder (404-class error) nor traverse across roots (`../<other>/...` rejected); job-overlap checks scoped per user (A's active job doesn't block B; still blocks A); deletion updates only the owner's global index,
- `tests/test_chroma_partition.py` (7): `user_collection_names` format; distinct names for two users; `patch_config_for_user` scopes collections + persist dir (original config untouched); HttpClient mode keeps shared persist dir with name-level isolation; empty user_key no-op; `_target_collection_names` on a patched config targets only owner `__u__` collections; real `PersistentClient` proof that two users' writes land in physically separate stores,
- `tests/test_api_isolation.py` (6 + 8 subtests): 401 without token on every protected route (documents list/get/delete, jobs, chat single/multi/find-relevant, watcher start); document list scoped to the token's user via the real `get_user_artifacts_root` dependency chain; user A's folder is 404 for user B; jobs list/get scoped per token; 409 delete-conflict behavior intact within one user; cross-user delete → 404 with files untouched,
- existing `tests/test_document_deletion.py` (5) still passes with user scoping applied.

Post-review fixes (2026-06-11) — suite now 57 tests + 8 subtests, all passing:

- Medium — added the missing authenticated cross-user coverage: `/v2/chat/single` and `/v2/chat/multi` with another user's folder name → 404; forged paths (`../<user_a>/...` traversal and absolute paths into A's subtree) → 400; summaries start/status/reset for another user's folder → 404; startup-recovery scoping test (see Phase 6),
- workflow-boundary tenancy tests in `tests/test_user_isolation.py`: `load_document`, `ask_question`, `ask_multi_document_question` (foreign folder mixed with owned one rejects the whole call), and both summary entry points raise `FileNotFoundError` for folders outside the caller's root; `retrieve_tree` raises `ValueError` for out-of-root selected folders; watcher `_scan_targets` exclusion-rules test (ignores `jobs/`, `chroma_db/`, non-user-key dirs; user docs get scoped config, legacy docs keep base config).

- Auth unit tests: signup, duplicate email generic failure, login success/failure, expired token, denylisted token, reset with correct/wrong answer, decoy question for unknown email, rate-limit 429s.
- Workflow isolation tests: same filename for two users → independent versions; `list_documents` scoped; `delete_documents` cannot cross user roots; job overlap scoped per user.
- Chroma partition tests: two users' writes land in distinct collections; cleanup touches only the owner's collections.
- API tests: 401 without token on every protected route; documents/chat/summaries/jobs operate only within the token's user scope; existing 409 delete-conflict behavior intact within one user.
- Reuse the patterns in `tests/test_document_deletion.py` (tempdir artifacts root, mocked workflows, `TestClient`).

Definition of done:

- the auth lifecycle and both isolation boundaries (filesystem, Chroma) are covered by automated tests,
- existing document-deletion tests still pass with user scoping applied.

## Test Scenarios

### Success cases

- register → login → upload → chat → delete, all scoped to one user,
- two users upload same-named files simultaneously; both succeed independently,
- password reset with correct security answer; old sessions revoked,
- API client logs in, uses token for 59 minutes, gets 401 at 61 minutes.

### Failure cases

- login with wrong password → generic 401,
- register with existing email → generic 400,
- reset/start with unknown email → decoy question, no enumeration,
- 6th login attempt in 60 s → 429,
- user B requests user A's folder name → 404,
- request with revoked (logged-out) token → 401.

## Manual Verification Checklist

- register two users in two browser sessions,
- upload different docs as each; confirm neither sees the other's docs,
- run chat simultaneously in both sessions; answers cite only the owner's corpus,
- reset user A's password via security question; confirm user A's old API token is rejected,
- restart API server; confirm users and docs persist (denylist loss is expected),
- run migration script on a copy of an existing single-tenant `artifacts/`; verify the owner sees all legacy docs.

## Out of Scope

- OAuth / SSO,
- refresh tokens,
- email verification and SMTP-based reset,
- roles, permissions, document sharing,
- admin UI,
- multi-worker / multi-server deployment,
- persistent browser sessions across page refresh.
