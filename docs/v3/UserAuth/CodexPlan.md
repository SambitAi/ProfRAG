# User Auth and Per-User Document Isolation

## Summary
Add first-party authentication with email + password, self-signup, and 1-hour JWT access tokens. Partition all document storage, indexes, jobs, and Chroma collections by authenticated user so a user can only list/query/delete their own documents. Keep Streamlit calling shared workflows directly, but require explicit user context in the workflow layer so the same ownership rules apply to both UI and API.

Security flaws in the raw proposal that this plan fixes:
- No plaintext password storage; use `argon2id` password hashing with per-password salt.
- Do not use raw email directly as a folder path or collection name; derive a safe internal `user_key`.
- Do not treat “selected documents” as authorization; authorization is ownership, selection is only UI state.
- Do not rely only on Chroma metadata filters; resolve user-scoped collections before query/delete/upsert.
- Do not keep the current single global bearer token; replace it with per-user signed 1-hour JWTs.
- Add login throttling and generic auth failures to reduce brute-force and account enumeration.

## Key Changes

### 1. Identity and auth model
- Add a local SQLite auth store for v1, for example `artifacts/auth.db` or `archive/auth.db`.
- Add a `users` table with at least: `id`, `email`, `email_normalized`, `user_key`, `password_hash`, `is_active`, `created_at`.
- Email is the login identifier; `email_normalized` is lowercase trimmed email.
- `user_key` is a filesystem-safe, stable derived key, not the raw email. Use a deterministic slug-plus-hash form such as `alice_example_com__a1b2c3d4`.
- Passwords are stored only as `argon2id` hashes.
- Add self-signup and login endpoints:
  - `POST /v2/auth/register`
  - `POST /v2/auth/login`
  - `GET /v2/auth/me`
- Issue a signed JWT access token with 1-hour expiry. Use HS256 with a strong secret from environment for v1.
- JWT claims: `sub` = user database id, `email`, `user_key`, `exp`, `iat`.
- No refresh token in v1; expired sessions require re-login.

### 2. Authorization and request/user context
- Replace the current single static bearer-token middleware model with JWT validation middleware or dependency-based auth for protected routes.
- Add a `get_current_user()` dependency returning a typed user context with `id`, `email`, `user_key`.
- Keep `/v2/health` public.
- Require auth for all document, chat, summary, and job routes.
- Add login rate limiting separate from the general per-IP limiter, with stricter thresholds.
- Return generic auth errors:
  - registration: generic duplicate/invalid messaging
  - login: `Invalid email or password`
- Streamlit login stores the authenticated user context and access token in `st.session_state`; UI continues calling workflows directly, but every workflow entrypoint used by the UI must accept `current_user` or `user_key`.

### 3. Per-user storage and data partitioning
- Change the document storage root shape from shared `artifacts/<doc>` to per-user storage:
  - `archive/<user_key>/<document_folder>`
- Also partition per-user supporting files:
  - global index: `archive/<user_key>/metadata.json`
  - jobs: `archive/<user_key>/jobs/*.json`
  - locks remain under each user root or under a user-scoped lock path
- Do not keep a shared cross-user `list_documents()` view; every list/load/inspect/delete path resolves only inside the current user root.
- Same-name document reuse/versioning is checked only within the current user’s archive subtree.

### 4. Chroma partitioning
- Use per-user collection names for each collection family instead of one global collection:
  - chunks: `<base_chunks>__u__<user_key>`
  - summaries: `<base_summaries>__u__<user_key>`
  - cards: `<base_cards>__u__<user_key>`
- Centralize collection-name resolution in one helper so all write, query, retrieval, summary indexing, and delete code use the same user-scoped names.
- Keep `document_folder` metadata as the per-user archive path under that user root.
- All retrieval, deletion, and summary indexing logic must operate only on the current user’s collections.
- Do not rely on metadata filtering alone for tenant isolation.

### 5. Workflow, UI, and API changes
- Introduce a shared user-aware context/helper layer that resolves:
  - user archive root
  - user global index path
  - user jobs path
  - user-scoped Chroma collections
- Update workflow entrypoints used by UI and API to accept user context, especially:
  - list/load/inspect
  - prepare document / URL ingest
  - summaries start/reset
  - delete documents
  - chat and multi-doc retrieval
- Streamlit:
  - add login and signup screens
  - gate the document/chat UI behind authenticated session state
  - list only current user documents
  - keep current selection behavior, but selection is only from that user’s visible docs
- API:
  - document routes no longer infer shared roots from global config alone; they derive user root from authenticated user
  - folder path validation stays basename-based but resolves inside current user root only
- Jobs:
  - create/list/get/recover jobs in the current user’s `jobs` directory
  - job overlap checks only inspect the current user’s job store

## Tests
- Auth tests:
  - successful self-signup
  - duplicate email rejected
  - successful login returns 1-hour JWT
  - invalid password rejected with generic message
  - expired JWT rejected
- Workflow isolation tests:
  - two users with same document filename get independent versioning and separate archive roots
  - `list_documents()` returns only the current user’s docs
  - `inspect_same_name_document()` checks only current user scope
  - `delete_documents()` cannot delete another user’s folder name
- Chroma partition tests:
  - writes for two users go to different collection names
  - retrieval for one user never reads the other user’s collections
  - delete only cleans the current user’s collections
- API tests:
  - protected routes return `401` without token
  - `/v2/documents`, `/v2/documents/delete`, chat, and summaries operate only within the token user scope
  - `409` overlap logic remains intact within a user’s own jobs
- UI-adjacent manual checks:
  - signup/login/logout flow
  - one user cannot see another user’s documents
  - same browser session after logout/login switches visible corpus correctly
  - delete, summary, and chat still work after login

## Assumptions and defaults
- Use self-signup in v1.
- Use SQLite for user storage.
- Use JWT access tokens with 1-hour expiry and no refresh token.
- Keep Streamlit direct-to-workflow, not UI-via-API.
- Use email as login identity, but not as raw path/collection key.
- Rename the storage root to `archive` and move all document storage under `archive/<user_key>/...`.
- No password reset, email verification, sharing, roles, or admin UI in v1.
