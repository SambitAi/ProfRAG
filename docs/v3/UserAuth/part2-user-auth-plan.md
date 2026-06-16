# V3 Part 2: User Authentication & Per-User Isolation — Final Plan

## Status

Approved design decisions (2026-06-10):

| Decision | Choice |
|---|---|
| Password reset | Security question fallback (self-service, no SMTP) |
| Storage root | Keep `artifacts/`, layout becomes `artifacts/<user_key>/<docs>` |
| Account creation | Self-signup (register screen in UI) |
| UI architecture | Streamlit stays direct-to-workflow; JWT is for external API clients only |

This plan supersedes and merges `Claude_plan.md` and `CodexPlan.md` in this folder.

---

## Context

ProfRAG is single-tenant today: one static Bearer token, one shared `artifacts/` root, one shared ChromaDB collection set, one shared jobs directory. The goal is multi-user operation: email + password login, self-signup, security-question password reset, per-user document storage, per-user ChromaDB collections, and safe concurrent access by multiple users.

The existing architecture is `Router (thin) → workflows.py (orchestrator) → services/ (stateless helpers)`, with Streamlit calling workflows directly. Auth is added as a layer in front of this — the orchestrator receives explicit user context as plain arguments and never imports from `api/`.

**This is NOT OAuth.** It is first-party session auth: the UI verifies credentials against a local user store and keeps user context in `st.session_state`; external API clients log in once and receive a signed 1-hour JWT.

---

## Identity model

### User record (SQLite, `artifacts/.users/users.db`)

| Column | Purpose |
|---|---|
| `id` | Integer primary key |
| `email` | Login identifier as entered |
| `email_normalized` | Lowercased, trimmed — unique index, used for lookup |
| `user_key` | Filesystem-safe stable key: `<email_slug>__<sha256(email_normalized)[:8]>`, e.g. `alice_example_com__a1b2c3d4` |
| `password_hash` | argon2id hash (per-password salt is built into argon2) |
| `security_question` | Plaintext question chosen at signup (from a fixed list) |
| `security_answer_hash` | argon2id hash of the normalized (lowercased, trimmed) answer |
| `is_active` | Soft-disable flag |
| `created_at` | ISO timestamp |

- `user_key` is computed **once at signup** and stored. Never re-derived at runtime.
- The human-readable slug prefix makes `ls artifacts/` debuggable; the hash suffix guarantees uniqueness and handles emails that slugify identically.
- SQLite opened with `PRAGMA journal_mode=WAL` — concurrent logins never block on each other (one writer, unlimited readers).

### Password & answer hashing

- **argon2id** via the `argon2-cffi` package (OWASP-recommended, memory-hard).
- Security answers are normalized (lowercase, strip, collapse whitespace) before hashing so "Fluffy " and "fluffy" match.

---

## Auth flows

### Signup (self-service)
1. UI register screen: email, password, security question (picked from a fixed list of ~6), answer.
2. Password policy: minimum 8 characters (no composition rules — length beats complexity).
3. Duplicate email → generic error ("Could not create account") to prevent enumeration.
4. On success: `user_key` computed, per-user root `artifacts/<user_key>/` created, user logged in.

### Login
- UI: verify against user store, put `UserContext(id, email, user_key)` in `st.session_state`.
- API: `POST /v2/auth/login` → 1-hour HS256 JWT. Claims: `sub` (user id), `email`, `user_key`, `jti`, `iat`, `exp`.
- Failure message is always exactly `Invalid email or password` (no enumeration).
- Rate limit: 5 attempts / 60 s per IP, separate from the global limiter.

### Password reset (security question)
1. Reset screen: user enters email.
2. **Always** show a security question — the real one if the email exists, a deterministic decoy (seeded from the email hash) if not. This prevents account enumeration via the reset flow.
3. User submits answer + new password. Wrong answer → generic failure.
4. Rate limit: 5 reset attempts / hour per email AND per IP (stricter than login — answers are more guessable than passwords).
5. On success: password re-hashed, all outstanding JWTs for that user revoked (denylist by `sub`).

Known weakness (accepted): security answers are weaker than passwords — guessable and phishable. Mitigations baked in: argon2id-hashed answers, aggressive rate limiting, decoy questions, revocation of existing sessions on reset.

### Logout
- UI: clear `st.session_state` auth keys, rerun → login screen.
- API: `POST /v2/auth/logout` adds the token's `jti` to an in-memory denylist. Tokens are otherwise valid until their 1-hour expiry.
- No refresh tokens in v1 — expired session means re-login. Cuts four attack surfaces (refresh theft, replay, rotation bugs, denylist sync).

---

## API surface

```
POST /v2/auth/register     {email, password, security_question, security_answer}  → 201 | 400 generic
POST /v2/auth/login        {email, password}                                      → {access_token, token_type, expires_in} | 401 | 429
GET  /v2/auth/me           Bearer token                                           → {email, user_key}
POST /v2/auth/logout       Bearer token                                           → 204 (jti denylisted)
POST /v2/auth/reset/start  {email}                                                → {security_question}   (real or decoy)
POST /v2/auth/reset/finish {email, answer, new_password}                          → 204 | 401 generic | 429
```

- `AuthMiddleware` replaced: validates JWT, sets `request.state.user_id` / `user_key` / `email`. Skips `/v2/health` and `/v2/auth/*` (except `/me` and `/logout`).
- The static `PROFRAG_API_TOKEN` mechanism is removed.
- JWT secret from `PROFRAG_JWT_SECRET` env var; startup fails if missing or < 32 chars.
- `api/deps.py` gains `get_current_user(request) → UserContext` and `get_user_artifacts_root(user) → str` (returns `artifacts/<user_key>`, mkdir-on-first-use).

---

## Per-user storage partitioning

```
artifacts/
├── .users/
│   └── users.db                      ← shared SQLite user store (WAL mode)
├── alice_example_com__a1b2c3d4/
│   ├── metadata.json                 ← per-user global index
│   ├── .locks/                       ← per-user lock files
│   ├── jobs/                         ← per-user job store
│   ├── chroma_db/                    ← per-user Chroma persist dir (PersistentClient mode)
│   └── my_document_v1/
└── bob_example_com__e5f6a7b8/
    └── ...
```

- `config["paths"]["artifacts_root"]` stays as the **base** root in YAML. Effective root is always `base / user_key`, resolved by the router (API) or session context (UI), passed explicitly into workflows.
- Every workflow entry point used by UI or API accepts the user-scoped `artifacts_root` (and `user_key` where Chroma is touched): list/load/inspect, prepare document, URL ingest, summaries start/reset, delete, chat, multi-doc retrieval.
- `resolve_document_folder` / `require_existing_document_folder` take `artifacts_root` as an explicit parameter (removes their hidden global).
- Same-name detection and versioning operate only within the current user's subtree.
- Job creation, listing, recovery, and delete-overlap checks operate only on the current user's `jobs/` directory.
- **Selection is not authorization**: `selected_document_folders` remains pure UI state; ownership is enforced by path scoping at the workflow layer.

---

## ChromaDB partitioning

One centralized resolver — the single point all read/write/delete code goes through:

```python
def user_collection_names(config: dict, user_key: str) -> dict[str, str]:
    return {
        "chunks":    f"chunks__u__{user_key}",
        "summaries": f"summaries__u__{user_key}",
        "cards":     f"cards__u__{user_key}",
    }
```

- Callers to update: `services/write_to_vector_db.py`, `services/metadata_cards.py`, `services/summarize_document.py`, `core/tree_retrieval.py`, `services/document_delete.py`.
- Collection-name isolation is the hard boundary — a missing metadata filter cannot leak rows across users. `document_folder` / `document_id` metadata stays for within-user filtering.
- Collections are created lazily on first access; no per-user provisioning step.

---

## Concurrency & multi-user simultaneous access

| Concern | Resolution |
|---|---|
| Concurrent logins hitting SQLite | WAL mode: reads never block |
| Cross-user lock contention | None by construction — each user has their own `.locks/`, `jobs/`, index |
| Within-user concurrent writes | Existing per-folder/index lock files, now user-scoped — unchanged semantics |
| Chroma concurrent access | Per-user collections: user A's upsert never contends with user B's query. Within one process, PersistentClient serializes writes per collection |
| **Background threads** (`_run_job`, `_run_summary_job`) | Must capture user-scoped `artifacts_root` **at spawn time as a closure argument**. Calling a path getter inside the thread after the request ends resolves the base root and writes to the wrong location — this is the highest-risk mechanical bug in the migration |
| Multiple Uvicorn workers | **Not supported in v1.** In-memory denylist and rate-limit windows are per-process, and Chroma PersistentClient is not multi-process safe. Run `--workers 1`; document the constraint. Scale-out path: Chroma HttpClient (`CHROMA_HOST` already supported) + Redis for denylist/rate limits |

---

## Migration of existing data

One-shot script `scripts/migrate_to_multiuser.py`:

1. Prompt for the owner's email + password + security question → create the first account.
2. Move each existing `artifacts/<doc_folder>/` → `artifacts/<owner_key>/<doc_folder>/`.
3. Move `artifacts/metadata.json` → `artifacts/<owner_key>/metadata.json`.
4. Move `artifacts/jobs/` → `artifacts/<owner_key>/jobs/`.
5. Move `artifacts/chroma_db/` → `artifacts/<owner_key>/chroma_db/` and rename collections to `chunks__u__<owner_key>` etc. (`collection.modify(name=...)`).
6. Idempotent: re-running detects already-migrated layout and exits.

---

## Dependencies added

- `argon2-cffi` (password + answer hashing)
- `PyJWT` (token signing/validation)

Both pure-Python-friendly, no service dependencies. SQLite is stdlib.

---

## Known limitations (accepted for v1)

- Security-question reset is weaker than email-based reset; mitigated by hashing, rate limits, decoys, session revocation.
- Logout/denylist is in-memory: revocations are lost on API restart; stolen API tokens remain valid up to 1 hour.
- Streamlit session is lost on page refresh → re-login (token persistence via browser storage is out of scope).
- Single Uvicorn worker only.
- No roles, sharing, admin UI, or email verification.
