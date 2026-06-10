# V3 Part 3: User Authentication & Per-User Isolation — Implementation Plan

## Context

ProfRAG is currently single-tenant: one static Bearer token, one `artifacts/` root, one shared ChromaDB collection set. The goal is to add email/password login, per-user document isolation (`artifacts/{user_slug}/`), and per-user ChromaDB collection namespacing. All callers (FastAPI routes and Streamlit UI) share one workflow layer; auth must be injected without forking that layer.

**What this is NOT:** OAuth. OAuth is a delegation protocol (third-party grants access on your behalf). This is JWT-based session auth — login → signed access token + refresh token. Simpler, self-contained, appropriate here.

---

## Security decisions

| Decision | Rationale |
|---|---|
| bcrypt (cost ≥ 12) for passwords | Industry baseline; never store plaintext |
| Email → `sha256(email)[:16]` hex as `user_slug` | Avoids PII in filesystem; avoids `@`, `+`, `.` path issues; stable and stored in DB |
| Access token: 1h JWT (HS256) | Short-lived; safe for `st.session_state` or localStorage |
| Refresh token: 7-day JWT + server-side denylist | Enables logout invalidation without per-request DB hit |
| In-memory token denylist (`set[str]` of `jti` values) | Prototype-appropriate; lost on restart; swap for Redis in production |
| Per-user ChromaDB collection name prefix | Hard isolation at collection boundary — a missing filter clause cannot leak rows |
| Login rate limit: 5 attempts / 60 s per IP | Brute-force mitigation on `/auth/login` only |
| JWT secret from `PROFRAG_JWT_SECRET` env var | Never from config YAML; validated at startup (min 32 chars) |
| HTTPS enforced at reverse proxy | Not in app code; documented as deployment requirement |

### Known limitation: JWT revocation gap
Access tokens are stateless. A stolen access token is valid until its 1h expiry even after logout. Logout revokes the `jti` in the in-memory denylist — effective for the current process lifetime, lost on restart. Acceptable for a prototype; mitigate in production with Redis-backed denylist.

---

## Architecture overview

```
Browser / Streamlit UI
        │
        │  POST /v2/auth/login  →  {access_token, refresh_token}
        │  Authorization: Bearer {access_token}  on every request
        ▼
FastAPI (api/)
  ├── AuthMiddleware          decodes JWT, sets request.state.user_id + user_slug
  ├── auth router             /login /refresh /logout
  ├── documents router        Depends(get_user_artifacts_root)
  ├── chat router             Depends(get_user_artifacts_root)
  └── jobs / summaries        Depends(get_user_artifacts_root)
        │
        ▼
workflows.py                  receives explicit artifacts_root + user_slug
        │
        ├── core/paths.py     unchanged (path helpers are pure)
        ├── core/global_index.py  per-user metadata.json at artifacts/{user_slug}/metadata.json
        ├── services/document_delete.py  collection names prefixed by user_slug
        └── ChromaDB          collections: {user_slug}_chunks, {user_slug}_summaries, {user_slug}_cards
```

---

## New files

### `core/user_store.py`
SQLite-backed user store at `{base_artifacts_root}/.users/users.db`.

```python
@dataclass
class UserRecord:
    user_id: str         # email — canonical login handle
    user_slug: str       # sha256(email.lower())[:16] — filesystem-safe, stable
    hashed_password: str # bcrypt hash
    created_at: str

def create_user(base_artifacts_root: Path, email: str, password: str) -> UserRecord:
    # normalise email to lowercase, compute slug, bcrypt hash password, insert row
    ...

def verify_user(base_artifacts_root: Path, email: str, password: str) -> UserRecord | None:
    # lookup by email, bcrypt.checkpw, return record or None
    ...

def get_user_by_id(base_artifacts_root: Path, email: str) -> UserRecord | None:
    ...
```

`user_slug` is computed once at account creation and stored. Never re-derived from email at runtime — this means the slug is stable even if the email could theoretically change.

### `api/security.py`
JWT creation, validation, and denylist.

```python
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

_DENYLIST: set[str] = set()   # jti values of revoked tokens

def create_access_token(user_id: str, user_slug: str) -> str:
    # payload: {"sub": user_id, "slug": user_slug, "jti": uuid4().hex,
    #            "type": "access", "exp": now + 60min}
    ...

def create_refresh_token(user_id: str, user_slug: str) -> str:
    # payload: same but "type": "refresh", exp = now + 7 days
    ...

def decode_token(token: str, expected_type: str = "access") -> dict:
    # raises HTTPException 401 if: expired, invalid signature, wrong type, jti in denylist
    ...

def revoke_token(jti: str) -> None:
    _DENYLIST.add(jti)
```

JWT secret read from `os.environ["PROFRAG_JWT_SECRET"]`. Validated at startup — raises `RuntimeError` if missing or shorter than 32 characters.

### `api/routers/auth.py`
No `Depends(get_current_user)` on login/refresh (they bootstrap the token):

```
POST /v2/auth/login
  body: {"email": str, "password": str}
  → 200: {"access_token": str, "refresh_token": str, "token_type": "bearer", "expires_in": 3600}
  → 401: invalid credentials
  → 429: rate limit exceeded (5/60s per IP)

POST /v2/auth/refresh
  body: {"refresh_token": str}
  → 200: {"access_token": str, "refresh_token": str, "token_type": "bearer", "expires_in": 3600}
       (old refresh token revoked, new refresh token issued — rotation)
  → 401: invalid/expired/denylisted token

POST /v2/auth/logout
  header: Authorization: Bearer {access_token}
  body: {"refresh_token": str}  (optional — to also revoke the refresh token)
  → 204: both tokens' jti values added to denylist
```

Login rate limit: per-IP sliding window (`collections.deque`) inside the route handler — 5 attempts in 60 s → 429. Same pattern as `RateLimitMiddleware` but scoped to this endpoint.

---

## Modified files

### `api/middleware.py`
Replace the static-token `AuthMiddleware` check with JWT validation:

- Skip paths: `/v2/health`, `/v2/auth/login`, `/v2/auth/refresh`
- Decode access token from `Authorization: Bearer ...` header
- On success: set `request.state.user_id` and `request.state.user_slug`
- On failure: return `JSONResponse(status_code=401, ...)`

The static `PROFRAG_API_TOKEN` / `auth_token` setting in `APISettings` becomes unused and can be removed.

### `api/deps.py`
Add user-scoped dependency functions:

```python
@dataclass(frozen=True)
class UserInfo:
    user_id: str
    user_slug: str

def get_current_user(request: Request) -> UserInfo:
    # reads request.state.user_id, request.state.user_slug (set by AuthMiddleware)
    return UserInfo(user_id=request.state.user_id, user_slug=request.state.user_slug)

def get_user_artifacts_root(user: UserInfo = Depends(get_current_user)) -> str:
    base = Path(get_app_config()["paths"]["artifacts_root"])
    root = base / user.user_slug
    root.mkdir(parents=True, exist_ok=True)
    return str(root)
```

Update `resolve_document_folder` and `require_existing_document_folder` to accept an explicit `artifacts_root: str` parameter instead of calling the global getter internally. This removes their hidden global dependency.

Keep the original `get_artifacts_root()` for the startup hook only (base root validation and directory creation).

### All routers — `documents.py`, `chat.py`, `jobs.py`, `summaries.py`
Each endpoint signature gains:
```python
artifacts_root: str = Depends(get_user_artifacts_root),
user: UserInfo = Depends(get_current_user),
```

`artifacts_root` is passed explicitly to workflows and thread targets. Background threads (`_run_job`, `_run_summary_job`) receive `artifacts_root` as a closure argument rather than calling `get_artifacts_root()` at thread time (which would otherwise resolve to the base root, ignoring the user scope).

This is mechanical — roughly 15 endpoint signatures + 2 thread wrapper calls.

### `services/document_delete.py`
`_target_collection_names(config)` → `_target_collection_names(config, user_slug: str)`:
- Returns `[f"{user_slug}_chunks", f"{user_slug}_summaries", f"{user_slug}_cards"]`
- `cleanup_document_vectors` already receives `config`; add `user_slug` to its signature and thread it through

`_get_chroma_client(config)` — unchanged. The ChromaDB client is shared; isolation is at the collection-name level.

### `services/write_to_vector_db.py`, `services/metadata_cards.py`, `services/summarize_document.py`, `core/tree_retrieval.py`
Any place that reads `config["vector_db"]["collection_name"]` to open a ChromaDB collection must instead use `f"{user_slug}_chunks"` (etc.). These services already receive `config` as a parameter; add `user_slug: str` alongside it.

The config values (`collection_name`, `summary_collection_name`, `card_collection_name`) become base suffixes or can be replaced with fixed strings (`chunks`, `summaries`, `cards`) since the user prefix already namespaces them.

### `workflows.py`
- `delete_documents(config_path, folders)` → `delete_documents(config_path, folders, user_slug: str)`
- Other workflow entry points that derive `artifacts_root` from config receive it as an explicit parameter from the router instead
- The `config["paths"]["artifacts_root"]` in YAML stays as the **base** root; all workflows use `base / user_slug` as the effective root

### `api/app.py`
- Register `auth_router` at `/v2`
- Add `/v2/auth/login` and `/v2/auth/refresh` to the middleware skip list (exact prefix match)
- Add startup validation: `os.environ.get("PROFRAG_JWT_SECRET")` — raise `RuntimeError` if absent or < 32 chars

### `config/app_config.yaml`
No changes. Auth config lives in env vars only; no secrets in config files.

### `ui.py` (Streamlit)
- Render a login form (email + password) when `st.session_state.get("access_token")` is falsy
- On successful login: store `access_token`, `refresh_token`, `user_slug` in `st.session_state`; `st.rerun()`
- All pipeline calls that go through the API pass `Authorization: Bearer {access_token}` header
- On any 401 API response: clear token keys from session state, `st.rerun()` → login form appears
- On token refresh (near expiry or on 401): call `/v2/auth/refresh` automatically, update `access_token` in session state
- Logout button: calls `/v2/auth/logout`, clears session state, reruns

Token lives in `st.session_state` only — lost on page refresh, user re-logs in. Acceptable for a prototype. For persistence across refresh, a cookie-based approach via a custom Streamlit component would be needed (out of scope here).

---

## ChromaDB per-user isolation — detail

| Collection purpose | Current name (config) | New name |
|---|---|---|
| Chunks | `pdf_rag_chunks_*` | `{user_slug}_chunks` |
| Summaries | `pdf_rag_summaries_*` | `{user_slug}_summaries` |
| Cards | `pdf_rag_cards_*` | `{user_slug}_cards` |

ChromaDB creates a collection on first access if it does not exist — no schema migration needed for new users.

For the double-enforcement principle: collection name prefix (hard boundary) + `user_slug` stored as metadata on every vector for auditability. The metadata field is not used as a filter in production queries (the collection name already isolates), but it enables forensic checks.

---

## Filesystem layout after migration

```
artifacts/
├── .users/
│   └── users.db                ← SQLite user store
├── {user_slug_A}/              ← e.g. "a3f9b2c1d4e5f6a7"
│   ├── metadata.json           ← this user's global index
│   ├── .locks/
│   ├── chroma_db/              ← this user's ChromaDB persistent store
│   └── my_document_v1/
│       ├── metadata.json
│       └── ...
└── {user_slug_B}/
    └── ...
```

---

## Data migration (existing single-tenant data)

1. Create a default admin user (e.g. the owner's email)
2. Move `artifacts/{doc_folder}/` → `artifacts/{admin_slug}/{doc_folder}/`
3. Move `artifacts/metadata.json` → `artifacts/{admin_slug}/metadata.json`
4. Move `artifacts/chroma_db/` → `artifacts/{admin_slug}/chroma_db/`
5. Rename ChromaDB collections: `pdf_rag_chunks_* → {admin_slug}_chunks` etc.
   - ChromaDB supports `collection.modify(name=new_name)` for HTTP client; for persistent client, rename the underlying directory in `chroma_db/`

A one-shot migration script at `scripts/migrate_to_multiuser.py` should handle steps 2–5.

---

## Verification checklist

1. `POST /v2/auth/login` with wrong password → 401
2. 6th login attempt within 60 s from same IP → 429
3. `GET /v2/documents` without token → 401
4. `GET /v2/documents` with valid token → returns only that user's documents
5. Upload doc as user A; login as user B → user B's document list is empty
6. `POST /v2/auth/logout`; reuse old access token on any endpoint → 401 (denylist hit)
7. Refresh token rotation: call `/v2/auth/refresh` → old refresh token rejected on second call
8. Wait 1 h (or mock `exp` in test) → access token rejected on any endpoint
9. ChromaDB probe: `col.get()` on `{user_A_slug}_chunks` returns zero records for user B's documents
10. Streamlit: login → upload → delete → logout → login again → state is correct at each step
11. Page refresh while logged in → login screen appears (token was in session_state only)

---

## Implementation order

1. `core/user_store.py` + `api/security.py` — standalone, unit-testable, no app wiring
2. `api/routers/auth.py` — login / refresh / logout endpoints
3. `api/middleware.py` — replace static token check with JWT decode + `request.state` population
4. `api/deps.py` — `get_current_user`, `get_user_artifacts_root`, update path helpers
5. `api/app.py` — register auth router, startup JWT secret validation
6. All routers — thread `artifacts_root` and `user` through endpoint signatures and background threads
7. `services/` — thread `user_slug` into collection name derivation (write_to_vector_db, metadata_cards, summarize_document, tree_retrieval, document_delete)
8. `workflows.py` — add `user_slug` to affected workflow entry points
9. `ui.py` — login screen, token storage, logout, 401 handling
10. `scripts/migrate_to_multiuser.py` — one-shot migration for existing data

---

## Out of scope

- OAuth / SSO (Google, GitHub)
- Role-based access control (RBAC) — all users have equal permissions on their own docs
- Document sharing between users
- Admin panel or user management UI
- Password reset flow (email delivery not available)
- Persistent login across page refresh in Streamlit (requires custom component)
- Redis-backed denylist (in-memory is sufficient for prototype)
