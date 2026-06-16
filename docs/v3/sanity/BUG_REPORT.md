# V3 Post-Completion Sanity Report

**Scope:** Full read of every V3-modified file after all phases (1–7) are marked done.  
**Date:** 2026-06-12  
**Files reviewed:** `api/middleware.py`, `api/deps.py`, `api/security.py`, `api/routers/auth.py`, `api/routers/chat.py`, `api/routers/documents.py`, `api/routers/jobs.py`, `api/routers/summaries.py`, `api/app.py`, `api/schemas/auth.py`, `core/user_store.py`, `core/collections.py`, `core/tree_retrieval.py`, `services/summary_watcher.py`, `workflows.py`, `main.py`, `ui.py`

---

## Bugs found — by severity

---

### BUG-01 · HIGH · `workflows.py:420` — `str(None)` passed as artifacts_root

**File:** [workflows.py:420](../../workflows.py#L420)  
**Function:** `_run_pipeline_from_metadata`

```python
if start_summary_after_unlock:
    start_summarization_background(
        config_path, document_folder,
        artifacts_root=str(artifacts_root),   # <-- BUG
        user_key=user_key,
    )
```

When `artifacts_root` is `None` (single-tenant / CLI path), `str(None)` produces the string `"None"`. This is truthy, so `start_summarization_background` enters the `if artifacts_root:` branch and calls `_require_folder_in_root(document_folder, "None")`. `Path("None").resolve()` resolves to something like `C:\...\ProfRAG\None`, and the containment check immediately raises `FileNotFoundError`. Background summarization silently dies for every document ingested via the single-tenant CLI.

**Fix:** remove the `str()` wrapper — pass the original reference:
```python
artifacts_root=artifacts_root,
```

**Impact:** V3 multi-user API path is unaffected (artifacts_root is always a real string there). The Streamlit UI is also unaffected post-auth because `_user_root()` returns a real path. The CLI / `main.py` single-tenant path breaks silently.

---

### BUG-02 · HIGH · `api/routers/summaries.py` — per-user config fed to global singleton watcher

**File:** [api/routers/summaries.py:18](../../api/routers/summaries.py#L18)  
**Function:** `_run_watcher_job`

```python
workflows.start_summary_watcher(get_config_path(), artifacts_root, user_key=user_key)
```

`summary_watcher.start()` is a process-wide singleton guarded by:
```python
if _WATCHER_THREAD and _WATCHER_THREAD.is_alive():
    return   # ignores the new config entirely
```

If Alice calls `POST /v2/summaries/watcher/start` before the Streamlit UI starts the watcher, the singleton is initialised with Alice's per-user config (`artifacts_root = artifacts/alice_.../`). Inside `_scan_targets`, `iter_user_roots(alice_root)` finds no user-key-named subdirs inside Alice's tree (Alice's root contains document folders, not user roots), so the watcher only ever processes Alice's documents. Bob's pending summaries are never picked up.

The watcher is designed to serve all users by scanning `iter_user_roots(base_root)`. It must always be started with the **base** artifacts root and no user scope.

**Fix:**
```python
# _run_watcher_job — drop the user-scoped args:
workflows.start_summary_watcher(get_config_path())

# start_watcher endpoint — same:
def start_watcher(artifacts_root: str = Depends(get_user_artifacts_root)) -> dict:
    # artifacts_root is not needed; create the job under the user's root for tracking,
    # but start the watcher with the base config.
    job = create_job(artifacts_root, "summaries.watcher.start")
    threading.Thread(
        target=_run_watcher_job,
        args=(job["job_id"], artifacts_root),   # no user_key
        daemon=True,
    ).start()
```

---

### BUG-03 · MEDIUM · Rate-limit dicts grow without bound (root fix)

**Files:**  
- [api/middleware.py](../../api/middleware.py)  
- [api/routers/auth.py](../../api/routers/auth.py)

**Original (pre-sanity) state:** Both the global rate-limit middleware and the per-endpoint login/reset limiter had a broken cleanup:
```python
store.pop(key, None)   # removes the key …
window = store[key]    # … but defaultdict immediately re-adds it
```
The pop was a no-op — keys accumulated indefinitely.

**First partial fix (BUG-03 v1):** Removed the dead pop+recreate block. This fixed the no-op, but keys are still never evicted: one-off IPs/emails make a request, their key is added, they never return, and the key stays forever.

**Root fix:** Lazy periodic eviction — once per window interval, sweep and remove every key whose newest timestamp predates the cutoff.

`api/middleware.py`:
```python
_WINDOW_SECONDS = 60.0
_last_purge: list[float] = [0.0]   # mutable container avoids global declaration in coroutine

# Inside RateLimitMiddleware.dispatch:
if now - _last_purge[0] >= _WINDOW_SECONDS:
    _last_purge[0] = now
    stale = [k for k, w in list(_REQUEST_WINDOWS.items()) if not w or w[-1] < cutoff]
    for k in stale:
        _REQUEST_WINDOWS.pop(k, None)
```

`api/routers/auth.py`:
```python
_PURGE_TIMESTAMPS: dict[int, float] = {}   # id(store) → last_purge_time (LOGIN and RESET tracked independently)

def _purge_stale(store, window_seconds):   # called inside _RATE_LOCK
    store_id = id(store)
    now = time.time()
    if now - _PURGE_TIMESTAMPS.get(store_id, 0.0) < window_seconds:
        return
    _PURGE_TIMESTAMPS[store_id] = now
    cutoff = now - window_seconds
    stale = [k for k, w in list(store.items()) if not w or w[-1] < cutoff]
    for k in stale:
        store.pop(k, None)

def _too_many_attempts(store, key, limit, window_seconds):
    with _RATE_LOCK:
        _purge_stale(store, window_seconds)   # ← eviction happens here
        …
```

**Properties of the fix:**
- Eviction runs at most once per `window_seconds` per dict → O(N) sweep amortised to O(1) per request
- Uses `list(store.items())` snapshot so mutation during iteration is safe
- `_LOGIN_ATTEMPTS` (60 s window) and `_RESET_ATTEMPTS` (3600 s window) are tracked independently via `id(store)`
- Auth eviction happens inside the existing `_RATE_LOCK` — no new lock needed

---

### BUG-04 · MEDIUM · Pydantic schema allows passwords shorter than the store minimum

**File:** [api/schemas/auth.py:7,44](../../api/schemas/auth.py#L7)

```python
class RegisterRequest(BaseModel):
    password: str = Field(min_length=1)      # <-- should be MIN_PASSWORD_LENGTH (8)

class ResetFinishRequest(BaseModel):
    new_password: str = Field(min_length=1)  # <-- same
```

`user_store.MIN_PASSWORD_LENGTH = 8`. A 1–7 character password passes Pydantic validation (no 422), reaches the store, triggers `UserValidationError("Password must be at least 8 characters.")`, and is caught by the router's generic except-block, which returns:
```
HTTP 400: "Could not create account."
```

The actual rejection reason is swallowed. Users with short passwords cannot self-diagnose without reading source code.

**Fix:**
```python
from core.user_store import MIN_PASSWORD_LENGTH

class RegisterRequest(BaseModel):
    password: str = Field(min_length=MIN_PASSWORD_LENGTH)

class ResetFinishRequest(BaseModel):
    new_password: str = Field(min_length=MIN_PASSWORD_LENGTH)
```

This surfaces a clean `422 Unprocessable Entity` with Pydantic's built-in message before the request even reaches the store. The generic catch-all still covers email duplication and other store errors.

---

## Issues — low severity / code smell

---

### ISSUE-01 · LOW · `get_user_artifacts_root` calls `mkdir` on every request

**File:** [api/deps.py:61–63](../../api/deps.py#L61)

```python
def get_user_artifacts_root(user: UserContext = Depends(get_current_user)) -> str:
    root = Path(get_artifacts_root()) / user.user_key
    root.mkdir(parents=True, exist_ok=True)   # syscall on every authenticated request
    return str(root)
```

`mkdir(exist_ok=True)` is safe but issues a filesystem syscall (stat + mkdir) on every request. For a single-worker prototype this is harmless. If request volume grows, add a process-level set of already-created roots.

---

### ISSUE-02 · LOW · `get_summaries` exposes absolute server paths in response

**File:** [api/routers/documents.py:247–256](../../api/routers/documents.py#L247)

```python
return {
    "levels": {
        "level1": {"exists": level1.exists(), "path": str(level1)},
        ...
    }
}
```

`str(level1)` is an absolute path like `C:\...\artifacts\alice_example_com__aaaaaaaa\doc_v1\summaries\level1_onepager.json`. It reveals the user_key in the path and exposes the server's directory layout. Remove the `path` field or replace it with a URL if clients need to fetch the file.

---

### ISSUE-03 · LOW · SQLite schema DDL runs on every connection

**File:** [core/user_store.py:83–97](../../core/user_store.py#L83)

`CREATE TABLE IF NOT EXISTS` is issued on every `_connect()` call. It is idempotent but still round-trips to SQLite. A `_SCHEMA_CREATED` module-level flag or a one-time migration pattern would eliminate it for the hot path.

---

## Orphan code

---

### ORPHAN-01 · `services/summary_watcher.py:stop()`

**File:** [services/summary_watcher.py:85–86](../../services/summary_watcher.py#L85)

```python
def stop() -> None:
    _STOP_EVENT.set()
```

`stop()` is never called from production code. The watcher runs as a daemon thread and dies when the process exits. This is acceptable for the current design, but the function is dead code. Remove it or wire it to a shutdown handler if graceful teardown is ever needed.

---

### ORPHAN-02 · Rate-limit cleanup block (dead logic, see BUG-03)

The `if not window: store.pop(client, None); window = store[client]` block in both rate-limit implementations does nothing useful (see BUG-03). It is effectively dead code — the pop is immediately undone.

---

## Summary table

| ID | Severity | File | Issue | Fixed |
|---|---|---|---|---|
| BUG-01 | HIGH | `workflows.py:420` | `str(artifacts_root)` converts None to `"None"` string; breaks single-tenant summarization | ✅ |
| BUG-02 | HIGH | `api/routers/summaries.py:18` | Per-user config fed to singleton watcher; only first caller's docs get served | ✅ |
| BUG-03 | MEDIUM | `api/middleware.py`, `api/routers/auth.py` | Rate-limit dicts grow without bound; root fix: lazy periodic eviction via `_last_purge` / `_purge_stale` | ✅ |
| BUG-04 | MEDIUM | `api/schemas/auth.py:7,44` | Password `min_length=1` bypasses user-visible validation; opaque 400 | ✅ |
| ISSUE-01 | LOW | `api/deps.py:61` | `mkdir` syscall on every authenticated request | open |
| ISSUE-02 | LOW | `api/routers/documents.py:247` | Absolute filesystem paths leaked in summary response | open |
| ISSUE-03 | LOW | `core/user_store.py:83` | DDL runs on every DB connection | open |
| ORPHAN-01 | — | `services/summary_watcher.py:85` | `stop()` never called from production | open |

---

## What is NOT broken

- Tenancy boundary: `_require_folder_in_root` is wired into every document-read/query/summary workflow entry point. Cross-user path traversal (relative and absolute) returns 404/400.
- `tree_retrieval.retrieve_tree` containment check (lines 130–146) is correct and independent of the workflow layer.
- `iter_user_roots` strict detection (user_key regex + infra dir exclusion + metadata.json exclusion) prevents junk `jobs/` trees under infra or legacy doc dirs.
- Per-user Chroma collection naming and `patch_config_for_user` are consistent: Phase 3 root patch is always applied before Phase 4 collection naming.
- JWT creation, validation, denylist, and user-revocation-timestamp logic are correct.
- `start_summarization_background` config capture (snapshot at spawn time) is correct — background threads never call `get_artifacts_root()` late.
- `_login_user` and `_logout` in `ui.py` correctly isolate session state between users.
- `startup` recovery `iter_user_roots(base_root)` is correct — infra dirs excluded.
- `delete_documents` containment check via `_resolve_document_folder_for_delete` is correct.
- `api/routers/auth.py:reset_finish` logging `record.user_key if record else ""` is null-safe.
- `str.startswith(tuple)` in `AuthMiddleware` is valid Python.
