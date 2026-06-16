# V3 SSO / OIDC Integration — Task List

## Goal

Add Google and Microsoft (Outlook) single sign-on to ProfRAG using OpenID Connect,
so users can authenticate via an existing identity provider instead of (or in addition to)
the local email/password store.

The existing JWT layer (`api/security.py`) and all downstream per-user isolation
(artifacts root, ChromaDB collection scoping, containment checks) are reused unchanged.
OIDC is just a different way to reach the `create_access_token` call.

## Delivery Strategy

Implement in vertical slices. Store extension first (standalone, no FastAPI),
then the callback routes, then the Streamlit flow, then tests.
No phase blocks a later phase from being reviewed independently.

## Cost

- Google OIDC: free (Google Cloud Console OAuth client, no per-user fee).
- Microsoft OIDC personal accounts (@outlook.com, @hotmail.com): free.
- Microsoft Azure AD organizational accounts: free up to 50,000 MAU.
- All required libraries (authlib, msal): Apache 2.0 / MIT, no cost.
- HTTPS required in production for redirect URIs; localhost HTTP allowed in dev.

## Implementation Status

- `Phase 1`: Pending
- `Phase 2`: Pending
- `Phase 3`: Pending
- `Phase 4`: Pending
- `Phase 5`: Pending
- `Phase 6`: Pending

## Task List

### Phase 1: User store extension for passwordless (OIDC) accounts

Status: Pending

- Add `auth_provider` column (`"local" | "google" | "microsoft"`) to the `users` table in `core/user_store.py`; default `"local"` for all existing rows.
- Add `provider_sub` column (the OIDC `sub` claim, globally unique per provider); nullable for local accounts.
- Add `create_or_update_oidc_user(base_root, email, provider, sub) -> UserRecord`:
  - normalise email, derive `user_key` (same formula as local — keeps artifact root stable),
  - upsert: if `email_normalized` already exists, update `provider_sub` if needed; if new, insert with null `password_hash`.
- Add `get_user_by_provider_sub(base_root, provider, sub) -> UserRecord | None` for the callback fast-path.
- `verify_user` must continue to work for local accounts; OIDC accounts with null `password_hash` return `None` on password check.
- Migration: add columns via `ALTER TABLE users ADD COLUMN … DEFAULT …` inside `_connect()` schema block, wrapped in a `try/except OperationalError` so re-runs are idempotent (SQLite does not support `ADD COLUMN IF NOT EXISTS` before 3.37).

Definition of done:

- `create_or_update_oidc_user` round-trips correctly in an isolated tempdir test,
- local account login is unaffected,
- the same email registered locally and then via OIDC merges to one `UserRecord` with the same `user_key`.

---

### Phase 2: OAuth app registration (one-time setup, no code)

Status: Pending

- **Google:** Create an OAuth 2.0 client in Google Cloud Console → Credentials → OAuth client ID (web application). Add authorised redirect URI: `http://localhost:8000/v2/auth/google/callback` (dev) and `https://<prod-domain>/v2/auth/google/callback` (prod). Note `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`.
- **Microsoft:** Register an app in Azure Portal → App registrations → New registration. Supported account types: "Personal Microsoft accounts only". Add redirect URI `http://localhost:8000/v2/auth/microsoft/callback`. Note `MICROSOFT_CLIENT_ID`; create a client secret under Certificates & secrets, note `MICROSOFT_CLIENT_SECRET`.
- Add to `.env` / shell environment (never to `config/app_config.yaml` or any committed file):
  ```
  GOOGLE_CLIENT_ID=...
  GOOGLE_CLIENT_SECRET=...
  MICROSOFT_CLIENT_ID=...
  MICROSOFT_CLIENT_SECRET=...
  OIDC_REDIRECT_BASE_URL=http://localhost:8501   # Streamlit port; overridden in prod
  ```
- Add `authlib` to `requirements.txt`.

Definition of done:

- credentials are in environment only,
- both provider consoles show the redirect URIs registered,
- `pip install authlib` succeeds in the project venv.

---

### Phase 3: FastAPI OIDC callback routes

Status: Pending

- Create `api/routers/sso.py` with four routes:
  ```
  GET /v2/auth/google/login       → redirect to Google OIDC authorize URL (sets signed state cookie)
  GET /v2/auth/google/callback    → exchange code, upsert user, issue ProfRAG JWT, redirect to UI
  GET /v2/auth/microsoft/login    → redirect to Microsoft OIDC authorize URL
  GET /v2/auth/microsoft/callback → same pattern
  ```
- Use `authlib.integrations.starlette_client.OAuth` to configure both providers:
  - Google discovery: `server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"`
  - Microsoft personal: `server_metadata_url = "https://login.microsoftonline.com/consumers/v2.0/.well-known/openid-configuration"`
- Callback flow (same for both providers):
  1. Verify `state` cookie (CSRF — `authlib` handles this automatically).
  2. Exchange authorisation code for ID token.
  3. Extract `email` and `sub` from verified ID token claims; reject if `email` is absent or `email_verified` is false.
  4. Call `user_store.create_or_update_oidc_user(base_root, email, provider, sub)`.
  5. Call `security.create_access_token(user_id=record.id, email=record.email_normalized, user_key=record.user_key)`.
  6. Redirect to `{OIDC_REDIRECT_BASE_URL}/?token=<jwt>` (Streamlit reads via `st.query_params`).
- Register `sso_router` in `api/app.py` at prefix `/v2`.
- Add all four paths (`/v2/auth/google/login`, `/v2/auth/google/callback`, `/v2/auth/microsoft/login`, `/v2/auth/microsoft/callback`) to `_AUTH_SKIP_PREFIXES` in `api/middleware.py`.

Definition of done:

- visiting `GET /v2/auth/google/login` redirects to Google's OAuth consent screen,
- completing the Google flow issues a valid ProfRAG JWT verifiable via `GET /v2/auth/me`,
- Microsoft flow works identically,
- a user with no prior account gets a new `UserRecord`; a returning user's `user_key` is unchanged.

---

### Phase 4: Streamlit UI changes

Status: Pending

- On the login screen in `ui.py`, add two buttons below the existing password form:
  - `st.link_button("Sign in with Google", "/v2/auth/google/login")` (Streamlit ≥ 1.27)
  - `st.link_button("Sign in with Microsoft", "/v2/auth/microsoft/login")`
- On every page load, inspect `st.query_params.get("token")`:
  - if present, call `security.decode_token(token)` — on success, populate `auth_user_key`, `auth_email`, `auth_artifacts_root` in `st.session_state`, then `st.query_params.clear()` and `st.rerun()`,
  - if `TokenError` is raised, show a generic "Sign-in failed." message and clear the param.
- Gate the password reset tab on `st.session_state.get("auth_provider") == "local"` — OIDC users reset credentials at their provider, not here.
- Logout behavior is unchanged: `st.session_state.clear()` + `st.rerun()`.
- Store `auth_provider` (`"google"` / `"microsoft"` / `"local"`) in session state alongside the existing auth keys so the UI can conditionally show the reset option.

Definition of done:

- clicking "Sign in with Google" completes the OAuth flow and lands back in the Streamlit app fully authenticated,
- the document list, chat, and upload panes are scoped to the OIDC user's `user_key` (same isolation as local auth),
- logout clears the session; re-login via Google restores the same corpus.

---

### Phase 5: Security hardening

Status: Pending

- **State / CSRF:** `authlib`'s `starlette_client` sets a signed state cookie automatically; verify it is enabled (do not pass `use_state=False`).
- **`email_verified` check:** after decoding the ID token, assert `claims.get("email_verified") is not False`; reject if false (Google sets this; Microsoft personal accounts always return verified emails but the check is harmless).
- **Allowed redirect domains:** validate `OIDC_REDIRECT_BASE_URL` against an allowlist at startup via a helper in `api/deps.py` settings; reject at server start if it does not match.
- **Scope:** request `openid email profile` only; do not request `offline_access`.
- **Production token delivery:** replace the query-param redirect with an `HttpOnly` session cookie for production deployments; document the tradeoff (query param is simpler for dev but briefly exposes the JWT in the address bar and referrer headers).

Definition of done:

- state mismatch on callback → 400, no session created,
- `email_verified: false` → 400,
- `OIDC_REDIRECT_BASE_URL` not in allowlist → server refuses to start.

---

### Phase 6: Tests

Status: Pending

- Create `tests/test_sso.py`:
  - `OidcUserStoreTests`:
    - `create_or_update_oidc_user` creates new record with correct `user_key`,
    - second call with same `sub` returns same `user_key` (idempotent upsert),
    - same email pre-registered locally → OIDC call returns same `user_key`,
    - local `verify_user` still works for local accounts,
    - OIDC account with null `password_hash` returns `None` on `verify_user`.
  - `SsoRouteTests` (FastAPI `TestClient` with `authlib` token exchange mocked):
    - `GET /v2/auth/google/login` → 302 with `Location` pointing to Google,
    - mocked callback with valid ID token → 302 to Streamlit with `?token=...`; `GET /v2/auth/me` with that token → 200 correct email,
    - callback with missing `email` claim → 400,
    - callback with `email_verified: false` → 400,
    - callback with state mismatch → 400.
  - Confirm all existing 82 tests still pass (OIDC is purely additive; no existing route or schema is changed).

Definition of done:

- OIDC callback issues JWTs that pass `GET /v2/auth/me`,
- all invalid/tampered callbacks are rejected with 400,
- zero regressions in the existing test suite.

---

## Test Scenarios

### Success cases

- New user signs in with Google → account created, corpus is empty, upload works normally.
- Returning Google user signs in again → same `user_key`, same document list.
- Local-account email re-used via Google → same `user_key`, same corpus, both login methods work.
- Microsoft personal account (@outlook.com) completes the same flow end-to-end.

### Failure cases

- Callback with forged `state` parameter → 400.
- ID token with no `email` claim → 400.
- ID token with `email_verified: false` → 400.
- Tampered or expired ID token → 400 (authlib validates signature and expiry).
- Redirect to an unapproved domain → server refuses at startup.

---

## Manual Verification Checklist

- Visit `http://localhost:8000/v2/auth/google/login` → Google consent screen appears.
- Complete Google sign-in → redirected to Streamlit with `?token=...` → app loads, correct user displayed.
- Click Logout → session cleared → login screen with SSO buttons reappears.
- Visit `http://localhost:8000/v2/auth/microsoft/login` → Microsoft sign-in screen appears.
- Complete Microsoft sign-in → same Streamlit flow.
- Register same email locally, then sign in via Google → same document list visible in both sessions.
- Manually craft a callback URL with a wrong `state` value → 400 returned, no session created.
- Confirm the reset-password tab is hidden when `auth_provider` is `"google"` or `"microsoft"`.

---

## Out of Scope

- SAML / enterprise SSO beyond Azure AD personal accounts.
- Refresh tokens for OIDC sessions (ProfRAG issues its own 1-hour JWT as before; the provider token is discarded after the callback).
- Role or permission mapping from provider groups or directory attributes.
- Email verification loop for local accounts.
- Admin UI for viewing or unlinking linked accounts.
- Multi-provider account merging beyond "same email → same `user_key`".
