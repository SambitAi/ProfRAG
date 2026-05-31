# API Runbook

## Start

1. Ensure dependencies are installed from `requirements.txt`.
2. Set env vars if needed:
   - `LLM_PROVIDER` (set explicitly; avoid implicit provider fallback)
   - `PROFRAG_CONFIG_PATH`
   - `PROFRAG_API_HOST`
   - `PROFRAG_API_PORT`
   - `PROFRAG_API_TOKEN` (optional auth)
   - `PROFRAG_API_RATE_LIMIT_PER_MINUTE`
   - `PROFRAG_API_MAX_REQUEST_BYTES`
3. Provider examples:
   - Google Vertex native:
     - `LLM_PROVIDER=google_native`
     - `GOOGLE_CLOUD_PROJECT=contract-reader-495223`
     - `GOOGLE_CLOUD_LOCATION=us-central1`
     - `gcloud auth application-default login`
   - Azure OpenAI:
     - `LLM_PROVIDER=azure`
     - `AZURE_OPENAI_API_KEY=...`
     - `AZURE_OPENAI_ENDPOINT=...`
4. Run:
   - `python api_server.py`
5. Retrieval path is unified:
   - tree retrieval is always on for single-doc and multi-doc flows (no `retrieval.tree_traversal` toggle).

## Health Checks

1. `GET /v2/health`
2. Expect `200` when ready.
3. `503` means config/storage/vector readiness failed.

## Job Lifecycle

1. Write endpoints return `job_id`.
2. Poll `GET /v2/jobs/{job_id}`.
3. Terminal states: `success`, `error`.

## Common Failure Triage

1. `401` unauthorized:
   - verify `Authorization: Bearer <token>` header
2. `429` rate limit:
   - reduce request burst or raise `PROFRAG_API_RATE_LIMIT_PER_MINUTE`
3. `413` payload too large:
   - reduce upload size or raise `PROFRAG_API_MAX_REQUEST_BYTES`
4. `503` health not ready:
   - check config path and vector backend availability

## Rollback

1. Stop API process.
2. Revert to previous commit/tag.
3. Restart with prior environment.
