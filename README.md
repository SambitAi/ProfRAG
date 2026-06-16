# Persistent PDF RAG (v1.1)

Metadata-driven, multimodal RAG for PDFs and web pages that scales to large corpora, reduces token waste, and improves â€œneedle in a haystackâ€ retrieval by narrowing search before chunk-level retrieval.

## What This Project Excels At

- Metadata-first retrieval orchestration with chunk-level evidence grounding
- Multimodal indexing (text, tables, images) with section-aware traversal
- Versioned artifacts and resumable processing for long-running pipelines
- Lower token cost via candidate narrowing before full chunk context assembly
- Single-doc and multi-doc Q&A with citation-aware outputs

## Core Retrieval Model (v1.1)

- `artifacts/metadata.json` and `{doc}/metadata.json` are control-plane catalogs.
- ChromaDB is the embedding similarity engine.
- Query flow:
  1. Use metadata/cards to narrow document and section scope.
  2. Run semantic retrieval in Chroma on chunks (with `document_id`/`section_path` filters).
  3. Rerank and send top evidence to the LLM.
  4. Return grounded answer with source metadata.

## Features

- PDF upload + URL ingest
- Section-aware chunking and metadata linking
- Chroma vector indexing for chunks/media/cards/summaries
- Background summary watcher (L1/L2/L3) with resumable progress
- Card-first routing with summary fallback compatibility
- Initial deterministic aggregation path using extracted structured fields

## Documentation

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Data storage and retrieval workflow: [docs/DATA_STORAGE_RETRIEVAL_WORKFLOW.md](docs/DATA_STORAGE_RETRIEVAL_WORKFLOW.md)

## Prerequisites

- Python 3.10+
- One LLM provider path:
  - OpenAI
  - Azure OpenAI
  - Google native (`google-genai`) or Google OpenAI-compatible endpoint

## Quick Start

```bash
# 1) Clone
git clone https://github.com/your-username/persistent-pdf-rag.git
cd persistent-pdf-rag

# 2) Create venv
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Configure environment
cp .env.example .env

# 5) Run app
streamlit run ui.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `openai` \| `azure` \| `google_native` \| `google` |
| `SUMMARIZER_LLM_PROVIDER` | Optional summarizer provider override |
| `OPENAI_API_KEY` | Required for `LLM_PROVIDER=openai` |
| `AZURE_OPENAI_API_KEY` | Required for Azure |
| `AZURE_OPENAI_ENDPOINT` | Required for Azure |
| `AZURE_OPENAI_API_VERSION` | Azure API version |
| `GOOGLE_API_KEY` | Optional key path for Google |
| `GOOGLE_CLOUD_PROJECT` | Google native/ADC project |
| `GOOGLE_CLOUD_LOCATION` | Google native/ADC region (for example `us-central1`) |
| `GOOGLE_OPENAI_ENDPOINT_ID` | Google-compatible endpoint id (default `openapi`) |
| `CHROMA_HOST` | Optional remote Chroma host |
| `CHROMA_PORT` | Optional remote Chroma port |

### Example: Google native for chat/ingest + Azure summarizer

```env
LLM_PROVIDER=google_native
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

SUMMARIZER_LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

## Key Config (`config/app_config.yaml`)

- Models:
  - `embeddings.model`
  - `chat.model`
  - `summarizer.model`
- Vector collections:
  - `vector_db.collection_name`
  - `vector_db.card_collection_name`
  - `vector_db.summary_collection_name`
- Retrieval:
  - `retrieval.top_k`
  - `retrieval.media_top_k`
- Summary reliability:
  - `summarizer.max_parallel_sections`
  - `summarizer.retry_attempts`
  - `summarizer.retry_base_seconds`
- Structured extraction (Phase 6 baseline):
  - `field_extraction.enabled`
  - `field_extraction.profiles`

## Operational Notes

- If embedding dimension changes, use new Chroma collection names for compatibility.
- If summarizer model fails or throttles, fallback and retry/backoff are applied.
- Google native path requires `google-genai` and valid ADC or API key.
- Collection names are auto-suffixed by embedding model by default (`vector_db.auto_collection_suffix: true`), so switching embedding models (for example 768 -> 1536) routes to a new compatible collection automatically.



## v2 Refactor Notes (Reader-Friendly)

- Retrieval is now unified: single-doc and multi-doc flows use the same tree-based retrieval path.
- Multi-doc coverage improved: retrieval walks each selected document before reranking.
- Version identity contract in code uses folder/version IDs (`<slug>_vN`) to avoid same-name/source collisions.
- Detailed implementation summaries are archived in:
  - `docs/v2/archive/multi_doc_retrieval_refactor_summary.md`
  - `docs/v2/archive/api_layer_refactor_summary.md`

## v3 Multi-User Deployment

### Auth & per-user storage

- Users live in `artifacts/.users/users.db` (SQLite, WAL mode, argon2id password/answer hashes).
- Each user owns `artifacts/<user_key>/` — documents, `metadata.json` global index, `jobs/`, `.locks/`, `chroma_db/`.
- Chroma collections are per-user: `<base>__u__<user_key>` (resolved by `core/collections.py`).
- API auth: HS256 JWT access tokens (1-hour expiry) with an in-memory revocation denylist.
- `PROFRAG_JWT_SECRET` (min 32 chars) must be set in the environment before starting the API. Never put it in config files.

### Migrating an existing single-tenant install

```bash
# Stop the API/UI first.
python scripts/migrate_to_multiuser.py            # interactive
python scripts/migrate_to_multiuser.py --email owner@example.com --password "..." --question-index 0 --answer "..."
```

Creates (or reuses) the owner account, moves all document folders, the global index, `jobs/`, and `chroma_db/` under `artifacts/<owner_key>/`, rewrites stored paths, and renames Chroma collections to the `__u__` format. Idempotent — safe to re-run.

### Single-worker constraint

The API **must run with exactly one worker process**:

```bash
uvicorn api.app:app --workers 1
```

Why:

- The JWT revocation denylist (`api/security.py`) is in-process memory — a second worker would not see tokens revoked by the first, so logout and password-reset revocation would silently stop working. (Denylist loss on restart is accepted: tokens expire within an hour.)
- The auth rate-limit counters (`api/routers/auth.py`) and the rate-limit middleware are in-process; multiple workers would multiply the effective limits.
- The summary watcher (`services/summary_watcher.py`) is a process-wide singleton thread; multiple workers would run duplicate scans.
- ChromaDB `PersistentClient` mode does not support concurrent access to one persist directory from multiple processes.

Background work (ingest, summarization) runs on daemon threads inside the single worker. All thread spawn sites capture the user-scoped `artifacts_root` and `user_key` at spawn time — never resolved inside a thread after request scope ends.

### Scale-out path

1. **Chroma**: set `vector_db.host`/`port` to a shared Chroma server (`HttpClient` mode). Per-user isolation is already at the collection-name level, so nothing else changes.
2. **Token revocation + rate limits**: move the denylist and counters from process memory to a shared store (Redis). Until then, `--workers 1`.
3. **Summary watcher**: run as a separate single-instance process instead of one thread per worker.
4. **Job store / locks**: file-based stores under `artifacts/<user_key>/` assume a single host; multi-host needs a shared queue (e.g. Redis).

