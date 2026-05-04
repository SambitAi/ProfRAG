# Persistent PDF RAG (v1.1)

Metadata-driven, multimodal RAG for PDFs and web pages that scales to large corpora, reduces token waste, and improves “needle in a haystack” retrieval by narrowing search before chunk-level retrieval.

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
- Data storage and retrieval workflow: [docs/USER_WORKFLOW.md](docs/USER_WORKFLOW.md)

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
  - `retrieval.tree_traversal`
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
