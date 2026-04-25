# Persistent PDF RAG

Chat with any PDF using a fully persistent, multimodal RAG pipeline — text, tables, images, and formulas all retrieved and rendered.

Every pipeline step is saved to disk. If processing fails mid-way, the next run resumes from the last successful step. Multiple versions of the same document are tracked independently.

---

## What it does

- **Upload PDFs** via a Streamlit web UI
- **Extract everything** — text, embedded images (with OCR captions), markdown tables (converted to CSV), and mathematical formulas
- **Section-aware chunking** — respects H1/H2/H3 heading hierarchy so chunks stay semantically coherent
- **Multimodal retrieval** — semantic search over text chunks *and* images/tables/formulas separately, then merged into a single prompt
- **Persistent artifacts** — every intermediate file is saved; re-opening a processed document skips straight to chat
- **Document versioning** — uploading the same filename again creates a new versioned folder (`_v2`, `_v3`, …) or resumes an incomplete run
- **Configurable** — swap PDF parser, chunking strategy, models, retrieval depth all via one YAML file

---

## Pipeline

```
PDF Upload
  ↓
pdf_to_markdown_pymupdf4llm     ← layout-aware text + image extraction
  ↓           ↓           ↓
extract_images  extract_tables  extract_formulas    ← parallel asset indexing
  ↓
markdown_chunker_section_aware  ← H1/H2/H3 hierarchy preserved
  ↓
write_to_vector_db              ← OpenAI embeddings → ChromaDB (text + media)
  ↓ (per user question)
retrieve_context                ← semantic search: top-k text + top-k media
  ↓
chat_response                   ← GPT answer with cited sources + inline images
```

Each stage writes its outputs to `artifacts/{document}_v{N}/` and records its status in `metadata.json`. The pipeline picks up from `last_successful_step` on any restart.

---

## Artifact layout

```
artifacts/
  my_report_v1/
    metadata.json          # pipeline state, chunk paths, step statuses
    source/
      my_report.pdf
    markdown/
      document.md
    images/
      *_page_N_img_M.png
      index.json           # caption, title, page, prev/next context per image
    tables/
      *_table_N.csv
      index.json           # title, headers, caption, prev/next context per table
    formulas/
      *_formula_N.txt
      index.json           # prev/next context per formula block
    chunks/
      *_chunk_000001.json  # text, section_path, image_paths, table_paths, formula_paths
      ...
    sections/
      {h1_slug}.json       # section-level asset registry for parent expansion
    vector/
      index_result.json
    retrieval/
      query_000001.json
    chat/
      query_000001.json
  chroma_db/               # shared ChromaDB store for all documents
```

---

## Prerequisites

- Python 3.10+
- An **OpenAI API key** *or* **Azure OpenAI** credentials (key + endpoint)

---

## Local setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/persistent-pdf-rag.git
cd persistent-pdf-rag

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# .\venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env and fill in your API keys (see table below)

# 5. Launch
streamlit run app.py
# → opens http://localhost:8501
```

---

## Environment variables

| Variable | When required | Description |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI users | Standard OpenAI API key |
| `AZURE_OPENAI_API_KEY` | Azure users | Azure OpenAI resource key |
| `AZURE_OPENAI_ENDPOINT` | Azure users | e.g. `https://my-resource.cognitiveservices.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | Azure users | e.g. `2025-01-01-preview` |
| `CHROMA_HOST` | Docker only | Hostname of ChromaDB HTTP server (e.g. `chromadb`) |
| `CHROMA_PORT` | Docker only | Port of ChromaDB HTTP server (default `8000`) |

The app auto-detects Azure vs OpenAI: if `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are both set, the Azure client is used; otherwise falls back to `OPENAI_API_KEY`.

---

## Configuration (`config/app_config.yaml`)

| Setting | Default | Options / Notes |
|---|---|---|
| `pdf.parser` | `pymupdf4llm` | `pymupdf` (text only) · `pymupdf4llm` (text + images + OCR) |
| `chunking.strategy` | `section_aware` | `section_aware` · `character_count` |
| `chunking.chunk_size` | `1200` | Characters per chunk |
| `chunking.chunk_overlap` | `50` | Overlap between consecutive chunks |
| `embeddings.model` | `text-embedding-3-small` | OpenAI model name or Azure deployment name |
| `chat.model` | `gpt-4.1-mini` | OpenAI model name or Azure deployment name |
| `retrieval.top_k` | `4` | Text chunks returned per query |
| `retrieval.media_top_k` | `4` | Images/tables/formulas returned per query |
| `retrieval.expand_parent` | `true` | Also retrieve sibling assets from the same H1 section |
| `extraction.images` | `true` | Enable/disable image extraction |
| `extraction.tables` | `true` | Enable/disable table extraction |
| `extraction.formulas` | `true` | Enable/disable formula extraction |
| `prompt.media_instruction` | *(see yaml)* | Injected into every LLM system prompt |

---

## Docker deployment

See [Docker section](#docker) below — two containers, one for the app and one for ChromaDB.

---

## Limitations

- **Scanned PDFs** — OCR runs via `pymupdf4llm` on a best-effort basis; heavily image-based PDFs may produce sparse text
- **Single Chroma collection** — all documents share one ChromaDB collection, filtered by `document_folder` at query time; very large deployments (100+ large PDFs) may benefit from per-document collections
- **Synchronous processing** — PDF upload blocks the Streamlit UI while the pipeline runs; for multi-user production use, move processing to a background worker
- **No authentication** — the app is designed for local or single-tenant use; add a reverse proxy with auth before exposing publicly

---

## Docker

### Architecture

Two containers communicate over a Docker network. ChromaDB runs as an HTTP server so its HNSW index stays warm across app restarts (eliminates the cold-start latency of reloading the index from disk on every request).

```
┌──────────────────────────────────┐      ┌────────────────────────┐
│  app  (Streamlit + pipeline)     │◄────►│  chromadb  (HTTP API)  │
│  :8501                           │      │  :8000                 │
│  volume: ./artifacts             │      │  volume: ./chroma_data │
└──────────────────────────────────┘      └────────────────────────┘
        │                                          │
   OpenAI / Azure OpenAI API (external)         persistent HNSW index
```

### Bottlenecks and mitigation

| Stage | Type | Mitigation |
|---|---|---|
| PDF → Markdown | CPU-bound | No easy mitigation; use `pymupdf` parser for speed at the cost of images |
| Embedding API calls | Network | Batch size limited by OpenAI rate limits; increase `chunk_size` to reduce chunk count |
| ChromaDB write | Disk I/O | Mount `./chroma_data` on SSD; separate container keeps index in RAM |
| ChromaDB query | Memory | `mem_limit: 1g` on chromadb container keeps HNSW index hot |
| LLM response | Network | Use `gpt-4o-mini` or `gpt-4.1-mini` for faster, cheaper responses |

### Quick start

```bash
cp .env.example .env   # fill in API keys
docker compose up --build
# → http://localhost:8501
```

### Files

**`Dockerfile`** — app image based on `python:3.11-slim`

**`docker-compose.yml`** — defines `app` and `chromadb` services with named volumes

**`.env.example`** — template for all required environment variables
