# Persistent PDF RAG

Chat with any PDF or web page using a fully persistent, multimodal RAG pipeline — text, tables, and images all retrieved and rendered. Query a single document or ask questions across your entire library at once.

Every pipeline step is saved to disk. If processing fails mid-way, the next run resumes from the last successful step. Multiple versions of the same document are tracked independently.

---

## What it does

- **Upload PDFs or paste a URL** via a Streamlit web UI — web pages are scraped to markdown automatically
- **Extract everything** — text, embedded images (with OCR captions), and markdown tables (converted to CSV)
- **Section-aware chunking** — respects H1/H2/H3 heading hierarchy so chunks stay semantically coherent
- **Multimodal retrieval** — semantic search over text chunks *and* images/tables separately, then merged into a single prompt
- **3-level summarization** — every document gets a per-section summary (L3), a structured medium summary (L2), and a 250-word executive summary (L1), all indexed for fast cross-document routing
- **Multi-document querying** — ask one question across your whole library; a 3-stage semantic funnel picks the most relevant documents before retrieving real chunks
- **Persistent artifacts** — every intermediate file is saved; re-opening a processed document skips straight to chat
- **Document versioning** — uploading the same filename or URL again creates a new versioned folder (`_v2`, `_v3`, …) or resumes an incomplete run
- **Configurable** — swap PDF parser, chunking strategy, models, retrieval depth all via one YAML file

---

## Pipeline

```
PDF Upload  ─────────────────────────────────────────────────────┐
                                                                  │
URL / Web Page ──► scrape_url_to_markdown                        │
                         │                                        │
                         ▼                                        ▼
               pdf_to_markdown_pymupdf4llm     ← layout-aware text + image extraction
                   ↓           ↓
             extract_images  extract_tables    ← parallel asset indexing
                   ↓
         markdown_chunker_section_aware        ← H1/H2/H3 hierarchy preserved
                   ↓
           write_to_vector_db                  ← OpenAI embeddings → ChromaDB (text + media)
                   ↓
           summarize_document                  ← L3 per-section → L2 medium → L1 exec summary
                   ↓                               (each level indexed in pdf_rag_summaries)
            ┌──────────────────────────────┐
            │  per user question           │
            └──────────────────────────────┘
                   ↓
           retrieve_context                    ← semantic search: top-k text + top-k media
                   ↓
           chat_response                       ← GPT answer with cited sources + inline images

  ── or ──

           multi_doc_query                     ← L1 summary filter → L3 section confirm
                   ↓                               → real chunk retrieval across N documents
           chat_response (cross-document)      ← per-document citations, then aggregate answer
```

Each stage writes its outputs to `artifacts/{document}_v{N}/` and records its status in `metadata.json`. The pipeline picks up from `last_successful_step` on any restart.

---

## Artifact layout

```
artifacts/
  my_report_v1/
    metadata.json          # pipeline state, chunk paths, step statuses, summary progress
    source/
      my_report.pdf
    markdown/
      document.md          # also written here for URL-ingested web pages
    images/
      *_page_N_img_M.png
      index.json           # caption, title, page, prev/next context per image
    tables/
      *_table_N.csv
      index.json           # title, headers, caption, prev/next context per table
    chunks/
      *_chunk_000001.json  # text, section_path, image_paths, table_paths
      ...
    sections/
      {h1_slug}.json       # section-level asset registry for parent expansion
    summaries/
      level3_detailed.json # per-H1 section summaries (parallel, resumable)
      level2_medium.json   # 4-6 paragraph structured summary
      level1_onepager.json # 250-word executive summary
    vector/
      index_result.json
    retrieval/
      query_000001.json
      multidoc_000001.json # written during cross-document queries
    chat/
      query_000001.json
  chroma_db/               # shared ChromaDB store — two collections:
    pdf_rag_chunks         #   chunk embeddings (text + media, filtered by document_folder)
    pdf_rag_summaries      #   L1 + L3 summary embeddings for multi-doc routing
```

---

## Prerequisites

- Python 3.10+
- An **OpenAI API key** *or* **Azure OpenAI** credentials (key + endpoint)
- *(Optional)* An **Azure AI Foundry** deployment for the summarizer model (e.g. DeepSeek-V3.2)

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
| `retrieval.media_top_k` | `4` | Images/tables returned per query |
| `retrieval.expand_parent` | `true` | Also retrieve sibling assets from the same H1 section |
| `extraction.images` | `true` | Enable/disable image extraction |
| `extraction.tables` | `true` | Enable/disable table extraction |
| `summarizer.model` | `DeepSeek-V3.2` | Model/deployment used for L1–L3 summarization; can differ from `chat.model` |
| `vector_db.collection_name` | `pdf_rag_chunks` | ChromaDB collection for chunk embeddings |
| `vector_db.summary_collection_name` | `pdf_rag_summaries` | ChromaDB collection for summary embeddings (multi-doc routing) |
| `prompt.media_instruction` | *(see yaml)* | Injected into every LLM system prompt |

---

## Docker deployment

See [Docker section](#docker) below — two containers, one for the app and one for ChromaDB.

---

## Limitations

- **Scanned PDFs** — OCR runs via `pymupdf4llm` on a best-effort basis; heavily image-based PDFs may produce sparse text
- **URL scraping** — Crawl4AI (primary) handles JS-heavy pages; trafilatura and html2text are used as fallbacks. Some paywalled or heavily obfuscated sites may still return sparse content
- **Single Chroma collection per type** — all documents share one chunk collection and one summary collection, filtered by `document_folder` at query time; very large deployments (100+ large PDFs) may benefit from per-document collections
- **Synchronous processing** — PDF upload blocks the Streamlit UI while the pipeline runs; for multi-user production use, move processing to a background worker
- **Summarizer model** — the default `summarizer.model` is an Azure AI Foundry deployment name; update it to match your actual deployment or any OpenAI-compatible model
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
| URL scraping | Network | Crawl4AI uses a headless browser; set `headless=True` (default) and expect ~2–5 s per page |
| Embedding API calls | Network | Batch size limited by OpenAI rate limits; increase `chunk_size` to reduce chunk count |
| Summarization API calls | Network | L3 runs up to 4 sections in parallel; uses `summarizer.model` which can be a cheaper/faster deployment |
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
