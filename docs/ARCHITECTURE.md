# Architecture

This document describes how the current system is built and how data moves through storage and retrieval.

## Pipeline

```text
PDF/URL ingest
  -> markdown conversion
  -> image/table extraction
  -> section-aware chunking
  -> vector indexing (chunks/media)
  -> card generation + card indexing
  -> optional field extraction
  -> background summarization (resumable)
  -> retrieval + answer generation
```

`main.py` is the orchestrator. Services are independent modules.

## Storage Model

```text
artifacts/
  metadata.json                    # global index (all docs)
  <doc>_vN/
    metadata.json                  # document-level state
    source/
    markdown/
    images/index.json
    tables/index.json
    chunks/*.json
    sections/*.json + sections/index.json
    summaries/*.json
    retrieval/query_*.json
    chat/query_*.json
    vector/index_result.json
  chroma_db/                       # local Chroma persistence (if not remote)
```

## Retrieval Model

- Metadata files are control-plane catalogs.
- Chroma is the semantic retrieval plane.
- Query flow:
  1. Narrow scope with metadata/cards (`document_id`, section hints).
  2. Query Chroma vectors (chunks/cards/summaries) with filters.
  3. Rerank evidence and generate answer with source metadata.

## Metadata Responsibilities

- `artifacts/metadata.json`
  - corpus catalog
  - document status and pointers
  - fast listing/routing without folder scans

- `{doc}/metadata.json`
  - per-document pipeline state
  - `chunk_paths`, summary progress, extracted fields
  - resume/checkpoint source of truth

## Provider Architecture

Provider path is environment-driven:

- `LLM_PROVIDER=openai`
- `LLM_PROVIDER=azure`
- `LLM_PROVIDER=google`
- `LLM_PROVIDER=google_native`

Optional:

- `SUMMARIZER_LLM_PROVIDER=<provider>` for summarizer-only override.

## Chroma Collections

- Primary chunks/media collection: `vector_db.collection_name`
- Card collection: `vector_db.card_collection_name`
- Summary collection: `vector_db.summary_collection_name`

If embedding dimensionality changes, switch to new collection names for compatibility.

## Summarization Runtime

- Levels: L3 (section), L2 (medium), L1 (executive)
- Runs asynchronously (watcher/background thread)
- Resumable via metadata checkpoints
- Rate-limit resilience via retry/backoff and configurable parallelism
