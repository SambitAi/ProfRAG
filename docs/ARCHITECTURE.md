# Architecture

This document describes how the current system is built, how data moves through storage/retrieval, and what changed in the v2 retrieval refactor.

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

`workflows.py` is the orchestration layer. Services remain independently testable modules.

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

## Retrieval Model (Current)

- Metadata files are control-plane catalogs.
- Chroma is the semantic retrieval plane.
- Query flow:
  1. Route candidates with metadata/cards.
  2. For selected/routed docs, run per-doc section + chunk retrieval.
  3. Apply reranking with per-doc retention safeguards.
  4. Generate grounded answer with source metadata.

## What Changed In V2 (For End Users)

### 1) Single retrieval path
- Single-doc and multi-doc chat now use a unified tree retrieval path.
- Result: more consistent behavior across UI/API calls.

### 2) Better multi-doc coverage
- Retrieval now walks each selected document before reranking.
- Rerank stage includes per-doc retention safeguards to reduce one-doc dominance.
- Result: better chance each selected document contributes usable context.

### 3) Version-safe document identity (code contract)
- Document identity in code is folder/version-derived (`<slug>_vN`) instead of hash of name+URL.
- Result: cleaner handling of versioned documents after re-ingest.

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

## Operational Note

Some migration outcomes (especially versioned-doc validation across old/new artifacts) require corpus re-ingest in an environment with working provider auth/network.
