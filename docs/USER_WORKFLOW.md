# Data Storage & Retrieval Workflow

This is the single workflow reference for how data is stored and how retrieval works.

## 1) Ingest to Storage Flow

1. Input accepted (PDF or URL).
2. Content converted to markdown.
3. Images and tables extracted with indexes.
4. Markdown split into section-aware chunks.
5. Chunk/media vectors written to Chroma.
6. Document/section cards created and indexed.
7. Optional structured fields extracted.
8. Metadata updated:
   - per-document: `{doc}/metadata.json`
   - global: `artifacts/metadata.json`
9. Summarization runs in background (resumable).

## 2) Storage Roles

### `artifacts/metadata.json` (global)
- Catalog of all documents.
- Used to locate documents and status quickly.
- Not used for embedding similarity search.

### `{doc}/metadata.json` (document-level)
- Full state for one document.
- Stores chunk paths, summary progress, extracted fields, and pipeline checkpoints.
- Used for resume and retrieval context mapping.

### `chunks/*.json` (document evidence units)
- Actual text evidence units.
- Each chunk carries section and adjacency context (`chunk_number`, `section_path`, assets).

### Chroma DB
- Stores vector embeddings and retrieval metadata.
- Performs similarity search for query embeddings.

## 3) Retrieval Traversal (Simple)

Before Chroma:

- decide candidate docs/scope (selected docs, ready status, doc ids, paths)
- build filters (document_id, section hints)

During Chroma:

- embedding similarity search on vectors (chunks/cards/summaries)

After Chroma:

- map results to human-readable source info
- build citations/source payload
- update/query state files as needed

So the embedding query goes to Chroma, while metadata files provide control-plane context around that query.

## 4) Retrieval Modes

### Single-doc
- Retrieval is constrained to one `document_id`.
- Top evidence chunks are returned.
- Answer is generated from retrieved chunk context.

### Multi-doc
- Query mode is classified (`specific`, `topical`, `aggregation`).
- Retrieval is bounded to selected docs (or routed candidates).
- Chroma returns cross-doc chunk evidence with filters.
- Aggregation mode can use extracted structured fields for deterministic totals.

## 5) Chunk + Retrieval Specifics

- Chunk files are the persisted evidence text units.
- Chroma chunk metadata links every hit back to:
  - `document_id`
  - `chunk_path`
  - `chunk_number`
  - `section_path`
- Tree traversal can narrow by document, then section, then chunk.
- If section-filtered retrieval is sparse, traversal broadens to doc-level chunks.
- Final answer uses retrieved chunk metadata for source mapping and citations.

## 6) Reliability Behaviors

- Metadata writes are atomic for top-level state files.
- Document versions are isolated by `document_id`-prefixed vector IDs.
- Summarization retries on rate limits with backoff.
- Background summary failures are logged and can be retried.
