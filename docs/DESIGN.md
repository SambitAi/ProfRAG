# ProfRAG — Architecture Design Document

## Vision

ProfRAG is a **metadata-driven multimodal RAG** system designed to find the needle in a
haystack across thousands of PDFs with the minimum possible LLM token cost. The guiding
principle is: **metadata does the filtering; the LLM only processes what metadata has
already proven is relevant.**

Scale target: tens of thousands of documents, terabytes of PDF content.

---

## Core Principle — The Metadata Tree

Every query traverses a strict hierarchy before the LLM sees a single token of document
content. Filtering happens at each level using semantic search over pre-built metadata —
not over raw chunks.

```
Query
  │
  ▼
[Level 0] artifacts/metadata.json  ←── global index: all docs, L1 summaries inline
  │  semantic rank → top-K docs (e.g. top 20 of 4,000)
  ▼
[Level 1] {doc}/summaries/level1_onepager.json  ←── 250-word exec summary per doc
  │  (already loaded from global index — no extra reads)
  ▼
[Level 3] {doc}/summaries/level3_detailed.json  ←── per-section summaries
  │  semantic rank → top-M sections (e.g. top 10 of 200 sections)
  ▼
[Leaf]  ChromaDB query — filtered to those sections only
  │  returns: chunks + images + tables
  ▼
LLM — receives compact, pre-filtered context → generates answer + cites sources
```

At no level does the LLM read content it didn't need. At no level does the system load
documents not proven relevant by the level above.

---

## Current State (What Is Built)

| Component | Status | Notes |
|---|---|---|
| PDF → Markdown pipeline | ✅ | PyMuPDF4LLM, 5 stages |
| Section-aware chunking | ✅ | H1/H2/H3 hierarchy, ~1200 chars |
| Image extraction + metadata | ✅ | captions, OCR, surrounding context |
| Table extraction + metadata | ✅ | CSV + structured index |
| ChromaDB write (chunks + media) | ✅ | text-embedding-3-small, persistent |
| L1 / L2 / L3 summary generation | ✅ | tree-summarize, Azure DeepSeek |
| Summary indexing in vector DB | ✅ | `pdf_rag_summaries` collection |
| Single-doc retrieval + chat | ✅ | top-k chunks + parent expansion |
| Multi-doc routing (L1+L3) | ✅ | ranks candidates, up to 5 docs |
| Per-doc metadata.json | ✅ | pipeline state, chunk paths, summary paths |
| UI — 3-pane parallel workflows | ✅ | dynamic refresh, pulse badge |

---

## What Is Missing

| Gap | Impact |
|---|---|
| **No `artifacts/metadata.json`** | `list_documents()` scans all folders on every call — O(N) disk reads. At 10K docs this is a bottleneck. No global entry point for L1 routing. |
| **No persistent watcher** | Summarization only starts within a UI session. App restart loses all pending work. No way to know which docs are awaiting summaries across restarts. |
| **Retrieval does not use the tree** | Multi-doc query retrieves chunks from all selected docs indiscriminately (top-4 per doc, up to 5 docs). Section-level filtering is not applied. L3 summaries are in ChromaDB but not used to filter which sections to retrieve from. |
| **L1 summaries not in global index** | The global routing entry point does not exist as a file — routing goes directly to ChromaDB `pdf_rag_summaries`. This cannot scale: at 10K docs, that collection holds 10K L1 embeddings and 100K+ L3 section embeddings with no on-disk pruning. |
| **No structured field extraction** | For aggregation queries (totals, dates, counts across docs), the LLM must read full chunks each time. Pre-extracted typed fields in metadata would make these near-free. |

---

## Target Architecture

### 1. Data Model — The Tree on Disk

```
artifacts/
├── metadata.json                      ← GLOBAL INDEX (new)
│
├── chroma_db/                         ← vector store (leaf retrieval only)
│
└── {doc_slug}_vN/
    ├── metadata.json                  ← DOCUMENT INDEX (enriched)
    ├── source/
    ├── markdown/
    ├── images/
    │   └── index.json
    ├── tables/
    │   └── index.json
    ├── chunks/
    │   └── {chunk_id}.json
    ├── sections/
    ├── summaries/
    │   ├── level1_onepager.json
    │   ├── level2_medium.json
    │   └── level3_detailed.json
    └── vector/
```

---

### 2. `artifacts/metadata.json` — Global Index Spec

Single file. Written at ingest completion and updated by watcher at each summary level.

```jsonc
{
  "schema_version": 1,
  "last_updated": "2025-04-27T12:00:00Z",
  "documents": {
    "1706_03762_v1": {
      "slug":              "1706_03762",
      "version":           1,
      "document_name":     "1706.03762.pdf",
      "document_folder":   "artifacts/1706_03762_v1",
      "ready_to_chat":     true,
      "total_chunks":      55,
      "summary_status":    "ready",       // pending | in_progress | ready | error
      "summary_level":     3,             // highest completed level (0 = none)
      "l1_summary":        "250-word executive summary inline...",
      "l3_sections":       ["Introduction", "Model Architecture", "Results"],
      "extracted_fields":  {},            // typed key-value pairs (future: invoice total, date, etc.)
      "indexed_at":        "2025-04-27T..."
    }
  }
}
```

**Why inline L1 summary?**
Loading the global index is one file read. The system immediately has all L1 summaries in
memory for semantic ranking — no per-document file reads required at routing time. At 10K
documents with ~300 chars each, the file is ~3MB — trivially fast to read and parse.

**Writes:**
- Pipeline writes the entry (without L1 summary) when `ready_to_chat` is set.
- Watcher updates `summary_status`, `summary_level`, `l1_summary` as summarization progresses.
- Append-only updates — each write merges into the existing JSON, never overwrites unrelated entries.

---

### 3. Document-Level `metadata.json` — Enriched Fields

Existing fields kept. New fields added:

```jsonc
{
  // ... existing fields unchanged ...
  "l3_section_names": ["Introduction", "Model Architecture", "..."],
  "l3_section_chunk_counts": { "Introduction": 4, "Model Architecture": 9 },
  "extracted_fields": {}
}
```

`l3_section_names` lets the retrieval layer know which sections exist before reading
`level3_detailed.json` — useful for budgeting queries.

---

### 4. Watcher — Persistent Background Summarizer

A single daemon thread that starts with the app process and survives UI session restarts.
It does NOT compete with active chat — it uses a separate LLM client and runs at low
priority.

**State machine per document:**

```
ready_to_chat=True
       │
       ▼
  [queue entry]  ←── watcher picks up on startup or on scan tick
       │
  ┌────▼────┐
  │  L3 run │  per-section, resumable: reads summary_progress.level3_sections_done
  └────┬────┘
       │ write metadata.json + artifacts/metadata.json after each section
       ▼
  ┌────────┐
  │  L2 run│  single call, resumable: checks level2_complete flag
  └────┬───┘
       ▼
  ┌────────┐
  │  L1 run│  single call, writes l1_summary to artifacts/metadata.json
  └────┬───┘
       │
  summary_status = "ready"  ←── UI reads this; pulse badge disappears
```

**Resumability:** Every level writes its completion flag to `metadata.json` before
starting the next. On restart, the watcher reads each doc's `summary_progress` and
resumes from the exact last completed step. No work is ever repeated.

**Concurrency:** Max 2 summarization workers running simultaneously (configurable). Chat
requests are unaffected — they run on the main OpenAI client, watcher uses its own.

**Scan interval:** Every 60 seconds. Checks `artifacts/metadata.json` for any
`ready_to_chat=True, summary_status != "ready"` entries not already in the queue.

**Startup sweep:** On process start, before accepting any UI requests, the watcher does
one full scan and enqueues all incomplete documents.

---

### 5. Retrieval — Tree Traversal

Replaces the current "query ChromaDB per doc" approach with a three-phase funnel.

#### Phase 1 — Global routing (L1 metadata)

```python
# Read once — no DB call
global_index = load_global_index("artifacts/metadata.json")

# Semantic similarity against all L1 summaries
# Uses cosine similarity on pre-loaded embeddings (no ChromaDB needed)
candidates = rank_by_l1_similarity(question_embedding, global_index, top_k=20)
```

If only a subset of docs is selected (user ticked checkboxes), filter to that subset
first, then rank.

Token cost of Phase 1: **0 LLM tokens**. One file read + CPU vector math.

#### Phase 2 — Section routing (L3 metadata)

```python
for doc in candidates[:20]:
    l3 = load_l3_summaries(doc["document_folder"])  # one file read per doc
    doc["relevant_sections"] = rank_sections_by_similarity(
        question_embedding, l3["sections"], top_n=3
    )
```

This returns at most 20 × 3 = 60 sections across all candidate documents. Each section
is tagged with its `section_name` and a relevance score.

Token cost of Phase 2: **0 LLM tokens**. 20 file reads + CPU vector math.

#### Phase 3 — Leaf retrieval (ChromaDB)

```python
# Query only the sections proven relevant by Phase 2
results = chroma_collection.query(
    query_embeddings=[question_embedding],
    n_results=top_k,
    where={
        "$and": [
            {"document_folder": {"$in": relevant_doc_folders}},
            {"section_path": {"$in": relevant_section_names}},
            {"item_type": {"$eq": "text"}}
        ]
    }
)
```

ChromaDB now operates as a **leaf retrieval engine**, not a routing engine. The filter
dramatically reduces the candidate pool — instead of searching 50K chunks across all docs,
it searches only the 300-500 chunks in the relevant sections.

Images and tables are fetched with the same section filter.

Token cost of Phase 3: **0 LLM tokens**. One filtered vector query.

#### LLM Call

The LLM receives only the pre-filtered chunks (~8-15 per query), their source metadata,
and any relevant images/tables. Typical context: 4,000–12,000 tokens. One LLM call.

---

### 6. Aggregation Pattern (Iterative Batching)

For queries that require data from many documents (e.g., "total cost across all
companies"):

```
Phase 1: L1 routing → all relevant docs (could be 400 of 4,000)
Phase 2: L3 sections → identify "invoice total" or "amount" section in each

Batch loop:
  for each batch of 20 docs:
    Phase 3: retrieve relevant chunks (invoice amount field)
    LLM mini-call: "extract the total amount from each document"
    → returns 20 structured values

Aggregation call:
  LLM: "sum these values and report total: [20 × 20 = 400 values]"
  → final answer
```

Total LLM calls: ~21 (20 batch + 1 aggregate). Each batch call is small (~5K tokens).
Total cost for 400 relevant docs: ~105K tokens. Without the tree, the same query would
require reading full documents — 400 × 55 chunks × 1200 chars = 26M chars.

The metadata tree is what makes this affordable.

---

### 7. Embedding Strategy

Phase 1 and Phase 2 semantic ranking currently requires embedding the question and
computing similarity against stored embeddings. Two options:

**Option A — Re-embed at query time (current approach for ChromaDB)**
- Embed question once → similarity against L1 texts in global index → similarity against L3 sections
- L1 and L3 texts are short enough to embed and compare in-memory with numpy
- No ChromaDB calls for routing — ChromaDB reserved for leaf retrieval only

**Option B — Pre-embed and cache L1/L3 in global index**
- Store L1 embedding vector in `artifacts/metadata.json` per document
- Store L3 section embeddings in document metadata
- Query time: load embeddings from JSON, compute cosine similarity locally
- Eliminates all ChromaDB calls except Phase 3
- Trade-off: `artifacts/metadata.json` grows (384 floats × 4 bytes × 10K docs = 15MB — still fine)

**Recommended**: Start with Option A (simpler, uses existing ChromaDB summary collection
for ranking). Move to Option B as doc count exceeds 50K.

---

## Implementation Phases

### Phase 1 — Global Index + Watcher (foundation)
1. Write `pipeline.write_global_index_entry()` — called at `ready_to_chat` and at each summary level
2. Write `pipeline.load_global_index()` — replaces directory scanning in `list_documents()`
3. Implement `WatcherThread` — startup sweep + periodic scan + resumable summarization queue
4. Update `artifacts/metadata.json` schema and write it to disk

### Phase 2 — Tree Traversal Retrieval
5. Implement `services/retrieve_context_tree.py` — three-phase funnel replacing `retrieve_context.py`
6. Update `multi_doc_query.py` to use Phase 1+2 from file (not ChromaDB summary collection)
7. Add section-level filter to ChromaDB Phase 3 query
8. Update `main.py` to route through the new retrieval function

### Phase 3 — Aggregation + Structured Fields
9. Add `extracted_fields` extraction to pipeline (configurable schema per document type)
10. Implement iterative batching loop for aggregation queries
11. UI: expose aggregation mode as a chat option when multiple docs selected

---

## Token Budget Per Query Type

| Query type | Docs selected | LLM tokens (approx) |
|---|---|---|
| Single doc, specific question | 1 | 4K–8K |
| Multi-doc, specific question | 100 | 8K–15K |
| Multi-doc, broad question | 4,000 | 10K–20K |
| Aggregation across 400 relevant docs | 4,000 | ~105K total (21 calls) |
| **Without tree (current)** | 100 | 40K–80K |
| **Without tree (current)** | 4,000 | impractical |

---

## Configuration

New fields to add to `app_config.yaml`:

```yaml
watcher:
  enabled: true
  scan_interval_seconds: 60
  max_concurrent_workers: 2
  model: DeepSeek-V3.2          # separate from chat model — can be cheaper

retrieval:
  tree_traversal: true          # when false, falls back to current direct retrieval
  l1_top_k: 20                  # docs to pass from Phase 1 → Phase 2
  l3_top_sections: 3            # sections per doc to pass from Phase 2 → Phase 3
  aggregation_batch_size: 20    # docs per LLM batch for aggregation queries

global_index:
  path: artifacts/metadata.json
  cache_ttl_seconds: 10         # reload from disk after N seconds (for multi-process)
```

---

## What Does Not Change

- The pipeline stages (pdf_to_markdown → extract_images → extract_tables →
  markdown_chunker → write_to_vector_db) are unchanged.
- The per-document folder structure is unchanged.
- ChromaDB remains as the leaf retrieval engine — it is not replaced.
- The UI 3-pane layout and all chat modes are unchanged.
- `level1_onepager.json`, `level2_medium.json`, `level3_detailed.json` files are unchanged.
- The summary generation logic (`summarize_document.py`) is unchanged; only triggering
  and progress tracking move to the watcher.
