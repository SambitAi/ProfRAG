# ProfRAG — User Workflow Guide

## Overview

ProfRAG has three fully independent parallel workflows. You can run all three at the same time:

```
┌─────────────────────┬────────────────────────┬──────────────────────────┐
│   LEFT PANE         │   MIDDLE PANE          │   RIGHT PANE             │
│   Add Documents     │   Generate Summaries   │   Chat                   │
│                     │                        │                          │
│  Independent of     │  Independent of chat   │  Always available;       │
│  summarization      │  and ingest            │  works on any ready doc  │
└─────────────────────┴────────────────────────┴──────────────────────────┘
```

---

## Workflow A — Add a Document

### Option 1: Upload a PDF
1. In the left pane, click **Add Document → Upload PDF**
2. Drag and drop or select a `.pdf` file
3. If a document with the same name already exists:
   - **Reuse** — open the already-processed version, go straight to chat
   - **Rebuild** — create a new version folder and reprocess
4. Click **Process uploaded PDF**
5. A spinner runs while the pipeline executes (5 stages, see below)
6. When complete, a green dot (🟢) appears next to the document name

### Option 2: From URL
1. Click **Add Document → From URL**
2. Paste any web page URL or direct PDF URL
3. Choose Reuse / Rebuild if the URL was previously ingested
4. Click **Process URL**

### Pipeline stages (run once per document)
| Stage | What it does |
|---|---|
| pdf_to_markdown | Converts PDF pages to structured markdown |
| extract_images | Extracts and captions images |
| extract_tables | Extracts tables to structured format |
| markdown_chunker | Splits document into ~1200-character section-aware chunks |
| write_to_vector_db | Embeds all chunks into the vector database |

Once **write_to_vector_db** completes → document shows 🟢 and is selectable for chat.
Summarization starts automatically in the background (see Workflow B).

### Status indicators (left pane)
| Indicator | Meaning |
|---|---|
| Green ring dot | Ready to chat — fully indexed (checkbox is enabled) |
| Red ring dot | Processing or failed — not yet selectable (checkbox is disabled) |
| 📝 (in label) | Summaries fully generated |
| ⏳ (in label) | Summarization in progress |
| ❌ (in label) | Summarization failed |
| ⟳ amber pulse badge | Summarization actively running — auto-updates every ~3 s |

---

## Workflow B — Generate Summaries

Summaries are **optional** but unlock the Routing chat mode (searching across all documents by question).

### When summaries start
Summarization begins automatically in the background as soon as a document finishes processing. You do not need to click anything. If the daemon was killed (e.g., app restart), click **Generate Summaries** in the middle pane.

### Three summary levels
| Level | File | What it contains | When to use |
|---|---|---|---|
| **1-Pager** | level1_onepager.json | Single compressed overview (~1 page) | Quick document orientation |
| **Medium** | level2_medium.json | Expanded summary across all sections | Understanding the full arc |
| **Detailed** | level3_detailed.json | Per-section summaries | Deep review or comparison |

### Summary pipeline order
```
Level 3 (per section) → Index L3 → Level 2 (whole doc) → Level 1 (compressed) → Index L1
```
The middle pane auto-refreshes every ~3 seconds while any summarization is active — you will see the progress badge update without clicking anything.

### Viewing summaries
1. Select a document using its checkbox in the left pane
2. In the middle pane, click **1-Pager**, **Medium**, or **Detailed**
3. The content expands below the button row
4. Click the active button again to collapse
5. Click **↻ Regenerate this level** to rerun that level and all downstream levels

---

## Workflow C — Chat

Chat is always available. Which mode activates depends on your document selection:

### Mode 1: Routing (no documents selected)
**When:** No checkboxes are ticked in the left pane.

```
Ask a question → system searches L1 + L3 summaries across all docs
→ shows ranked candidate list with match % → you pick which docs to search
→ switches to Deep Search mode
```

Best for: "I don't know which document has what I need."
Requires: At least some documents have summaries generated (L1 indexed).

### Mode 2: Single-Document Chat
**When:** Exactly one document checkbox is selected.

```
Ask a question → retrieves top chunks from that doc → answer + source sections
```

Sources show the section hierarchy (e.g., "3.2 Methodology → Results") and a text snippet. Images or tables from adjacent chunks are displayed if available.

### Mode 3: Direct Multi-Document Chat
**When:** Two or more document checkboxes are selected.

```
Ask a question → ranks selected docs by summary relevance → retrieves top chunks
from up to 5 most relevant docs → single synthesized answer with per-doc citations
```

The system automatically filters to the most relevant docs — selecting 10 documents and asking a narrow question will focus on the 1–5 that are actually relevant.

### Mode 4: Deep Search (post-routing)
**When:** You confirmed routing candidates (clicked "Confirm & Search").

```
Active document set is locked → ask follow-up questions → multi-doc retrieval
→ type /back to return to routing and change documents
```

Shortcut: Type `/back` in the chat input to exit Deep Search and return to Routing mode.

---

## Chat mode transition map

```
No selection
    │
    ▼
ROUTING MODE ──[ask question]──► show ranked candidates
    │                                     │
    │                        [Confirm & Search]
    │                                     │
    │                                     ▼
    │                           DEEP SEARCH MODE
    │                                     │
    │                              [/back or
    │                           Change Documents]
    │◄────────────────────────────────────┘
    │
    │  [select 1 doc checkbox]
    ▼
SINGLE-DOC CHAT
    │
    │  [select 2+ doc checkboxes]
    ▼
MULTI-DOC CHAT
    │
    │  [Deselect All]
    ▼
ROUTING MODE
```

---

## Tips

- **Collapse panes** using the ◀ button inside each pane to give chat more space. Expand with the ▶ strip on the edge.
- **Reuse vs. Rebuild:** Reuse is instant — it skips reprocessing and jumps straight to chat. Use Rebuild only if the file has changed.
- **Pipeline resume:** If processing was interrupted (app crash, restart), uploading the same file again automatically resumes from the last completed step.
- **Multiple documents at once:** You can upload a second document while chatting with the first — the upload pane and chat are completely independent.
- **Summaries don't block chat:** A document is usable for chat as soon as it shows 🟢, even if summaries are still generating.
- **Routing accuracy:** Routing relies on L1 and L3 summaries. Documents without any summaries won't appear as routing candidates; select them manually instead.
- **Document versions:** Rebuilding a document creates a new versioned folder (e.g., `report_v2`). Old versions remain available and selectable.
