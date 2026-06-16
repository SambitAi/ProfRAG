# V3 Model Tier (lite → pro) with Deeper Tree Retrieval — Task List

## Goal

Keep `gemini-2.5-flash-lite` as the cheap default for chat. When a user is unsatisfied
with an answer (notably in multidoc mode, where flash-lite hallucinates and under-reads),
they flip a sticky session toggle to a **deep** tier (`gemini-2.5-pro`) that:

1. swaps the answering model, and
2. **raises the retrieval caps** so the existing per-document tree walk fetches more
   sections and leaf chunks per doc and more total chunks — still a tree traversal, never
   a flat similarity dump,
3. **groups the multidoc prompt by document** to support comparison and combined
   multi-document analysis.

## Background — why this is needed

Investigation of the retrieval stack established two facts:

- **Retrieval is already a tree.** Both the single-doc path (`workflows.ask_question` →
  `retrieve_context_tree.retrieve`) and the multi-doc path
  (`workflows.ask_multi_document_question` → `services/multi_doc_query.ask_across_documents`)
  route through `retrieve_tree` → `_walk_per_doc` in `core/tree_retrieval.py`. For each
  selected document it queries the **cards** collection for relevant sections, then queries
  the **chunks** collection constrained to those sections. The tree is not the bottleneck.

- **Three hardcoded, model-independent caps are the bottleneck.** A bigger model alone
  would still only ever see ~20 flat-labeled chunks:
  - `multi_doc_chunk_cap` default **20** — ceiling on chunks sent to the LLM
    (`effective_top_k = min(max(8, docs × per_doc_leaf_k), 20)`,
    `core/tree_retrieval.py:370-373`).
  - `per_doc_section_k` clamped to **max 8** and `per_doc_leaf_k` auto-computed ~6 — both
    **ignore config** (`core/tree_retrieval.py:290-294`; helpers `_auto_section_k` /
    `_auto_leaf_k` lines 29-34).
  - `multi_doc_max_docs: 5` caps documents walked (`config/app_config.yaml`).

  The model is hardcoded to `config["chat"]["model"]` in both answer paths
  (`services/chat_response.py:160`, `services/multi_doc_query.py:492`) with no
  per-request override.

The per-request override mirrors the existing `patch_config_for_user`
(`core/collections.py`) shallow-copy pattern.

## Design Decisions

- **2 tiers**: `standard` = `gemini-2.5-flash-lite` (default), `deep` = `gemini-2.5-pro`.
- **Sticky session toggle** in the UI; persists until changed, user re-asks after flipping.
- **Deep tier groups prompt context by document**; standard tier prompt assembly unchanged
  (lowest risk).

## Implementation Status

- `Phase 1`: Pending
- `Phase 2`: Pending
- `Phase 3`: Pending
- `Phase 4`: Pending
- `Phase 5`: Pending
- `Phase 6`: Pending

## Task List

### Phase 1: Declare the tiers in config

Status: Pending

- Add a `tiers` block under `chat:` in `config/app_config.yaml`. Keep `chat.model` as the
  standard default so nothing breaks when no tier is passed.

  ```yaml
  chat:
    model: gemini-2.5-flash-lite   # standard default (unchanged)
    temperature: 0
    tiers:
      standard:
        model: gemini-2.5-flash-lite
        retrieval:
          multi_doc_chunk_cap: 20
          per_doc_section_k: 8
          per_doc_leaf_k: 6
          multi_doc_max_docs: 5
          group_by_document: false
      deep:
        model: gemini-2.5-pro
        retrieval:
          multi_doc_chunk_cap: 60
          per_doc_section_k: 14
          per_doc_leaf_k: 14
          multi_doc_max_docs: 8
          group_by_document: true
  ```

- Add `per_doc_section_k` and `per_doc_leaf_k` to the existing top-level `retrieval:` block
  so non-tier callers retain current behavior explicitly.

Definition of done:

- config loads with no schema errors,
- absence of any tier key yields exactly today's behavior.

---

### Phase 2: Tier override helper

Status: Pending

- Add `patch_config_for_tier(config, tier)` to `core/config.py`, mirroring
  `patch_config_for_user`'s shallow-copy approach:

  ```python
  def patch_config_for_tier(config: dict, tier: str | None) -> dict:
      tiers = config.get("chat", {}).get("tiers", {})
      spec = tiers.get(tier or "standard")
      if not spec:
          return config
      patched = {**config, "chat": {**config.get("chat", {})}}
      if spec.get("model"):
          patched["chat"]["model"] = spec["model"]
      if spec.get("retrieval"):
          patched["retrieval"] = {**config.get("retrieval", {}), **spec["retrieval"]}
      return patched
  ```

- It must not mutate the input dict and must return the config unchanged for unknown/None tier.

Definition of done:

- helper overlays model + retrieval keys for "deep",
- returns input unchanged for unknown/None tier,
- input dict is never mutated.

---

### Phase 3: Make tree retrieval honor config caps

Status: Pending

This is the load-bearing change. Today `core/tree_retrieval.py:290-294` auto-computes and
clamps the per-doc K values, ignoring config. Change them to **prefer explicit config
values**, falling back to the current auto-compute when absent, and drop the hard
`min(8, …)` ceiling when config sets a higher value:

```python
retrieval_cfg = config.get("retrieval", {})
per_doc_section_k = int(retrieval_cfg.get("per_doc_section_k")
                        or max(2, min(8, section_k_total // doc_count)))
per_doc_leaf_k = int(retrieval_cfg.get("per_doc_leaf_k")
                     or max(2, min(leaf_k_total, max(4, leaf_k_total // doc_count))))
```

- `multi_doc_chunk_cap` is already read from config (line 370) — no change; the deep tier
  value (60) now flows through.
- The per-doc section→leaf walk, dedup, rerank, and post-rerank per-doc floor logic stay
  exactly as-is — only the K/cap magnitudes grow.
- `multi_doc_max_docs`: confirm enforcement points (discovery `_auto_doc_k` and the folder
  cap in `multi_doc_query`) and have both read `config["retrieval"]["multi_doc_max_docs"]`
  so the deep tier (8) takes effect.

Definition of done:

- with deep config, `effective_top_k` and per-doc K rise,
- the traversal shape is unchanged (still section→leaf per doc),
- standard config reproduces today's numbers.

---

### Phase 4: Thread `model_tier` through the call chain

Status: Pending

- **API schemas** (`api/schemas/` chat): add `model_tier: str = "standard"` (optional,
  defaulted) to `SingleChatRequest` and `MultiChatRequest`.
- **API routers** (`api/routers/chat.py`): forward `req.model_tier` into the workflow calls.
- **workflows.py**: `ask_question(...)` and `ask_multi_document_question(...)` gain a
  `model_tier: str | None = None` parameter; both call
  `config = patch_config_for_tier(patch_config_for_user(config, user_key or ""), model_tier)`
  (tier applied **after** user scoping so collection isolation is preserved).
- Single-doc path already forwards `config["chat"]["model"]` to `chat_response.run` — it now
  carries the tier model automatically. Multi-doc already reads `config["chat"]["model"]`.

Definition of done:

- a request with `model_tier="deep"` reaches the LLM call with `gemini-2.5-pro`,
- omitting `model_tier` is identical to today.

---

### Phase 5: Deep-tier prompt grouping

Status: Pending

- In `services/multi_doc_query.py` prompt assembly (lines 446-489), when
  `config["retrieval"].get("group_by_document")` is true, build context grouped per document
  instead of the flat `[Source: …]` list:

  ```
  ### Document: <doc_name>
  [section <path>] <chunk text>
  [section <path>] <chunk text>

  ### Document: <other_doc>
  ...
  ```

- Keep the per-sentence citation instruction and the existing "show per-document values
  first, then the aggregate" directive — grouping reinforces it for comparison and
  combined-analysis questions.
- Standard tier keeps the existing flat assembly.

Definition of done:

- deep tier context is grouped per document,
- standard tier output is byte-for-byte unchanged,
- comparison questions return per-document values then aggregate, with grouped citations.

---

### Phase 6: UI sticky tier toggle + tests

Status: Pending

- **`ui.py`**:
  - `st.session_state.setdefault("model_tier", "standard")`.
  - Render a segmented control / radio near the chat action bar (`render_action_bar`,
    ~line 1114): **⚡ Standard** / **🌳 Deep (slower, more thorough)**, bound to
    `st.session_state["model_tier"]`.
  - Pass the tier into both `pipeline.ask_question(...)` (~line 953) and
    `pipeline.ask_multi_document_question(...)` (~line 1064).
  - Add a caption: "Deep mode reads more of each document with a stronger model."
- **`pipeline` wrappers** (the module `ui.py` imports): forward `model_tier` to the workflow
  functions.
- **Tests** (`tests/`):
  - `patch_config_for_tier` overlays model + retrieval keys for "deep", returns config
    unchanged for unknown/None tier, does not mutate input.
  - Standard-tier multidoc retrieval still yields ≤20 chunks and `flash-lite`.
  - Confirm all existing tests still pass
    (`.\venvcursor\Scripts\python.exe -m unittest discover -s tests`).

Definition of done:

- toggle persists across the session and drives the tier on both chat paths,
- new unit test passes,
- zero regressions in the existing suite.

---

## Files Touched

| File | Change |
|---|---|
| `config/app_config.yaml` | add `chat.tiers`; add `per_doc_*` to `retrieval` |
| `core/config.py` | new `patch_config_for_tier` |
| `core/tree_retrieval.py` | honor config `per_doc_section_k`/`per_doc_leaf_k`; tier `multi_doc_max_docs` |
| `services/multi_doc_query.py` | optional group-by-document prompt; tier doc cap |
| `api/schemas/*` (chat) | `model_tier` field on single/multi requests |
| `api/routers/chat.py` | forward `model_tier` |
| `workflows.py` | `model_tier` param + `patch_config_for_tier` in both ask_* fns |
| `ui.py` (+ `pipeline` wrappers) | sticky toggle, forward tier |

## Verification

1. **Standard unchanged**: ask a multidoc question with no toggle → identical model
   (`flash-lite`) and ≤20 chunks. Existing tests pass.
2. **Deep raises breadth**: flip to Deep, re-ask. Inspect `effective_top_k` and per-doc K —
   must rise (cap 60, per_doc_leaf_k 14); answer cites chunks from **all** selected docs.
3. **Model swaps**: assert the deep call uses `gemini-2.5-pro`.
4. **Tree preserved**: confirm `_walk_per_doc` still runs section→leaf per doc on deep tier.
5. **Comparison query**: 3 related docs on Deep → per-document values then aggregate,
   citations grouped by document.
6. **Unit test**: `patch_config_for_tier` overlay / no-mutation / unknown-tier behavior.

## Out of Scope

- A third (mid / `gemini-2.5-flash`) tier — only `standard` and `deep` for now.
- Automatic tier escalation (the user chooses; no auto-retry on low confidence).
- Per-tier embeddings or reranker model changes (embeddings stay `gemini-embedding-001`).
- Cost metering / usage caps per tier.
- Streaming responses.
