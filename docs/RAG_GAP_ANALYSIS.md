# RAG Gap Analysis For 1000s Of PDFs (Low Cost, Fast, Accurate, Multimodal)

## Target

System goal:
- low cost at scale
- fast retrieval for large corpus (1000s of PDFs)
- high answer accuracy and faithfulness
- multimodal evidence (text, table, image)

Current strengths in this repo:
- section-aware chunking + hierarchical summaries
- metadata-driven scope control (`artifacts/metadata.json` + per-doc metadata)
- multimodal extraction pipeline (text/tables/images)
- retrieval traversal that can narrow by document and section

---

## Executive Gap Summary

High-impact gaps to address first:
1. Hybrid retrieval (sparse + dense) is not first-class.
2. Adaptive retrieval depth is limited (query-intent-based control should be stronger).
3. Reranking is present but not yet a robust cross-encoder stage for hard queries.
4. Retrieval confidence and omission handling need stronger recovery loops.
5. Evaluation harness for production quality is not yet complete (recall/faithfulness dashboards).
6. Index and query concurrency strategy for 1000s corpus + mixed load needs explicit tuning plan.

---

## Gap Analysis By Layer

## 1) Embeddings

Current:
- single embedding model path configured at runtime
- model-suffixed collections prevent dimension collision (good)

Gap:
- no explicit multi-tier embedding strategy by doc class/query class
- no benchmark matrix for cost/latency/recall trade-offs per model

Recommendation:
- benchmark 2-3 embedding tiers (small/medium/large) on your own corpus.
- adopt policy:
  - default small for broad retrieval
  - escalate to larger model for low-confidence queries or complex technical asks

Expected outcome:
- lower average cost with high-quality fallback for difficult queries

---

## 2) Index Strategy (ANN behavior)

Current:
- Chroma default ANN behavior used, with document/section filtering

Gap:
- no explicit ANN tuning protocol by corpus size, query rate, and filter selectivity
- no formal policy for high-cardinality metadata filtering under load

Recommendation:
- define index tuning playbook per corpus tier (10k, 100k, 1M chunks).
- measure p50/p95/p99 latency with/without metadata filters.
- enforce two-stage retrieval for scale:
  1) doc/section candidate narrowing
  2) chunk ANN over narrowed scope

Expected outcome:
- stable latency under scale and better recall/latency balance

---

## 3) Retrieval Algorithm

Current:
- metadata-aware routing and tree traversal available
- multi-doc routing exists

Gap:
- hybrid sparse+dense retrieval not central in current path
- alpha (sparse/dense blend) is not dynamically tuned by query intent
- retrieval stopping/expansion policy can still exit early on difficult questions

Recommendation:
- add hybrid retrieval as first-class:
  - BM25/keyword sparse score + embedding dense score
- add intent-aware adaptive alpha:
  - entity/fact queries -> higher sparse weight
  - conceptual/explanatory -> higher dense weight
- implement controlled deepening loop:
  - start summary/card level
  - drill down only if confidence low
  - broaden scope before abstain

Expected outcome:
- better cross-document hit rate, fewer false negatives, lower premature abstains

---

## 4) Reranking

Current:
- reranking logic exists

Gap:
- no explicit heavy rerank path (cross-encoder) for ambiguous or high-stakes queries

Recommendation:
- add selective cross-encoder reranking on top-N candidates only (for example N=20-50).
- gate it by query complexity and confidence.

Expected outcome:
- improved top-k precision and answer faithfulness where it matters most

---

## 5) Multimodal Evidence Handling

Current:
- image/table extraction and linking present
- media can be attached in answer path

Gap:
- multimodal scoring is still mostly text-led
- limited explicit fusion strategy between text chunks and nearby table/image evidence

Recommendation:
- add evidence fusion scoring:
  - text relevance score
  - table relevance score
  - image/caption relevance score
- create multimodal citation policy:
  - force at least one relevant evidence type citation when media is central

Expected outcome:
- more reliable table/image-grounded answers and fewer text-only misses

---

## 6) Reliability And Omission Control

Current:
- citation checks and retries exist

Gap:
- omission hallucination control not fully formalized
- weak retrieval cases can still produce confident but incomplete responses

Recommendation:
- add retrieval confidence gate before generation.
- if low confidence:
  1) query rewrite
  2) hybrid retry
  3) broaden document scope
  4) only then abstain
- track omission errors separately from generation hallucinations.

Expected outcome:
- improved trustworthiness and clearer abstain behavior

---

## 7) Evaluation And Production Metrics

Current:
- qualitative testing and scenario checks done

Gap:
- no complete quantitative eval loop tied to release gates

Recommendation (must-have metrics):
- Retrieval:
  - recall@k
  - nDCG@k
  - MRR
- Generation:
  - citation coverage ratio
  - faithfulness score (claim-evidence support)
  - answer relevance score
- Safety/behavior:
  - abstain precision/recall
  - omission rate
- Ops:
  - p95/p99 retrieval latency
  - cost per successful answer

Expected outcome:
- objective release quality control instead of manual-only checks

---

## 8) Scale And Concurrency

Current:
- background summarization and ingestion flow available

Gap:
- no explicit workload isolation strategy for heavy ingest + live query concurrency

Recommendation:
- isolate indexing and query workloads with queue/backpressure.
- add batching + jittered retries on provider and vector DB calls.
- define SLOs by mode:
  - ingest throughput
  - query p95 latency

Expected outcome:
- predictable performance during concurrent ingestion and Q&A

---

## Prioritized Exploration Roadmap

## Priority 1 (Immediate)
1. Hybrid sparse+dense retrieval + adaptive alpha.
2. Retrieval confidence gate + recovery loop before abstain.
3. Evaluation harness with recall@k, faithfulness, citation coverage, p95 latency.

## Priority 2
1. Selective cross-encoder reranking for complex queries.
2. Multimodal evidence fusion scoring (text+table+image).
3. ANN/filter performance benchmark suite by corpus size.

## Priority 3
1. Multi-tier embedding policy (small default, large fallback).
2. Workload isolation for high-concurrency ingest/query.
3. Cost optimizer policy (quality target with budget cap).

---

## What To Explore Next (Concrete Experiments)

1. Embedding A/B/C on same eval set:
- measure recall@k, cost/query, latency/query.

2. Dense-only vs hybrid retrieval:
- compare omission rate and answer relevance on multi-doc questions.

3. Rerank strategy experiment:
- baseline rerank vs cross-encoder gated rerank.

4. Adaptive traversal experiment:
- static depth vs confidence-driven deepening.

5. Multimodal fusion experiment:
- text-only ranking vs text+table+image fused ranking.

Success criteria:
- improve faithfulness and recall while reducing cost per accepted answer.

---

## Recommendation For Your Current Direction

Before full decentralization implementation, prioritize retrieval quality/cost experiments first (Priority 1 above). This will lock the target behavior so provider/backend decoupling preserves the right performance envelope when you later swap Chroma, LLM providers, or cloud paths.
