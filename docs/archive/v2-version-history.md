# v2 Version History (Canonical Referral)

This document is the single reference summary for the v2 work done in this folder. It replaces the need to read the separate execution notes in docs/v2 and docs/v2/archive once the implementation and validation path are understood.

Use this file as the cleanup reference for deleting the older v2 execution documents after the migration is complete and verified.

---

## What v2 was trying to achieve

v2 aimed to make the system:
- more reliable for multi-document retrieval,
- easier to operate through a real API layer,
- and cleaner to maintain by moving orchestration and infrastructure responsibilities into dedicated modules.

The main workstreams were:
1. API-layer delivery and operational hardening,
2. workflow/main.py cleanup and ownership separation,
3. retrieval correctness and version-unique identity fixes,
4. staged validation and migration planning.

---

## 1. API layer: the v2 delivery surface

- Objective:
  Make the system usable from non-UI clients and support automation, job tracking, and production-style operations.

- What changed:
  - Added a FastAPI-based v2 API surface for health, documents, summaries, chat, and jobs.
  - Added request/response schemas, middleware for auth, rate limits, logging, and request-size control.
  - Added job-based async execution so upload, ingest, and summarization flows could be tracked and retried.
  - Added a deployment and rollback runbook for startup, health checks, and failure triage.

- Why it was done:
  - The original UI-only path was not enough for automation or production operations.
  - Long-running tasks needed a stable lifecycle model instead of synchronous-only execution.

- Where the work landed:
  - docs/v2/API_RUNBOOK.md
  - docs/v2/archive/api_layer_refactor_summary.md
  - docs/v2/archive/api_layer_tasks_checklist.md
  - api/app.py
  - api/routers/*
  - api/schemas/*
  - api/deps.py
  - api/middleware.py
  - core/job_store.py

---

## 2. Workflow and main.py cleanup

- Objective:
  Centralize orchestration logic and reduce coupling between UI entry points and production workflow behavior.

- What changed:
  - Moved rendering helpers into services/image_render.py.
  - Moved infra helpers into core/global_index.py and core/paths.py.
  - Introduced workflows.py as the primary orchestration layer.
  - Reduced main.py to a thin compatibility wrapper.

- Why it was done:
  - The old layout mixed UI orchestration, retrieval logic, and infra concerns.
  - A cleaner ownership model was needed for both UI stability and API readiness.

- Where the work landed:
  - docs/v2/archive/cleanup_main.md
  - docs/v2/archive/worklow.md
  - workflows.py
  - core/global_index.py
  - core/paths.py
  - services/image_render.py
  - main.py

---

## 3. Retrieval fix: versioned documents and multi-doc coverage

- Objective:
  Fix the v2 regression where multi-document retrieval could collapse to one dominant document or fail to distinguish versions of the same source.

- What changed:
  - Replaced hash-based document identity with version-unique folder-derived IDs such as slug_v1 and slug_v2.
  - Reworked tree retrieval so selection mode walks each document independently instead of relying on global section bias.
  - Kept discovery mode for broad routing questions, but routed it through the same per-document retrieval path after candidate selection.
  - Added retrieval config keys for per-doc section/chunk behavior and output limits.

- Why it was done:
  - v2 introduced identity collisions across versions and biased global section retrieval.
  - The v1.1 per-document traversal model was needed to restore reliable multi-doc coverage.

- Where the work landed:
  - docs/v2/multi_doc_retrieval_rewrite.md
  - docs/v2/archive/multi_doc_retrieval_refactor_summary.md
  - docs/v2/archive/multi_doc_retrieval_rca_fix_checklist.md
  - core/tree_retrieval.py
  - core/paths.py
  - services/write_to_vector_db.py
  - services/metadata_cards.py
  - services/summarize_document.py
  - services/pdf_upload.py
  - services/web_upload.py
  - core/global_index.py

---

## 4. Validation and migration status

- Objective:
  Prove the changes are correct after data migration and re-ingestion.

- What changed:
  - Validation plans were documented for document identity, selected-doc coverage, multi-doc quality, and regression checks.
  - The migration path was staged into identity migration, re-ingest, and validation gates.

- Why it matters:
  - The retrieval and identity changes are high-risk and require real corpus validation, not just code inspection.
  - The current environment still has auth/network blockers for full re-ingest and post-migration verification.

- Where the validation notes live:
  - docs/v2/multi_doc_retrieval_rewrite.md
  - docs/v2/archive/multi_doc_retrieval_rca_fix_checklist.md
  - docs/v2/archive/multi_doc_retrieval_refactor_summary.md

---

## 5. Cleanup guidance for the v2 execution docs

The execution-style notes in this folder are historical working documents. This file is the canonical referral record for the actual v2 work.

When the migration and validation cycle is complete, the older execution notes can be removed because the important decisions, scope, and outcomes are already captured here:
- docs/v2/API_RUNBOOK.md
- docs/v2/multi_doc_retrieval_rewrite.md
- docs/v2/archive/*

This file should remain as the durable handoff document for anyone reviewing or deleting the v2 execution artifacts.
