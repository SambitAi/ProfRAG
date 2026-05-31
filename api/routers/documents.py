from __future__ import annotations

import base64
from pathlib import Path
import shutil
import threading
import time

from fastapi import APIRouter, Header, HTTPException, Query

import workflows
from core.config import load_app_config
from api.deps import get_artifacts_root, get_config_path, require_existing_document_folder
from api.schemas.documents import (
    DocumentInspectResponse,
    DocumentStatusResponse,
    IngestUrlRequest,
    SummaryResetRequest,
    UploadDocumentRequest,
)
from core.global_index import delete_global_index_entry, global_index_path
from core.job_store import create_job, find_job_by_idempotency_key, list_jobs, update_job


router = APIRouter(prefix="/documents", tags=["documents"])


def _run_job(job_id: str, fn, *args) -> None:
    artifacts_root = get_artifacts_root()
    try:
        update_job(artifacts_root, job_id, state="running")
        result = fn(*args)
        update_job(artifacts_root, job_id, state="success", result={"metadata": result if isinstance(result, dict) else {}})
    except Exception as exc:
        update_job(
            artifacts_root,
            job_id,
            state="error",
            error={"code": "workflow_error", "message": str(exc)},
            retryable=True,
        )


def _run_summary_job(job_id: str, document_folder: str, starter_fn, *starter_args) -> None:
    artifacts_root = get_artifacts_root()
    timeout_seconds = 60 * 60
    deadline = time.time() + timeout_seconds
    try:
        update_job(artifacts_root, job_id, state="running")
        starter_fn(*starter_args)

        # Wait until summary pipeline reaches terminal state.
        while True:
            metadata = workflows.load_document(document_folder)
            status = str(metadata.get("summary_status", "pending"))
            if status == "ready" or bool(metadata.get("summary_ready", False)):
                update_job(
                    artifacts_root,
                    job_id,
                    state="success",
                    result={
                        "summary_status": status,
                        "summary_ready": bool(metadata.get("summary_ready", False)),
                    },
                )
                return
            if status == "error":
                update_job(
                    artifacts_root,
                    job_id,
                    state="error",
                    error={"code": "summary_failed", "message": "Summary generation failed"},
                    retryable=True,
                )
                return
            if time.time() >= deadline:
                update_job(
                    artifacts_root,
                    job_id,
                    state="error",
                    error={"code": "summary_timeout", "message": f"Summary job timed out after {timeout_seconds}s"},
                    retryable=True,
                )
                return
            time.sleep(1.0)
    except Exception as exc:
        update_job(
            artifacts_root,
            job_id,
            state="error",
            error={"code": "workflow_error", "message": str(exc)},
            retryable=True,
        )


@router.get("")
def list_documents() -> list[dict]:
    return workflows.list_documents(get_config_path())


# IMPORTANT: keep static routes above parameterized routes to avoid path capture.
# `/documents/inspect`, `/documents/upload`, and `/documents/ingest-url`
# must be registered before `/documents/{folder}`.
@router.get("/inspect", response_model=DocumentInspectResponse)
def inspect_document(file_name: str = Query(..., min_length=1)) -> DocumentInspectResponse:
    metadata = workflows.inspect_same_name_document(get_config_path(), file_name)
    return DocumentInspectResponse(exists=metadata is not None, metadata=metadata)


@router.post("/upload")
def upload_document(
    req: UploadDocumentRequest,
    x_idempotency_key: str | None = Header(default=None),
    x_correlation_id: str | None = Header(default=None),
) -> dict:
    try:
        file_bytes = base64.b64decode(req.file_content_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 file_content_b64")

    artifacts_root = get_artifacts_root()
    existing = find_job_by_idempotency_key(artifacts_root, "documents.upload", x_idempotency_key or "")
    if existing:
        return {"job_id": existing["job_id"], "state": existing["state"]}
    job = create_job(
        artifacts_root,
        "documents.upload",
        payload={"file_name": req.file_name, "user_choice": req.user_choice, "idempotency_key": x_idempotency_key or ""},
        correlation_id=x_correlation_id or "",
    )
    threading.Thread(
        target=_run_job,
        args=(job["job_id"], workflows.prepare_document, get_config_path(), req.file_name, file_bytes, req.user_choice),
        daemon=True,
    ).start()
    return {"job_id": job["job_id"], "state": job["state"]}


@router.post("/ingest-url")
def ingest_url(
    req: IngestUrlRequest,
    x_idempotency_key: str | None = Header(default=None),
    x_correlation_id: str | None = Header(default=None),
) -> dict:
    artifacts_root = get_artifacts_root()
    existing = find_job_by_idempotency_key(artifacts_root, "documents.ingest_url", x_idempotency_key or "")
    if existing:
        return {"job_id": existing["job_id"], "state": existing["state"]}
    job = create_job(
        artifacts_root,
        "documents.ingest_url",
        payload={"url": str(req.url), "user_choice": req.user_choice, "idempotency_key": x_idempotency_key or ""},
        correlation_id=x_correlation_id or "",
    )
    threading.Thread(
        target=_run_job,
        args=(job["job_id"], workflows.prepare_url_document, get_config_path(), str(req.url), req.user_choice),
        daemon=True,
    ).start()
    return {"job_id": job["job_id"], "state": job["state"]}


@router.get("/{folder}")
def get_document(folder: str) -> dict:
    document_folder = require_existing_document_folder(folder)
    metadata = workflows.load_document(document_folder)
    return metadata


@router.get("/{folder}/status", response_model=DocumentStatusResponse)
def get_document_status(folder: str) -> DocumentStatusResponse:
    document_folder = require_existing_document_folder(folder)
    metadata = workflows.load_document(document_folder)

    return DocumentStatusResponse(
        folder_name=folder,
        document_name=metadata.get("document_name", folder),
        last_successful_step=metadata.get("last_successful_step", "unknown"),
        ready_to_chat=bool(metadata.get("ready_to_chat", False)),
        summary_ready=bool(metadata.get("summary_ready", False)),
        summary_status=str(metadata.get("summary_status", "pending")),
        total_chunks=int(metadata.get("total_chunks", 0)),
    )


@router.get("/{folder}/summaries/status")
def get_summaries_status(folder: str) -> dict:
    document_folder = require_existing_document_folder(folder)
    metadata = workflows.load_document(document_folder)

    return {
        "folder_name": folder,
        "summary_ready": bool(metadata.get("summary_ready", False)),
        "summary_status": metadata.get("summary_status", "pending"),
        "summary_progress": metadata.get("summary_progress", {}),
        "summary_paths": metadata.get("summary_paths", {}),
    }


@router.get("/{folder}/summaries")
def get_summaries(folder: str) -> dict:
    document_folder = require_existing_document_folder(folder)
    metadata = workflows.load_document(document_folder)
    summaries_dir = Path(document_folder) / "summaries"
    level1 = (summaries_dir / "level1_onepager.json")
    level2 = (summaries_dir / "level2_medium.json")
    level3 = (summaries_dir / "level3_detailed.json")
    return {
        "folder_name": folder,
        "summary_ready": bool(metadata.get("summary_ready", False)),
        "summary_status": metadata.get("summary_status", "pending"),
        "levels": {
            "level1": {"exists": level1.exists(), "path": str(level1)},
            "level2": {"exists": level2.exists(), "path": str(level2)},
            "level3": {"exists": level3.exists(), "path": str(level3)},
        },
    }


@router.get("/{folder}/summaries/{level}")
def get_summary_level(folder: str, level: str) -> dict:
    if level not in {"level1", "level2", "level3"}:
        raise HTTPException(status_code=400, detail="Invalid summary level")
    document_folder = require_existing_document_folder(folder)
    summaries_dir = Path(document_folder) / "summaries"
    file_map = {
        "level1": summaries_dir / "level1_onepager.json",
        "level2": summaries_dir / "level2_medium.json",
        "level3": summaries_dir / "level3_detailed.json",
    }
    target = file_map[level]
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Summary not found for {level}")
    from core.storage import read_json
    return {"folder_name": folder, "level": level, "payload": read_json(target, default={})}


@router.post("/{folder}/summaries/start")
def start_summaries(
    folder: str,
    x_idempotency_key: str | None = Header(default=None),
    x_correlation_id: str | None = Header(default=None),
) -> dict:
    document_folder = require_existing_document_folder(folder)
    artifacts_root = get_artifacts_root()
    existing = find_job_by_idempotency_key(artifacts_root, "summaries.start", x_idempotency_key or "")
    if existing:
        return {"job_id": existing["job_id"], "state": existing["state"]}
    job = create_job(
        artifacts_root,
        "summaries.start",
        payload={"folder": folder, "idempotency_key": x_idempotency_key or ""},
        correlation_id=x_correlation_id or "",
    )
    threading.Thread(
        target=_run_summary_job,
        args=(job["job_id"], str(document_folder), workflows.start_summarization_background, get_config_path(), str(document_folder)),
        daemon=True,
    ).start()
    return {"job_id": job["job_id"], "state": job["state"]}


@router.post("/{folder}/summaries/reset")
def reset_summaries(
    folder: str,
    req: SummaryResetRequest,
    x_idempotency_key: str | None = Header(default=None),
    x_correlation_id: str | None = Header(default=None),
) -> dict:
    document_folder = require_existing_document_folder(folder)
    artifacts_root = get_artifacts_root()
    existing = find_job_by_idempotency_key(artifacts_root, "summaries.reset", x_idempotency_key or "")
    if existing:
        return {"job_id": existing["job_id"], "state": existing["state"]}
    job = create_job(
        artifacts_root,
        "summaries.reset",
        payload={"folder": folder, "level": req.level, "idempotency_key": x_idempotency_key or ""},
        correlation_id=x_correlation_id or "",
    )
    threading.Thread(
        target=_run_summary_job,
        args=(
            job["job_id"],
            str(document_folder),
            workflows.reset_summary_level,
            get_config_path(),
            str(document_folder),
            req.level,
        ),
        daemon=True,
    ).start()
    return {"job_id": job["job_id"], "state": job["state"]}


@router.delete("/{folder}")
def delete_document(folder: str) -> dict:
    document_folder = require_existing_document_folder(folder)
    artifacts_root = get_artifacts_root()
    active_jobs = list_jobs(artifacts_root)
    for job in active_jobs:
        state = str(job.get("state", ""))
        if state not in {"pending", "running"}:
            continue
        payload = job.get("payload", {})
        payload_folder = str(payload.get("folder", ""))
        if payload_folder and payload_folder == folder:
            raise HTTPException(status_code=409, detail="Document has active jobs; retry after completion.")
    config = load_app_config(get_config_path())
    delete_global_index_entry(global_index_path(config), document_folder)
    shutil.rmtree(document_folder)
    return {"deleted": True, "folder": folder}
