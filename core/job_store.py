from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid
import logging

from core.locks import job_store_lock
from core.paths import ensure_directory
from core.storage import read_json, write_json_atomic


VALID_JOB_STATES = {"pending", "running", "success", "error"}
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jobs_dir(artifacts_root: str | Path) -> Path:
    return ensure_directory(Path(artifacts_root) / "jobs")


def _job_path(artifacts_root: str | Path, job_id: str) -> Path:
    return _jobs_dir(artifacts_root) / f"{job_id}.json"


def _validate_state(state: str) -> None:
    if state not in VALID_JOB_STATES:
        raise ValueError(f"Invalid job state '{state}'. Expected one of: {sorted(VALID_JOB_STATES)}")


def create_job(
    artifacts_root: str | Path,
    job_type: str,
    payload: dict[str, Any] | None = None,
    initial_state: str = "pending",
    correlation_id: str = "",
) -> dict[str, Any]:
    _validate_state(initial_state)
    job_id = uuid.uuid4().hex
    now = _now_iso()
    job = {
        "job_id": job_id,
        "job_type": job_type,
        "state": initial_state,
        "created_at": now,
        "started_at": now if initial_state == "running" else "",
        "completed_at": now if initial_state in {"success", "error"} else "",
        "updated_at": now,
        "payload": payload or {},
        "result": {},
        "error": {},
        "retryable": False,
        "correlation_id": correlation_id,
    }
    with job_store_lock(artifacts_root):
        write_json_atomic(_job_path(artifacts_root, job_id), job)
    logger.info("job_create", extra={"job_id": job_id, "job_type": job_type, "state": initial_state})
    return job


def update_job(
    artifacts_root: str | Path,
    job_id: str,
    *,
    state: str | None = None,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    retryable: bool | None = None,
    payload_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with job_store_lock(artifacts_root):
        path = _job_path(artifacts_root, job_id)
        current = read_json(path, default={})
        if not current:
            raise FileNotFoundError(f"Job not found: {job_id}")

        now = _now_iso()
        if state is not None:
            _validate_state(state)
            previous_state = str(current.get("state", ""))
            current["state"] = state
            if state == "running" and not current.get("started_at"):
                current["started_at"] = now
            if state in {"success", "error"}:
                if not current.get("started_at") and previous_state == "pending":
                    current["started_at"] = now
                current["completed_at"] = now
        if result is not None:
            current["result"] = result
        if error is not None:
            current["error"] = error
        if retryable is not None:
            current["retryable"] = bool(retryable)
        if payload_patch:
            payload = current.setdefault("payload", {})
            payload.update(payload_patch)

        current["updated_at"] = now
        write_json_atomic(path, current)
        logger.info(
            "job_update",
            extra={
                "job_id": job_id,
                "job_type": current.get("job_type", ""),
                "state": current.get("state", ""),
                "retryable": current.get("retryable", False),
            },
        )
        return current


def get_job(artifacts_root: str | Path, job_id: str) -> dict[str, Any] | None:
    path = _job_path(artifacts_root, job_id)
    data = read_json(path, default={})
    return data or None


def list_jobs(
    artifacts_root: str | Path,
    status_filter: str | None = None,
    job_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    if status_filter is not None:
        _validate_state(status_filter)

    jobs: list[dict[str, Any]] = []
    for path in _jobs_dir(artifacts_root).glob("*.json"):
        data = read_json(path, default={})
        if not data:
            continue
        if status_filter is not None and data.get("state") != status_filter:
            continue
        if job_type_filter is not None and data.get("job_type") != job_type_filter:
            continue
        jobs.append(data)

    return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)


def find_job_by_idempotency_key(
    artifacts_root: str | Path,
    job_type: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    if not idempotency_key:
        return None
    for job in list_jobs(artifacts_root, job_type_filter=job_type):
        payload = job.get("payload", {})
        if payload.get("idempotency_key") == idempotency_key and job.get("state") in {"pending", "running", "success"}:
            return job
    return None


def recover_inflight_jobs(artifacts_root: str | Path) -> int:
    recovered = 0
    for job in list_jobs(artifacts_root):
        state = str(job.get("state", ""))
        if state in {"pending", "running"}:
            update_job(
                artifacts_root,
                str(job["job_id"]),
                state="error",
                error={"code": "job_interrupted", "message": "Job interrupted by process restart"},
                retryable=True,
            )
            recovered += 1
    return recovered
