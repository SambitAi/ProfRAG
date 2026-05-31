from __future__ import annotations

import threading

from fastapi import APIRouter

import workflows
from api.deps import get_artifacts_root, get_config_path
from core.job_store import create_job, update_job


router = APIRouter(prefix="/summaries", tags=["summaries"])


def _run_watcher_job(job_id: str) -> None:
    artifacts_root = get_artifacts_root()
    try:
        update_job(artifacts_root, job_id, state="running")
        workflows.start_summary_watcher(get_config_path())
        update_job(artifacts_root, job_id, state="success", result={"started": True})
    except Exception as exc:
        update_job(
            artifacts_root,
            job_id,
            state="error",
            error={"code": "watcher_start_failed", "message": str(exc)},
            retryable=True,
        )


@router.post("/watcher/start")
def start_watcher() -> dict:
    artifacts_root = get_artifacts_root()
    job = create_job(artifacts_root, "summaries.watcher.start")
    threading.Thread(target=_run_watcher_job, args=(job["job_id"],), daemon=True).start()
    return {"job_id": job["job_id"], "state": job["state"]}

