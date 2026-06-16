from __future__ import annotations

import threading

from fastapi import APIRouter, Depends

import workflows
from api.deps import get_config_path, get_user_artifacts_root
from core.job_store import create_job, update_job


router = APIRouter(prefix="/summaries", tags=["summaries"])


def _run_watcher_job(job_id: str, artifacts_root: str) -> None:
    try:
        update_job(artifacts_root, job_id, state="running")
        # Always start with the base config (no user scope) so the singleton
        # watcher scans iter_user_roots and serves every user's pending docs,
        # not just the caller's.
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
def start_watcher(
    artifacts_root: str = Depends(get_user_artifacts_root),
) -> dict:
    job = create_job(artifacts_root, "summaries.watcher.start")
    threading.Thread(target=_run_watcher_job, args=(job["job_id"], artifacts_root), daemon=True).start()
    return {"job_id": job["job_id"], "state": job["state"]}
