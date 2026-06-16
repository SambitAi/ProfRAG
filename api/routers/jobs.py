from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_user_artifacts_root
from core.job_store import get_job, list_jobs
from api.schemas.jobs import JobResponse


router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobResponse])
def list_jobs_route(
    artifacts_root: str = Depends(get_user_artifacts_root),
    status: str | None = Query(default=None),
    job_type: str | None = Query(default=None),
) -> list[JobResponse]:
    try:
        jobs = list_jobs(artifacts_root, status_filter=status, job_type_filter=job_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return [JobResponse(**job) for job in jobs]


@router.get("/{job_id}", response_model=JobResponse)
def get_job_route(
    job_id: str,
    artifacts_root: str = Depends(get_user_artifacts_root),
) -> JobResponse:
    job = get_job(artifacts_root, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return JobResponse(**job)
