from __future__ import annotations

from pydantic import BaseModel


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    state: str
    created_at: str
    started_at: str = ""
    completed_at: str = ""
    updated_at: str
    payload: dict
    result: dict
    error: dict
    retryable: bool
    correlation_id: str = ""

