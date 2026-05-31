from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: str
    message: str
    retryable: bool = False
    context: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    service: str


class JobStatus(BaseModel):
    state: str
