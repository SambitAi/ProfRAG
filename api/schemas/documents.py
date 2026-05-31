from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl
from pydantic import BaseModel, Field


class DocumentStatusResponse(BaseModel):
    folder_name: str
    document_name: str
    last_successful_step: str
    ready_to_chat: bool
    summary_ready: bool
    summary_status: str
    total_chunks: int


class DocumentInspectResponse(BaseModel):
    exists: bool
    metadata: dict | None = None


class UploadDocumentRequest(BaseModel):
    file_name: str = Field(min_length=1)
    file_content_b64: str = Field(min_length=1)
    user_choice: Literal["reuse", "reprocess", "new_version", "ignore"] = "reuse"


class IngestUrlRequest(BaseModel):
    url: AnyHttpUrl
    user_choice: Literal["reuse", "reprocess", "new_version", "ignore"] = "reuse"


class SummaryResetRequest(BaseModel):
    level: str = Field(pattern="^level[123]$")
