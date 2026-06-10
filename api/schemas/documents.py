from __future__ import annotations

from typing import Annotated
from typing import Literal

from pydantic import AnyHttpUrl
from pydantic import BaseModel, Field
from pydantic import StringConstraints


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


NonEmptyFolderName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DeleteDocumentsRequest(BaseModel):
    folders: list[NonEmptyFolderName] = Field(min_length=1)


class DeleteDocumentsCollectionDetail(BaseModel):
    collection_name: str
    status: str
    selector: str
    matched_ids: int
    delete_mode: str


class DeleteDocumentsFolderDetail(BaseModel):
    folder: str
    vector_cleanup: list[DeleteDocumentsCollectionDetail] = Field(default_factory=list)
    supports_where_delete: bool = False
    global_index_removed: bool = False
    filesystem_removed: bool = False


class DeleteDocumentsResponse(BaseModel):
    deleted: bool
    folders: list[str]
    collections_cleaned: list[str]
    global_index_removed: list[str]
    details: list[DeleteDocumentsFolderDetail] = Field(default_factory=list)


class UploadDocumentRequest(BaseModel):
    file_name: str = Field(min_length=1)
    file_content_b64: str = Field(min_length=1)
    user_choice: Literal["reuse", "reprocess", "new_version", "ignore"] = "reuse"


class IngestUrlRequest(BaseModel):
    url: AnyHttpUrl
    user_choice: Literal["reuse", "reprocess", "new_version", "ignore"] = "reuse"


class SummaryResetRequest(BaseModel):
    level: str = Field(pattern="^level[123]$")
