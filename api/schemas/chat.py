from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class SourceItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    chunk_number: int | None = None
    document_folder: str | None = None
    chunk_path: str | None = None
    section_path: str | None = None
    score: float | None = None


class SingleChatRequest(BaseModel):
    document_folder: str = Field(min_length=1)
    question: str = Field(min_length=1)


class MultiChatRequest(BaseModel):
    document_folders: list[str] = Field(min_length=1)
    question: str = Field(min_length=1)


class FindRelevantRequest(BaseModel):
    question: str = Field(min_length=1)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    question: str
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    abstain: bool = False
    confidence: str = ""
    mode: str = ""
    document_name: str = ""


class MultiChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    question: str
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    abstain: bool = False
    confidence: str = ""
    mode: str = ""


class RelevantDocument(BaseModel):
    model_config = ConfigDict(extra="allow")
    folder: str | None = None
    folder_name: str | None = None
    document_folder: str | None = None
    document_name: str | None = None
    score: float | None = None
