from __future__ import annotations

from fastapi import APIRouter, HTTPException

import workflows
from api.deps import get_config_path, resolve_document_folder
from api.schemas.chat import (
    ChatResponse,
    FindRelevantRequest,
    MultiChatRequest,
    MultiChatResponse,
    RelevantDocument,
    SingleChatRequest,
)


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/single", response_model=ChatResponse)
def chat_single(req: SingleChatRequest) -> ChatResponse:
    folder = resolve_document_folder(req.document_folder)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {req.document_folder}")
    payload = workflows.ask_question(get_config_path(), str(folder), req.question)
    return ChatResponse(**payload)


@router.post("/multi", response_model=MultiChatResponse)
def chat_multi(req: MultiChatRequest) -> MultiChatResponse:
    resolved_folders = [(folder, resolve_document_folder(folder)) for folder in req.document_folders]
    missing = [raw for raw, resolved in resolved_folders if not resolved.exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Documents not found: {missing}")
    folders = [str(resolved) for _, resolved in resolved_folders]
    payload = workflows.ask_multi_document_question(get_config_path(), folders, req.question)
    return MultiChatResponse(**payload)


@router.post("/find-relevant", response_model=list[RelevantDocument])
def find_relevant(req: FindRelevantRequest) -> list[RelevantDocument]:
    docs = workflows.find_relevant_documents(get_config_path(), req.question)
    return [RelevantDocument(**doc) for doc in docs]
