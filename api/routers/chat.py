from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

import workflows
from api.deps import get_config_path, get_user_artifacts_root, get_user_key, resolve_document_folder
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
def chat_single(
    req: SingleChatRequest,
    artifacts_root: str = Depends(get_user_artifacts_root),
    user_key: str = Depends(get_user_key),
) -> ChatResponse:
    folder = resolve_document_folder(req.document_folder, artifacts_root)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {req.document_folder}")
    payload = workflows.ask_question(get_config_path(), str(folder), req.question, artifacts_root=artifacts_root, user_key=user_key)
    return ChatResponse(**payload)


@router.post("/multi", response_model=MultiChatResponse)
def chat_multi(
    req: MultiChatRequest,
    artifacts_root: str = Depends(get_user_artifacts_root),
    user_key: str = Depends(get_user_key),
) -> MultiChatResponse:
    resolved_folders = [(folder, resolve_document_folder(folder, artifacts_root)) for folder in req.document_folders]
    missing = [raw for raw, resolved in resolved_folders if not resolved.exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Documents not found: {missing}")
    folders = [str(resolved) for _, resolved in resolved_folders]
    payload = workflows.ask_multi_document_question(get_config_path(), folders, req.question, artifacts_root=artifacts_root, user_key=user_key)
    return MultiChatResponse(**payload)


@router.post("/find-relevant", response_model=list[RelevantDocument])
def find_relevant(
    req: FindRelevantRequest,
    artifacts_root: str = Depends(get_user_artifacts_root),
    user_key: str = Depends(get_user_key),
) -> list[RelevantDocument]:
    docs = workflows.find_relevant_documents(get_config_path(), req.question, artifacts_root, user_key=user_key)
    return [RelevantDocument(**doc) for doc in docs]
