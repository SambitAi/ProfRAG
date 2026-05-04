from __future__ import annotations

from pathlib import Path

from core.retrieval_engine import run_retrieval


def run(
    question: str,
    document_folder: str | Path,
    chroma_persist_dir: str | Path,
    collection_name: str,
    embedding_model: str,
    retrieval_output_path: str | Path,
    top_k: int,
    media_top_k: int = 4,
    expand_parent: bool = True,
    chroma_host: str | None = None,
    chroma_port: int = 8000,
) -> str:
    return run_retrieval(
        question=question,
        document_folder=Path(document_folder),
        chroma_persist_dir=chroma_persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        retrieval_output_path=retrieval_output_path,
        top_k=top_k,
        media_top_k=media_top_k,
        expand_parent=expand_parent,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
    )
