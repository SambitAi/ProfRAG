from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import load_app_config
from core.metadata import load_metadata
from core.paths import ensure_directory, slugify_filename
from core.storage import read_json
from services import chat_response, extract_images, extract_tables, pdf_upload, retrieve_context, write_to_vector_db
from services import url_ingest, web_upload
from services import multi_doc_query


def _config(config_path: str | Path) -> dict[str, Any]:
    return load_app_config(config_path)


def _get_pdf_extractor(config: dict[str, Any]) -> Any:
    if config.get("pdf", {}).get("parser") == "pymupdf4llm":
        from services import pdf_to_markdown_pymupdf4llm as m
    else:
        from services import pdf_to_markdown as m
    return m


def _get_chunker(config: dict[str, Any]) -> Any:
    if config.get("chunking", {}).get("strategy") == "section_aware":
        from services import markdown_chunker_section_aware as m
    else:
        from services import markdown_chunker as m
    return m


def read_chunk(chunk_path: str | Path) -> dict[str, Any]:
    return read_json(chunk_path, default={})


def read_prev_chunk(document_folder: str | Path, chunk_number: int) -> dict[str, Any]:
    """Return the chunk JSON for chunk N-1, or {} if at start or folder unknown."""
    if not document_folder or not chunk_number:
        return {}
    metadata = load_metadata(document_folder)
    chunk_paths = metadata.get("chunk_paths", [])
    prev_idx = chunk_number - 2  # chunk_number is 1-indexed; prev chunk is at index chunk_number-2
    if 0 <= prev_idx < len(chunk_paths):
        return read_json(chunk_paths[prev_idx], default={})
    return {}


def read_next_chunk(document_folder: str | Path, chunk_number: int) -> dict[str, Any]:
    """Return the chunk JSON for chunk N+1, or {} if at end or folder unknown."""
    if not document_folder or not chunk_number:
        return {}
    metadata = load_metadata(document_folder)
    chunk_paths = metadata.get("chunk_paths", [])
    next_idx = chunk_number  # chunk_number is 1-indexed; next chunk is at index chunk_number
    if next_idx < len(chunk_paths):
        return read_json(chunk_paths[next_idx], default={})
    return {}


def _artifact_paths(document_folder: str | Path, question_number: int) -> dict[str, Path]:
    folder = Path(document_folder)
    return {
        "markdown": folder / "markdown" / "document.md",
        "chunks": folder / "chunks",
        "vector_result": folder / "vector" / "index_result.json",
        "retrieval": folder / "retrieval" / f"query_{question_number:06d}.json",
        "chat": folder / "chat" / f"query_{question_number:06d}.json",
    }


def _find_document_folders(artifacts_root: str | Path) -> list[Path]:
    root = Path(artifacts_root)
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda item: item.name)


def list_documents(config_path: str | Path) -> list[dict[str, Any]]:
    config = _config(config_path)
    artifacts_root = config["paths"]["artifacts_root"]
    documents: list[dict[str, Any]] = []

    for folder in _find_document_folders(artifacts_root):
        metadata_path = folder / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = read_json(metadata_path, default={})
        documents.append(
            {
                "folder_name": folder.name,
                "document_folder": str(folder),
                "document_name": metadata.get("document_name", folder.name),
                "last_successful_step": metadata.get("last_successful_step", "unknown"),
                "ready_to_chat": metadata.get("ready_to_chat", False),
                "total_chunks": metadata.get("total_chunks", 0),
                "summary_ready": metadata.get("summary_ready", False),
                "summary_status": metadata.get("summary_status", "pending"),
            }
        )

    return sorted(documents, key=lambda item: item["folder_name"], reverse=True)


def _find_latest_same_name_document(artifacts_root: str | Path, file_name: str) -> Path | None:
    slug = slugify_filename(file_name)
    matches: list[Path] = []
    for folder in _find_document_folders(artifacts_root):
        metadata_path = folder / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = read_json(metadata_path, default={})
        if slugify_filename(metadata.get("document_name", "")) == slug:
            matches.append(folder)
    return sorted(matches, key=lambda item: item.name)[-1] if matches else None


def inspect_same_name_document(config_path: str | Path, file_name: str) -> dict[str, Any] | None:
    config = _config(config_path)
    existing_folder = _find_latest_same_name_document(config["paths"]["artifacts_root"], file_name)
    if existing_folder is None:
        return None
    return load_metadata(existing_folder)


def _run_pipeline_from_metadata(config_path: str | Path, document_folder: str | Path) -> dict[str, Any]:
    config = _config(config_path)
    metadata = load_metadata(document_folder)
    outputs = _artifact_paths(document_folder, 1)
    last_step = metadata.get("last_successful_step", "created")

    if last_step == "upload":
        _get_pdf_extractor(config).run(metadata["source_pdf_path"], outputs["markdown"], document_folder)
        last_step = "pdf_to_markdown"

    extraction_cfg = config.get("extraction", {})

    if last_step in ("pdf_to_markdown", "extract_images", "extract_tables"):
        if extraction_cfg.get("images", True):
            extract_images.run(outputs["markdown"], document_folder)
        if extraction_cfg.get("tables", True):
            extract_tables.run(outputs["markdown"], document_folder)
        last_step = "extract_assets"

    if last_step == "extract_assets":
        _get_chunker(config).run(
            outputs["markdown"],
            outputs["chunks"],
            document_folder,
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"],
        )
        last_step = "markdown_chunker"

    if last_step == "markdown_chunker":
        metadata = load_metadata(document_folder)
        # Start summarization before vectorisation — both only need chunk_paths,
        # so the daemon runs concurrently while write_to_vector_db blocks.
        start_summarization_background(config_path, document_folder)
        write_to_vector_db.run(
            chunk_paths=metadata["chunk_paths"],
            document_folder=document_folder,
            chroma_persist_dir=config["vector_db"]["persist_directory"],
            collection_name=config["vector_db"]["collection_name"],
            embedding_model=config["embeddings"]["model"],
            index_result_path=outputs["vector_result"],
        )
        return load_metadata(document_folder)

    # Reached when the document was already vectorised but summarisation is incomplete.
    # Always restart: the daemon may have been killed (status stays "in_progress" on restart)
    # and summarize_document.run() is idempotent — it skips already-completed checkpoints.
    final_metadata = load_metadata(document_folder)
    if not final_metadata.get("summary_ready"):
        start_summarization_background(config_path, document_folder)
    return final_metadata


def prepare_document(
    config_path: str | Path,
    file_name: str,
    file_bytes: bytes,
    user_choice: str,
) -> dict[str, Any]:
    config = _config(config_path)
    artifacts_root = config["paths"]["artifacts_root"]
    ensure_directory(artifacts_root)
    existing_folder = _find_latest_same_name_document(artifacts_root, file_name)

    version = None
    if existing_folder is not None:
        existing_metadata = load_metadata(existing_folder)
        if existing_metadata.get("ready_to_chat") and user_choice == "reuse":
            if not existing_metadata.get("summary_ready"):
                return _run_pipeline_from_metadata(config_path, existing_folder)
            return existing_metadata
        if not existing_metadata.get("ready_to_chat"):
            return _run_pipeline_from_metadata(config_path, existing_folder)
        version = int(existing_metadata["document_version"]) + 1

    document_folder = pdf_upload.save_uploaded_pdf(
        file_bytes=file_bytes,
        original_file_name=file_name,
        artifacts_root=artifacts_root,
        version=version,
    )
    return _run_pipeline_from_metadata(config_path, document_folder)


def inspect_same_url_document(config_path: str | Path, url: str) -> dict[str, Any] | None:
    return inspect_same_name_document(config_path, url_ingest.url_to_document_name(url))


def prepare_url_document(
    config_path: str | Path,
    url: str,
    user_choice: str,
) -> dict[str, Any]:
    from urllib.parse import urlparse as _urlparse

    document_name = url_ingest.url_to_document_name(url)
    config = _config(config_path)
    artifacts_root = config["paths"]["artifacts_root"]
    ensure_directory(artifacts_root)
    existing_folder = _find_latest_same_name_document(artifacts_root, document_name)

    # Early reuse check before any network request
    if existing_folder:
        existing_metadata = load_metadata(existing_folder)
        if existing_metadata.get("ready_to_chat") and user_choice == "reuse":
            if not existing_metadata.get("summary_ready"):
                return _run_pipeline_from_metadata(config_path, existing_folder)
            return existing_metadata
        if not existing_metadata.get("ready_to_chat"):
            return _run_pipeline_from_metadata(config_path, existing_folder)

    # PDF URL → download and run full PDF pipeline
    if url_ingest.is_pdf_url(url):
        body_bytes = url_ingest.fetch_pdf_bytes(url)
        path_segment = _urlparse(url).path.rstrip("/").split("/")[-1]
        file_name = path_segment or "document.pdf"
        if not file_name.lower().endswith(".pdf"):
            file_name += ".pdf"
        return prepare_document(config_path, file_name, body_bytes, user_choice)

    # HTML → scrape directly to markdown (no PDF conversion)
    markdown_text = url_ingest.scrape_url_to_markdown(url)

    version = None
    if existing_folder:
        version = int(load_metadata(existing_folder)["document_version"]) + 1

    document_folder = web_upload.save_scraped_web_page(
        markdown_text=markdown_text,
        document_name=document_name,
        artifacts_root=artifacts_root,
        source_url=url,
        version=version,
    )
    return _run_pipeline_from_metadata(config_path, document_folder)


def load_document(document_folder: str | Path) -> dict[str, Any]:
    return load_metadata(document_folder)


def find_relevant_documents(config_path: str | Path, question: str) -> list[dict[str, Any]]:
    config = _config(config_path)
    return multi_doc_query.find_relevant_documents(question, config)


def ask_multi_document_question(
    config_path: str | Path,
    document_folders: list[str],
    question: str,
) -> dict[str, Any]:
    config = _config(config_path)
    return multi_doc_query.ask_across_documents(question, document_folders, config)


def start_summarization_background(config_path: str | Path, document_folder: str | Path) -> None:
    """Non-blocking: marks in_progress and spawns a daemon thread for summarization."""
    import threading
    from core.metadata import update_summary_progress

    config = _config(config_path)
    update_summary_progress(document_folder, "started", True)

    def _run() -> None:
        from services import summarize_document
        try:
            summarize_document.run(document_folder, config)
        except Exception:
            pass  # errors are written to metadata by summarize_document.run()

    threading.Thread(target=_run, daemon=True).start()


def reset_summary_level(config_path: str | Path, document_folder: str | Path, level: str) -> None:
    """Clear checkpoints for *level* and all downstream levels, then restart summarization.

    level: "level1" | "level2" | "level3"
    Regenerating a level invalidates everything that was derived from it.
    """
    from core.metadata import load_metadata, save_metadata
    from core.storage import write_json

    folder = Path(document_folder)
    metadata = load_metadata(folder)
    progress = metadata.setdefault("summary_progress", {})

    _downstream: dict[str, list[str]] = {
        "level3": ["level3_sections_done", "level3_complete", "level3_indexed",
                   "level2_complete", "level1_complete", "level1_indexed"],
        "level2": ["level2_complete", "level1_complete", "level1_indexed"],
        "level1": ["level1_complete", "level1_indexed"],
    }
    for key in _downstream.get(level, []):
        progress.pop(key, None)

    metadata["summary_ready"] = False
    metadata["summary_status"] = "pending"
    save_metadata(folder, metadata)

    # Reset L3 file to empty so sections don't accumulate duplicates on re-run
    if level == "level3":
        write_json(folder / "summaries" / "level3_detailed.json", {"sections": []})

    start_summarization_background(config_path, folder)


def ask_question(config_path: str | Path, document_folder: str | Path, question: str) -> dict[str, Any]:
    config = _config(config_path)
    metadata = load_metadata(document_folder)
    question_count = len(list((Path(document_folder) / "chat").glob("query_*.json"))) + 1
    outputs = _artifact_paths(document_folder, question_count)
    retrieval_context_path = retrieve_context.run(
        question=question,
        document_folder=document_folder,
        chroma_persist_dir=config["vector_db"]["persist_directory"],
        collection_name=config["vector_db"]["collection_name"],
        embedding_model=config["embeddings"]["model"],
        retrieval_output_path=outputs["retrieval"],
        top_k=config["retrieval"]["top_k"],
        media_top_k=config["retrieval"].get("media_top_k", 4),
        expand_parent=config["retrieval"].get("expand_parent", True),
    )
    answer_path = chat_response.run(
        retrieval_input_path=retrieval_context_path,
        output_answer_path=outputs["chat"],
        document_folder=document_folder,
        chat_model=config["chat"]["model"],
        media_instruction=config.get("prompt", {}).get("media_instruction", ""),
    )
    answer_payload = read_json(answer_path, default={})
    answer_payload["document_name"] = metadata["document_name"]
    return answer_payload
