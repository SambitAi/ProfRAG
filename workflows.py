from __future__ import annotations

from pathlib import Path
import logging
import shutil
import threading
from time import perf_counter
from typing import Any

from core.config import load_app_config
from core.global_index import (
    build_global_entry,
    delete_global_index_entry,
    find_document_folders,
    find_latest_same_name_document,
    global_index_path,
    load_global_index,
    write_global_index_entry,
)
from core.job_store import list_jobs
from core.locks import document_lock
from core.metadata import load_metadata, save_metadata, update_summary_progress
from core.paths import artifact_paths, ensure_directory
from core.storage import read_json, write_json
from services import chat_response, extract_images, extract_tables, field_extractor, metadata_cards, pdf_upload, write_to_vector_db
from services import document_delete
from services import image_render
from services import markdown_chunker_section_aware as markdown_chunker
from services import multi_doc_query
from services import pdf_to_markdown_pymupdf4llm as pdf_to_markdown
from services import retrieve_context_tree
from services import summary_watcher
from services import url_ingest, web_upload

logger = logging.getLogger(__name__)


class DocumentDeletionError(Exception):
    """Base workflow-layer error for document deletion."""


class DocumentDeletionValidationError(DocumentDeletionError):
    """Raised when delete input is invalid."""


class DocumentDeletionNotFoundError(DocumentDeletionError):
    """Raised when a target document folder is missing or outside artifacts root."""


class DocumentDeletionConflictError(DocumentDeletionError):
    """Raised when a target document cannot be deleted due to active work."""


def _folder_cleanup_result(folder_name: str) -> dict[str, Any]:
    return {
        "folder": folder_name,
        "vector_cleanup": [],
        "supports_where_delete": False,
        "global_index_removed": False,
        "filesystem_removed": False,
    }


def _log_workflow_enter(workflow: str, stage: str, document_id: str = "") -> float:
    logger.info(
        "enter workflow",
        extra={"workflow": workflow, "stage": stage, "document_id": document_id, "status": "enter"},
    )
    return perf_counter()


def _log_workflow_exit(start_ts: float, workflow: str, stage: str, document_id: str = "", status: str = "ok") -> None:
    logger.info(
        "exit workflow",
        extra={
            "workflow": workflow,
            "stage": stage,
            "document_id": document_id,
            "status": status,
            "duration_ms": int((perf_counter() - start_ts) * 1000),
        },
    )


def list_documents(config_path: str | Path) -> list[dict[str, Any]]:
    config = load_app_config(config_path)
    artifacts_root = config["paths"]["artifacts_root"]
    artifacts_root_path = Path(artifacts_root)
    documents: list[dict[str, Any]] = []

    global_index = load_global_index(
        global_index_path(config),
        cache_ttl_seconds=float(config.get("global_index", {}).get("cache_ttl_seconds", 5)),
    )
    indexed_docs = global_index.get("documents", {}) if isinstance(global_index, dict) else {}
    indexed_keys: set[str] = set()

    for folder_name, entry in indexed_docs.items():
        if not isinstance(entry, dict):
            continue
        indexed_keys.add(folder_name)
        doc_folder = entry.get("document_folder") or str(artifacts_root_path / folder_name)
        documents.append(
            {
                "folder_name": folder_name,
                "document_folder": doc_folder,
                "document_name": entry.get("document_name", folder_name),
                "last_successful_step": entry.get("last_successful_step", "unknown"),
                "ready_to_chat": entry.get("ready_to_chat", False),
                "total_chunks": entry.get("total_chunks", 0),
                "summary_ready": entry.get("summary_ready", False),
                "summary_status": entry.get("summary_status", "pending"),
            }
        )

    for folder in find_document_folders(artifacts_root):
        if folder.name in indexed_keys:
            continue
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


def _resolve_document_folder_for_delete(artifacts_root: str | Path, folder: str) -> Path:
    if not folder or not str(folder).strip():
        raise DocumentDeletionValidationError("Folder name is required.")

    root = Path(artifacts_root).resolve()
    resolved = (root / str(folder)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DocumentDeletionValidationError(f"Invalid folder: {folder}") from exc
    if not resolved.exists() or not resolved.is_dir():
        raise DocumentDeletionNotFoundError(f"Document folder not found: {folder}")
    return resolved


def _iter_active_jobs(artifacts_root: str | Path) -> list[dict[str, Any]]:
    return [
        job for job in list_jobs(artifacts_root)
        if str(job.get("state", "")) in {"pending", "running"}
    ]


def _is_documents_job_overlap(job: dict[str, Any], metadata_by_folder: dict[str, dict[str, Any]]) -> str | None:
    job_type = str(job.get("job_type", ""))
    if not job_type.startswith("documents."):
        return None

    payload = job.get("payload", {})
    file_name = str(payload.get("file_name", "")).strip()
    source_url = str(payload.get("url", "")).strip()

    for folder_name, metadata in metadata_by_folder.items():
        if not isinstance(metadata, dict) or not metadata:
            continue
        metadata_url = str(metadata.get("source_url", "")).strip()
        source_pdf_path = str(metadata.get("source_pdf_path", "") or "").strip()
        source_pdf_name = Path(source_pdf_path).name if source_pdf_path else ""
        if file_name and file_name == source_pdf_name:
            return folder_name
        if source_url and metadata_url and source_url == metadata_url:
            return folder_name
    return None


def _assert_no_active_folder_jobs(
    artifacts_root: str | Path,
    folders: set[str],
    metadata_by_folder: dict[str, dict[str, Any]],
) -> None:
    for job in _iter_active_jobs(artifacts_root):
        payload = job.get("payload", {})
        payload_folder = str(payload.get("folder", "")).strip()
        if payload_folder and payload_folder in folders:
            raise DocumentDeletionConflictError(
                f"Document '{payload_folder}' has active job '{job.get('job_type', '')}'; retry after completion."
            )

        overlapping_folder = _is_documents_job_overlap(job, metadata_by_folder)
        if overlapping_folder:
            raise DocumentDeletionConflictError(
                f"Document '{overlapping_folder}' has active job '{job.get('job_type', '')}'; retry after completion."
            )


def delete_documents(config_path: str | Path, folders: list[str]) -> dict[str, Any]:
    if not folders:
        raise DocumentDeletionValidationError("At least one folder is required.")

    config = load_app_config(config_path)
    artifacts_root = config["paths"]["artifacts_root"]
    unique_folders = list(dict.fromkeys(str(folder).strip() for folder in folders if str(folder).strip()))
    if not unique_folders:
        raise DocumentDeletionValidationError("At least one folder is required.")
    delete_run_id = ",".join(unique_folders)
    start_ts = _log_workflow_enter("documents", "delete_documents", delete_run_id)

    try:
        resolved_folders = [
            _resolve_document_folder_for_delete(artifacts_root, folder_name)
            for folder_name in unique_folders
        ]
        metadata_by_folder: dict[str, dict[str, Any]] = {}
        for folder_name, document_folder in zip(unique_folders, resolved_folders):
            try:
                metadata = load_metadata(document_folder)
            except Exception:
                logger.exception(
                    "document_delete_metadata_load_failed",
                    extra={"folder": folder_name, "path": str(document_folder)},
                )
                metadata = {}
            metadata_by_folder[folder_name] = metadata
        _assert_no_active_folder_jobs(artifacts_root, set(unique_folders), metadata_by_folder)

        index_path = global_index_path(config)
        collections_cleaned: list[str] = []
        global_index_removed: list[str] = []
        details: list[dict[str, Any]] = []

        for folder_name, document_folder in zip(unique_folders, resolved_folders):
            logger.info("document_delete_start", extra={"folder": folder_name, "path": str(document_folder)})
            metadata = metadata_by_folder.get(folder_name, {})
            document_id = str((metadata or {}).get("document_id", "")).strip()
            indexed_document_folder = str((metadata or {}).get("document_folder", "")).strip() or str(document_folder)
            folder_result = _folder_cleanup_result(folder_name)

            logger.info(
                "document_delete_vector_cleanup_start",
                extra={"folder": folder_name, "document_id": document_id, "document_folder": indexed_document_folder},
            )
            cleanup_result = document_delete.cleanup_document_vectors(
                config,
                document_folder=indexed_document_folder,
                document_id=document_id,
            )
            folder_result["vector_cleanup"] = list(cleanup_result.get("details", []))
            folder_result["supports_where_delete"] = bool(cleanup_result.get("supports_where_delete", False))
            for collection_name in cleanup_result.get("collections_cleaned", []):
                if collection_name not in collections_cleaned:
                    collections_cleaned.append(collection_name)
            logger.info(
                "document_delete_vector_cleanup_complete",
                extra={
                    "folder": folder_name,
                    "document_id": document_id,
                    "collections_cleaned": cleanup_result.get("collections_cleaned", []),
                    "details": cleanup_result.get("details", []),
                },
            )

            try:
                logger.info(
                    "document_delete_index_remove_start",
                    extra={"folder": folder_name, "index_path": str(index_path)},
                )
                delete_global_index_entry(index_path, document_folder)
                folder_result["global_index_removed"] = True
                global_index_removed.append(folder_name)
                logger.info("document_delete_index_removed", extra={"folder": folder_name, "index_path": str(index_path)})
            except Exception:
                logger.exception(
                    "document_delete_index_failed",
                    extra={"folder": folder_name, "index_path": str(index_path)},
                )
                raise

            try:
                logger.info("document_delete_files_remove_start", extra={"folder": folder_name, "path": str(document_folder)})
                shutil.rmtree(document_folder)
                folder_result["filesystem_removed"] = True
                logger.info("document_delete_files_removed", extra={"folder": folder_name, "path": str(document_folder)})
            except Exception:
                logger.exception(
                    "document_delete_files_failed",
                    extra={"folder": folder_name, "path": str(document_folder)},
                )
                raise

            details.append(folder_result)

        result = {
            "deleted": True,
            "folders": unique_folders,
            "collections_cleaned": collections_cleaned,
            "global_index_removed": global_index_removed,
            "details": details,
        }
        _log_workflow_exit(start_ts, "documents", "delete_documents", delete_run_id, "ok")
        return result
    except Exception:
        _log_workflow_exit(start_ts, "documents", "delete_documents", delete_run_id, "error")
        raise


def inspect_same_name_document(config_path: str | Path, file_name: str) -> dict[str, Any] | None:
    config = load_app_config(config_path)
    existing_folder = find_latest_same_name_document(config["paths"]["artifacts_root"], file_name)
    if existing_folder is None:
        return None
    return load_metadata(existing_folder)


def _run_pipeline_from_metadata(config_path: str | Path, document_folder: str | Path) -> dict[str, Any]:
    config = load_app_config(config_path)
    metadata = load_metadata(document_folder)
    document_id = metadata.get("document_id", "")
    start_ts = _log_workflow_enter("pipeline", "_run_pipeline_from_metadata", document_id)
    outputs = artifact_paths(document_folder, 1)
    start_summary_after_unlock = False

    try:
        with document_lock(document_folder):
            metadata = load_metadata(document_folder)
            last_step = metadata.get("last_successful_step", "created")
            if last_step == "upload":
                pdf_to_markdown.run(metadata["source_pdf_path"], outputs["markdown"], document_folder)
                last_step = "pdf_to_markdown"

            extraction_cfg = config.get("extraction", {})

            if last_step in ("pdf_to_markdown", "extract_images", "extract_tables"):
                if extraction_cfg.get("images", True):
                    extract_images.run(outputs["markdown"], document_folder)
                if extraction_cfg.get("tables", True):
                    extract_tables.run(outputs["markdown"], document_folder)
                last_step = "extract_assets"

            if last_step == "extract_assets":
                markdown_chunker.run(
                    outputs["markdown"],
                    outputs["chunks"],
                    document_folder,
                    chunk_size=config["chunking"]["chunk_size"],
                    chunk_overlap=config["chunking"]["chunk_overlap"],
                )
                last_step = "markdown_chunker"

            if last_step == "markdown_chunker":
                metadata = load_metadata(document_folder)
                write_to_vector_db.run(
                    chunk_paths=metadata["chunk_paths"],
                    document_folder=document_folder,
                    chroma_persist_dir=config["vector_db"]["persist_directory"],
                    collection_name=config["vector_db"]["collection_name"],
                    embedding_model=config["embeddings"]["model"],
                    index_result_path=outputs["vector_result"],
                    chroma_host=config["vector_db"].get("host"),
                    chroma_port=int(config["vector_db"].get("port", 8000)),
                )
                field_extractor.run(document_folder, config)
                metadata_cards.run(document_folder, config)
                final_metadata = load_metadata(document_folder)
                write_global_index_entry(
                    global_index_path(config),
                    document_folder,
                    build_global_entry(final_metadata, Path(document_folder)),
                )
                if not final_metadata.get("summary_ready"):
                    start_summary_after_unlock = True
                result_metadata = final_metadata
            else:
                # Reached when the document was already vectorised but summarisation is incomplete.
                # Always restart: the daemon may have been killed (status stays "in_progress" on restart)
                # and summarize_document.run() is idempotent - it skips already-completed checkpoints.
                final_metadata = load_metadata(document_folder)
                if not final_metadata.get("summary_ready"):
                    start_summary_after_unlock = True
                result_metadata = final_metadata

        if start_summary_after_unlock:
            start_summarization_background(config_path, document_folder)
        _log_workflow_exit(
            start_ts,
            "pipeline",
            "_run_pipeline_from_metadata",
            result_metadata.get("document_id", document_id),
            "ok",
        )
        return result_metadata
    except Exception:
        _log_workflow_exit(start_ts, "pipeline", "_run_pipeline_from_metadata", document_id, "error")
        raise


def prepare_document(
    config_path: str | Path,
    file_name: str,
    file_bytes: bytes,
    user_choice: str,
) -> dict[str, Any]:
    start_ts = _log_workflow_enter("ingestion", "prepare_document", "")
    doc_id = ""
    config = load_app_config(config_path)
    artifacts_root = config["paths"]["artifacts_root"]
    ensure_directory(artifacts_root)
    existing_folder = find_latest_same_name_document(artifacts_root, file_name)

    version = None
    if existing_folder is not None:
        existing_metadata = load_metadata(existing_folder)
        doc_id = existing_metadata.get("document_id", "")
        if existing_metadata.get("ready_to_chat") and user_choice == "reuse":
            if not existing_metadata.get("summary_ready"):
                result = _run_pipeline_from_metadata(config_path, existing_folder)
                _log_workflow_exit(start_ts, "ingestion", "prepare_document", result.get("document_id", doc_id), "ok")
                return result
            _log_workflow_exit(start_ts, "ingestion", "prepare_document", doc_id, "ok")
            return existing_metadata
        if not existing_metadata.get("ready_to_chat"):
            result = _run_pipeline_from_metadata(config_path, existing_folder)
            _log_workflow_exit(start_ts, "ingestion", "prepare_document", result.get("document_id", doc_id), "ok")
            return result
        version = int(existing_metadata["document_version"]) + 1

    try:
        document_folder = pdf_upload.save_uploaded_pdf(
            file_bytes=file_bytes,
            original_file_name=file_name,
            artifacts_root=artifacts_root,
            version=version,
        )
        result = _run_pipeline_from_metadata(config_path, document_folder)
        _log_workflow_exit(start_ts, "ingestion", "prepare_document", result.get("document_id", doc_id), "ok")
        return result
    except Exception:
        _log_workflow_exit(start_ts, "ingestion", "prepare_document", doc_id, "error")
        raise


def prepare_url_document(
    config_path: str | Path,
    url: str,
    user_choice: str,
) -> dict[str, Any]:
    from urllib.parse import urlparse as _urlparse

    start_ts = _log_workflow_enter("ingestion", "prepare_url_document", "")
    doc_id = ""
    document_name = url_ingest.url_to_document_name(url)
    config = load_app_config(config_path)
    artifacts_root = config["paths"]["artifacts_root"]
    ensure_directory(artifacts_root)
    existing_folder = find_latest_same_name_document(artifacts_root, document_name)

    # Early reuse check before any network request
    if existing_folder:
        existing_metadata = load_metadata(existing_folder)
        doc_id = existing_metadata.get("document_id", "")
        if existing_metadata.get("ready_to_chat") and user_choice == "reuse":
            if not existing_metadata.get("summary_ready"):
                result = _run_pipeline_from_metadata(config_path, existing_folder)
                _log_workflow_exit(start_ts, "ingestion", "prepare_url_document", result.get("document_id", doc_id), "ok")
                return result
            _log_workflow_exit(start_ts, "ingestion", "prepare_url_document", doc_id, "ok")
            return existing_metadata
        if not existing_metadata.get("ready_to_chat"):
            result = _run_pipeline_from_metadata(config_path, existing_folder)
            _log_workflow_exit(start_ts, "ingestion", "prepare_url_document", result.get("document_id", doc_id), "ok")
            return result

    # PDF URL -> download and run full PDF pipeline
    if url_ingest.is_pdf_url(url):
        try:
            body_bytes = url_ingest.fetch_pdf_bytes(url)
            path_segment = _urlparse(url).path.rstrip("/").split("/")[-1]
            file_name = path_segment or "document.pdf"
            if not file_name.lower().endswith(".pdf"):
                file_name += ".pdf"
            result = prepare_document(config_path, file_name, body_bytes, user_choice)
            _log_workflow_exit(start_ts, "ingestion", "prepare_url_document", result.get("document_id", doc_id), "ok")
            return result
        except Exception:
            _log_workflow_exit(start_ts, "ingestion", "prepare_url_document", doc_id, "error")
            raise

    try:
        # HTML -> scrape directly to markdown (no PDF conversion)
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
        result = _run_pipeline_from_metadata(config_path, document_folder)
        _log_workflow_exit(start_ts, "ingestion", "prepare_url_document", result.get("document_id", doc_id), "ok")
        return result
    except Exception:
        _log_workflow_exit(start_ts, "ingestion", "prepare_url_document", doc_id, "error")
        raise


def load_document(document_folder: str | Path) -> dict[str, Any]:
    return load_metadata(document_folder)


def find_relevant_documents(config_path: str | Path, question: str) -> list[dict[str, Any]]:
    config = load_app_config(config_path)
    return multi_doc_query.find_relevant_documents(question, config)


def ask_multi_document_question(
    config_path: str | Path,
    document_folders: list[str],
    question: str,
) -> dict[str, Any]:
    config = load_app_config(config_path)
    return multi_doc_query.ask_across_documents(question, document_folders, config)


# Concurrency boundary: spawns a daemon thread + calls update_summary_progress.
# Do not refactor thread ownership here until core/job_store.py is in place (Phase B).
def start_summarization_background(config_path: str | Path, document_folder: str | Path) -> None:
    """Non-blocking: marks in_progress and spawns a daemon thread for summarization."""

    metadata = load_metadata(document_folder)
    document_id = metadata.get("document_id", "")
    start_ts = _log_workflow_enter("summary", "start_summarization_background", document_id)
    try:
        config = load_app_config(config_path)
        update_summary_progress(document_folder, "started", True)

        def _run() -> None:
            from services import summarize_document

            try:
                summarize_document.run(document_folder, config)
            except Exception:
                failed = load_metadata(document_folder)
                failed["summary_ready"] = False
                failed["summary_status"] = "error"
                save_metadata(document_folder, failed)
                logger.exception("Background summarization failed for %s", document_folder)

        threading.Thread(target=_run, daemon=True).start()
        _log_workflow_exit(start_ts, "summary", "start_summarization_background", document_id, "ok")
    except Exception:
        _log_workflow_exit(start_ts, "summary", "start_summarization_background", document_id, "error")
        raise


def start_summary_watcher(config_path: str | Path) -> None:
    config = load_app_config(config_path)
    summary_watcher.start(config)


def reset_summary_level(config_path: str | Path, document_folder: str | Path, level: str) -> None:
    """Clear checkpoints for *level* and all downstream levels, then restart summarization.

    level: "level1" | "level2" | "level3"
    Regenerating a level invalidates everything that was derived from it.
    """

    folder = Path(document_folder)
    metadata = load_metadata(folder)
    document_id = metadata.get("document_id", "")
    start_ts = _log_workflow_enter("summary", "reset_summary_level", document_id)
    try:
        progress = metadata.setdefault("summary_progress", {})
        _downstream: dict[str, list[str]] = {
            "level3": ["level3_sections_done", "level3_complete", "level3_indexed", "level2_complete", "level1_complete", "level1_indexed"],
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
        _log_workflow_exit(start_ts, "summary", "reset_summary_level", document_id, "ok")
    except Exception:
        _log_workflow_exit(start_ts, "summary", "reset_summary_level", document_id, "error")
        raise


def ask_question(config_path: str | Path, document_folder: str | Path, question: str) -> dict[str, Any]:
    config = load_app_config(config_path)
    metadata = load_metadata(document_folder)
    document_id = metadata.get("document_id", "")
    start_ts = _log_workflow_enter("qa", "ask_question", document_id)
    question_count = len(list((Path(document_folder) / "chat").glob("query_*.json"))) + 1
    outputs = artifact_paths(document_folder, question_count)

    try:
        tree = retrieve_context_tree.retrieve(question, config, selected_document_folders=[str(document_folder)])
        retrieval_payload = {
            "question": question,
            "documents": tree.get("documents", []),
            "metadatas": tree.get("metadatas", []),
            "ids": tree.get("ids", []),
            "media_items": [],
            "section_media": {"image_paths": [], "table_paths": []},
            "images_meta": [],
            "confidence": tree.get("confidence", "normal"),
            "mode": tree.get("mode", "specific"),
            "abstain": tree.get("abstain", False),
        }
        write_json(outputs["retrieval"], retrieval_payload)
        retrieval_context_path = str(outputs["retrieval"])

        retrieval_payload = read_json(retrieval_context_path, default={})
        if retrieval_payload.get("abstain"):
            result = {
                "question": question,
                "answer": "I cannot answer this from the available document context.",
                "sources": [],
                "image_paths": [],
                "abstain": True,
                "confidence": retrieval_payload.get("confidence", "low"),
                "mode": retrieval_payload.get("mode", "specific"),
                "document_name": metadata["document_name"],
            }
            _log_workflow_exit(start_ts, "qa", "ask_question", document_id, "ok")
            return result

        answer_path = chat_response.run(
            retrieval_input_path=retrieval_context_path,
            output_answer_path=outputs["chat"],
            document_folder=document_folder,
            chat_model=config["chat"]["model"],
            media_instruction=config.get("prompt", {}).get("media_instruction", ""),
        )
        answer_payload = read_json(answer_path, default={})
        answer_payload["document_name"] = metadata["document_name"]
        answer_payload["image_paths"] = image_render.resolve_top_chunk_images(answer_payload, str(document_folder))
        answer_payload["abstain"] = False
        _log_workflow_exit(start_ts, "qa", "ask_question", document_id, "ok")
        return answer_payload
    except Exception:
        _log_workflow_exit(start_ts, "qa", "ask_question", document_id, "error")
        raise
