from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

import main as pipeline


CONFIG_PATH = Path(__file__).parent / "config" / "app_config.yaml"


# ── Session state ────────────────────────────────────────────────────────────

def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("upload_key", 0)
    st.session_state.setdefault("url_input_key", 0)
    st.session_state.setdefault("url_processing", False)
    st.session_state.setdefault("pending_url", "")
    st.session_state.setdefault("pending_user_choice", "new_version")
    st.session_state.setdefault("url_error", "")
    # PDF upload processing state (mirrors url_processing pattern)
    st.session_state.setdefault("pdf_processing", False)
    st.session_state.setdefault("pending_pdf_name", "")
    st.session_state.setdefault("pending_pdf_bytes", b"")
    st.session_state.setdefault("pending_pdf_choice", "new_version")
    st.session_state.setdefault("pdf_info", "")
    # Multi-doc / routing state
    st.session_state.setdefault("selected_document_folders", [])
    st.session_state.setdefault("chat_mode", "routing")        # "routing" | "deep_search"
    st.session_state.setdefault("active_deep_search_folders", [])
    st.session_state.setdefault("pending_routing_question", "")
    st.session_state.setdefault("routing_candidates", [])
    # Pane visibility state
    st.session_state.setdefault("show_doc_pane", True)
    st.session_state.setdefault("show_summary_pane", True)
    st.session_state.setdefault("_sum_pane_active_doc", "")
    st.session_state.setdefault("_sum_pane_active_level", "")
    # Tracks folders whose summarization daemon was started in this session.
    # Daemon threads die on process exit, so any "in_progress" not in this
    # set is stale from a previous session.
    st.session_state.setdefault("active_summarization_folders", set())


def reset_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["routing_candidates"] = []
    st.session_state["pending_routing_question"] = ""


def _track_summarization(folder: str) -> None:
    """Start background summarization and register in the session-level tracking set."""
    pipeline.start_summarization_background(CONFIG_PATH, folder)
    st.session_state["active_summarization_folders"].add(str(folder))


# ── Pipeline status ──────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    "upload",
    "pdf_to_markdown",
    "extract_images",
    "extract_tables",
    "markdown_chunker",
    "write_to_vector_db",
    "retrieve_context",
    "chat_response",
]


def _status_label(metadata: dict, step_name: str) -> str:
    step_info = metadata.get("steps", {}).get(step_name, {})
    if step_info.get("status") == "success":
        return "done"
    return "pending"


def render_pipeline_status(metadata: dict, title: str) -> None:
    with st.expander(title, expanded=False):
        st.write(f"Ready to chat: `{metadata.get('ready_to_chat', False)}`")
        st.write(f"Last successful step: `{metadata.get('last_successful_step', 'unknown')}`")
        st.write(f"Total chunks: `{metadata.get('total_chunks', 0)}`")
        for step_name in PIPELINE_STEPS:
            st.write(f"- `{step_name}`: `{_status_label(metadata, step_name)}`")


# ── Summary helpers ───────────────────────────────────────────────────────────

def _summary_progress_label(metadata: dict) -> str:
    p = metadata.get("summary_progress", {})
    if p.get("level1_indexed"):
        return "finalizing"
    if p.get("level1_complete"):
        return "indexing L1"
    if p.get("level2_complete"):
        return "running L1"
    if p.get("level3_indexed"):
        return "running L2"
    if p.get("level3_complete"):
        return "indexing L3"
    return "starting"


# ── Summary pane (middle column) ─────────────────────────────────────────────

_LEVEL_FILES = {
    "level1": "level1_onepager.json",
    "level2": "level2_medium.json",
    "level3": "level3_detailed.json",
}
_LEVEL_SHORT = {"level1": "1-Pager", "level2": "Medium", "level3": "Detailed"}
_LEVEL_LONG  = {"level1": "1-Pager Summary", "level2": "Medium Summary", "level3": "Detailed Summary"}


def render_summary_pane(selected_folders: list[str]) -> None:
    from core.storage import read_json as _read_json

    # Invisible anchor used by CSS :has(#summary-pane-marker) to colour
    # this specific stColumn without touching inner button-row columns.
    st.markdown('<span id="summary-pane-marker"></span>', unsafe_allow_html=True)

    col_title, col_close = st.columns([4, 1])
    with col_title:
        st.subheader("Summaries")
    with col_close:
        if st.button("◀", key="sum_pane_close", use_container_width=True):
            st.session_state["show_summary_pane"] = False
            st.rerun()

    # Clear active selection when its document is no longer selected
    active_doc   = st.session_state.get("_sum_pane_active_doc", "")
    active_level = st.session_state.get("_sum_pane_active_level", "")
    if active_doc and active_doc not in selected_folders:
        st.session_state["_sum_pane_active_doc"]   = ""
        st.session_state["_sum_pane_active_level"] = ""
        active_doc = active_level = ""

    st.divider()

    if not selected_folders:
        st.caption("Select documents from the left panel to view their summaries here.")

    for folder in selected_folders:
        metadata   = pipeline.load_document(folder)
        doc_name   = metadata.get("document_name", Path(folder).name)
        status     = metadata.get("summary_status", "pending")
        sum_ready  = metadata.get("summary_ready", False)
        folder_slug = Path(folder).name
        summaries_dir = Path(folder) / "summaries"
        available  = {k for k, f in _LEVEL_FILES.items() if (summaries_dir / f).exists()}

        st.markdown(f"**{doc_name}**")

        # Status / generate controls
        if not sum_ready:
            if status == "in_progress":
                st.caption(f"⏳ {_summary_progress_label(metadata)}")
                if st.button("Restart", key=f"sp_restart_{folder_slug}", use_container_width=True):
                    _track_summarization(folder)
                    st.rerun()
            elif status == "error":
                st.caption("❌ Generation failed")
                if st.button("Retry", key=f"sp_retry_{folder_slug}", use_container_width=True):
                    _track_summarization(folder)
                    st.rerun()
            elif not available:
                if st.button("Generate Summaries", type="primary",
                             key=f"sp_gen_{folder_slug}", use_container_width=True):
                    _track_summarization(folder)
                    st.rerun()

        # Level buttons (disabled when file not yet generated)
        if available or sum_ready:
            c1, c2, c3 = st.columns(3)
            for level_key, col in [("level1", c1), ("level2", c2), ("level3", c3)]:
                is_active = (active_doc == folder and active_level == level_key)
                btn_key   = f"sp_btn_{level_key}_{folder_slug}"
                with col:
                    if level_key in available:
                        btype = "primary" if is_active else "secondary"
                        if st.button(_LEVEL_SHORT[level_key], key=btn_key,
                                     use_container_width=True, type=btype):
                            if is_active:
                                st.session_state["_sum_pane_active_doc"]   = ""
                                st.session_state["_sum_pane_active_level"] = ""
                            else:
                                st.session_state["_sum_pane_active_doc"]   = folder
                                st.session_state["_sum_pane_active_level"] = level_key
                            st.rerun()
                    else:
                        st.button(_LEVEL_SHORT[level_key], key=btn_key,
                                  use_container_width=True, disabled=True)

        st.divider()

    # ── Shared summary content area (bottom, collapsible) ────────────────────
    active_doc   = st.session_state.get("_sum_pane_active_doc", "")
    active_level = st.session_state.get("_sum_pane_active_level", "")

    if active_doc and active_level:
        summaries_dir = Path(active_doc) / "summaries"
        data = {}
        try:
            data = _read_json(str(summaries_dir / _LEVEL_FILES[active_level]), default={})
        except Exception:
            pass

        doc_name = pipeline.load_document(active_doc).get("document_name", Path(active_doc).name)
        with st.expander(f"{_LEVEL_LONG[active_level]} — {doc_name}", expanded=True):
            if active_level == "level3":
                for sec in data.get("sections", []):
                    st.markdown(f"**{sec.get('section', '')}**")
                    st.markdown(sec.get("summary", ""))
            else:
                st.markdown(data.get("summary", "No summary available."))

            if st.button("↻ Regenerate this level",
                         key=f"sp_regen_{Path(active_doc).name}_{active_level}",
                         use_container_width=True):
                pipeline.reset_summary_level(CONFIG_PATH, active_doc, active_level)
                st.session_state["active_summarization_folders"].add(str(active_doc))
                st.session_state["_sum_pane_active_doc"]   = ""
                st.session_state["_sum_pane_active_level"] = ""
                st.rerun()


# ── Left panel: existing documents ───────────────────────────────────────────

def render_existing_documents() -> None:
    documents = pipeline.list_documents(CONFIG_PATH)
    if not documents:
        st.info("No persisted documents found yet.")
        return

    # ── Select All / Deselect All ────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Select All", use_container_width=True):
            for doc in documents:
                if doc["ready_to_chat"]:
                    st.session_state[f"doc_chk_{doc['folder_name']}"] = True
            st.rerun()
    with col_b:
        if st.button("Deselect All", use_container_width=True):
            for doc in documents:
                st.session_state[f"doc_chk_{doc['folder_name']}"] = False
            st.session_state["selected_document_folders"] = []
            st.session_state["chat_mode"] = "routing"
            st.session_state["active_deep_search_folders"] = []
            reset_chat()
            st.rerun()

    # ── Checkbox list ────────────────────────────────────────────────────────
    for doc in documents:
        key = f"doc_chk_{doc['folder_name']}"
        st.session_state.setdefault(key, False)

        summary_badge = ""
        if doc.get("summary_ready"):
            summary_badge = "  📝"
        elif doc.get("summary_status") == "in_progress":
            summary_badge = "  ⏳"
        elif doc.get("summary_status") == "error":
            summary_badge = "  ❌"

        ready_badge = "🟢" if doc["ready_to_chat"] else "🔴"
        label = f"{ready_badge} {doc['folder_name']}{summary_badge}"

        st.checkbox(label, key=key, disabled=not doc["ready_to_chat"])

    # Derive selected list from checkbox states
    selected = [
        doc["document_folder"]
        for doc in documents
        if st.session_state.get(f"doc_chk_{doc['folder_name']}", False)
    ]
    st.session_state["selected_document_folders"] = selected

    if selected:
        st.caption(f"{len(selected)} document(s) selected")

    # Pipeline status for single selected doc
    if len(selected) == 1:
        doc_meta = pipeline.load_document(selected[0])
        render_pipeline_status(doc_meta, "Pipeline status")


# ── Sidebar: document ingest ─────────────────────────────────────────────────

def render_upload_panel() -> None:
    if st.session_state["pdf_processing"]:
        st.file_uploader(
            "Upload a PDF", type=["pdf"], accept_multiple_files=False,
            key="pdf_uploader_processing", disabled=True,
        )
        st.button("Processing...", disabled=True, use_container_width=True, key="pdf_btn_proc")
        with st.spinner("Running document pipeline..."):
            try:
                metadata = pipeline.prepare_document(
                    config_path=CONFIG_PATH,
                    file_name=st.session_state["pending_pdf_name"],
                    file_bytes=st.session_state["pending_pdf_bytes"],
                    user_choice=st.session_state["pending_pdf_choice"],
                )
                if metadata.get("summary_status") == "in_progress":
                    st.session_state["active_summarization_folders"].add(
                        metadata.get("document_folder", "")
                    )
                reset_chat()
                if metadata.get("ready_to_chat"):
                    st.toast(f"`{metadata['document_name']}` is ready!")
                else:
                    st.session_state["pdf_info"] = (
                        f"`{metadata['document_name']}` resumed to step "
                        f"`{metadata.get('last_successful_step', 'unknown')}`."
                    )
            finally:
                st.session_state["pdf_processing"] = False
                st.session_state["pending_pdf_name"] = ""
                st.session_state["pending_pdf_bytes"] = b""
                st.session_state["pending_pdf_choice"] = "new_version"
                st.session_state["upload_key"] += 1
        st.rerun()
        return

    if st.session_state["pdf_info"]:
        st.info(st.session_state["pdf_info"])
        st.session_state["pdf_info"] = ""

    uploaded_file = st.file_uploader(
        "Upload a PDF", type=["pdf"], accept_multiple_files=False,
        key=f"pdf_uploader_{st.session_state['upload_key']}",
    )
    if not uploaded_file:
        return

    user_choice = "new_version"
    same_name_document = pipeline.inspect_same_name_document(CONFIG_PATH, uploaded_file.name)
    if same_name_document and same_name_document.get("ready_to_chat"):
        user_choice = st.radio(
            "A stored file with the same name exists. Choose what to do.",
            options=["reuse", "rebuild"],
            captions=[
                "Open the already processed version and go straight to chat.",
                "Create a new version folder and process the upload again.",
            ],
            horizontal=False,
        )
    elif same_name_document and not same_name_document.get("ready_to_chat"):
        st.info(
            "A same-name document exists but is not fully processed yet. "
            "This upload will resume from its last successful step."
        )
        user_choice = "reuse"

    if st.button("Process uploaded PDF", use_container_width=True):
        st.session_state["pending_pdf_name"] = uploaded_file.name
        st.session_state["pending_pdf_bytes"] = uploaded_file.getvalue()
        st.session_state["pending_pdf_choice"] = user_choice
        st.session_state["pdf_processing"] = True
        st.rerun()


def render_url_panel() -> None:
    if st.session_state["url_processing"]:
        st.text_input(
            "Web page or PDF URL",
            value="",
            disabled=True,
            placeholder="Processing...",
            key="url_input_processing",
        )
        st.button("Processing...", disabled=True, use_container_width=True, key="url_btn_proc")
        with st.spinner("Fetching and processing URL..."):
            try:
                metadata = pipeline.prepare_url_document(
                    CONFIG_PATH,
                    st.session_state["pending_url"],
                    st.session_state["pending_user_choice"],
                )
                if metadata.get("summary_status") == "in_progress":
                    st.session_state["active_summarization_folders"].add(
                        metadata.get("document_folder", "")
                    )
                st.session_state["url_input_key"] += 1
                reset_chat()
                if metadata.get("ready_to_chat"):
                    st.toast(f"`{metadata['document_name']}` is ready!")
            except Exception as exc:
                st.session_state["url_error"] = str(exc)
            finally:
                st.session_state["url_processing"] = False
                st.session_state["pending_url"] = ""
                st.session_state["pending_user_choice"] = "new_version"
        st.rerun()

    if st.session_state["url_error"]:
        st.error(st.session_state["url_error"])
        st.session_state["url_error"] = ""

    url = st.text_input(
        "Web page or PDF URL",
        placeholder="https://example.com/report.pdf  or  https://example.com/article",
        key=f"url_input_{st.session_state['url_input_key']}",
    ).strip()
    if not url:
        return

    user_choice = "new_version"
    if url.startswith(("http://", "https://")) and "." in url[8:]:
        same = pipeline.inspect_same_url_document(CONFIG_PATH, url)
        if same and same.get("ready_to_chat"):
            user_choice = st.radio(
                "A stored document with the same URL already exists.",
                options=["reuse", "rebuild"],
                captions=["Use the existing processed version.", "Re-scrape and reprocess."],
                horizontal=False,
                key="url_radio",
            )
        elif same and not same.get("ready_to_chat"):
            st.info("Existing document not fully processed — will resume from last step.")
            user_choice = "reuse"

    if st.button("Process URL", use_container_width=True, key="url_submit"):
        st.session_state["pending_url"] = url
        st.session_state["pending_user_choice"] = user_choice
        st.session_state["url_processing"] = True
        st.rerun()


def render_ingest_panel() -> None:
    tab_pdf, tab_url = st.tabs(["Upload PDF", "From URL"])
    with tab_pdf:
        render_upload_panel()
    with tab_url:
        render_url_panel()


# ── Chat helpers ─────────────────────────────────────────────────────────────

def _resolve_image_path(p: str) -> str | None:
    ph = Path(p)
    if ph.is_absolute():
        return str(ph) if ph.exists() else None
    resolved = Path(__file__).parent / p
    return str(resolved) if resolved.exists() else None


def _render_message_history() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            img_paths = message.get("image_paths", [])
            if img_paths:
                with st.expander(f"📷 {len(img_paths)} image(s)"):
                    cols = st.columns(min(len(img_paths), 3))
                    for i, img_path in enumerate(img_paths):
                        cols[i % 3].image(img_path, use_container_width=True)
            sources = message.get("chunk_sources", [])
            if sources:
                with st.expander(f"📄 {len(sources)} source(s)", expanded=False):
                    for src in sources:
                        doc = src.get("document_name", "Unknown")
                        section = src.get("section_path", "")
                        chunk_num = src.get("chunk_number", "")
                        snippet = src.get("chunk_text_snippet", "")
                        label = f"**{doc}**"
                        if section:
                            label += f" — {section}"
                        elif chunk_num:
                            label += f" — Chunk {chunk_num}"
                        st.markdown(label)
                        if snippet:
                            st.caption(f'"{snippet}..."')


def _render_routing_candidates() -> None:
    candidates = st.session_state.get("routing_candidates", [])
    if not candidates:
        return

    st.info(
        "I found these documents may be relevant. "
        "Select which to search, then confirm:"
    )
    for cand in candidates:
        key = f"rchk_{cand['folder']}"
        st.session_state.setdefault(key, True)
        section_hint = f" — *{cand['top_section']}*" if cand.get("top_section") else ""
        score_pct = int(cand.get("score", 0) * 100)
        label = f"**{cand['doc_name']}**{section_hint}  `{score_pct}% match`"
        st.checkbox(label, key=key)

    selected_folders = [
        c["folder"]
        for c in candidates
        if st.session_state.get(f"rchk_{c['folder']}", True)
    ]

    if st.button("Confirm & Search selected documents", type="primary"):
        if not selected_folders:
            st.warning("Select at least one document.")
        else:
            st.session_state["active_deep_search_folders"] = selected_folders
            st.session_state["chat_mode"] = "deep_search"
            st.session_state["routing_candidates"] = []
            st.rerun()


# ── Main chat area ───────────────────────────────────────────────────────────

def render_chat() -> None:
    selected_folders = st.session_state.get("selected_document_folders", [])
    chat_mode = st.session_state.get("chat_mode", "routing")
    active_folders = st.session_state.get("active_deep_search_folders", [])

    if len(selected_folders) == 1:
        _render_single_doc_chat(selected_folders[0])
        return

    if len(selected_folders) > 1:
        _render_direct_multi_doc_chat(selected_folders)
        return

    # ── No docs selected: routing mode or deep search (post-routing) ─────────
    if chat_mode == "deep_search" and active_folders:
        _render_deep_search_chat(active_folders)
    else:
        _render_routing_chat()


def _render_single_doc_chat(document_folder: str) -> None:
    metadata = pipeline.load_document(document_folder)
    st.subheader(f"Chatting with `{metadata['document_name']}`")
    st.caption(
        f"Folder: `{Path(document_folder).name}` | "
        f"chunks: `{metadata.get('total_chunks', 0)}`"
    )

    if not metadata.get("ready_to_chat"):
        st.warning("This document is not ready to chat yet.")
        return

    _render_message_history()

    question = st.chat_input("Ask a question about this document")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            answer_payload = pipeline.ask_question(CONFIG_PATH, document_folder, question)

            image_paths: list[str] = []
            sources = answer_payload.get("sources", [])
            if sources:
                top_meta = sources[0]
                doc_folder = top_meta.get("document_folder", "")
                chunk_num = top_meta.get("chunk_number", 0)
                chunk_path = top_meta.get("chunk_path", "")
                image_paths = pipeline.read_chunk(chunk_path).get("image_paths", []) if chunk_path else []
                if not image_paths:
                    image_paths = pipeline.read_next_chunk(doc_folder, chunk_num).get("image_paths", [])
                if not image_paths:
                    image_paths = pipeline.read_prev_chunk(doc_folder, chunk_num).get("image_paths", [])

            image_paths = [r for p in dict.fromkeys(image_paths) if p for r in [_resolve_image_path(p)] if r]

            sources_text = "\n".join(
                f"- {item.get('section_path') or ('Chunk ' + str(item.get('chunk_number', 'unknown')))}"
                for item in sources
            )
            answer_text = f"{answer_payload.get('answer', 'No answer generated.')}\n\nSources:\n{sources_text}"

        st.markdown(answer_text)
        if image_paths:
            with st.expander(f"📷 {len(image_paths)} image(s)"):
                cols = st.columns(min(len(image_paths), 3))
                for i, img_path in enumerate(image_paths):
                    cols[i % 3].image(img_path, use_container_width=True)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer_text,
        "image_paths": image_paths,
    })


def _render_direct_multi_doc_chat(document_folders: list[str]) -> None:
    doc_names = [pipeline.load_document(f).get("document_name", Path(f).name) for f in document_folders]
    st.subheader(f"Multi-doc mode — {len(document_folders)} documents")
    st.caption(", ".join(doc_names))

    _render_message_history()

    question = st.chat_input("Ask a question across selected documents")
    if not question:
        return

    _process_multi_doc_question(question, document_folders)


def _render_routing_chat() -> None:
    st.subheader("Find relevant documents")
    st.caption(
        "Ask a question and the system will search across all document summaries "
        "to identify which documents are relevant — then you confirm which to search."
    )

    _render_message_history()
    _render_routing_candidates()

    question = st.chat_input("Ask a question to find relevant documents...")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    st.session_state["pending_routing_question"] = question

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching document library..."):
        candidates = pipeline.find_relevant_documents(CONFIG_PATH, question)

    if not candidates:
        msg = (
            "No relevant documents found. Make sure documents have been summarized, "
            "or select documents manually in the left panel."
        )
        st.session_state["messages"].append({"role": "assistant", "content": msg})
        st.session_state["pending_routing_question"] = ""
        st.rerun()
        return

    st.session_state["routing_candidates"] = candidates
    st.rerun()  # raises RerunException — stops execution here


def _render_deep_search_chat(active_folders: list[str]) -> None:
    doc_names = [pipeline.load_document(f).get("document_name", Path(f).name) for f in active_folders]

    # Banner
    col_names, col_btn = st.columns([4, 1])
    with col_names:
        st.subheader("Deep search — " + ", ".join(doc_names))
    with col_btn:
        if st.button("Change Documents", use_container_width=True):
            st.session_state["chat_mode"] = "routing"
            st.session_state["active_deep_search_folders"] = []
            st.session_state["routing_candidates"] = []
            reset_chat()
            st.rerun()

    # Process pending question from routing confirmation
    pending = st.session_state.get("pending_routing_question", "")
    if pending:
        st.session_state["pending_routing_question"] = ""
        _process_multi_doc_question(pending, active_folders)
        st.rerun()

    _render_message_history()

    question = st.chat_input("Ask a question about these documents... (type /back to change documents)")
    if not question:
        return

    if question.strip().lower() == "/back":
        st.session_state["chat_mode"] = "routing"
        st.session_state["active_deep_search_folders"] = []
        st.session_state["routing_candidates"] = []
        reset_chat()
        st.rerun()
        return

    _process_multi_doc_question(question, active_folders)


def _process_multi_doc_question(question: str, document_folders: list[str]) -> None:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching across documents..."):
            payload = pipeline.ask_multi_document_question(CONFIG_PATH, document_folders, question)

        answer = payload.get("answer", "No answer generated.")
        sources = payload.get("sources", [])

        st.markdown(answer)

        if sources:
            with st.expander(f"📄 {len(sources)} source(s)", expanded=False):
                for src in sources:
                    doc = src.get("document_name", "Unknown")
                    section = src.get("section_path", "")
                    chunk_num = src.get("chunk_number", "")
                    snippet = src.get("chunk_text_snippet", "")
                    label = f"**{doc}**"
                    if section:
                        label += f" — {section}"
                    elif chunk_num:
                        label += f" — Chunk {chunk_num}"
                    st.markdown(label)
                    if snippet:
                        st.caption(f'"{snippet}..."')

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "chunk_sources": sources,
    })


# ── App entry point ──────────────────────────────────────────────────────────

_NAVBAR_HTML = (
    '<div class="profrag-navbar">'
    '<span class="profrag-navbar-title">📄 ProfRAG</span>'
    "</div>"
)

_APP_CSS = """
<style>
/* ── Navbar ──────────────────────────────────────────────────────────────── */
.profrag-navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 52px;
    z-index: 999999;
    background: linear-gradient(90deg, #1a3558 0%, #2563a8 100%);
    display: flex;
    align-items: center;
    padding: 0 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
.profrag-navbar-title {
    color: #ffffff;
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

/* ── Main layout ─────────────────────────────────────────────────────────── */
[data-testid="stMainBlockContainer"],
.block-container {
    padding-left:  0 !important;
    padding-right: 0 !important;
    padding-top:   52px !important;
    max-width:     100% !important;
}
section[data-testid="stMain"] {
    background-color: #edf4fc !important;
}

/* ── Open pane columns — independently scrollable ────────────────────────── */
[data-testid="stColumn"]:has(#doc-col-marker) {
    background-color: #b8d0eb !important;
    border-right:     2px solid #6fa3d8 !important;
    height:           calc(100vh - 52px);
    overflow-y:       auto;
    overflow-x:       hidden;
}
[data-testid="stColumn"]:has(#summary-pane-marker) {
    background-color: #d4e8f8 !important;
    border-right:     2px solid #6fa3d8 !important;
    height:           calc(100vh - 52px);
    overflow-y:       auto;
    overflow-x:       hidden;
}
[data-testid="stColumn"]:has(#chat-col-marker) {
    height:           calc(100vh - 52px);
    overflow-y:       auto;
    overflow-x:       hidden;
}

/* ── Collapsed pane tabs (narrow expand strips) ──────────────────────────── */
[data-testid="stColumn"]:has(#doc-tab-marker),
[data-testid="stColumn"]:has(#sum-tab-marker) {
    height:   calc(100vh - 52px);
    overflow: hidden;
    cursor:   pointer;
}
[data-testid="stColumn"]:has(#doc-tab-marker) {
    background-color: #b8d0eb !important;
    border-right:     2px solid #6fa3d8 !important;
}
[data-testid="stColumn"]:has(#sum-tab-marker) {
    background-color: #d4e8f8 !important;
    border-right:     2px solid #6fa3d8 !important;
}
/* Strip default vertical-block padding so button fills the narrow column */
[data-testid="stColumn"]:has(#doc-tab-marker) [data-testid="stVerticalBlock"],
[data-testid="stColumn"]:has(#sum-tab-marker) [data-testid="stVerticalBlock"] {
    padding: 0 !important;
    gap:     0 !important;
}
/* Expand-arrow button style */
[data-testid="stColumn"]:has(#doc-tab-marker) button,
[data-testid="stColumn"]:has(#sum-tab-marker) button {
    background:  transparent !important;
    border:      none !important;
    box-shadow:  none !important;
    color:       #1a3558 !important;
    font-size:   1.2rem !important;
    font-weight: 900 !important;
    min-height:  64px !important;
    width:       100% !important;
    padding:     10px 0 !important;
}
[data-testid="stColumn"]:has(#doc-tab-marker) button:hover,
[data-testid="stColumn"]:has(#sum-tab-marker) button:hover {
    background: rgba(26, 53, 88, 0.14) !important;
    color:      #2563a8 !important;
}

/* ── Responsive — tablet (≤ 1024px) ─────────────────────────────────────── */
@media (max-width: 1024px) {
    .profrag-navbar-title { font-size: 1.25rem; }
}

/* ── Responsive — mobile (≤ 768px) ──────────────────────────────────────── */
@media (max-width: 768px) {
    .profrag-navbar-title { font-size: 1rem; letter-spacing: 0.05em; }
    [data-testid="stHorizontalBlock"]:has(#chat-col-marker) {
        flex-wrap: wrap !important;
    }
    [data-testid="stColumn"]:has(#doc-col-marker),
    [data-testid="stColumn"]:has(#doc-tab-marker),
    [data-testid="stColumn"]:has(#summary-pane-marker),
    [data-testid="stColumn"]:has(#sum-tab-marker),
    [data-testid="stColumn"]:has(#chat-col-marker) {
        min-width:    100% !important;
        width:        100% !important;
        height:       auto !important;
        max-height:   55vh;
        border-right: none !important;
        border-bottom: 2px solid #6fa3d8;
    }
}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="ProfRAG", page_icon="📄", layout="wide")
    st.markdown(_APP_CSS, unsafe_allow_html=True)
    st.markdown(_NAVBAR_HTML, unsafe_allow_html=True)
    initialize_state()

    show_doc = st.session_state.get("show_doc_pane", True)
    show_sum = st.session_state.get("show_summary_pane", True)

    doc_col = sum_col = doc_tab = sum_tab = None

    # Four layout states — each pane is independent of the other
    if show_doc and show_sum:
        doc_col, sum_col, chat_col = st.columns([20, 30, 50])
    elif show_doc:                  # summary pane collapsed → narrow tab
        doc_col, sum_tab, chat_col = st.columns([20, 3, 77])
    elif show_sum:                  # doc pane collapsed → narrow tab
        doc_tab, sum_col, chat_col = st.columns([3, 30, 67])
    else:                           # both panes collapsed
        doc_tab, sum_tab, chat_col = st.columns([3, 3, 94])

    # ── Collapsed doc tab ────────────────────────────────────────────────────
    if doc_tab is not None:
        with doc_tab:
            st.markdown('<span id="doc-tab-marker"></span>', unsafe_allow_html=True)
            if st.button("▶", key="doc_expand", use_container_width=True,
                         help="Expand Documents"):
                st.session_state["show_doc_pane"] = True
                st.rerun()

    # ── Open doc pane ────────────────────────────────────────────────────────
    # Rendered first so render_existing_documents() writes the current checkbox
    # state into selected_document_folders before the summary pane reads it.
    if doc_col is not None:
        with doc_col:
            st.markdown('<span id="doc-col-marker"></span>', unsafe_allow_html=True)
            _, coll_col = st.columns([6, 1])
            with coll_col:
                if st.button("◀", key="doc_collapse", use_container_width=True,
                             help="Collapse Documents"):
                    st.session_state["show_doc_pane"] = False
                    st.rerun()
            st.subheader("Documents")
            render_existing_documents()
            st.divider()
            st.subheader("Add Document")
            render_ingest_panel()

    # Read selection AFTER render_existing_documents() has updated session state
    # so the summary pane and chat see the current checkbox values, not last frame's.
    selected = st.session_state.get("selected_document_folders", [])

    # ── Collapsed sum tab ────────────────────────────────────────────────────
    if sum_tab is not None:
        with sum_tab:
            st.markdown('<span id="sum-tab-marker"></span>', unsafe_allow_html=True)
            if st.button("▶", key="sum_expand", use_container_width=True,
                         help="Expand Summaries"):
                st.session_state["show_summary_pane"] = True
                st.rerun()

    # ── Open summary pane ────────────────────────────────────────────────────
    if sum_col is not None:
        with sum_col:
            render_summary_pane(selected)

    # ── Chat column (always visible) ─────────────────────────────────────────
    with chat_col:
        st.markdown('<span id="chat-col-marker"></span>', unsafe_allow_html=True)
        st.caption("Select documents from the left panel or ask a question to find relevant documents automatically.")
        render_chat()


if __name__ == "__main__":
    main()
