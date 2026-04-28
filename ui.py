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
    # Action bar panel toggles
    st.session_state.setdefault("show_pdf_panel", False)
    st.session_state.setdefault("show_url_panel", False)
    st.session_state.setdefault("show_add_dropdown", False)
    st.session_state.setdefault("_bar_question", "")
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

    col_title, col_close = st.columns([4, 1], vertical_alignment="center")
    with col_title:
        st.markdown('<span id="sum-close-sticky-marker"></span>', unsafe_allow_html=True)
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
                label = _summary_progress_label(metadata)
                st.markdown(
                    f'<span class="summary-progress-badge">⟳ {label}</span>',
                    unsafe_allow_html=True,
                )
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

    # ── Search + scrollable list ─────────────────────────────────────────────
    with st.container(height=320, border=False):
        search = st.text_input(
            "Search",
            placeholder="Filter documents...",
            key="doc_search",
            label_visibility="collapsed",
        ).strip().lower()

        filtered = (
            [d for d in documents if search in d["folder_name"].lower()]
            if search else documents
        )

        for doc in filtered:
            key = f"doc_chk_{doc['folder_name']}"
            st.session_state.setdefault(key, False)
            is_selected = st.session_state.get(key, False)
            is_ready    = doc["ready_to_chat"]

            summary_badge = ""
            if doc.get("summary_ready"):
                summary_badge = " 📝"
            elif doc.get("summary_status") == "in_progress":
                summary_badge = " ⏳"
            elif doc.get("summary_status") == "error":
                summary_badge = " ❌"

            # Colored square file icon
            icon_color = "#22C55E" if is_ready else "#EF4444"
            icon_bg    = "rgba(34,197,94,0.12)" if is_ready else "rgba(239,68,68,0.12)"
            icon_col, btn_col = st.columns([0.5, 10.5], gap="small")
            with icon_col:
                st.markdown(
                    f'<div style="margin-top:9px;line-height:0;flex-shrink:0">'
                    f'<svg width="14" height="18" viewBox="0 0 14 18" fill="none" xmlns="http://www.w3.org/2000/svg">'
                    f'<path d="M0 0h9l5 5v13H0z" fill="{icon_color}"/>'
                    f'<path d="M9 0l5 5H9z" fill="rgba(0,0,0,0.2)"/>'
                    f'<line x1="2" y1="9" x2="12" y2="9" stroke="white" stroke-width="1.2" stroke-linecap="round"/>'
                    f'<line x1="2" y1="12" x2="12" y2="12" stroke="white" stroke-width="1.2" stroke-linecap="round"/>'
                    f'<line x1="2" y1="15" x2="8" y2="15" stroke="white" stroke-width="1.2" stroke-linecap="round"/>'
                    f'</svg></div>',
                    unsafe_allow_html=True,
                )
            with btn_col:
                label = f"{doc['folder_name']}{summary_badge}"
                btn_type = "primary" if is_selected else "secondary"
                if is_ready:
                    if st.button(label, key=f"doc_btn_{doc['folder_name']}",
                                 use_container_width=True, type=btn_type):
                        st.session_state[key] = not is_selected
                        st.rerun()
                else:
                    st.button(label, key=f"doc_btn_{doc['folder_name']}",
                              use_container_width=True, disabled=True)

    # Derive selected from ALL documents (not just filtered view)
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
        if user_choice == "reuse":
            # Nothing to rebuild — load existing metadata and return to main page
            with st.spinner("Loading existing document..."):
                metadata = pipeline.prepare_document(
                    config_path=CONFIG_PATH,
                    file_name=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    user_choice="reuse",
                )
            reset_chat()
            if metadata.get("ready_to_chat"):
                st.toast(f"`{metadata['document_name']}` is ready!")
            st.rerun()  # closes dialog, no pdf_processing flag set
        else:
            # Full pipeline needed — hand off to the processing state machine
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
        return

    if st.session_state["url_error"]:
        st.error(st.session_state["url_error"])
        st.session_state["url_error"] = ""

    url = st.text_input(
        "Web page or PDF URL",
        placeholder="https://example.com/report.pdf  or  https://example.com/article",
        key=f"url_input_{st.session_state['url_input_key']}",
    ).strip()

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

    if st.button("Process URL", disabled=not bool(url), use_container_width=True, key="url_submit"):
        st.session_state["pending_url"] = url
        st.session_state["pending_user_choice"] = user_choice
        st.session_state["url_processing"] = True
        st.session_state["show_url_panel"] = False
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

def _scroll_chat_to_bottom() -> None:
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        function scroll() {
            try {
                var d = window.parent.document;
                var marker = d.getElementById('chat-messages-area');
                if (!marker) return;
                var el = marker;
                while (el && !(el.getAttribute && el.getAttribute('data-testid') === 'stVerticalBlock')) {
                    el = el.parentElement;
                }
                if (el) el.scrollTop = el.scrollHeight;
            } catch(e) {}
        }
        scroll();
        setTimeout(scroll, 200);
        setTimeout(scroll, 700);
    })();
    </script>
    """, height=0)


def render_chat(question: str | None = None) -> None:
    selected_folders = st.session_state.get("selected_document_folders", [])
    chat_mode = st.session_state.get("chat_mode", "routing")
    active_folders = st.session_state.get("active_deep_search_folders", [])

    if len(selected_folders) == 1:
        _render_single_doc_chat(selected_folders[0], question)
        _scroll_chat_to_bottom()
    elif len(selected_folders) > 1:
        _render_direct_multi_doc_chat(selected_folders, question)
        _scroll_chat_to_bottom()
    elif chat_mode == "deep_search" and active_folders:
        _render_deep_search_chat(active_folders, question)
        _scroll_chat_to_bottom()
    else:
        _render_routing_chat(question)
        _scroll_chat_to_bottom()


def _render_single_doc_chat(document_folder: str, question: str | None = None) -> None:
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


def _render_direct_multi_doc_chat(document_folders: list[str], question: str | None = None) -> None:
    doc_names = [pipeline.load_document(f).get("document_name", Path(f).name) for f in document_folders]
    st.subheader(f"Multi-doc mode — {len(document_folders)} documents")
    st.caption(", ".join(doc_names))

    _render_message_history()

    if not question:
        return

    _process_multi_doc_question(question, document_folders)


def _render_routing_chat(question: str | None = None) -> None:
    st.subheader("Find relevant documents")
    st.caption(
        "Ask a question and the system will search across all document summaries "
        "to identify which documents are relevant — then you confirm which to search."
    )

    _render_message_history()
    _render_routing_candidates()

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


def _render_deep_search_chat(active_folders: list[str], question: str | None = None) -> None:
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
    '<span class="profrag-navbar-title">ProfRAG</span>'
    "</div>"
)

_APP_CSS = """
<style>
/* Design tokens */
:root {
    --c-accent:    #4F6EF7;
    --c-success:   #22C55E;
    --c-error:     #EF4444;
    --c-warning:   #F59E0B;
    --c-text1:     #111827;
    --c-text2:     #6B7280;
    --c-border:    #E5E7EB;
    --c-bg-page:   #F3F4F6;
    --c-pane-doc:  #FFFFFF;
    --c-pane-sum:  #FFFFFF;
    --c-pane-chat: #F7F8FA;
    --navbar-h:    52px;
    --bar-h:       68px;
}

/* Navbar */
.profrag-navbar {
    position: fixed; top: 0; left: 0; right: 0;
    height: 52px; z-index: 999999;
    background: #ffffff;
    border-bottom: 2px solid #000;
    display: flex; align-items: center;
    padding: 0 1.5rem;
}
.profrag-navbar-title {
    color: var(--c-text1);
    font-size: 1.1rem; font-weight: 700; letter-spacing: 0.01em;
    font-family: 'Inter','Segoe UI',system-ui,sans-serif;
}

/* Page lock */
html, body, .stApp { overflow: hidden !important; height: 100% !important; }
section[data-testid="stMain"] { background: var(--c-bg-page) !important; overflow: hidden !important; padding-bottom: 0 !important; margin-bottom: 0 !important; }
section[data-testid="stMain"] > div { padding-bottom: 0 !important; margin-bottom: 0 !important; }

/* Main layout */
[data-testid="stMainBlockContainer"], .block-container {
    padding: var(--navbar-h) 0 0 0 !important; max-width: 100% !important;
}
[data-testid="stMainBlockContainer"] > div:first-child { margin-top: 0 !important; padding-top: 0 !important; }
/* #chat-col-marker is always present — reliably targets the main 3-column block */
[data-testid="stHorizontalBlock"]:has(#chat-col-marker),
[data-testid="stHorizontalBlock"]:has(#doc-col-marker),
[data-testid="stHorizontalBlock"]:has(#doc-tab-marker) {
    margin-top: 0 !important; padding-top: 0 !important;
    margin-bottom: 0 !important; padding-bottom: 0 !important;
    gap: 0 !important; column-gap: 0 !important; align-items: stretch !important;
}
/* Zero only horizontal padding on column flex items — don't touch vertical/margin */
[data-testid="stHorizontalBlock"]:has(#chat-col-marker) > [data-testid="stColumn"],
[data-testid="stHorizontalBlock"]:has(#doc-col-marker) > [data-testid="stColumn"],
[data-testid="stHorizontalBlock"]:has(#doc-tab-marker) > [data-testid="stColumn"] {
    padding-left: 0 !important; padding-right: 0 !important;
}

/* Pane heights */
[data-testid="stColumn"]:has(#doc-col-marker),
[data-testid="stColumn"]:has(#summary-pane-marker),
[data-testid="stColumn"]:has(#chat-col-marker),
[data-testid="stColumn"]:has(#doc-tab-marker),
[data-testid="stColumn"]:has(#sum-tab-marker) {
    height: calc(100vh - var(--navbar-h)) !important; overflow: hidden;
}
[data-testid="stColumn"]:has(#doc-col-marker),
[data-testid="stColumn"]:has(#summary-pane-marker) {
    overflow-y: auto !important; overflow-x: hidden !important;
}

/* Doc pane: white */
[data-testid="stColumn"]:has(#doc-col-marker) {
    background: var(--c-pane-doc) !important;
    border-radius: 0 !important;
    border-right: 1px solid #000 !important;
}
[data-testid="stColumn"]:has(#doc-col-marker) > div { padding-left: 0.875rem !important; padding-right: 0.875rem !important; }
[data-testid="stColumn"]:has(#doc-col-marker) p,
[data-testid="stColumn"]:has(#doc-col-marker) span,
[data-testid="stColumn"]:has(#doc-col-marker) h1,
[data-testid="stColumn"]:has(#doc-col-marker) h2,
[data-testid="stColumn"]:has(#doc-col-marker) h3,
[data-testid="stColumn"]:has(#doc-col-marker) h4,
[data-testid="stColumn"]:has(#doc-col-marker) label,
[data-testid="stColumn"]:has(#doc-col-marker) small,
[data-testid="stColumn"]:has(#doc-col-marker) li,
[data-testid="stColumn"]:has(#doc-col-marker) strong,
[data-testid="stColumn"]:has(#doc-col-marker) a { color: var(--c-text1) !important; }
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stCaptionContainer"] p,
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stCaptionContainer"] span { color: var(--c-text2) !important; }
[data-testid="stColumn"]:has(#doc-col-marker) hr { border-color: var(--c-border) !important; }
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important; border: none !important; box-shadow: none !important;
}
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stTextInputRootElement"] input {
    background: #F9FAFB !important; border: 1px solid #000 !important;
    border-radius: 0 !important; color: var(--c-text1) !important;
}
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stTextInputRootElement"] input::placeholder { color: var(--c-text2) !important; }
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stBaseButton-secondary"] {
    background: transparent !important; border: none !important; box-shadow: none !important;
    border-radius: 8px !important; color: var(--c-text1) !important;
    text-align: left !important; font-size: 0.85rem !important; transition: background 0.12s !important;
}
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stBaseButton-secondary"]:hover { background: #F3F4F6 !important; }
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stBaseButton-primary"] {
    background: rgba(79,110,247,0.10) !important; color: var(--c-accent) !important;
    border: none !important; border-radius: 8px !important; font-size: 0.85rem !important;
    text-align: left !important; box-shadow: none !important;
}
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stBaseButton-secondary"] p,
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stBaseButton-primary"] p {
    overflow: hidden !important; text-overflow: ellipsis !important;
    white-space: nowrap !important; max-width: 100% !important; margin: 0 !important;
}
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stAlert"],
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stAlert"] p { color: var(--c-text1) !important; }
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stExpander"] summary,
[data-testid="stColumn"]:has(#doc-col-marker) [data-testid="stExpander"] summary p { color: var(--c-text1) !important; }

/* Sticky doc header */
[data-testid="stHorizontalBlock"]:has(#doc-collapse-sticky-marker) {
    position: sticky !important; top: 0 !important; z-index: 10 !important;
    background: var(--c-pane-doc) !important; padding-bottom: 2px !important;
    align-items: center !important;
}
/* Collapse/expand buttons — borderless, icon-only */
[data-testid="stHorizontalBlock"]:has(#doc-collapse-sticky-marker) [data-testid="stBaseButton-secondary"],
[data-testid="stHorizontalBlock"]:has(#sum-close-sticky-marker) [data-testid="stBaseButton-secondary"],
[data-testid="stHorizontalBlock"]:has(#doc-collapse-sticky-marker) button,
[data-testid="stHorizontalBlock"]:has(#sum-close-sticky-marker) button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    outline: none !important; color: var(--c-text2) !important;
    font-size: 1rem !important; padding: 4px 8px !important;
}
[data-testid="stHorizontalBlock"]:has(#doc-collapse-sticky-marker) [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stHorizontalBlock"]:has(#sum-close-sticky-marker) [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stHorizontalBlock"]:has(#doc-collapse-sticky-marker) button:hover,
[data-testid="stHorizontalBlock"]:has(#sum-close-sticky-marker) button:hover {
    background: rgba(0,0,0,0.05) !important; border: none !important;
}
/* Zero out h3 margin so title and button are the same height in the flex row */
[data-testid="stHorizontalBlock"]:has(#doc-collapse-sticky-marker) h3,
[data-testid="stHorizontalBlock"]:has(#sum-close-sticky-marker) h3 {
    margin-top: 0 !important; margin-bottom: 0 !important; line-height: 1.2 !important;
}

/* Summary pane */
[data-testid="stColumn"]:has(#summary-pane-marker) {
    background: var(--c-pane-sum) !important;
    border-radius: 0 !important;
    border-left: 1px solid #000 !important;
    border-right: 1px solid #000 !important;
}
[data-testid="stColumn"]:has(#summary-pane-marker) > div { padding-left: 0.875rem !important; padding-right: 0.875rem !important; }
[data-testid="stColumn"]:has(#summary-pane-marker) p,
[data-testid="stColumn"]:has(#summary-pane-marker) span,
[data-testid="stColumn"]:has(#summary-pane-marker) label,
[data-testid="stColumn"]:has(#summary-pane-marker) strong,
[data-testid="stColumn"]:has(#summary-pane-marker) li { color: var(--c-text1) !important; }
[data-testid="stColumn"]:has(#summary-pane-marker) h1,
[data-testid="stColumn"]:has(#summary-pane-marker) h2,
[data-testid="stColumn"]:has(#summary-pane-marker) h3,
[data-testid="stColumn"]:has(#summary-pane-marker) h4 { color: var(--c-text1) !important; }
[data-testid="stColumn"]:has(#summary-pane-marker) [data-testid="stCaptionContainer"] p,
[data-testid="stColumn"]:has(#summary-pane-marker) [data-testid="stCaptionContainer"] span { color: var(--c-text2) !important; }
[data-testid="stHorizontalBlock"]:has(#sum-close-sticky-marker) {
    position: sticky !important; top: 0 !important; z-index: 10 !important;
    background: var(--c-pane-sum) !important; padding-bottom: 2px !important;
    align-items: center !important;
}

/* Chat pane */
[data-testid="stColumn"]:has(#chat-col-marker) {
    background: #ffffff !important;
    border-radius: 0 !important;
    border-left: 1px solid #000 !important;
}
[data-testid="stColumn"]:has(#chat-col-marker) > div {
    padding-left: 1rem !important; padding-right: 1rem !important;
    padding-bottom: 0 !important;
}
[data-testid="stColumn"]:has(#chat-col-marker) p,
[data-testid="stColumn"]:has(#chat-col-marker) span,
[data-testid="stColumn"]:has(#chat-col-marker) label,
[data-testid="stColumn"]:has(#chat-col-marker) strong,
[data-testid="stColumn"]:has(#chat-col-marker) li { color: var(--c-text1) !important; }
[data-testid="stColumn"]:has(#chat-col-marker) h1,
[data-testid="stColumn"]:has(#chat-col-marker) h2,
[data-testid="stColumn"]:has(#chat-col-marker) h3,
[data-testid="stColumn"]:has(#chat-col-marker) h4 { color: var(--c-text1) !important; }
[data-testid="stColumn"]:has(#chat-col-marker) [data-testid="stCaptionContainer"] p,
[data-testid="stColumn"]:has(#chat-col-marker) [data-testid="stCaptionContainer"] span {
    color: var(--c-text2) !important; font-size: 0.88rem !important;
}

/* Collapsed tab strips */
[data-testid="stColumn"]:has(#doc-tab-marker) {
    background: var(--c-pane-doc) !important; border-radius: 0 !important;
    border-right: 1px solid #000 !important;
    position: relative !important;
    display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
}
[data-testid="stColumn"]:has(#sum-tab-marker) {
    background: var(--c-pane-sum) !important; border-radius: 0 !important;
    border-left: 1px solid #000 !important;
    border-right: 1px solid #000 !important;
    position: relative !important;
    display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
}
[data-testid="stColumn"]:has(#doc-tab-marker)::before {
    content: 'DOCUMENTS';
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%) rotate(-90deg);
    color: #9CA3AF; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.06em; white-space: nowrap; pointer-events: none;
}
[data-testid="stColumn"]:has(#sum-tab-marker)::before {
    content: 'SUMMARIES';
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%) rotate(-90deg);
    color: #9CA3AF; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.06em; white-space: nowrap; pointer-events: none;
}
[data-testid="stColumn"]:has(#doc-tab-marker) > div,
[data-testid="stColumn"]:has(#sum-tab-marker) > div,
[data-testid="stColumn"]:has(#doc-tab-marker) > div > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stColumn"]:has(#sum-tab-marker) > div > [data-testid="stVerticalBlockBorderWrapper"] {
    display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
    flex: 1 1 auto !important; height: 100% !important;
}
[data-testid="stColumn"]:has(#doc-tab-marker) [data-testid="stVerticalBlock"],
[data-testid="stColumn"]:has(#sum-tab-marker) [data-testid="stVerticalBlock"] {
    padding: 0 !important; gap: 0 !important;
    display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
    height: 100% !important;
}
[data-testid="stColumn"]:has(#doc-tab-marker) [data-testid="stMarkdownContainer"],
[data-testid="stColumn"]:has(#sum-tab-marker) [data-testid="stMarkdownContainer"] {
    flex: 0 0 auto !important; height: auto !important;
}
[data-testid="stColumn"]:has(#doc-tab-marker) button,
[data-testid="stColumn"]:has(#sum-tab-marker) button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: var(--c-text2) !important; font-size: 1.1rem !important; font-weight: 600 !important;
    min-height: 44px !important; width: 100% !important; padding: 6px 0 !important;
}
[data-testid="stColumn"]:has(#doc-tab-marker) button:hover,
[data-testid="stColumn"]:has(#sum-tab-marker) button:hover {
    background: rgba(0,0,0,0.04) !important; color: var(--c-text1) !important;
}

/* Summary progress pulse */
@keyframes pulse-opacity { 0%,100%{opacity:1} 50%{opacity:0.35} }
.summary-progress-badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    background: rgba(245,158,11,0.12); color: #B45309;
    animation: pulse-opacity 2s ease-in-out infinite;
}

/* Chat column — two-section layout: messages area + fixed bar below */
[data-testid="stColumn"]:has(#chat-col-marker) {
    position: relative !important;
    overflow: hidden !important;
}
/* Messages area: absolute, fills the column above the bar */
[data-testid="stVerticalBlockBorderWrapper"]:has(#chat-messages-area) {
    position: absolute !important;
    top: 0 !important; left: 0 !important; right: 0 !important; bottom: var(--bar-h) !important;
    overflow: hidden !important;
    background: transparent !important; border: none !important;
    padding: 0 !important; margin: 0 !important;
}
[data-testid="stVerticalBlock"]:has(#chat-messages-area) {
    height: 100% !important;
    overflow-y: auto !important; overflow-x: hidden !important;
    padding-bottom: 12px !important;
}

/* Action bar — scoped inside chat column so it never matches the main layout block */
[data-testid="stColumn"]:has(#chat-col-marker) [data-testid="stHorizontalBlock"]:has(#action-bar-marker) {
    position: absolute !important;
    bottom: 0 !important; left: 0 !important; right: 0 !important;
    background: #ffffff !important;
    border: none !important; border-top: 1px solid #000 !important; border-radius: 0 !important;
    padding: 12px 16px !important; gap: 12px !important;
    align-items: center !important; min-height: 68px !important;
    box-shadow: none !important; margin: 0 !important; z-index: 100 !important;
}
[data-testid="stColumn"]:has(#chat-col-marker) [data-testid="stHorizontalBlock"]:has(#action-bar-marker) > [data-testid="stColumn"] {
    flex-shrink: 0 !important;
}
[data-testid="stColumn"]:has(#chat-col-marker) [data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stBaseButton-secondary"] {
    background: #F3F4F6 !important; border: 1px solid var(--c-border) !important;
    border-radius: 10px !important;
    min-height: 38px !important; max-height: 38px !important; min-width: 38px !important;
    color: var(--c-text2) !important; font-size: 1.2rem !important;
    box-shadow: none !important; padding: 0 !important; transition: background 0.12s !important;
}
[data-testid="stColumn"]:has(#chat-col-marker) [data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stBaseButton-secondary"]:hover { background: #E5E7EB !important; }
[data-testid="stColumn"]:has(#chat-col-marker) [data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stForm"] {
    background: transparent !important; border: none !important; padding: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stTextInputRootElement"],
[data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stTextInputRootElement"] > div {
    background: transparent !important; border: none !important; box-shadow: none !important;
}
[data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stTextInputRootElement"] input {
    border: none !important; background: transparent !important; box-shadow: none !important;
    font-size: 0.95rem !important; color: var(--c-text1) !important; padding: 6px 10px !important;
}
[data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stTextInputRootElement"] input:focus {
    outline: none !important; box-shadow: none !important; border: none !important;
}
[data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stTextInputRootElement"] label { display: none !important; }
[data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stFormSubmitButton"] button {
    background: var(--c-accent) !important; color: #fff !important;
    border-radius: 10px !important; min-height: 38px !important; max-height: 38px !important;
    padding: 0 20px !important; border: none !important;
    font-size: 0.9rem !important; font-weight: 600 !important;
    box-shadow: none !important; transition: background 0.12s !important; white-space: nowrap !important;
}
[data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stFormSubmitButton"] button:hover { background: #3B55D4 !important; }
[data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stForm"] [data-testid="stVerticalBlock"] {
    display: flex !important; flex-direction: row !important;
    align-items: center !important; gap: 6px !important;
}
[data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stTextInputRootElement"] {
    flex: 1 1 auto !important; min-width: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stFormSubmitButton"] {
    flex: 0 0 auto !important; width: auto !important;
}
[data-testid="stHorizontalBlock"]:has(#action-bar-marker) [data-testid="stFormSubmitButton"] button {
    width: auto !important;
}


/* Kill Streamlit's injected top spacing at every nesting level */
section[data-testid="stMain"] > div { margin-top: 0 !important; padding-top: 0 !important; }
[data-testid="stMainBlockContainer"] > div,
[data-testid="stMainBlockContainer"] > div > div { margin-top: 0 !important; padding-top: 0 !important; }
/* Chat column inner wrappers fill full height so absolute children anchor correctly */
[data-testid="stColumn"]:has(#chat-col-marker) > div,
[data-testid="stColumn"]:has(#chat-col-marker) > div > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stColumn"]:has(#chat-col-marker) > div > [data-testid="stVerticalBlock"] {
    height: 100% !important;
}

/* Responsive */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"]:has(#doc-col-marker),
    [data-testid="stHorizontalBlock"]:has(#chat-col-marker) { flex-wrap: wrap !important; }
    [data-testid="stColumn"]:has(#doc-col-marker),
    [data-testid="stColumn"]:has(#doc-tab-marker),
    [data-testid="stColumn"]:has(#summary-pane-marker),
    [data-testid="stColumn"]:has(#sum-tab-marker),
    [data-testid="stColumn"]:has(#chat-col-marker) {
        min-width: 100% !important; width: 100% !important;
        height: auto !important; max-height: 50vh; border-radius: 12px !important;
    }
}
</style>

"""



@st.dialog("Add Document", width="large")
def show_add_document_dialog() -> None:
    """Modal overlay with Upload PDF / From URL tabs."""
    tab_pdf, tab_url = st.tabs(["📎 Upload PDF", "🔗 From URL"])
    with tab_pdf:
        render_upload_panel()
    with tab_url:
        render_url_panel()


def render_action_bar() -> str | None:
    """Floating bar inside chat column: + button opens modal | chat input | send."""
    selected  = st.session_state.get("selected_document_folders", [])
    chat_mode = st.session_state.get("chat_mode", "routing")
    active_ds = st.session_state.get("active_deep_search_folders", [])

    if len(selected) == 1:
        placeholder = "Ask a question about this document..."
    elif len(selected) > 1:
        placeholder = "Ask a question across selected documents..."
    elif chat_mode == "deep_search" and active_ds:
        placeholder = "Ask a follow-up...  (/back to change documents)"
    else:
        placeholder = "Ask anything..."

    c_plus, c_form = st.columns([1, 22])

    with c_plus:
        st.markdown('<span id="action-bar-marker"></span>', unsafe_allow_html=True)
        if st.button("+", key="bar_plus_btn", use_container_width=True,
                     type="secondary"):
            show_add_document_dialog()

    with c_form:
        with st.form("chat_bar_form", clear_on_submit=True, border=False):
            question = st.text_input(
                "Ask",
                placeholder=placeholder,
                label_visibility="collapsed",
                key="bar_chat_field",
            )
            submitted = st.form_submit_button("↗ Send")

        if submitted and question:
            return question.strip()

    return None


def _maybe_auto_refresh() -> None:
    """Poll summarization daemons and rerun every 3 s while any are active."""
    import time
    active = st.session_state.get("active_summarization_folders", set())
    if not active:
        return
    still_running = {
        f for f in active
        if pipeline.load_document(f).get("summary_status") == "in_progress"
    }
    st.session_state["active_summarization_folders"] = still_running
    if still_running:
        time.sleep(3)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="ProfRAG", page_icon="📄", layout="wide")
    st.markdown(_APP_CSS, unsafe_allow_html=True)
    st.markdown(_NAVBAR_HTML, unsafe_allow_html=True)
    initialize_state()

    pending_q = st.session_state.pop("_bar_question", None) or None

    # Re-open the upload dialog if processing is still active (st.rerun() inside a
    # @st.dialog closes it, so we call it again here to keep the spinner visible).
    if st.session_state.get("pdf_processing") or st.session_state.get("url_processing"):
        show_add_document_dialog()

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
            coll_space, coll_col = st.columns([6, 1], vertical_alignment="center")
            with coll_space:
                st.markdown('<span id="doc-collapse-sticky-marker"></span>', unsafe_allow_html=True)
                st.subheader("Documents")
            with coll_col:
                if st.button("◀", key="doc_collapse", use_container_width=True,
                             help="Collapse Documents"):
                    st.session_state["show_doc_pane"] = False
                    st.rerun()
            render_existing_documents()

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
        with st.container():
            st.markdown('<span id="chat-messages-area"></span>', unsafe_allow_html=True)
            render_chat(question=pending_q)
            question = render_action_bar()
        if question:
            st.session_state["_bar_question"] = question
            st.rerun()

    _maybe_auto_refresh()


if __name__ == "__main__":
    main()
