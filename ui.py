from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import streamlit as st

import main as pipeline


CONFIG_PATH = Path(__file__).parent / "config" / "app_config.yaml"
_CITE_TOKEN_RE = re.compile(r"\[Source:\s*([^\]]+)\]")


# ── Session state ────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "messages": [],
    "upload_key": 0,
    "url_input_key": 0,
    "url_processing": False,
    "pending_url": "",
    "pending_user_choice": "new_version",
    "url_error": "",
    "pdf_processing": False,
    "pending_pdf_name": "",
    "pending_pdf_bytes": b"",
    "pending_pdf_choice": "new_version",
    "pdf_info": "",
    "selected_document_folders": [],
    "chat_mode": "routing",
    "active_deep_search_folders": [],
    "pending_routing_question": "",
    "routing_candidates": [],
    "show_doc_pane": True,
    "show_summary_pane": True,
    "_bar_question": "",
    "_sum_pane_active_doc": "",
    "_sum_pane_active_level": "",
    # Daemon threads die on process exit, so any "in_progress" not in this
    # set is stale from a previous session.
    "active_summarization_folders": set(),
}


def initialize_state() -> None:
    for key, default in _DEFAULTS.items():
        st.session_state.setdefault(key, default)


def reset_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["routing_candidates"] = []
    st.session_state["pending_routing_question"] = ""


def _exit_to_routing() -> None:
    """Drop deep_search state and return to routing mode."""
    st.session_state["chat_mode"] = "routing"
    st.session_state["active_deep_search_folders"] = []
    reset_chat()


def _track_summarization(folder: str) -> None:
    """Start background summarization and register in the session-level tracking set."""
    pipeline.start_summarization_background(CONFIG_PATH, folder)
    st.session_state["active_summarization_folders"].add(str(folder))


def _marker(name: str) -> None:
    """Emit an invisible marker span used by CSS :has() selectors to scope styles."""
    st.markdown(f'<span id="{name}"></span>', unsafe_allow_html=True)


def _set_sum_active(doc: str = "", level: str = "") -> None:
    """Set or clear the summary-pane's active doc/level (defaults clear both)."""
    st.session_state["_sum_pane_active_doc"]   = doc
    st.session_state["_sum_pane_active_level"] = level


def _chat_state() -> tuple[list, str, list]:
    """Return (selected_folders, chat_mode, active_deep_search_folders)."""
    return (
        st.session_state.get("selected_document_folders", []),
        st.session_state.get("chat_mode", "routing"),
        st.session_state.get("active_deep_search_folders", []),
    )


# ── Summary helpers ───────────────────────────────────────────────────────────

_PROGRESS_LABELS: tuple[tuple[str, str], ...] = (
    ("level1_indexed",  "finalizing"),
    ("level1_complete", "indexing L1"),
    ("level2_complete", "running L1"),
    ("level3_indexed",  "running L2"),
    ("level3_complete", "indexing L3"),
)


def _summary_progress_label(metadata: dict) -> str:
    p = metadata.get("summary_progress", {})
    return next((label for flag, label in _PROGRESS_LABELS if p.get(flag)), "starting")


# ── Summary pane (middle column) ─────────────────────────────────────────────

_LEVEL_FILES = {
    "level1": "level1_onepager.json",
    "level2": "level2_medium.json",
    "level3": "level3_detailed.json",
}
_LEVEL_SHORT = {"level1": "1-Pager", "level2": "Medium", "level3": "Detailed"}
_LEVEL_LONG  = {"level1": "1-Pager Summary", "level2": "Medium Summary", "level3": "Detailed Summary"}


def _render_summary_doc_card(folder: str, active_doc: str, active_level: str) -> None:
    """Render one document's status + level-button row in the summary pane."""
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
                        _set_sum_active() if is_active else _set_sum_active(folder, level_key)
                        st.rerun()
                else:
                    st.button(_LEVEL_SHORT[level_key], key=btn_key,
                              use_container_width=True, disabled=True)

    st.divider()


def render_summary_pane(selected_folders: list[str]) -> None:
    from core.storage import read_json

    _marker("summary-pane-marker")

    col_title, col_close = st.columns([4, 1], vertical_alignment="center")
    with col_title:
        _marker("sum-close-sticky-marker")
        st.subheader("Summaries")
    with col_close:
        if st.button("◀", key="sum_pane_close", use_container_width=True):
            st.session_state["show_summary_pane"] = False
            st.rerun()

    watcher = _watcher_status_snapshot()
    st.caption(
        f"Watcher: queued `{watcher['queued']}` | running `{watcher['running']}` | failed `{len(watcher['failed'])}`"
    )
    if watcher["failed"]:
        if st.button("Retry failed summaries", key="retry_failed_summaries", use_container_width=True):
            for doc in watcher["failed"]:
                folder = doc.get("document_folder", "")
                if folder:
                    _track_summarization(folder)
            st.rerun()

    # Clear active selection when its document is no longer selected
    active_doc   = st.session_state.get("_sum_pane_active_doc", "")
    active_level = st.session_state.get("_sum_pane_active_level", "")
    if active_doc and active_doc not in selected_folders:
        _set_sum_active()
        active_doc = active_level = ""

    if not selected_folders:
        st.caption("Select documents from the left panel to view their summaries here.")

    for folder in selected_folders:
        _render_summary_doc_card(folder, active_doc, active_level)

    # ── Shared summary content area (bottom, collapsible) ────────────────────
    active_doc   = st.session_state.get("_sum_pane_active_doc", "")
    active_level = st.session_state.get("_sum_pane_active_level", "")

    if active_doc and active_level:
        summaries_dir = Path(active_doc) / "summaries"
        data = {}
        try:
            data = read_json(str(summaries_dir / _LEVEL_FILES[active_level]), default={})
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
                _set_sum_active()
                st.rerun()


# ── Left panel: existing documents ───────────────────────────────────────────

_STATUS_BADGES = {"in_progress": " ⏳", "error": " ❌"}


_DOC_ICON_SVG = (
    '<div style="margin-top:9px;line-height:0;flex-shrink:0">'
    '<svg width="14" height="18" viewBox="0 0 14 18" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M0 0h9l5 5v13H0z" fill="{color}"/>'
    '<path d="M9 0l5 5H9z" fill="rgba(0,0,0,0.2)"/>'
    '<line x1="2" y1="9" x2="12" y2="9" stroke="white" stroke-width="1.2" stroke-linecap="round"/>'
    '<line x1="2" y1="12" x2="12" y2="12" stroke="white" stroke-width="1.2" stroke-linecap="round"/>'
    '<line x1="2" y1="15" x2="8" y2="15" stroke="white" stroke-width="1.2" stroke-linecap="round"/>'
    '</svg></div>'
)


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
            _exit_to_routing()
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

            summary_badge = (
                " 📝" if doc.get("summary_ready")
                else _STATUS_BADGES.get(doc.get("summary_status", ""), "")
            )

            icon_color = "#22C55E" if is_ready else "#EF4444"
            icon_col, btn_col = st.columns([0.5, 10.5], gap="small")
            with icon_col:
                st.markdown(_DOC_ICON_SVG.format(color=icon_color), unsafe_allow_html=True)
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


# ── Sidebar: document ingest ─────────────────────────────────────────────────

def _post_ingest(metadata: dict) -> None:
    """Shared post-ingest tail: track summarization, reset chat, toast if ready."""
    if metadata.get("summary_status") == "in_progress":
        st.session_state["active_summarization_folders"].add(metadata.get("document_folder", ""))
    reset_chat()
    if metadata.get("ready_to_chat"):
        st.toast(f"`{metadata['document_name']}` is ready!")


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
                _post_ingest(metadata)
                if not metadata.get("ready_to_chat"):
                    st.session_state["pdf_info"] = (
                        f"`{metadata['document_name']}` resumed to step "
                        f"`{metadata.get('last_successful_step', 'unknown')}`."
                    )
            finally:
                st.session_state.update({
                    "pdf_processing": False,
                    "pending_pdf_name": "",
                    "pending_pdf_bytes": b"",
                    "pending_pdf_choice": "new_version",
                    "upload_key": st.session_state["upload_key"] + 1,
                })
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
            with st.spinner("Loading existing document..."):
                metadata = pipeline.prepare_document(
                    config_path=CONFIG_PATH,
                    file_name=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    user_choice="reuse",
                )
            _post_ingest(metadata)
            st.rerun()
        else:
            st.session_state.update({
                "pending_pdf_name": uploaded_file.name,
                "pending_pdf_bytes": uploaded_file.getvalue(),
                "pending_pdf_choice": user_choice,
                "pdf_processing": True,
            })
            st.rerun()


def render_url_panel() -> None:
    if st.session_state["url_processing"]:
        st.text_input(
            "Web page or PDF URL",
            value="", disabled=True, placeholder="Processing...",
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
                _post_ingest(metadata)
                st.session_state["url_input_key"] += 1
            except Exception as exc:
                st.session_state["url_error"] = str(exc)
            finally:
                st.session_state.update({
                    "url_processing": False,
                    "pending_url": "",
                    "pending_user_choice": "new_version",
                })
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
        same = pipeline.inspect_same_name_document(CONFIG_PATH, pipeline.url_ingest.url_to_document_name(url))
        if same and same.get("ready_to_chat"):
            user_choice = st.radio(
                "A stored document with the same URL already exists.",
                options=["reuse", "rebuild"],
                captions=["Use the existing processed version.", "Re-scrape and reprocess."],
                horizontal=False, key="url_radio",
            )
        elif same and not same.get("ready_to_chat"):
            st.info("Existing document not fully processed — will resume from last step.")
            user_choice = "reuse"

    if st.button("Process URL", disabled=not bool(url), use_container_width=True, key="url_submit"):
        st.session_state.update({
            "pending_url": url,
            "pending_user_choice": user_choice,
            "url_processing": True,
        })
        st.rerun()


# ── Chat helpers ─────────────────────────────────────────────────────────────

def _render_images_expander(image_paths: list[str]) -> None:
    if not image_paths:
        return
    with st.expander(f"📷 {len(image_paths)} image(s)"):
        cols = st.columns(min(len(image_paths), 3))
        for i, img_path in enumerate(image_paths):
            cols[i % 3].image(img_path, use_container_width=True)


def _render_sources_expander(sources: list[dict]) -> None:
    if not sources:
        return
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


def _render_answer_with_clickable_citations(answer: str, anchor_prefix: str = "cite") -> None:
    """Render answer text with inline [Source: ...] tokens as clickable references."""
    text = (answer or "").strip()
    if not text:
        st.markdown("No answer generated.")
        return

    cites: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        if label not in cites:
            cites.append(label)
        idx = cites.index(label) + 1
        return f"[{idx}](#{anchor_prefix}-{idx})"

    rendered = _CITE_TOKEN_RE.sub(_replace, text)
    st.markdown(rendered)

    if cites:
        st.markdown("**References**")
        for idx, label in enumerate(cites, start=1):
            st.markdown(f'<a id="{anchor_prefix}-{idx}"></a>{idx}. `{label}`', unsafe_allow_html=True)


def _render_message_history() -> None:
    for i, message in enumerate(st.session_state["messages"], start=1):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_answer_with_clickable_citations(message["content"], anchor_prefix=f"cite-msg-{i}")
            else:
                st.markdown(message["content"])
            _render_images_expander(message.get("image_paths", []))
            _render_sources_expander(message.get("chunk_sources", []))


def _watcher_status_snapshot() -> dict[str, Any]:
    docs = pipeline.list_documents(CONFIG_PATH)
    queued = sum(
        1 for d in docs
        if d.get("ready_to_chat") and d.get("summary_status", "pending") == "pending"
    )
    running = sum(1 for d in docs if d.get("summary_status") == "in_progress")
    failed = [d for d in docs if d.get("summary_status") == "error"]
    return {"queued": queued, "running": running, "failed": failed}


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
        label = f"**{cand['doc_name']}**{section_hint}  `relevance {score_pct}`"
        st.checkbox(label, key=key)
        with st.expander(f"Why this matched: {cand['doc_name']}", expanded=False):
            meta = pipeline.load_document(cand["folder"])
            card = (meta or {}).get("document_card", {}) or {}
            opening = (card.get("opening_text", "") or "").strip()
            l1_summary = (card.get("l1_summary", "") or "").strip()
            top_section = (cand.get("top_section", "") or "").strip()

            if top_section:
                st.caption(f"Matched section signal: {top_section}")
            if opening:
                opening_preview = opening[:220] + ("..." if len(opening) > 220 else "")
                st.caption(f"Opening text signal: {opening_preview}")
            if l1_summary:
                summary_preview = l1_summary[:300] + ("..." if len(l1_summary) > 300 else "")
                st.caption(f"L1 summary signal: {summary_preview}")

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
    """Scroll the chat messages container to the bottom. Runs three times to handle
    Streamlit's async post-injection rendering."""
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (() => {
      const scroll = () => {
        let el = window.parent.document.getElementById('chat-messages-area');
        while (el && el.getAttribute?.('data-testid') !== 'stVerticalBlock') el = el.parentElement;
        if (el) el.scrollTop = el.scrollHeight;
      };
      [0, 200, 700].forEach(t => setTimeout(scroll, t));
    })();
    </script>
    """, height=0)


def render_chat(question: str | None = None) -> None:
    selected_folders, chat_mode, active_folders = _chat_state()

    if len(selected_folders) == 1:
        _render_single_doc_chat(selected_folders[0], question)
    elif len(selected_folders) > 1:
        _render_direct_multi_doc_chat(selected_folders, question)
    elif chat_mode == "deep_search" and active_folders:
        _render_deep_search_chat(active_folders, question)
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
            sources = answer_payload.get("sources", [])
            image_paths = answer_payload.get("image_paths", [])
            if answer_payload.get("abstain"):
                answer_text = "I cannot answer this from the available document context."
                sources = []
                image_paths = []
            else:
                sources_text = "\n".join(
                    f"- {item.get('section_path') or ('Chunk ' + str(item.get('chunk_number', 'unknown')))}"
                    for item in sources
                )
                answer_text = f"{answer_payload.get('answer', 'No answer generated.')}\n\nSources:\n{sources_text}"

        st.markdown(answer_text)
        _render_images_expander(image_paths)

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
            _exit_to_routing()
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
        _exit_to_routing()
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
        confidence = payload.get("confidence", "")
        citation_warning = payload.get("citation_warning", "")

        if confidence == "low":
            st.warning("Low confidence: selected documents may only partially cover this question.")
        if citation_warning:
            st.info(f"Citation warning: {citation_warning}")
        _render_answer_with_clickable_citations(answer, anchor_prefix="cite-live")
        _render_sources_expander(sources)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "chunk_sources": sources,
        "confidence": confidence,
        "citation_warning": citation_warning,
    })


# ── App entry point ──────────────────────────────────────────────────────────

_NAVBAR_HTML = (
    '<div class="profrag-navbar">'
    '<span class="profrag-navbar-title">ProfRAG</span>'
    "</div>"
)

_APP_CSS = (
    "<style>" +
    (Path(__file__).parent / "static" / "styles.css").read_text(encoding="utf-8") +
    "</style>"
)



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
    selected, chat_mode, active_ds = _chat_state()

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
        _marker("action-bar-marker")
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
    pipeline.start_summary_watcher(CONFIG_PATH)

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
            _marker("doc-tab-marker")
            if st.button("▶", key="doc_expand", use_container_width=True,
                         help="Expand Documents"):
                st.session_state["show_doc_pane"] = True
                st.rerun()

    # ── Open doc pane ────────────────────────────────────────────────────────
    # Rendered first so render_existing_documents() writes the current checkbox
    # state into selected_document_folders before the summary pane reads it.
    if doc_col is not None:
        with doc_col:
            _marker("doc-col-marker")
            coll_space, coll_col = st.columns([6, 1], vertical_alignment="center")
            with coll_space:
                _marker("doc-collapse-sticky-marker")
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
            _marker("sum-tab-marker")
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
        _marker("chat-col-marker")
        with st.container():
            _marker("chat-messages-area")
            render_chat(question=pending_q)
            question = render_action_bar()
        if question:
            st.session_state["_bar_question"] = question
            st.rerun()

    _maybe_auto_refresh()


if __name__ == "__main__":
    main()
