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
    # Multi-doc / routing state
    st.session_state.setdefault("selected_document_folders", [])
    st.session_state.setdefault("chat_mode", "routing")        # "routing" | "deep_search"
    st.session_state.setdefault("active_deep_search_folders", [])
    st.session_state.setdefault("pending_routing_question", "")
    st.session_state.setdefault("routing_candidates", [])


def reset_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["routing_candidates"] = []
    st.session_state["pending_routing_question"] = ""


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


# ── Summary buttons ──────────────────────────────────────────────────────────

def render_summary_buttons(metadata: dict, document_folder: str) -> None:
    if not metadata.get("summary_ready"):
        status = metadata.get("summary_status", "pending")
        if status == "in_progress":
            st.caption("⏳ Generating summaries...")
        elif status == "error":
            st.caption("❌ Summary generation failed.")
            if st.button("Retry Summarization", use_container_width=True, key="sum_retry"):
                pipeline.start_summarization_background(CONFIG_PATH, document_folder)
                st.rerun()
        else:
            if st.button("Generate Summaries", type="primary", use_container_width=True, key="sum_generate"):
                pipeline.start_summarization_background(CONFIG_PATH, document_folder)
                st.rerun()
        return

    summary_paths = metadata.get("summary_paths", {})
    st.caption("Document Summaries")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("1-Pager", use_container_width=True, key="sum_l1"):
            st.session_state["_show_summary"] = "level1"
    with col2:
        if st.button("Medium", use_container_width=True, key="sum_l2"):
            st.session_state["_show_summary"] = "level2"
    with col3:
        if st.button("Detailed", use_container_width=True, key="sum_l3"):
            st.session_state["_show_summary"] = "level3"

    level_labels = {"level1": "1-Pager Summary", "level2": "Medium Summary", "level3": "Detailed Summary"}

    show = st.session_state.get("_show_summary")
    if show and show in summary_paths:
        path = summary_paths[show]
        data = {}
        try:
            from core.storage import read_json
            data = read_json(path, default={})
        except Exception:
            pass
        label = level_labels.get(show, show)
        with st.expander(label, expanded=True):
            if show == "level3":
                sections = data.get("sections", [])
                for sec in sections:
                    st.markdown(f"**{sec.get('section', '')}**")
                    st.markdown(sec.get("summary", ""))
            else:
                st.markdown(data.get("summary", "No summary available."))


# ── Sidebar: existing documents ──────────────────────────────────────────────

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

    # ── Summary buttons for single selected doc ──────────────────────────────
    if len(selected) == 1:
        doc_meta = pipeline.load_document(selected[0])
        render_pipeline_status(doc_meta, "Pipeline status")
        render_summary_buttons(doc_meta, selected[0])


# ── Sidebar: document ingest ─────────────────────────────────────────────────

def render_upload_panel() -> None:
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
        with st.spinner("Running document pipeline and generating summaries..."):
            metadata = pipeline.prepare_document(
                config_path=CONFIG_PATH,
                file_name=uploaded_file.name,
                file_bytes=uploaded_file.getvalue(),
                user_choice=user_choice,
            )
        reset_chat()
        if metadata.get("ready_to_chat"):
            st.toast(f"`{metadata['document_name']}` is ready!")
            st.session_state["upload_key"] += 1
            st.rerun()
        else:
            st.warning(
                f"`{metadata['document_name']}` resumed to step "
                f"`{metadata.get('last_successful_step', 'unknown')}`."
            )


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

    # ── Single document: existing behavior + summary buttons in sidebar ──────
    if len(selected_folders) == 1:
        _render_single_doc_chat(selected_folders[0])
        return

    # ── Multiple docs selected directly in sidebar: direct multi-doc mode ────
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
            "or select documents manually in the sidebar."
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

_SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"] {
    min-width: 420px;
    max-width: 420px;
}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="MultiModalRAG", page_icon="📄", layout="wide")
    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)
    initialize_state()

    st.title("Persistent PDF RAG")
    st.caption("Select documents from the sidebar or ask a question to find relevant documents automatically.")

    with st.sidebar:
        st.subheader("Documents")
        render_existing_documents()
        st.divider()
        st.subheader("Add Document")
        render_ingest_panel()

    render_chat()

    # Auto-refresh sidebar while any document is generating summaries in background
    docs = pipeline.list_documents(CONFIG_PATH)
    if any(d.get("summary_status") == "in_progress" for d in docs):
        import time
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()
