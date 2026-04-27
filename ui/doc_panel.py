from __future__ import annotations

import streamlit as st

import main as pipeline
from ui.state import CONFIG_PATH, reset_chat

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


def _step_label(metadata: dict, step_name: str) -> str:
    step_info = metadata.get("steps", {}).get(step_name, {})
    return "done" if step_info.get("status") == "success" else "pending"


def render_pipeline_status(metadata: dict, title: str) -> None:
    with st.expander(title, expanded=False):
        st.write(f"Ready to chat: `{metadata.get('ready_to_chat', False)}`")
        st.write(f"Last successful step: `{metadata.get('last_successful_step', 'unknown')}`")
        st.write(f"Total chunks: `{metadata.get('total_chunks', 0)}`")
        for step in PIPELINE_STEPS:
            st.write(f"- `{step}`: `{_step_label(metadata, step)}`")


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
        st.checkbox(
            f"{ready_badge} {doc['folder_name']}{summary_badge}",
            key=key,
            disabled=not doc["ready_to_chat"],
        )

    # Derive and persist selection from current checkbox state
    selected = [
        doc["document_folder"]
        for doc in documents
        if st.session_state.get(f"doc_chk_{doc['folder_name']}", False)
    ]
    st.session_state["selected_document_folders"] = selected

    if selected:
        st.caption(f"{len(selected)} document(s) selected")

    if len(selected) == 1:
        render_pipeline_status(pipeline.load_document(selected[0]), "Pipeline status")
