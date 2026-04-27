from __future__ import annotations

from pathlib import Path

import streamlit as st

import main as pipeline
from ui.state import CONFIG_PATH, reset_chat

_PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_image_path(p: str) -> str | None:
    ph = Path(p)
    if ph.is_absolute():
        return str(ph) if ph.exists() else None
    resolved = _PROJECT_ROOT / p
    return str(resolved) if resolved.exists() else None


# ── Message history ───────────────────────────────────────────────────────────

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
                        _render_source_entry(src)


def _render_source_entry(src: dict) -> None:
    doc     = src.get("document_name", "Unknown")
    section = src.get("section_path", "")
    chunk   = src.get("chunk_number", "")
    snippet = src.get("chunk_text_snippet", "")
    label   = f"**{doc}**"
    if section:
        label += f" — {section}"
    elif chunk:
        label += f" — Chunk {chunk}"
    st.markdown(label)
    if snippet:
        st.caption(f'"{snippet}..."')


# ── Routing candidate UI ──────────────────────────────────────────────────────

def _render_routing_candidates() -> None:
    candidates = st.session_state.get("routing_candidates", [])
    if not candidates:
        return

    st.info("I found these documents may be relevant. Select which to search, then confirm:")
    for cand in candidates:
        key = f"rchk_{cand['folder']}"
        st.session_state.setdefault(key, True)
        section_hint = f" — *{cand['top_section']}*" if cand.get("top_section") else ""
        score_pct    = int(cand.get("score", 0) * 100)
        st.checkbox(f"**{cand['doc_name']}**{section_hint}  `{score_pct}% match`", key=key)

    selected_folders = [
        c["folder"] for c in candidates
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


# ── Chat modes ────────────────────────────────────────────────────────────────

def render_chat() -> None:
    selected_folders = st.session_state.get("selected_document_folders", [])
    chat_mode        = st.session_state.get("chat_mode", "routing")
    active_folders   = st.session_state.get("active_deep_search_folders", [])

    if len(selected_folders) == 1:
        _render_single_doc_chat(selected_folders[0])
    elif len(selected_folders) > 1:
        _render_direct_multi_doc_chat(selected_folders)
    elif chat_mode == "deep_search" and active_folders:
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
            image_paths, answer_text = _build_single_doc_response(answer_payload, document_folder)

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


def _build_single_doc_response(
    answer_payload: dict, document_folder: str
) -> tuple[list[str], str]:
    sources    = answer_payload.get("sources", [])
    image_paths: list[str] = []

    if sources:
        top = sources[0]
        doc_folder = top.get("document_folder", "")
        chunk_num  = top.get("chunk_number", 0)
        chunk_path = top.get("chunk_path", "")
        image_paths = pipeline.read_chunk(chunk_path).get("image_paths", []) if chunk_path else []
        if not image_paths:
            image_paths = pipeline.read_next_chunk(doc_folder, chunk_num).get("image_paths", [])
        if not image_paths:
            image_paths = pipeline.read_prev_chunk(doc_folder, chunk_num).get("image_paths", [])

    image_paths = [
        r for p in dict.fromkeys(image_paths) if p
        for r in [_resolve_image_path(p)] if r
    ]

    sources_text = "\n".join(
        f"- {item.get('section_path') or ('Chunk ' + str(item.get('chunk_number', 'unknown')))}"
        for item in sources
    )
    answer_text = (
        f"{answer_payload.get('answer', 'No answer generated.')}\n\nSources:\n{sources_text}"
    )
    return image_paths, answer_text


def _render_direct_multi_doc_chat(document_folders: list[str]) -> None:
    doc_names = [pipeline.load_document(f).get("document_name", Path(f).name) for f in document_folders]
    st.subheader(f"Multi-doc mode — {len(document_folders)} documents")
    st.caption(", ".join(doc_names))

    _render_message_history()

    question = st.chat_input("Ask a question across selected documents")
    if question:
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
    st.rerun()


def _render_deep_search_chat(active_folders: list[str]) -> None:
    doc_names = [pipeline.load_document(f).get("document_name", Path(f).name) for f in active_folders]

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

        answer  = payload.get("answer", "No answer generated.")
        sources = payload.get("sources", [])

        st.markdown(answer)
        if sources:
            with st.expander(f"📄 {len(sources)} source(s)", expanded=False):
                for src in sources:
                    _render_source_entry(src)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "chunk_sources": sources,
    })
