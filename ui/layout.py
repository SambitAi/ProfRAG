from __future__ import annotations

import streamlit as st

from ui.state import initialize_state
from ui.styles import APP_CSS, NAVBAR_HTML
from ui.doc_panel import render_existing_documents
from ui.ingest_panel import render_ingest_panel
from ui.summary_pane import render_summary_pane
from ui.chat_panel import render_chat


def main() -> None:
    st.set_page_config(page_title="ProfRAG", page_icon="📄", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(NAVBAR_HTML, unsafe_allow_html=True)
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
        st.caption(
            "Select documents from the left panel or ask a question "
            "to find relevant documents automatically."
        )
        render_chat()
