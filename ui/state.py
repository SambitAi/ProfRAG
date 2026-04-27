from __future__ import annotations

from pathlib import Path

import streamlit as st

import main as pipeline

CONFIG_PATH = Path(__file__).parent.parent / "config" / "app_config.yaml"


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("upload_key", 0)
    st.session_state.setdefault("url_input_key", 0)
    st.session_state.setdefault("url_processing", False)
    st.session_state.setdefault("pending_url", "")
    st.session_state.setdefault("pending_user_choice", "new_version")
    st.session_state.setdefault("url_error", "")
    st.session_state.setdefault("pdf_processing", False)
    st.session_state.setdefault("pending_pdf_name", "")
    st.session_state.setdefault("pending_pdf_bytes", b"")
    st.session_state.setdefault("pending_pdf_choice", "new_version")
    st.session_state.setdefault("pdf_info", "")
    st.session_state.setdefault("selected_document_folders", [])
    st.session_state.setdefault("chat_mode", "routing")        # "routing" | "deep_search"
    st.session_state.setdefault("active_deep_search_folders", [])
    st.session_state.setdefault("pending_routing_question", "")
    st.session_state.setdefault("routing_candidates", [])
    st.session_state.setdefault("show_doc_pane", True)
    st.session_state.setdefault("show_summary_pane", True)
    st.session_state.setdefault("_sum_pane_active_doc", "")
    st.session_state.setdefault("_sum_pane_active_level", "")
    # Daemon threads die on process exit; track which folders were started in
    # this session so stale "in_progress" entries from prior sessions are ignored.
    st.session_state.setdefault("active_summarization_folders", set())


def reset_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["routing_candidates"] = []
    st.session_state["pending_routing_question"] = ""


def track_summarization(folder: str) -> None:
    """Start background summarization and register the folder in the session tracking set."""
    pipeline.start_summarization_background(CONFIG_PATH, folder)
    st.session_state["active_summarization_folders"].add(str(folder))
