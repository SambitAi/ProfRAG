from __future__ import annotations

from pathlib import Path

import streamlit as st

import main as pipeline
from ui.state import CONFIG_PATH, track_summarization

_LEVEL_FILES = {
    "level1": "level1_onepager.json",
    "level2": "level2_medium.json",
    "level3": "level3_detailed.json",
}
_LEVEL_SHORT = {"level1": "1-Pager", "level2": "Medium",         "level3": "Detailed"}
_LEVEL_LONG  = {"level1": "1-Pager Summary", "level2": "Medium Summary", "level3": "Detailed Summary"}


def _summary_progress_label(metadata: dict) -> str:
    p = metadata.get("summary_progress", {})
    if p.get("level1_indexed"):  return "finalizing"
    if p.get("level1_complete"): return "indexing L1"
    if p.get("level2_complete"): return "running L1"
    if p.get("level3_indexed"):  return "running L2"
    if p.get("level3_complete"): return "indexing L3"
    return "starting"


def render_summary_pane(selected_folders: list[str]) -> None:
    from core.storage import read_json as _read_json

    # CSS anchor — colours this column via :has(#summary-pane-marker)
    st.markdown('<span id="summary-pane-marker"></span>', unsafe_allow_html=True)

    col_title, col_close = st.columns([4, 1])
    with col_title:
        st.subheader("Summaries")
    with col_close:
        if st.button("◀", key="sum_pane_close", use_container_width=True):
            st.session_state["show_summary_pane"] = False
            st.rerun()

    # Clear active selection if its document is no longer selected
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
        _render_doc_summary_row(folder, active_doc, active_level)

    # ── Shared content viewer ────────────────────────────────────────────────
    active_doc   = st.session_state.get("_sum_pane_active_doc", "")
    active_level = st.session_state.get("_sum_pane_active_level", "")

    if active_doc and active_level:
        _render_summary_content(active_doc, active_level, _read_json)


def _render_doc_summary_row(
    folder: str, active_doc: str, active_level: str
) -> None:
    metadata      = pipeline.load_document(folder)
    doc_name      = metadata.get("document_name", Path(folder).name)
    status        = metadata.get("summary_status", "pending")
    sum_ready     = metadata.get("summary_ready", False)
    folder_slug   = Path(folder).name
    summaries_dir = Path(folder) / "summaries"
    available     = {k for k, f in _LEVEL_FILES.items() if (summaries_dir / f).exists()}

    st.markdown(f"**{doc_name}**")

    # Status / generate controls
    if not sum_ready:
        if status == "in_progress":
            st.caption(f"⏳ {_summary_progress_label(metadata)}")
            if st.button("Restart", key=f"sp_restart_{folder_slug}", use_container_width=True):
                track_summarization(folder)
                st.rerun()
        elif status == "error":
            st.caption("❌ Generation failed")
            if st.button("Retry", key=f"sp_retry_{folder_slug}", use_container_width=True):
                track_summarization(folder)
                st.rerun()
        elif not available:
            if st.button("Generate Summaries", type="primary",
                         key=f"sp_gen_{folder_slug}", use_container_width=True):
                track_summarization(folder)
                st.rerun()

    # Level buttons
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


def _render_summary_content(active_doc: str, active_level: str, read_json) -> None:
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
            st.session_state["_sum_pane_active_doc"]   = ""
            st.session_state["_sum_pane_active_level"] = ""
            st.rerun()
