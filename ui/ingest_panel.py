from __future__ import annotations

import streamlit as st

import main as pipeline
from ui.state import CONFIG_PATH, reset_chat


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
    same = pipeline.inspect_same_name_document(CONFIG_PATH, uploaded_file.name)
    if same and same.get("ready_to_chat"):
        user_choice = st.radio(
            "A stored file with the same name exists. Choose what to do.",
            options=["reuse", "rebuild"],
            captions=[
                "Open the already processed version and go straight to chat.",
                "Create a new version folder and process the upload again.",
            ],
            horizontal=False,
        )
    elif same and not same.get("ready_to_chat"):
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
            "Web page or PDF URL", value="", disabled=True,
            placeholder="Processing...", key="url_input_processing",
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
