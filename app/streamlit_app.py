from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.index_setup_page import render_index_setup_page
from app.search_page import render_search_configuration, render_search_page
from app.ui_helpers import get_snowflake_session, inject_css
from src.config import AppConfig
from src.snowflake_io import collect_search_options, get_embedding_table_status


st.set_page_config(
    page_title="Claims Similarity Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    inject_css()
    config = AppConfig.from_app_config()

    st.title("Claims Similarity Explorer")
    st.caption("Build and search Snowflake-hosted claim embeddings with Snowpark.")

    try:
        session = get_snowflake_session()
    except Exception as exc:
        st.error(f"Snowflake connection failed: {exc}")
        return

    setup_tab, search_tab = st.tabs(["Index Setup", "Similar Claims Search"])

    with setup_tab:
        render_index_setup_page(session, config)
    with search_tab:
        try:
            status = get_embedding_table_status(session, config)
        except Exception as exc:
            st.error(f"Could not inspect the Snowflake embedding table: {exc}")
            return
        if status is None or not status.models:
            st.info("Open Index Setup and initialize at least one Snowflake embedding model.")
            return
        search_configuration = render_search_configuration(status.models)
        if search_configuration is None:
            st.info("Select an embedding model and similarity metric.")
            return
        selected_model, selected_metric = search_configuration
        try:
            options = collect_search_options(session, status.table_name)
        except Exception as exc:
            st.error(f"Could not load Snowflake search options: {exc}")
            return
        render_search_page(session, status.table_name, selected_model, selected_metric, options)


if __name__ == "__main__":
    main()
