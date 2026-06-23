from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.config import AppConfig, EMBEDDING_MODELS
from src.snowflake_io import get_embedding_table_status, initialize_embeddings


def render_index_setup_page(session: Any, config: AppConfig) -> None:
    st.subheader("Index Setup")
    st.caption("Load claims with Snowpark, embed them with Snowflake Cortex, and save vectors in Snowflake.")

    st.markdown("**Snowflake embedding models**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "model": model.key,
                    "dimensions": model.dimensions,
                    "language": model.language,
                }
                for model in EMBEDDING_MODELS
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    render_column_mapping(config)

    try:
        config.validate_source()
    except ValueError as exc:
        st.warning(str(exc))
        source_error = exc
    else:
        source_error = None
        st.success(f"Snowflake source table: `{config.snowflake_table}`")
        st.caption(f"Embedding table: `{config.embedding_table}`")
        if config.snowflake_row_limit:
            st.caption(f"Initialization uses a random sample of `{config.snowflake_row_limit}` rows.")
        else:
            st.caption("Initialization loads all rows from the configured Snowflake table.")

    selected_models = st.multiselect(
        "Embedding models",
        options=[model.key for model in EMBEDDING_MODELS],
    )
    render_embedding_table_status(session, config)

    clicked = st.button(
        "Initialize or replace embeddings",
        type="primary",
        disabled=source_error is not None or not selected_models,
        use_container_width=True,
    )
    if not clicked:
        return

    with st.spinner("Building Snowflake embeddings..."):
        try:
            result = initialize_embeddings(session, config, selected_models)
        except Exception as exc:
            st.error(f"Embedding initialization failed: {exc}")
            return

    st.success(
        f"Saved {result.row_count:,} claims to `{result.table_name}` "
        f"with {len(result.models)} embedding model(s)."
    )
    render_embedding_table_status(session, config)


def render_column_mapping(config: AppConfig) -> None:
    st.markdown("**Snowflake column mapping**")
    st.caption("Mappings are read from `app_config.toml`; credentials stay in your local Snowflake files.")
    st.dataframe(pd.DataFrame(config.columns.mapping_rows()), use_container_width=True, hide_index=True)


def render_embedding_table_status(session: Any, config: AppConfig) -> None:
    try:
        status = get_embedding_table_status(session, config)
    except Exception as exc:
        st.warning(f"Could not inspect `{config.embedding_table}`: {exc}")
        return
    if status is None:
        st.info("No Snowflake embedding table exists yet.")
        return
    st.json(
        {
            "table": status.table_name,
            "record_count": status.row_count,
            "models": [model.key for model in status.models],
        }
    )
