from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.config import AppConfig, SnowflakeConfig, available_embedding_models, available_reranker_models
from src.diagnostics import read_json
from src.indexing import active_collection_name, build_index_from_snowflake


def render_index_setup_page(config: AppConfig) -> None:
    st.subheader("Index Setup")
    st.caption("Load Snowflake claims once, embed them with a local model, and save the index in Chroma.")

    embedding_models = available_embedding_models()
    reranker_models = available_reranker_models()
    render_model_tables(embedding_models, reranker_models)
    render_column_mapping(config)

    try:
        snowflake = SnowflakeConfig.from_env()
    except RuntimeError as exc:
        st.warning(str(exc))
        snowflake = None
    else:
        st.success(f"Snowflake source configured: `{snowflake.qualified_table}`")

    if not embedding_models:
        st.info("Add local embedding models under `models/embeddings` before indexing.")
        return
    try:
        config.columns.validate_required()
        column_error = None
    except ValueError as exc:
        st.warning(str(exc))
        column_error = exc

    model_by_key = {model.key: model for model in embedding_models}
    default_index = next(
        (index for index, model in enumerate(embedding_models) if model.key == config.model_key),
        0,
    )
    selected_key = st.selectbox(
        "Embedding model to index",
        options=[model.key for model in embedding_models],
        index=default_index,
        format_func=lambda key: model_by_key[key].label,
    )
    selected_model = model_by_key[selected_key]
    manifest = read_json(config.index_manifest_path_for_model(selected_model.key))
    render_active_index_summary(config, selected_model.key, manifest)

    clicked = st.button(
        "Load or refresh index",
        type="primary",
        disabled=snowflake is None or column_error is not None,
        use_container_width=True,
    )
    if not clicked:
        return

    with st.spinner("Loading claims and preparing the Chroma index..."):
        try:
            result = build_index_from_snowflake(config, selected_model, snowflake)
        except Exception as exc:
            st.error(f"Index build failed: {exc}")
            return

    if result.status == "current":
        st.success("Index is current. Existing Chroma collection was reused.")
    elif result.status == "activated_existing":
        st.success("Matching Chroma collection already exists and is now active.")
    else:
        st.success("Index built and saved in Chroma.")
    render_active_index_summary(config, selected_model.key, result.manifest)


def render_model_tables(embedding_models: list[Any], reranker_models: list[Any]) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Embedding models**")
        st.table(model_rows(embedding_models))
    with right:
        st.markdown("**Rerank models**")
        st.table(model_rows(reranker_models))


def render_column_mapping(config: AppConfig) -> None:
    st.markdown("**Snowflake column mapping**")
    st.caption("Required mappings must be set. Optional mappings can be blank to skip the field.")
    st.table(pd.DataFrame(config.columns.mapping_rows()))


def model_rows(models: list[Any]) -> pd.DataFrame:
    rows = [
        {
            "key": model.key,
            "label": model.label,
            "local_path": str(model.model_dir),
        }
        for model in models
    ]
    return pd.DataFrame(rows or [{"key": "", "label": "No models found", "local_path": ""}])


def render_active_index_summary(config: AppConfig, model_key: str, manifest: dict[str, Any]) -> None:
    if not active_collection_name(config, model_key, manifest):
        st.info("No active hash-based index manifest exists for this embedding model yet.")
        return
    st.json(
        {
            "collection_name": manifest.get("collection_name"),
            "record_count": manifest.get("record_count"),
            "index_hash": manifest.get("index_hash"),
            "dataset_hash": manifest.get("dataset_hash"),
            "model_fingerprint": manifest.get("model_fingerprint"),
            "refreshed_at_utc": manifest.get("refreshed_at_utc"),
            "refresh_strategy": manifest.get("refresh_strategy"),
        }
    )
