from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from app.ui_helpers import cluster_claim_columns
from src.config import AppConfig, EmbeddingModelConfig, available_embedding_models
from src.diagnostics import collection_diagnostics


def render_diagnostics_page(
    config: AppConfig,
    selected_model: EmbeddingModelConfig,
    collection_name: str,
    collection: Any,
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    clusters: dict[str, Any],
) -> None:
    diagnostics = collection_diagnostics(collection)
    manifest_diagnostics = manifest.get("diagnostics", {})

    cols = st.columns(4)
    cols[0].metric("Chroma records", diagnostics.get("record_count", 0))
    cols[1].metric("Manifest records", manifest.get("record_count", 0))
    cols[2].metric("Clusters", diagnostics.get("cluster_count", 0))
    cols[3].metric("Avg description chars", diagnostics.get("average_description_length", 0))

    st.subheader("Index")
    st.json(
        {
            "source": manifest.get("source"),
            "refreshed_at_utc": manifest.get("refreshed_at_utc"),
            "embedding_model": manifest.get("embedding_model"),
            "embedding_model_path": manifest.get("embedding_model_path"),
            "embedding_version": manifest.get("embedding_version"),
            "selected_model_key": selected_model.key,
            "chroma_dir": str(config.chroma_dir),
            "collection_name": collection_name,
        }
    )

    st.subheader("Data Quality")
    st.json(
        {
            "missing_descriptions": manifest_diagnostics.get("missing_descriptions", 0),
            "short_descriptions": manifest_diagnostics.get("short_descriptions", 0),
            "duplicate_descriptions": manifest_diagnostics.get("duplicate_descriptions", 0),
            "metadata_columns": diagnostics.get("metadata_columns", []),
        }
    )

    render_embedding_model_smoke_test(selected_model)

    st.subheader("Cluster Artifact")
    st.json(
        {
            "n_clusters": clusters.get("n_clusters"),
            "algorithm": clusters.get("algorithm"),
            "clustered_at_utc": clusters.get("clustered_at_utc"),
        }
    )

    if not frame.empty:
        st.subheader("Local Claims Snapshot")
        st.dataframe(cluster_claim_columns(frame).head(100), use_container_width=True, hide_index=True)


def render_embedding_model_smoke_test(selected_model: EmbeddingModelConfig) -> None:
    st.subheader("Local Embedding Models")
    models = available_embedding_models()
    st.dataframe(pd.DataFrame([model_status_row(model) for model in models]), use_container_width=True, hide_index=True)
    st.caption(f"Smoke test uses the active model: `{selected_model.repo_id}`.")

    sample_text = st.text_input("Dummy text", "Rohrbruch verursachte Wasserschaden in einer Lagerhalle.")
    if st.button("Load local model and embed dummy text"):
        start = time.perf_counter()
        try:
            from src.embeddings import load_embedding_model

            model = load_embedding_model(str(selected_model.model_dir), selected_model.repo_id)
            embedding = model.encode_queries([sample_text])
        except Exception as exc:
            st.error(f"Model smoke test failed: {exc}")
            return
        elapsed = time.perf_counter() - start
        st.success("Local model loaded and embedded the dummy text.")
        st.json(
            {
                "model_key": selected_model.key,
                "model_name": selected_model.repo_id,
                "model_path": str(selected_model.model_dir),
                "sample_text": sample_text,
                "embedding_shape": list(embedding.shape),
                "squared_norm": round(float((embedding[0] ** 2).sum()), 6),
                "elapsed_seconds": round(elapsed, 3),
            }
        )


def model_status_row(model: EmbeddingModelConfig) -> dict[str, Any]:
    files = list(model.model_dir.rglob("*")) if model.model_dir.exists() else []
    size_mb = sum(path.stat().st_size for path in files if path.is_file()) / 1024 / 1024
    return {
        "key": model.key,
        "model": model.repo_id,
        "installed": model.model_dir.exists(),
        "size_mb": round(size_mb, 1),
        "local_path": str(model.model_dir),
        "notes": model.notes,
    }
