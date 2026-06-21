from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.cluster_page import render_cluster_page
from app.diagnostics_page import render_diagnostics_page
from app.index_setup_page import render_index_setup_page
from app.search_page import render_search_model_selector, render_search_page
from app.ui_helpers import (
    inject_css,
    load_claims_frame,
    local_data_version,
)
from src.chroma_store import get_collection
from src.config import AppConfig
from src.diagnostics import read_json
from src.indexing import active_collection_name


st.set_page_config(
    page_title="Claims Similarity Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    inject_css()
    load_dotenv_if_available()
    config = AppConfig.from_env()

    st.title("Claims Similarity Explorer")
    st.caption("Local semantic search and exploratory KMeans clustering for insurance claims.")

    setup_tab, search_tab, cluster_tab, diagnostics_tab = st.tabs(
        ["Index Setup", "Similar Claims Search", "Cluster Explorer", "Data & Embedding Diagnostics"]
    )

    selected_model = None
    collection_name = ""
    collection = None
    claims_frame = None
    manifest = {}
    clusters = {}
    cluster_map = {}

    with setup_tab:
        render_index_setup_page(config)
    with search_tab:
        selected_model = render_search_model_selector(config)
        if selected_model is not None:
            manifest = read_json(config.index_manifest_path_for_model(selected_model.key))
            collection_name = active_collection_name(config, selected_model.key, manifest)
            if collection_name:
                collection = get_collection(config.chroma_dir, collection_name)
                claims_frame = load_claims_frame(
                    str(config.chroma_dir),
                    collection_name,
                    local_data_version(config, selected_model.key),
                )
                clusters, cluster_map = read_cluster_artifacts(config, selected_model.key, manifest)
            else:
                claims_frame = pd.DataFrame()
            render_search_page(config, selected_model, collection, claims_frame, manifest)
    with cluster_tab:
        if selected_model is None or claims_frame is None:
            st.info("Select an embedding model in Similar Claims Search first.")
        else:
            render_cluster_page(config, selected_model, claims_frame, clusters, cluster_map)
    with diagnostics_tab:
        if selected_model is None:
            st.info("Select an embedding model in Similar Claims Search first.")
        elif collection is None or claims_frame is None:
            st.info("Open Index Setup and click `Load or refresh index` before viewing diagnostics.")
        else:
            render_diagnostics_page(config, selected_model, collection_name, collection, claims_frame, manifest, clusters)


def read_cluster_artifacts(config: AppConfig, model_key: str, manifest: dict) -> tuple[dict, dict]:
    clusters = read_json(config.clusters_path_for_model(model_key))
    cluster_map = read_json(config.cluster_map_path_for_model(model_key))
    index_hash = manifest.get("index_hash")
    if index_hash and clusters.get("index_hash") != index_hash:
        return {}, {}
    return clusters, cluster_map


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


if __name__ == "__main__":
    main()
