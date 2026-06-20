from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.chroma_store import collection_to_frame, get_collection
from src.config import AppConfig


def local_data_version(config: AppConfig, model_key: str) -> str:
    parts = [
        str(path.stat().st_mtime_ns)
        for path in [
            config.index_manifest_path_for_model(model_key),
            config.clusters_path_for_model(model_key),
            config.cluster_map_path_for_model(model_key),
        ]
        if path.exists()
    ]
    return ":".join(parts) or "empty"


@st.cache_data(show_spinner=False)
def load_claims_frame(chroma_dir: str, collection_name: str, _version: str) -> pd.DataFrame:
    collection = get_collection(Path(chroma_dir), collection_name)
    return collection_to_frame(collection)


def format_amount(value: float) -> str:
    return f"{value:,.0f}"


def cluster_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "claim_id",
        "document",
        "line_of_business",
        "claim_type",
        "cause_of_loss",
        "country",
        "claim_status",
        "loss_year",
        "reserve_amount",
        "paid_amount",
        "currency",
    ]
    return frame[[column for column in columns if column in frame]].copy()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: #f7f8fa;
            border: 1px solid #e6e8eb;
            border-radius: 8px;
            padding: 0.8rem 1rem;
        }
        section[data-testid="stSidebar"] {
            background: #f7f8fa;
        }
        .stButton button {
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
