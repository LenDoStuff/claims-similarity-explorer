from __future__ import annotations

import streamlit as st


@st.cache_resource
def get_snowflake_session():
    from snowflake.snowpark import Session

    return Session.builder.create()


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
