from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import AppConfig, ColumnConfig, EMBEDDING_MODELS_BY_KEY
from src.snowflake_io import EmbeddingTableStatus


def run_streamlit_app() -> None:
    from app.streamlit_app import main

    main()


def make_app_config() -> AppConfig:
    return AppConfig(
        columns=ColumnConfig(
            claim_id="claim_id",
            description="claim_description",
            line_of_business="line_of_business",
            country="country",
        ),
        snowflake_table="DB.SCHEMA.CLAIMS",
        snowflake_row_limit=10,
    )


def configure_app(monkeypatch, config: AppConfig) -> None:
    status = EmbeddingTableStatus(
        "DB.SCHEMA.CLAIMS_EMBEDDINGS",
        1,
        [EMBEDDING_MODELS_BY_KEY["voyage-multilingual-2"]],
    )
    options = {
        "equality": {"country": ["DE"]},
        "ranges": {},
        "claims": [{"CLAIM_ID": "1", "COUNTRY": "DE"}],
    }
    monkeypatch.setattr("app.streamlit_app.AppConfig.from_app_config", lambda: config)
    monkeypatch.setattr("app.streamlit_app.get_snowflake_session", lambda: object())
    monkeypatch.setattr("app.streamlit_app.get_embedding_table_status", lambda *args: status)
    monkeypatch.setattr("app.index_setup_page.get_embedding_table_status", lambda *args: status)
    monkeypatch.setattr("app.streamlit_app.collect_search_options", lambda *args: options)


def test_streamlit_renders_snowflake_initialization_and_search(monkeypatch) -> None:
    configure_app(monkeypatch, make_app_config())

    app = AppTest.from_function(run_streamlit_app)
    app.run(timeout=60)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Index Setup", "Similar Claims Search"]
    model_select = next(select for select in app.selectbox if select.label == "Embedding model")
    assert model_select.options == ["Voyage Multilingual 2"]
    assert model_select.value is None
    metric_select = next(select for select in app.selectbox if select.label == "Similarity metric")
    assert metric_select.options == [
        "Cosine similarity",
        "Inner product",
        "Manhattan distance",
        "Euclidean distance",
    ]
    assert metric_select.value is None
    model_multiselect = next(select for select in app.multiselect if select.label == "Embedding models")
    assert model_multiselect.value == []
    assert any("Snowflake source table" in success.value for success in app.success)
    assert any(button.label == "Initialize or replace embeddings" for button in app.button)


def test_streamlit_embedding_search_renders_results(monkeypatch) -> None:
    configure_app(monkeypatch, make_app_config())
    monkeypatch.setattr(
        "app.search_page.query_similar_claims",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "claim_id": "1",
                    "document": "Claim description: water leakage in warehouse",
                    "country": "DE",
                    "metric_value": 0.9,
                }
            ]
        ),
    )

    app = AppTest.from_function(run_streamlit_app)
    app.run(timeout=60)

    next(select for select in app.selectbox if select.label == "Embedding model").set_value("voyage-multilingual-2")
    next(select for select in app.selectbox if select.label == "Similarity metric").set_value("cosine")
    app.run(timeout=60)

    next(area for area in app.text_area if area.label == "Free-text claim description").set_value("water leakage")
    next(button for button in app.button if button.label == "Search").click()
    app.run(timeout=60)

    assert not app.exception
    assert any("claim-result-card" in markdown.value for markdown in app.markdown)
