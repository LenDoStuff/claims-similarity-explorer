from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

from src.chroma_store import reset_collection
from src.config import AppConfig


def run_streamlit_app(config) -> None:
    from app.streamlit_app import main

    main(config)


def make_app_config(tmp_path) -> AppConfig:
    return AppConfig(
        snowflake_table="CLAIMS",
        snowflake_row_limit=10,
        chroma_dir=tmp_path / "chroma",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_streamlit_model_dropdown_and_missing_index_message(tmp_path) -> None:
    config = make_app_config(tmp_path)

    app = AppTest.from_function(run_streamlit_app, args=(config,))
    app.run(timeout=60)

    assert not app.exception
    assert any(tab.label == "Index Setup" for tab in app.tabs)
    assert any(subheader.value == "Index Setup" for subheader in app.subheader)
    assert any("Snowflake column mapping" in markdown.value for markdown in app.markdown)
    assert any("random Snowflake sample" in caption.value for caption in app.caption)
    model_select = next(select for select in app.selectbox if select.label == "Embedding model")
    assert "Multilingual E5 Small" in model_select.options
    assert model_select.value == "multilingual-e5-small"
    assert any("Index Setup" in info.value for info in app.info)


def test_streamlit_search_configuration_controls_render(tmp_path) -> None:
    config = make_app_config(tmp_path)
    collection_name = "claims_multilingual_e5_small_active_test"
    collection = reset_collection(config.chroma_dir, collection_name)
    collection.upsert(
        ids=["1"],
        documents=["Claim description: water leakage in warehouse"],
        metadatas=[{"claim_id": "1", "country": "DE", "description_length": 43}],
        embeddings=[[1.0, 0.0]],
    )
    manifest_path = config.index_manifest_path_for_model("multilingual-e5-small")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "collection_name": collection_name,
                "record_count": 1,
                "index_hash": "active-test",
            }
        ),
        encoding="utf-8",
    )

    app = AppTest.from_function(run_streamlit_app, args=(config,))
    app.run(timeout=60)

    assert not app.exception
    assert any(subheader.value == "Search Configuration" for subheader in app.subheader)
    assert any(select.label == "Embedding model" for select in app.selectbox)
    assert any(select.label == "Retrieval mode" for select in app.selectbox)
    keyword_select = next(select for select in app.selectbox if select.label == "Keyword search algorithm")
    assert keyword_select.options == ["BM25"]
    rerank_select = next(select for select in app.selectbox if select.label == "Rerank model")
    assert "Off" in rerank_select.options
    assert any("mmarco" in option.lower() for option in rerank_select.options)
    assert any(slider.label == "Candidate pool" for slider in app.slider)
    query_source = next(radio for radio in app.radio if radio.label == "Query source")
    assert query_source.options == ["Free text", "Existing claim"]

    next(select for select in app.selectbox if select.label == "Retrieval mode").set_value("BM25")
    next(area for area in app.text_area if area.label == "Free-text claim description").set_value("water leakage")
    next(button for button in app.button if button.label == "Search").click()
    app.run(timeout=60)

    assert not app.exception
    assert any("claim-result-card" in markdown.value for markdown in app.markdown)


def test_streamlit_cluster_explorer_tabs_and_review_controls_render(tmp_path) -> None:
    config = make_app_config(tmp_path)
    collection_name = "claims_multilingual_e5_small_cluster_test"
    index_hash = "active-cluster-test"
    collection = reset_collection(config.chroma_dir, collection_name)
    collection.upsert(
        ids=["1", "2"],
        documents=[
            "Claim description: water leakage in warehouse",
            "Claim description: sprinkler leakage damaged stock",
        ],
        metadatas=[
            {
                "claim_id": "1",
                "cluster_id": 0,
                "line_of_business": "Property",
                "claim_type": "Water Damage",
                "cause_of_loss": "Leakage",
                "country": "DE",
                "claim_status": "Open",
                "loss_year": 2024,
                "reserve_amount": 1000.0,
                "paid_amount": 100.0,
                "currency": "EUR",
                "description_length": 43,
            },
            {
                "claim_id": "2",
                "cluster_id": 0,
                "line_of_business": "Property",
                "claim_type": "Water Damage",
                "cause_of_loss": "Leakage",
                "country": "AT",
                "claim_status": "Closed",
                "loss_year": 2024,
                "reserve_amount": 500.0,
                "paid_amount": 250.0,
                "currency": "EUR",
                "description_length": 48,
            },
        ],
        embeddings=[[1.0, 0.0], [0.9, 0.1]],
    )
    manifest_path = config.index_manifest_path_for_model("multilingual-e5-small")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "collection_name": collection_name,
                "record_count": 2,
                "index_hash": index_hash,
            }
        ),
        encoding="utf-8",
    )
    clusters_path = config.clusters_path_for_model("multilingual-e5-small")
    clusters_path.parent.mkdir(parents=True, exist_ok=True)
    clusters_path.write_text(
        json.dumps(
            {
                "algorithm": "kmeans",
                "n_clusters": 1,
                "clustered_at_utc": "2026-06-19T00:00:00Z",
                "index_hash": index_hash,
                "clusters": [
                    {
                        "cluster_id": 0,
                        "label": "Water damage / leakage",
                        "frequent_terms": ["water", "leakage", "sprinkler"],
                        "representative_claims": [
                            {
                                "claim_id": "1",
                                "description": "Claim description: water leakage in warehouse",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config.cluster_map_path_for_model("multilingual-e5-small").write_text(
        json.dumps(
            {
                "projection": {"method": "UMAP", "metric": "cosine", "random_state": 42, "n_neighbors": 2},
                "index_hash": index_hash,
                "points": [
                    {
                        "claim_id": "1",
                        "cluster_id": 0,
                        "x": 0.0,
                        "y": 0.0,
                        "description": "water leakage in warehouse",
                    },
                    {
                        "claim_id": "2",
                        "cluster_id": 0,
                        "x": 1.0,
                        "y": 0.1,
                        "description": "sprinkler leakage damaged stock",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    app = AppTest.from_function(run_streamlit_app, args=(config,))
    app.run(timeout=60)

    assert not app.exception
    assert {"Overview", "Cluster Detail", "Map", "Analytics", "Outliers", "Review"}.issubset(
        {tab.label for tab in app.tabs}
    )
    assert any(select.label == "Selected cluster" for select in app.selectbox)
    assert any(input_box.label == "Manual cluster label" for input_box in app.text_input)
    assert any(select.label == "Review status" for select in app.selectbox)
    assert any(select.label == "Color by" for select in app.selectbox)
    assert any(select.label == "Size by" for select in app.selectbox)
    assert any(button.label == "Save review" for button in app.button)


def test_streamlit_cluster_map_missing_artifact_message(tmp_path) -> None:
    config = make_app_config(tmp_path)
    collection_name = "claims_multilingual_e5_small_missing_map_test"
    index_hash = "active-missing-map-test"
    collection = reset_collection(config.chroma_dir, collection_name)
    collection.upsert(
        ids=["1"],
        documents=["Claim description: water leakage in warehouse"],
        metadatas=[
            {
                "claim_id": "1",
                "cluster_id": 0,
                "line_of_business": "Property",
                "claim_type": "Water Damage",
                "cause_of_loss": "Leakage",
                "country": "DE",
                "claim_status": "Open",
                "loss_year": 2024,
                "reserve_amount": 1000.0,
                "paid_amount": 100.0,
                "description_length": 43,
            }
        ],
        embeddings=[[1.0, 0.0]],
    )
    manifest_path = config.index_manifest_path_for_model("multilingual-e5-small")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "collection_name": collection_name,
                "record_count": 1,
                "index_hash": index_hash,
            }
        ),
        encoding="utf-8",
    )
    clusters_path = config.clusters_path_for_model("multilingual-e5-small")
    clusters_path.parent.mkdir(parents=True, exist_ok=True)
    clusters_path.write_text(
        json.dumps(
            {
                "algorithm": "kmeans",
                "n_clusters": 1,
                "clustered_at_utc": "2026-06-19T00:00:00Z",
                "index_hash": index_hash,
                "clusters": [
                    {
                        "cluster_id": 0,
                        "label": "Water damage / leakage",
                        "frequent_terms": ["water", "leakage"],
                        "representative_claims": [
                            {"claim_id": "1", "description": "water leakage in warehouse"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    app = AppTest.from_function(run_streamlit_app, args=(config,))
    app.run(timeout=60)

    assert not app.exception
    assert any("No cluster map artifact" in info.value for info in app.info)
