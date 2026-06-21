from __future__ import annotations

import pytest

from src.config import (
    AppConfig,
    ColumnConfig,
    available_embedding_models,
    available_reranker_models,
    discover_embedding_models,
    discover_reranker_models,
    model_collection_name,
    model_storage_key,
    versioned_collection_name,
)


def test_model_storage_key_replaces_hyphens() -> None:
    assert model_storage_key("multilingual-e5-small") == "multilingual_e5_small"


def test_model_specific_collection_and_artifact_names() -> None:
    config = AppConfig()

    assert model_collection_name("claims", "multilingual-e5-base") == "claims_multilingual_e5_base"
    assert versioned_collection_name("claims", "multilingual-e5-base", "abcdef1234567890") == "claims_multilingual_e5_base_abcdef123456"
    assert config.index_manifest_path_for_model("multilingual-e5-small").name == "index_manifest_multilingual_e5_small.json"
    assert config.clusters_path_for_model("multilingual-e5-small").name == "clusters_multilingual_e5_small.json"
    assert config.cluster_map_path_for_model("multilingual-e5-small").name == "cluster_map_multilingual_e5_small.json"


def test_column_config_allows_blank_optional_mappings(monkeypatch) -> None:
    monkeypatch.setenv("LINE_OF_BUSINESS_COLUMN", "")
    monkeypatch.setenv("CLAIM_TYPE_COLUMN", "")

    columns = ColumnConfig.from_env()

    assert "line_of_business" not in columns.selected_columns
    assert "claim_type" not in columns.selected_columns
    assert columns.selected_columns[:2] == ["claim_id", "claim_description"]
    assert columns.embedding_columns == ["claim_description", "cause_of_loss", "damaged_object", "country"]
    assert any(row["source_column"] == "(skipped)" for row in columns.mapping_rows())


def test_column_config_rejects_blank_required_mappings() -> None:
    with pytest.raises(ValueError, match="CLAIM_ID_COLUMN"):
        ColumnConfig(claim_id="").validate_required()

    with pytest.raises(ValueError, match="CLAIM_DESCRIPTION_COLUMN"):
        ColumnConfig(description="").validate_required()


def test_model_scanners_find_typed_subfolders(tmp_path) -> None:
    embedding_root = tmp_path / "embeddings"
    reranker_root = tmp_path / "rerankers"
    (embedding_root / "custom-embedding").mkdir(parents=True)
    (reranker_root / "custom-reranker").mkdir(parents=True)

    embedding_models = discover_embedding_models(embedding_root)
    reranker_models = discover_reranker_models(reranker_root)

    assert [model.key for model in embedding_models] == ["custom-embedding"]
    assert embedding_models[0].label == "Custom Embedding"
    assert [model.key for model in reranker_models] == ["custom-reranker"]
    assert reranker_models[0].label == "Custom Reranker"


def test_available_models_use_typed_model_dirs(monkeypatch, tmp_path) -> None:
    (tmp_path / "embeddings" / "embedding-a").mkdir(parents=True)
    (tmp_path / "embeddings" / "embedding-b").mkdir(parents=True)
    (tmp_path / "rerankers" / "reranker-a").mkdir(parents=True)
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))

    config = AppConfig()
    collection_names = [model_collection_name(config.collection_name, model.key) for model in available_embedding_models()]

    assert collection_names == ["claims_embedding_a", "claims_embedding_b"]
    assert [model.key for model in available_reranker_models()] == ["reranker-a"]
    assert len(collection_names) == len(set(collection_names))
