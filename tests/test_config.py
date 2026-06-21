from __future__ import annotations

import pytest

from src.config import (
    AppConfig,
    ColumnConfig,
    discover_embedding_models,
    discover_reranker_models,
    model_storage_key,
    versioned_collection_name,
)


def test_model_storage_key_replaces_hyphens() -> None:
    assert model_storage_key("multilingual-e5-small") == "multilingual_e5_small"


def test_model_specific_collection_and_artifact_names() -> None:
    config = AppConfig()

    assert versioned_collection_name("claims", "multilingual-e5-base", "abcdef1234567890") == "claims_multilingual_e5_base_abcdef123456"
    assert config.index_manifest_path_for_model("multilingual-e5-small").name == "index_manifest_multilingual_e5_small.json"
    assert config.clusters_path_for_model("multilingual-e5-small").name == "clusters_multilingual_e5_small.json"
    assert config.cluster_map_path_for_model("multilingual-e5-small").name == "cluster_map_multilingual_e5_small.json"


def test_column_config_allows_blank_optional_mappings() -> None:
    columns = ColumnConfig.from_mapping(
        {
            "line_of_business": "",
            "claim_type": "",
        }
    )

    assert "line_of_business" not in columns.selected_columns
    assert "claim_type" not in columns.selected_columns
    assert columns.selected_columns[:2] == ["claim_id", "claim_description"]
    assert columns.embedding_columns == ["claim_description", "cause_of_loss", "damaged_object", "country"]
    assert any(row["source_column"] == "(skipped)" for row in columns.mapping_rows())


def test_app_config_reads_table_and_columns(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "app_config.toml"
    config_path.write_text(
        """
[snowflake]
table = "CLAIMS"
row_limit = 250

[columns]
claim_id = "ID"
description = "TEXT"
line_of_business = ""
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.config.DEFAULT_APP_CONFIG_PATH", config_path)

    config = AppConfig.from_app_config()

    assert config.snowflake_table == "CLAIMS"
    assert config.snowflake_row_limit == 250
    assert config.columns.claim_id == "ID"
    assert config.columns.description == "TEXT"
    assert config.columns.line_of_business == ""
    assert "line_of_business" not in config.columns.selected_columns


def test_column_config_rejects_blank_required_mappings() -> None:
    with pytest.raises(ValueError, match="columns.claim_id"):
        ColumnConfig(claim_id="").validate_required()

    with pytest.raises(ValueError, match="columns.description"):
        ColumnConfig(description="").validate_required()


def test_app_config_requires_table(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "app_config.toml"
    config_path.write_text(
        """
[snowflake]
table = ""

[columns]
claim_id = "claim_id"
description = "claim_description"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.config.DEFAULT_APP_CONFIG_PATH", config_path)

    config = AppConfig.from_app_config()

    with pytest.raises(ValueError, match="snowflake.table"):
        config.validate_source()


def test_app_config_requires_required_columns(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "app_config.toml"
    config_path.write_text(
        """
[snowflake]
table = "CLAIMS"

[columns]
claim_id = ""
description = "claim_description"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.config.DEFAULT_APP_CONFIG_PATH", config_path)

    config = AppConfig.from_app_config()

    with pytest.raises(ValueError, match="columns.claim_id"):
        config.validate_source()


def test_app_config_rejects_non_positive_row_limit(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "app_config.toml"
    config_path.write_text(
        """
[snowflake]
table = "CLAIMS"
row_limit = 0

[columns]
claim_id = "claim_id"
description = "claim_description"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.config.DEFAULT_APP_CONFIG_PATH", config_path)

    config = AppConfig.from_app_config()

    with pytest.raises(ValueError, match="snowflake.row_limit"):
        config.validate_source()


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


def test_model_scanners_sort_typed_subfolders(tmp_path) -> None:
    embedding_root = tmp_path / "embeddings"
    reranker_root = tmp_path / "rerankers"
    (embedding_root / "embedding-b").mkdir(parents=True)
    (embedding_root / "embedding-a").mkdir(parents=True)
    (reranker_root / "reranker-a").mkdir(parents=True)

    assert [model.key for model in discover_embedding_models(embedding_root)] == ["embedding-a", "embedding-b"]
    assert [model.key for model in discover_reranker_models(reranker_root)] == ["reranker-a"]
