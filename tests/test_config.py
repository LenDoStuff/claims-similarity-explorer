from __future__ import annotations

import pytest

from src.config import (
    AppConfig,
    COLUMN_MAPPING_FIELDS,
    ColumnConfig,
    available_embedding_models,
    embedding_column_for_model,
    model_for_embedding_column,
)


def full_column_mapping(**overrides: str) -> dict[str, str]:
    values = {
        "claim_id": "claim_id",
        "description": "claim_description",
        "line_of_business": "line_of_business",
        "claim_type": "claim_type",
        "cause_of_loss": "cause_of_loss",
        "damaged_object": "damaged_object",
        "country": "country",
        "claim_status": "claim_status",
        "loss_date": "loss_date",
        "reserve_amount": "reserve_amount",
        "paid_amount": "paid_amount",
        "currency": "currency",
        "policy_type": "policy_type",
    }
    values.update(overrides)
    return values


def columns_toml(**overrides: str) -> str:
    values = full_column_mapping(**overrides)
    return "\n".join(f'{key} = "{values[key]}"' for _, key, _, _ in COLUMN_MAPPING_FIELDS)


def full_columns(**overrides: str) -> ColumnConfig:
    return ColumnConfig.from_mapping(full_column_mapping(**overrides))


def test_embedding_model_registry_contains_all_snowflake_text_models() -> None:
    models = available_embedding_models()

    assert len(models) == 8
    assert "voyage-multilingual-2" in {model.key for model in models}
    assert {model.dimensions for model in models} == {768, 1024}


def test_embedding_column_round_trip() -> None:
    column = embedding_column_for_model("voyage-multilingual-2")

    assert column == "EMBEDDING_VOYAGE_MULTILINGUAL_2"
    assert model_for_embedding_column(column).key == "voyage-multilingual-2"


def test_embedding_table_uses_source_schema_and_suffix() -> None:
    assert AppConfig(columns=full_columns(), snowflake_table="DB.SCHEMA.CLAIMS").embedding_table == "DB.SCHEMA.CLAIMS_EMBEDDINGS"
    assert AppConfig(columns=full_columns(), snowflake_table='"DB"."SCHEMA"."Claims"').embedding_table == '"DB"."SCHEMA"."Claims_EMBEDDINGS"'


def test_column_config_allows_blank_optional_mappings() -> None:
    columns = full_columns(line_of_business="", claim_type="")

    assert columns.selected_columns[:2] == ["claim_id", "claim_description"]
    assert "line_of_business" not in columns.selected_columns
    assert any(row["source_column"] == "(skipped)" for row in columns.mapping_rows())


def test_app_config_reads_table_and_columns(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "app_config.toml"
    config_path.write_text(
        f"""
[snowflake]
table = "CLAIMS"
row_limit = 250

[columns]
{columns_toml(claim_id="ID", description="TEXT", line_of_business="")}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.config.DEFAULT_APP_CONFIG_PATH", config_path)

    config = AppConfig.from_app_config()

    assert config.snowflake_table == "CLAIMS"
    assert config.snowflake_row_limit == 250
    assert config.columns.claim_id == "ID"
    assert config.columns.description == "TEXT"


def test_app_config_rejects_missing_optional_mappings(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "app_config.toml"
    config_path.write_text(
        """
[snowflake]
table = "CLAIMS"

[columns]
claim_id = "claim_id"
description = "claim_description"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.config.DEFAULT_APP_CONFIG_PATH", config_path)

    with pytest.raises(ValueError, match="Set optional mappings"):
        AppConfig.from_app_config()


def test_app_config_rejects_unknown_values(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "app_config.toml"
    config_path.write_text(
        f"""
[snowflake]
table = "CLAIMS"
schema = "PUBLIC"

[columns]
{columns_toml()}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.config.DEFAULT_APP_CONFIG_PATH", config_path)

    with pytest.raises(ValueError, match="snowflake.schema"):
        AppConfig.from_app_config()


def test_config_validation_requires_source_and_required_columns() -> None:
    with pytest.raises(ValueError, match="snowflake.table"):
        AppConfig(columns=full_columns(), snowflake_table="").validate_source()
    with pytest.raises(ValueError, match="columns.claim_id"):
        AppConfig(columns=full_columns(claim_id=""), snowflake_table="CLAIMS").validate_source()
    with pytest.raises(ValueError, match="snowflake.row_limit"):
        AppConfig(columns=full_columns(), snowflake_table="CLAIMS", snowflake_row_limit=0).validate_source()
