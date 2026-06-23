from __future__ import annotations

import pytest
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.types import StringType, StructField, StructType, VectorType

from src.config import AppConfig, ColumnConfig
from src.snowflake_io import (
    get_embedding_table_status,
    initialize_embeddings,
    initialized_models,
    prepare_claims_frame,
)


def required_columns_only() -> ColumnConfig:
    return ColumnConfig(claim_id="CLAIM_ID", description="CLAIM_DESCRIPTION")


class FakeDataFrame:
    def __init__(self, schema_names: list[str], calls: dict, *, count: int = 1) -> None:
        self.schema = type("Schema", (), {"names": schema_names})()
        self.calls = calls
        self._count = count
        self.write = FakeWriter(calls)

    def sample(self, *, n: int):
        self.calls["sample_n"] = n
        return self

    def filter(self, condition):
        self.calls.setdefault("filters", []).append(str(condition))
        return self

    def select(self, *columns):
        self.calls["selected"] = [str(column) for column in columns]
        return self

    def with_column(self, name, value):
        self.calls.setdefault("embedding_columns", []).append((name, str(value)))
        return self

    def count(self):
        return self._count


class FakeWriter:
    def __init__(self, calls: dict) -> None:
        self.calls = calls

    def mode(self, mode: str):
        self.calls["mode"] = mode
        return self

    def save_as_table(self, table_name: str) -> None:
        self.calls["saved_table"] = table_name


class FakeSession:
    def __init__(self, frame: FakeDataFrame, calls: dict) -> None:
        self.frame = frame
        self.calls = calls

    def table(self, table_name: str):
        self.calls["table"] = table_name
        return self.frame


def test_prepare_claims_frame_uses_snowpark_and_builds_derived_columns() -> None:
    calls = {}
    frame = FakeDataFrame(["CLAIM_ID", "CLAIM_DESCRIPTION"], calls)
    config = AppConfig(
        columns=required_columns_only(),
        snowflake_table="DB.SCHEMA.CLAIMS",
        snowflake_row_limit=5,
    )

    result = prepare_claims_frame(FakeSession(frame, calls), config)

    assert result is frame
    assert calls["table"] == "DB.SCHEMA.CLAIMS"
    assert calls["sample_n"] == 5
    assert calls["filters"]
    assert any('AS "DOCUMENT"' in column for column in calls["selected"])
    assert any('AS "SOURCE_TEXT_HASH"' in column for column in calls["selected"])


def test_prepare_claims_frame_validates_source_columns_before_sampling() -> None:
    calls = {}
    frame = FakeDataFrame(["CLAIM_ID"], calls)
    config = AppConfig(columns=required_columns_only(), snowflake_table="CLAIMS", snowflake_row_limit=5)

    with pytest.raises(ValueError, match="CLAIM_DESCRIPTION"):
        prepare_claims_frame(FakeSession(frame, calls), config)

    assert "sample_n" not in calls


def test_initialize_embeddings_overwrites_table_with_selected_models(monkeypatch) -> None:
    calls = {}
    frame = FakeDataFrame(["DOCUMENT"], calls, count=12)
    config = AppConfig(columns=required_columns_only(), snowflake_table="DB.SCHEMA.CLAIMS")
    monkeypatch.setattr("src.snowflake_io.prepare_claims_frame", lambda *args: frame)

    result = initialize_embeddings(object(), config, ["voyage-multilingual-2", "e5-base-v2"])

    assert [name for name, _ in calls["embedding_columns"]] == [
        "EMBEDDING_VOYAGE_MULTILINGUAL_2",
        "EMBEDDING_E5_BASE_V2",
    ]
    assert all("ai_embed" in expression for _, expression in calls["embedding_columns"])
    assert calls["mode"] == "overwrite"
    assert calls["saved_table"] == "DB.SCHEMA.CLAIMS_EMBEDDINGS"
    assert result.row_count == 12


def test_initialize_embeddings_requires_a_model() -> None:
    with pytest.raises(ValueError, match="at least one"):
        initialize_embeddings(object(), AppConfig(columns=required_columns_only(), snowflake_table="CLAIMS"), [])


def test_initialize_embeddings_rejects_empty_data(monkeypatch) -> None:
    frame = FakeDataFrame(["DOCUMENT"], {}, count=0)
    monkeypatch.setattr("src.snowflake_io.prepare_claims_frame", lambda *args: frame)

    with pytest.raises(RuntimeError, match="No claims"):
        initialize_embeddings(
            object(),
            AppConfig(columns=required_columns_only(), snowflake_table="CLAIMS"),
            ["voyage-multilingual-2"],
        )


def test_initialized_models_uses_vector_schema_columns() -> None:
    frame = type(
        "Frame",
        (),
        {
            "schema": StructType(
                [
                    StructField("CLAIM_ID", StringType()),
                    StructField("EMBEDDING_VOYAGE_MULTILINGUAL_2", VectorType(float, 1024)),
                    StructField("EMBEDDING_E5_BASE_V2", VectorType(float, 768)),
                ]
            )
        },
    )()

    assert [model.key for model in initialized_models(frame)] == [
        "voyage-multilingual-2",
        "e5-base-v2",
    ]


def test_missing_embedding_table_uses_snowflake_error_code() -> None:
    class MissingSession:
        def table(self, table_name: str):
            raise SnowparkSQLException("missing", sql_error_code=2003)

    config = AppConfig(columns=required_columns_only(), snowflake_table="CLAIMS")

    assert get_embedding_table_status(MissingSession(), config) is None


def test_other_snowflake_errors_are_not_hidden() -> None:
    class BrokenSession:
        def table(self, table_name: str):
            raise SnowparkSQLException("denied", sql_error_code=2001)

    config = AppConfig(columns=required_columns_only(), snowflake_table="CLAIMS")

    with pytest.raises(SnowparkSQLException, match="denied"):
        get_embedding_table_status(BrokenSession(), config)
