from __future__ import annotations

import pandas as pd
import pytest

from src.config import ColumnConfig, SnowflakeConfig
from src.snowflake_io import build_claims_query, load_claims_from_snowflake, snowpark_connection_parameters


def test_build_claims_query_aliases_columns_to_configured_names() -> None:
    snowflake = SnowflakeConfig(
        account="acct",
        user="user",
        password="pw",
        warehouse="wh",
        database="DB",
        schema="SCHEMA",
        table="CLAIMS",
    )

    query = build_claims_query(snowflake, ColumnConfig(), limit=10)

    assert 'claim_id AS "claim_id"' in query
    assert 'claim_description AS "claim_description"' in query
    assert 'FROM "DB"."SCHEMA"."CLAIMS"' in query
    assert "LIMIT 10" in query


def test_build_claims_query_skips_blank_optional_columns() -> None:
    snowflake = SnowflakeConfig(
        account="acct",
        user="user",
        password="pw",
        warehouse="wh",
        database="DB",
        schema="SCHEMA",
        table="CLAIMS",
    )
    columns = ColumnConfig(
        line_of_business="",
        claim_type="",
        cause_of_loss="",
        damaged_object="",
        country="",
        claim_status="",
        loss_date="",
        reserve_amount="",
        paid_amount="",
        currency="",
        policy_type="",
    )

    query = build_claims_query(snowflake, columns)

    assert query == 'SELECT claim_id AS "claim_id", claim_description AS "claim_description" FROM "DB"."SCHEMA"."CLAIMS"'


def test_build_claims_query_rejects_blank_required_columns() -> None:
    snowflake = SnowflakeConfig(
        account="acct",
        user="user",
        password="pw",
        warehouse="wh",
        database="DB",
        schema="SCHEMA",
        table="CLAIMS",
    )

    with pytest.raises(ValueError, match="CLAIM_DESCRIPTION_COLUMN"):
        build_claims_query(snowflake, ColumnConfig(description=""))


def test_snowpark_connection_parameters_include_optional_role() -> None:
    snowflake = SnowflakeConfig(
        account="acct",
        user="user",
        password="pw",
        warehouse="wh",
        database="DB",
        schema="SCHEMA",
        table="CLAIMS",
        role="ROLE",
    )

    parameters = snowpark_connection_parameters(snowflake)

    assert parameters["account"] == "acct"
    assert parameters["database"] == "DB"
    assert parameters["role"] == "ROLE"


def test_load_claims_from_snowflake_uses_snowpark_session(monkeypatch) -> None:
    snowflake = SnowflakeConfig(
        account="acct",
        user="user",
        password="pw",
        warehouse="wh",
        database="DB",
        schema="SCHEMA",
        table="CLAIMS",
    )
    calls = {}

    class FakeDataFrame:
        def to_pandas(self) -> pd.DataFrame:
            calls["to_pandas"] = True
            return pd.DataFrame([{"claim_id": "1"}])

    class FakeSession:
        def sql(self, query: str) -> FakeDataFrame:
            calls["query"] = query
            return FakeDataFrame()

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr("src.snowflake_io.create_snowpark_session", lambda config: FakeSession())

    frame = load_claims_from_snowflake(snowflake, ColumnConfig(), limit=5)

    assert frame["claim_id"].tolist() == ["1"]
    assert 'FROM "DB"."SCHEMA"."CLAIMS"' in calls["query"]
    assert "LIMIT 5" in calls["query"]
    assert calls["to_pandas"]
    assert calls["closed"]
