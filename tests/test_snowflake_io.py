from __future__ import annotations

from src.config import ColumnConfig, SnowflakeConfig
from src.snowflake_io import build_claims_query


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

