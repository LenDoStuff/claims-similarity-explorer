from __future__ import annotations

import pandas as pd
import pytest

from src.config import ColumnConfig
from src.snowflake_io import build_claims_query, load_claims_from_snowflake


def test_build_claims_query_aliases_columns_to_configured_names() -> None:
    query = build_claims_query("CLAIMS", ColumnConfig(), row_limit=10)

    assert 'claim_id AS "claim_id"' in query
    assert 'claim_description AS "claim_description"' in query
    assert "FROM CLAIMS" in query
    assert "SAMPLE (10 ROWS)" in query


def test_build_claims_query_skips_blank_optional_columns() -> None:
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

    query = build_claims_query("CLAIMS", columns)

    assert query == 'SELECT claim_id AS "claim_id", claim_description AS "claim_description" FROM CLAIMS'


def test_build_claims_query_rejects_blank_table() -> None:
    with pytest.raises(ValueError, match="snowflake.table"):
        build_claims_query("", ColumnConfig())


def test_build_claims_query_rejects_non_positive_row_limit() -> None:
    with pytest.raises(ValueError, match="snowflake.row_limit"):
        build_claims_query("CLAIMS", ColumnConfig(), row_limit=0)


def test_build_claims_query_rejects_blank_required_columns() -> None:
    with pytest.raises(ValueError, match="columns.description"):
        build_claims_query("CLAIMS", ColumnConfig(description=""))


def test_load_claims_from_snowflake_uses_default_snowpark_session(monkeypatch) -> None:
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

    class FakeBuilder:
        def create(self) -> FakeSession:
            calls["created"] = True
            return FakeSession()

    class FakeSessionClass:
        builder = FakeBuilder()

    monkeypatch.setattr("snowflake.snowpark.Session", FakeSessionClass)

    frame = load_claims_from_snowflake("CLAIMS", ColumnConfig(), row_limit=5)

    assert frame["claim_id"].tolist() == ["1"]
    assert "FROM CLAIMS" in calls["query"]
    assert "SAMPLE (5 ROWS)" in calls["query"]
    assert calls["created"]
    assert calls["to_pandas"]
    assert calls["closed"]
