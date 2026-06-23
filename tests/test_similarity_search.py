from __future__ import annotations

import pandas as pd
import pytest

from src.similarity_search import SearchFilters, query_similar_claims


class FakeDataFrame:
    def __init__(self, calls: dict, *, query: bool = False) -> None:
        self.calls = calls
        self.query = query
        self.schema = type(
            "Schema",
            (),
            {
                "names": [
                    "CLAIM_ID",
                    "DOCUMENT",
                    "COUNTRY",
                    "LOSS_YEAR",
                    "EMBEDDING_VOYAGE_MULTILINGUAL_2",
                ]
            },
        )()

    def filter(self, condition):
        self.calls.setdefault("filters", []).append(str(condition))
        return self

    def select(self, *columns):
        key = "query_select" if self.query else "result_select"
        self.calls[key] = [str(column) for column in columns]
        return self

    def cross_join(self, other):
        self.calls["cross_join"] = True
        return self

    def sort(self, *columns):
        self.calls["sort"] = [
            type(column._expression.direction).__name__
            if hasattr(column._expression, "direction")
            else str(column)
            for column in columns
        ]
        return self

    def limit(self, count: int):
        self.calls["limit"] = count
        return self

    def to_pandas(self):
        return pd.DataFrame(
            [
                {
                    "CLAIM_ID": "2",
                    "DOCUMENT": "Claim description: water damage",
                    "COUNTRY": "DE",
                    "LOSS_YEAR": 2024,
                    "METRIC_VALUE": 0.91,
                }
            ]
        )


class FakeSession:
    def __init__(self, calls: dict) -> None:
        self.calls = calls
        self.frame = FakeDataFrame(calls)

    def table(self, table_name: str):
        self.calls["table"] = table_name
        return self.frame

    def create_dataframe(self, rows, schema):
        self.calls["query_rows"] = rows
        self.calls["query_schema"] = schema
        return FakeDataFrame(self.calls, query=True)


@pytest.mark.parametrize(
    ("metric_key", "function_name", "sort_direction"),
    [
        ("cosine", "VECTOR_COSINE_SIMILARITY", "Descending"),
        ("inner_product", "VECTOR_INNER_PRODUCT", "Descending"),
        ("l1", "VECTOR_L1_DISTANCE", "Ascending"),
        ("l2", "VECTOR_L2_DISTANCE", "Ascending"),
    ],
)
def test_query_similar_claims_uses_selected_snowflake_metric(
    metric_key: str,
    function_name: str,
    sort_direction: str,
) -> None:
    calls = {}
    result = query_similar_claims(
        FakeSession(calls),
        "DB.SCHEMA.CLAIMS_EMBEDDINGS",
        "voyage-multilingual-2",
        "water leakage",
        metric_key=metric_key,
        filters=SearchFilters(equality={"country": "DE"}, loss_year_range=(2023, 2025)),
        top_n=10,
        exclude_claim_id="1",
    )

    assert calls["table"] == "DB.SCHEMA.CLAIMS_EMBEDDINGS"
    assert calls["query_rows"] == [["water leakage"]]
    assert len(calls["filters"]) == 4
    assert calls["cross_join"]
    assert any(function_name in column for column in calls["result_select"])
    assert calls["sort"][0] == sort_direction
    assert calls["limit"] == 10
    assert result.iloc[0]["claim_id"] == "2"
    assert result.iloc[0]["metric_value"] == 0.91
