from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import EMBEDDING_MODELS_BY_KEY, embedding_column_for_model


EQUALITY_FILTER_FIELDS = [
    "line_of_business",
    "claim_type",
    "cause_of_loss",
    "country",
    "claim_status",
    "policy_type",
]


@dataclass(frozen=True)
class SimilarityMetric:
    key: str
    label: str
    function_name: str
    descending: bool
    bounded_range: tuple[float, float] | None = None

    @property
    def direction_label(self) -> str:
        return "Higher is better" if self.descending else "Lower is better"


SIMILARITY_METRICS = (
    SimilarityMetric("cosine", "Cosine similarity", "VECTOR_COSINE_SIMILARITY", True, (-1.0, 1.0)),
    SimilarityMetric("inner_product", "Inner product", "VECTOR_INNER_PRODUCT", True),
    SimilarityMetric("l1", "Manhattan distance", "VECTOR_L1_DISTANCE", False),
    SimilarityMetric("l2", "Euclidean distance", "VECTOR_L2_DISTANCE", False),
)
SIMILARITY_METRICS_BY_KEY = {metric.key: metric for metric in SIMILARITY_METRICS}


@dataclass(frozen=True)
class SearchFilters:
    equality: dict[str, Any] = field(default_factory=dict)
    loss_year_range: tuple[int | None, int | None] = (None, None)
    reserve_amount_range: tuple[float | None, float | None] = (None, None)
    paid_amount_range: tuple[float | None, float | None] = (None, None)


def query_similar_claims(
    session: Any,
    table_name: str,
    model_key: str,
    query_text: str,
    *,
    metric_key: str,
    filters: SearchFilters,
    top_n: int,
    exclude_claim_id: str | None = None,
) -> pd.DataFrame:
    from snowflake.snowpark.functions import ai_embed, call_builtin, col, lit

    if model_key not in EMBEDDING_MODELS_BY_KEY:
        raise ValueError(f"Unknown Snowflake embedding model: {model_key}")
    try:
        metric = SIMILARITY_METRICS_BY_KEY[metric_key]
    except KeyError as exc:
        raise ValueError(f"Unknown similarity metric: {metric_key}") from exc
    frame = session.table(table_name)

    for field in EQUALITY_FILTER_FIELDS:
        value = filters.equality.get(field)
        column = field.upper()
        if value not in (None, "", "All") and column in frame.schema.names:
            frame = frame.filter(col(column) == lit(value))

    frame = apply_range(frame, "LOSS_YEAR", filters.loss_year_range)
    frame = apply_range(frame, "RESERVE_AMOUNT", filters.reserve_amount_range)
    frame = apply_range(frame, "PAID_AMOUNT", filters.paid_amount_range)
    if exclude_claim_id:
        frame = frame.filter(col("CLAIM_ID") != lit(exclude_claim_id))

    query = session.create_dataframe([[query_text]], schema=["QUERY_TEXT"]).select(
        ai_embed(model_key, col("QUERY_TEXT")).alias("QUERY_VECTOR")
    )
    metric_value = call_builtin(
        metric.function_name,
        col(embedding_column_for_model(model_key)),
        col("QUERY_VECTOR"),
    ).alias("METRIC_VALUE")

    result_columns = [
        column
        for column in [
            "CLAIM_ID",
            "DOCUMENT",
            "LINE_OF_BUSINESS",
            "CLAIM_TYPE",
            "CAUSE_OF_LOSS",
            "DAMAGED_OBJECT",
            "COUNTRY",
            "CLAIM_STATUS",
            "LOSS_DATE",
            "LOSS_YEAR",
            "RESERVE_AMOUNT",
            "PAID_AMOUNT",
            "CURRENCY",
            "POLICY_TYPE",
        ]
        if column in frame.schema.names
    ]
    results = (
        frame.cross_join(query)
        .select(*[col(column) for column in result_columns], metric_value)
        .sort(
            col("METRIC_VALUE").desc() if metric.descending else col("METRIC_VALUE").asc(),
            col("CLAIM_ID"),
        )
        .limit(top_n)
        .to_pandas()
    )
    results.columns = [column.lower() for column in results.columns]
    return results


def apply_range(frame: Any, column: str, bounds: tuple[int | float | None, int | float | None]) -> Any:
    from snowflake.snowpark.functions import col, lit

    if column not in frame.schema.names:
        return frame
    lower, upper = bounds
    if lower is not None:
        frame = frame.filter(col(column) >= lit(lower))
    if upper is not None:
        frame = frame.filter(col(column) <= lit(upper))
    return frame
