from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import (
    AppConfig,
    EMBEDDING_MODELS_BY_COLUMN,
    EMBEDDING_MODELS_BY_KEY,
    EmbeddingModelConfig,
    embedding_column_for_model,
)


CANONICAL_COLUMNS = {
    "claim_id": "CLAIM_ID",
    "description": "CLAIM_DESCRIPTION",
    "line_of_business": "LINE_OF_BUSINESS",
    "claim_type": "CLAIM_TYPE",
    "cause_of_loss": "CAUSE_OF_LOSS",
    "damaged_object": "DAMAGED_OBJECT",
    "country": "COUNTRY",
    "claim_status": "CLAIM_STATUS",
    "loss_date": "LOSS_DATE",
    "reserve_amount": "RESERVE_AMOUNT",
    "paid_amount": "PAID_AMOUNT",
    "currency": "CURRENCY",
    "policy_type": "POLICY_TYPE",
}

DOCUMENT_FIELDS = [
    ("description", "Claim description"),
    ("line_of_business", "Line of business"),
    ("claim_type", "Claim type"),
    ("cause_of_loss", "Cause of loss"),
    ("damaged_object", "Damaged object"),
    ("country", "Country"),
]


@dataclass(frozen=True)
class EmbeddingTableStatus:
    table_name: str
    row_count: int
    models: list[EmbeddingModelConfig]


def validate_configured_columns(schema_names: list[str], config: AppConfig) -> None:
    missing = [column for column in config.columns.selected_columns if column not in schema_names]
    if missing:
        available = ", ".join(schema_names) if schema_names else "(none)"
        raise ValueError(
            f"Configured Snowflake column(s) not found: {', '.join(missing)}. "
            f"Available table columns: {available}"
        )


def clean_text_column(column: Any) -> Any:
    from snowflake.snowpark.functions import coalesce, lit, regexp_replace, trim

    return trim(regexp_replace(coalesce(column.cast("string"), lit("")), lit(r"\s+"), lit(" ")))


def prepare_claims_frame(session: Any, config: AppConfig) -> Any:
    from snowflake.snowpark.functions import (
        array_construct_compact,
        array_to_string,
        col,
        concat,
        length,
        lit,
        sha2,
        when,
        year,
    )

    config.validate_source()

    frame = session.table(config.snowflake_table)
    validate_configured_columns(list(frame.schema.names), config)
    if config.snowflake_row_limit is not None:
        frame = frame.sample(n=config.snowflake_row_limit)

    source = {}
    for attr in CANONICAL_COLUMNS:
        name = getattr(config.columns, attr)
        if name:
            escaped_name = name.replace('"', '""')
            source[attr] = col(f'"{escaped_name}"')
    claim_id = clean_text_column(source["claim_id"])
    description = clean_text_column(source["description"])
    frame = frame.filter((length(claim_id) > 0) & (length(description) > 0))

    document_parts = []
    for attr, label in DOCUMENT_FIELDS:
        if attr not in source:
            continue
        value = clean_text_column(source[attr])
        document_parts.append(
            when(length(value) > 0, concat(lit(f"{label}: "), value))
        )
    document = array_to_string(array_construct_compact(*document_parts), lit("\n"))

    selected = [
        claim_id.alias("CLAIM_ID"),
        description.alias("CLAIM_DESCRIPTION"),
        document.alias("DOCUMENT"),
    ]
    for attr, canonical in CANONICAL_COLUMNS.items():
        if attr in {"claim_id", "description"} or attr not in source:
            continue
        selected.append(source[attr].alias(canonical))
    if "loss_date" in source:
        selected.append(year(source["loss_date"]).alias("LOSS_YEAR"))
    selected.extend(
        [
            length(description).alias("DESCRIPTION_LENGTH"),
            sha2(document, 256).alias("SOURCE_TEXT_HASH"),
        ]
    )
    return frame.select(*selected)


def initialized_models(frame: Any) -> list[EmbeddingModelConfig]:
    from snowflake.snowpark.types import VectorType

    models = []
    for field in frame.schema.fields:
        model = EMBEDDING_MODELS_BY_COLUMN.get(field.name.strip('"').upper())
        if model is not None and isinstance(field.datatype, VectorType):
            models.append(model)
    return models


def get_embedding_table_status(session: Any, config: AppConfig) -> EmbeddingTableStatus | None:
    from snowflake.snowpark.exceptions import SnowparkSQLException

    try:
        frame = session.table(config.embedding_table)
        models = initialized_models(frame)
        row_count = frame.count()
    except SnowparkSQLException as exc:
        if exc.sql_error_code == 2003:
            return None
        raise
    return EmbeddingTableStatus(config.embedding_table, row_count, models)


@dataclass(frozen=True)
class IndexBuildResult:
    table_name: str
    row_count: int
    models: list[str]


def initialize_embeddings(
    session: Any,
    config: AppConfig,
    model_keys: list[str],
) -> IndexBuildResult:
    from snowflake.snowpark.functions import ai_embed, col

    if not model_keys:
        raise ValueError("Select at least one embedding model.")

    frame = prepare_claims_frame(session, config)
    row_count = frame.count()
    if row_count == 0:
        raise RuntimeError("No claims with non-empty claim IDs and descriptions were available to index.")

    for model_key in model_keys:
        if model_key not in EMBEDDING_MODELS_BY_KEY:
            raise ValueError(f"Unknown Snowflake embedding model: {model_key}")
        frame = frame.with_column(
            embedding_column_for_model(model_key),
            ai_embed(model_key, col("DOCUMENT")),
        )

    frame.write.mode("overwrite").save_as_table(config.embedding_table)
    return IndexBuildResult(config.embedding_table, row_count, model_keys)


def collect_search_options(session: Any, table_name: str) -> dict[str, Any]:
    from snowflake.snowpark.functions import col, max as max_, min as min_

    frame = session.table(table_name)
    schema_names = set(frame.schema.names)

    equality = {}
    for column in ["LINE_OF_BUSINESS", "CLAIM_TYPE", "CAUSE_OF_LOSS", "COUNTRY", "CLAIM_STATUS", "POLICY_TYPE"]:
        if column not in schema_names:
            continue
        rows = (
            frame.select(col(column))
            .filter(col(column).is_not_null())
            .distinct()
            .sort(col(column))
            .collect()
        )
        equality[column.lower()] = [row[0] for row in rows if row[0] not in (None, "")]

    ranges = {}
    for column in ["LOSS_YEAR", "RESERVE_AMOUNT", "PAID_AMOUNT"]:
        if column not in schema_names:
            continue
        row = frame.select(
            min_(col(column)).alias("MIN_VALUE"),
            max_(col(column)).alias("MAX_VALUE"),
        ).collect()[0]
        ranges[column.lower()] = (row[0], row[1])

    claim_columns = [
        column
        for column in ["CLAIM_ID", "LINE_OF_BUSINESS", "CLAIM_TYPE", "COUNTRY", "CLAIM_STATUS"]
        if column in schema_names
    ]
    claims = [
        row.as_dict(recursive=True)
        for row in frame.select(*[col(column) for column in claim_columns]).sort(col("CLAIM_ID")).collect()
    ]
    return {"equality": equality, "ranges": ranges, "claims": claims}


def get_claim_document(session: Any, table_name: str, claim_id: str) -> str:
    from snowflake.snowpark.functions import col, lit

    rows = (
        session.table(table_name)
        .filter(col("CLAIM_ID") == lit(claim_id))
        .select(col("DOCUMENT"))
        .limit(1)
        .collect()
    )
    return str(rows[0][0]) if rows else ""
