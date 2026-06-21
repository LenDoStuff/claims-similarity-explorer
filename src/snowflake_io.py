from __future__ import annotations

import re

import pandas as pd

from src.config import ColumnConfig, quote_identifier


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def build_claims_query(table: str, columns: ColumnConfig, *, row_limit: int | None = None) -> str:
    if not table:
        raise ValueError("Required app config value cannot be blank: snowflake.table")
    if row_limit is not None and row_limit <= 0:
        raise ValueError("snowflake.row_limit must be a positive integer when set")
    columns.validate_required()
    selected = ", ".join(
        f"{source_identifier(column)} AS {quote_identifier(column)}" for column in columns.selected_columns
    )
    query = f"SELECT {selected} FROM {source_identifier(table)}"
    if row_limit is not None:
        query += f" SAMPLE ({int(row_limit)} ROWS)"
    return query


def source_identifier(identifier: str) -> str:
    if identifier.startswith('"') and identifier.endswith('"'):
        return identifier
    if SAFE_IDENTIFIER_RE.match(identifier):
        return identifier
    return quote_identifier(identifier)


def load_claims_from_snowflake(
    table: str,
    columns: ColumnConfig,
    *,
    row_limit: int | None = None,
) -> pd.DataFrame:
    from snowflake.snowpark import Session

    query = build_claims_query(table, columns, row_limit=row_limit)
    session = Session.builder.create()
    try:
        return session.sql(query).to_pandas()
    finally:
        session.close()
