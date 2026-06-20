from __future__ import annotations

import re
from typing import Any

import pandas as pd
import snowflake.connector

from src.config import ColumnConfig, SnowflakeConfig, quote_identifier


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def build_claims_query(snowflake: SnowflakeConfig, columns: ColumnConfig, *, limit: int | None = None) -> str:
    selected = ", ".join(
        f"{source_identifier(column)} AS {quote_identifier(column)}" for column in columns.selected_columns
    )
    query = f"SELECT {selected} FROM {snowflake.qualified_table}"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return query


def source_identifier(identifier: str) -> str:
    if identifier.startswith('"') and identifier.endswith('"'):
        return identifier
    if SAFE_IDENTIFIER_RE.match(identifier):
        return identifier
    return quote_identifier(identifier)


def load_claims_from_snowflake(
    snowflake: SnowflakeConfig,
    columns: ColumnConfig,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    connection_kwargs: dict[str, Any] = {
        "account": snowflake.account,
        "user": snowflake.user,
        "password": snowflake.password,
        "warehouse": snowflake.warehouse,
        "database": snowflake.database,
        "schema": snowflake.schema,
    }
    if snowflake.role:
        connection_kwargs["role"] = snowflake.role

    query = build_claims_query(snowflake, columns, limit=limit)
    with snowflake.connector.connect(**connection_kwargs) as conn:
        return pd.read_sql(query, conn)
