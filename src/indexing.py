from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import AppConfig
from src.snowflake_io import add_embedding_columns, prepare_claims_frame


@dataclass(frozen=True)
class IndexBuildResult:
    table_name: str
    row_count: int
    models: list[str]


def initialize_embeddings(
    session: Any,
    config: AppConfig,
    model_keys: list[str],
    *,
    limit: int | None = None,
) -> IndexBuildResult:
    if not model_keys:
        raise ValueError("Select at least one embedding model.")

    frame = prepare_claims_frame(session, config, limit=limit)
    row_count = frame.count()
    if row_count == 0:
        raise RuntimeError("No claims with non-empty claim IDs and descriptions were available to index.")

    embedded = add_embedding_columns(frame, model_keys)
    embedded.write.mode("overwrite").save_as_table(config.embedding_table)
    return IndexBuildResult(config.embedding_table, row_count, model_keys)
