from __future__ import annotations

import pytest

from src.config import AppConfig, ColumnConfig
from src.indexing import initialize_embeddings


class FakeWriter:
    def __init__(self, calls: dict) -> None:
        self.calls = calls

    def mode(self, mode: str):
        self.calls["mode"] = mode
        return self

    def save_as_table(self, table_name: str) -> None:
        self.calls["saved_table"] = table_name


class FakeFrame:
    def __init__(self, calls: dict, count: int) -> None:
        self.calls = calls
        self._count = count
        self.write = FakeWriter(calls)

    def count(self) -> int:
        return self._count


def config() -> AppConfig:
    return AppConfig(
        columns=ColumnConfig(claim_id="CLAIM_ID", description="DESCRIPTION"),
        snowflake_table="DB.SCHEMA.CLAIMS",
    )


def test_initialize_embeddings_overwrites_derived_table(monkeypatch) -> None:
    calls = {}
    base = FakeFrame(calls, 12)
    embedded = FakeFrame(calls, 12)
    monkeypatch.setattr("src.indexing.prepare_claims_frame", lambda *args, **kwargs: base)
    monkeypatch.setattr("src.indexing.add_embedding_columns", lambda frame, models: embedded)

    result = initialize_embeddings(object(), config(), ["voyage-multilingual-2", "e5-base-v2"])

    assert calls["mode"] == "overwrite"
    assert calls["saved_table"] == "DB.SCHEMA.CLAIMS_EMBEDDINGS"
    assert result.row_count == 12
    assert result.models == ["voyage-multilingual-2", "e5-base-v2"]


def test_initialize_embeddings_requires_models() -> None:
    with pytest.raises(ValueError, match="at least one"):
        initialize_embeddings(object(), config(), [])


def test_initialize_embeddings_rejects_empty_prepared_data(monkeypatch) -> None:
    monkeypatch.setattr("src.indexing.prepare_claims_frame", lambda *args, **kwargs: FakeFrame({}, 0))

    with pytest.raises(RuntimeError, match="No claims"):
        initialize_embeddings(object(), config(), ["voyage-multilingual-2"])
