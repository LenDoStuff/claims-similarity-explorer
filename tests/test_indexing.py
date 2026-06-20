from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AppConfig, EmbeddingModelConfig
from src.indexing import build_index_hash, model_fingerprint, records_dataset_hash
from src.indexing import build_index_from_frame


def test_records_dataset_hash_is_stable_and_content_sensitive() -> None:
    records = [
        {"id": "2", "document": "fire", "metadata": {"country": "DE"}},
        {"id": "1", "document": "water", "metadata": {"country": "AT"}},
    ]
    reordered = list(reversed(records))
    changed = [
        {"id": "2", "document": "fire", "metadata": {"country": "DE"}},
        {"id": "1", "document": "water damage", "metadata": {"country": "AT"}},
    ]

    assert records_dataset_hash(records) == records_dataset_hash(reordered)
    assert records_dataset_hash(records) != records_dataset_hash(changed)


def test_model_fingerprint_ignores_cache_and_tracks_model_files(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    first = model_fingerprint(model_dir)
    cache_dir = model_dir / ".cache"
    cache_dir.mkdir()
    (cache_dir / "ignored.bin").write_text("changed", encoding="utf-8")

    assert model_fingerprint(model_dir) == first

    (model_dir / "config.json").write_text('{"changed": true}', encoding="utf-8")

    assert model_fingerprint(model_dir) != first


def test_index_hash_changes_with_dataset_or_model() -> None:
    base = {
        "source": "snowflake",
        "source_identity": '"DB"."SCHEMA"."CLAIMS"',
        "selected_columns": ["claim_id", "claim_description"],
        "dataset_hash": "dataset-a",
        "model_key": "multilingual-e5-small",
        "model_fingerprint": "model-a",
        "embedding_version": "e5-v1",
    }

    first = build_index_hash(**base)
    assert build_index_hash(**{**base, "dataset_hash": "dataset-b"}) != first
    assert build_index_hash(**{**base, "model_fingerprint": "model-b"}) != first


def test_build_index_from_frame_reuses_current_collection(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "models" / "embeddings" / "test-embedding"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    config = AppConfig(chroma_dir=tmp_path / "chroma", artifacts_dir=tmp_path / "artifacts")
    model_config = EmbeddingModelConfig(
        key="test-embedding",
        label="Test Embedding",
        repo_id="test-embedding",
        model_dir=model_dir,
        notes="",
    )
    frame = pd.DataFrame(
        [
            {
                "claim_id": "1",
                "claim_description": "Water leakage in warehouse",
                "line_of_business": "Property",
                "claim_type": "Water Damage",
                "cause_of_loss": "Leakage",
                "damaged_object": "Stock",
                "country": "DE",
            }
        ]
    )

    class FakeModel:
        def encode_passages(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr("src.embeddings.load_embedding_model", lambda *args, **kwargs: FakeModel())

    first = build_index_from_frame(config, model_config, frame, source="test", source_identity="claims")
    second = build_index_from_frame(config, model_config, frame, source="test", source_identity="claims")

    assert first.status == "rebuilt"
    assert second.status == "current"
    assert first.collection_name == second.collection_name
    assert first.manifest["record_count"] == 1
