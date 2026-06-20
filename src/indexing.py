from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.chroma_store import get_existing_collection, reset_collection, upsert_records
from src.config import AppConfig, EmbeddingModelConfig, SnowflakeConfig, versioned_collection_name
from src.diagnostics import read_json, write_json
from src.text_preprocessing import prepare_claim_records


@dataclass(frozen=True)
class IndexBuildResult:
    status: str
    collection_name: str
    manifest_path: Path
    manifest: dict[str, Any]


def build_index_from_snowflake(
    config: AppConfig,
    model_config: EmbeddingModelConfig,
    snowflake: SnowflakeConfig,
    *,
    limit: int | None = None,
    batch_size: int = 64,
    chroma_batch_size: int = 500,
) -> IndexBuildResult:
    from src.snowflake_io import load_claims_from_snowflake

    frame = load_claims_from_snowflake(snowflake, config.columns, limit=limit)
    return build_index_from_frame(
        config,
        model_config,
        frame,
        source="snowflake",
        source_identity=snowflake.qualified_table,
        batch_size=batch_size,
        chroma_batch_size=chroma_batch_size,
    )


def build_index_from_frame(
    config: AppConfig,
    model_config: EmbeddingModelConfig,
    frame: pd.DataFrame,
    *,
    source: str,
    source_identity: str,
    batch_size: int = 64,
    chroma_batch_size: int = 500,
) -> IndexBuildResult:
    records, diagnostics = prepare_claim_records(
        frame,
        config.columns,
        model_name=model_config.repo_id,
        model_path=str(model_config.model_dir),
        embedding_version=config.embedding_version,
    )
    if not records:
        raise RuntimeError("No claims with non-empty descriptions were available to index.")

    dataset_hash = records_dataset_hash(records)
    fingerprint = model_fingerprint(model_config.model_dir)
    index_hash = build_index_hash(
        source=source,
        source_identity=source_identity,
        selected_columns=config.columns.selected_columns,
        dataset_hash=dataset_hash,
        model_key=model_config.key,
        model_fingerprint=fingerprint,
        embedding_version=config.embedding_version,
    )
    collection_name = versioned_collection_name(config.collection_name, model_config.key, index_hash)
    manifest_path = config.index_manifest_path_for_model(model_config.key)
    current_manifest = read_json(manifest_path)

    if (
        current_manifest.get("index_hash") == index_hash
        and current_manifest.get("collection_name")
        and collection_has_count(config, current_manifest["collection_name"], len(records))
    ):
        return IndexBuildResult("current", current_manifest["collection_name"], manifest_path, current_manifest)

    manifest = build_manifest(
        config,
        model_config,
        source=source,
        source_identity=source_identity,
        collection_name=collection_name,
        dataset_hash=dataset_hash,
        model_fingerprint=fingerprint,
        index_hash=index_hash,
        diagnostics=diagnostics,
        record_count=len(records),
        refresh_strategy="activated_existing_index",
    )
    if collection_has_count(config, collection_name, len(records)):
        write_json(manifest_path, manifest)
        return IndexBuildResult("activated_existing", collection_name, manifest_path, manifest)

    from src.embeddings import load_embedding_model

    model = load_embedding_model(str(model_config.model_dir), model_config.repo_id)
    embeddings = model.encode_passages([record["document"] for record in records], batch_size=batch_size)
    collection = reset_collection(config.chroma_dir, collection_name)
    upsert_records(collection, records, embeddings, batch_size=chroma_batch_size)

    manifest = build_manifest(
        config,
        model_config,
        source=source,
        source_identity=source_identity,
        collection_name=collection_name,
        dataset_hash=dataset_hash,
        model_fingerprint=fingerprint,
        index_hash=index_hash,
        diagnostics=diagnostics,
        record_count=collection.count(),
        refresh_strategy="full_rebuild_with_hash_check",
    )
    write_json(manifest_path, manifest)
    return IndexBuildResult("rebuilt", collection_name, manifest_path, manifest)


def records_dataset_hash(records: list[dict[str, Any]]) -> str:
    payload = [
        {
            "id": record.get("id"),
            "document": record.get("document"),
            "metadata": record.get("metadata", {}),
        }
        for record in sorted(records, key=lambda item: str(item.get("id", "")))
    ]
    return stable_hash(payload)


def model_fingerprint(model_dir: Path) -> str:
    files = []
    for path in sorted(Path(model_dir).rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        stat = path.stat()
        files.append(
            {
                "path": path.relative_to(model_dir).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return stable_hash(files)


def build_index_hash(
    *,
    source: str,
    source_identity: str,
    selected_columns: list[str],
    dataset_hash: str,
    model_key: str,
    model_fingerprint: str,
    embedding_version: str,
) -> str:
    return stable_hash(
        {
            "source": source,
            "source_identity": source_identity,
            "selected_columns": selected_columns,
            "dataset_hash": dataset_hash,
            "model_key": model_key,
            "model_fingerprint": model_fingerprint,
            "embedding_version": embedding_version,
        }
    )


def active_collection_name(config: AppConfig, model_key: str, manifest: dict[str, Any]) -> str:
    return str(manifest.get("collection_name") or config.collection_name_for_model(model_key))


def collection_has_count(config: AppConfig, collection_name: str, expected_count: int) -> bool:
    collection = get_existing_collection(config.chroma_dir, collection_name)
    return bool(collection and collection.count() == expected_count)


def build_manifest(
    config: AppConfig,
    model_config: EmbeddingModelConfig,
    *,
    source: str,
    source_identity: str,
    collection_name: str,
    dataset_hash: str,
    model_fingerprint: str,
    index_hash: str,
    diagnostics: dict[str, Any],
    record_count: int,
    refresh_strategy: str,
) -> dict[str, Any]:
    manifest = {
        "refreshed_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "source_identity": source_identity,
        "collection_name": collection_name,
        "chroma_dir": str(config.chroma_dir),
        "model_key": model_config.key,
        "embedding_model": model_config.repo_id,
        "embedding_model_path": str(model_config.model_dir),
        "embedding_version": config.embedding_version,
        "selected_columns": config.columns.selected_columns,
        "dataset_hash": dataset_hash,
        "model_fingerprint": model_fingerprint,
        "index_hash": index_hash,
        "diagnostics": diagnostics,
        "record_count": record_count,
        "refresh_strategy": refresh_strategy,
    }
    if source == "snowflake":
        manifest["snowflake_table"] = source_identity
    return manifest


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
