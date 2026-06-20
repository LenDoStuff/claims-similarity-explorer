from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
from chromadb.api.models.Collection import Collection


def get_chroma_client(chroma_dir: Path) -> chromadb.PersistentClient:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_collection(chroma_dir: Path, collection_name: str) -> Collection:
    client = get_chroma_client(chroma_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def get_existing_collection(chroma_dir: Path, collection_name: str) -> Collection | None:
    client = get_chroma_client(chroma_dir)
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        return None


def reset_collection(chroma_dir: Path, collection_name: str) -> Collection:
    client = get_chroma_client(chroma_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_records(
    collection: Collection,
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    batch_size: int = 500,
) -> None:
    if len(records) != len(embeddings):
        raise ValueError("Record and embedding counts do not match.")
    for start in range(0, len(records), batch_size):
        end = start + batch_size
        batch = records[start:end]
        collection.upsert(
            ids=[record["id"] for record in batch],
            documents=[record["document"] for record in batch],
            metadatas=[record["metadata"] for record in batch],
            embeddings=embeddings[start:end].tolist(),
        )


def fetch_all_records(collection: Collection, *, include_embeddings: bool = False) -> dict[str, Any]:
    include = ["documents", "metadatas"]
    if include_embeddings:
        include.append("embeddings")
    return collection.get(include=include)


def collection_to_frame(collection: Collection) -> pd.DataFrame:
    data = fetch_all_records(collection)
    rows: list[dict[str, Any]] = []
    for claim_id, document, metadata in zip(data.get("ids", []), data.get("documents", []), data.get("metadatas", [])):
        row = dict(metadata or {})
        row["id"] = claim_id
        row["document"] = document
        rows.append(row)
    return pd.DataFrame(rows)


def get_existing_claim_options(collection: Collection) -> pd.DataFrame:
    frame = collection_to_frame(collection)
    if frame.empty:
        return frame
    cols = [col for col in ["claim_id", "line_of_business", "claim_type", "country", "claim_status", "loss_year"] if col in frame]
    return frame[cols].sort_values("claim_id")


def update_metadata_values(collection: Collection, ids: list[str], updates: list[dict[str, Any]]) -> None:
    if len(ids) != len(updates):
        raise ValueError("Metadata update ids and values do not match.")
    if not ids:
        return
    existing = collection.get(ids=ids, include=["metadatas"])
    current_by_id = dict(zip(existing.get("ids", []), existing.get("metadatas", [])))
    merged = []
    for claim_id, update in zip(ids, updates):
        metadata = dict(current_by_id.get(claim_id) or {})
        metadata.update({key: value for key, value in update.items() if value is not None})
        merged.append(metadata)
    collection.update(ids=ids, metadatas=merged)
