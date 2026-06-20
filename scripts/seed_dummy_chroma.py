from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DUMMY_CLAIMS_PATH = Path(__file__).with_name("dummy_claims.json")


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()

    from src.config import AppConfig, available_embedding_models, get_embedding_model

    config = AppConfig.from_env()
    frame = build_dummy_claims()

    if args.all_models:
        models = available_embedding_models()
    else:
        models = [get_embedding_model(args.model_key or config.model_key)]

    for model_config in models:
        seed_model(config, model_config, frame, args, started_at)


def seed_model(config, model_config, frame: pd.DataFrame, args: argparse.Namespace, started_at: float) -> None:
    log_step(started_at, f"Seeding {model_config.key}")
    log_step(started_at, "Importing seed dependencies")
    from src.chroma_store import get_collection
    from src.clustering import (
        assign_clusters,
        build_cluster_summary,
        cluster_embeddings,
        load_cluster_input,
        write_cluster_artifact,
        write_cluster_map_artifact,
    )
    from src.diagnostics import write_json
    from src.indexing import build_index_from_frame

    clusters_path = config.clusters_path_for_model(model_config.key)
    cluster_map_path = config.cluster_map_path_for_model(model_config.key)
    result = build_index_from_frame(
        config,
        model_config,
        frame,
        source="dummy_seed",
        source_identity=str(DUMMY_CLAIMS_PATH),
        batch_size=args.batch_size,
    )
    collection_name = result.collection_name
    manifest_path = result.manifest_path
    log_step(started_at, f"Index status: {result.status}")
    collection = get_collection(config.chroma_dir, collection_name)

    log_step(started_at, "Clustering embeddings")
    ids, documents, metadatas, stored_embeddings = load_cluster_input(collection)
    labels, centers = cluster_embeddings(stored_embeddings, args.clusters)
    assign_clusters(collection, ids, labels)
    cluster_summary = build_cluster_summary(ids, documents, metadatas, stored_embeddings, labels, centers)
    cluster_summary["clustered_at_utc"] = datetime.now(UTC).isoformat()
    cluster_summary["algorithm"] = "KMeans"
    cluster_summary["random_state"] = 42
    cluster_summary["index_hash"] = result.manifest.get("index_hash")
    write_cluster_artifact(clusters_path, cluster_summary, n_clusters=min(args.clusters, len(ids)))
    log_step(started_at, "Building cluster map with fast SVD projection")
    cluster_map = build_dummy_cluster_map(ids, documents, metadatas, stored_embeddings, labels)
    cluster_map["created_at_utc"] = datetime.now(UTC).isoformat()
    cluster_map["index_hash"] = result.manifest.get("index_hash")
    write_cluster_map_artifact(cluster_map_path, cluster_map)
    log_step(started_at, "Finished cluster map")

    manifest = dict(result.manifest)
    manifest["cluster_count"] = len(set(int(label) for label in labels))
    write_json(manifest_path, manifest)

    print(f"Seeded {collection.count()} dummy claims into {collection_name}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote clusters to {clusters_path}")
    print(f"Wrote cluster map to {cluster_map_path}")


def log_step(started_at: float, message: str) -> None:
    print(f"[{time.perf_counter() - started_at:6.1f}s] {message}", flush=True)


def build_dummy_cluster_map(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict:
    from src.clustering import claim_description_from_document

    coordinates = project_embeddings_fast(embeddings)
    points = []
    for claim_id, document, metadata, label, coordinate in zip(ids, documents, metadatas, labels, coordinates):
        row = dict(metadata or {})
        points.append(
            {
                "claim_id": str(row.get("claim_id") or claim_id),
                "cluster_id": int(label),
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
                "description": claim_description_from_document(document),
            }
        )
    return {
        "projection": {
            "method": "SVD",
            "metric": "cosine",
            "random_state": 42,
            "n_neighbors": None,
        },
        "points": points,
    }


def project_embeddings_fast(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(embeddings) == 1:
        return np.asarray([[0.0, 0.0]], dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ components[:2].T
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(len(coordinates), dtype=np.float32)])
    return np.asarray(coordinates[:, :2], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed local ChromaDB with synthetic demo claims.")
    parser.add_argument("--clusters", type=int, default=8, help="Requested number of KMeans clusters.")
    parser.add_argument("--batch-size", type=int, default=32, help="SentenceTransformer embedding batch size.")
    parser.add_argument("--all-models", action="store_true", help="Seed one Chroma collection per local model.")
    parser.add_argument(
        "--model-key",
        default=None,
        help="Local embedding model folder name under models/embeddings.",
    )
    return parser.parse_args()


def build_dummy_claims() -> pd.DataFrame:
    claims = json.loads(DUMMY_CLAIMS_PATH.read_text(encoding="utf-8"))
    return pd.DataFrame(claims)


if __name__ == "__main__":
    main()
