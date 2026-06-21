from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    args = parse_args()

    from src.chroma_store import get_existing_collection
    from src.clustering import (
        assign_clusters,
        build_cluster_map,
        build_cluster_summary,
        cluster_embeddings,
        load_cluster_input,
        write_cluster_artifact,
        write_cluster_map_artifact,
    )
    from src.config import AppConfig, get_embedding_model
    from src.diagnostics import read_json
    from src.indexing import active_collection_name

    config = AppConfig.from_app_config()
    model_config = get_embedding_model(args.model_key or config.model_key)
    manifest = read_json(config.index_manifest_path_for_model(model_config.key))
    collection_name = active_collection_name(config, model_config.key, manifest)
    if not collection_name:
        raise RuntimeError(
            f"No active hash-based index manifest collection is available for {model_config.key}. "
            "Use Index Setup or run scripts/build_chroma_index.py first."
        )
    collection = get_existing_collection(config.chroma_dir, collection_name)
    if collection is None:
        raise RuntimeError(
            f"Active Chroma collection '{collection_name}' is missing. "
            "Use Index Setup or run scripts/build_chroma_index.py first."
        )
    ids, documents, metadatas, embeddings = load_cluster_input(collection)
    if len(ids) == 0:
        raise RuntimeError("No Chroma records with embeddings are available. Run build_chroma_index.py first.")

    labels, centers = cluster_embeddings(embeddings, args.clusters)
    assign_clusters(collection, ids, labels)
    summary = build_cluster_summary(ids, documents, metadatas, embeddings, labels, centers)
    summary["clustered_at_utc"] = datetime.now(UTC).isoformat()
    summary["algorithm"] = "KMeans"
    summary["random_state"] = 42
    summary["index_hash"] = manifest.get("index_hash")
    write_cluster_artifact(config.clusters_path_for_model(model_config.key), summary, n_clusters=min(args.clusters, len(ids)))
    cluster_map = build_cluster_map(ids, documents, metadatas, embeddings, labels)
    cluster_map["created_at_utc"] = datetime.now(UTC).isoformat()
    cluster_map["index_hash"] = manifest.get("index_hash")
    write_cluster_map_artifact(config.cluster_map_path_for_model(model_config.key), cluster_map)
    print(f"Assigned clusters for {len(ids)} claims.")
    print(f"Wrote cluster summary to {config.clusters_path_for_model(model_config.key)}")
    print(f"Wrote cluster map to {config.cluster_map_path_for_model(model_config.key)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exploratory KMeans claim clusters from local ChromaDB.")
    parser.add_argument("--clusters", type=int, default=12, help="Requested number of KMeans clusters.")
    parser.add_argument("--model-key", default=None, help="Local embedding model folder name under models/embeddings.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
