from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    args = parse_args()

    from src.config import AppConfig, get_embedding_model
    from src.indexing import build_index_from_snowflake

    config = AppConfig.from_app_config()
    model_config = get_embedding_model(args.model_key or config.model_key)
    result = build_index_from_snowflake(
        config,
        model_config,
        limit=args.limit,
        batch_size=args.batch_size,
        chroma_batch_size=args.chroma_batch_size,
    )
    print(f"Index status: {result.status}")
    print(f"Active collection: {result.collection_name}")
    print(f"Record count: {result.manifest.get('record_count')}")
    print(f"Wrote manifest to {result.manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local ChromaDB index from Snowflake claims.")
    parser.add_argument("--limit", type=int, default=None, help="Optional random sample row limit; overrides app_config.toml.")
    parser.add_argument("--batch-size", type=int, default=64, help="SentenceTransformer embedding batch size.")
    parser.add_argument("--chroma-batch-size", type=int, default=500, help="Chroma upsert batch size.")
    parser.add_argument(
        "--model-key",
        default=None,
        help="Local embedding model folder name under models/embeddings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
