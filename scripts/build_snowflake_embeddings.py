from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    args = parse_args()

    from src.config import AppConfig
    from src.indexing import initialize_embeddings
    from src.snowflake_io import create_snowflake_session

    config = AppConfig.from_app_config()
    session = create_snowflake_session()
    try:
        result = initialize_embeddings(session, config, args.models, limit=args.limit)
    finally:
        session.close()
    print(f"Embedding table: {result.table_name}")
    print(f"Record count: {result.row_count}")
    print(f"Models: {', '.join(result.models)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Snowflake-hosted claim embeddings with Snowpark.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["voyage-multilingual-2"],
        help="Snowflake AI_EMBED text model names.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit overriding app_config.toml.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
