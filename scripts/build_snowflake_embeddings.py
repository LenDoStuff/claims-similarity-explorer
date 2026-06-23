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
    from snowflake.snowpark import Session

    from src.snowflake_io import initialize_embeddings

    config = AppConfig.from_app_config()
    session = Session.builder.create()
    try:
        result = initialize_embeddings(session, config, args.models)
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
        required=True,
        help="Snowflake AI_EMBED text model names.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
