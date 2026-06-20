from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.chroma_store import collection_to_frame


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def collection_diagnostics(collection: Any) -> dict[str, Any]:
    frame = collection_to_frame(collection)
    if frame.empty:
        return {
            "record_count": 0,
            "metadata_columns": [],
            "average_description_length": 0,
            "cluster_count": 0,
        }
    cluster_count = int(frame["cluster_id"].nunique()) if "cluster_id" in frame else 0
    avg_description_length = float(pd.to_numeric(frame.get("description_length"), errors="coerce").mean() or 0)
    return {
        "record_count": int(len(frame)),
        "metadata_columns": sorted(
            column for column in frame.columns if column not in {"id", "document"}
        ),
        "average_description_length": round(avg_description_length, 1),
        "cluster_count": cluster_count,
    }

