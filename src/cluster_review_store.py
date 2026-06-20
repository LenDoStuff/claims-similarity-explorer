from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


REVIEW_STATUSES = ["unreviewed", "accepted", "needs_split", "needs_merge", "too_mixed", "not_useful"]


def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=review_columns())
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("reviews", [])
    if not rows:
        return pd.DataFrame(columns=review_columns())
    return pd.DataFrame(rows)


def save_cluster_review(
    path: Path,
    *,
    model_key: str,
    cluster_version: str,
    cluster_id: int,
    manual_label: str,
    review_status: str,
    notes: str,
) -> None:
    if review_status not in REVIEW_STATUSES:
        raise ValueError(f"Unknown review status: {review_status}")

    reviews = load_reviews(path)
    row = {
        "model_key": model_key,
        "cluster_version": cluster_version,
        "cluster_id": int(cluster_id),
        "manual_label": manual_label.strip(),
        "review_status": review_status,
        "notes": notes.strip(),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }

    if reviews.empty:
        reviews = pd.DataFrame([row])
    else:
        mask = (
            (reviews["model_key"] == model_key)
            & (reviews["cluster_version"] == cluster_version)
            & (reviews["cluster_id"].astype(int) == int(cluster_id))
        )
        reviews = reviews[~mask]
        reviews = pd.concat([reviews, pd.DataFrame([row])], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[dict[str, Any]]] = {
        "reviews": reviews[review_columns()].to_dict(orient="records")
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def review_columns() -> list[str]:
    return [
        "model_key",
        "cluster_version",
        "cluster_id",
        "manual_label",
        "review_status",
        "notes",
        "updated_at_utc",
    ]
