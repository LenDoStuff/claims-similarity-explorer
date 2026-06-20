from __future__ import annotations

import json

import pytest

from src.cluster_review_store import load_reviews, review_columns, save_cluster_review


def test_load_reviews_returns_empty_frame_for_missing_file(tmp_path) -> None:
    reviews = load_reviews(tmp_path / "reviews.json")

    assert reviews.empty
    assert reviews.columns.tolist() == review_columns()


def test_save_cluster_review_creates_and_updates_one_review(tmp_path) -> None:
    path = tmp_path / "artifacts" / "cluster_reviews.json"

    save_cluster_review(
        path,
        model_key="multilingual-e5-small",
        cluster_version="v1",
        cluster_id=7,
        manual_label=" Water leakage ",
        review_status="accepted",
        notes=" Looks useful ",
    )
    save_cluster_review(
        path,
        model_key="multilingual-e5-small",
        cluster_version="v1",
        cluster_id=7,
        manual_label="Pipe leakage",
        review_status="needs_split",
        notes="Mixed with sprinkler losses",
    )

    reviews = load_reviews(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(reviews) == 1
    assert len(payload["reviews"]) == 1
    assert reviews.iloc[0]["manual_label"] == "Pipe leakage"
    assert reviews.iloc[0]["review_status"] == "needs_split"
    assert reviews.iloc[0]["notes"] == "Mixed with sprinkler losses"
    assert reviews.iloc[0]["updated_at_utc"]


def test_save_cluster_review_rejects_unknown_status(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown review status"):
        save_cluster_review(
            tmp_path / "reviews.json",
            model_key="multilingual-e5-small",
            cluster_version="v1",
            cluster_id=1,
            manual_label="",
            review_status="done",
            notes="",
        )
