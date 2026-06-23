from __future__ import annotations

import pandas as pd

from app.search_page import metric_score_bar, result_card_html, score_details_html
from src.similarity_search import get_similarity_metric


def test_result_card_omits_empty_metadata_sections() -> None:
    row = pd.Series(
        {
            "claim_id": "1",
            "document": "Claim description: Pipe leak in building.",
            "final_score": 0.8,
        }
    )

    html = result_card_html(1, row, get_similarity_metric("cosine"))

    assert "claim-chip-row" not in html
    assert "claim-tile-row" not in html
    assert "Pipe leak in building." in html


def test_result_card_score_details_show_snowflake_similarity() -> None:
    row = pd.Series(
        {
            "claim_id": "1",
            "document": "Claim description: Pipe leak in building.",
            "metric_value": 0.879,
        }
    )
    metric = get_similarity_metric("cosine")

    html = result_card_html(1, row, metric)
    score_html = score_details_html(row, metric)

    assert "claim-score-detail-grid" in html
    assert score_html.startswith('<div class="claim-score-details">')
    assert "\n" not in score_html
    assert "Cosine similarity" in html
    assert "Higher is better" in html
    assert "<span>|</span>" not in html
    assert "similarity 0.879" not in html


def test_unbounded_metric_omits_score_bar() -> None:
    metric = get_similarity_metric("l2")
    row = pd.Series(
        {
            "claim_id": "1",
            "document": "Claim description: Pipe leak in building.",
            "metric_value": 1.25,
        }
    )

    html = result_card_html(1, row, metric)

    assert "Euclidean distance" in html
    assert "Lower is better" in html
    assert "claim-score-track" not in html
    assert metric_score_bar(1.25, metric) == ""
