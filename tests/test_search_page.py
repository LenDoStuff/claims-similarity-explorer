from __future__ import annotations

import pandas as pd

from app.search_page import result_card_html, score_details_html


def test_result_card_omits_empty_metadata_sections() -> None:
    row = pd.Series(
        {
            "claim_id": "1",
            "document": "Claim description: Pipe leak in building.",
            "final_score": 0.8,
        }
    )

    html = result_card_html(1, row)

    assert "claim-chip-row" not in html
    assert "claim-tile-row" not in html
    assert "Pipe leak in building." in html


def test_result_card_score_details_are_grid_without_duplicate_similarity() -> None:
    row = pd.Series(
        {
            "claim_id": "1",
            "document": "Claim description: Pipe leak in building.",
            "final_score": 0.915,
            "semantic_score": 0.879,
            "similarity": 0.879,
            "bm25_score": 1.0,
            "distance": 0.121,
        }
    )

    html = result_card_html(1, row)
    score_html = score_details_html(row)

    assert "claim-score-detail-grid" in html
    assert score_html.startswith('<div class="claim-score-details">')
    assert "\n" not in score_html
    assert "Semantic similarity" in html
    assert "BM25 normalized" in html
    assert "Chroma distance" in html
    assert "<span>|</span>" not in html
    assert "similarity 0.879" not in html
