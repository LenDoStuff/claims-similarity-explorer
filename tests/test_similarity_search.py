from __future__ import annotations

import pandas as pd
import numpy as np

from src.similarity_search import (
    SearchFilters,
    apply_pandas_filters,
    apply_rerank_scores,
    bm25_scores,
    build_chroma_where,
    merge_search_candidates,
    query_bm25_candidates,
    results_to_frame,
    tokenize_for_bm25,
)


def test_build_chroma_where_uses_equality_filters() -> None:
    filters = SearchFilters(equality={"country": "DE", "claim_status": "Open"})

    where = build_chroma_where(filters)

    assert where == {"$and": [{"country": {"$eq": "DE"}}, {"claim_status": {"$eq": "Open"}}]}


def test_build_chroma_where_returns_none_without_filters() -> None:
    assert build_chroma_where(SearchFilters()) is None


def test_apply_pandas_filters_applies_ranges() -> None:
    frame = pd.DataFrame(
        [
            {"claim_id": "1", "loss_year": 2023, "reserve_amount": 10.0, "paid_amount": 1.0},
            {"claim_id": "2", "loss_year": 2024, "reserve_amount": 50.0, "paid_amount": 5.0},
            {"claim_id": "3", "loss_year": 2025, "reserve_amount": 100.0, "paid_amount": 10.0},
        ]
    )
    filters = SearchFilters(
        loss_year_range=(2024, 2025),
        reserve_amount_range=(20.0, 90.0),
        paid_amount_range=(None, 8.0),
    )

    filtered = apply_pandas_filters(frame, filters)

    assert filtered["claim_id"].tolist() == ["2"]


def test_results_to_frame_adds_similarity() -> None:
    results = {
        "ids": [["1"]],
        "documents": [["Claim description: fire"]],
        "metadatas": [[{"claim_id": "1", "country": "DE"}]],
        "distances": [[0.25]],
    }

    frame = results_to_frame(results)

    assert frame.iloc[0]["claim_id"] == "1"
    assert frame.iloc[0]["similarity"] == 0.75


def test_tokenize_for_bm25_handles_german_terms() -> None:
    assert tokenize_for_bm25("Rohrbruch verursachte Wasserschäden.") == [
        "rohrbruch",
        "verursachte",
        "wasserschäden",
    ]


def test_bm25_scores_rank_matching_document_higher() -> None:
    scores = bm25_scores("water leakage", ["water leakage in warehouse", "fire damage"])

    assert scores[0] > scores[1]


def test_query_bm25_candidates_filters_and_sorts() -> None:
    frame = pd.DataFrame(
        [
            {"claim_id": "1", "document": "water leakage warehouse", "country": "DE"},
            {"claim_id": "2", "document": "fire damage", "country": "DE"},
            {"claim_id": "3", "document": "water leakage apartment", "country": "AT"},
        ]
    )

    result = query_bm25_candidates(
        frame,
        "water leakage",
        filters=SearchFilters(equality={"country": "DE"}),
        candidate_pool=10,
    )

    assert result["claim_id"].tolist() == ["1"]
    assert result.iloc[0]["bm25_score"] == 1.0


def test_merge_search_candidates_blends_scores() -> None:
    semantic = pd.DataFrame(
        [{"claim_id": "1", "document": "water", "semantic_score": 0.8, "distance": 0.2}]
    )
    bm25 = pd.DataFrame(
        [
            {"claim_id": "1", "document": "water", "bm25_score": 0.4},
            {"claim_id": "2", "document": "leakage", "bm25_score": 1.0},
        ]
    )

    result = merge_search_candidates(semantic, bm25, retrieval_mode="Hybrid", semantic_weight=0.5)

    assert result.iloc[0]["claim_id"] == "1"
    assert round(result.iloc[0]["final_score"], 4) == 0.6


def test_apply_rerank_scores_replaces_final_score() -> None:
    frame = pd.DataFrame(
        [
            {"claim_id": "1", "document": "water", "final_score": 0.9},
            {"claim_id": "2", "document": "fire", "final_score": 0.1},
        ]
    )

    result = apply_rerank_scores(frame, np.asarray([0.1, 0.9], dtype=np.float32))

    assert result["claim_id"].tolist() == ["2", "1"]
    assert result.iloc[0]["rerank_score"] == 1.0
