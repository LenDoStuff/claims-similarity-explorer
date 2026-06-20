from __future__ import annotations

import pandas as pd

from src.cluster_profiles import (
    ClusterFilters,
    apply_cluster_filters,
    build_cluster_overview,
    cluster_financials,
    cluster_version,
    common_metadata,
    selected_cluster_report,
    weak_assignment_candidates,
)


def test_cluster_overview_uses_cluster_artifact_and_review_data() -> None:
    claims = sample_claims()
    clusters = sample_clusters()
    version = cluster_version(clusters, "multilingual-e5-small")
    reviews = pd.DataFrame(
        [
            {
                "model_key": "multilingual-e5-small",
                "cluster_version": version,
                "cluster_id": 0,
                "manual_label": "Manual water label",
                "review_status": "accepted",
                "notes": "Useful cluster",
            }
        ]
    )

    overview = build_cluster_overview(
        claims,
        clusters,
        reviews,
        model_key="multilingual-e5-small",
        version=version,
    )
    row = overview[overview["cluster_id"] == 0].iloc[0]

    assert row["display_label"] == "Manual water label"
    assert row["review_status"] == "accepted"
    assert row["cluster_size"] == 2
    assert row["total_reserve"] == 400.0
    assert row["top_keywords"] == "water, pipe, leakage"


def test_cluster_filters_recalculate_visible_overview_values() -> None:
    claims = sample_claims()
    clusters = sample_clusters()
    version = cluster_version(clusters, "multilingual-e5-small")
    reviews = pd.DataFrame(
        [
            {
                "model_key": "multilingual-e5-small",
                "cluster_version": version,
                "cluster_id": 0,
                "manual_label": "",
                "review_status": "accepted",
                "notes": "",
            }
        ]
    )
    overview = build_cluster_overview(
        claims,
        clusters,
        reviews,
        model_key="multilingual-e5-small",
        version=version,
    )

    filtered_claims, filtered_overview = apply_cluster_filters(
        claims,
        overview,
        ClusterFilters(
            line_of_business=["Property"],
            loss_year_range=(2024, 2024),
            reserve_amount_range=(0, 200),
            description_search="water",
            cluster_search="pipe",
            review_status=["accepted"],
        ),
    )

    assert filtered_claims["claim_id"].tolist() == ["C1"]
    assert filtered_overview["cluster_id"].tolist() == [0]
    assert filtered_overview.iloc[0]["cluster_size"] == 1
    assert filtered_overview.iloc[0]["total_reserve"] == 100.0


def test_cluster_filters_can_filter_by_review_status() -> None:
    claims = sample_claims()
    clusters = sample_clusters()
    version = cluster_version(clusters, "multilingual-e5-small")
    reviews = pd.DataFrame(
        [
            {
                "model_key": "multilingual-e5-small",
                "cluster_version": version,
                "cluster_id": 0,
                "manual_label": "",
                "review_status": "accepted",
                "notes": "",
            }
        ]
    )
    overview = build_cluster_overview(
        claims,
        clusters,
        reviews,
        model_key="multilingual-e5-small",
        version=version,
    )

    filtered_claims, filtered_overview = apply_cluster_filters(
        claims,
        overview,
        ClusterFilters(review_status=["not_useful"]),
    )

    assert filtered_claims.empty
    assert filtered_overview.empty


def test_profile_helpers_shape_financial_metadata_outliers_and_report() -> None:
    claims = sample_claims()
    clusters = sample_clusters()
    overview = build_cluster_overview(
        claims,
        clusters,
        pd.DataFrame(),
        model_key="multilingual-e5-small",
        version=cluster_version(clusters, "multilingual-e5-small"),
    )

    financials = cluster_financials(claims[claims["cluster_id"] == 0])
    metadata = common_metadata(claims)
    outliers = weak_assignment_candidates(claims, overview)
    report = selected_cluster_report(
        profile=overview.iloc[0].to_dict(),
        keywords=["water", "pipe"],
        representatives=clusters["clusters"][0]["representative_claims"],
        financials=financials,
    )

    assert financials["total_paid"] == 40.0
    assert metadata["country"][0] == {"value": "DE", "count": 2}
    assert "High severity claim" in outliers["outlier_reason"].tolist()
    assert "# Cluster" in report
    assert "Representative Claims" in report


def sample_claims() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "C1",
                "document": "Burst pipe caused water damage in warehouse.",
                "cluster_id": 0,
                "line_of_business": "Property",
                "claim_type": "Water Damage",
                "cause_of_loss": "Leakage",
                "country": "DE",
                "claim_status": "Open",
                "loss_year": 2024,
                "reserve_amount": 100.0,
                "paid_amount": 10.0,
                "description_length": 43,
            },
            {
                "claim_id": "C2",
                "document": "Sprinkler leakage damaged goods.",
                "cluster_id": 0,
                "line_of_business": "Property",
                "claim_type": "Water Damage",
                "cause_of_loss": "Leakage",
                "country": "AT",
                "claim_status": "Closed",
                "loss_year": 2024,
                "reserve_amount": 300.0,
                "paid_amount": 30.0,
                "description_length": 31,
            },
            {
                "claim_id": "C3",
                "document": "Fire explosion destroyed stored inventory.",
                "cluster_id": 1,
                "line_of_business": "Property",
                "claim_type": "Fire",
                "cause_of_loss": "Explosion",
                "country": "DE",
                "claim_status": "Closed",
                "loss_year": 2023,
                "reserve_amount": 1000.0,
                "paid_amount": 900.0,
                "description_length": 42,
            },
            {
                "claim_id": "C4",
                "document": "Customer injury.",
                "cluster_id": 1,
                "line_of_business": "Liability",
                "claim_type": "Bodily Injury",
                "cause_of_loss": "Slip",
                "country": "US",
                "claim_status": "Open",
                "loss_year": 2024,
                "reserve_amount": 50.0,
                "paid_amount": 20.0,
                "description_length": 16,
            },
        ]
    )


def sample_clusters() -> dict:
    return {
        "algorithm": "kmeans",
        "n_clusters": 2,
        "clustered_at_utc": "2026-06-19T00:00:00Z",
        "clusters": [
            {
                "cluster_id": 0,
                "label": "Water damage / leakage",
                "frequent_terms": ["water", "pipe", "leakage"],
                "representative_claims": [
                    {"claim_id": "C1", "description": "Burst pipe caused water damage in warehouse."}
                ],
            },
            {
                "cluster_id": 1,
                "label": "Fire and injury",
                "frequent_terms": ["fire", "injury"],
                "representative_claims": [
                    {"claim_id": "C3", "description": "Fire explosion destroyed stored inventory."}
                ],
            },
        ],
    }
