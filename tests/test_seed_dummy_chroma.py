from __future__ import annotations

import numpy as np

from scripts.seed_dummy_chroma import build_dummy_claims, build_dummy_cluster_map


def test_dummy_claims_cover_expected_demo_shape() -> None:
    frame = build_dummy_claims()

    assert len(frame) == 32
    assert frame["claim_id"].is_unique
    assert frame["claim_description"].str.len().min() > 20
    assert {"DE", "AT", "NL"}.issubset(set(frame["country"]))
    assert {
        "Water Damage",
        "Fire",
        "Machinery Breakdown",
        "Transport Damage",
        "Public Liability",
        "Weather Damage",
        "Construction Defect",
        "Bodily Injury",
    }.issubset(set(frame["claim_type"]))


def test_dummy_cluster_map_uses_fast_projection() -> None:
    cluster_map = build_dummy_cluster_map(
        ["1", "2", "3"],
        [
            "Claim description: Water leakage damaged stock.",
            "Claim description: Pipe leakage damaged goods.",
            "Claim description: Fire damaged machinery.",
        ],
        [{"claim_id": "1"}, {"claim_id": "2"}, {"claim_id": "3"}],
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        np.asarray([0, 0, 1]),
    )

    assert cluster_map["projection"]["method"] == "SVD"
    assert len(cluster_map["points"]) == 3
    assert cluster_map["points"][0]["description"] == "Water leakage damaged stock."
