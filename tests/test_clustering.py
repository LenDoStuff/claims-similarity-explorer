from __future__ import annotations

import numpy as np
import pytest

from src.clustering import (
    build_cluster_map,
    build_cluster_summary,
    claim_description_from_document,
    cluster_embeddings,
    frequent_terms,
    keyword_text_from_document,
)


def test_frequent_terms_filters_common_stop_words() -> None:
    terms = frequent_terms(["Water damage and leakage in warehouse", "Wasser leakage warehouse"])

    assert "leakage" in terms
    assert "warehouse" in terms
    assert "and" not in terms
    assert "damage" not in terms


def test_keyword_text_uses_event_fields_without_embedding_labels() -> None:
    text = keyword_text_from_document(labeled_document("Burst pipe damaged warehouse stock."))

    assert "Burst pipe damaged warehouse stock." in text
    assert "Water Damage" in text
    assert "Pipe Leakage" in text
    assert "Warehouse inventory" in text
    assert "Line of business" not in text
    assert "Country" not in text


def test_frequent_terms_penalizes_common_terms_across_corpus() -> None:
    water = keyword_text_from_document(labeled_document("Burst pipe caused water leakage in warehouse."))
    sprinkler = keyword_text_from_document(
        labeled_document("Sprinkler leakage damaged electronics.", cause="Sprinkler Leakage")
    )
    fire = keyword_text_from_document(
        labeled_document(
            "Electrical short circuit caused fire and smoke contamination.",
            claim_type="Fire",
            cause="Electrical Fault",
            damaged_object="Production hall",
        )
    )

    terms = frequent_terms([water, sprinkler], corpus_texts=[water, sprinkler, fire])

    assert "leakage" in terms[:3]
    assert "claim" not in terms
    assert "description" not in terms
    assert "damaged" not in terms


def test_claim_description_from_document_returns_raw_description() -> None:
    description = claim_description_from_document(labeled_document("Burst pipe damaged warehouse stock."))

    assert description == "Burst pipe damaged warehouse stock."


def test_cluster_embeddings_caps_cluster_count() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)

    labels, centers = cluster_embeddings(embeddings, n_clusters=5)

    assert len(labels) == 2
    assert centers.shape == (2, 2)


def test_build_cluster_map_returns_umap_points() -> None:
    ids = ["1", "2", "3", "4"]
    documents = [
        labeled_document("Water leakage damaged warehouse stock."),
        labeled_document("Pipe leakage damaged stored goods."),
        labeled_document("Fire damaged production hall.", claim_type="Fire", cause="Electrical Fault"),
        labeled_document("Explosion damaged machine.", claim_type="Fire", cause="Explosion"),
    ]
    metadatas = [{"claim_id": claim_id} for claim_id in ids]
    embeddings = np.asarray(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.9, 0.1]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1])

    cluster_map = build_cluster_map(ids, documents, metadatas, embeddings, labels)

    assert cluster_map["projection"]["method"] == "UMAP"
    assert len(cluster_map["points"]) == 4
    first = cluster_map["points"][0]
    assert {"claim_id", "cluster_id", "x", "y", "description"}.issubset(first)
    assert first["description"] == "Water leakage damaged warehouse stock."


def test_build_cluster_map_requires_enough_embeddings_for_umap() -> None:
    with pytest.raises(ValueError, match="At least 3 embeddings"):
        build_cluster_map(
            ["1", "2"],
            [labeled_document("Water leakage."), labeled_document("Fire loss.")],
            [{"claim_id": "1"}, {"claim_id": "2"}],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.asarray([0, 1]),
        )


def test_build_cluster_summary_returns_representatives_and_metadata() -> None:
    ids = ["1", "2", "3"]
    documents = [
        labeled_document("Water leakage damaged warehouse stock."),
        labeled_document("Pipe leakage damaged stored goods."),
        labeled_document(
            "Fire explosion damaged machine.",
            claim_type="Fire",
            cause="Explosion",
            damaged_object="Machine",
        ),
    ]
    metadatas = [
        {"claim_id": "1", "line_of_business": "Property", "country": "DE"},
        {"claim_id": "2", "line_of_business": "Property", "country": "DE"},
        {"claim_id": "3", "line_of_business": "Property", "country": "AT"},
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1])
    centers = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    summary = build_cluster_summary(ids, documents, metadatas, embeddings, labels, centers)

    assert len(summary["clusters"]) == 2
    first = summary["clusters"][0]
    assert first["size"] == 2
    assert first["representative_claims"]
    assert "Claim description" not in first["representative_claims"][0]["description"]
    assert "claim" not in first["frequent_terms"]
    assert "line" not in first["frequent_terms"]
    assert "leakage" in first["frequent_terms"]
    assert first["common_metadata"]["country"][0]["value"] == "DE"


def labeled_document(
    description: str,
    *,
    claim_type: str = "Water Damage",
    cause: str = "Pipe Leakage",
    damaged_object: str = "Warehouse inventory",
) -> str:
    return "\n".join(
        [
            f"Claim description: {description}",
            "Line of business: Property",
            f"Claim type: {claim_type}",
            f"Cause of loss: {cause}",
            f"Damaged object: {damaged_object}",
            "Country: DE",
        ]
    )
