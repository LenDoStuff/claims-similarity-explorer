from __future__ import annotations

import pandas as pd

from src.config import ColumnConfig
from src.text_preprocessing import (
    build_metadata,
    clean_text,
    prepare_claim_records,
    prepare_embedding_text,
    source_text_hash,
)


def test_clean_text_collapses_whitespace_and_handles_missing() -> None:
    assert clean_text("  Wasser\n\nschaden\t in   Halle  ") == "Wasser schaden in Halle"
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_prepare_embedding_text_uses_event_fields_only() -> None:
    columns = ColumnConfig()
    row = {
        "claim_id": "CLM-1",
        "claim_description": "Burst pipe damaged stock.",
        "line_of_business": "Property",
        "claim_type": "Water Damage",
        "cause_of_loss": "Pipe Leakage",
        "damaged_object": "Warehouse",
        "country": "DE",
        "claim_status": "Open",
        "reserve_amount": 50000,
    }

    text = prepare_embedding_text(row, columns)

    assert "Claim description: Burst pipe damaged stock." in text
    assert "Line of business: Property" in text
    assert "Claim type: Water Damage" in text
    assert "Cause of loss: Pipe Leakage" in text
    assert "Damaged object: Warehouse" in text
    assert "Country: DE" in text
    assert "CLM-1" not in text
    assert "50000" not in text
    assert "Open" not in text


def test_source_hash_is_stable() -> None:
    assert source_text_hash("same text") == source_text_hash("same text")
    assert source_text_hash("same text") != source_text_hash("different text")


def test_build_metadata_converts_scalars_and_derives_loss_year() -> None:
    columns = ColumnConfig()
    row = {
        "claim_id": "CLM-123",
        "claim_description": "Fire in production site.",
        "line_of_business": "Property",
        "claim_type": "Fire",
        "cause_of_loss": "Explosion",
        "damaged_object": "Machine",
        "country": "DE",
        "claim_status": "Open",
        "loss_date": "2024-03-10",
        "reserve_amount": 125000.0,
        "paid_amount": 30000.0,
        "currency": "EUR",
        "policy_type": "Industrial",
    }

    metadata = build_metadata(
        row,
        columns,
        cleaned_description=row["claim_description"],
        embedding_text="Fire in production site.",
        model_name="model",
        model_path="models/model",
        embedding_version="v1",
    )

    assert metadata["claim_id"] == "CLM-123"
    assert metadata["loss_year"] == 2024
    assert metadata["reserve_amount"] == 125000.0
    assert metadata["embedding_model"] == "model"
    assert "source_text_hash" in metadata


def test_prepare_claim_records_skips_blank_descriptions_and_counts_duplicates() -> None:
    frame = pd.DataFrame(
        [
            {"claim_id": "1", "claim_description": "Pipe leak in building."},
            {"claim_id": "2", "claim_description": "Pipe leak in building."},
            {"claim_id": "3", "claim_description": ""},
            {"claim_id": "", "claim_description": "Fire."},
        ]
    )

    records, diagnostics = prepare_claim_records(
        frame,
        ColumnConfig(),
        model_name="model",
        model_path="models/model",
        embedding_version="v1",
    )

    assert len(records) == 2
    assert diagnostics["missing_descriptions"] == 1
    assert diagnostics["missing_claim_ids"] == 1
    assert diagnostics["duplicate_descriptions"] == 1


def test_prepare_claim_records_supports_required_columns_only() -> None:
    columns = ColumnConfig(
        line_of_business="",
        claim_type="",
        cause_of_loss="",
        damaged_object="",
        country="",
        claim_status="",
        loss_date="",
        reserve_amount="",
        paid_amount="",
        currency="",
        policy_type="",
    )
    frame = pd.DataFrame([{"claim_id": "1", "claim_description": "Pipe leak in building."}])

    records, diagnostics = prepare_claim_records(
        frame,
        columns,
        model_name="model",
        model_path="models/model",
        embedding_version="v1",
    )

    assert diagnostics["indexed_rows"] == 1
    assert records[0]["document"] == "Claim description: Pipe leak in building."
    assert records[0]["metadata"]["claim_id"] == "1"
    assert "line_of_business" not in records[0]["metadata"]
