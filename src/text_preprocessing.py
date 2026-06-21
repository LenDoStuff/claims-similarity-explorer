from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from src.config import ColumnConfig


WHITESPACE_RE = re.compile(r"\s+")
SHORT_DESCRIPTION_CHARS = 20


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\x00", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def is_short_description(description: str, min_chars: int = SHORT_DESCRIPTION_CHARS) -> bool:
    return bool(description) and len(description) < min_chars


def prepare_embedding_text(row: Mapping[str, Any], columns: ColumnConfig) -> str:
    parts = [
        ("Claim description", row.get(columns.description)),
        ("Line of business", row.get(columns.line_of_business)),
        ("Claim type", row.get(columns.claim_type)),
        ("Cause of loss", row.get(columns.cause_of_loss)),
        ("Damaged object", row.get(columns.damaged_object)),
        ("Country", row.get(columns.country)),
    ]
    lines = [f"{label}: {cleaned}" for label, value in parts if (cleaned := clean_text(value))]
    return "\n".join(lines)


def source_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if pd.notna(value) else None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return clean_text(value)


def derive_loss_year(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.year)


def build_metadata(
    row: Mapping[str, Any],
    columns: ColumnConfig,
    *,
    cleaned_description: str,
    embedding_text: str,
    model_name: str,
    model_path: str,
    embedding_version: str,
) -> dict[str, str | int | float | bool]:
    claim_id = clean_text(row.get(columns.claim_id))
    metadata: dict[str, str | int | float | bool | None] = {
        "claim_id": claim_id,
        "line_of_business": to_json_scalar(row.get(columns.line_of_business)),
        "claim_type": to_json_scalar(row.get(columns.claim_type)),
        "cause_of_loss": to_json_scalar(row.get(columns.cause_of_loss)),
        "damaged_object": to_json_scalar(row.get(columns.damaged_object)),
        "country": to_json_scalar(row.get(columns.country)),
        "claim_status": to_json_scalar(row.get(columns.claim_status)),
        "loss_date": to_json_scalar(row.get(columns.loss_date)),
        "loss_year": derive_loss_year(row.get(columns.loss_date)),
        "reserve_amount": to_json_scalar(row.get(columns.reserve_amount)),
        "paid_amount": to_json_scalar(row.get(columns.paid_amount)),
        "currency": to_json_scalar(row.get(columns.currency)),
        "policy_type": to_json_scalar(row.get(columns.policy_type)),
        "description_length": len(cleaned_description),
        "embedding_model": model_name,
        "embedding_model_path": model_path,
        "embedding_version": embedding_version,
        "source_text_hash": source_text_hash(embedding_text),
    }
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def prepare_claim_records(
    frame: pd.DataFrame,
    columns: ColumnConfig,
    *,
    model_name: str,
    model_path: str,
    embedding_version: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    columns.validate_required()
    records: list[dict[str, Any]] = []
    seen_descriptions: dict[str, int] = {}
    diagnostics = {
        "source_rows": int(len(frame)),
        "indexed_rows": 0,
        "missing_descriptions": 0,
        "short_descriptions": 0,
        "duplicate_descriptions": 0,
        "missing_claim_ids": 0,
    }

    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        claim_id = clean_text(row_dict.get(columns.claim_id))
        description = clean_text(row_dict.get(columns.description))
        if not claim_id:
            diagnostics["missing_claim_ids"] += 1
            continue
        if not description:
            diagnostics["missing_descriptions"] += 1
            continue
        if is_short_description(description):
            diagnostics["short_descriptions"] += 1

        normalized_description = description.casefold()
        seen_descriptions[normalized_description] = seen_descriptions.get(normalized_description, 0) + 1
        if seen_descriptions[normalized_description] > 1:
            diagnostics["duplicate_descriptions"] += 1

        embedding_text = prepare_embedding_text(row_dict, columns)
        metadata = build_metadata(
            row_dict,
            columns,
            cleaned_description=description,
            embedding_text=embedding_text,
            model_name=model_name,
            model_path=model_path,
            embedding_version=embedding_version,
        )
        records.append(
            {
                "id": claim_id,
                "document": embedding_text,
                "claim_description": description,
                "metadata": metadata,
            }
        )

    diagnostics["indexed_rows"] = len(records)
    return records, diagnostics
