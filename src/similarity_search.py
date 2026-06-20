from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.text_preprocessing import clean_text


EQUALITY_FILTER_FIELDS = [
    "line_of_business",
    "claim_type",
    "cause_of_loss",
    "country",
    "claim_status",
    "policy_type",
    "currency",
    "cluster_id",
]
TOKEN_RE = re.compile(r"[0-9A-Za-zÄÖÜäöüß]+")


@dataclass(frozen=True)
class SearchFilters:
    equality: dict[str, Any] = field(default_factory=dict)
    loss_year_range: tuple[int | None, int | None] = (None, None)
    reserve_amount_range: tuple[float | None, float | None] = (None, None)
    paid_amount_range: tuple[float | None, float | None] = (None, None)


def build_chroma_where(filters: SearchFilters) -> dict[str, Any] | None:
    clauses = []
    for field in EQUALITY_FILTER_FIELDS:
        value = filters.equality.get(field)
        if value not in (None, "", "All"):
            clauses.append({field: {"$eq": value}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def apply_pandas_filters(frame: pd.DataFrame, filters: SearchFilters) -> pd.DataFrame:
    result = frame.copy()
    result = _apply_range(result, "loss_year", filters.loss_year_range)
    result = _apply_range(result, "reserve_amount", filters.reserve_amount_range)
    result = _apply_range(result, "paid_amount", filters.paid_amount_range)
    return result


def apply_all_filters(frame: pd.DataFrame, filters: SearchFilters) -> pd.DataFrame:
    result = frame.copy()
    for field in EQUALITY_FILTER_FIELDS:
        value = filters.equality.get(field)
        if value not in (None, "", "All") and field in result:
            result = result[result[field] == value]
    return apply_pandas_filters(result, filters)


def _apply_range(frame: pd.DataFrame, column: str, bounds: tuple[int | float | None, int | float | None]) -> pd.DataFrame:
    lower, upper = bounds
    if column not in frame or (lower is None and upper is None):
        return frame
    series = pd.to_numeric(frame[column], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if lower is not None:
        mask &= series >= lower
    if upper is not None:
        mask &= series <= upper
    return frame[mask]


def tokenize_for_bm25(text: Any) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(clean_text(text))]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    minimum = float(scores.min())
    maximum = float(scores.max())
    if math.isclose(maximum, minimum):
        if maximum > 0:
            return np.ones_like(scores, dtype=np.float32)
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - minimum) / (maximum - minimum)


def bm25_scores(query: str, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    query_terms = tokenize_for_bm25(query)
    if not query_terms or not documents:
        return np.zeros(len(documents), dtype=np.float32)

    tokenized_documents = [tokenize_for_bm25(document) for document in documents]
    document_lengths = np.asarray([len(tokens) for tokens in tokenized_documents], dtype=np.float32)
    average_length = float(document_lengths.mean()) if len(document_lengths) else 0.0
    if average_length == 0:
        return np.zeros(len(documents), dtype=np.float32)

    document_frequencies: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequencies.update(set(tokens))

    total_documents = len(tokenized_documents)
    scores = np.zeros(total_documents, dtype=np.float32)
    for term in query_terms:
        frequency = document_frequencies.get(term, 0)
        if frequency == 0:
            continue
        idf = math.log(1.0 + (total_documents - frequency + 0.5) / (frequency + 0.5))
        for index, tokens in enumerate(tokenized_documents):
            term_frequency = tokens.count(term)
            if term_frequency == 0:
                continue
            denominator = term_frequency + k1 * (1.0 - b + b * document_lengths[index] / average_length)
            scores[index] += idf * term_frequency * (k1 + 1.0) / denominator
    return scores


def query_bm25_candidates(
    claims_frame: pd.DataFrame,
    query_text: str,
    *,
    filters: SearchFilters,
    candidate_pool: int,
    exclude_claim_id: str | None = None,
) -> pd.DataFrame:
    frame = apply_all_filters(claims_frame, filters)
    if exclude_claim_id and not frame.empty and "claim_id" in frame:
        frame = frame[frame["claim_id"] != exclude_claim_id]
    if frame.empty:
        return frame

    documents = frame["document"].fillna("").astype(str).tolist()
    raw_scores = bm25_scores(query_text, documents)
    result = frame.copy()
    result["bm25_score"] = normalize_scores(raw_scores)
    result = result[result["bm25_score"] > 0]
    if result.empty:
        return result
    return (
        result.sort_values(["bm25_score", "claim_id"], ascending=[False, True])
        .head(candidate_pool)
        .reset_index(drop=True)
    )


def query_similar_claims(
    collection: Any,
    query_embedding: np.ndarray,
    *,
    filters: SearchFilters,
    top_n: int,
    candidate_multiplier: int = 4,
    exclude_claim_id: str | None = None,
) -> pd.DataFrame:
    if query_embedding.ndim == 2:
        query_embedding = query_embedding[0]
    n_results = max(top_n * candidate_multiplier, top_n, 50)
    where = build_chroma_where(filters)
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)
    frame = results_to_frame(results)
    if exclude_claim_id and not frame.empty:
        frame = frame[frame["claim_id"] != exclude_claim_id]
    frame = apply_pandas_filters(frame, filters)
    if not frame.empty:
        frame = frame.sort_values(["distance", "claim_id"], ascending=[True, True]).head(top_n)
    return frame.reset_index(drop=True)


def results_to_frame(results: dict[str, Any]) -> pd.DataFrame:
    ids = _first(results.get("ids"))
    documents = _first(results.get("documents"))
    metadatas = _first(results.get("metadatas"))
    distances = _first(results.get("distances"))
    rows: list[dict[str, Any]] = []
    for claim_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        row = dict(metadata or {})
        row["id"] = claim_id
        row["claim_id"] = clean_text(row.get("claim_id") or claim_id)
        row["document"] = document
        row["distance"] = float(distance)
        row["similarity"] = max(0.0, 1.0 - float(distance))
        row["semantic_score"] = row["similarity"]
        rows.append(row)
    return pd.DataFrame(rows)


def merge_search_candidates(
    semantic_frame: pd.DataFrame,
    bm25_frame: pd.DataFrame,
    *,
    retrieval_mode: str,
    semantic_weight: float,
) -> pd.DataFrame:
    frames = []
    if retrieval_mode in {"Semantic", "Hybrid"} and not semantic_frame.empty:
        frames.append(semantic_frame.copy())
    if retrieval_mode in {"BM25", "Hybrid"} and not bm25_frame.empty:
        frames.append(bm25_frame.copy())
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "semantic_score" not in combined:
        combined["semantic_score"] = 0.0
    else:
        combined["semantic_score"] = pd.to_numeric(combined["semantic_score"], errors="coerce").fillna(0.0)
    if "bm25_score" not in combined:
        combined["bm25_score"] = 0.0
    else:
        combined["bm25_score"] = pd.to_numeric(combined["bm25_score"], errors="coerce").fillna(0.0)
    rows = []
    for claim_id, group in combined.groupby("claim_id", sort=False):
        row = group.iloc[0].copy()
        row["claim_id"] = claim_id
        row["semantic_score"] = float(group["semantic_score"].max())
        row["bm25_score"] = float(group["bm25_score"].max())
        if "distance" in group:
            distance = pd.to_numeric(group["distance"], errors="coerce").dropna()
            if not distance.empty:
                row["distance"] = float(distance.min())
        rows.append(row.to_dict())

    result = pd.DataFrame(rows)
    if retrieval_mode == "Semantic":
        result["final_score"] = result["semantic_score"]
    elif retrieval_mode == "BM25":
        result["final_score"] = result["bm25_score"]
    else:
        result["final_score"] = (
            semantic_weight * result["semantic_score"] + (1.0 - semantic_weight) * result["bm25_score"]
        )
    return result.sort_values(["final_score", "claim_id"], ascending=[False, True]).reset_index(drop=True)


def apply_rerank_scores(frame: pd.DataFrame, rerank_scores: np.ndarray) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy().head(len(rerank_scores))
    result["rerank_score"] = normalize_scores(rerank_scores)
    result["final_score"] = result["rerank_score"]
    return result.sort_values(["final_score", "claim_id"], ascending=[False, True]).reset_index(drop=True)


def _first(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return value


def display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "final_score",
        "semantic_score",
        "bm25_score",
        "rerank_score",
        "similarity",
        "distance",
        "claim_id",
        "line_of_business",
        "claim_type",
        "cause_of_loss",
        "country",
        "claim_status",
        "loss_year",
        "reserve_amount",
        "paid_amount",
        "currency",
    ]
    available = [column for column in columns if column in frame.columns]
    result = frame[available].copy() if available else frame.copy()
    if "similarity" in result:
        result["similarity"] = result["similarity"].map(lambda value: round(float(value), 4))
    if "distance" in result:
        result["distance"] = result["distance"].map(lambda value: round(float(value), 4))
    for column in ["final_score", "semantic_score", "bm25_score", "rerank_score"]:
        if column in result:
            result[column] = result[column].map(lambda value: round(float(value), 4))
    return result
