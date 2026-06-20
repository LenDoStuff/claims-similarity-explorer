from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.cluster_review_store import REVIEW_STATUSES
from src.text_preprocessing import clean_text


@dataclass(frozen=True)
class ClusterFilters:
    line_of_business: list[str] = field(default_factory=list)
    country: list[str] = field(default_factory=list)
    claim_type: list[str] = field(default_factory=list)
    cause_of_loss: list[str] = field(default_factory=list)
    claim_status: list[str] = field(default_factory=list)
    loss_year_range: tuple[int | None, int | None] = (None, None)
    reserve_amount_range: tuple[float | None, float | None] = (None, None)
    paid_amount_range: tuple[float | None, float | None] = (None, None)
    min_cluster_size: int = 1
    description_search: str = ""
    cluster_search: str = ""
    review_status: list[str] = field(default_factory=list)


def cluster_version(clusters: dict[str, Any], model_key: str) -> str:
    algorithm = clean_text(clusters.get("algorithm")) or "unknown"
    n_clusters = clean_text(clusters.get("n_clusters")) or "unknown"
    created_at = clean_text(clusters.get("clustered_at_utc")) or "unknown"
    return f"{model_key}|{algorithm}|{n_clusters}|{created_at}"


def build_cluster_overview(
    claims: pd.DataFrame,
    clusters: dict[str, Any],
    reviews: pd.DataFrame,
    *,
    model_key: str,
    version: str,
) -> pd.DataFrame:
    if claims.empty or "cluster_id" not in claims:
        return pd.DataFrame()

    cluster_info = {int(cluster["cluster_id"]): cluster for cluster in clusters.get("clusters", [])}
    review_map = review_lookup(reviews, model_key=model_key, version=version)
    rows = []
    for cluster_id, cluster_claims in claims.groupby("cluster_id", dropna=False):
        cluster_id = int(cluster_id)
        info = cluster_info.get(cluster_id, {})
        review = review_map.get(cluster_id, {})
        reserve = pd.to_numeric(cluster_claims.get("reserve_amount"), errors="coerce").fillna(0)
        paid = pd.to_numeric(cluster_claims.get("paid_amount"), errors="coerce").fillna(0)
        size = int(len(cluster_claims))
        outlier_count = 1 if size == 1 else 0
        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": clean_text(info.get("label")) or f"Cluster {cluster_id}",
                "manual_label": clean_text(review.get("manual_label")),
                "cluster_size": size,
                "top_keywords": ", ".join(info.get("frequent_terms", [])[:6]),
                "dominant_line_of_business": dominant_value(cluster_claims, "line_of_business"),
                "dominant_cause_of_loss": dominant_value(cluster_claims, "cause_of_loss"),
                "dominant_country": dominant_value(cluster_claims, "country"),
                "total_reserve": float(reserve.sum()),
                "total_paid": float(paid.sum()),
                "avg_reserve": float(reserve.mean()) if size else 0.0,
                "avg_paid": float(paid.mean()) if size else 0.0,
                "outlier_count": outlier_count,
                "outlier_share": float(outlier_count / size) if size else 0.0,
                "review_status": clean_text(review.get("review_status")) or "unreviewed",
                "review_notes": clean_text(review.get("notes")),
                "display_label": clean_text(review.get("manual_label"))
                or clean_text(info.get("label"))
                or f"Cluster {cluster_id}",
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)


def apply_cluster_filters(
    claims: pd.DataFrame,
    overview: pd.DataFrame,
    filters: ClusterFilters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_claims = claims.copy()
    for column, values in [
        ("line_of_business", filters.line_of_business),
        ("country", filters.country),
        ("claim_type", filters.claim_type),
        ("cause_of_loss", filters.cause_of_loss),
        ("claim_status", filters.claim_status),
    ]:
        if values and column in filtered_claims:
            filtered_claims = filtered_claims[filtered_claims[column].isin(values)]

    filtered_claims = apply_range(filtered_claims, "loss_year", filters.loss_year_range)
    filtered_claims = apply_range(filtered_claims, "reserve_amount", filters.reserve_amount_range)
    filtered_claims = apply_range(filtered_claims, "paid_amount", filters.paid_amount_range)

    if filters.description_search:
        query = filters.description_search.casefold()
        filtered_claims = filtered_claims[
            filtered_claims.get("document", pd.Series("", index=filtered_claims.index))
            .fillna("")
            .astype(str)
            .str.casefold()
            .str.contains(query, regex=False)
        ]

    if filtered_claims.empty or "cluster_id" not in filtered_claims or overview.empty:
        return filtered_claims.iloc[0:0].reset_index(drop=True), overview.iloc[0:0].reset_index(drop=True)

    base_overview = overview.set_index("cluster_id")
    rows = []
    for cluster_id, cluster_claims in filtered_claims.groupby("cluster_id", dropna=False):
        cluster_id = int(cluster_id)
        if cluster_id not in base_overview.index:
            continue
        row = base_overview.loc[cluster_id].to_dict()
        reserve = pd.to_numeric(cluster_claims.get("reserve_amount"), errors="coerce").fillna(0)
        paid = pd.to_numeric(cluster_claims.get("paid_amount"), errors="coerce").fillna(0)
        size = int(len(cluster_claims))
        row.update(
            {
                "cluster_id": cluster_id,
                "cluster_size": size,
                "dominant_line_of_business": dominant_value(cluster_claims, "line_of_business"),
                "dominant_cause_of_loss": dominant_value(cluster_claims, "cause_of_loss"),
                "dominant_country": dominant_value(cluster_claims, "country"),
                "total_reserve": float(reserve.sum()),
                "total_paid": float(paid.sum()),
                "avg_reserve": float(reserve.mean()) if size else 0.0,
                "avg_paid": float(paid.mean()) if size else 0.0,
                "outlier_count": 1 if size == 1 else 0,
                "outlier_share": float(1 / size) if size == 1 else 0.0,
            }
        )
        rows.append(row)

    filtered_overview = pd.DataFrame(rows)
    if filtered_overview.empty:
        return filtered_claims.iloc[0:0].reset_index(drop=True), overview.iloc[0:0].reset_index(drop=True)

    filtered_overview = filtered_overview[filtered_overview["cluster_size"] >= filters.min_cluster_size]

    if filters.cluster_search and not filtered_overview.empty:
        query = filters.cluster_search.casefold()
        haystack = (
            filtered_overview["cluster_label"].fillna("")
            + " "
            + filtered_overview["manual_label"].fillna("")
            + " "
            + filtered_overview["top_keywords"].fillna("")
        ).str.casefold()
        filtered_overview = filtered_overview[haystack.str.contains(query, regex=False)]

    if filters.review_status and not filtered_overview.empty:
        filtered_overview = filtered_overview[filtered_overview["review_status"].isin(filters.review_status)]

    cluster_ids = set(filtered_overview["cluster_id"].dropna().astype(int)) if not filtered_overview.empty else set()
    filtered_claims = filtered_claims[filtered_claims["cluster_id"].astype(int).isin(cluster_ids)] if cluster_ids else filtered_claims.iloc[0:0]
    return filtered_claims.reset_index(drop=True), filtered_overview.reset_index(drop=True)


def cluster_financials(claims: pd.DataFrame) -> dict[str, float]:
    reserve = pd.to_numeric(claims.get("reserve_amount"), errors="coerce").fillna(0)
    paid = pd.to_numeric(claims.get("paid_amount"), errors="coerce").fillna(0)
    return {
        "total_reserve": float(reserve.sum()),
        "total_paid": float(paid.sum()),
        "avg_reserve": float(reserve.mean()) if len(reserve) else 0.0,
        "avg_paid": float(paid.mean()) if len(paid) else 0.0,
        "median_reserve": float(reserve.median()) if len(reserve) else 0.0,
        "largest_reserve": float(reserve.max()) if len(reserve) else 0.0,
    }


def common_metadata(claims: pd.DataFrame, *, limit: int = 5) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for column in ["line_of_business", "claim_type", "cause_of_loss", "country", "claim_status", "loss_year"]:
        if column not in claims:
            continue
        counts = claims[column].dropna().astype(str).value_counts().head(limit)
        result[column] = [{"value": value, "count": int(count)} for value, count in counts.items()]
    return result


def weak_assignment_candidates(claims: pd.DataFrame, overview: pd.DataFrame, *, limit: int = 50) -> pd.DataFrame:
    if claims.empty:
        return claims
    sizes = overview.set_index("cluster_id")["cluster_size"].to_dict() if not overview.empty else {}
    result = claims.copy()
    result["cluster_size"] = result["cluster_id"].map(sizes).fillna(0).astype(int)
    reserve = pd.to_numeric(result.get("reserve_amount"), errors="coerce").fillna(0)
    high_reserve_threshold = reserve.quantile(0.9) if len(reserve) else 0
    description_length = pd.to_numeric(result.get("description_length"), errors="coerce").fillna(0)
    result["outlier_reason"] = ""
    result.loc[result["cluster_size"] <= 1, "outlier_reason"] = "Singleton cluster"
    result.loc[(description_length > 0) & (description_length < 40), "outlier_reason"] = "Low-information description"
    result.loc[(reserve >= high_reserve_threshold) & (reserve > 0), "outlier_reason"] = "High severity claim"
    result = result[result["outlier_reason"] != ""]
    if result.empty:
        return result
    result["reserve_amount"] = reserve.loc[result.index]
    return result.sort_values(["reserve_amount", "claim_id"], ascending=[False, True]).head(limit).reset_index(drop=True)


def selected_cluster_report(
    *,
    profile: dict[str, Any],
    keywords: list[str],
    representatives: list[dict[str, Any]],
    financials: dict[str, float],
) -> str:
    lines = [
        f"# Cluster {profile.get('cluster_id')} - {profile.get('display_label')}",
        "",
        f"Size: {profile.get('cluster_size', 0)} claims",
        f"Review status: {profile.get('review_status', 'unreviewed')}",
        "",
        "## Keywords",
        ", ".join(keywords) if keywords else "No keywords available.",
        "",
        "## Representative Claims",
    ]
    for representative in representatives:
        lines.append(f"- {representative.get('claim_id')}: {representative.get('description')}")
    lines.extend(
        [
            "",
            "## Financials",
            f"Total reserve: {financials['total_reserve']:.2f}",
            f"Total paid: {financials['total_paid']:.2f}",
            f"Average reserve: {financials['avg_reserve']:.2f}",
            f"Largest reserve: {financials['largest_reserve']:.2f}",
        ]
    )
    return "\n".join(lines)


def apply_range(frame: pd.DataFrame, column: str, bounds: tuple[int | float | None, int | float | None]) -> pd.DataFrame:
    lower, upper = bounds
    if column not in frame or (lower is None and upper is None):
        return frame
    values = pd.to_numeric(frame[column], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if lower is not None:
        mask &= values >= lower
    if upper is not None:
        mask &= values <= upper
    return frame[mask]


def dominant_value(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    values = frame[column].dropna().astype(str)
    if values.empty:
        return ""
    return values.value_counts().index[0]


def review_lookup(reviews: pd.DataFrame, *, model_key: str, version: str) -> dict[int, dict[str, Any]]:
    if reviews.empty:
        return {}
    filtered = reviews[(reviews["model_key"] == model_key) & (reviews["cluster_version"] == version)]
    return {int(row["cluster_id"]): row.to_dict() for _, row in filtered.iterrows()}


def review_status_options() -> list[str]:
    return REVIEW_STATUSES
