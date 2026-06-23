from __future__ import annotations

import re
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from src.config import EmbeddingModelConfig
from src.similarity_search import (
    SIMILARITY_METRICS,
    SIMILARITY_METRICS_BY_KEY,
    SearchFilters,
    SimilarityMetric,
    query_similar_claims,
)
from src.snowflake_io import get_claim_document


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return WHITESPACE_RE.sub(" ", str(value).replace("\x00", " ")).strip()


def render_search_configuration(
    models: list[EmbeddingModelConfig],
) -> tuple[EmbeddingModelConfig, SimilarityMetric] | None:
    st.subheader("Search Configuration")
    models_by_key = {model.key: model for model in models}
    model_column, metric_column = st.columns(2)
    with model_column:
        selected_model_key = st.selectbox(
            "Embedding model",
            options=[model.key for model in models],
            index=None,
            placeholder="Select an embedding model",
            format_func=lambda key: models_by_key[key].label,
        )
    with metric_column:
        selected_metric_key = st.selectbox(
            "Similarity metric",
            options=[metric.key for metric in SIMILARITY_METRICS],
            index=None,
            placeholder="Select a similarity metric",
            format_func=lambda key: SIMILARITY_METRICS_BY_KEY[key].label,
        )
    if selected_model_key is None or selected_metric_key is None:
        return None
    return models_by_key[selected_model_key], SIMILARITY_METRICS_BY_KEY[selected_metric_key]


def render_search_page(
    session: Any,
    table_name: str,
    selected_model: EmbeddingModelConfig,
    selected_metric: SimilarityMetric,
    options: dict[str, Any],
) -> None:
    filters = render_filters(options)

    top_n = st.slider("Top results", min_value=5, max_value=50, value=10, step=5)

    st.subheader("Search")
    st.caption(
        f"Using `{selected_model.key}` with {selected_metric.label.lower()} "
        f"from `{table_name}`."
    )
    query_source = st.radio("Query source", ["Free text", "Existing claim"], horizontal=True)
    selected_claim = None
    query_text = ""
    if query_source == "Free text":
        query_text = st.text_area("Free-text claim description", height=130)
    else:
        selected_claim = existing_claim_selector(options["claims"])
    search_clicked = st.button("Search", type="primary", use_container_width=True)

    source_claim_id = None
    search_text = clean_text(query_text)
    if not search_clicked:
        st.info("Enter a claim description or select an existing claim, then run search.")
        return
    if query_source == "Existing claim" and selected_claim:
        source_claim_id = selected_claim
        try:
            search_text = clean_text(get_claim_document(session, table_name, selected_claim))
        except Exception as exc:
            st.error(f"Could not load the selected claim from Snowflake: {exc}")
            return
    if not search_text:
        if query_source == "Free text":
            st.warning("Provide free text.")
        else:
            st.warning("Select an existing claim.")
        return

    try:
        results = query_similar_claims(
            session,
            table_name,
            selected_model.key,
            search_text,
            metric_key=selected_metric.key,
            filters=filters,
            top_n=top_n,
            exclude_claim_id=source_claim_id,
        )
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    if results.empty:
        st.warning("No matching claims found for the selected filters.")
        return

    render_result_cards(results, search_text, filters, selected_metric)


def render_result_cards(
    results: pd.DataFrame,
    query_text: str,
    filters: SearchFilters,
    metric: SimilarityMetric,
) -> None:
    inject_result_card_css()
    filter_count = active_filter_count(filters)
    st.markdown(
        f"""
        <div class="claim-results-header">
            <div>
                <h2>Results</h2>
                <p>Showing the most similar claims to your query.</p>
            </div>
            <div class="claim-results-toolbar">
                <div class="claim-query-box">{escape(query_text)}</div>
                <div class="claim-toolbar-pill">Filters <span>{filter_count}</span></div>
                <div class="claim-toolbar-pill">Sort by: {escape(metric.label)}</div>
            </div>
            <div class="claim-result-count">{len(results)} results found</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        st.markdown(result_card_html(rank, row, metric), unsafe_allow_html=True)


def result_card_html(rank: int, row: pd.Series, metric: SimilarityMetric) -> str:
    claim_id = escape(clean_text(row.get("claim_id")))
    description = escape(result_description(row))
    score = score_float(row.get("metric_value"))
    score_text = format_score(score)
    score_bar = metric_score_bar(score, metric)
    score_row_class = "claim-score-row" if score_bar else "claim-score-row claim-score-row-raw"
    chips = "".join(result_chip_html(row, field, label, status=(field == "claim_status")) for label, field in [
        ("Line", "line_of_business"),
        ("Type", "claim_type"),
        ("Cause", "cause_of_loss"),
        ("Object", "damaged_object"),
        ("Country", "country"),
        ("Status", "claim_status"),
    ])
    tiles = "".join(result_tile_html(row, field, label) for label, field in [
        ("Reserve", "reserve_amount"),
        ("Paid", "paid_amount"),
        ("Loss date", "loss_date"),
        ("Policy", "policy_type"),
    ])
    chips_html = f'<div class="claim-chip-row">{chips}</div>' if chips else ""
    tiles_html = f'<div class="claim-tile-row">{tiles}</div>' if tiles else ""
    scores_html = score_details_html(row, metric)
    return f"""
    <div class="claim-result-card">
        <div class="claim-card-top">
            <div class="claim-card-identity">
                <span class="claim-rank">#{rank}</span>
                <span class="claim-id">{claim_id}</span>
            </div>
            <div class="claim-score-block">
                <div class="claim-score-label">{escape(metric.label)} · {metric.direction_label}</div>
                <div class="{score_row_class}"><span>{score_text}</span>{score_bar}</div>
            </div>
        </div>
        <div class="claim-description">{description}</div>
        {chips_html}
        {tiles_html}
        {scores_html}
    </div>
    """


def result_chip_html(row: pd.Series, field: str, label: str, *, status: bool = False) -> str:
    if field not in row or pd.isna(row.get(field)):
        return ""
    value = escape(format_metadata_value(field, row.get(field)))
    class_name = "claim-chip claim-chip-status" if status else "claim-chip"
    return f'<span class="{class_name}"><span>{escape(label)}</span> {value}</span>'


def result_tile_html(row: pd.Series, field: str, label: str) -> str:
    if field not in row or pd.isna(row.get(field)):
        return ""
    if field in {"reserve_amount", "paid_amount"}:
        value = format_amount_html(row, field)
    else:
        value = escape(format_metadata_value(field, row.get(field)))
    return f'<div class="claim-tile"><span>{escape(label)}</span><strong>{value}</strong></div>'


def score_details_html(row: pd.Series, metric: SimilarityMetric) -> str:
    items = result_score_items(row, metric)
    if not items:
        return ""
    cells = "".join(
        f'<div class="claim-score-detail"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in items
    )
    return (
        '<div class="claim-score-details">'
        '<div class="claim-score-title">Scoring details</div>'
        f'<div class="claim-score-detail-grid">{cells}</div>'
        '</div>'
    )


def inject_result_card_css() -> None:
    st.markdown(
        """
        <style>
        .claim-results-header {
            margin-top: 1.2rem;
            margin-bottom: 1rem;
        }
        .claim-results-header h2 {
            margin: 0 0 0.2rem 0;
            font-size: 1.7rem;
        }
        .claim-results-header p,
        .claim-result-count {
            color: #64748b;
            margin: 0;
            font-size: 0.9rem;
        }
        .claim-results-toolbar {
            display: grid;
            grid-template-columns: minmax(260px, 1fr) auto auto;
            gap: 1rem;
            align-items: center;
            margin: 1.25rem 0 0.75rem;
        }
        .claim-query-box,
        .claim-toolbar-pill {
            border: 1px solid #dfe7f1;
            border-radius: 8px;
            background: #ffffff;
            min-height: 42px;
            display: flex;
            align-items: center;
            color: #26364d;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }
        .claim-query-box {
            padding: 0 1rem;
        }
        .claim-toolbar-pill {
            padding: 0 1rem;
            gap: 0.6rem;
            white-space: nowrap;
            font-size: 0.9rem;
        }
        .claim-toolbar-pill span {
            display: inline-flex;
            min-width: 22px;
            min-height: 22px;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: #dcfce7;
            color: #15803d;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .claim-result-card {
            border: 1px solid #dfe7f1;
            border-radius: 8px;
            background: #ffffff;
            padding: 1.1rem 1.25rem 0.7rem;
            margin: 1rem 0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        .claim-card-top {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
        }
        .claim-card-identity {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        .claim-rank {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            border-radius: 8px;
            background: #dcfce7;
            color: #0f3d25;
            font-weight: 800;
            font-size: 1.05rem;
        }
        .claim-id {
            color: #039855;
            background: #f0fdf4;
            border-radius: 6px;
            padding: 0.35rem 0.55rem;
            font-family: monospace;
            font-weight: 800;
            letter-spacing: 0;
        }
        .claim-score-block {
            min-width: 310px;
        }
        .claim-score-label {
            color: #64748b;
            font-size: 0.78rem;
            margin-bottom: 0.25rem;
        }
        .claim-score-row {
            display: grid;
            grid-template-columns: 58px 1fr;
            gap: 0.75rem;
            align-items: center;
        }
        .claim-score-row span {
            color: #16a34a;
            font-size: 1.05rem;
            font-weight: 800;
        }
        .claim-score-row-raw {
            grid-template-columns: 1fr;
        }
        .claim-score-track {
            height: 8px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
        }
        .claim-score-track div {
            height: 100%;
            background: #16a34a;
            border-radius: inherit;
        }
        .claim-description {
            margin: 1.1rem 0 0.75rem;
            color: #0f172a;
            font-size: 1.15rem;
            line-height: 1.45;
            font-weight: 750;
        }
        .claim-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.8rem;
        }
        .claim-chip {
            border: 1px solid #dfe7f1;
            border-radius: 6px;
            padding: 0.38rem 0.65rem;
            color: #334155;
            background: #ffffff;
            font-size: 0.82rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }
        .claim-chip span {
            color: #64748b;
            margin-right: 0.3rem;
        }
        .claim-chip-status {
            background: #f0fdf4;
            color: #15803d;
            border-color: #d7f5df;
        }
        .claim-tile-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            border: 1px solid #dfe7f1;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 0.75rem;
        }
        .claim-tile {
            padding: 0.8rem 1rem;
            min-height: 62px;
            border-right: 1px solid #dfe7f1;
        }
        .claim-tile:last-child {
            border-right: none;
        }
        .claim-tile span {
            display: block;
            color: #64748b;
            font-size: 0.75rem;
            margin-bottom: 0.25rem;
        }
        .claim-tile strong {
            color: #0f172a;
            font-size: 0.92rem;
            font-weight: 700;
        }
        .claim-score-details {
            border-top: 1px solid #e5e7eb;
            padding-top: 0.65rem;
            display: grid;
            grid-template-columns: 120px minmax(0, 1fr);
            gap: 0.75rem;
            align-items: start;
        }
        .claim-score-title {
            color: #334155;
            font-weight: 700;
            font-size: 0.82rem;
            padding-top: 0.45rem;
        }
        .claim-score-detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.5rem;
        }
        .claim-score-detail {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            background: #f8fafc;
            padding: 0.45rem 0.6rem;
        }
        .claim-score-detail span {
            display: block;
            color: #64748b;
            font-size: 0.74rem;
            margin-bottom: 0.18rem;
        }
        .claim-score-detail strong {
            color: #0f172a;
            font-size: 0.88rem;
        }
        @media (max-width: 900px) {
            .claim-results-toolbar,
            .claim-tile-row,
            .claim-score-details {
                grid-template-columns: 1fr;
            }
            .claim-card-top {
                flex-direction: column;
            }
            .claim-score-block {
                min-width: 100%;
            }
            .claim-tile {
                border-right: none;
                border-bottom: 1px solid #dfe7f1;
            }
            .claim-tile:last-child {
                border-bottom: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def result_description(row: pd.Series) -> str:
    document = clean_text(row.get("document"))
    prefix = "Claim description:"
    if document.casefold().startswith(prefix.casefold()):
        description = document[len(prefix):].strip()
        marker = " line of business:"
        marker_index = description.casefold().find(marker)
        if marker_index >= 0:
            description = description[:marker_index].strip()
        return description
    return document


def result_score_items(row: pd.Series, metric: SimilarityMetric) -> list[tuple[str, str]]:
    value = row.get("metric_value")
    if pd.isna(value):
        return []
    return [(f"{metric.label} · {metric.direction_label}", format_score(value))]


def metric_score_bar(value: float, metric: SimilarityMetric) -> str:
    if metric.bounded_range is None:
        return ""
    minimum, maximum = metric.bounded_range
    percent = max(0.0, min((value - minimum) / (maximum - minimum), 1.0)) * 100
    return f'<div class="claim-score-track"><div style="width: {percent:.0f}%"></div></div>'


def active_filter_count(filters: SearchFilters) -> int:
    count = len(filters.equality)
    count += int(filters.loss_year_range != (None, None))
    count += int(filters.reserve_amount_range != (None, None))
    count += int(filters.paid_amount_range != (None, None))
    return count


def format_score(value: Any) -> str:
    return f"{score_float(value):.3f}"


def score_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_metadata_value(field: str, value: Any) -> str:
    if field in {"reserve_amount", "paid_amount"}:
        try:
            return f"{float(value):,.0f}"
        except (TypeError, ValueError):
            return clean_text(value)
    if field == "loss_year":
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return clean_text(value)
    return clean_text(value)


def format_amount_html(row: pd.Series, field: str) -> str:
    try:
        amount = f"{float(row.get(field)):,.0f}"
    except (TypeError, ValueError):
        amount = escape(clean_text(row.get(field)))
    currency = clean_text(row.get("currency"))
    symbols = {"EUR": "&euro;", "GBP": "&pound;", "USD": "$"}
    prefix = symbols.get(currency, f"{escape(currency)} " if currency else "")
    return f"{prefix}{amount}"


def existing_claim_selector(claims: list[dict[str, Any]]) -> str | None:
    label_by_id = {}
    for row in claims:
        claim_id = clean_text(row.get("CLAIM_ID"))
        summary = " | ".join(
            clean_text(row.get(field))
            for field in ["LINE_OF_BUSINESS", "CLAIM_TYPE", "COUNTRY", "CLAIM_STATUS"]
            if clean_text(row.get(field))
        )
        label_by_id[claim_id] = f"{claim_id} - {summary}" if summary else claim_id
    labels = [""] + list(label_by_id.values())
    selected_label = st.selectbox("Existing claim", labels, index=0)
    if not selected_label:
        return None
    for claim_id, label in label_by_id.items():
        if label == selected_label:
            return claim_id
    return None


def render_filters(options: dict[str, Any]) -> SearchFilters:
    with st.sidebar:
        st.header("Filters")
        equality = {}
        for field in ["line_of_business", "claim_type", "cause_of_loss", "country", "claim_status", "policy_type"]:
            values = options["equality"].get(field, [])
            if values:
                choices = ["All"] + [clean_text(value) for value in values]
                value = st.selectbox(field.replace("_", " ").title(), choices)
                if value != "All":
                    equality[field] = value

        loss_year_range = numeric_range(options["ranges"].get("loss_year"), "Loss year", integer=True)
        reserve_range = numeric_range(options["ranges"].get("reserve_amount"), "Reserve amount")
        paid_range = numeric_range(options["ranges"].get("paid_amount"), "Paid amount")

    return SearchFilters(
        equality=equality,
        loss_year_range=loss_year_range,
        reserve_amount_range=reserve_range,
        paid_amount_range=paid_range,
    )


def numeric_range(
    bounds: tuple[Any, Any] | None,
    label: str,
    *,
    integer: bool = False,
) -> tuple[int | float | None, int | float | None]:
    if not bounds or bounds[0] is None or bounds[1] is None:
        return (None, None)
    low = int(bounds[0]) if integer else float(bounds[0])
    high = int(bounds[1]) if integer else float(bounds[1])
    if low == high:
        return (None, None)
    selected = st.slider(label, min_value=low, max_value=high, value=(low, high))
    if selected == (low, high):
        return (None, None)
    return selected
