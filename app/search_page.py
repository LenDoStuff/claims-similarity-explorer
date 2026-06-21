from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from src.config import (
    AppConfig,
    EmbeddingModelConfig,
    RerankerModelConfig,
    available_embedding_models,
    available_reranker_models,
)
from src.similarity_search import (
    SearchFilters,
    apply_rerank_scores,
    merge_search_candidates,
    query_bm25_candidates,
    query_similar_claims,
)
from src.text_preprocessing import clean_text


def render_search_model_selector(config: AppConfig) -> EmbeddingModelConfig | None:
    st.subheader("Search Configuration")
    models = available_embedding_models()
    if not models:
        st.warning("No local embedding models were found in `models/embeddings`.")
        return None
    default_index = next(
        (index for index, model in enumerate(models) if model.key == config.model_key),
        0,
    )
    selected_key = st.selectbox(
        "Embedding model",
        options=[model.key for model in models],
        index=default_index,
        format_func=lambda key: next(model.label for model in models if model.key == key),
    )
    return next(model for model in models if model.key == selected_key)


def render_search_page(
    config: AppConfig,
    selected_model: EmbeddingModelConfig,
    collection: Any,
    claims_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    if claims_frame.empty:
        if manifest.get("collection_name") and manifest.get("index_hash"):
            st.info(
                f"The active Chroma collection for {selected_model.label} has no claims. "
                "Open Index Setup and click `Load or refresh index`."
            )
        else:
            st.info(
                f"No active Chroma index is available for {selected_model.label}. "
                "Open Index Setup and click `Load or refresh index`."
            )
        return

    filters = render_filters(claims_frame)
    index_model_name = selected_model.repo_id
    index_model_path = selected_model.model_dir

    top_n, retrieval_mode, keyword_algorithm, semantic_weight, candidate_pool, reranker_model, rerank_top_k = render_search_controls()

    st.subheader("Search")
    st.caption(f"Using `{index_model_name}` from `{manifest.get('collection_name', '')}`.")
    query_source = st.radio("Query source", ["Free text", "Existing claim"], horizontal=True)
    selected_claim = None
    query_text = ""
    if query_source == "Free text":
        query_text = st.text_area("Free-text claim description", height=130)
    else:
        selected_claim = existing_claim_selector(claims_frame)
    search_clicked = st.button("Search", type="primary", use_container_width=True)

    source_claim_id = None
    search_text = clean_text(query_text)
    if query_source == "Existing claim" and selected_claim:
        source_row = claims_frame[claims_frame["claim_id"] == selected_claim].head(1)
        if not source_row.empty:
            source_claim_id = selected_claim
            search_text = clean_text(source_row.iloc[0].get("document"))

    if not search_clicked:
        st.info("Enter a claim description or select an existing claim, then run search.")
        return
    if not search_text:
        if query_source == "Free text":
            st.warning("Provide free text.")
        else:
            st.warning("Select an existing claim.")
        return

    try:
        semantic_results = pd.DataFrame()
        if retrieval_mode in {"Semantic", "Hybrid"}:
            from src.embeddings import load_embedding_model

            model = load_embedding_model(str(index_model_path), index_model_name)
            query_embedding = model.encode_queries([search_text])
            semantic_results = query_similar_claims(
                collection,
                query_embedding,
                filters=filters,
                top_n=candidate_pool,
                candidate_multiplier=1,
                exclude_claim_id=source_claim_id,
            )

        bm25_results = pd.DataFrame()
        if retrieval_mode in {"BM25", "Hybrid"} and keyword_algorithm == "BM25":
            bm25_results = query_bm25_candidates(
                claims_frame,
                search_text,
                filters=filters,
                candidate_pool=candidate_pool,
                exclude_claim_id=source_claim_id,
            )

        results = merge_search_candidates(
            semantic_results,
            bm25_results,
            retrieval_mode=retrieval_mode,
            semantic_weight=semantic_weight,
        )
        if reranker_model is not None and not results.empty:
            from src.embeddings import load_reranker_model

            rerank_candidates = results.head(rerank_top_k)
            reranker = load_reranker_model(str(reranker_model.model_dir), reranker_model.repo_id)
            rerank_scores = reranker.predict(search_text, rerank_candidates["document"].fillna("").astype(str).tolist())
            results = apply_rerank_scores(rerank_candidates, rerank_scores)

        results = results.head(top_n)
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    if results.empty:
        st.warning("No matching claims found for the selected filters.")
        return

    render_result_cards(results, search_text, filters)


def render_search_controls() -> tuple[int, str, str, float, int, RerankerModelConfig | None, int]:
    reranker_models = available_reranker_models()
    reranker_by_key = {model.key: model for model in reranker_models}
    top, keyword, rerank, results = st.columns(4)
    with top:
        retrieval_mode = st.selectbox("Retrieval mode", ["Semantic", "BM25", "Hybrid"], index=0)
    with keyword:
        keyword_algorithm = st.selectbox("Keyword search algorithm", ["BM25"], index=0)
    with rerank:
        rerank_options = ["Off"] + [model.key for model in reranker_models]
        rerank_mode = st.selectbox(
            "Rerank model",
            rerank_options,
            index=0,
            format_func=lambda key: "Off" if key == "Off" else reranker_by_key[key].label,
        )
    with results:
        top_n = st.slider("Top results", min_value=5, max_value=50, value=10, step=5)

    left, middle, right = st.columns(3)
    with left:
        candidate_pool = st.slider("Candidate pool", min_value=50, max_value=300, value=100, step=25)
    with middle:
        semantic_weight = 1.0
        if retrieval_mode == "Hybrid":
            semantic_weight = st.slider("Semantic weight", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    with right:
        rerank_top_k = min(50, candidate_pool)
        if rerank_mode != "Off":
            rerank_top_k = st.slider(
                "Rerank top-K",
                min_value=top_n,
                max_value=candidate_pool,
                value=min(max(50, top_n), candidate_pool),
                step=5,
            )
    reranker_model = None if rerank_mode == "Off" else reranker_by_key[rerank_mode]
    return top_n, retrieval_mode, keyword_algorithm, semantic_weight, candidate_pool, reranker_model, rerank_top_k


def render_result_cards(results: pd.DataFrame, query_text: str, filters: SearchFilters) -> None:
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
                <div class="claim-toolbar-pill">Sort by: Match score</div>
            </div>
            <div class="claim-result-count">{len(results)} results found</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        st.markdown(result_card_html(rank, row), unsafe_allow_html=True)


def result_card_html(rank: int, row: pd.Series) -> str:
    claim_id = escape(clean_text(row.get("claim_id") or row.get("id")))
    description = escape(result_description(row))
    score = score_float(row.get("final_score", row.get("similarity", 0)))
    score_text = format_score(score)
    score_percent = max(0.0, min(score, 1.0)) * 100
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
    scores = " <span>|</span> ".join(escape(item) for item in result_score_items(row))
    return f"""
    <div class="claim-result-card">
        <div class="claim-card-top">
            <div class="claim-card-identity">
                <span class="claim-rank">#{rank}</span>
                <span class="claim-id">{claim_id}</span>
            </div>
            <div class="claim-score-block">
                <div class="claim-score-label">Match score</div>
                <div class="claim-score-row">
                    <span>{score_text}</span>
                    <div class="claim-score-track"><div style="width: {score_percent:.0f}%"></div></div>
                </div>
            </div>
        </div>
        <div class="claim-description">{description}</div>
        <div class="claim-chip-row">{chips}</div>
        <div class="claim-tile-row">{tiles}</div>
        <div class="claim-score-details"><span>Scoring details</span> {scores}</div>
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
            color: #64748b;
            font-size: 0.82rem;
        }
        .claim-score-details > span:first-child {
            color: #334155;
            font-weight: 700;
            margin-right: 1rem;
        }
        @media (max-width: 900px) {
            .claim-results-toolbar,
            .claim-tile-row {
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


def result_score_items(row: pd.Series) -> list[str]:
    items = []
    for label, field in [
        ("semantic", "semantic_score"),
        ("BM25", "bm25_score"),
        ("rerank", "rerank_score"),
        ("similarity", "similarity"),
        ("distance", "distance"),
    ]:
        if field in row and pd.notna(row.get(field)):
            items.append(f"{label} {format_score(row.get(field))}")
    return items


def active_filter_count(filters: SearchFilters) -> int:
    count = len(filters.equality)
    count += int(filters.loss_year_range != (None, None))
    count += int(filters.reserve_amount_range != (None, None))
    count += int(filters.paid_amount_range != (None, None))
    return count


def result_metadata_items(row: pd.Series) -> list[str]:
    items = []
    for label, field in [
        ("Line", "line_of_business"),
        ("Type", "claim_type"),
        ("Cause", "cause_of_loss"),
        ("Object", "damaged_object"),
        ("Country", "country"),
        ("Status", "claim_status"),
        ("Loss year", "loss_year"),
        ("Loss date", "loss_date"),
        ("Reserve", "reserve_amount"),
        ("Paid", "paid_amount"),
        ("Currency", "currency"),
        ("Policy", "policy_type"),
    ]:
        if field in row and pd.notna(row.get(field)):
            items.append(f"{label}: {format_metadata_value(field, row.get(field))}")
    return items


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


def existing_claim_selector(frame: pd.DataFrame) -> str | None:
    label_by_id = {}
    for _, row in frame.sort_values("claim_id").iterrows():
        claim_id = clean_text(row.get("claim_id"))
        summary = " | ".join(
            clean_text(row.get(field))
            for field in ["line_of_business", "claim_type", "country", "claim_status"]
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


def render_filters(frame: pd.DataFrame) -> SearchFilters:
    with st.sidebar:
        st.header("Filters")
        equality = {}
        for field in ["line_of_business", "claim_type", "cause_of_loss", "country", "claim_status", "policy_type"]:
            if field in frame:
                options = ["All"] + sorted(clean_text(value) for value in frame[field].dropna().unique() if clean_text(value))
                value = st.selectbox(field.replace("_", " ").title(), options)
                if value != "All":
                    equality[field] = value

        loss_year_range = numeric_range(frame, "loss_year", "Loss year", integer=True)
        reserve_range = numeric_range(frame, "reserve_amount", "Reserve amount")
        paid_range = numeric_range(frame, "paid_amount", "Paid amount")

    return SearchFilters(
        equality=equality,
        loss_year_range=loss_year_range,
        reserve_amount_range=reserve_range,
        paid_amount_range=paid_range,
    )


def numeric_range(
    frame: pd.DataFrame,
    column: str,
    label: str,
    *,
    integer: bool = False,
) -> tuple[int | float | None, int | float | None]:
    if column not in frame:
        return (None, None)
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return (None, None)
    low = int(values.min()) if integer else float(values.min())
    high = int(values.max()) if integer else float(values.max())
    if low == high:
        return (None, None)
    selected = st.slider(label, min_value=low, max_value=high, value=(low, high))
    if selected == (low, high):
        return (None, None)
    return selected
