from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_helpers import cluster_claim_columns, format_amount
from src.cluster_profiles import (
    ClusterFilters,
    apply_cluster_filters,
    build_cluster_overview,
    cluster_financials,
    cluster_version,
    common_metadata,
    review_status_options,
    selected_cluster_report,
    weak_assignment_candidates,
)
from src.cluster_review_store import load_reviews, save_cluster_review
from src.config import AppConfig, EmbeddingModelConfig
from src.text_preprocessing import clean_text


def render_cluster_page(
    config: AppConfig,
    selected_model: EmbeddingModelConfig,
    frame: pd.DataFrame,
    clusters: dict[str, Any],
    cluster_map: dict[str, Any],
) -> None:
    if frame.empty or "cluster_id" not in frame:
        st.info(
            f"No current cluster artifact is available for {selected_model.label}. "
            "Index Setup only builds the Chroma search index; build clusters separately for this active index."
        )
        return

    version = cluster_version(clusters, selected_model.key)
    reviews = load_reviews(config.cluster_reviews_path)
    overview = build_cluster_overview(frame, clusters, reviews, model_key=selected_model.key, version=version)
    filters = render_cluster_filters(frame, overview, version)
    filtered, overview = apply_cluster_filters(frame, overview, filters)
    if overview.empty:
        st.warning("No clusters match the selected filters.")
        return

    selected_cluster = st.selectbox(
        "Selected cluster",
        overview["cluster_id"].tolist(),
        format_func=lambda cluster_id: cluster_option_label(overview, cluster_id),
    )
    cluster_frame = filtered[filtered["cluster_id"] == selected_cluster]
    cluster_info = next(
        (cluster for cluster in clusters.get("clusters", []) if cluster.get("cluster_id") == selected_cluster),
        {},
    )
    profile = overview[overview["cluster_id"] == selected_cluster].iloc[0].to_dict()
    financials = cluster_financials(cluster_frame)

    overview_tab, detail_tab, map_tab, analytics_tab, outliers_tab, review_tab = st.tabs(
        ["Overview", "Cluster Detail", "Map", "Analytics", "Outliers", "Review"]
    )

    with overview_tab:
        render_cluster_overview_tab(filtered, overview, reviews, version, selected_model.key)
    with detail_tab:
        render_cluster_detail_tab(profile, cluster_info, cluster_frame, financials)
    with map_tab:
        render_cluster_map_tab(filtered, overview, cluster_map, selected_cluster)
    with analytics_tab:
        render_cluster_analytics_tab(filtered, overview)
    with outliers_tab:
        render_cluster_outliers_tab(filtered, overview)
    with review_tab:
        render_cluster_review_tab(config, selected_model, version, selected_cluster, profile, reviews)


def render_cluster_filters(frame: pd.DataFrame, overview: pd.DataFrame, version: str) -> ClusterFilters:
    with st.sidebar:
        st.header("Cluster Filters")
        st.selectbox("Cluster version", [version], key="cluster_version")
        line_of_business = multiselect_filter(frame, "line_of_business", "Line of business")
        country = multiselect_filter(frame, "country", "Country")
        claim_type = multiselect_filter(frame, "claim_type", "Claim type")
        cause_of_loss = multiselect_filter(frame, "cause_of_loss", "Cause of loss")
        claim_status = multiselect_filter(frame, "claim_status", "Claim status")
        loss_year_range = cluster_numeric_range(frame, "loss_year", "Cluster loss year", integer=True)
        reserve_range = cluster_numeric_range(frame, "reserve_amount", "Cluster reserve amount")
        paid_range = cluster_numeric_range(frame, "paid_amount", "Cluster paid amount")
        max_size = int(overview["cluster_size"].max()) if not overview.empty else 1
        min_cluster_size = 1
        if max_size > 1:
            min_cluster_size = st.slider("Minimum cluster size", 1, max_size, 1)
        description_search = st.text_input("Search claim descriptions")
        cluster_search = st.text_input("Search cluster labels / keywords")
        review_status = st.multiselect("Review status", review_status_options())

    return ClusterFilters(
        line_of_business=line_of_business,
        country=country,
        claim_type=claim_type,
        cause_of_loss=cause_of_loss,
        claim_status=claim_status,
        loss_year_range=loss_year_range,
        reserve_amount_range=reserve_range,
        paid_amount_range=paid_range,
        min_cluster_size=min_cluster_size,
        description_search=description_search,
        cluster_search=cluster_search,
        review_status=review_status,
    )


def render_cluster_overview_tab(
    frame: pd.DataFrame,
    overview: pd.DataFrame,
    reviews: pd.DataFrame,
    version: str,
    model_key: str,
) -> None:
    reviewed = reviews[(reviews.get("model_key") == model_key) & (reviews.get("cluster_version") == version)] if not reviews.empty else reviews
    cols = st.columns(6)
    cols[0].metric("Claims shown", len(frame))
    cols[1].metric("Clusters", overview["cluster_id"].nunique())
    cols[2].metric("Outliers", int(overview["outlier_count"].sum()))
    cols[3].metric("Largest cluster", int(overview["cluster_size"].max()))
    cols[4].metric("Total reserve", format_amount(overview["total_reserve"].sum()))
    cols[5].metric("Reviewed", f"{len(reviewed)}/{len(overview)}")

    st.download_button(
        "Download cluster overview CSV",
        overview.to_csv(index=False).encode("utf-8"),
        file_name="cluster_overview.csv",
        mime="text/csv",
    )
    st.dataframe(overview_table(overview), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Cluster Size")
        st.plotly_chart(px.bar(overview, x="cluster_id", y="cluster_size"), use_container_width=True, key="overview_cluster_size")
    with right:
        st.subheader("Top Reserve")
        top_reserve = overview.sort_values("total_reserve", ascending=False).head(20)
        st.plotly_chart(px.bar(top_reserve, x="cluster_id", y="total_reserve"), use_container_width=True, key="overview_top_reserve")


def render_cluster_detail_tab(
    profile: dict[str, Any],
    cluster_info: dict[str, Any],
    cluster_frame: pd.DataFrame,
    financials: dict[str, float],
) -> None:
    st.subheader(f"Cluster {profile['cluster_id']} - {profile['display_label']}")
    cols = st.columns(5)
    cols[0].metric("Claims", int(profile["cluster_size"]))
    cols[1].metric("Total reserve", format_amount(financials["total_reserve"]))
    cols[2].metric("Total paid", format_amount(financials["total_paid"]))
    cols[3].metric("Avg reserve", format_amount(financials["avg_reserve"]))
    cols[4].metric("Largest claim", format_amount(financials["largest_reserve"]))

    left, right = st.columns([2, 1])
    with left:
        st.write("Keywords")
        st.write(", ".join(cluster_info.get("frequent_terms", [])) or "No keywords available.")
        st.write("Representative claims")
        for representative in cluster_info.get("representative_claims", []):
            st.markdown(f"- `{representative.get('claim_id')}` {representative.get('description')}")
    with right:
        st.write("Common metadata")
        st.json(common_metadata(cluster_frame))

    search = st.text_input("Search descriptions in selected cluster")
    table = cluster_frame.copy()
    if search:
        table = table[table["document"].fillna("").astype(str).str.casefold().str.contains(search.casefold(), regex=False)]
    view = st.selectbox(
        "Claims view",
        ["Representative claims", "Largest claims", "Newest claims", "Weakest assigned claims"],
    )
    if view == "Largest claims":
        table = table.sort_values("reserve_amount", ascending=False)
    elif view == "Newest claims" and "loss_date" in table:
        table = table.sort_values("loss_date", ascending=False)
    elif view == "Weakest assigned claims":
        table = table.sort_values("description_length", ascending=True)
    st.download_button(
        "Download selected cluster CSV",
        cluster_claim_columns(table).to_csv(index=False).encode("utf-8"),
        file_name=f"cluster_{profile['cluster_id']}_claims.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download cluster report Markdown",
        selected_cluster_report(
            profile=profile,
            keywords=cluster_info.get("frequent_terms", []),
            representatives=cluster_info.get("representative_claims", []),
            financials=financials,
        ).encode("utf-8"),
        file_name=f"cluster_{profile['cluster_id']}_report.md",
        mime="text/markdown",
    )
    st.dataframe(cluster_claim_columns(table).head(200), use_container_width=True, hide_index=True)


def render_cluster_map_tab(
    frame: pd.DataFrame,
    overview: pd.DataFrame,
    cluster_map: dict[str, Any],
    selected_cluster: int,
) -> None:
    st.subheader("Cluster Map")
    points = pd.DataFrame(cluster_map.get("points", []))
    if points.empty:
        st.info("No cluster map artifact is available. Run `python scripts/seed_dummy_chroma.py --all-models`.")
        return
    if frame.empty or "claim_id" not in frame:
        st.info("No claims match the current filters.")
        return

    points["claim_id"] = points["claim_id"].astype(str)
    filtered_claims = frame.copy()
    filtered_claims["claim_id"] = filtered_claims["claim_id"].astype(str)
    map_frame = points[["claim_id", "x", "y", "description"]].merge(filtered_claims, on="claim_id", how="inner")
    if map_frame.empty:
        st.info("No mapped claims match the current filters.")
        return

    label_by_cluster = overview.set_index("cluster_id")["display_label"].to_dict()
    map_frame["cluster_label"] = map_frame["cluster_id"].map(label_by_cluster).fillna("Unknown")
    map_frame["selected_cluster"] = map_frame["cluster_id"].map(
        lambda cluster_id: "Selected cluster" if int(cluster_id) == int(selected_cluster) else "Other clusters"
    )
    reserve_values = map_frame["reserve_amount"] if "reserve_amount" in map_frame else pd.Series(index=map_frame.index, dtype=float)
    map_frame["reserve_bucket"] = reserve_bucket(reserve_values)
    map_frame["short_description"] = map_frame["description"].fillna("").astype(str).str.slice(0, 140)

    color_options = ["cluster_id"] + [
        column for column in ["line_of_business", "country", "claim_status"] if column in map_frame
    ] + ["reserve bucket"]
    size_options = ["Fixed"] + [column for column in ["reserve_amount", "paid_amount"] if column in map_frame]
    controls = st.columns(2)
    with controls[0]:
        color_by = st.selectbox(
            "Color by",
            color_options,
        )
    with controls[1]:
        size_by = st.selectbox("Size by", size_options)

    color_column = "reserve_bucket" if color_by == "reserve bucket" else color_by
    if color_by == "cluster_id":
        map_frame["cluster_id"] = map_frame["cluster_id"].astype(str)

    size_column = None
    if size_by != "Fixed" and size_by in map_frame:
        map_frame["point_size"] = pd.to_numeric(map_frame[size_by], errors="coerce").fillna(0)
        if map_frame["point_size"].max() > 0:
            size_column = "point_size"

    fig = px.scatter(
        map_frame,
        x="x",
        y="y",
        color=color_column,
        size=size_column,
        symbol="selected_cluster",
        symbol_map={"Selected cluster": "circle", "Other clusters": "circle-open"},
        hover_data=[
            column
            for column in [
                "claim_id",
                "short_description",
                "cluster_id",
                "cluster_label",
                "line_of_business",
                "country",
                "reserve_amount",
                "paid_amount",
            ]
            if column in map_frame
        ],
        labels={color_column: color_by, "point_size": size_by, "short_description": "description"},
        size_max=24,
    )
    fig.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title=color_by,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    if size_column is None:
        fig.update_traces(marker={"size": 11})
    st.caption(f"{len(map_frame)} mapped claims shown. Projection: {cluster_map.get('projection', {}).get('method', 'unknown')}.")
    st.plotly_chart(fig, use_container_width=True, key="cluster_map")


def render_cluster_analytics_tab(frame: pd.DataFrame, overview: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Top Clusters by Claim Count")
        st.plotly_chart(px.bar(overview.head(20), x="cluster_id", y="cluster_size"), use_container_width=True, key="analytics_cluster_count")
        st.subheader("Average Reserve")
        avg_reserve = overview.sort_values("avg_reserve", ascending=False).head(20)
        st.plotly_chart(px.bar(avg_reserve, x="cluster_id", y="avg_reserve"), use_container_width=True, key="analytics_avg_reserve")
    with right:
        st.subheader("Total Paid")
        paid = overview.sort_values("total_paid", ascending=False).head(20)
        st.plotly_chart(px.bar(paid, x="cluster_id", y="total_paid"), use_container_width=True, key="analytics_total_paid")
        st.subheader("Status Distribution")
        if "claim_status" in frame:
            status = frame["claim_status"].fillna("Unknown").value_counts().reset_index()
            status.columns = ["claim_status", "count"]
            st.plotly_chart(px.bar(status, x="claim_status", y="count"), use_container_width=True, key="analytics_status_distribution")

    if "loss_year" in frame:
        st.subheader("Claims Over Time by Cluster")
        trend = frame.groupby(["loss_year", "cluster_id"]).size().reset_index(name="count")
        st.plotly_chart(px.line(trend, x="loss_year", y="count", color="cluster_id"), use_container_width=True, key="analytics_claims_over_time")


def render_cluster_outliers_tab(frame: pd.DataFrame, overview: pd.DataFrame) -> None:
    st.subheader("Weak Assignments / Unusual Claims")
    outliers = weak_assignment_candidates(frame, overview)
    if outliers.empty:
        st.info("No weak assignment candidates found with the MVP heuristics.")
        return
    columns = [
        "claim_id",
        "document",
        "cluster_id",
        "cluster_size",
        "outlier_reason",
        "reserve_amount",
        "paid_amount",
        "line_of_business",
        "claim_type",
        "country",
    ]
    st.dataframe(outliers[[column for column in columns if column in outliers]], use_container_width=True, hide_index=True)


def render_cluster_review_tab(
    config: AppConfig,
    selected_model: EmbeddingModelConfig,
    version: str,
    cluster_id: int,
    profile: dict[str, Any],
    reviews: pd.DataFrame,
) -> None:
    st.subheader(f"Review Cluster {cluster_id}")
    current = reviews[
        (reviews.get("model_key") == selected_model.key)
        & (reviews.get("cluster_version") == version)
        & (reviews.get("cluster_id").astype(str) == str(cluster_id))
    ] if not reviews.empty else pd.DataFrame()
    current_row = current.iloc[0].to_dict() if not current.empty else {}
    manual_label = st.text_input("Manual cluster label", value=clean_text(current_row.get("manual_label")) or profile["display_label"])
    review_status = st.selectbox(
        "Review status",
        review_status_options(),
        index=review_status_options().index(clean_text(current_row.get("review_status")) or "unreviewed"),
    )
    notes = st.text_area("Review notes", value=clean_text(current_row.get("notes")))
    if st.button("Save review", type="primary"):
        save_cluster_review(
            config.cluster_reviews_path,
            model_key=selected_model.key,
            cluster_version=version,
            cluster_id=int(cluster_id),
            manual_label=manual_label,
            review_status=review_status,
            notes=notes,
        )
        st.success("Review saved.")
        st.rerun()

    st.subheader("Review History")
    if reviews.empty:
        st.info("No cluster reviews saved yet.")
    else:
        st.dataframe(reviews.sort_values("updated_at_utc", ascending=False), use_container_width=True, hide_index=True)


def multiselect_filter(frame: pd.DataFrame, column: str, label: str) -> list[str]:
    if column not in frame:
        return []
    options = sorted(clean_text(value) for value in frame[column].dropna().unique() if clean_text(value))
    return st.multiselect(label, options, key=f"cluster_filter_{column}")


def cluster_numeric_range(
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
    selected = st.slider(label, min_value=low, max_value=high, value=(low, high), key=f"cluster_range_{column}")
    if selected == (low, high):
        return (None, None)
    return selected


def cluster_option_label(overview: pd.DataFrame, cluster_id: int) -> str:
    row = overview[overview["cluster_id"] == cluster_id].iloc[0]
    return f"{cluster_id} - {row['display_label']} ({int(row['cluster_size'])} claims)"


def overview_table(overview: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cluster_id",
        "display_label",
        "manual_label",
        "cluster_size",
        "top_keywords",
        "dominant_line_of_business",
        "dominant_cause_of_loss",
        "dominant_country",
        "total_reserve",
        "total_paid",
        "avg_reserve",
        "outlier_share",
        "review_status",
    ]
    table = overview[[column for column in columns if column in overview]].copy()
    for column in ["total_reserve", "total_paid", "avg_reserve", "outlier_share"]:
        if column in table:
            table[column] = table[column].map(lambda value: round(float(value), 4))
    return table


def reserve_bucket(values: Any) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[float("-inf"), 50_000, 100_000, 250_000, float("inf")],
        labels=["<=50k", "50k-100k", "100k-250k", ">250k"],
    ).astype(str).replace("nan", "Unknown")
