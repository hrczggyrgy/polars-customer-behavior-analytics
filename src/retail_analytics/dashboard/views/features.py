"""Features view module."""

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from retail_analytics.visualization.chart_components import (
    build_color_map,
    get_color,
    safe_list,
    sampling_disclosure,
    format_axis_title,
    METRICS,
)
from retail_analytics.dashboard.app import collect, table, PAGE_ROWS


def render_features(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Features view - distributions, scatter plots, and feature diagnostics."""
    st.subheader("Feature Distributions & Scatterplots")
    st.caption(
        "Distributions of customer features and their relationships. "
        "Sampled up to 2,000 hash-selected finite observations. "
        "Missing values are not zeros. No customer IDs are sent to charts."
    )

    features = [f for f in [
        "spend_90d", "purchases_90d", "recency", "median_basket", "median_gap",
        "gap_cv", "lateness", "new_share", "repeat_share", "established_share",
        "trial_conversion", "category_hhi", "products_last6days", "categories_last6days",
        "frequency_change", "basket_change",
    ] if f in rows.collect_schema().names()]

    expressions = [pl.len().alias("n")]
    for f in features:
        finite = pl.col(f).filter(pl.col(f).is_finite())
        expressions += [finite.count().alias(f + "_n"), finite.median().alias(f + "_median")]
    stats = collect(rows.select(expressions)).row(0, named=True)
    diagnostic = pl.DataFrame(
        [
            dict(
                feature=f,
                finite=stats[f + "_n"],
                unavailable=stats["n"] - stats[f + "_n"],
                median=stats[f + "_median"],
            )
            for f in features
        ]
    )

    from retail_analytics.dashboard.app import bars
    bars(
        diagnostic,
        "feature",
        "unavailable",
        "Missing or non-finite feature values",
        "missingness",
        color_map,
    )
    table(diagnostic)

    feature = st.selectbox("Distribution feature", features)
    sample = collect(
        rows.filter(pl.col(feature).is_finite())
        .sort(pl.col("customer_id").hash(seed=29))
        .limit(PAGE_ROWS)
        .select(feature)
    )
    if sample.height:
        for box, trace, title in zip(
            st.columns(2),
            [
                go.Histogram(x=sample[feature].to_list(), nbinsx=30),
                go.Box(x=sample[feature].to_list(), boxpoints=False),
            ],
            ["Histogram", "Box plot"],
        ):
            with box:
                fig = go.Figure(trace)
                fig.update_layout(
                    title=title,
                    template="plotly_white",
                    height=400,
                    margin=dict(l=20, r=20, t=65, b=50),
                )
                st.plotly_chart(fig, use_container_width=True)

    x = st.selectbox("Scatter X", features)
    y = st.selectbox("Scatter Y", features, index=min(1, len(features) - 1))
    dim = st.selectbox("Scatter color", ["behavior", "trajectory", "lifecycle", "value_tier", "evidence"])
    if x != y:
        sample = collect(
            rows.filter(pl.col(x).is_finite() & pl.col(y).is_finite())
            .sort(pl.col("customer_id").hash(seed=29))
            .limit(PAGE_ROWS)
            .select(x, y, dim)
        )
        fig = go.Figure()
        for name in sorted(sample[dim].unique().to_list()):
            part = sample.filter(pl.col(dim) == name)
            fig.add_trace(
                go.Scattergl(
                    x=part[x].to_list(),
                    y=part[y].to_list(),
                    mode="markers",
                    name=name,
                    marker=dict(size=6, opacity=0.55, color=get_color(name, color_map)),
                )
            )
        # Add sampling disclosure
        total_finite = collect(
            rows.filter(pl.col(x).is_finite() & pl.col(y).is_finite()).select(pl.len())
        ).item()
        disclosure = sampling_disclosure(
            sampled_n=sample.height,
            population_n=total_finite,
            method="deterministic hash (seed=29)",
            stratified=True,
            strata=sample[dim].unique().to_list(),
        )
        st.caption(disclosure)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Distributions and scatter use at most 2,000 hash-selected finite observations, not the full population. "
        "Missing values are not zeros. No customer IDs are sent to these charts."
    )
    st.caption(
        "repeat_share begins on the second distinct product-purchase day; established_share begins on the third. "
        "Existing repeat-style rules use established_share."
    )


__all__ = ["render_features"]