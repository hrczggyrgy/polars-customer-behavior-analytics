"""Customer intelligence UI over the repository's existing Polars outputs.

Run from an installed checkout: streamlit run dashboard_v2.py
Set RETAIL_RESULTS or select a pipeline output directory in the sidebar.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from retail_analytics.visualization.visual_insights_core import (
    cohort_summary,
    concentration,
    segment_summary,
    transition_summary,
)

PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#000000",
]
BLUE, ORANGE, GREEN, RED = PALETTE[:4]
SCALE = [[0, "#FFFFFF"], [0.5, "#56B4E9"], [1, BLUE]]
PAGE_SIZE = 50


@dataclass(frozen=True)
class Filters:
    """Explicit reporting interval and end-snapshot population filters."""

    start: date
    end: date
    segments: tuple[str, ...]
    categories: tuple[str, ...]


def read_json(path: Path) -> dict[str, Any]:
    """Read optional pipeline metadata."""
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {}
    )


def fingerprint(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Invalidate cached results when parquet or metadata files change."""
    files = sorted(
        p
        for p in root.rglob("*")
        if p.suffix in {".parquet", ".json"}
    )
    return tuple(
        (
            str(p.relative_to(root)),
            p.stat().st_size,
            p.stat().st_mtime_ns,
        )
        for p in files
    )


@st.cache_data(show_spinner=False, max_entries=4)
def load_catalog(root_name: str, stamp: tuple) -> dict[str, Any]:
    """Discover result snapshots, categories and coverage metadata."""
    root = Path(root_name)

    if not (root / "validated.parquet").exists():
        raise ValueError(
            "Expected validated.parquet in the selected results directory."
        )

    snapshots = sorted(
        p.name
        for p in (root / "snapshots").glob("*")
        if p.is_dir() and list(p.glob("*.parquet"))
    )
    if not snapshots:
        raise ValueError(
            "No customer snapshots found. Run the repository pipeline first."
        )

    for name in snapshots:
        date.fromisoformat(name)

    tx = pl.scan_parquet(root / "validated.parquet")
    required = {
        "customer_id",
        "date",
        "value",
        "department",
        "category",
        "category_key",
    }
    missing = required - set(tx.collect_schema().names())
    if missing:
        raise ValueError(
            f"Validated transaction columns missing: {sorted(missing)}"
        )

    bounds = (
        tx.select(
            pl.col("date").min().alias("first"),
            pl.col("date").max().alias("last"),
        )
        .collect()
        .row(0, named=True)
    )
    if bounds["first"] is None:
        raise ValueError("Validated transactions are empty.")

    cats = (
        tx.select("category_key", "department", "category")
        .unique()
        .sort("department", "category")
        .collect()
    )

    quality = read_json(root / "quality.json")
    manifest = (
        read_json(root / "configuration.json")
        | read_json(root / "run.json")
    )
    coverage = (
        quality.get(
            "coverage_confirmed",
            manifest.get("coverage_confirmed", False),
        )
        is True
    )

    return dict(
        snapshots=snapshots,
        categories=cats,
        quality=quality,
        manifest=manifest,
        coverage=coverage,
        **bounds,
    )


@st.cache_data(show_spinner=False, max_entries=12)
def load_snapshot(
    root_name: str,
    cutoff: date,
    stamp: tuple,
) -> pl.DataFrame:
    """Load canonical customer features without reimplementing labels."""
    files = sorted(
        (Path(root_name) / "snapshots" / cutoff.isoformat())
        .glob("*.parquet")
    )
    if not files:
        raise ValueError(
            f"No snapshot at {cutoff}; "
            "generate that cutoff with retail-analytics."
        )

    frame = pl.scan_parquet(files).collect(engine="streaming")

    required = {
        "customer_id",
        "first_purchase",
        "recency",
        "behavior",
        "lifecycle",
        "spend_history",
        "spend_90d",
        "purchases_90d",
        "trajectory_tags",
        "style_eligible",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Snapshot columns missing: {sorted(missing)}"
        )

    if frame["customer_id"].n_unique() != frame.height:
        raise ValueError("Snapshot customer IDs are not unique.")

    if (
        "snapshot" in frame.columns
        and frame.filter(pl.col("snapshot") != cutoff).height
    ):
        raise ValueError(
            "Snapshot dates disagree with their directory."
        )

    return frame


def population(
    frame: pl.DataFrame,
    lines: pl.LazyFrame,
    f: Filters,
) -> pl.DataFrame:
    """Filter labels at this cutoff and category participation in its period."""
    if f.segments:
        frame = frame.filter(
            pl.col("behavior").is_in(f.segments)
        )

    if f.categories:
        ids = lines.filter(
            pl.col("date").is_between(f.start, f.end)
        )
        ids = (
            ids.filter(
                pl.col("category_key").is_in(f.categories)
            )
            .select("customer_id")
            .unique()
        )
        frame = frame.join(
            ids.collect(engine="streaming"),
            on="customer_id",
            how="semi",
        )

    return frame


def basket_scan(
    root_name: str,
    cutoff: date,
) -> pl.LazyFrame:
    """Reuse basket-grain event outputs from the canonical pipeline."""
    files = sorted(
        (Path(root_name) / "events" / cutoff.isoformat())
        .glob("*/baskets.parquet")
    )
    if not files:
        raise ValueError(
            f"Missing events/{cutoff}/*/baskets.parquet; "
            "regenerate pipeline outputs."
        )

    return pl.scan_parquet(files).filter(
        pl.col("date") <= cutoff
    )


def period_metrics(
    lines: pl.LazyFrame,
    rows: pl.DataFrame,
    f: Filters,
) -> dict[str, Any]:
    """Compute interval KPIs using canonical values and basket identity."""
    part = (
        lines.filter(
            pl.col("date").is_between(f.start, f.end)
        )
        .join(
            rows.select("customer_id").lazy(),
            on="customer_id",
            how="semi",
        )
    )

    if f.categories:
        part = part.filter(
            pl.col("category_key").is_in(f.categories)
        )

    basket = (
        part.group_by(
            "customer_id",
            "transaction_id",
            "date",
        )
        .agg(pl.col("value").sum().alias("spend"))
    )
    buyers = basket.group_by("customer_id").agg(
        pl.len().alias("orders")
    )

    result = (
        basket.select(
            pl.len().alias("orders"),
            pl.col("spend").sum().alias("revenue"),
        )
        .collect()
        .row(0, named=True)
    )
    counts = (
        buyers.select(
            pl.len().alias("customers"),
            (pl.col("orders") >= 2)
            .sum()
            .alias("repeaters"),
        )
        .collect()
        .row(0, named=True)
    )

    eligible = rows.filter(
        pl.col("style_eligible").fill_null(False)
    )
    overdue = eligible.filter(
        pl.col("trajectory_tags")
        .list.contains("overdue")
        .fill_null(False)
    ).height

    return result | counts | dict(
        aov=(
            result["revenue"] / result["orders"]
            if result["orders"]
            else None
        ),
        repeat=(
            counts["repeaters"] / counts["customers"]
            if counts["customers"]
            else None
        ),
        risk=(
            overdue / eligible.height
            if eligible.height
            else None
        ),
        eligible=eligible.height,
        overdue=overdue,
        population=rows.height,
    )


@st.cache_data(
    show_spinner="Computing customer views...",
    max_entries=8,
)
def analyze(
    root_name: str,
    stamp: tuple,
    f: Filters,
    coverage: bool,
) -> dict[str, Any]:
    """Cache filter-dependent aggregates; retain all eligible customer records."""
    root = Path(root_name)
    tx = pl.scan_parquet(
        root / "validated.parquet"
    ).filter(pl.col("date") <= f.end)

    rows = population(
        load_snapshot(root_name, f.end, stamp),
        tx,
        f,
    )
    metrics = period_metrics(tx, rows, f)

    if rows.is_empty():
        return dict(rows=rows, metrics=metrics)

    ids = rows.select("customer_id").lazy()
    baskets = basket_scan(root_name, f.end).join(
        ids,
        on="customer_id",
        how="semi",
    )

    line = (
        tx.join(ids, on="customer_id", how="semi")
        .filter(pl.col("date").is_between(f.start, f.end))
    )
    if f.categories:
        line = line.filter(
            pl.col("category_key").is_in(f.categories)
        )

    monthly = (
        line.group_by(
            pl.col("date")
            .dt.truncate("1mo")
            .alias("month")
        )
        .agg(pl.col("value").sum().alias("revenue"))
        .sort("month")
        .collect()
    )

    categories = (
        line.group_by(
            "category_key",
            "department",
            "category",
        )
        .agg(
            pl.col("value").sum().alias("revenue"),
            pl.col("customer_id")
            .n_unique()
            .alias("buyers"),
        )
        .collect()
        .with_columns(
            (pl.col("buyers") / rows.height)
            .alias("participation"),
            pl.when(pl.col("revenue").sum() > 0)
            .then(
                pl.col("revenue")
                / pl.col("revenue").sum()
            )
            .otherwise(None)
            .alias("revenue_share"),
            pl.concat_str(
                "department",
                pl.lit(" / "),
                "category",
            ).alias("label"),
        )
    )

    segments = segment_summary(
        rows.lazy(),
        "behavior",
        "spend_90d",
    )

    if {"new_share", "evidence"} <= set(rows.columns):
        profiles = rows.group_by("behavior").agg(
            pl.col("recency")
            .median()
            .alias("median_recency_days"),
            pl.col("purchases_90d")
            .median()
            .alias("median_baskets_90d"),
            pl.col("new_share")
            .mean()
            .alias("mean_new_product_share"),
            pl.col("evidence")
            .value_counts()
            .alias("evidence_counts"),
        )
    else:
        profiles = rows.select("behavior").unique()

    segments = segments.join(
        profiles,
        on="behavior",
        how="left",
    )

    conc = concentration(
        rows.select(
            pl.col("spend_history").alias("spend_90d")
        ).lazy()
    )

    ordered = (
        baskets.sort(
            "customer_id",
            "date",
            "transaction_id",
        )
        .with_columns(
            pl.col("date")
            .cum_count()
            .over("customer_id")
            .alias("order_number")
        )
    )
    second = (
        ordered.filter(pl.col("order_number") == 2)
        .select(
            "customer_id",
            pl.col("date").alias("second_date"),
        )
    )
    second = (
        rows.select(
            "customer_id",
            "first_purchase",
        )
        .lazy()
        .join(
            second,
            on="customer_id",
            how="left",
        )
        .with_columns(
            (
                pl.col("second_date")
                - pl.col("first_purchase")
            )
            .dt.total_days()
            .alias("days_to_second")
        )
        .collect()
    )

    repeat_summary = cohort_summary(
        baskets,
        rows.lazy(),
        f.end,
        coverage,
        minimum=20,
    )
    retention = monthly_retention(
        baskets,
        rows,
        f,
        coverage,
    )

    return dict(
        rows=rows,
        metrics=metrics,
        monthly=monthly,
        categories=categories,
        segments=segments,
        concentration=conc,
        second=second,
        repeats=repeat_summary,
        retention=retention,
    )


def month_offset(
    later: pl.Expr,
    earlier: pl.Expr,
) -> pl.Expr:
    """Calendar-month difference, not an elapsed-day approximation."""
    return (
        (later.dt.year() - earlier.dt.year()) * 12
        + later.dt.month()
        - earlier.dt.month()
    )


def monthly_retention(
    baskets: pl.LazyFrame,
    rows: pl.DataFrame,
    f: Filters,
    coverage: bool,
) -> pl.DataFrame:
    """Age-align monthly activity and mask incomplete observation months."""
    members = (
        rows.filter(
            pl.col("first_purchase")
            .is_between(f.start, f.end)
        )
        .select(
            "customer_id",
            pl.col("first_purchase")
            .dt.truncate("1mo")
            .alias("cohort"),
        )
    )
    sizes = (
        members.group_by("cohort")
        .agg(pl.len().alias("cohort_n"))
        .sort("cohort")
        .tail(24)
    )

    if sizes.is_empty():
        return pl.DataFrame()

    activity = (
        baskets.join(
            members.lazy(),
            on="customer_id",
            how="inner",
        )
        .with_columns(
            month_offset(
                pl.col("date"),
                pl.col("cohort"),
            ).alias("age")
        )
    )
    counts = (
        activity.group_by("cohort", "age")
        .agg(
            pl.col("customer_id")
            .n_unique()
            .alias("active_n")
        )
        .collect()
    )

    grid = sizes.join(
        pl.DataFrame({"age": list(range(13))}),
        how="cross",
    )
    grid = (
        grid.join(
            counts,
            on=["cohort", "age"],
            how="left",
        )
        .with_columns(
            pl.col("active_n").fill_null(0),
            pl.col("cohort")
            .dt.offset_by("1mo")
            .dt.offset_by("-1d")
            .alias("cohort_end"),
        )
    )

    # A cohort-age cell is observable only after that entire calendar month ends.
    grid = grid.with_columns(
        pl.col("cohort")
        .dt.offset_by(
            pl.concat_str(
                pl.col("age") + 1,
                pl.lit("mo"),
            )
        )
        .dt.offset_by("-1d")
        .alias("activity_end")
    )

    return grid.with_columns(
        pl.when(
            pl.lit(coverage)
            & (pl.col("cohort_n") >= 20)
            & (pl.col("activity_end") <= f.end)
        )
        .then(
            pl.col("active_n")
            / pl.col("cohort_n")
        )
        .otherwise(None)
        .alias("rate")
    )


def colors(labels: list[str]) -> dict[str, str]:
    """Keep label colors stable across a reporting session."""
    return {
        name: PALETTE[i % len(PALETTE)]
        for i, name in enumerate(sorted(set(labels)))
    }


def show(
    fig: go.Figure,
    title: str,
    x: str,
    y: str,
    takeaway: str,
    key: str,
) -> None:
    """Apply one visual theme and attach an interpretive chart caption."""
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=430,
        colorway=PALETTE,
        margin=dict(l=20, r=20, t=70, b=60),
        xaxis_title=x,
        yaxis_title=y,
        font=dict(size=13),
        legend=dict(orientation="h", y=-0.25),
        hovermode="closest",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={"displaylogo": False},
    )
    st.caption(takeaway)


def fmt(
    value: float | int | None,
    percent: bool = False,
) -> str:
    """Render unavailable values separately from genuine zeros."""
    if value is None:
        return "N/A"
    return (
        f"{value:.1%}"
        if percent
        else f"{value:,.2f}"
    )


def delta(
    current: float | None,
    previous: float | None,
    rate: bool = False,
) -> str | None:
    """Use percentage-point deltas for rates and relative deltas otherwise."""
    if current is None or previous is None:
        return None

    if rate:
        return f"{(current - previous) * 100:+.1f} pp"

    return (
        f"{(current / previous - 1) * 100:+.1f}%"
        if previous != 0
        else None
    )


def overview(
    data: dict[str, Any],
    previous: dict[str, Any] | None,
    f: Filters,
) -> None:
    """Headline interval KPIs and full-population monthly revenue."""
    m, old = data["metrics"], previous or {}

    specs = [
        ("customers", "Purchasing customers", False),
        ("revenue", "Recorded revenue", False),
        ("aov", "AOV", False),
        ("repeat", "Repeat-basket rate", True),
        ("risk", "Overdue churn proxy", True),
    ]

    for box, (key, label, rate) in zip(
        st.columns(5),
        specs,
    ):
        value = (
            str(m[key])
            if key == "customers"
            else fmt(m[key], rate)
        )
        box.metric(
            label,
            value,
            delta(m[key], old.get(key), rate),
            delta_color=(
                "inverse"
                if key == "risk"
                else "normal"
            ),
        )

    st.caption(
        "Customer KPI counts interval buyers; "
        f"{m['population']:,} observed customers are in "
        "the snapshot population. Repeat rate = buyers "
        "with at least two qualifying baskets / interval "
        f"buyers. Overdue = {m['overdue']:,} / "
        f"{m['eligible']:,} style-eligible customers at "
        "the cutoff, not predicted churn."
    )

    if f.categories:
        st.caption(
            "Category AOV uses selected-category line value "
            "per qualifying basket, not the full-basket amount."
        )

    monthly = data["monthly"]
    if monthly.is_empty():
        st.info(
            "No purchases in this reporting interval; "
            "historical customer views remain available."
        )
        return

    calendar = pl.DataFrame(
        {
            "month": pl.date_range(
                f.start.replace(day=1),
                f.end.replace(day=1),
                interval="1mo",
                eager=True,
            )
        }
    )
    monthly = (
        calendar.join(
            monthly,
            on="month",
            how="left",
        )
        .with_columns(pl.col("revenue").fill_null(0))
    )
    peak = (
        monthly.sort("revenue", descending=True)
        .row(0, named=True)
    )

    fig = go.Figure(
        go.Scatter(
            x=monthly["month"].to_list(),
            y=monthly["revenue"].to_list(),
            mode="lines+markers",
            line=dict(color=BLUE, width=3),
            hovertemplate=(
                "Month %{x|%Y-%m}<br>"
                "Recorded revenue %{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )
    show(
        fig,
        f"{peak['month']:%b %Y} has the highest "
        "recorded revenue in this interval",
        "Calendar month",
        "Recorded revenue (input currency units)",
        "Monthly trend reveals timing, not proven recurring "
        "seasonality; boundary months may be partial and zero "
        "observed activity does not establish complete capture.",
        "monthly",
    )


def segmentation(
    data: dict[str, Any],
    cmap: dict[str, str],
) -> None:
    """Compare existing behavior labels without inventing clusters."""
    table = data["segments"]
    fig = go.Figure()

    for r in table.to_dicts():
        fig.add_trace(
            go.Scatter(
                x=[r["customer_share"]],
                y=[r["spend_share"]],
                mode="markers",
                name=r["behavior"],
                marker=dict(
                    color=cmap[r["behavior"]],
                    size=(
                        12
                        + 48 * math.sqrt(r["customer_share"])
                    ),
                ),
                customdata=[
                    [
                        r["customers"],
                        r["median"],
                        r["active_share"],
                    ]
                ],
                hovertemplate=(
                    "%{fullData.name}<br>"
                    "Customer share %{x:.1%}<br>"
                    "90d revenue share %{y:.1%}<br>"
                    "Customers %{customdata[0]}<br>"
                    "Median 90d spend %{customdata[1]:,.2f}<br>"
                    "Active %{customdata[2]:.1%}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Equal customer and revenue share",
            line=dict(color="#000000", dash="dot"),
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(
        tickformat=".0%",
        range=[0, 1],
    )
    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
    )

    leader = (
        table.sort("spend", descending=True)
        .row(0, named=True)
    )
    show(
        fig,
        f"{leader['behavior']} contributes the most "
        "trailing-90-day revenue",
        "Share of filtered customers",
        "Share of trailing-90-day revenue",
        "Segments above the equality line contribute "
        "disproportionate value; marker size encodes segment "
        "size. Labels and 90-day windows are inherited "
        "unchanged from the pipeline.",
        "segments",
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Profiles include unavailable-feature counts and "
        "history exposure. Multi-tag behavior labels are "
        "treated as whole, non-overlapping combinations."
    )


def lifecycle(
    data: dict[str, Any],
    f: Filters,
    coverage: bool,
) -> None:
    """Separate observed retention, incomplete follow-up and overdue risk."""
    r = data["retention"]

    if not coverage:
        st.info(
            "Retention estimates are unavailable because "
            "capture coverage was not explicitly confirmed "
            "by the pipeline run."
        )
    elif (
        not r.is_empty()
        and r["rate"].is_not_null().any()
    ):
        cohorts = (
            r["cohort"]
            .unique()
            .sort()
            .to_list()
        )
        lookup = {
            (v["cohort"], v["age"]): v
            for v in r.to_dicts()
        }
        z = [
            [
                lookup[(c, a)]["rate"]
                for a in range(13)
            ]
            for c in cohorts
        ]
        n = [
            [
                lookup[(c, a)]["cohort_n"]
                for a in range(13)
            ]
            for c in cohorts
        ]

        fig = go.Figure(
            go.Heatmap(
                x=list(range(13)),
                y=[
                    c.strftime("%Y-%m")
                    for c in cohorts
                ],
                z=z,
                customdata=n,
                colorscale=SCALE,
                zmin=0,
                zmax=1,
                hoverongaps=False,
                colorbar=dict(tickformat=".0%"),
                hovertemplate=(
                    "First-observed cohort %{y}<br>"
                    "Calendar age %{x} months<br>"
                    "Active %{z:.1%}<br>"
                    "Cohort customers %{customdata}"
                    "<extra></extra>"
                ),
            )
        )
        show(
            fig,
            "Age-aligned cohorts separate retention "
            "from unequal tenure",
            "Months since first-observed calendar month",
            "First-observed cohort",
            "Cells show any purchase in that age-month / "
            "fixed cohort size; blanks are incomplete months "
            "or cohorts below 20 customers. Latest 24 cohorts, "
            "ages 0–12. First-observed dates are not verified "
            "acquisition dates; current segment/category "
            "membership makes this a retrospective population view.",
            "retention",
        )
    else:
        st.info(
            "No completed cohort-month cells meet the "
            "20-customer minimum in this interval."
        )

    with st.expander(
        "Repository age-aligned repeat rates: 30 / 60 / 90 days"
    ):
        st.dataframe(
            data["repeats"],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Reuses cohort_summary: repeat on a later shopping "
            "day, coverage required, minimum 20 eligible "
            "customers, Wilson 95% intervals. This table covers "
            "all first-observed cohorts in the filtered population."
        )

    second = data["second"].filter(
        pl.col("first_purchase")
        .is_between(f.start, f.end)
    )
    observed = second.filter(
        pl.col("days_to_second").is_not_null()
    )

    if observed.height:
        fig = go.Figure(
            go.Histogram(
                x=observed["days_to_second"].to_list(),
                nbinsx=30,
                marker_color=BLUE,
                hovertemplate=(
                    "Days %{x}<br>"
                    "Customers %{y}"
                    "<extra></extra>"
                ),
            )
        )
        median = observed["days_to_second"].median()
        show(
            fig,
            "Observed repeaters reach a second basket "
            f"in a median {median:g} days",
            "Days from first to second observed basket",
            "Observed repeaters",
            f"Conditional distribution: {observed.height:,} "
            f"observed repeaters out of {second.height:,} "
            "first-observed customers in the date range; "
            "non-repeaters have incomplete or censored outcomes, "
            "not zero wait. A distinct second basket on the "
            "same day counts as day zero.",
            "second",
        )
    else:
        st.info(
            "No observed second baskets for customers "
            "first seen in this date range."
        )

    rows = data["rows"]
    risk = (
        rows.filter(
            pl.col("style_eligible").fill_null(False)
        )
        .with_columns(
            pl.col("trajectory_tags")
            .list.contains("overdue")
            .fill_null(False)
            .alias("is_overdue")
        )
    )

    if risk.height and "lateness" in risk.columns:
        risk = (
            risk.filter(pl.col("lateness").is_finite())
            .sort(pl.col("customer_id").hash(seed=29))
            .head(3000)
        )
        fig = go.Figure()

        for state, name, color in [
            (False, "Not overdue", BLUE),
            (True, "Overdue proxy", RED),
        ]:
            p = risk.filter(
                pl.col("is_overdue") == state
            )
            fig.add_trace(
                go.Scattergl(
                    x=p["lateness"].to_list(),
                    y=p["spend_90d"].to_list(),
                    mode="markers",
                    name=name,
                    marker=dict(
                        color=color,
                        size=7,
                        opacity=0.55,
                    ),
                    hovertemplate=(
                        "Lateness %{x:.2f}x<br>"
                        "90d recorded spend %{y:,.2f}"
                        "<extra></extra>"
                    ),
                )
            )

        fig.add_vline(
            x=2,
            line_dash="dot",
            line_color=RED,
        )
        show(
            fig,
            "Overdue customers with high recorded value "
            "deserve investigation",
            "Recency / typical shopping-day gap",
            "Trailing-90-day spend (currency units)",
            "Uses the pipeline's overdue tag, not a probability "
            "or causal recommendation. At most 3,000 deterministic "
            "hash-selected eligible customers; unknown rhythm "
            "is excluded. High spend reflects past value, "
            "not future loss.",
            "risk",
        )


def value_section(data: dict[str, Any]) -> None:
    """Observed lifetime revenue and the canonical descending Pareto curve."""
    rows = data["rows"]
    c = data["concentration"]

    sample = (
        rows.sort(
            pl.col("customer_id").hash(seed=29)
        )
        .head(5000)
    )

    fig = go.Figure(
        go.Histogram(
            x=sample["spend_history"].to_list(),
            nbinsx=40,
            marker_color=GREEN,
            hovertemplate=(
                "Observed historical spend %{x:,.2f}<br>"
                "Sample customers %{y}"
                "<extra></extra>"
            ),
        )
    )
    median = rows["spend_history"].median()

    show(
        fig,
        "Median observed customer value is "
        f"{median:,.2f} currency units",
        "All recorded spend through cutoff (currency units)",
        "Sample customers",
        "Observed revenue-to-date is the supported lifetime-value "
        "proxy, not forecast CLV or profit. Median uses all "
        "customers; histogram uses at most 5,000 deterministic "
        "observations. Unequal tenure and left-censoring "
        "affect comparisons.",
        "value_hist",
    )

    if not c["curve"]:
        st.info(
            "A revenue-share curve is undefined because "
            "total historical spend is zero."
        )
        return

    x, y = zip(*c["curve"])
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Descending-spend Pareto",
            line=dict(color=BLUE, width=3),
            hovertemplate=(
                "Top %{x:.1%} customers<br>"
                "Historical revenue %{y:.1%}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Equal distribution",
            line=dict(color="#000000", dash="dot"),
            hoverinfo="skip",
        )
    )

    for t in c["top"]:
        fig.add_trace(
            go.Scatter(
                x=[t["actual_fraction"]],
                y=[t["spend_share"]],
                mode="markers",
                name=f"Top {t['requested_fraction']:.0%}",
                marker=dict(color=ORANGE, size=9),
                hovertemplate=(
                    "Actual customer share %{x:.1%}<br>"
                    "Historical revenue share %{y:.1%}"
                    "<extra></extra>"
                ),
            )
        )

    top = c["top"][-1]
    fig.update_xaxes(
        tickformat=".0%",
        range=[0, 1],
    )
    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
    )

    show(
        fig,
        "Top 20% of customers generate "
        f"{top['spend_share']:.1%} of historical revenue",
        "Cumulative customer share, highest spend first",
        "Cumulative historical revenue share",
        "Uses the repository concentration function on "
        f"spend_history; all {c['n']:,} filtered customers "
        "are included. Top fractions round up to whole "
        "customers. This is Pareto, not an ascending Lorenz curve.",
        "pareto",
    )
    st.dataframe(
        pl.DataFrame(c["top"]),
        use_container_width=True,
        hide_index=True,
    )


def behavior(
    data: dict[str, Any],
    cmap: dict[str, str],
) -> None:
    """Joint frequency/recency patterns and selected-interval category affinity."""
    rows = data["rows"]
    sample = (
        rows.sort(
            pl.col("customer_id").hash(seed=29)
        )
        .head(3000)
    )
    fig = go.Figure()

    for name in sorted(
        sample["behavior"].unique().to_list()
    ):
        p = sample.filter(
            pl.col("behavior") == name
        )
        fig.add_trace(
            go.Scattergl(
                x=p["recency"].to_list(),
                y=p["purchases_90d"].to_list(),
                mode="markers",
                name=name,
                marker=dict(
                    color=cmap[name],
                    size=6,
                    opacity=0.5,
                ),
                customdata=p["spend_90d"].to_list(),
                hovertemplate=(
                    "%{fullData.name}<br>"
                    "Recency %{x} days<br>"
                    "90d baskets %{y}<br>"
                    "90d spend %{customdata:,.2f}"
                    "<extra></extra>"
                ),
            )
        )

    show(
        fig,
        "Recency and purchase frequency distinguish "
        "active from dormant behavior",
        "Days since last observed purchase",
        "Baskets in trailing 90 days",
        "Up to 3,000 deterministic customers; windows and "
        "features are unchanged pipeline outputs. Categories "
        "filter customer membership here, not customer-wide "
        "recency or frequency.",
        "recency",
    )

    cat = (
        data["categories"]
        .filter(pl.col("revenue_share").is_not_null())
        .sort("revenue", descending=True)
    )
    if cat.is_empty():
        st.info(
            "No category revenue in the reporting interval."
        )
        return

    shown = cat.head(20).reverse()
    fig = go.Figure()

    for col, name, color in [
        ("participation", "Customer participation", BLUE),
        ("revenue_share", "Revenue share", ORANGE),
    ]:
        fig.add_trace(
            go.Scatter(
                x=shown[col].to_list(),
                y=shown["label"].to_list(),
                mode="markers",
                name=name,
                marker=dict(color=color, size=10),
                customdata=list(
                    zip(
                        shown["buyers"].to_list(),
                        shown["revenue"].to_list(),
                    )
                ),
                hovertemplate=(
                    "%{y}<br>"
                    "%{fullData.name}: %{x:.1%}<br>"
                    "Buyers %{customdata[0]}<br>"
                    "Revenue %{customdata[1]:,.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(tickformat=".0%")
    show(
        fig,
        f"{cat['label'][0]} leads category revenue "
        "in the selected interval",
        "Share of customers or recorded revenue",
        "Department / category",
        "Paired dots compare reach with monetary contribution; "
        "participation uses the entire filtered snapshot "
        "population and can sum above 100%. Top 20 categories "
        "by revenue; all categories remain in the table. "
        "This is descriptive affinity, not product-pair "
        "association mining.",
        "affinity",
    )
    st.dataframe(
        cat,
        use_container_width=True,
        hide_index=True,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def migration(
    root_name: str,
    stamp: tuple,
    origin: date,
    end: date,
    ids: tuple[str, ...],
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Reuse matched-customer lifecycle transitions for the selected population."""
    before = load_snapshot(
        root_name,
        origin,
        stamp,
    ).filter(pl.col("customer_id").is_in(ids))

    after = load_snapshot(
        root_name,
        end,
        stamp,
    ).filter(pl.col("customer_id").is_in(ids))

    return transition_summary(
        before.lazy(),
        after.lazy(),
        "lifecycle",
    )


def migration_section(
    root_name: str,
    stamp: tuple,
    snapshots: list[str],
    data: dict[str, Any],
    end: date,
) -> None:
    """Show all matched lifecycle flows, preserving new-customer reconciliation."""
    earlier = [
        d
        for d in snapshots
        if d < end.isoformat()
    ]
    if not earlier:
        st.info(
            "Lifecycle migration requires an earlier "
            "pipeline snapshot."
        )
        return

    origin = date.fromisoformat(
        st.selectbox(
            "Migration origin",
            earlier,
            index=len(earlier) - 1,
        )
    )
    flows, counts = migration(
        root_name,
        stamp,
        origin,
        end,
        tuple(data["rows"]["customer_id"].to_list()),
    )
    if flows.is_empty():
        st.info(
            "No selected customers appear in both snapshots."
        )
        return

    left = sorted(
        flows["from_label"].unique().to_list()
    )
    right = sorted(
        flows["to_label"].unique().to_list()
    )
    cmap = colors(left + right)

    fig = go.Figure(
        go.Sankey(
            node=dict(
                label=(
                    [f"Then: {s}" for s in left]
                    + [f"Now: {s}" for s in right]
                ),
                color=[
                    cmap[s]
                    for s in left + right
                ],
                pad=18,
                thickness=18,
            ),
            link=dict(
                source=[
                    left.index(s)
                    for s in flows["from_label"]
                ],
                target=[
                    len(left) + right.index(s)
                    for s in flows["to_label"]
                ],
                value=flows["len"].to_list(),
                customdata=flows["rate"].to_list(),
                color="rgba(0,114,178,0.25)",
                hovertemplate=(
                    "%{source.label} → %{target.label}<br>"
                    "Customers %{value}<br>"
                    "Share of matched origin %{customdata:.1%}"
                    "<extra></extra>"
                ),
            ),
        )
    )

    show(
        fig,
        f"{counts['matched_n']:,} customers connect "
        "the two lifecycle snapshots",
        "",
        "",
        "Flow width encodes matched customer count; "
        f"{counts['destination_unmatched']:,} selected "
        "end-snapshot customers lack an origin record "
        "and are excluded from flows. This follows the "
        "current filtered population backward, not a "
        "prospective acquisition funnel.",
        "migration",
    )


def safe_csv(frame: pl.DataFrame) -> bytes:
    """Flatten nested columns and escape spreadsheet-formula prefixes."""
    cols = [
        c
        for c, dtype in frame.schema.items()
        if not dtype.is_nested()
    ]
    result = frame.select(cols)
    strings = [
        c
        for c, dtype in result.schema.items()
        if dtype == pl.String
    ]

    result = result.with_columns(
        [
            pl.when(
                pl.col(c).str.contains(r"^[=+\-@\t\r]")
            )
            .then(
                pl.concat_str(
                    pl.lit("'"),
                    pl.col(c),
                )
            )
            .otherwise(pl.col(c))
            .alias(c)
            for c in strings
        ]
    )
    return result.write_csv().encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=20)
def customer_history(
    root_name: str,
    stamp: tuple,
    customer_id: str,
    f: Filters,
) -> pl.DataFrame:
    """Retrieve only the customer's globally filtered basket history."""
    line = pl.scan_parquet(
        Path(root_name) / "validated.parquet"
    ).filter(
        (pl.col("customer_id") == customer_id)
        & pl.col("date").is_between(f.start, f.end)
    )
    if f.categories:
        line = line.filter(
            pl.col("category_key").is_in(f.categories)
        )

    return (
        line.group_by("date", "transaction_id")
        .agg(
            pl.col("value")
            .sum()
            .alias("recorded_spend"),
            pl.col("product_id")
            .n_unique()
            .alias("products"),
            pl.col("category_key")
            .n_unique()
            .alias("categories"),
        )
        .sort("date", "transaction_id")
        .collect()
    )


def explorer(
    root_name: str,
    stamp: tuple,
    data: dict[str, Any],
    f: Filters,
) -> None:
    """Paginated segment/customer search and a cutoff-consistent drill-down."""
    rows = data["rows"]
    query = st.text_input(
        "Customer ID contains",
        key="customer_query",
    )
    selected = (
        rows.filter(
            pl.col("customer_id")
            .str.contains(query, literal=True)
        )
        if query
        else rows
    )

    order = st.selectbox(
        "Sort customers by",
        ["spend_history", "spend_90d", "recency"],
    )
    selected = selected.sort(
        [order, "customer_id"],
        descending=[True, False],
    )

    if selected.is_empty():
        st.info(
            "No customer IDs match this search."
        )
        return

    pages = math.ceil(selected.height / PAGE_SIZE)
    page = int(
        st.number_input(
            "Page",
            min_value=1,
            max_value=pages,
            value=1,
            step=1,
        )
    )
    shown = selected.slice(
        (page - 1) * PAGE_SIZE,
        PAGE_SIZE,
    )
    columns = [
        c
        for c in [
            "customer_id",
            "behavior",
            "lifecycle",
            "value_tier",
            "evidence",
            "recency",
            "purchases_90d",
            "spend_90d",
            "spend_history",
            "trajectory",
            "left_censored",
        ]
        if c in rows.columns
    ]

    st.dataframe(
        shown.select(columns),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Showing page {page} / {pages}; "
        f"{selected.height:,} matches. Snapshot features "
        "retain their documented historical windows."
    )
    st.download_button(
        "Export this customer page",
        safe_csv(shown.select(columns)),
        "customer_page.csv",
        "text/csv",
    )

    chosen = st.selectbox(
        "Inspect a customer on this page",
        shown["customer_id"].to_list(),
    )
    record = shown.filter(
        pl.col("customer_id") == chosen
    )

    with st.expander("All canonical customer features"):
        st.dataframe(
            record,
            use_container_width=True,
            hide_index=True,
        )

    history = customer_history(
        root_name,
        stamp,
        chosen,
        f,
    )
    if history.is_empty():
        st.info(
            "This customer has no qualifying baskets "
            "in the date/category interval."
        )
        return

    fig = go.Figure(
        go.Scatter(
            x=history["date"].to_list(),
            y=history["recorded_spend"].to_list(),
            mode="markers+lines",
            marker=dict(color=BLUE, size=9),
            customdata=history["transaction_id"].to_list(),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "Basket %{customdata}<br>"
                "Recorded spend %{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )
    show(
        fig,
        "Basket history makes individual purchase "
        "patterns inspectable",
        "Purchase date",
        "Qualifying basket spend (currency units)",
        "Only the global date/category selection is shown; "
        "connecting lines indicate sequence, not purchases "
        "between observations. With categories selected, "
        "amounts are selected-category contributions.",
        "customer_history",
    )
    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """Configure a read-only dashboard over existing pipeline outputs."""
    st.set_page_config(
        page_title="Customer intelligence",
        layout="wide",
    )
    st.title("Customer intelligence")
    st.caption(
        "Who customers are behaviorally, how they purchase, "
        "observed value and signs of dormancy."
    )

    root = Path(
        st.sidebar.text_input(
            "Pipeline results directory",
            os.getenv("RETAIL_RESULTS", "results"),
        )
    ).expanduser().resolve()

    if not root.is_dir():
        st.info(
            "Select a pipeline results directory. This app "
            "reads existing outputs; it does not generate "
            "demo data or rerun the pipeline."
        )
        return

    try:
        stamp = fingerprint(root)
        catalog = load_catalog(str(root), stamp)

        end = date.fromisoformat(
            st.sidebar.selectbox(
                "End date / feature snapshot",
                catalog["snapshots"],
                index=len(catalog["snapshots"]) - 1,
            )
        )
        first = catalog["first"]
        if end < first:
            raise ValueError(
                "Selected snapshot precedes all observed transactions."
            )

        start = st.sidebar.date_input(
            "Start date",
            value=max(
                first,
                end - timedelta(days=89),
            ),
            min_value=first,
            max_value=end,
        )
        if not isinstance(start, date):
            raise ValueError("Choose one start date.")

        raw = load_snapshot(str(root), end, stamp)
        labels = sorted(
            raw["behavior"]
            .drop_nulls()
            .unique()
            .to_list()
        )
        segments = tuple(
            st.sidebar.multiselect(
                "Behavior segment (empty = all)",
                labels,
            )
        )

        cats = catalog["categories"]
        names = {
            r["category_key"]: (
                f"{r['department']} / {r['category']} "
                f"[{r['category_key']}]"
            )
            for r in cats.to_dicts()
        }
        categories = tuple(
            st.sidebar.multiselect(
                "Category (empty = all)",
                list(names),
                format_func=lambda k: names[k],
            )
        )

        f = Filters(start, end, segments, categories)

        st.sidebar.caption(
            "No geographic or demographic fields exist "
            "in the canonical schema. Segment and category "
            "membership are evaluated at the selected "
            "cutoff/interval."
        )

        data = analyze(
            str(root),
            stamp,
            f,
            catalog["coverage"],
        )
        if data["rows"].is_empty():
            st.warning(
                "No customers match these filters. Clear "
                "a segment/category selection or broaden "
                "the date range."
            )
            return

        days = (end - start).days + 1
        previous_end = start - timedelta(days=1)
        previous_start = (
            previous_end - timedelta(days=days - 1)
        )
        previous: dict[str, Any] | None = None

        if (
            previous_start >= first
            and catalog["coverage"]
        ):
            prev_filter = Filters(
                previous_start,
                previous_end,
                segments,
                categories,
            )

            if not segments and not categories:
                # Prior buyers cannot first appear after the prior cutoff;
                # labels are not used here.
                prior_rows = raw.filter(
                    pl.col("first_purchase") <= previous_end
                )
                previous = period_metrics(
                    pl.scan_parquet(
                        root / "validated.parquet"
                    ),
                    prior_rows,
                    prev_filter,
                )
                previous["risk"] = None

            elif (
                previous_end.isoformat()
                in catalog["snapshots"]
            ):
                prior_rows = population(
                    load_snapshot(
                        str(root),
                        previous_end,
                        stamp,
                    ),
                    pl.scan_parquet(
                        root / "validated.parquet"
                    ),
                    prev_filter,
                )
                previous = period_metrics(
                    pl.scan_parquet(
                        root / "validated.parquet"
                    ),
                    prior_rows,
                    prev_filter,
                )

            if (
                previous_end.isoformat()
                in catalog["snapshots"]
                and previous is not None
            ):
                prior_rows = population(
                    load_snapshot(
                        str(root),
                        previous_end,
                        stamp,
                    ),
                    pl.scan_parquet(
                        root / "validated.parquet"
                    ),
                    prev_filter,
                )
                previous["risk"] = period_metrics(
                    pl.scan_parquet(
                        root / "validated.parquet"
                    ),
                    prior_rows,
                    prev_filter,
                )["risk"]

        st.caption(
            f"Reporting {start} through {end}, inclusive. "
            f"Previous comparison: {previous_start} through "
            f"{previous_end} ({days} days). Missing deltas "
            "mean insufficient capture/history, a zero baseline, "
            "or no exact prior snapshot for label-dependent "
            "metrics; rates use percentage-point changes. "
            "Populations are evaluated separately, "
            "not a fixed customer panel."
        )

        if previous is None:
            st.info(
                "Previous-period deltas are unavailable: "
                "complete history and confirmed capture are "
                "required; segment/category comparisons also "
                "require a snapshot at the exact "
                "previous-period end."
            )

        view = st.radio(
            "View",
            [
                "Overview",
                "Segmentation",
                "Lifecycle",
                "Value",
                "Behavior",
                "Explorer",
                "Quality and methods",
            ],
            horizontal=True,
        )
        cmap = colors(labels)

        if view == "Overview":
            overview(data, previous, f)

        elif view == "Segmentation":
            segmentation(data, cmap)
            migration_section(
                str(root),
                stamp,
                catalog["snapshots"],
                data,
                end,
            )

        elif view == "Lifecycle":
            lifecycle(
                data,
                f,
                catalog["coverage"],
            )

        elif view == "Value":
            value_section(data)

        elif view == "Behavior":
            behavior(data, cmap)

        elif view == "Explorer":
            explorer(
                str(root),
                stamp,
                data,
                f,
            )

        else:
            st.subheader("Run metadata and quality")
            st.json(catalog["quality"])
            st.json(catalog["manifest"])

            st.write(
                "Purchase-only revenue; returns, margin, "
                "currency conversion and predictive CLV are "
                "not supported. Invalid rows are excluded "
                "upstream according to the run configuration. "
                "Duplicate handling follows the pipeline run."
            )
            st.write(
                "Interval start filters sales, category "
                "participation and first-observed lifecycle "
                "cohorts. End date anchors all snapshot features. "
                "Value, segments, risk and recency retain their "
                "documented historical/rolling windows and are "
                "not recalculated inside the selected interval."
            )
            st.write(
                "Monthly retention and repeat-rate inference "
                "require explicitly confirmed capture. Cohorts "
                "and migration are retrospective when filtered "
                "by current membership. First-observed cohorts "
                "may be left-censored; second-basket waiting "
                "times are conditional on an observed repeat "
                "and are not survival estimates."
            )
            st.write(
                "Chart rationale: bubble = segment size versus "
                "value; heatmap = age-aligned cohort activity; "
                "Sankey = matched lifecycle migration; "
                "Pareto = concentration; paired dots = category "
                "reach versus value; scatter = frequency/recency "
                "or risk/value; time series = calendar timing."
            )
            st.write(
                "All transformations use Polars. Plotly receives "
                "Python lists; no pandas dependency or conversion "
                "is used. Large point charts use disclosed "
                "deterministic samples; aggregate tables and "
                "Pareto use all filtered records."
            )

    except (
        ValueError,
        OSError,
        KeyError,
        pl.exceptions.PolarsError,
    ) as exc:
        st.error(
            f"Cannot render this result set: {exc}"
        )
        st.caption(
            "Check that the directory contains "
            "validated.parquet, snapshots/<date>/*.parquet "
            "and events/<date>/*/baskets.parquet "
            "from one completed pipeline run."
        )


if __name__ == "__main__":
    main()