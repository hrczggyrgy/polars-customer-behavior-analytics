"""Bounded, coverage-aware visual analytics for the shared Polars pipeline."""

import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import date, timedelta
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from retail_analytics.core.retail_polars_workflow import COLS
from retail_analytics.core.stats import wilson
from retail_analytics.dashboard.dashboard_backend import cutoffs
from retail_analytics.visualization.visual_insights_core import (
    METRICS,
    cohort_summary,
    collect,
    concentration,
    transition_summary,
    momentum_map,
    cadence_map,
    category_participation_comparison,
    trial_to_repeat_map,
    revenue_waterfall,
    value_tier_transition_waterfall,
    behavior_trajectory_matrix,
    purchase_rhythm_panel,
    basket_composition_slope,
    category_sankey,
    future_outcome_validation,
)
from retail_analytics.core.constants import (
    QUANTILE_VALUE_LOW,
    QUANTILE_VALUE_HIGH,
    QUANTILE_STYLE_LOW,
    QUANTILE_STYLE_HIGH,
)

MAX_BYTES = 100 * 1024 * 1024
PAGE_ROWS = 2000
TIMEOUT = 600
FEATURES = [
    "spend_90d",
    "purchases_90d",
    "recency",
    "median_basket",
    "median_gap",
    "gap_cv",
    "lateness",
    "new_share",
    "repeat_share",
    "established_share",
    "trial_conversion",
    "category_hhi",
    "products_last6days",
    "categories_last6days",
    "frequency_change",
    "basket_change",
]
DIMS = ["behavior", "trajectory", "lifecycle", "value_tier", "evidence"]

# Colorblind-safe Okabe-Ito palette (8 colors)
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]


def build_color_map(labels):
    """Build deterministic color map from sorted labels using Okabe-Ito palette."""
    sorted_labels = sorted(set(str(label) for label in labels))
    return {label: OKABE_ITO[i % len(OKABE_ITO)] for i, label in enumerate(sorted_labels)}


def get_color(label, color_map):
    """Get color for label from color map, fallback to hash."""
    return color_map.get(str(label), "#64748b")


def format_axis_title(metric_name):
    """Convert metric column name to human-readable title with units."""
    if metric_name in METRICS:
        return f"{METRICS[metric_name][0]} ({METRICS[metric_name][1]})"
    # Fallback: title case with underscores replaced
    return metric_name.replace("_", " ").title()


def bounded_grid(records, row, column, value, limit=24):
    if not 1 <= limit <= 36:
        raise ValueError("Grid limit must be 1..36")
    ys = sorted({str(r[row]) for r in records})[-limit:]
    xs = sorted({str(r[column]) for r in records})[-limit:]
    lookup = {(str(r[row]), str(r[column])): r[value] for r in records}
    return xs, ys, [[lookup.get((y, x)) for x in xs] for y in ys]


def read_json(path):
    return json.loads(Path(path).read_text())


def reset():
    workspace = st.session_state.pop("workspace", None)
    if workspace:
        workspace.cleanup()
    for key in [
        "fingerprint",
        "inspection_key",
        "result",
        "result_key",
        "preview_schema",
        "preview_key",
        "column_mapping",
    ]:
        st.session_state.pop(key, None)


def drop_result():
    old = st.session_state.pop("result", None)
    st.session_state.pop("result_key", None)
    if old and "workspace" in st.session_state:
        parent = Path(st.session_state["workspace"].name).resolve()
        target = Path(old).resolve()
        if target.parent == parent and target.name.startswith("run_"):
            shutil.rmtree(target, ignore_errors=True)


def worker(job):
    root = Path(job["workspace"])
    control = root / "job.json"
    control.write_text(json.dumps(job))
    status = root / "worker_status.json"
    status.unlink(missing_ok=True)
    try:
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("dashboard_backend.py")),
                "--job",
                str(control),
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise ValueError("Analysis timed out. Use a smaller upload or fewer cutoffs.")
    if not status.exists():
        raise ValueError("Worker stopped unexpectedly; no result was accepted.")
    report = read_json(status)
    if process.returncode or not report["ok"]:
        message = (
            report.get("message", "Worker failed")
            if report.get("error_type") in ["ValueError", "AssertionError"]
            else "Processing failed. Check formats, dependencies and server resources."
        )
        raise ValueError(message)


def table(frame):
    data = (
        collect(frame.limit(PAGE_ROWS))
        if isinstance(frame, pl.LazyFrame)
        else frame.head(PAGE_ROWS)
    )
    st.dataframe(data, hide_index=True, use_container_width=True)
    st.caption("Tables display at most 2,000 rows.")


def export_csv(data):
    columns = [x for x, dtype in data.schema.items() if dtype == pl.String]
    safe = data.with_columns(
        *[
            pl.when(pl.col(x).str.contains(r"^[=+\-@\t\r]"))
            .then(pl.concat_str([pl.lit("'"), pl.col(x)]))
            .otherwise(pl.col(x))
            .alias(x)
            for x in columns
        ]
    )
    return safe.write_csv().encode("utf8")


def chart(figure, title, key, x=None, y=None):
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=65, b=50),
        xaxis_title=x,
        yaxis_title=y,
        uirevision=key,
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        key=key,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}},
    )


def bars(data, group, metric, title, key, color_map=None):
    if data.is_empty():
        st.info("No observations match this view.")
        return
    shown = data.sort(metric, descending=True).head(25).reverse()
    if color_map is None:
        color_map = build_color_map(shown[group].to_list())
    chart(
        go.Figure(
            go.Bar(
                x=shown[metric].to_list(),
                y=shown[group].to_list(),
                orientation="h",
                marker_color=[get_color(v, color_map) for v in shown[group].to_list()],
            )
        ),
        title,
        key,
        format_axis_title(metric),
        group,
    )
    st.caption(f"Showing up to 25 groups out of {data.height}; omitted groups are not merged.")


def apply_filters(rows, filters):
    for dimension, values in filters.items():
        if values:
            rows = rows.filter(pl.col(dimension).is_in(values))
    return rows


def display_results(root, coverage):
    manifest = read_json(root / "run.json")
    thresholds = read_json(root / "thresholds.json")
    coverage = bool(coverage and manifest.get("coverage_confirmed"))
    
    # Show threshold calibration info
    if thresholds.get("reference_selection", {}).get("selected"):
        ref_sel = thresholds["reference_selection"]
        st.success(
            f"✓ Thresholds auto-calibrated from snapshot **{ref_sel['reference_date']}** "
            f"({ref_sel['style_population']:,} style-eligible, {ref_sel['value_population']:,} value-eligible customers)"
        )
        with st.expander("Calibrated thresholds", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Value tier low (P33)", f"{thresholds.get('value_low', 0):,.0f}")
                st.metric("Value tier high (P80)", f"{thresholds.get('value_high', 0):,.0f}")
                st.metric("Basket low (P33)", f"{thresholds.get('basket_low', 0):,.0f}")
                st.metric("Basket high (P80)", f"{thresholds.get('basket_high', 0):,.0f}")
            with col2:
                st.metric("Gap low (P33)", f"{thresholds.get('gap_low', 0):.1f} days")
                st.metric("Gap high (P80)", f"{thresholds.get('gap_high', 0):.1f} days")
                lateness = thresholds.get('lateness_overdue_min')
                if lateness:
                    st.metric("Lateness threshold", f"{lateness:.2f}")
                freq_thresh = thresholds.get('frequency_change_threshold')
                if freq_thresh:
                    st.metric("Frequency change threshold", f"±{freq_thresh:.2f}")
            
            # Show warnings from data profile
            data_profile = thresholds.get("data_profile", {})
            warnings = data_profile.get("warnings", [])
            if warnings:
                for w in warnings:
                    st.warning(w)
            
            # Show label share warnings
            label_shares = thresholds.get("label_shares", {})
            ls_warnings = label_shares.get("warnings", [])
            if ls_warnings:
                for w in ls_warnings:
                    st.info(w)
    elif thresholds.get("reference_date") is None and thresholds.get("reference_selection", {}).get("selected") is False:
        st.warning(
            "⚠ Thresholds not calibrated: no snapshot met minimum population criteria. "
            "Labels using quantile thresholds (large_basket_episodic, small_basket_frequent, value tiers) "
            "will show as 'reference_unavailable'."
        )
    
    chosen = st.selectbox(
        "Result cutoff", manifest["snapshots"], index=len(manifest["snapshots"]) - 1
    )
    raw = pl.scan_parquet(str(root / "snapshots" / chosen / "*.parquet"))
    filters = {}
    with st.expander("Population filters", expanded=False):
        for box, dim in zip(st.columns(2), ["lifecycle", "evidence"]):
            options = collect(raw.select(dim).unique().sort(dim))[dim].to_list()
            with box:
                filters[dim] = st.multiselect(dim, options, key=f"{root.name}_{dim}")
        st.caption(
            "Empty selections mean all customers. Historical filters are evaluated separately at each cutoff; transitions filter the destination population."
        )
    rows = apply_filters(raw, filters)

    # Build deterministic color map from all dimension labels
    all_labels = set()
    for dim in DIMS:
        labels = collect(rows.select(dim).unique().sort(dim))[dim].to_list()
        all_labels.update(str(label) for label in labels)
    color_map = build_color_map(all_labels)

    metrics = collect(
        rows.select(
            pl.len().alias("customers"),
            pl.col("spend_90d").sum().alias("sales"),
            pl.col("purchases_90d").sum().alias("baskets"),
            (pl.col("purchases_90d") > 0).sum().alias("active"),
        )
    ).row(0, named=True)
    for box, label, value in zip(
        st.columns(4),
        ["Filtered customers", "Recorded spend: 90d", "Baskets: 90d", "Active customers: 90d"],
        [metrics["customers"], f"{metrics['sales']:,.2f}", metrics["baskets"], metrics["active"]],
    ):
        box.metric(label, value)
    st.caption(
        "Recorded amounts are currency units, not profit. Windows end at the cutoff; inactive observed customers remain included."
    )
    if not metrics["customers"]:
        st.info("No customers match the filters. Clear or broaden the selections.")
        return
    view = st.radio(
        "Analytical view",
        [
            "Overview",
            "Segments",
            "Features",
            "Customer Momentum",
            "Purchase Cadence",
            "Category Participation",
            "Trial to Repeat",
            "Revenue Waterfall",
            "Value-Tier Transitions",
            "Behavior × Trajectory",
            "Purchase Rhythm",
            "Basket Evolution",
            "Category Sankey",
            "Future Validation",
            "Cohorts",
            "Transitions",
            "Quality",
            "Customer",
            "Methods",
        ],
        horizontal=True,
    )
    history = [x for x in manifest["snapshots"] if x <= chosen]
    if view == "Overview":
        historical = apply_filters(
            pl.scan_parquet(
                [
                    str(root / "snapshots" / d / p.name)
                    for d in history
                    for p in (root / "snapshots" / d).glob("*.parquet")
                ]
            ),
            filters,
        )
        trend = collect(
            historical.group_by("snapshot")
            .agg(
                pl.col("spend_90d").sum().alias("spend"),
                (pl.col("purchases_90d") > 0).sum().alias("active"),
                pl.len().alias("customers"),
            )
            .sort("snapshot")
        )
        for box, metric in zip(st.columns(2), ["spend", "active"]):
            with box:
                chart(
                    go.Figure(
                        go.Scatter(
                            x=trend["snapshot"].to_list(),
                            y=trend[metric].to_list(),
                            mode="lines+markers",
                        )
                    ),
                    f"Rolling 90-day {metric}",
                    f"trend_{metric}",
                    "Cutoff",
                    format_axis_title(metric) if metric in METRICS else metric,
                )
        st.caption(
            "Overlapping windows are not independent; filtered membership and observation exposure may change."
        )
        dim = st.selectbox("Composition", DIMS, key="overview_dim")
        data = collect(
            rows.group_by(dim).agg(
                pl.len().alias("customers"), pl.col("spend_90d").sum().alias("spend_90d")
            )
        )
        for box, metric in zip(st.columns(2), ["customers", "spend_90d"]):
            with box:
                bars(data, dim, metric, f"{metric} by {dim}", f"composition_{metric}", color_map)
    elif view == "Segments":
        dim = st.selectbox("Segmentation layer", DIMS)
        horizon = st.selectbox("Future-purchase horizon (days)", [30, 90])
        minimum = st.number_input("Minimum eligible customers per outcome estimate", 1, 10000, 20)
        outcome, eligible = f"future_purchases_{horizon}d", f"eligible_followup_{horizon}d"
        data = collect(
            rows.group_by(dim)
            .agg(
                pl.len().alias("customers"),
                pl.col("spend_90d").sum().alias("spend_90d"),
                pl.col("median_gap").median().alias("median_gap"),
                pl.col(outcome).filter(pl.col(eligible)).count().alias("followup_n"),
                (pl.col(outcome) > 0).filter(pl.col(eligible)).sum().alias("future_buyers"),
            )
            .sort("customers", descending=True)
        )
        records = []
        for record in data.to_dicts():
            n, buyers = record["followup_n"], record["future_buyers"]
            allowed = coverage and n >= minimum
            lo, hi = wilson(buyers, n) if allowed else (None, None)
            record.update(rate=buyers / n if allowed else None, low=lo, high=hi)
            if not coverage:
                record["followup_n"], record["future_buyers"] = None, None
            records.append(record)
        data = pl.DataFrame(records)
        bars(
            data,
            dim,
            st.selectbox("Segment bar measure", ["customers", "spend_90d"]),
            "Segment comparison",
            "segment_bars",
            color_map,
        )
        valid = [r for r in records if r["rate"] is not None][:25]
        if valid:
            fig = go.Figure(
                go.Scatter(
                    x=[r["rate"] for r in valid],
                    y=[r[dim] for r in valid],
                    mode="markers",
                    marker=dict(size=10, color=[get_color(r[dim], color_map) for r in valid]),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[r["high"] - r["rate"] for r in valid],
                        arrayminus=[r["rate"] - r["low"] for r in valid],
                    ),
                    customdata=[[r["future_buyers"], r["followup_n"]] for r in valid],
                    hovertemplate="%{y}<br>Rate: %{x:.1%}<br>Buyers: %{customdata[0]} / %{customdata[1]}<extra></extra>",
                )
            )
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            chart(
                fig,
                f"Next-{horizon}-day purchase rate: 95% Wilson intervals",
                "outcome_intervals",
                "Purchase proportion",
                dim,
            )
        else:
            st.info(
                "No outcome estimates: coverage is unconfirmed, follow-up is incomplete, or groups are too small."
            )
        st.caption(
            "Intervals assume independent customers at one cutoff, not causal effects. Up to 25 estimates are shown; exploratory comparisons are not multiplicity-adjusted."
        )
        table(data)
        st.download_button(
            "Export filtered segment summary", export_csv(data), "segments.csv", "text/csv"
        )
    elif view == "Features":
        features = [f for f in FEATURES if f in rows.collect_schema().names()]
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
                    chart(
                        go.Figure(trace),
                        title,
                        title.lower().replace(" ", "_"),
                        feature,
                        "Sample count" if title == "Histogram" else None,
                    )
        x = st.selectbox("Scatter X", features)
        y = st.selectbox("Scatter Y", features, index=min(1, len(features) - 1))
        dim = st.selectbox("Scatter color", DIMS)
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
            chart(fig, "Feature relationships", "scatter", x, y)
        st.caption(
            "Distributions and scatter use at most 2,000 hash-selected finite observations, not the full population. Missing values are not zeros. No customer IDs are sent to these charts."
        )
        st.caption(
            "repeat_share begins on the second distinct product-purchase day; established_share begins on the third. Existing repeat-style rules use established_share."
        )
    elif view == "Customer Momentum":
        st.subheader("Customer momentum map: frequency vs. basket change")
        st.caption(
            "X-axis: frequency_change (recent 90d vs previous 90d purchase frequency). "
            "Y-axis: basket_change (recent 90d vs previous 90d AOV). "
            "Colour: value_tier. Only customers with comparable windows (tenure ≥ 179 days) and non-zero previous purchases are shown."
        )

        mom = momentum_map(rows)
        sample = mom["sample"]
        quad = mom["quadrant_summary"]
        not_comp = mom["not_comparable_n"]
        finite_n = mom["finite_n"]

        if sample.is_empty():
            st.info("No customers with comparable frequency and basket changes in this population.")
        else:
            # Quadrant summary
            st.write("**Quadrant summary (all comparable customers)**")
            if not quad.is_empty():
                quad_display = quad.with_columns(
                    pl.col("quadrant").str.replace_all("_", " ").str.to_title_case().alias("Quadrant"),
                    pl.col("customer_share").alias("Customer share"),
                    pl.col("spend_share").alias("Spend share"),
                ).select("Quadrant", "customers", "Customer share", "spend_90d", "Spend share")
                st.dataframe(quad_display, hide_index=True, use_container_width=True)
            st.caption(f"Customers with comparable changes: {finite_n:,}. Not comparable (missing changes): {not_comp:,}. Not comparable customers are excluded from the scatter and quadrant counts; they are not placed at zero.")

            # Scatter plot
            fig = go.Figure()
            quadrant_colors = {
                "growing_both": "#009E73",           # bluish green
                "more_frequent_smaller_baskets": "#56B4E9",  # sky blue
                "less_frequent_larger_baskets": "#E69F00",   # orange
                "declining_both": "#D55E00",         # vermillion
            }
            quadrant_labels = {
                "growing_both": "Growing on both dimensions",
                "more_frequent_smaller_baskets": "More frequent, lighter shopping",
                "less_frequent_larger_baskets": "Less frequent, heavier shopping",
                "declining_both": "Declining on both dimensions",
            }

            for name in sorted(sample["value_tier"].unique().to_list()):
                part = sample.filter(pl.col("value_tier") == name)
                fig.add_trace(
                    go.Scattergl(
                        x=part["frequency_change"].to_list(),
                        y=part["basket_change"].to_list(),
                        mode="markers",
                        name=str(name),
                        marker=dict(size=6, opacity=0.55, color=get_color(name, color_map)),
                        customdata=list(zip(
                            part["spend_90d"].to_list(),
                            part["purchases_90d"].to_list(),
                            part["purchases_previous90d"].to_list(),
                            part["aov_recent"].to_list(),
                            part["aov_previous"].to_list(),
                            part["lifecycle"].to_list(),
                            part["evidence"].to_list(),
                        )),
                        hovertemplate=(
                            "Value tier: %{fullData.name}<br>"
                            "Frequency change: %{x:.1%}<br>"
                            "Basket change: %{y:.1%}<br>"
                            "Recent spend: %{customdata[0]:,.2f}<br>"
                            "Recent purchases: %{customdata[1]}<br>"
                            "Previous purchases: %{customdata[2]}<br>"
                            "Recent AOV: %{customdata[3]:,.2f}<br>"
                            "Previous AOV: %{customdata[4]:,.2f}<br>"
                            "Lifecycle: %{customdata[5]}<br>"
                            "Evidence: %{customdata[6]}<extra></extra>"
                        ),
                    )
                )

            # Add zero reference lines
            fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

            # Add quadrant annotations
            fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                             text="Growing on both", showarrow=False, font=dict(size=10, color="#009E73"), opacity=0.5)
            fig.add_annotation(x=0.5, y=0.05, xref="paper", yref="paper",
                             text="More frequent, lighter", showarrow=False, font=dict(size=10, color="#56B4E9"), opacity=0.5)
            fig.add_annotation(x=0.05, y=0.5, xref="paper", yref="paper",
                             text="Less frequent, heavier", showarrow=False, font=dict(size=10, color="#E69F00"), opacity=0.5)
            fig.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper",
                             text="Declining on both", showarrow=False, font=dict(size=10, color="#D55E00"), opacity=0.5)

            chart(fig, "Customer momentum map", "momentum_scatter",
                  format_axis_title("frequency_change"), format_axis_title("basket_change"))

            st.caption(
                f"Scatter shows up to {PAGE_ROWS:,} hash-sampled customers with finite changes. "
                f"Zero reference lines and quadrant labels are visual guides only. "
                f"Missing changes (not comparable) are not shown at zero; count: {not_comp:,}."
            )
            table(sample.select("customer_id", "frequency_change", "basket_change", "value_tier", "lifecycle", "evidence",
                              "spend_90d", "purchases_90d", "purchases_previous90d", "aov_recent", "aov_previous"))

    elif view == "Purchase Cadence":
        st.subheader("Purchase cadence map: recency vs. median gap")
        st.caption(
            "X-axis: median_gap (typical inter-purchase interval in days). "
            "Y-axis: recency (days since last purchase). "
            "Reference lines: recency = median_gap (on cadence) and recency = 2 × median_gap (overdue). "
            "This visualises the lateness = recency / median_gap metric."
        )

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            evidence_options = collect(rows.select("evidence").unique().sort("evidence"))["evidence"].to_list()
            evidence_filter = st.multiselect("Evidence level", evidence_options, key="cadence_evidence")
        with col2:
            vt_options = collect(rows.select("value_tier").unique().sort("value_tier"))["value_tier"].to_list()
            vt_filter = st.multiselect("Value tier", vt_options, key="cadence_vt")

        cad = cadence_map(rows, evidence_filter=evidence_filter or None, value_tier_filter=vt_filter or None)
        sample = cad["sample"]
        eligible_n = cad["eligible_n"]
        total_n = cad["total_n"]

        if sample.is_empty():
            st.info("No customers with finite median_gap and recency in this population.")
        else:
            st.write(f"**Eligible customers (style_eligible): {eligible_n:,} of {total_n:,}**")

            fig = go.Figure()
            for name in sorted(sample["value_tier"].unique().to_list()):
                part = sample.filter(pl.col("value_tier") == name)
                fig.add_trace(
                    go.Scattergl(
                        x=part["median_gap"].to_list(),
                        y=part["recency"].to_list(),
                        mode="markers",
                        name=str(name),
                        marker=dict(size=6, opacity=0.55, color=get_color(name, color_map)),
                        customdata=list(zip(
                            part["lateness"].to_list(),
                            part["evidence"].to_list(),
                            part["lifecycle"].to_list(),
                            part["gap_cv"].to_list(),
                            part["days_365d"].to_list(),
                        )),
                        hovertemplate=(
                            "Value tier: %{fullData.name}<br>"
                            "Median gap: %{x:.1f} days<br>"
                            "Recency: %{y:.1f} days<br>"
                            "Lateness: %{customdata[0]:.2f}<br>"
                            "Evidence: %{customdata[1]}<br>"
                            "Lifecycle: %{customdata[2]}<br>"
                            "Gap CV: %{customdata[3]:.2f}<br>"
                            "Shopping days (365d): %{customdata[4]}<extra></extra>"
                        ),
                    )
                )

            # Reference lines: recency = median_gap and recency = 2 * median_gap
            # We need to add these as traces since they're diagonal lines
            max_gap = max(sample["median_gap"].max(), sample["recency"].max()) if sample.height > 0 else 100
            x_line = [0, max_gap * 1.1]
            fig.add_trace(go.Scatter(
                x=x_line, y=x_line, mode="lines", name="Recency = Median gap (on cadence)",
                line=dict(color="gray", dash="dash", width=1), hoverinfo="skip", showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=x_line, y=[2 * v for v in x_line], mode="lines", name="Recency = 2× Median gap (overdue)",
                line=dict(color="red", dash="dash", width=1), hoverinfo="skip", showlegend=True
            ))

            # Highlight overdue region
            fig.add_vrect(x0=0, x1=max_gap * 1.1, y0=0, y1=max_gap * 1.1,
                          fillcolor="rgba(0,158,115,0.05)", line_width=0, annotation_text="On cadence",
                          annotation_position="top left", annotation_font_size=10, annotation_font_color="#009E73")
            fig.add_vrect(x0=0, x1=max_gap * 1.1, y0=max_gap * 1.1, y1=max_gap * 2.2,
                          fillcolor="rgba(213,94,0,0.05)", line_width=0, annotation_text="Overdue (1-2×)",
                          annotation_position="top left", annotation_font_size=10, annotation_font_color="#D55E00")
            fig.add_vrect(x0=0, x1=max_gap * 1.1, y0=max_gap * 2.2, y1=max_gap * 3.3,
                          fillcolor="rgba(213,94,0,0.1)", line_width=0, annotation_text="Overdue (≥2×)",
                          annotation_position="top left", annotation_font_size=10, annotation_font_color="#D55E00")

            chart(fig, "Purchase cadence map", "cadence_scatter",
                  "Median gap (days)", "Recency (days)")

            st.caption(
                f"Shows up to {PAGE_ROWS:,} hash-sampled customers with finite median_gap > 0 and recency. "
                f"Grey line: recency = median gap (on cadence). Red line: recency = 2× median gap (overdue threshold). "
                f"Coloured regions: green = on cadence, light orange = 1-2× overdue, darker orange = ≥2× overdue. "
                f"Lateness = recency / median_gap. Only style_eligible customers have meaningful median_gap."
            )
            table(sample.select("customer_id", "median_gap", "recency", "lateness", "value_tier", "evidence", "lifecycle",
                              "gap_cv", "days_365d"))

    elif view == "Category Participation":
        st.subheader("Category participation: recent 90d vs previous 90d")
        st.caption(
            "Participation = customers purchasing from category during period ÷ fixed comparison population. "
            "Recent 90d: [C-89, C]. Previous 90d: [C-179, C-90]. "
            "Same population denominator for both periods. Change in percentage points shown."
        )

        # Load validated data for category participation (needs per-transaction dates)
        validated_path = root / "validated.parquet"
        if not validated_path.exists():
            st.info("No validated data available for this analysis.")
        else:
            validated = pl.scan_parquet(validated_path)
            part = category_participation_comparison(validated, rows)

            if part.is_empty():
                st.info("No category participation data for this population.")
            else:
                # Top categories by recent participation
                top_n = st.slider("Top N categories to show", 5, 50, 20)
                display = part.head(top_n)

                # Paired horizontal bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=display["label"].to_list(),
                    x=display["recent_participation"].to_list(),
                    name="Recent 90d",
                    orientation="h",
                    marker_color="#0072B2",
                    customdata=list(zip(
                        display["recent_buyers"].to_list(),
                        display["recent_spend"].to_list(),
                        display["previous_buyers"].to_list(),
                        display["previous_spend"].to_list(),
                        display["participation_change_pp"].to_list(),
                    )),
                    hovertemplate=(
                        "%{y}<br>"
                        "Recent participation: %{x:.1%}<br>"
                        "Recent buyers: %{customdata[0]:,}<br>"
                        "Recent spend: %{customdata[1]:,.2f}<br>"
                        "Previous buyers: %{customdata[2]:,}<br>"
                        "Previous spend: %{customdata[3]:,.2f}<br>"
                        "Change (pp): %{customdata[4]:.2f}<extra></extra>"
                    ),
                ))
                fig.add_trace(go.Bar(
                    y=display["label"].to_list(),
                    x=display["previous_participation"].to_list(),
                    name="Previous 90d",
                    orientation="h",
                    marker_color="#E69F00",
                    opacity=0.6,
                    customdata=list(zip(
                        display["recent_buyers"].to_list(),
                        display["recent_spend"].to_list(),
                        display["previous_buyers"].to_list(),
                        display["previous_spend"].to_list(),
                        display["participation_change_pp"].to_list(),
                    )),
                    hovertemplate=(
                        "%{y}<br>"
                        "Previous participation: %{x:.1%}<br>"
                        "Recent buyers: %{customdata[0]:,}<br>"
                        "Recent spend: %{customdata[1]:,.2f}<br>"
                        "Previous buyers: %{customdata[2]:,}<br>"
                        "Previous spend: %{customdata[3]:,.2f}<br>"
                        "Change (pp): %{customdata[4]:.2f}<extra></extra>"
                    ),
                ))
                fig.update_layout(barmode="group")
                chart(fig, f"Category participation (top {top_n})", "cat_participation_bars",
                      "Participation (share of filtered population)", "Category")

                # Table with changes
                table(display.select("label", "population_n", "recent_participation", "previous_participation",
                                     "participation_change_pp", "buyer_change", "recent_buyers", "previous_buyers",
                                     "recent_spend", "previous_spend"))

                st.caption(
                    f"Population (denominator): {part['population_n'][0] if part.height > 0 else 0:,} customers (fixed for both periods). "
                    f"Participation is share of this population. Change in percentage points = recent - previous. "
                    f"Categories with zero buyers in a period show 0% participation, not missing."
                )

    elif view == "Trial to Repeat":
        st.subheader("Product discovery to repeat: trial conversion by product")
        st.caption(
            "X-axis: number of eligible trials (first purchase in [C-364, C-90]). "
            "Y-axis: pooled trial conversion (repeats within 90d / eligible trials). "
            "Bubble size: distinct trialling customers. "
            "Pooled conversion = Σ repeats / Σ eligible trials across all customers."
        )

        # Load product history for the chosen cutoff
        prod_history_paths = list((root / "events" / chosen).glob("*/product_history.parquet"))
        if not prod_history_paths:
            st.info("No product history available for this cutoff.")
        else:
            prod_history = pl.scan_parquet(prod_history_paths)
            trial_data = trial_to_repeat_map(prod_history, rows)

            if trial_data.is_empty():
                st.info("No eligible trials in this population.")
            else:
                # Filter to products with minimum trials
                min_trials = st.number_input("Minimum eligible trials per product", 1, 1000, 5)
                plot_data = trial_data.filter(pl.col("eligible_trials") >= min_trials)

                if plot_data.is_empty():
                    st.info(f"No products with ≥ {min_trials} eligible trials.")
                else:
                    # Bubble chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=plot_data["eligible_trials"].to_list(),
                        y=plot_data["trial_conversion"].to_list(),
                        mode="markers",
                        marker=dict(
                            size=[max(8, min(60, math.sqrt(c) * 3)) for c in plot_data["trialling_customers"].to_list()],
                            opacity=0.6,
                            color="#0072B2",
                        ),
                        customdata=list(zip(
                            plot_data["eligible_trials"].to_list(),
                            plot_data["repeated_trials"].to_list(),
                            plot_data["trialling_customers"].to_list(),
                            plot_data["product_id"].to_list(),
                        )),
                        hovertemplate=(
                            "Product: %{customdata[3]}<br>"
                            "Eligible trials: %{customdata[0]}<br>"
                            "Repeated trials: %{customdata[1]}<br>"
                            "Trialling customers: %{customdata[2]}<br>"
                            "Trial conversion: %{y:.1%}<extra></extra>"
                        ),
                    ))
                    fig.update_yaxes(range=[0, 1], tickformat=".0%")
                    chart(fig, f"Trial to repeat (min {min_trials} trials)", "trial_repeat_bubble",
                          "Eligible trials", "Trial conversion (pooled)")

                    st.caption(
                        f"Pooled conversion = total repeated trials / total eligible trials across customers. "
                        f"Not the mean of customer-level conversion rates. "
                        f"Eligible trials: first purchase in [C-364, C-90] with 90-day follow-up window fully observed. "
                        f"Categories not shown; requires product→category lookup."
                    )
                    table(plot_data.select("product_id", "eligible_trials", "repeated_trials",
                                           "trialling_customers", "trial_conversion"))

    elif view == "Revenue Waterfall":
        st.subheader("Revenue driver waterfall: recent 90d vs previous 90d")
        st.caption(
            "Decomposes spend change into: purchasing customers × baskets per purchaser × spend per basket (AOV). "
            "Uses a fixed-order bridge (customer count → frequency → AOV). "
            "Residual captures interaction effects. Method documented; contributions sum to observed change."
        )

        # Need previous cutoff data
        if len(history) < 2:
            st.info("Need at least two cutoffs for waterfall. Select an earlier snapshot in the sidebar.")
        else:
            origin = st.selectbox("Previous cutoff", history[:-1], index=len(history) - 2)
            before_rows = apply_filters(
                pl.scan_parquet([str(p) for p in (root / "snapshots" / origin).glob("*.parquet")]),
                filters,
            )
            cutoff_date = date.fromisoformat(chosen)
            waterfall = revenue_waterfall(rows, before_rows, cutoff_date)

            if waterfall.is_empty():
                st.info("Could not compute waterfall for this comparison.")
            else:
                # Waterfall chart
                fig = go.Figure()
                components = waterfall["component"].to_list()
                values = waterfall["value"].to_list()

                # Calculate cumulative for waterfall
                cumulative = [values[0]]
                for i in range(1, len(values) - 1):
                    cumulative.append(cumulative[-1] + values[i])
                cumulative.append(values[-1])  # final value

                colors = []
                for i, v in enumerate(values):
                    if i == 0 or i == len(values) - 1:
                        colors.append("#0072B2")  # start/end
                    elif v >= 0:
                        colors.append("#009E73")  # positive
                    else:
                        colors.append("#D55E00")  # negative

                fig.add_trace(go.Waterfall(
                    name="Spend decomposition",
                    orientation="v",
                    measure=["absolute"] + ["relative"] * (len(values) - 2) + ["total"],
                    x=components,
                    y=values,
                    text=[f"{v:,.2f}" for v in values],
                    textposition="outside",
                    connector=dict(line=dict(color="rgb(63, 63, 63)")),
                    increasing=dict(marker=dict(color="#009E73")),
                    decreasing=dict(marker=dict(color="#D55E00")),
                    totals=dict(marker=dict(color="#0072B2")),
                ))

                chart(fig, f"Revenue waterfall: {origin} to {chosen}", "revenue_waterfall",
                      "Component", "Spend change (currency units)")

                table(waterfall)
                st.caption(
                    "Fixed-order bridge: (1) purchasing customers change × prev frequency × prev AOV, "
                    "(2) current customers × frequency change × prev AOV, "
                    "(3) current customers × current frequency × AOV change. "
                    "Residual = interaction effects. Attribution depends on decomposition order."
                )

    elif view == "Value-Tier Transitions":
        st.subheader("Value-tier transition waterfall")
        st.caption(
            "Net customer movement between value tiers between two snapshots. "
            "Inflow = customers entering tier, Outflow = customers leaving tier, "
            "Net = inflow - outflow. Uses transition data from segment_transitions.parquet."
        )

        if len(history) < 2:
            st.info("Need at least two cutoffs. Select an earlier snapshot in the sidebar.")
        else:
            origin = st.selectbox("Previous cutoff", history[:-1], index=len(history) - 2, key="vt_origin")
            # Waterfall only works for value_tier; other dimensions use Transitions view
            dim = "value_tier"
            st.info("Showing value_tier transitions (waterfall). For other dimensions, use the Transitions view.")

            before = apply_filters(
                pl.scan_parquet([str(p) for p in (root / "snapshots" / origin).glob("*.parquet")]),
                filters,
            )
            after = apply_filters(
                pl.scan_parquet([str(p) for p in (root / "snapshots" / chosen).glob("*.parquet")]),
                filters,
            )
            flows, counts = transition_summary(before, after, dim)

            if flows.is_empty():
                st.info("No matched customers at both cutoffs.")
            else:
                    waterfall = value_tier_transition_waterfall(flows)

                    if not waterfall.is_empty():
                        # Waterfall chart
                        fig = go.Figure()
                        tiers = waterfall["tier"].to_list()
                        inflows = waterfall["inflow"].to_list()
                        outflows = waterfall["outflow"].to_list()
                        net = waterfall["net_change"].to_list()

                        fig.add_trace(go.Bar(
                            x=tiers, y=inflows, name="Inflow", marker_color="#009E73",
                            hovertemplate="Tier: %{x}<br>Customers entering: %{y}<extra></extra>"
                        ))
                        fig.add_trace(go.Bar(
                            x=tiers, y=[-o for o in outflows], name="Outflow", marker_color="#D55E00",
                            hovertemplate="Tier: %{x}<br>Customers leaving: %{y}<extra></extra>"
                        ))
                        fig.add_trace(go.Scatter(
                            x=tiers, y=net, name="Net change", mode="lines+markers",
                            line=dict(color="#0072B2", width=3), marker=dict(size=10),
                            hovertemplate="Tier: %{x}<br>Net change: %{y}<extra></extra>"
                        ))

                        fig.update_layout(barmode="relative")
                        chart(fig, f"Value-tier net movement: {origin} to {chosen}", "vt_waterfall",
                              "Value tier", "Customer count")

                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total inflow", int(waterfall["inflow"].sum()))
                        with col2:
                            st.metric("Total outflow", int(waterfall["outflow"].sum()))
                        with col3:
                            st.metric("Net change", int(waterfall["net_change"].sum()))

                        table(waterfall.select("tier", "inflow", "outflow", "net_change", "direction"))
                        st.caption(
                            f"Matched customers: {counts['matched_n']:,}. "
                            f"Inflow = customers entering tier. Outflow = customers leaving tier. "
                            f"Unmatched origin customers are not churn; unmatched destination are new arrivals."
                        )
                    else:
                        st.info("No value tier transitions found.")

    elif view == "Behavior × Trajectory":
        st.subheader("Behavior × Trajectory bubble matrix")
        st.caption(
            "X-axis: behavioral style (behavior). Y-axis: trajectory tags. "
            "Bubble size: customer count. Bubble color: median 90-day spend. "
            "Shows which style-trajectory combinations are growing, declining, or at risk."
        )

        matrix = behavior_trajectory_matrix(rows)

        if matrix.is_empty():
            st.info("No behavior-trajectory combinations in this population.")
        else:
            # Filter to combinations with sufficient customers
            min_customers = st.number_input("Minimum customers per combination", 1, 1000, 5, key="bt_min")
            plot_data = matrix.filter(pl.col("customers") >= min_customers)

            if plot_data.is_empty():
                st.info(f"No combinations with ≥ {min_customers} customers.")
            else:
                # Bubble chart
                fig = go.Figure()

                behaviors = sorted(plot_data["behavior"].unique().to_list())
                trajectories = sorted(plot_data["trajectory"].unique().to_list())

                for _, row in plot_data.iter_rows(named=True):
                    fig.add_trace(go.Scatter(
                        x=[row["behavior"]], y=[row["trajectory"]],
                        mode="markers",
                        marker=dict(
                            size=max(10, min(80, math.sqrt(row["customers"]) * 2)),
                            color=row["median_spend"] if row["median_spend"] else 0,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Median 90d spend", thickness=15),
                            opacity=0.7,
                        ),
                        name=f"{row['behavior']} × {row['trajectory']}",
                        hovertemplate=(
                            "Behavior: %{x}<br>"
                            "Trajectory: %{y}<br>"
                            "Customers: %{customdata[0]:,}<br>"
                            "Median spend: %{customdata[1]:,.2f}<br>"
                            "Future 30d rate: %{customdata[2]:.1%}<br>"
                            "Future 90d rate: %{customdata[3]:.1%}<extra></extra>"
                        ),
                        customdata=[[row["customers"], row["median_spend"] or 0,
                                     row["future_30d_rate"] or 0, row["future_90d_rate"] or 0]],
                        showlegend=False,
                    ))

                chart(fig, f"Behavior × Trajectory matrix (min {min_customers} customers)", "bt_matrix",
                      "Behavioral style", "Trajectory")

                # Summary table with future outcome rates
                table(plot_data.select("behavior", "trajectory", "customers", "median_spend", "mean_spend",
                                       "future_30d_rate", "future_90d_rate", "eligible_30d", "eligible_90d"))
                st.caption(
                    "Bubble size = customer count. Color = median 90-day spend. "
                    "Future rates only shown for eligible follow-up customers. "
                    "Behavior and trajectory tags can overlap; matrix preserves multi-tag combinations."
                )

    elif view == "Purchase Rhythm":
        st.subheader("Purchase rhythm panel")
        st.caption(
            "Distributions of median_gap (typical cadence), gap_cv (irregularity), and lateness (overdue). "
            "Segment-level summary shows how rhythm differs by behavioral style. "
            "Regular shoppers: low gap_cv, low lateness. Episodic: high median_gap, variable gap_cv."
        )

        rhythm = purchase_rhythm_panel(rows)

        if rhythm["segment_rhythm"].is_empty():
            st.info("No style-eligible customers in this population.")
        else:
            # Three histograms
            col1, col2, col3 = st.columns(3)

            with col1:
                if rhythm["median_gap_hist"].height > 0:
                    fig = go.Figure(go.Histogram(
                        x=rhythm["median_gap_hist"]["median_gap"].to_list(), nbinsx=30,
                        marker_color="#0072B2", opacity=0.7
                    ))
                    chart(fig, "Median gap distribution", "rhythm_median_gap",
                          "Median gap (days)", "Customer count")

            with col2:
                if rhythm["gap_cv_hist"].height > 0:
                    fig = go.Figure(go.Histogram(
                        x=rhythm["gap_cv_hist"]["gap_cv"].to_list(), nbinsx=30,
                        marker_color="#E69F00", opacity=0.7
                    ))
                    chart(fig, "Gap CV distribution", "rhythm_gap_cv",
                          "Gap coefficient of variation", "Customer count")

            with col3:
                if rhythm["lateness_hist"].height > 0:
                    fig = go.Figure(go.Histogram(
                        x=rhythm["lateness_hist"]["lateness"].to_list(), nbinsx=30,
                        marker_color="#D55E00", opacity=0.7
                    ))
                    fig.add_vline(x=1.0, line_dash="dash", line_color="gray",
                                  annotation_text="On cadence", annotation_position="top")
                    fig.add_vline(x=2.0, line_dash="dash", line_color="red",
                                  annotation_text="Overdue", annotation_position="top")
                    chart(fig, "Lateness distribution", "rhythm_lateness",
                          "Lateness (recency / median_gap)", "Customer count")

            # Segment rhythm summary
            st.write("**Segment rhythm summary**")
            seg = rhythm["segment_rhythm"]
            table(seg.select("behavior", "customers", "median_gap", "gap_cv", "lateness",
                             "median_recency", "avg_frequency_90d"))
            st.caption(
                "Median gap = typical inter-purchase interval. Gap CV = irregularity (std/mean). "
                "Lateness = recency / median_gap (≥2 = overdue). "
                "Only style-eligible customers (days_365d≥6, tenure≥90, gap_count≥4) included."
            )

    elif view == "Basket Evolution":
        st.subheader("Basket composition slope chart")
        st.caption(
            "Before/after comparison for customers with comparable 90-day windows (tenure ≥ 179 days). "
            "Shows shifts in breadth, AOV, spend, and product familiarity shares. "
            "Only customers with finite recent and previous metrics are included."
        )

        basket = basket_composition_slope(rows)

        summary = basket["summary"]
        sample = basket["sample"]

        if summary.is_empty() or summary["comparable_customers"][0] == 0:
            st.info("No customers with comparable windows in this population.")
        else:
            comp_n = summary["comparable_customers"][0]

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Comparable customers", f"{comp_n:,}")
            with col2:
                db = summary["mean_breadth_recent"][0] - summary["mean_breadth_previous"][0]
                st.metric("Δ Breadth (mean)", f"{db:+.2f}")
            with col3:
                da = summary["mean_aov_recent"][0] - summary["mean_aov_previous"][0]
                st.metric("Δ AOV (mean)", f"{da:+,.2f}")
            with col4:
                ds = summary["mean_spend_recent"][0] - summary["mean_spend_previous"][0]
                st.metric("Δ Spend (mean)", f"{ds:+,.2f}")

            # Slope chart for mean values
            fig = go.Figure()
            metrics = ["Breadth", "AOV", "Spend", "New share", "Repeat share", "Established share"]
            prev_vals = [
                summary["mean_breadth_previous"][0], summary["mean_aov_previous"][0],
                summary["mean_spend_previous"][0], summary["mean_new_share"][0],
                summary["mean_repeat_share"][0], summary["mean_established_share"][0]
            ]
            recent_vals = [
                summary["mean_breadth_recent"][0], summary["mean_aov_recent"][0],
                summary["mean_spend_recent"][0], summary["mean_new_share"][0],
                summary["mean_repeat_share"][0], summary["mean_established_share"][0]
            ]

            for i, (m, p, r) in enumerate(zip(metrics, prev_vals, recent_vals)):
                color = "#009E73" if r > p else ("#D55E00" if r < p else "#64748b")
                fig.add_trace(go.Scatter(
                    x=["Previous 90d", "Recent 90d"], y=[p, r],
                    mode="lines+markers", name=m, line=dict(color=color, width=2),
                    marker=dict(size=8), hovertemplate=f"{m}: %{{y:.3f}}<extra></extra>"
                ))

            chart(fig, f"Mean basket composition shift (n={comp_n:,})", "basket_slope",
                  "Period", "Value")

            # Dumbbell for median values
            fig2 = go.Figure()
            metrics2 = ["Breadth", "AOV"]
            prev_med = [summary["median_breadth_previous"][0], summary["median_aov_previous"][0]]
            recent_med = [summary["median_breadth_recent"][0], summary["median_aov_recent"][0]]

            for i, (m, p, r) in enumerate(zip(metrics2, prev_med, recent_med)):
                color = "#009E73" if r > p else ("#D55E00" if r < p else "#64748b")
                fig2.add_trace(go.Scatter(
                    x=[p, r], y=[m, m], mode="lines", line=dict(color=color, width=4),
                    showlegend=False, hoverinfo="skip"
                ))
                fig2.add_trace(go.Scatter(
                    x=[p], y=[m], mode="markers", marker=dict(size=12, color="#E69F00"),
                    name="Previous" if i == 0 else None, hovertemplate=f"{m} previous: %{{x:.2f}}<extra></extra>"
                ))
                fig2.add_trace(go.Scatter(
                    x=[r], y=[m], mode="markers", marker=dict(size=12, color="#0072B2"),
                    name="Recent" if i == 0 else None, hovertemplate=f"{m} recent: %{{x:.2f}}<extra></extra>"
                ))

            chart(fig2, f"Median basket composition shift (n={comp_n:,})", "basket_dumbbell",
                  "Value", "Metric")

            # Customer-level scatter (sampled)
            if sample.height > 0:
                st.write("**Customer-level shifts (sampled)**")
                metric_choice = st.selectbox(
                    "Metric", ["Breadth", "AOV", "Spend", "Frequency change", "Basket change"],
                    key="basket_metric"
                )

                if metric_choice == "Breadth":
                    x_col, y_col = "breadth_previous", "breadth_recent"
                    x_title, y_title = "Previous breadth", "Recent breadth"
                elif metric_choice == "AOV":
                    x_col, y_col = "aov_previous", "aov_recent"
                    x_title, y_title = "Previous AOV", "Recent AOV"
                elif metric_choice == "Spend":
                    x_col, y_col = "spend_previous90d", "spend_90d"
                    x_title, y_title = "Previous spend", "Recent spend"
                elif metric_choice == "Frequency change":
                    x_col, y_col = "frequency_change", "basket_change"
                    x_title, y_title = "Frequency change", "Basket change"
                else:  # Basket change
                    x_col, y_col = "basket_change", "frequency_change"
                    x_title, y_title = "Basket change", "Frequency change"

                fig3 = go.Figure()
                for name in sorted(sample["value_tier"].unique().to_list()):
                    part = sample.filter(pl.col("value_tier") == name)
                    fig3.add_trace(go.Scattergl(
                        x=part[x_col].to_list(), y=part[y_col].to_list(),
                        mode="markers", name=str(name),
                        marker=dict(size=6, opacity=0.55, color=get_color(name, color_map)),
                        hovertemplate=f"Value tier: %{{fullData.name}}<br>{x_title}: %{{x}}<br>{y_title}: %{{y}}<extra></extra>"
                    ))

                # Identity line
                max_val = max(sample[x_col].max(), sample[y_col].max())
                min_val = min(sample[x_col].min(), sample[y_col].min())
                fig3.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode="lines", line=dict(color="gray", dash="dash"),
                    showlegend=False, hoverinfo="skip"
                ))

                chart(fig3, f"Customer-level: {x_title} vs {y_title}", "basket_scatter",
                      x_title, y_title)

            st.caption(
                f"Comparable customers: {comp_n:,} (tenure ≥ 179 days, non-zero previous purchases). "
                "Slope chart shows mean shifts. Dumbbell shows median shifts. "
                "Identity line: points above = increase, below = decrease. "
                "Missing/recent/previous values excluded from calculations."
            )

    elif view == "Category Sankey":
        st.subheader("Category participation Sankey")
        st.caption(
            "Shows category participation change between two snapshots. "
            "Participation = customers buying from category / filtered population (same denominator). "
            "Uses validated transaction data for both periods."
        )

        if len(history) < 2:
            st.info("Need at least two cutoffs. Select an earlier snapshot in the sidebar.")
        else:
            origin = st.selectbox("Previous cutoff", history[:-1], index=len(history) - 2, key="cat_origin")

            # Load validated data
            validated_path = root / "validated.parquet"
            if not validated_path.exists():
                st.info("No validated data available.")
            else:
                validated = pl.scan_parquet(validated_path)

                # Fixed population: current filtered population
                ids = rows.select("customer_id").unique()
                n = collect(ids.select(pl.len())).item()

                # Current cutoff date ranges
                cutoff_curr = date.fromisoformat(chosen)
                cutoff_prev = date.fromisoformat(origin)

                from retail_analytics.core.constants import WINDOW_RECENT, WINDOW_PREVIOUS_START_OFFSET, WINDOW_YEAR

                # Current period: [C-364, C] (365-day window for category history)
                curr_start = cutoff_curr - timedelta(days=WINDOW_YEAR - 1)
                curr_end = cutoff_curr

                # Previous period: [origin-364, origin]
                prev_start = cutoff_prev - timedelta(days=WINDOW_YEAR - 1)
                prev_end = cutoff_prev

                # Current participation
                curr_cats = (
                    validated.filter(pl.col("date").is_between(curr_start, curr_end))
                    .join(ids, on="customer_id", how="semi")
                    .group_by("department", "category")
                    .agg(
                        pl.col("customer_id").n_unique().alias("curr_customers"),
                        pl.col("value").sum().alias("curr_spend"),
                    )
                    .with_columns(
                        pl.lit(n).alias("curr_population"),
                        (pl.col("curr_customers") / pl.lit(n)).alias("curr_participation"),
                    )
                )

                # Previous participation
                prev_ids = apply_filters(
                    pl.scan_parquet([str(p) for p in (root / "snapshots" / origin).glob("*.parquet")]),
                    filters,
                ).select("customer_id").unique()
                prev_n = collect(prev_ids.select(pl.len())).item()

                prev_cats = (
                    validated.filter(pl.col("date").is_between(prev_start, prev_end))
                    .join(prev_ids, on="customer_id", how="semi")
                    .group_by("department", "category")
                    .agg(
                        pl.col("customer_id").n_unique().alias("prev_customers"),
                        pl.col("value").sum().alias("prev_spend"),
                    )
                    .with_columns(
                        pl.lit(prev_n).alias("prev_population"),
                        (pl.col("prev_customers") / pl.lit(prev_n)).alias("prev_participation"),
                    )
                )

                merged = collect(
                    curr_cats.join(prev_cats, on=["department", "category"], how="full")
                    .with_columns(
                        pl.col("curr_participation").fill_null(0),
                        pl.col("prev_participation").fill_null(0),
                        pl.col("curr_customers").fill_null(0),
                        pl.col("prev_customers").fill_null(0),
                        pl.col("curr_spend").fill_null(0),
                        pl.col("prev_spend").fill_null(0),
                        pl.col("curr_population").fill_null(n),
                        pl.col("prev_population").fill_null(prev_n),
                    )
                    .with_columns(
                        (pl.col("curr_participation") - pl.col("prev_participation")).alias("part_change"),
                        pl.concat_str([pl.col("department"), pl.lit(" / "), pl.col("category")]).alias("label"),
                    )
                    .sort("curr_participation", descending=True)
                )

                if merged.is_empty():
                    st.info("No category participation data for one or both periods.")
                else:
                    top_n = st.slider("Top N categories", 5, 50, 20, key="cat_sankey_top")
                    display = merged.head(top_n)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=display["label"].to_list(),
                        x=display["curr_participation"].to_list(),
                        name=f"Current ({chosen})", orientation="h",
                        marker_color="#0072B2",
                        customdata=list(zip(
                            display["curr_customers"].to_list(),
                            display["curr_spend"].to_list(),
                            display["prev_customers"].to_list(),
                            display["prev_spend"].to_list(),
                            display["part_change"].to_list(),
                        )),
                        hovertemplate=(
                            "%{y}<br>Current participation: %{x:.1%}<br>"
                            "Current buyers: %{customdata[0]:,}<br>"
                            "Current spend: %{customdata[1]:,.2f}<br>"
                            "Previous buyers: %{customdata[2]:,}<br>"
                            "Previous spend: %{customdata[3]:,.2f}<br>"
                            "Change (pp): %{customdata[4]:.2f}<extra></extra>"
                        ),
                    ))
                    fig.add_trace(go.Bar(
                        y=display["label"].to_list(),
                        x=display["prev_participation"].to_list(),
                        name=f"Previous ({origin})", orientation="h",
                        marker_color="#E69F00", opacity=0.6,
                        customdata=list(zip(
                            display["curr_customers"].to_list(),
                            display["curr_spend"].to_list(),
                            display["prev_customers"].to_list(),
                            display["prev_spend"].to_list(),
                            display["part_change"].to_list(),
                        )),
                        hovertemplate=(
                            "%{y}<br>Previous participation: %{x:.1%}<br>"
                            "Current buyers: %{customdata[0]:,}<br>"
                            "Current spend: %{customdata[1]:,.2f}<br>"
                            "Previous buyers: %{customdata[2]:,}<br>"
                            "Previous spend: %{customdata[3]:,.2f}<br>"
                            "Change (pp): %{customdata[4]:.2f}<extra></extra>"
                        ),
                    ))
                    fig.update_layout(barmode="group")
                    chart(fig, f"Category participation change: {origin} → {chosen} (top {top_n})", "cat_sankey",
                          "Participation (share of period population)", "Category")

                    table(display.select("label", "curr_population", "prev_population",
                                         "curr_participation", "prev_participation", "part_change",
                                         "curr_customers", "prev_customers", "curr_spend", "prev_spend"))

                    st.caption(
                        "Participation = category buyers / period population (same denominator per period). "
                        "Current period uses current filtered population; previous uses previous filtered population. "
                        "For fixed-population comparison, use Category Participation view."
                    )

    elif view == "Future Validation":
        st.subheader("Future outcome validation")
        st.caption(
            "Do current behavioral tags predict 30/90-day future purchases and spend? "
            "Only customers with fully observed follow-up windows (eligible_followup) are included. "
            "Wilson 95% confidence intervals shown for purchase rates."
        )

        if not coverage:
            st.warning("Coverage unconfirmed: future outcome validation unavailable. "
                       "Enable coverage confirmation in sidebar to see this view.")
        else:
            validation = future_outcome_validation(rows)

            if validation.is_empty():
                st.info("No eligible follow-up customers in this population.")
            else:
                horizon = st.selectbox("Horizon", [30, 90], key="fv_horizon")
                dim = st.selectbox("Dimension", ["behavior", "trajectory", "value_tier", "lifecycle", "evidence"],
                                   key="fv_dim")

                plot_data = validation.filter(
                    (pl.col("horizon") == horizon) & (pl.col("dimension") == dim)
                ).sort("rate", descending=True)

                if plot_data.is_empty():
                    st.info(f"No data for {dim} at {horizon}-day horizon.")
                else:
                    # Bar chart with Wilson intervals
                    from retail_analytics.core.stats import wilson

                    fig = go.Figure()
                    for row in plot_data.iter_rows(named=True):
                        n = row["eligible_n"]
                        buyers = row["buyers"]
                        rate = row["rate"]
                        lo, hi = wilson(buyers, n) if n >= 20 else (None, None)

                        fig.add_trace(go.Bar(
                            x=[row[dim]], y=[rate],
                            name=str(row[dim]),
                            marker_color=get_color(row[dim], color_map),
                            error_y=dict(
                                type="data", array=[hi - rate] if hi else None,
                                arrayminus=[rate - lo] if lo else None,
                                visible=True
                            ) if lo is not None else None,
                            customdata=[[buyers, n, row["mean_spend"], row["mean_spend_buyers"]]],
                            hovertemplate=(
                                "%{x}<br>Rate: %{y:.1%}<br>"
                                "Buyers: %{customdata[0]} / %{customdata[1]}<br>"
                                "Mean spend: %{customdata[2]:,.2f}<br>"
                                "Mean spend (buyers): %{customdata[3]:,.2f}<extra></extra>"
                            ),
                            showlegend=False,
                        ))

                    fig.update_yaxes(range=[0, 1], tickformat=".0%")
                    chart(fig, f"Future {horizon}-day purchase rate by {dim}", "fv_rate",
                          dim, "Purchase rate (Wilson 95% CI)")

                    # Spend chart
                    fig2 = go.Figure()
                    for row in plot_data.iter_rows(named=True):
                        fig2.add_trace(go.Bar(
                            x=[row[dim]], y=[row["mean_spend"]],
                            name=str(row[dim]),
                            marker_color=get_color(row[dim], color_map),
                            customdata=[[row["buyers"], row["eligible_n"]]],
                            hovertemplate=(
                                "%{x}<br>Mean future spend: %{y:,.2f}<br>"
                                "Buyers: %{customdata[0]} / %{customdata[1]}<extra></extra>"
                            ),
                            showlegend=False,
                        ))

                    chart(fig2, f"Mean future {horizon}-day spend by {dim}", "fv_spend",
                          dim, "Mean spend (currency units)")

                    table(plot_data.select("dimension", dim, "eligible_n", "buyers", "rate",
                                           "mean_spend", "mean_spend_buyers"))
                    st.caption(
                        f"Only customers with eligible_followup_{horizon}d=True included. "
                        f"Wilson 95% CI shown for groups with ≥20 eligible customers. "
                        f"Descriptive association, not causal prediction. "
                        f"Multiple comparisons not multiplicity-adjusted."
                    )

    elif view == "Cohorts":
        historical = apply_filters(
            pl.scan_parquet(
                [str(p) for d in history for p in (root / "snapshots" / d).glob("*.parquet")]
            ),
            filters,
        )

        # Calendar activity heatmap (existing)
        cohort = collect(
            historical.with_columns(pl.col("first_purchase").dt.truncate("1mo").alias("cohort"))
            .group_by("cohort", "snapshot")
            .agg(
                pl.len().alias("customers"), (pl.col("purchases_30d") > 0).mean().alias("activity")
            )
        )
        xs, ys, z = bounded_grid(cohort.to_dicts(), "cohort", "snapshot", "activity")
        _, _, counts = bounded_grid(cohort.to_dicts(), "cohort", "snapshot", "customers")
        fig = go.Figure(
            go.Heatmap(
                x=xs,
                y=ys,
                z=z,
                customdata=counts,
                zmin=0,
                zmax=1,
                colorscale="Blues",
                colorbar=dict(tickformat=".0%"),
                hovertemplate="Cohort %{y}<br>Cutoff %{x}<br>30-day activity %{z:.1%}<br>Customers %{customdata}<extra></extra>",
                hoverongaps=False,
            )
        )
        chart(
            fig,
            "First-observed cohorts: rolling 30-day activity",
            "cohort_heatmap",
            "Calendar cutoff",
            "First-observed month",
        )
        st.caption(
            "Latest 24 cohorts/cutoffs only. Blank cells are absent observations, not zero activity. This is calendar activity, not age-aligned retention or verified acquisition cohorts; exposure and filtered membership differ."
        )
        table(cohort.sort("cohort", "snapshot"))

        # Age-aligned repeat purchase chart using cohort_summary (Wilson intervals)
        st.subheader("Age-aligned repeat purchase (cohort_summary)")
        cutoff_date = date.fromisoformat(chosen)
        first_date = collect(historical.select(pl.col("first_purchase").min())).item()
        if first_date is not None:
            # Get baskets for the chosen cutoff
            baskets_cutoff = pl.scan_parquet(
                [str(p) for p in (root / "events" / chosen).glob("*/baskets.parquet")]
            )
            # Filter to chosen cutoff population
            ids = rows.select("customer_id", "first_purchase").unique()
            cohort_data = cohort_summary(
                baskets_cutoff, ids.lazy(), cutoff_date, coverage, minimum=20
            )
            if not cohort_data.is_empty():
                # Show for 30-day horizon
                horizon_data = cohort_data.filter(pl.col("horizon") == 30)
                if not horizon_data.is_empty():
                    fig = go.Figure()
                    for status in ["eligible", "capture_unconfirmed", "insufficient_followup_or_n"]:
                        subset = horizon_data.filter(pl.col("status") == status)
                        if not subset.is_empty():
                            label = {
                                "eligible": "Eligible (Wilson 95%)",
                                "capture_unconfirmed": "Capture unconfirmed",
                                "insufficient_followup_or_n": "Insufficient follow-up or N",
                            }.get(status, status)
                            color = {
                                "eligible": "#0072B2",
                                "capture_unconfirmed": "#CC79A7",
                                "insufficient_followup_or_n": "#D55E00",
                            }.get(status, "#64748b")
                            fig.add_trace(
                                go.Scatter(
                                    x=subset["cohort"].dt.strftime("%Y-%m").to_list(),
                                    y=subset["rate"].to_list()
                                    if status == "eligible"
                                    else [None] * subset.height,
                                    mode="markers+lines" if status == "eligible" else "markers",
                                    name=label,
                                    marker=dict(size=10, color=color),
                                    error_y=dict(
                                        type="data",
                                        symmetric=False,
                                        array=(subset["high"] - subset["rate"]).to_list()
                                        if status == "eligible"
                                        else None,
                                        arrayminus=(subset["rate"] - subset["low"]).to_list()
                                        if status == "eligible"
                                        else None,
                                        visible=True,
                                    )
                                    if status == "eligible"
                                    else None,
                                    hovertemplate="Cohort: %{x}<br>Repeat rate: %{y:.1%}<br>Eligible: %{customdata[0]}<br>Repeats: %{customdata[1]}<extra></extra>",
                                    customdata=list(
                                        zip(
                                            subset["eligible_n"].to_list()
                                            if "eligible_n" in subset.columns
                                            else [None] * subset.height,
                                            subset["repeat_n"].to_list()
                                            if "repeat_n" in subset.columns
                                            else [None] * subset.height,
                                        )
                                    )
                                    if status == "eligible"
                                    else None,
                                )
                            )
                    # Mark first cohort as left-censored
                    first_cohort = cohort_data["cohort"].min()
                    if first_cohort is not None:
                        fig.add_vline(
                            x=first_cohort.strftime("%Y-%m"),
                            line_dash="dash",
                            line_color="red",
                            annotation_text="First cohort (left-censored)",
                            annotation_position="top left",
                        )
                    fig.update_xaxes(tickformat="%Y-%m")
                    fig.update_yaxes(range=[0, 1], tickformat=".0%")
                    chart(
                        fig,
                        "Age-aligned 30-day repeat purchase rate by cohort (Wilson 95% CI)",
                        "cohort_repeat_rate",
                        "Cohort month",
                        "Repeat rate",
                    )
                    st.caption(
                        "Age-aligned: repeat purchase within 30 days of first purchase. Wilson 95% CI for eligible cohorts (min 20 customers). First cohort marked as left-censored (first purchase = observation start)."
                    )
                    table(horizon_data.sort("cohort"))

        # Concentration (Lorenz) chart
        st.subheader("Customer concentration (Lorenz curve)")
        conc = concentration(rows)
        if conc["n"] > 0:
            # Plot Lorenz curve
            curve = conc["curve"]
            x_vals, y_vals = zip(*curve) if curve else ([], [])
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name="Lorenz curve",
                    line=dict(color="#0072B2", width=2),
                    hovertemplate="Population share: %{x:.1%}<br>Spend share: %{y:.1%}<extra></extra>",
                )
            )
            # Add equality line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Equality",
                    line=dict(color="gray", dash="dash"),
                    hoverinfo="skip",
                )
            )
            # Highlight top fractions
            for top in conc["top"]:
                fig.add_trace(
                    go.Scatter(
                        x=[top["actual_fraction"]],
                        y=[top["spend_share"]],
                        mode="markers",
                        name=f"Top {top['requested_fraction']:.0%}",
                        marker=dict(size=10, color="#D55E00"),
                        hovertemplate=f"Top {top['requested_fraction']:.0%} customers<br>Population share: {top['actual_fraction']:.1%}<br>Spend share: {top['spend_share']:.1%}<extra></extra>",
                    )
                )
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            chart(
                fig,
                f"Lorenz curve: concentration of 90-day spend (n={conc['n']})",
                "lorenz_curve",
                "Population share (customers)",
                "Spend share",
            )
            st.caption(
                f"Total spend: {conc['total']:,.2f} ({conc['positive_n']} with positive spend). Top fractions round up to whole customers."
            )
        table(cohort.sort("cohort", "snapshot"))
    elif view == "Transitions":
        if len(history) < 2:
            st.info("Choose a cutoff with an earlier snapshot to compare transitions.")
            return
        origin = st.selectbox("Origin cutoff", history[:-1], index=len(history) - 2)
        dim = st.selectbox("Transition layer", DIMS[:-1])
        before = apply_filters(
            pl.scan_parquet([str(p) for p in (root / "snapshots" / origin).glob("*.parquet")]),
            filters,
        )
        after = apply_filters(
            pl.scan_parquet([str(p) for p in (root / "snapshots" / chosen).glob("*.parquet")]),
            filters,
        )
        flows, counts = transition_summary(before, after, dim)
        if flows.is_empty():
            st.info("No matched customers at both cutoffs.")
            return
        # Show reconciliation counts
        st.caption(
            f"Reconciliation: origin={counts['origin_n']}, destination={counts['destination_n']}, "
            f"matched={counts['matched_n']}, origin_unmatched={counts['origin_unmatched']}, "
            f"destination_unmatched={counts['destination_unmatched']}. "
            f"Unmatched origin customers are not churn; unmatched destination customers are new arrivals."
        )
        # Sankey diagram
        sources = sorted(flows["from_label"].unique().to_list())
        targets = sorted(flows["to_label"].unique().to_list())
        fig = go.Figure(
            go.Sankey(
                node=dict(
                    label=["From: " + s for s in sources] + ["To: " + t for t in targets],
                    color=[get_color(s, color_map) for s in sources + targets],
                    pad=15,
                ),
                link=dict(
                    source=[sources.index(r["from_label"]) for r in flows.to_dicts()],
                    target=[len(sources) + targets.index(r["to_label"]) for r in flows.to_dicts()],
                    value=flows["len"].to_list(),
                ),
            )
        )
        chart(fig, f"{dim}: {origin} to {chosen}", "transition_sankey")
        # Heatmap
        xs, ys, z = bounded_grid(flows.to_dicts(), "from_label", "to_label", "rate", limit=20)
        fig = go.Figure(
            go.Heatmap(
                x=xs,
                y=ys,
                z=z,
                zmin=0,
                zmax=1,
                colorscale="Blues",
                colorbar=dict(tickformat=".0%"),
                hoverongaps=False,
            )
        )
        chart(
            fig,
            "Origin-normalized transition proportions",
            "transition_matrix",
            "Destination label",
            "Origin label",
        )
        st.caption(
            f"Matched customers: {flows['len'].sum():,}. Sankey shows the 30 largest of {flows.height} flows; heatmap axes are capped at 20 labels. "
            f"Rates use all matched, destination-filtered customers per origin, so displayed cells need not sum to 100%. "
            f"Missing pairs are blank. Unmatched origin customers are not churn; unmatched destination customers are new arrivals."
        )
        table(flows)
        st.download_button(
            "Export transition summary", export_csv(flows), "transitions.csv", "text/csv"
        )
    elif view == "Quality":
        inspection, quality = (
            read_json(root.parent / "inspection.json"),
            read_json(root / "quality.json"),
        )
        labels = [
            "invalid_rows",
            "invalid_dates",
            "unknown_department_rows",
            "unknown_category_rows",
        ]
        counts = pl.DataFrame({"issue": labels, "rows": [inspection.get(k, 0) for k in labels]})
        bars(counts, "issue", "rows", "Upload-wide quality flags", "quality_flags", color_map)
        st.caption(
            "Flags can overlap; these counts cover the upload, not the filtered customer population."
        )
        st.json(inspection)
        st.json(quality)
        table(pl.scan_parquet(root / "product_variation.parquet"))
    elif view == "Customer":
        consent = st.checkbox("Show identifiable customer details in this session")
        customer = st.text_input("Exact customer_id; no full ID list is loaded")
        if consent and customer:
            profile = rows.filter(pl.col("customer_id") == customer)
            if not coverage:
                profile = profile.select(pl.exclude("^future_.*$", "^eligible_followup_.*$"))
            table(profile)
            ids = rows.filter(pl.col("customer_id") == customer).select("customer_id")
            paths = list((root / "events" / chosen).glob("*/baskets.parquet"))
            timeline = collect(
                pl.scan_parquet(paths)
                .join(ids, on="customer_id", how="inner", validate="m:1")
                .group_by("date")
                .agg(pl.col("spend").sum().alias("spend"), pl.len().alias("baskets"))
                .sort("date")
                .with_columns(pl.col("spend").cum_sum().alias("cumulative_spend"))
                .tail(PAGE_ROWS)
            )
            if timeline.height:
                for box, metric in zip(st.columns(2), ["spend", "cumulative_spend"]):
                    with box:
                        chart(
                            go.Figure(
                                go.Scatter(
                                    x=timeline["date"].to_list(),
                                    y=timeline[metric].to_list(),
                                    mode="lines+markers",
                                )
                            ),
                            metric,
                            "customer_" + metric,
                            "Purchase day",
                            "Recorded amount",
                        )
                st.caption(
                    "Latest 2,000 purchase days; cumulative values include all historical purchase days through this cutoff."
                )
            else:
                st.info("Customer not found in this filtered population.")
    else:
        visual_manifest = dict(
            manifest, result_cutoff=chosen, population_filters=filters, chart_sample_limit=PAGE_ROWS
        )
        st.json(visual_manifest)
        st.json(read_json(root / "thresholds.json"))
        st.info(
            "Descriptive rules, not calibrated prediction, causal effects, permanent churn, profit CLV or price elasticity. No seasonal adjustment or package-attribute inference."
        )
        st.caption(
            "Colors are deterministic labels, not ranks or confidence. Charts support zoom, hover, legend toggles and PNG export. Only the selected view computes; no shared customer-data cache is used. Files remain on this app server in the temporary session workspace; clearing is not secure erasure."
        )
        st.download_button(
            "Export analysis and population manifest",
            json.dumps(visual_manifest, indent=2),
            "analysis_manifest.json",
            "application/json",
        )


def main():
    st.set_page_config(page_title="Retail behavioral science", page_icon="📊", layout="wide")
    st.title("Retail behavioral science dashboard")
    st.caption(
        "Polars analytics · bounded interactive charts · explicit eligibility and uncertainty"
    )
    st.sidebar.warning(
        "Uploads are processed on this server. Use a trusted, access-controlled deployment."
    )
    if st.sidebar.button("Clear data and results"):
        reset()
        st.session_state["upload_nonce"] = st.session_state.get("upload_nonce", 0) + 1
        st.rerun()
    nonce = st.session_state.get("upload_nonce", 0)
    uploaded = st.sidebar.file_uploader(
        "Upload Parquet or UTF-8 CSV", type=["parquet", "csv"], key=f"upload_{nonce}"
    )
    if uploaded is None:
        reset()
        st.info("Required columns: " + ", ".join(COLS))
        return
    if uploaded.size > MAX_BYTES:
        st.error("Upload limit is 100 MiB. Reduce the file or use the CLI.")
        return
    digest = hashlib.sha256(uploaded.getbuffer()).hexdigest()
    if st.session_state.get("fingerprint") != digest:
        reset()
        st.session_state["workspace"] = tempfile.TemporaryDirectory(prefix="retail_dashboard_")
        st.session_state["fingerprint"] = digest
        source = Path(st.session_state["workspace"].name) / (
            "input" + Path(uploaded.name).suffix.lower()
        )
        source.write_bytes(uploaded.getbuffer())
    root = Path(st.session_state["workspace"].name)
    source = next(root.glob("input.*"))
    kind = "parquet" if source.suffix == ".parquet" else "csv"
    mode = st.sidebar.selectbox(
        "Meaning of price", ["Select explicitly", "Unit selling price", "Line total"]
    )
    if mode == "Select explicitly":
        st.info("Confirm what price represents before analysis.")
        return
    fmt = st.sidebar.text_input("Date format (blank = ISO YYYY-MM-DD)")
    decimal = st.sidebar.checkbox("Decimal comma; no thousands separators")
    separator = (
        st.sidebar.selectbox("CSV delimiter", [",", ";", "\t", "|"]) if kind == "csv" else ","
    )
    timezone = st.sidebar.text_input(
        "Timezone (for timezone-aware Datetime)",
        value="",
        help="e.g., 'UTC', 'US/Eastern'. Leave empty for naive dates.",
    )
    base = dict(
        workspace=str(root),
        source=str(source),
        kind=kind,
        separator=separator,
        date_format=fmt or None,
        decimal_comma=decimal,
        price_mode="unit" if mode == "Unit selling price" else "line_total",
        sha256=digest,
        timezone=timezone or None,
    )
    preview_key = json.dumps(base, sort_keys=True)
    if st.session_state.get("preview_key") != preview_key:
        st.session_state.pop("preview_schema", None)
        st.session_state.pop("column_mapping", None)
        st.session_state["preview_key"] = preview_key
    if st.sidebar.button("Preview columns / change mapping"):
        drop_result()
        st.session_state.pop("inspection_key", None)
        try:
            worker(dict(base, action="preview"))
            st.session_state["preview_schema"] = read_json(root / "preview.json")
        except Exception as exc:
            st.error(str(exc))
            return
    preview = st.session_state.get("preview_schema")
    mapping = st.session_state.get("column_mapping", {})
    if preview:
        options = [""] + preview.get("columns", [])
        mapping = {}
        st.sidebar.subheader("Column mapping")
        for required in COLS:
            default = options.index(required) if required in options else 0
            selected = st.sidebar.selectbox(
                required, options, index=default, key=f"map_{root.name}_{required}"
            )
            if selected:
                mapping[required] = selected
        if len(mapping) != len(COLS) or len(set(mapping.values())) != len(COLS):
            st.info("Map every required field to a different source column.")
            return
        st.session_state["column_mapping"] = mapping
    if mapping:
        base["column_mapping"] = mapping
    inspection_key = json.dumps(base, sort_keys=True)
    if st.session_state.get("inspection_key") != inspection_key:
        drop_result()
        if not st.sidebar.button("Validate upload"):
            st.info(
                "Validate the upload to continue. Preview and mapping are optional for canonical column names."
            )
            return
        try:
            with st.spinner("Validating upload in a separate worker..."):
                worker(dict(base, action="inspect"))
            st.session_state["inspection_key"] = inspection_key
            st.session_state.pop("preview_schema", None)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
            if (root / "inspection.json").exists():
                st.json(read_json(root / "inspection.json"))
            return
    metadata = read_json(root / "inspection.json")
    first, last = (
        date.fromisoformat(metadata["minimum_date"]),
        date.fromisoformat(metadata["maximum_date"]),
    )
    data_span_days = (last - first).days
    
    # Auto-calculate smart defaults
    # Default cutoff: leave 90 days for follow-up (so 30-day follow-up works)
    default_chosen = last - timedelta(days=90)
    if default_chosen < first:
        default_chosen = first + timedelta(days=max(1, data_span_days // 4))
    
    # Auto-calculate number of historical cutoffs (monthly, max 12)
    default_count = min(12, max(2, data_span_days // 30))
    
    # Auto-calculate partitions based on customer count (target ~500 customers/partition)
    valid_customers = metadata["valid_customers"]
    default_partitions = max(1, min(32, valid_customers // 500))

    st.sidebar.caption(
        f"{metadata['rows']:,} rows · {valid_customers:,} customers · {first} to {last} "
        f"({data_span_days} days)"
    )

    # Threshold calibration settings (auto by default)
    with st.sidebar.expander("Threshold calibration (auto)", expanded=False):
        st.caption(
            "Thresholds are auto-calibrated from the earliest snapshot with "
            "sufficient eligible customers. Uses distribution-free CIs with stability gates."
        )
        min_ref_pop = st.number_input(
            "Min reference population",
            min_value=10,
            max_value=10000,
            value=100,
            help="Minimum eligible customers for stable quantile estimates. "
                 "Lower = more snapshots eligible; higher = more stable thresholds."
        )
        st.caption(
            f"Default quantiles: value tier {QUANTILE_VALUE_LOW:.0%}/{QUANTILE_VALUE_HIGH:.0%}, "
            f"style {QUANTILE_STYLE_LOW:.0%}/{QUANTILE_STYLE_HIGH:.0%}. "
            f"CI width gate: 50% of estimate."
        )

    exclude = st.sidebar.checkbox("Exclude invalid records after review")
    if metadata["invalid_rows"] and not exclude:
        st.warning("Review invalid rows and explicitly select exclusion to continue.")
        st.json(metadata)
        return
    
    # Coverage confirmation - highlighted
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Coverage confirmation required for future-purchase analysis**")
    st.sidebar.caption(
        "Check this only if you have COMPLETE purchase data through the coverage end date. "
        "Without it, future-purchase rates and outcome estimates are suppressed."
    )
    coverage = st.sidebar.checkbox("I confirm complete purchase capture through the date below")
    
    end = (
        st.sidebar.date_input(
            "Confirmed coverage end",
            value=last,
            min_value=last,
            max_value=max(last, date.today()),
            disabled=not coverage,
            help="Must be ≥ latest data date. Future-purchase windows cannot extend beyond this."
        )
        if coverage
        else last
    )
    
    # Latest analysis cutoff - auto-default to leave room for follow-up
    st.sidebar.markdown("**📅 Analysis cutoffs**")
    st.sidebar.caption(
        f"Choose a cutoff that leaves future data for follow-up. "
        f"Recommended: {last - timedelta(days=90)} (leaves 90 days for 30/90-day follow-up)."
    )
    chosen = st.sidebar.date_input(
        "Latest analysis cutoff",
        value=default_chosen,
        min_value=first + timedelta(days=30),
        max_value=end - timedelta(days=1) if coverage else end,
        help="Cutoff date for feature computation. Future windows start after this date."
    )
    
    # Auto-calculated cutoffs
    dates = cutoffs(chosen, first, default_count)
    st.sidebar.caption(f"Cutoffs (monthly, {len(dates)}): " + ", ".join(map(str, dates)))
    
    # Override option for advanced users
    with st.sidebar.expander("Advanced: override auto settings", expanded=False):
        count = st.slider("Historical cutoffs", 1, 12, default_count)
        dates = cutoffs(chosen, first, count)
        partitions = st.slider("Customer partitions", 1, 32, default_partitions)
        st.caption(f"Partitions: {partitions} (≈{valid_customers//max(1,partitions)} customers each)")

    job = dict(
        base,
        action="analyze",
        snapshots=[str(x) for x in dates],
        buckets=partitions,
        exclude_invalid=exclude,
        coverage_confirmed=coverage,
        observed_through=str(end),
        min_reference_population=min_ref_pop,
    )
    result_key = json.dumps(job, sort_keys=True)
    if st.session_state.get("result_key") != result_key:
        drop_result()
    if st.sidebar.button("Run analysis", type="primary"):
        drop_result()
        target = root / ("run_" + uuid.uuid4().hex)
        try:
            with st.spinner("Building historical features and segments; 10-minute limit..."):
                worker(dict(job, output=str(target)))
            st.session_state["result"], st.session_state["result_key"] = str(target), result_key
        except Exception as exc:
            st.error(str(exc))
            if (target / "quality.json").exists():
                st.json(read_json(target / "quality.json"))
            shutil.rmtree(target, ignore_errors=True)
            return
    if not coverage:
        st.warning("Coverage is unconfirmed: future-purchase comparisons are suppressed.")
    if st.session_state.get("result"):
        try:
            display_results(Path(st.session_state["result"]), coverage)
        except Exception as exc:
            st.error(
                "A view could not render. Check output schemas, dependencies and server resources."
            )
            st.caption(type(exc).__name__)
    else:
        st.json(metadata)
        st.info("Run analysis to create the visual dashboard.")


if __name__ == "__main__":
    main()
