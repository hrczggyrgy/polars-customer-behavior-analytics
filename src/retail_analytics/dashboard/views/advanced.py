"""Advanced Analytics views: Behavior×Trajectory, Purchase Rhythm, Basket Evolution, Category Sankey, Transitions, Quality, Customer."""

from datetime import date
from pathlib import Path
import math

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from retail_analytics.visualization.chart_components import (
    ChartEvidence,
    ChartCard,
    create_chart_card,
    build_color_map,
    get_color,
    safe_list,
)
from retail_analytics.visualization.visual_insights_core import (
    behavior_trajectory_matrix,
    purchase_rhythm_panel,
    basket_composition_slope,
    segment_alluvial_flow,
    transition_summary,
    concentration,
    full_population_distribution,
)
from retail_analytics.dashboard.app import read_json, table, apply_filters, PAGE_ROWS


def render_behavior_trajectory(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Behavior × Trajectory Bubble Matrix."""
    st.subheader("Behavior × Trajectory Bubble Matrix")
    st.caption(
        "X-axis: behavioral style (behavior). Y-axis: trajectory tags. "
        "Bubble size: customer count. Bubble color: median 90-day spend. "
        "Shows which style-trajectory combinations are growing, declining, or at risk."
    )

    try:
        matrix = behavior_trajectory_matrix(rows)
    except Exception as e:
        st.error(f"behavior_trajectory_matrix failed: {type(e).__name__}: {e}")
        st.caption("This may be due to missing behavior/trajectory columns.")
        return

    if matrix.is_empty():
        st.info("No behavior-trajectory combinations in this population.")
        return

    # Filter to combinations with sufficient customers
    min_customers = st.number_input(
        "Minimum customers per combination", 1, 1000, 5, key="bt_min"
    )
    plot_data = matrix.filter(pl.col("customers") >= min_customers)

    if plot_data.is_empty():
        st.info(f"No combinations with ≥ {min_customers} customers.")
        return

    # Bubble chart
    fig = go.Figure()

    for _, row in plot_data.iter_rows(named=True):
        fig.add_trace(
            go.Scatter(
                x=[row["behavior"]],
                y=[row["trajectory"]],
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
                customdata=[
                    [
                        row["customers"],
                        row["median_spend"] or 0,
                        row["future_30d_rate"] or 0,
                        row["future_90d_rate"] or 0,
                    ]
                ],
                showlegend=False,
            )
        )

    card = create_chart_card(
        key="bt_matrix",
        title=f"Behavior × Trajectory Matrix (min {min_customers} customers)",
        figure=fig,
        question="Which behavior-trajectory combinations are most common and how do they differ in future outcomes?",
        takeaway=f"{plot_data.height} combinations; total customers: {plot_data['customers'].sum():,}",
        evidence=ChartEvidence(
            numerator="Customer count per combination",
            denominator="Total filtered population",
            population_size=plot_data["customers"].sum(),
            comparison_period="Current snapshot",
            confidence_interval="Not applicable",
            interval_type="Descriptive scatter",
            eligibility_criteria=f"Min customers per combination: {min_customers}",
        ),
        how_to_read=(
            "X-axis: behavioral style. Y-axis: trajectory. "
            "Bubble size = customer count. Color = median 90d spend. "
            "Hover shows exact counts, spend, and future rates."
        ),
        caveat=(
            "Behavior and trajectory tags can overlap; matrix preserves multi-tag combinations. "
            "Future rates only shown for eligible follow-up customers."
        ),
        next_step="Check 'Transitions' view for label migration patterns.",
        alt_description="Bubble matrix showing behavior × trajectory combinations with size by customer count and color by median spend.",
    )
    card.render()

    # Summary table with future outcome rates
    from retail_analytics.dashboard.app import table
    table(
        plot_data.select(
            "behavior",
            "trajectory",
            "customers",
            "median_spend",
            "mean_spend",
            "future_30d_rate",
            "future_90d_rate",
            "eligible_30d",
            "eligible_90d",
        )
    )
    st.caption(
        "Bubble size = customer count. Color = median 90-day spend. "
        "Future rates only shown for eligible follow-up customers. "
        "Behavior and trajectory tags can overlap; matrix preserves multi-tag combinations."
    )


def render_purchase_rhythm(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Purchase Rhythm Panel."""
    st.subheader("Purchase Rhythm Panel")
    st.caption(
        "Distributions of median_gap (typical cadence), gap_cv (irregularity), and lateness (overdue). "
        "Segment-level summary shows how rhythm differs by behavioral style. "
        "Regular shoppers: low gap_cv, low lateness. Episodic: high median_gap, variable gap_cv."
    )

    from retail_analytics.visualization.visual_insights_core import purchase_rhythm_panel

    rhythm = purchase_rhythm_panel(rows)

    if rhythm["segment_rhythm"].is_empty():
        st.info("No style-eligible customers in this population.")
    else:
        # Three histograms
        col1, col2, col3 = st.columns(3)

        with col1:
            if rhythm["median_gap_hist"].height > 0:
                fig = go.Figure(
                    go.Histogram(
                        x=rhythm["median_gap_hist"]["median_gap"].to_list(),
                        nbinsx=30,
                        marker_color="#0072B2",
                        opacity=0.7,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if rhythm["gap_cv_hist"].height > 0:
                fig = go.Figure(
                    go.Histogram(
                        x=rhythm["gap_cv_hist"]["gap_cv"].to_list(),
                        nbinsx=30,
                        marker_color="#E69F00",
                        opacity=0.7,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

        with col3:
            if rhythm["lateness_hist"].height > 0:
                fig = go.Figure(
                    go.Histogram(
                        x=rhythm["lateness_hist"]["lateness"].to_list(),
                        nbinsx=30,
                        marker_color="#D55E00",
                        opacity=0.7,
                    )
                )
                fig.add_vline(
                    x=1.0,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="On cadence",
                    annotation_position="top",
                )
                fig.add_vline(
                    x=2.0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Overdue",
                    annotation_position="top",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Segment rhythm summary
        st.write("**Segment Rhythm Summary**")
        seg = rhythm["segment_rhythm"]
        table(
            seg.select(
                "behavior",
                "customers",
                "median_gap",
                "gap_cv",
                "lateness",
                "median_recency",
                "avg_frequency_90d",
            )
        )
        st.caption(
            "Median gap = typical inter-purchase interval. Gap CV = irregularity (std/mean). "
            "Lateness = recency / median_gap (≥2 = overdue). "
            "Only style-eligible customers (days_365d≥6, tenure≥90, gap_count≥4) included."
        )


def render_basket_evolution(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Basket Evolution (Slope Chart)."""
    st.subheader("Basket Composition Slope Chart")
    st.caption(
        "Before/after comparison for customers with comparable 90-day windows (tenure ≥ 179 days). "
        "Shows shifts in breadth, AOV, spend, and product familiarity shares. "
        "Only customers with finite recent and previous metrics are included."
    )

    from retail_analytics.visualization.visual_insights_core import basket_composition_slope

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
            summary["mean_breadth_previous"][0],
            summary["mean_aov_previous"][0],
            summary["mean_spend_previous"][0],
            summary["mean_new_share"][0],
            summary["mean_repeat_share"][0],
            summary["mean_established_share"][0],
        ]
        recent_vals = [
            summary["mean_breadth_recent"][0],
            summary["mean_aov_recent"][0],
            summary["mean_spend_recent"][0],
            summary["mean_new_share"][0],
            summary["mean_repeat_share"][0],
            summary["mean_established_share"][0],
        ]

        for i, (m, p, r) in enumerate(zip(metrics, prev_vals, recent_vals)):
            color = "#009E73" if r > p else ("#D55E00" if r < p else "#64748b")
            fig.add_trace(
                go.Scatter(
                    x=["Previous 90d", "Recent 90d"],
                    y=[p, r],
                    mode="lines+markers",
                    name=m,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    hovertemplate=f"{m}: %{{y:.3f}}<extra></extra>",
                )
            )

        st.plotly_chart(fig, use_container_width=True)

        # Dumbbell for median values
        fig2 = go.Figure()
        metrics2 = ["Breadth", "AOV"]
        prev_med = [summary["median_breadth_previous"][0], summary["median_aov_previous"][0]]
        recent_med = [summary["median_breadth_recent"][0], summary["median_aov_recent"][0]]

        for i, (m, p, r) in enumerate(zip(metrics2, prev_med, recent_med)):
            color = "#009E73" if r > p else ("#D55E00" if r < p else "#64748b")
            fig2.add_trace(
                go.Scatter(
                    x=[p, r],
                    y=[m, m],
                    mode="lines",
                    line=dict(color=color, width=4),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=[p],
                    y=[m],
                    mode="markers",
                    marker=dict(size=12, color="#E69F00"),
                    name="Previous" if i == 0 else None,
                    hovertemplate=f"{m} previous: %{{x:.2f}}<extra></extra>",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=[r],
                    y=[m],
                    mode="markers",
                    marker=dict(size=12, color="#0072B2"),
                    name="Recent" if i == 0 else None,
                    hovertemplate=f"{m} recent: %{{x:.2f}}<extra></extra>",
                )
            )

        st.plotly_chart(fig2, use_container_width=True)

        # Customer-level scatter (sampled)
        if sample.height > 0:
            st.write("**Customer-level shifts (sampled)**")
            metric_choice = st.selectbox(
                "Metric",
                ["Breadth", "AOV", "Spend", "Frequency change", "Basket change"],
                key="basket_metric",
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
                fig3.add_trace(
                    go.Scattergl(
                        x=part[x_col].to_list(),
                        y=part[y_col].to_list(),
                        mode="markers",
                        name=str(name),
                        marker=dict(size=6, opacity=0.55, color=get_color(name, color_map)),
                        hovertemplate=f"Value tier: %{{fullData.name}}<br>{x_title}: %{{x}}<br>{y_title}: %{{y}}<extra></extra>",
                    )
                )

            # Identity line
            max_val = max(sample[x_col].max(), sample[y_col].max())
            min_val = min(sample[x_col].min(), sample[y_col].min())
            fig3.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            f"Comparable customers: {comp_n:,} (tenure ≥ 179 days, non-zero previous purchases). "
            "Slope chart shows mean shifts. Dumbbell shows median shifts. "
            "Identity line: points above = increase, below = decrease. "
            "Missing/recent/previous values excluded from calculations."
        )


def render_category_sankey(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Category Sankey - Category Participation Comparison."""
    from retail_analytics.visualization.visual_insights_core import category_participation_comparison
    from retail_analytics.dashboard.app import read_json, apply_filters, collect

    st.subheader("Category Participation Comparison")
    st.caption(
        "Shows category participation change between two snapshots using 365-day windows. "
        "Participation = customers buying from category / eligible population for each snapshot. "
        "Uses validated transaction data for both periods. "
        "Categories appearing in only one period are shown with zero participation in the other."
    )

    manifest = read_json(root.parent / "run.json")
    history = [x for x in manifest["snapshots"] if x <= st.session_state.get("chosen", "")]

    if len(history) < 2:
        st.info("Need at least two cutoffs. Select an earlier snapshot in the sidebar.")
        return

    origin = st.selectbox("Previous cutoff", history[:-1], index=len(history) - 2, key="cat_origin")

    # Load validated data
    validated_path = root / "validated.parquet"
    if not validated_path.exists():
        st.info("No validated data available.")
        return

    validated = pl.scan_parquet(validated_path)

    # Get customer populations for both snapshots
    curr_ids = rows.select("customer_id").unique()
    curr_n = collect(curr_ids.select(pl.len())).item()

    prev_rows = apply_filters(
        pl.scan_parquet(
            [str(p) for p in (root / "snapshots" / origin).glob("*.parquet")]
        ),
        {},
    )
    prev_ids = prev_rows.select("customer_id").unique()
    prev_n = collect(prev_ids.select(pl.len())).item()

    # Current cutoff date ranges (365-day window)
    chosen = st.session_state.get("chosen", str(date.today()))
    cutoff_curr = date.fromisoformat(chosen)
    cutoff_prev = date.fromisoformat(origin)

    from retail_analytics.core.constants import WINDOW_YEAR

    curr_start = cutoff_curr - timedelta(days=WINDOW_YEAR - 1)
    curr_end = cutoff_curr
    prev_start = cutoff_prev - timedelta(days=WINDOW_YEAR - 1)
    prev_end = cutoff_prev

    # Current participation
    curr_cats = (
        validated.filter(pl.col("date").is_between(curr_start, curr_end))
        .join(curr_ids, on="customer_id", how="semi")
        .group_by("department", "category")
        .agg(
            pl.col("customer_id").n_unique().alias("curr_customers"),
            pl.col("value").sum().alias("curr_spend"),
        )
        .with_columns(
            pl.lit(curr_n).alias("curr_population"),
            (pl.col("curr_customers") / pl.lit(curr_n)).alias("curr_participation"),
        )
    )

    # Previous participation
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

    # Full join with coalesce to preserve previous-only and current-only categories
    merged = collect(
        curr_cats.join(
            prev_cats, on=["department", "category"], how="full", coalesce=True
        )
        .with_columns(
            pl.col("curr_participation").fill_null(0),
            pl.col("prev_participation").fill_null(0),
            pl.col("curr_customers").fill_null(0),
            pl.col("prev_customers").fill_null(0),
            pl.col("curr_spend").fill_null(0),
            pl.col("prev_spend").fill_null(0),
            pl.col("curr_population").fill_null(curr_n),
            pl.col("prev_population").fill_null(prev_n),
        )
        .with_columns(
            (pl.col("curr_participation") - pl.col("prev_participation")).alias(
                "part_change"
            ),
            pl.concat_str(
                [pl.col("department"), pl.lit(" / "), pl.col("category")]
            ).alias("label"),
        )
        .sort("curr_participation", descending=True)
    )

    if merged.is_empty():
        st.info("No category participation data for one or both periods.")
    else:
        top_n = st.slider("Top N categories", 5, 50, 20, key="cat_sankey_top")
        display = merged.head(top_n)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=display["label"].to_list(),
                x=display["curr_participation"].to_list(),
                name=f"Current ({chosen})",
                orientation="h",
                marker_color="#0072B2",
                customdata=list(
                    zip(
                        display["curr_customers"].to_list(),
                        display["curr_spend"].to_list(),
                        display["prev_customers"].to_list(),
                        display["prev_spend"].to_list(),
                        display["part_change"].to_list(),
                    )
                ),
                hovertemplate=(
                    "%{y}<br>Current participation: %{x:.1%}<br>"
                    "Current buyers: %{customdata[0]:,}<br>"
                    "Current spend: %{customdata[1]:,.2f}<br>"
                    "Previous buyers: %{customdata[2]:,}<br>"
                    "Previous spend: %{customdata[3]:,.2f}<br>"
                    "Change (pp): %{customdata[4]:.2f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Bar(
                y=display["label"].to_list(),
                x=display["prev_participation"].to_list(),
                name=f"Previous ({origin})",
                orientation="h",
                marker_color="#E69F00",
                opacity=0.6,
                customdata=list(
                    zip(
                        display["curr_customers"].to_list(),
                        display["curr_spend"].to_list(),
                        display["prev_customers"].to_list(),
                        display["prev_spend"].to_list(),
                        display["part_change"].to_list(),
                    )
                ),
                hovertemplate=(
                    "%{y}<br>Previous participation: %{x:.1%}<br>"
                    "Current buyers: %{customdata[0]:,}<br>"
                    "Current spend: %{customdata[1]:,.2f}<br>"
                    "Previous buyers: %{customdata[2]:,}<br>"
                    "Previous spend: %{customdata[3]:,.2f}<br>"
                    "Change (pp): %{customdata[4]:.2f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        table(
            display.select(
                "label",
                "curr_population",
                "prev_population",
                "curr_participation",
                "prev_participation",
                "part_change",
                "curr_customers",
                "prev_customers",
                "curr_spend",
                "prev_spend",
            )
        )

        st.caption(
            "Participation = category buyers / period population (same denominator per period). "
            "Current period uses current filtered population; previous uses previous filtered population. "
            "For fixed-population comparison, use Category Participation view."
        )


__all__ = [
    "render_behavior_trajectory",
    "render_purchase_rhythm",
    "render_basket_evolution",
    "render_category_sankey",
]