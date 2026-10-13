"""Cohorts view module."""

from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from retail_analytics.visualization.chart_components import (
    apply_filters,
    bounded_grid,
    collect,
    read_json,
    safe_list,
    table,
)
from retail_analytics.visualization.visual_insights_core import (
    cohort_summary,
    concentration,
)


def render_cohorts(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Cohorts view with calendar activity heatmap, age-aligned repeat, and concentration."""

    manifest = read_json(root / "run.json")
    history = [x for x in manifest["snapshots"] if x <= st.session_state.get("chosen", "")]

    chosen = st.session_state.get("chosen", str(date.today()))
    chosen_date = date.fromisoformat(chosen)

    # Calendar activity heatmap (existing)
    st.subheader("Calendar Activity Heatmap")
    try:
        historical = apply_filters(
            pl.scan_parquet(
                [str(p) for d in history for p in (root / "snapshots" / d).glob("*.parquet")]
            ),
            {},
        )

        cohort = collect(
            historical.with_columns(pl.col("first_purchase").dt.truncate("1mo").alias("cohort"))
            .group_by("cohort", "snapshot")
            .agg(
                pl.len().alias("customers"),
                (pl.col("purchases_30d") > 0).mean().alias("activity"),
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
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Latest 24 cohorts/cutoffs only. Blank cells are absent observations, not zero activity. This is calendar activity, not age-aligned retention or verified acquisition cohorts; exposure and filtered membership differ."
        )
        table(cohort.sort("cohort", "snapshot"))
    except Exception as e:
        st.error(f"Cohort heatmap failed: {type(e).__name__}: {e}")

    # Age-aligned repeat purchase chart using cohort_summary (Wilson intervals)
    st.subheader("Age-Aligned Repeat Purchase (cohort_summary)")
    try:
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
                    for status in [
                        "eligible",
                        "capture_unconfirmed",
                        "insufficient_followup_or_n",
                    ]:
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
                                    x=safe_list(subset["cohort"].dt.strftime("%Y-%m")),
                                    y=safe_list(subset["rate"])
                                    if status == "eligible"
                                    else [None] * subset.height,
                                    mode="markers+lines" if status == "eligible" else "markers",
                                    name=label,
                                    marker=dict(size=10, color=color),
                                    error_y=dict(
                                        type="data",
                                        symmetric=False,
                                        array=safe_list(subset["high"] - subset["rate"])
                                        if status == "eligible"
                                        else None,
                                        arrayminus=safe_list(subset["rate"] - subset["low"])
                                        if status == "eligible"
                                        else None,
                                        visible=True,
                                    )
                                    if status == "eligible"
                                    else None,
                                    hovertemplate="Cohort: %{x}<br>Repeat rate: %{y:.1%}<br>Eligible: %{customdata[0]}<br>Repeats: %{customdata[1]}<extra></extra>",
                                    customdata=list(
                                        zip(
                                            safe_list(subset["eligible_n"])
                                            if "eligible_n" in subset.columns
                                            else [None] * subset.height,
                                            safe_list(subset["repeat_n"])
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
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Age-aligned: repeat purchase within 30 days of first purchase. Wilson 95% CI for eligible cohorts (min 20 customers). First cohort marked as left-censored (first purchase = observation start)."
                    )
                    table(horizon_data.sort("cohort"))
    except Exception as e:
        st.error(f"Age-aligned repeat purchase analysis failed: {type(e).__name__}: {e}")
        st.caption("This may be due to missing baskets data, or insufficient cohort data.")

    # Concentration (Lorenz) chart
    st.subheader("Customer Concentration (Lorenz Curve)")
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
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Total spend: {conc['total']:,.2f} ({conc['positive_n']} with positive spend). Top fractions round up to whole customers."
        )
    table(cohort.sort("cohort", "snapshot"))


__all__ = ["render_cohorts"]
