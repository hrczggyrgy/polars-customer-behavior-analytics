"""Lifecycle & Repeat views: Cohort-Age Retention, Time to Repeat, Multi-Horizon Cohorts, Purchase Cadence, Cadence Validation."""

from datetime import date, timedelta
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from retail_analytics.visualization.chart_components import (
    ChartEvidence,
    ChartSpec,
    ChartCard,
    create_chart_card,
    SupportThresholds,
    EvidenceState,
    evaluate_evidence,
    build_color_map,
    get_color,
    safe_list,
    format_axis_title,
    check_horizon_eligibility,
)
from retail_analytics.visualization.visual_insights_core import (
    cohort_age_retention_heatmap,
    time_to_second_purchase,
    multi_horizon_cohort_repeat,
    cadence_map,
    cadence_return_validation,
)
from retail_analytics.dashboard.app import read_json


def render_cohort_age_retention(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Cohort-Age Retention Heatmap."""
    st.subheader("Cohort-Age Retention Heatmap")
    st.caption(
        "Question: How does observed activity change after the first recorded purchase? "
        "Rows: first-observed month (cohort). Columns: months since first purchase (age). "
        "Age-0 is the cohort month (baseline). Blank cells = not yet observed (cohort too young). "
        "Consistent 0-100% color scale across all cohorts for comparability."
    )
    
    min_cohort = st.number_input("Minimum cohort size", 5, 1000, 20, key="car_min_cohort")
    
    validated_path = root / "validated.parquet"
    if not validated_path.exists():
        st.info("No validated data available for cohort-age retention.")
        return
    
    validated = pl.scan_parquet(validated_path)
    cutoff_date = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
    
    try:
        heatmap_data = cohort_age_retention_heatmap(
            validated, rows, cutoff_date, min_cohort
        )
        
        if heatmap_data.is_empty():
            st.info("Insufficient data for cohort-age retention heatmap.")
            return
        
        # Determine max observed age per cohort for censoring
        cutoff_month = date(cutoff_date.year, cutoff_date.month, 1)
        heatmap_data = heatmap_data.with_columns(
            (
                (pl.lit(cutoff_month).dt.year() - pl.col("cohort_month").dt.year()) * 12
                + (pl.lit(cutoff_month).dt.month() - pl.col("cohort_month").dt.month())
            ).alias("max_age_observed")
        )
        
        # Pivot for heatmap
        pivot = heatmap_data.pivot(
            values="activity_rate",
            index="cohort_month",
            on="age_month",
        )
        active_pivot = heatmap_data.pivot(
            values="active_customers",
            index="cohort_month",
            on="age_month",
        )
        size_pivot = heatmap_data.pivot(
            values="cohort_size",
            index="cohort_month",
            on="age_month",
        )
        max_obs_pivot = heatmap_data.pivot(
            values="max_age_observed",
            index="cohort_month",
            on="age_month",
        )
        
        # Limit to last 24 cohorts/ages for display
        cohort_labels = sorted(pivot["cohort_month"].to_list())[-24:]
        age_cols = [c for c in pivot.columns if c != "cohort_month"]
        age_cols = sorted([c for c in age_cols if isinstance(c, int)])[-24:]
        
        z = []
        hover = []
        total_cohorts = 0
        total_customers = 0
        observed_cells = 0
        
        for cohort in cohort_labels:
            z_row = []
            hover_row = []
            for age in age_cols:
                cohort_row = heatmap_data.filter(
                    (pl.col("cohort_month") == cohort) & (pl.col("age_month") == age)
                )
                if cohort_row.height > 0:
                    rate = cohort_row["activity_rate"][0]
                    active = cohort_row["active_customers"][0]
                    size = cohort_row["cohort_size"][0]
                    max_obs = cohort_row["max_age_observed"][0]
                    total_customers += size
                    total_cohorts += 1
                    # Blank future cells
                    if age > max_obs:
                        z_row.append(None)
                        hover_row.append(
                            f"Cohort: {cohort}<br>Age: {age}m<br>Not yet observed"
                        )
                    else:
                        z_row.append(rate)
                        hover_row.append(
                            f"Cohort: {cohort}<br>Age: {age}m<br>Active: {active}/{size} ({rate:.1%})"
                        )
                        observed_cells += 1
                else:
                    z_row.append(None)
                    hover_row.append(f"Cohort: {cohort}<br>Age: {age}m<br>No data")
            z.append(z_row)
            hover.append(hover_row)
        
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=[str(a) for a in age_cols],
                y=[
                    c.strftime("%Y-%m") if hasattr(c, "strftime") else str(c)
                    for c in cohort_labels
                ],
                colorscale="Blues",
                zmin=0,
                zmax=1,
                colorbar=dict(tickformat=".0%", title="Activity rate"),
                hovertext=hover,
                hoverinfo="text",
                hoverongaps=False,
            )
        )
        
        card = create_chart_card(
            key="cohort_age_retention_card",
            title="Cohort-Age Retention Heatmap",
            figure=fig,
            question="How does observed activity change after the first recorded purchase?",
            takeaway=f"{total_cohorts} cohorts, {total_customers:,} total customers, {observed_cells} observed age-cohort cells",
            evidence=ChartEvidence(
                numerator="Active customers in cohort at each age",
                denominator="Original cohort size",
                population_size=total_customers,
                comparison_period="Monthly since first observed purchase",
                confidence_interval="Not shown (raw rates)",
                interval_type="Descriptive rates only",
                eligibility_criteria=f"Min cohort size: {min_cohort}",
            ),
            how_to_read=(
                "Rows: first-observed purchase month (cohort). Columns: months since first purchase (age). "
                "Color: proportion of original cohort making a purchase in each month (0-100%). "
                "Age-0 is the cohort month (baseline). Blank cells = not yet observed (cohort too young). "
                "Consistent 0-100% color scale across all cohorts for comparability."
            ),
            caveat=(
                "First-observed cohort may be left-censored (capture start ≠ acquisition). "
                "Cohorts have unequal observation windows - immature cohorts appear worse. "
                "Raw rates not comparable without exposure adjustment. "
                "This is observed activity retention, not contractual retention."
            ),
            next_step="Check 'Multi-Horizon Cohorts' for repeat rates with Wilson intervals by horizon.",
            alt_description="Heatmap showing cohort-age retention with cohorts as rows and months since first purchase as columns, color-coded by activity rate.",
        )
        card.render()
        
        table(heatmap_data.sort("cohort_month", "age_month"))
        
    except Exception as e:
        st.error(f"Cohort-age retention failed: {type(e).__name__}: {e}")


def render_time_to_repeat(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Time to Repeat (Kaplan-Meier)."""
    st.subheader("Time to Second Purchase: Kaplan–Meier Survival Curve")
    st.caption(
        "Non-contractual customer activity model (Schmittlein, Morrison & Colombo, 1987). "
        "Duration starts at first observed shopping date; ends at second distinct shopping date. "
        "Customers without a second purchase by cutoff are right-censored. "
        "Shaded bands: 95% Greenwood confidence intervals."
    )
    
    try:
        baskets_paths = list((root / "events" / st.session_state.get("chosen", "")).glob("*/baskets.parquet"))
        if not baskets_paths:
            st.info("No basket data available for this cutoff.")
            return
        
        baskets = pl.scan_parquet(baskets_paths)
        cutoff_date = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
        km = time_to_second_purchase(baskets, rows, cutoff_date, max_days=180)
        
        if km.is_empty():
            st.info("Insufficient data for survival curve.")
            return
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=km["days"].to_list(),
                y=(1 - km["survival_prob"]).to_list(),
                mode="lines",
                name="Cumulative repeat probability",
                line=dict(color="#0072B2", width=2),
                hovertemplate="Day: %{x}<br>Repeat prob: %{y:.1%}<br>At risk: %{customdata[0]}<br>Events: %{customdata[1]}<extra></extra>",
                customdata=list(zip(km["at_risk"].to_list(), km["events"].to_list())),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=km["days"].to_list(),
                y=(1 - km["upper_ci"]).to_list(),
                mode="lines",
                name="Upper CI (95%)",
                line=dict(color="#0072B2", width=1, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=km["days"].to_list(),
                y=(1 - km["lower_ci"]).to_list(),
                mode="lines",
                name="Lower CI (95%)",
                line=dict(color="#0072B2", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(0,114,178,0.1)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        
        for pct in [0.25, 0.5, 0.75]:
            idx = (km["cum_repeat_prob"] - pct).abs().arg_min()
            day = km["days"][idx]
            fig.add_vline(
                x=day,
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
                annotation_text=f"P{pct:.0%}={day}d",
                annotation_position="top",
            )
        
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        
        card = create_chart_card(
            key="km_survival",
            title="Time to Second Observed Shopping Day (Kaplan–Meier)",
            figure=fig,
            question="How long does it take for a customer to make a second observed shopping event?",
            takeaway=f"Median time to repeat: {km.filter(pl.col('cum_repeat_prob') >= 0.5)['days'].min()}d; {km.filter(pl.col('days') <= 30)['cum_repeat_prob'].max():.1%} repeat within 30 days",
            evidence=ChartEvidence(
                numerator="Observed second purchases",
                denominator="Customers at risk at day 1",
                population_size=km["at_risk"][0] if km.height > 0 else 0,
                comparison_period="Up to 180 days since first observed purchase",
                confidence_interval="Greenwood 95%",
                interval_type="Kaplan-Meier survival CI",
                eligibility_criteria="Customers with first purchase before cutoff",
            ),
            how_to_read=(
                "Blue line: cumulative repeat probability (1 - survival). "
                "Shaded band: 95% Greenwood confidence interval. "
                "Vertical dotted lines: 25th, 50th, 75th percentile days. "
                "Right-censored at cutoff for customers without second purchase."
            ),
            caveat=(
                "Right-censored at cutoff. First observed purchase ≠ true acquisition. "
                "Greenwood 95% CI. Research: Schmittlein, Morrison & Colombo (1987), Management Science."
            ),
            next_step="Check 'Cadence Validation' for cadence-based return probability.",
            alt_description="Kaplan-Meier survival curve showing cumulative repeat probability over days since first observed purchase with 95% confidence bands.",
        )
        card.render()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total = km["at_risk"][0] if km.height > 0 else 0
            st.metric("Customers at risk (day 1)", f"{total:,}")
        with col2:
            total_events = km["events"].sum()
            st.metric("Observed second purchases", f"{total_events:,}")
        with col3:
            median_day = km.filter(pl.col("cum_repeat_prob") >= 0.5)["days"].min()
            st.metric(
                "Median time to repeat",
                f"{median_day}d" if median_day is not None else "Not reached",
            )
        with col4:
            pct_30 = km.filter(pl.col("days") <= 30)["cum_repeat_prob"].max()
            st.metric(
                "Repeat within 30 days",
                f"{pct_30:.1%}" if pct_30 is not None else "N/A",
            )
        
        table(
            km.select(
                "days",
                "at_risk",
                "events",
                "censored",
                "cum_repeat_prob",
                "lower_ci",
                "upper_ci",
            )
        )
        
    except Exception as e:
        st.error(f"Time-to-repeat analysis failed: {type(e).__name__}: {e}")


def render_multi_horizon_cohorts(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Multi-Horizon Cohorts."""
    manifest = read_json(root / "run.json")
    chosen_date = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
    
    horizon_options = [30, 60, 90]
    horizon_eligibility = {}
    for h in horizon_options:
        eligible, reason = check_horizon_eligibility(manifest, h, chosen_date)
        horizon_eligibility[h] = (eligible, reason)
    
    if not coverage:
        st.warning(
            "⚠ **Coverage not confirmed** — Future outcome validation is unavailable.\n\n"
            "To enable multi-horizon cohorts, re-run analysis with coverage confirmation "
            "and a complete-through date."
        )
        return
    
    # Show eligibility for each horizon
    for h in horizon_options:
        eligible, reason = horizon_eligibility[h]
        if eligible:
            st.success(f"✅ **{h}-day horizon:** {reason}")
        else:
            st.error(f"❌ **{h}-day horizon:** {reason}")
    
    eligible_horizons = [h for h in horizon_options if horizon_eligibility[h][0]]
    if not eligible_horizons:
        st.info("No horizons are eligible for multi-horizon cohort analysis with the current coverage settings.")
        return
    
    try:
        min_cohort = st.number_input("Minimum cohort size", 5, 1000, 20)
        
        baskets_paths = list((root / "events" / st.session_state.get("chosen", "")).glob("*/baskets.parquet"))
        if not baskets_paths:
            st.info("No basket data available for this cutoff.")
            return
        
        baskets = pl.scan_parquet(baskets_paths)
        cutoff_date = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
        cohort_data = multi_horizon_cohort_repeat(
            baskets, rows, cutoff_date, coverage, eligible_horizons, min_cohort
        )
        
        if cohort_data.is_empty():
            st.info("No eligible cohorts.")
            return
        
        fig = go.Figure()
        colors = {30: "#0072B2", 60: "#009E73", 90: "#E69F00"}
        
        for horizon in eligible_horizons:
            h_data = cohort_data.filter(pl.col("horizon") == horizon).sort("cohort")
            if h_data.is_empty():
                continue
            
            eligible = h_data.filter(pl.col("status") == "eligible")
            if not eligible.is_empty():
                fig.add_trace(
                    go.Scatter(
                        x=safe_list(eligible["cohort"].dt.strftime("%Y-%m")),
                        y=safe_list(eligible["rate"]),
                        mode="lines+markers",
                        name=f"{horizon}d (eligible)",
                        line=dict(color=colors[horizon], width=2),
                        marker=dict(size=8),
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=safe_list(
                                eligible["upper_ci"] - eligible["rate"]
                            ),
                            arrayminus=safe_list(
                                eligible["rate"] - eligible["lower_ci"]
                            ),
                            visible=True,
                        ),
                        customdata=list(
                            zip(
                                safe_list(eligible["eligible_n"]),
                                safe_list(eligible["repeat_n"]),
                            )
                        ),
                        hovertemplate="Cohort: %{x}<br>Rate: %{y:.1%}<br>Eligible: %{customdata[0]:,}<br>Repeats: %{customdata[1]:,}<extra></extra>",
                    )
                )
            
            ineligible = h_data.filter(pl.col("status") != "eligible")
            if not ineligible.is_empty():
                fig.add_trace(
                    go.Scatter(
                        x=safe_list(ineligible["cohort"].dt.strftime("%Y-%m")),
                        y=[None] * ineligible.height,
                        mode="markers",
                        name=f"{horizon}d (insufficient)",
                        marker=dict(color=colors[horizon], size=8, symbol="x"),
                        showlegend=True,
                    )
                )
        
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
        
        card = create_chart_card(
            key="multi_horizon_cohorts",
            title="Age-Aligned Repeat Purchase Rates by Cohort (30/60/90d)",
            figure=fig,
            question="How do repeat purchase rates vary by cohort age and horizon?",
            takeaway=f"Age-aligned repeat rates for {len(eligible_horizons)} horizons across {cohort_data['cohort'].n_unique()} cohorts",
            evidence=ChartEvidence(
                numerator="Customers with repeat purchase within horizon",
                denominator="Original cohort size",
                population_size=None,
                comparison_period="30/60/90 days since first observed purchase",
                confidence_interval="Wilson 95% for eligible cohorts",
                interval_type="Binomial proportion CI",
                eligibility_criteria="Coverage confirmed + complete follow-up + min cohort size",
            ),
            how_to_read=(
                "Lines show age-aligned repeat rates by cohort month for each horizon. "
                "Markers with error bars: eligible cohorts with Wilson 95% CI. "
                "X markers: ineligible cohorts (insufficient follow-up or N). "
                "Red dashed line: first cohort (left-censored)."
            ),
            caveat=(
                "Age-aligned: repeat within X days of first purchase. Wilson 95% CI for eligible cohorts. "
                "First cohort left-censored. Caveat: first observed ≠ true acquisition."
            ),
            next_step="Check 'Cohort-Age Retention' for full retention heatmap.",
            alt_description="Line chart showing age-aligned repeat purchase rates by cohort month for 30, 60, and 90-day horizons with Wilson confidence intervals.",
        )
        card.render()
        
        table(cohort_data.sort("cohort", "horizon"))
        
    except Exception as e:
        st.error(f"Multi-horizon cohort analysis failed: {type(e).__name__}: {e}")


def render_purchase_cadence(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Purchase Cadence Map."""
    st.subheader("Purchase Cadence Map: Recency vs. Median Gap")
    st.caption(
        "X-axis: median_gap (typical inter-purchase interval in days). "
        "Y-axis: recency (days since last purchase). "
        "Reference lines: recency = median_gap (on cadence) and recency = 2 × median_gap (overdue). "
        "This visualises the lateness = recency / median_gap metric."
    )
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        evidence_options = collect(rows.select("evidence").unique().sort("evidence"))[
            "evidence"
        ].to_list()
        evidence_filter = st.multiselect(
            "Evidence level", evidence_options, key="cadence_evidence"
        )
    with col2:
        vt_options = collect(rows.select("value_tier").unique().sort("value_tier"))[
            "value_tier"
        ].to_list()
        vt_filter = st.multiselect("Value tier", vt_options, key="cadence_vt")
    
    cad = cadence_map(
        rows, evidence_filter=evidence_filter or None, value_tier_filter=vt_filter or None
    )
    sample = cad["sample"]
    eligible_n = cad["eligible_n"]
    total_n = cad["total_n"]
    
    if sample.is_empty():
        st.info("No customers with finite median_gap and recency in this population.")
        return
    
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
                customdata=list(
                    zip(
                        part["lateness"].to_list(),
                        part["evidence"].to_list(),
                        part["lifecycle"].to_list(),
                        part["gap_cv"].to_list(),
                        part["days_365d"].to_list(),
                    )
                ),
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
    
    # Reference lines
    max_gap = (
        max(sample["median_gap"].max(), sample["recency"].max())
        if sample.height > 0
        else 100
    )
    x_line = [0, max_gap * 1.1]
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=x_line,
            mode="lines",
            name="Recency = Median gap (on cadence)",
            line=dict(color="gray", dash="dash", width=1),
            hoverinfo="skip",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=[2 * v for v in x_line],
            mode="lines",
            name="Recency = 2× Median gap (overdue)",
            line=dict(color="red", dash="dash", width=1),
            hoverinfo="skip",
            showlegend=True,
        )
    )
    
    # Highlight regions
    fig.add_vrect(
        x0=0,
        x1=max_gap * 1.1,
        y0=0,
        y1=max_gap * 1.1,
        fillcolor="rgba(0,158,115,0.05)",
        line_width=0,
        annotation_text="On cadence",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#009E73",
    )
    fig.add_vrect(
        x0=0,
        x1=max_gap * 1.1,
        y0=max_gap * 1.1,
        y1=max_gap * 2.2,
        fillcolor="rgba(213,94,0,0.05)",
        line_width=0,
        annotation_text="Overdue (1-2×)",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#D55E00",
    )
    fig.add_vrect(
        x0=0,
        x1=max_gap * 1.1,
        y0=max_gap * 2.2,
        y1=max_gap * 3.3,
        fillcolor="rgba(213,94,0,0.1)",
        line_width=0,
        annotation_text="Overdue (≥2×)",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#D55E00",
    )
    
    card = create_chart_card(
        key="cadence_scatter",
        title="Purchase Cadence Map",
        figure=fig,
        question="Which customers are overdue relative to their typical purchase rhythm?",
        takeaway=f"{eligible_n:,} eligible customers; lateness = recency / median_gap",
        evidence=ChartEvidence(
            numerator="Customers with finite median_gap and recency",
            denominator="Style-eligible customers",
            population_size=eligible_n,
            comparison_period="Current snapshot",
            confidence_interval="Not applicable",
            interval_type="Descriptive scatter",
            eligibility_criteria="Style-eligible (days_365d≥6, tenure≥90, gap_count≥4)",
        ),
        how_to_read=(
            "X-axis: median_gap (typical inter-purchase interval in days). "
            "Y-axis: recency (days since last purchase). "
            "Grey line: recency = median gap (on cadence). "
            "Red line: recency = 2× median gap (overdue threshold). "
            "Colored regions: green = on cadence, light orange = 1-2× overdue, darker orange = ≥2× overdue. "
            "Lateness = recency / median_gap. Only style_eligible customers have meaningful median_gap."
        ),
        caveat=(
            "Shows up to 2,000 hash-sampled customers with finite median_gap > 0 and recency. "
            "Grey line: recency = median gap (on cadence). Red line: recency = 2× median gap (overdue threshold). "
            "Coloured regions: green = on cadence, light orange = 1-2× overdue, darker orange = ≥2× overdue. "
            "Lateness = recency / median_gap. Only style_eligible customers have meaningful median_gap."
        ),
        next_step="Check 'Cadence Validation' for return probability by lateness.",
        alt_description="Scatter plot showing recency vs median gap with cadence reference lines and overdue regions.",
    )
    card.render()
    
    table(
        sample.select(
            "customer_id",
            "median_gap",
            "recency",
            "lateness",
            "value_tier",
            "evidence",
            "lifecycle",
            "gap_cv",
            "days_365d",
        )
    )


def render_cadence_validation(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Cadence Validation."""
    manifest = read_json(root / "run.json")
    chosen_date = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
    
    horizon_options = [30, 90]
    horizon_eligibility = {}
    for h in horizon_options:
        eligible, reason = check_horizon_eligibility(manifest, h, chosen_date)
        horizon_eligibility[h] = (eligible, reason)
    
    if not coverage:
        st.warning(
            "⚠ **Coverage not confirmed** — Future outcome validation is unavailable.\n\n"
            "To enable cadence validation, re-run analysis with coverage confirmation "
            "and a complete-through date."
        )
        return
    
    for h in horizon_options:
        eligible, reason = horizon_eligibility[h]
        if eligible:
            st.success(f"✅ **{h}-day horizon:** {reason}")
        else:
            st.error(f"❌ **{h}-day horizon:** {reason}")
    
    eligible_horizons = [h for h in horizon_options if horizon_eligibility[h][0]]
    if not eligible_horizons:
        st.info("No horizons are eligible for cadence validation with the current coverage settings.")
        return
    
    try:
        max_lateness = st.slider("Max lateness to display", 2.0, 10.0, 5.0, 0.5)
        bins = st.slider("Number of lateness bins", 5, 20, 10)
        
        val = cadence_return_validation(
            rows, coverage, max_lateness=max_lateness, bins=bins
        )
        
        if "warning" in val.columns:
            st.warning(val["warning"][0])
        elif "error" in val.columns:
            st.error(val["error"][0])
        elif val.is_empty():
            st.info("No eligible customers with finite cadence metrics.")
        else:
            for horizon in eligible_horizons:
                h_data = val.filter(pl.col("horizon") == horizon)
                if h_data.is_empty():
                    continue
                
                fig = go.Figure()
                has_upper = h_data["upper_ci"].is_not_null().any()
                has_lower = h_data["lower_ci"].is_not_null().any()
                upper_diff = (h_data["upper_ci"] - h_data["purchase_rate"]).to_list() if has_upper else None
                lower_diff = (h_data["purchase_rate"] - h_data["lower_ci"]).to_list() if has_lower else None
                
                fig.add_trace(
                    go.Bar(
                        x=h_data["lateness_bin"].to_list(),
                        y=h_data["purchase_rate"].to_list(),
                        name=f"{horizon}d return rate",
                        marker_color="#0072B2" if horizon == 30 else "#E69F00",
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=upper_diff,
                            arrayminus=lower_diff,
                            visible=True,
                        )
                        if has_upper or has_lower
                        else None,
                        customdata=list(
                            zip(
                                h_data["eligible_n"].to_list(),
                                h_data["buyers"].to_list(),
                            )
                        ),
                        hovertemplate="Lateness bin: %{x}<br>Return rate: %{y:.1%}<br>Eligible: %{customdata[0]:,}<br>Buyers: %{customdata[1]:,}<extra></extra>",
                    )
                )
                fig.add_vline(
                    x="1.5–2.0" if "1.5–2.0" in h_data["lateness_bin"].to_list() else 2.0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="2× median_gap boundary",
                    annotation_position="top",
                )
                
                fig.update_yaxes(range=[0, 1], tickformat=".0%")
                
                card = create_chart_card(
                    key=f"cadence_val_{horizon}",
                    title=f"{horizon}-Day Return Rate by Lateness Bin",
                    figure=fig,
                    question="Does the lateness boundary correspond to actual changes in return probability?",
                    takeaway=f"{horizon}-day return rate by lateness bins with Wilson 95% CI",
                    evidence=ChartEvidence(
                        numerator="Customers with future purchase within horizon",
                        denominator="Eligible customers per lateness bin",
                        population_size=None,
                        comparison_period=f"{horizon}-day follow-up",
                        confidence_interval="Wilson 95% per bin",
                        interval_type="Binomial proportion CI",
                        eligibility_criteria="Coverage confirmed + style-eligible + complete follow-up",
                    ),
                    how_to_read=(
                        "Bars show future purchase rate per lateness bin. "
                        "Error bars: Wilson 95% CI (bins with ≥20 eligible). "
                        "Red dashed line: 2× median_gap boundary (overdue threshold). "
                        "Lateness = recency / median_gap."
                    ),
                    caveat=(
                        "Wilson 95% CI shown for bins with ≥20 eligible customers. "
                        "Lateness = recency / median_gap. Research: BG/NBD repeat-buying models."
                    ),
                    next_step="Check 'Purchase Cadence' for customer-level cadence map.",
                    alt_description=f"Bar chart showing {horizon}-day return rate by lateness bin with Wilson confidence intervals.",
                )
                card.render()
            
            table(
                val.select(
                    "horizon",
                    "lateness_bin",
                    "eligible_n",
                    "buyers",
                    "purchase_rate",
                    "lower_ci",
                    "upper_ci",
                )
            )
            st.caption(
                "Wilson 95% CI shown for bins with ≥20 eligible customers. "
                "Lateness = recency / median_gap. Research: BG/NBD repeat-buying models."
            )
            
    except Exception as e:
        st.error(f"Cadence validation failed: {type(e).__name__}: {e}")