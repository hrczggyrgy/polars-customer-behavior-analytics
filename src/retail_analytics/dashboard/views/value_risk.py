"""Value & Risk views: RFM Response Surface, Value-at-Risk, Revenue Bridge, Customer Momentum, Spend Concentration."""

from datetime import date
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
    check_horizon_eligibility,
)
from retail_analytics.visualization.visual_insights_core import (
    rfm_response_surface,
    value_at_risk_scatter,
    revenue_bridge,
    revenue_waterfall,
    momentum_map,
    concentration,
    behavior_trajectory_matrix,
)
from retail_analytics.dashboard.app import read_json


def render_rfm_response_surface(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """RFM Response Surface."""
    manifest = read_json(root / "run.json")
    chosen_date = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
    eligible, reason = check_horizon_eligibility(manifest, 30, chosen_date)
    
    if not coverage:
        st.warning(
            "⚠ **Coverage not confirmed** — Future outcome validation is unavailable.\n\n"
            "To enable RFM response surface, re-run analysis with coverage confirmation "
            "and a complete-through date."
        )
        return
    elif not eligible:
        st.error(f"❌ **30-day horizon ineligible:** {reason}")
        return
    
    st.success(f"✅ **30-day horizon:** {reason}")
    
    try:
        rec_bins = st.slider("Recency quantile bins", 3, 10, 5)
        freq_bins = st.slider("Frequency quantile bins", 3, 10, 5)
        
        surface = rfm_response_surface(
            rows, coverage, recency_bins=rec_bins, frequency_bins=freq_bins
        )
        
        if "warning" in surface.columns:
            st.warning(surface["warning"][0])
        elif "error" in surface.columns:
            st.error(surface["error"][0])
        elif surface.is_empty():
            st.info("No eligible customers for response surface.")
        else:
            z = []
            hover = []
            total_eligible = 0
            cells_with_ci = 0
            for r_bin in sorted(surface["recency_bin"].unique()):
                z_row = []
                hover_row = []
                for f_bin in sorted(surface["frequency_bin"].unique()):
                    cell = surface.filter(
                        (pl.col("recency_bin") == r_bin)
                        & (pl.col("frequency_bin") == f_bin)
                    )
                    if cell.height > 0:
                        z_row.append(
                            cell["rate"][0] if cell["rate"][0] is not None else None
                        )
                        n = cell["eligible_n"][0]
                        buyers = cell["buyers"][0]
                        total_eligible += n
                        lo = (
                            cell["lower_ci"][0] if cell["lower_ci"][0] is not None else None
                        )
                        hi = (
                            cell["upper_ci"][0] if cell["upper_ci"][0] is not None else None
                        )
                        if lo is not None:
                            cells_with_ci += 1
                        hover_row.append(
                            f"Recency: {r_bin}<br>Freq: {f_bin}<br>Rate: {cell['rate'][0]:.1%}<br>N: {n}<br>Buyers: {buyers}<br>CI: [{lo:.1%}, {hi:.1%}]"
                            if lo
                            else f"Recency: {r_bin}<br>Freq: {f_bin}<br>N: {n}"
                        )
                    else:
                        z_row.append(None)
                        hover_row.append(f"Recency: {r_bin}<br>Freq: {f_bin}<br>No data")
                z.append(z_row)
                hover.append(hover_row)
            
            fig = go.Figure(
                go.Heatmap(
                    z=z,
                    x=sorted(surface["frequency_bin"].unique()),
                    y=sorted(surface["recency_bin"].unique()),
                    colorscale="RdYlGn",
                    zmin=0,
                    zmax=1,
                    colorbar=dict(tickformat=".0%", title="30d repeat rate"),
                    hovertext=hover,
                    hoverinfo="text",
                )
            )
            
            card = create_chart_card(
                key="rfm_surface_card",
                title="RFM Response Surface: 30-Day Repeat Probability",
                figure=fig,
                question="Which combinations of recency and frequency show different future repeat rates?",
                takeaway=f"Surface shows {cells_with_ci} cells with Wilson 95% CI across {rec_bins}×{freq_bins} bins; total eligible: {total_eligible:,}",
                evidence=ChartEvidence(
                    numerator="Customers with repeat purchase within 30 days",
                    denominator="Eligible customers per recency×frequency bin",
                    population_size=total_eligible,
                    comparison_period="30-day follow-up",
                    confidence_interval="Wilson 95% per cell",
                    interval_type="Binomial proportion CI",
                    eligibility_criteria="Coverage confirmed + 30-day follow-up observed",
                ),
                how_to_read=(
                    "Heatmap cells show 30-day repeat rate (color: green=high, red=low). "
                    "X-axis: frequency quantile bins (purchases_90d). Y-axis: recency quantile bins. "
                    "Hover shows exact rate, eligible N, buyers, and Wilson CI. "
                    "Blank cells = insufficient eligible customers (<20)."
                ),
                caveat=(
                    "Quantile bins on recency and frequency_90d. Wilson 95% CI for cells with ≥20 eligible. "
                    "Descriptive association, not causal prediction or CLV forecast. "
                    "Multiple comparisons not multiplicity-adjusted. "
                    "First observed purchase ≠ true acquisition."
                ),
                next_step="Drill into 'Segments' view to see repeat rates by behavioral segment.",
                alt_description="Heatmap showing 30-day repeat probability across recency and frequency quantile bins with Wilson confidence intervals.",
            )
            card.render()
            
            table(
                surface.select(
                    "recency_bin",
                    "frequency_bin",
                    "eligible_n",
                    "buyers",
                    "rate",
                    "lower_ci",
                    "upper_ci",
                )
            )
            
    except Exception as e:
        st.error(f"RFM response surface failed: {type(e).__name__}: {e}")


def render_value_at_risk_scatter(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Value-at-Risk Scatter."""
    st.subheader("Value-at-Risk: High-Spend Customers Overdue on Cadence")
    st.caption(
        "X-axis: lateness = recency / median_gap (how overdue relative to typical cadence). "
        "Y-axis: spend_365d (recorded spend in last 365 days). "
        "Bubble size: purchases_365d. Color: value_tier or trajectory. "
        "Reference lines at 1× and 2× median gap. Identifies commercially important customers whose purchasing rhythm has deteriorated."
    )
    
    try:
        lateness_thresh = st.slider(
            "Overdue threshold (lateness)", 1.0, 5.0, 2.0, 0.25, key="var_thresh"
        )
        color_by = st.selectbox(
            "Color by", ["value_tier", "trajectory", "lifecycle", "evidence"], key="var_color"
        )
        sample_limit = st.slider(
            "Sample limit (scatter)", 500, 5000, 2000, 500, key="var_sample"
        )
        
        var_data = value_at_risk_scatter(
            rows, lateness_threshold=lateness_thresh, sample_limit=sample_limit
        )
        
        sample = var_data["sample"]
        summary = var_data["summary"]
        
        if sample.is_empty():
            st.info("No customers with finite cadence metrics in this population.")
            return
        
        # Summary metrics
        if not summary.is_empty():
            s = summary.row(0, named=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Eligible customers", f"{s['eligible_customers']:,}")
            with col2:
                st.metric("Total 365d spend", f"{s['total_spend_365d']:,.2f}")
            with col3:
                st.metric(
                    f"Overdue (≥{lateness_thresh}×)",
                    f"{s['overdue_customers']:,} ({s['overdue_share']:.1%})",
                )
            with col4:
                st.metric("At-risk 365d spend", f"{s['at_risk_spend_365d']:,.2f}")
        
        # Scatter plot
        fig = go.Figure()
        
        color_map_local = build_color_map(sample[color_by].unique().to_list())
        
        for name in sorted(sample[color_by].unique().to_list()):
            part = sample.filter(pl.col(color_by) == name)
            fig.add_trace(
                go.Scattergl(
                    x=part["lateness"].to_list(),
                    y=part["spend_365d"].to_list(),
                    mode="markers",
                    name=str(name),
                    marker=dict(
                        size=[
                            max(6, min(40, (p ** 0.5) * 1.5))
                            for p in part["purchases_365d"].to_list()
                        ],
                        opacity=0.6,
                        color=get_color(name, color_map_local),
                    ),
                    customdata=list(
                        zip(
                            part["recency"].to_list(),
                            part["median_gap"].to_list(),
                            part["spend_90d"].to_list(),
                            part["purchases_90d"].to_list(),
                            part["evidence"].to_list(),
                            part["lifecycle"].to_list(),
                        )
                    ),
                    hovertemplate=(
                        f"{color_by}: %{{fullData.name}}<br>"
                        "Lateness: %{x:.2f}<br>"
                        "Spend 365d: %{y:,.2f}<br>"
                        "Recency: %{customdata[0]:.1f} days<br>"
                        "Median gap: %{customdata[1]:.1f} days<br>"
                        "Spend 90d: %{customdata[2]:,.2f}<br>"
                        "Purchases 90d: %{customdata[3]}<br>"
                        "Evidence: %{customdata[4]}<br>"
                        "Lifecycle: %{customdata[5]}<extra></extra>"
                    ),
                )
            )
        
        # Reference lines
        max_x = (
            max(sample["lateness"].max(), lateness_thresh * 1.5) if sample.height > 0 else 5
        )
        fig.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            annotation_text="On cadence (1×)",
        )
        fig.add_vline(
            x=2.0,
            line_dash="dash",
            line_color="red",
            line_width=1,
            annotation_text="Overdue (2×)",
        )
        fig.add_vline(
            x=lateness_thresh,
            line_dash="dot",
            line_color="orange",
            line_width=2,
            annotation_text=f"Threshold ({lateness_thresh}×)",
        )
        
        # Shaded regions
        fig.add_vrect(
            x0=0,
            x1=1.0,
            fillcolor="rgba(0,158,115,0.05)",
            line_width=0,
            annotation_text="On cadence",
            annotation_position="top left",
        )
        fig.add_vrect(
            x0=1.0,
            x1=2.0,
            fillcolor="rgba(213,94,0,0.05)",
            line_width=0,
            annotation_text="1-2× overdue",
            annotation_position="top left",
        )
        fig.add_vrect(
            x0=2.0,
            x1=max_x,
            fillcolor="rgba(213,94,0,0.1)",
            line_width=0,
            annotation_text="≥2× overdue",
            annotation_position="top left",
        )
        
        card = create_chart_card(
            key="var_scatter",
            title=f"Value-at-Risk: Lateness vs 365d Spend (n={sample.height:,} sampled)",
            figure=fig,
            question="Which high-spend customers are overdue relative to their typical cadence?",
            takeaway=f"{summary['eligible_customers'][0] if not summary.is_empty() else 0:,} eligible; {summary['overdue_customers'][0] if not summary.is_empty() else 0:,} overdue (≥{lateness_thresh}×)",
            evidence=ChartEvidence(
                numerator="Overdue high-spend customers",
                denominator="Style-eligible customers with finite cadence",
                population_size=summary["eligible_customers"][0] if not summary.is_empty() else 0,
                comparison_period="365-day window",
                confidence_interval="Not applicable",
                interval_type="Descriptive scatter",
                eligibility_criteria="Finite median_gap > 0, finite recency, style-eligible",
            ),
            how_to_read=(
                "X-axis: Lateness = recency / median_gap (how overdue). "
                "Y-axis: Recorded spend in last 365 days. "
                "Bubble size: Purchase count in 365d. "
                "Grey dashed: on cadence (1×). Red dashed: overdue (2×). "
                "Orange dotted: user threshold. Green/Orange/Red regions: on cadence / 1-2× / ≥2× overdue."
            ),
            caveat=(
                "Not a churn prediction; describes observed inactivity relative to historical cadence. "
                "Spend is recorded amount, not profit. Customers may be inactive for many reasons."
            ),
            next_step="Check 'Customer Brief' for individual customer timeline.",
            alt_description="Scatter plot showing lateness (recency/median_gap) vs 365-day spend, bubble size by purchase count, colored regions for cadence status.",
        )
        card.render()
        
        st.caption(
            f"Shows up to {sample_limit:,} hash-sampled customers with finite median_gap > 0 and recency. "
            f"Full population: {summary['eligible_customers'][0] if not summary.is_empty() else 0:,} eligible, "
            f"{summary['overdue_customers'][0] if not summary.is_empty() else 0:,} overdue (≥{lateness_thresh}×). "
            f"Not a churn prediction; describes observed inactivity relative to historical cadence."
        )
        
        table(
            sample.select(
                "customer_id",
                "lateness",
                "spend_365d",
                "spend_90d",
                "purchases_365d",
                "value_tier",
                "trajectory",
                "evidence",
                "lifecycle",
            )
        )
        
    except Exception as e:
        st.error(f"Value-at-risk scatter failed: {type(e).__name__}: {e}")


def render_revenue_bridge(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Revenue Bridge (Calendar Period Decomposition)."""
    st.subheader("Revenue Bridge: Calendar-Period Decomposition")
    st.caption(
        "Compares two calendar periods (non-overlapping) with matched-population option. "
        "Decomposes spend change into: purchasing customers, baskets per customer, spend per basket (AOV). "
        "Also shows calendar-period trend with independent monthly windows."
    )
    
    try:
        validated_path = root / "validated.parquet"
        if not validated_path.exists():
            st.info("No validated data available for calendar-period trend.")
            return
        
        validated = pl.scan_parquet(validated_path)
        
        # Calendar-period trend
        st.write("**Calendar-Period Revenue Trend**")
        first_date = collect(rows.select(pl.col("first_purchase").min())).item()
        if first_date:
            from retail_analytics.visualization.visual_insights_core import calendar_period_revenue_trend
            
            trend_data = calendar_period_revenue_trend(
                validated, rows, first_date, date.fromisoformat(st.session_state.get("chosen", str(date.today()))), months=12
            )
            
            if not trend_data.is_empty():
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=trend_data["month"].to_list(),
                        y=trend_data["spend"].to_list(),
                        mode="lines+markers",
                        name="Monthly spend",
                        line=dict(color="#0072B2", width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=trend_data["month"].to_list(),
                        y=trend_data["buyers"].to_list(),
                        mode="lines+markers",
                        name="Purchasing customers",
                        line=dict(color="#009E73", width=2),
                        yaxis="y2",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=trend_data["month"].to_list(),
                        y=trend_data["aov"].to_list(),
                        mode="lines+markers",
                        name="AOV",
                        line=dict(color="#E69F00", width=2),
                        yaxis="y3",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=trend_data["month"].to_list(),
                        y=trend_data["baskets_per_buyer"].to_list(),
                        mode="lines+markers",
                        name="Baskets/buyer",
                        line=dict(color="#D55E00", width=2),
                        yaxis="y4",
                    )
                )
                
                fig.update_layout(
                    xaxis=dict(title="Month"),
                    yaxis=dict(title="Spend", side="left", color="#0072B2"),
                    yaxis2=dict(
                        title="Buyers", side="right", overlaying="y", color="#009E73"
                    ),
                    yaxis3=dict(
                        title="AOV",
                        side="right",
                        overlaying="y",
                        position=0.85,
                        color="#E69F00",
                    ),
                    yaxis4=dict(
                        title="Baskets/buyer",
                        side="right",
                        overlaying="y",
                        position=0.95,
                        color="#D55E00",
                    ),
                )
                
                card = create_chart_card(
                    key="cal_trend",
                    title="Calendar-Period Revenue Components (12 months)",
                    figure=fig,
                    question="How do monthly spend, buyers, AOV, and baskets/buyer trend over calendar periods?",
                    takeaway="12-month calendar trend with non-overlapping windows; fixed population",
                    evidence=ChartEvidence(
                        numerator="Monthly metrics",
                        denominator="Population customers",
                        population_size=None,
                        comparison_period="12 calendar months",
                        confidence_interval="Not applicable",
                        interval_type="Descriptive monthly series",
                        eligibility_criteria="Fixed population (current filtered customers). Partial months marked.",
                    ),
                    how_to_read=(
                        "Blue: Monthly spend. Green: Purchasing customers (right axis). "
                        "Orange: AOV (right axis). Red: Baskets/buyer (right axis). "
                        "Non-overlapping monthly windows. Independent observations per month."
                    ),
                    caveat=(
                        "Non-overlapping windows are independent. Rolling 90-day windows overlap and are NOT independent. "
                        "Fixed population = current filtered customers; partial months at edges may be incomplete."
                    ),
                    next_step="Check 'Revenue Bridge' below for period-over-period decomposition.",
                    alt_description="Multi-axis line chart showing monthly spend, buyers, AOV, and baskets per buyer over 12 calendar months.",
                )
                card.render()
                
                st.caption(
                    "Non-overlapping monthly windows. Fixed population (current filtered customers). Partial months marked."
                )
        
        # Revenue bridge
        st.write("**Revenue Decomposition Bridge**")
        
        from retail_analytics.dashboard.app import apply_filters
        
        manifest = read_json(root / "run.json")
        history = [x for x in manifest["snapshots"] if x <= st.session_state.get("chosen", "")]
        
        if len(history) < 2:
            st.info("Need at least two cutoffs for bridge comparison.")
        else:
            origin = st.selectbox(
                "Previous cutoff", history[:-1], index=len(history) - 2, key="bridge_origin"
            )
            match_pop = st.checkbox(
                "Match customers across periods",
                value=True,
                help="Use only customers present in both snapshots for frequency/AOV comparison",
            )
            
            before_rows = apply_filters(
                pl.scan_parquet(
                    [str(p) for p in (root / "snapshots" / origin).glob("*.parquet")]
                ),
                {},
            )
            cutoff_date = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
            bridge = revenue_bridge(
                rows, before_rows, cutoff_date, match_customers=match_pop
            )
            
            if bridge.is_empty():
                st.info("Could not compute bridge for this comparison.")
            else:
                # Waterfall chart
                fig = go.Figure()
                components = bridge["component"].to_list()
                values = bridge["value"].to_list()
                desc = bridge["description"].to_list()
                pop_types = bridge["population_type"].to_list()
                
                cumulative = [values[0]]
                for i in range(1, len(values) - 1):
                    cumulative.append(cumulative[-1] + values[i])
                cumulative.append(values[-1])
                
                colors = []
                for i, v in enumerate(values):
                    if i == 0 or i == len(values) - 1:
                        colors.append("#0072B2")
                    elif v >= 0:
                        colors.append("#009E73")
                    else:
                        colors.append("#D55E00")
                
                fig = go.Figure()
                fig.add_trace(
                    go.Waterfall(
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
                        customdata=list(zip(desc, pop_types)),
                        hovertemplate="%{x}<br>Value: %{y:,.2f}<br>%{customdata[0]}<br>Population: %{customdata[1]}<extra></extra>",
                    )
                )
                
                card = create_chart_card(
                    key="revenue_bridge",
                    title=f"Revenue Bridge: {origin} to {st.session_state.get('chosen')} ({'matched' if match_pop else 'unmatched'} population)",
                    figure=fig,
                    question="How much of observed spend change comes from customers active in both periods vs entering/leaving?",
                    takeaway=f"Spend change decomposed into customers, frequency, AOV, and residual ({'matched' if match_pop else 'unmatched'} population)",
                    evidence=ChartEvidence(
                        numerator="Component contributions to spend change",
                        denominator="Previous period spend",
                        population_size=None,
                        comparison_period=f"{origin} → {st.session_state.get('chosen')}",
                        confidence_interval="Not applicable",
                        interval_type="Arithmetic decomposition",
                        eligibility_criteria="Fixed-order bridge: N → F → A",
                    ),
                    how_to_read=(
                        "Waterfall bars: blue=start/end totals, green=positive contribution, red=negative. "
                        "Components: (1) Purchasing customers change × prev freq × prev AOV, "
                        "(2) Current customers × freq change × prev AOV, "
                        "(3) Current customers × current freq × AOV change. "
                        "Residual = interaction effects. Hover shows component description."
                    ),
                    caveat=(
                        "Fixed-order bridge: attribution depends on decomposition order. "
                        "Not causal attribution. Recorded spend, not profit. "
                        f"{'Matched customers across periods' if match_pop else 'Full snapshot populations'}."
                    ),
                    next_step="Check 'Revenue Waterfall' for calendar-period trend.",
                    alt_description="Waterfall chart decomposing spend change into customer count, frequency, AOV, and residual components.",
                )
                card.render()
                
                table(bridge)
                st.caption(
                    f"Fixed-order bridge: customers → frequency → AOV. {'Matched population' if match_pop else 'Full snapshot populations'}. "
                    f"Residual captures interaction effects. Order-dependent; not causal attribution."
                )
        
    except Exception as e:
        st.error(f"Revenue bridge failed: {type(e).__name__}: {e}")


def render_customer_momentum(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Customer Momentum Map."""
    st.subheader("Customer Momentum Map: Frequency vs. Basket Change")
    st.caption(
        "X-axis: frequency_change (recent 90d vs previous 90d purchase frequency). "
        "Y-axis: basket_change (recent 90d vs previous 90d AOV). "
        "Colour: value_tier. Only customers with comparable windows (tenure ≥ 179 days) and non-zero previous purchases are shown."
    )
    
    try:
        mom = momentum_map(rows)
    except Exception as e:
        st.error(f"momentum_map failed: {type(e).__name__}: {e}")
        st.caption(
            "This may be due to missing frequency_change/basket_change columns or no comparable customers."
        )
        return
    
    sample = mom["sample"]
    quad = mom["quadrant_summary"]
    not_comp = mom["not_comparable_n"]
    finite_n = mom["finite_n"]
    
    if sample.is_empty():
        st.info("No customers with comparable frequency and basket changes in this population.")
        return
    
    # Quadrant summary
    st.write("**Quadrant Summary (all comparable customers)**")
    if not quad.is_empty():
        quad_display = quad.with_columns(
            pl.col("quadrant")
            .str.replace_all("_", " ")
            .str.to_title_case()
            .alias("Quadrant"),
            pl.col("customer_share").alias("Customer share"),
            pl.col("spend_share").alias("Spend share"),
        ).select("Quadrant", "customers", "Customer share", "spend_90d", "Spend share")
        st.dataframe(quad_display, hide_index=True, use_container_width=True)
    st.caption(
        f"Customers with comparable changes: {finite_n:,}. Not comparable (missing changes): {not_comp:,}. Not comparable customers are excluded from the scatter and quadrant counts; they are not placed at zero."
    )
    
    # Scatter plot
    fig = go.Figure()
    
    for name in sorted(sample["value_tier"].unique().to_list()):
        part = sample.filter(pl.col("value_tier") == name)
        fig.add_trace(
            go.Scattergl(
                x=part["frequency_change"].to_list(),
                y=part["basket_change"].to_list(),
                mode="markers",
                name=str(name),
                marker=dict(size=6, opacity=0.55, color=get_color(name, color_map)),
                customdata=list(
                    zip(
                        part["spend_90d"].to_list(),
                        part["purchases_90d"].to_list(),
                        part["purchases_previous90d"].to_list(),
                        part["aov_recent"].to_list(),
                        part["aov_previous"].to_list(),
                        part["lifecycle"].to_list(),
                        part["evidence"].to_list(),
                    )
                ),
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
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Growing on both",
        showarrow=False,
        font=dict(size=10, color="#009E73"),
        opacity=0.5,
    )
    fig.add_annotation(
        x=0.5,
        y=0.05,
        xref="paper",
        yref="paper",
        text="More frequent, lighter",
        showarrow=False,
        font=dict(size=10, color="#56B4E9"),
        opacity=0.5,
    )
    fig.add_annotation(
        x=0.05,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Less frequent, heavier",
        showarrow=False,
        font=dict(size=10, color="#E69F00"),
        opacity=0.5,
    )
    fig.add_annotation(
        x=0.05,
        y=0.05,
        xref="paper",
        yref="paper",
        text="Declining on both",
        showarrow=False,
        font=dict(size=10, color="#D55E00"),
        opacity=0.5,
    )
    
    card = create_chart_card(
        key="momentum_scatter",
        title="Customer Momentum Map",
        figure=fig,
        question="Did customers' observed shopping frequency and basket size change between comparable windows?",
        takeaway=f"{finite_n:,} comparable customers; {not_comp:,} not comparable (excluded)",
        evidence=ChartEvidence(
            numerator="Customers with positive/negative change",
            denominator="Comparable customers (tenure ≥ 179d, non-zero previous purchases)",
            population_size=finite_n,
            comparison_period="Recent 90d vs Previous 90d",
            confidence_interval="Not applicable",
            interval_type="Descriptive scatter",
            eligibility_criteria="Tenure ≥ 179 days, non-zero previous purchases, finite frequency_change & basket_change",
        ),
        how_to_read=(
            "X-axis: Frequency change (recent vs previous 90d purchases). "
            "Y-axis: Basket change (recent vs previous 90d AOV). "
            "Quadrants: Growing on both (top-right), More frequent/lighter (bottom-right), "
            "Less frequent/heavier (top-left), Declining on both (bottom-left). "
            "Color: value tier. Zero lines: no change reference."
        ),
        caveat=(
            "Only customers with comparable windows shown. "
            "Missing changes (not comparable) are not shown at zero. "
            "Observed spend, not profit. Quadrant labels are visual guides only."
        ),
        next_step="Check 'Basket Evolution' for slope/dumbbell comparison.",
        alt_description="Scatter plot showing frequency change vs basket change by value tier, with quadrant annotations and zero reference lines.",
    )
    card.render()
    
    st.caption(
        f"Scatter shows up to 2,000 hash-sampled customers with finite changes. "
        f"Zero reference lines and quadrant labels are visual guides only. "
        f"Missing changes (not comparable) are not shown at zero; count: {not_comp:,}."
    )
    
    table(
        sample.select(
            "customer_id",
            "frequency_change",
            "basket_change",
            "value_tier",
            "lifecycle",
            "evidence",
            "spend_90d",
            "purchases_90d",
            "purchases_previous90d",
            "aov_recent",
            "aov_previous",
        )
    )


def render_value_tier_transitions(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Value-Tier Transitions Waterfall."""
    st.subheader("Value-Tier Transition Waterfall")
    st.caption(
        "Net customer movement between value tiers between two snapshots. "
        "Inflow = customers entering tier, Outflow = customers leaving tier, "
        "Net = inflow - outflow. Uses transition data from segment_transitions.parquet."
    )

    manifest = read_json(root / "run.json")
    history = [x for x in manifest["snapshots"] if x <= st.session_state.get("chosen", "")]

    if len(history) < 2:
        st.info("Need at least two cutoffs. Select an earlier snapshot in the sidebar.")
        return

    origin = st.selectbox("Previous cutoff", history[:-1], index=len(history) - 2, key="vt_origin")
    dim = "value_tier"
    st.info("Showing value_tier transitions (waterfall). For other dimensions, use the Transitions view.")

    from retail_analytics.dashboard.app import apply_filters

    before = apply_filters(
        pl.scan_parquet([str(p) for p in (root / "snapshots" / origin).glob("*.parquet")]),
        {},
    )
    after = apply_filters(
        pl.scan_parquet([str(p) for p in (root / "snapshots" / st.session_state.get("chosen", "")).glob("*.parquet")]),
        {},
    )
    flows, counts = transition_summary(before, after, dim)

    if flows.is_empty():
        st.info("No matched customers at both cutoffs.")
        return

    waterfall = value_tier_transition_waterfall(flows)

    if not waterfall.is_empty():
        # Waterfall chart
        fig = go.Figure()
        tiers = waterfall["tier"].to_list()
        inflows = waterfall["inflow"].to_list()
        outflows = waterfall["outflow"].to_list()
        net = waterfall["net_change"].to_list()

        fig.add_trace(
            go.Bar(
                x=tiers,
                y=inflows,
                name="Inflow",
                marker_color="#009E73",
                hovertemplate="Tier: %{x}<br>Customers entering: %{y}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=tiers,
                y=[-o for o in outflows],
                name="Outflow",
                marker_color="#D55E00",
                hovertemplate="Tier: %{x}<br>Customers leaving: %{y}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=tiers,
                y=net,
                name="Net change",
                mode="lines+markers",
                line=dict(color="#0072B2", width=3),
                marker=dict(size=10),
                hovertemplate="Tier: %{x}<br>Net change: %{y}<extra></extra>",
            )
        )

        fig.update_layout(barmode="relative")

        card = create_chart_card(
            key="vt_waterfall",
            title=f"Value-Tier Net Movement: {origin} to {st.session_state.get('chosen')}",
            figure=fig,
            question="How did customer counts shift between value tiers?",
            takeaway=f"Total inflow: {int(waterfall['inflow'].sum()):,}; Total outflow: {int(waterfall['outflow'].sum()):,}; Net: {int(waterfall['net_change'].sum()):+d}",
            evidence=ChartEvidence(
                numerator="Inflow/outflow customer counts",
                denominator="Matched customers per tier",
                population_size=counts["matched_n"],
                comparison_period=f"{origin} → {st.session_state.get('chosen')}",
                confidence_interval="Not applicable",
                interval_type="Descriptive flow",
                eligibility_criteria="Matched customers at both cutoffs",
            ),
            how_to_read=(
                "Green bars: inflow (customers entering tier). "
                "Red bars: outflow (customers leaving tier). "
                "Blue line: net change per tier. "
                "Hover for exact counts."
            ),
            caveat=(
                f"Matched customers: {counts['matched_n']:,}. "
                f"Inflow = customers entering tier. Outflow = customers leaving tier. "
                f"Unmatched origin customers are not churn; unmatched destination are new arrivals."
            ),
            next_step="Check 'Transitions' view for other dimension flows.",
            alt_description="Waterfall chart showing value-tier customer inflows, outflows, and net changes.",
        )
        card.render()

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


def render_spend_concentration(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Spend Concentration (Lorenz Curve)."""
    st.subheader("Customer Concentration (Lorenz Curve)")
    st.caption(
        "Cumulative share of customers vs cumulative share of spend, "
        "plus equality reference line and top-share summaries. "
        "Shows how concentrated recorded spending is among customers."
    )
    
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
        
        card = create_chart_card(
            key="lorenz_curve",
            title=f"Lorenz Curve: Concentration of 90-Day Spend (n={conc['n']})",
            figure=fig,
            question="How concentrated is recorded spending among customers?",
            takeaway=f"Gini: {conc.get('gini', 0):.3f}; Top 10% hold {next((t['spend_share'] for t in conc['top'] if t['requested_fraction']==0.1), 0):.1%} of spend",
            evidence=ChartEvidence(
                numerator="Cumulative spend share",
                denominator="Total spend",
                population_size=conc["n"],
                comparison_period="90-day window",
                confidence_interval="Not applicable",
                interval_type="Descriptive concentration",
                eligibility_criteria="Customers with finite spend_90d",
            ),
            how_to_read=(
                "Blue curve: Lorenz curve (cumulative spend share vs cumulative customer share). "
                "Gray dashed: perfect equality line. "
                "Red markers: top fraction spend shares. "
                "Area between curves = Gini coefficient × 2."
            ),
            caveat=(
                "Concentration is population-specific and depends on inclusion rules. "
                "Do not infer profitability or customer dependency without margin/cost context. "
                "Recorded spend, not profit."
            ),
            next_step="Check 'Segments' view for spend distribution by behavioral segment.",
            alt_description="Lorenz curve showing cumulative spend share vs cumulative customer share with equality reference line and top fraction markers.",
        )
        card.render()
        
        st.caption(
            f"Total spend: {conc['total']:,.2f} ({conc['positive_n']} with positive spend). Top fractions round up to whole customers."
        )


def render_revenue_waterfall(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Revenue Waterfall (Legacy - Fixed-Order Bridge)."""
    st.subheader("Revenue driver waterfall: recent 90d vs previous 90d")
    st.caption(
        "Decomposes spend change into: purchasing customers × baskets per purchaser × spend per basket (AOV). "
        "Uses a fixed-order bridge (customer count → frequency → AOV). "
        "Residual captures interaction effects. Method documented; contributions sum to observed change."
    )

    manifest = read_json(root / "run.json")
    history = [x for x in manifest["snapshots"] if x <= st.session_state.get("chosen", "")]

    # Need previous cutoff data
    if len(history) < 2:
        st.info(
            "Need at least two cutoffs for waterfall. Select an earlier snapshot in the sidebar."
        )
        return

    from retail_analytics.dashboard.app import apply_filters

    origin = st.selectbox("Previous cutoff", history[:-1], index=len(history) - 2)
    before_rows = apply_filters(
        pl.scan_parquet([str(p) for p in (root / "snapshots" / origin).glob("*.parquet")]),
        {},
    )
    cutoff_date = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
    waterfall = revenue_waterfall(rows, before_rows, cutoff_date)

    if waterfall.is_empty():
        st.info("Could not compute waterfall for this comparison.")
        return

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

    fig = go.Figure()
    fig.add_trace(
        go.Waterfall(
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
        )
    )

    card = create_chart_card(
        key="revenue_waterfall",
        title=f"Revenue waterfall: {origin} to {st.session_state.get('chosen')}",
        figure=fig,
        question="How much of observed spend change comes from customers active in both periods vs entering/leaving?",
        takeaway=f"Spend change decomposed into customers, frequency, AOV, and residual",
        evidence=ChartEvidence(
            numerator="Component contributions to spend change",
            denominator="Previous period spend",
            population_size=None,
            comparison_period=f"{origin} → {st.session_state.get('chosen')}",
            confidence_interval="Not applicable",
            interval_type="Arithmetic decomposition",
            eligibility_criteria="Fixed-order bridge: N → F → A",
        ),
        how_to_read=(
            "Waterfall bars: blue=start/end totals, green=positive contribution, red=negative. "
            "Components: (1) purchasing customers change × prev frequency × prev AOV, "
            "(2) current customers × frequency change × prev AOV, "
            "(3) current customers × current frequency × AOV change. "
            "Residual = interaction effects. Hover shows component description."
        ),
        caveat=(
            "Fixed-order bridge: attribution depends on decomposition order. "
            "Not causal attribution. Recorded spend, not profit. "
            "Full snapshot populations."
        ),
        next_step="Check 'Revenue Bridge' for calendar-period decomposition.",
        alt_description="Waterfall chart decomposing spend change into customer count, frequency, AOV, and residual components.",
    )
    card.render()

    table(waterfall)
    st.caption(
        "Fixed-order bridge: (1) purchasing customers change × prev frequency × prev AOV, "
        "(2) current customers × frequency change × prev AOV, "
        "(3) current customers × current frequency × AOV change. "
        "Residual = interaction effects. Attribution depends on decomposition order."
    )


__all__ = [
    "render_rfm_response_surface",
    "render_value_at_risk_scatter",
    "render_revenue_bridge",
    "render_revenue_waterfall",
    "render_customer_momentum",
    "render_spend_concentration",
]