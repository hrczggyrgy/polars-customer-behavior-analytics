"""Customer Explorer views: Customer Brief and Customer Timeline."""

from datetime import date
from pathlib import Path

import numpy as np
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
)
from retail_analytics.dashboard.app import collect, read_json, table, PAGE_ROWS


def render_customer_brief(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Customer Brief - Narrative profile with evidence and limitations."""
    from retail_analytics.visualization.chart_components import (
        ChartEvidence,
        ChartCard,
        create_chart_card,
    )
    
    st.subheader("👤 Customer Brief")
    st.caption(
        "**Question:** What has actually been observed about this customer? "
        "A concise narrative profile with supporting evidence and clear limitations. "
        "Identifiers are masked by default."
    )

    # Customer selector
    customer_ids = collect(rows.select("customer_id").unique().sort("customer_id"))["customer_id"].to_list()
    if not customer_ids:
        st.info("No customers in filtered population.")
        return

    # Mask identifiers by default
    mask_ids = st.checkbox("Mask customer IDs", value=True, key="mask_customer_ids")
    display_ids = [f"Cust_{i+1:04d}" for i in range(len(customer_ids))] if mask_ids else customer_ids
    id_map = dict(zip(display_ids, customer_ids))

    selected_display = st.selectbox("Customer", display_ids, key="customer_brief_id")
    selected_id = id_map[selected_display]
    if not selected_id:
        return

    # Get customer data
    customer = collect(rows.filter(pl.col("customer_id") == selected_id))
    if customer.is_empty():
        st.warning("Customer not found in current filtered population.")
        return

    c = customer.row(0, named=True)

    # ============================================================
    # 1. OBSERVED RELATIONSHIP
    # ============================================================
    st.markdown("### 1. Observed Relationship")
    
    first_purchase = c.get("first_purchase")
    last_purchase = c.get("last_purchase")
    tenure = c.get("tenure", 0)
    recency = c.get("recency", None)
    purchases_90d = c.get("purchases_90d", 0)
    days_365d = c.get("days_365d", 0)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("First Purchase", str(first_purchase) if first_purchase else "N/A")
    col2.metric("Last Purchase", str(last_purchase) if last_purchase else "N/A")
    col3.metric("Tenure (days)", f"{tenure:,}" if tenure else "N/A")
    col4.metric("Recency (days)", f"{recency:,}" if recency is not None else "N/A")
    
    col1, col2 = st.columns(2)
    col1.metric("Purchase Days (90d)", f"{purchases_90d:,}")
    col2.metric("Shopping Days (365d)", f"{days_365d:,}")
    
    # Evidence badge for relationship
    rel_state = evaluate_evidence(
        n=days_365d if days_365d else 0,
        thresholds=SupportThresholds(),
        analysis_type="default"
    )
    st.caption(f"Evidence: {format_evidence_badge(rel_state)}")

    # ============================================================
    # 2. SHOPPING RHYTHM
    # ============================================================
    st.markdown("### 2. Shopping Rhythm")
    
    median_gap = c.get("median_gap")
    gap_cv = c.get("gap_cv")
    gap_count = c.get("gap_count", 0)
    lateness = c.get("lateness")
    
    if median_gap is not None and median_gap > 0:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median Gap", f"{median_gap:.1f} days")
        col2.metric("Gap CV", f"{gap_cv:.2f}" if gap_cv is not None else "N/A")
        col3.metric("Usable Gaps", f"{gap_count:,}")
        col4.metric("Lateness", f"{lateness:.2f}×" if lateness is not None else "N/A")
        
        # Rhythm interpretation
        if lateness is not None:
            if lateness < 1.0:
                rhythm_status = "🟢 On cadence (purchasing ahead of typical rhythm)"
            elif lateness < 2.0:
                rhythm_status = "🟡 1-2× overdue"
            else:
                rhythm_status = "🔴 ≥2× overdue (significantly past typical interval)"
            st.caption(f"Rhythm status: **{rhythm_status}**")
        
        # Eligibility for cadence metrics
        cadence_state = evaluate_evidence(
            n=gap_count,
            thresholds=SupportThresholds(),
            analysis_type="default"
        )
        st.caption(f"Cadence evidence: {format_evidence_badge(cadence_state)}")
        if not cadence_state.has_evidence:
            st.warning("⚠ Insufficient gaps for reliable rhythm estimate (need ≥4 gaps, tenure ≥90d, 6+ shopping days in 365d).")
    else:
        st.warning("⚠ No median gap available — insufficient purchase history for rhythm analysis.")
        st.caption("Requires: ≥6 shopping days in 365d, tenure ≥90d, ≥4 usable gaps")

    # ============================================================
    # 3. BASKET / VALUE PATTERN
    # ============================================================
    st.markdown("### 3. Basket & Value Pattern (90-day window)")
    
    spend_90d = c.get("spend_90d", 0)
    median_basket = c.get("median_basket")
    aov_recent = c.get("aov_recent")
    aov_previous = c.get("aov_previous")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"{spend_90d:,.2f}")
    col2.metric("Median Basket", f"{median_basket:,.2f}" if median_basket is not None else "N/A")
    col3.metric("Recent AOV", f"{aov_recent:,.2f}" if aov_recent is not None else "N/A")
    col4.metric("Previous AOV", f"{aov_previous:,.2f}" if aov_previous is not None else "N/A")
    
    # Peer comparison
    peer_median_spend = collect(
        rows.select(pl.col("spend_90d").median())
    ).item()
    if peer_median_spend:
        diff_pct = ((spend_90d - peer_median_spend) / peer_median_spend * 100) if peer_median_spend else 0
        st.caption(f"Peer median 90d spend: {peer_median_spend:,.2f} | This customer: {diff_pct:+.1f}% vs peer median")
    
    st.caption("⚠ Monetary figures are recorded spend, not profit.")

    # ============================================================
    # 4. BREADTH & PREFERENCE
    # ============================================================
    st.markdown("### 4. Breadth & Preference")
    
    cat_count = c.get("category_count", 0)
    cat_hhi = c.get("category_hhi")
    dom_cat = c.get("dominant_category", "N/A")
    dom_dept = c.get("dominant_department", "N/A")
    dominant_share = c.get("dominant_share")
    core_size = c.get("core_size", 0)
    core_spend_share = c.get("core_spend_share", 0)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Categories Shopped", f"{cat_count:,}")
    col2.metric("Dominant Category", f"{dom_cat} ({dom_dept})")
    col3.metric("Dominant Share", f"{dominant_share:.1%}" if dominant_share is not None else "N/A")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Category HHI", f"{cat_hhi:.3f}" if cat_hhi is not None else "N/A")
    col2.metric("Core Products", f"{core_size:,}")
    col3.metric("Core Spend Share", f"{core_spend_share:.1%}" if core_spend_share is not None else "N/A")
    
    # Concentration interpretation
    if cat_hhi is not None:
        if cat_hhi > 0.5:
            conc_desc = "Highly concentrated (few categories dominate)"
        elif cat_hhi > 0.25:
            conc_desc = "Moderately concentrated"
        else:
            conc_desc = "Diverse (broad category spread)"
        st.caption(f"Concentration: **{conc_desc}** (HHI = {cat_hhi:.3f})")

    # ============================================================
    # 5. CHANGE & EVIDENCE
    # ============================================================
    st.markdown("### 5. Recent vs Previous Period (90d vs Prior 90d)")
    
    if c.get("comparable_windows"):
        freq_chg = c.get("frequency_change")
        basket_chg = c.get("basket_change")
        spend_chg = c.get("spend_change")
        breadth_chg = c.get("breadth_change")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Frequency Δ", f"{freq_chg:+.1%}" if freq_chg is not None else "N/A")
        col2.metric("Basket Size Δ", f"{basket_chg:+.1%}" if basket_chg is not None else "N/A")
        col3.metric("Spend Δ", f"{spend_chg:+.1%}" if spend_chg is not None else "N/A")
        col4.metric("Breadth Δ", f"{breadth_chg:+.2f}" if breadth_chg is not None else "N/A")
        
        # Evidence quality for change
        tenure = c.get("tenure", 0)
        st.caption(f"Comparison eligible: tenure ≥ 179d ({'✓' if tenure >= 179 else '✗'}, actual: {tenure}d)")
    else:
        tenure = c.get("tenure", 0)
        st.info(f"⚠ Insufficient tenure for period-over-period comparison (requires ≥ 179 days, current: {tenure}d).")
    
    # Evidence summary
    st.markdown("### Evidence Summary")
    evidence_level = c.get("evidence", "unknown")
    lifecycle = c.get("lifecycle", "unknown")
    behavior = c.get("behavior", "unknown")
    trajectory = c.get("trajectory", "unknown")
    value_tier = c.get("value_tier", "unknown")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Lifecycle:** {lifecycle}")
        st.write(f"**Evidence Level:** {evidence_level}")
        st.write(f"**Behavior:** {behavior}")
    with col2:
        st.write(f"**Trajectory:** {trajectory}")
        st.write(f"**Value Tier:** {value_tier}")
        if c.get("style_eligible"):
            st.write("**Style Eligible:** ✓ (sufficient history for cadence metrics)")
        else:
            st.write("**Style Eligible:** ✗ (insufficient history for cadence metrics)")
    
    # Future outcomes (if coverage confirmed)
    if coverage:
        future_30 = c.get("future_purchases_30d")
        future_90 = c.get("future_purchases_90d")
        eligible_30 = c.get("eligible_followup_30d")
        eligible_90 = c.get("eligible_followup_90d")
        
        if eligible_30 or eligible_90:
            st.markdown("### Future Outcomes (Observed)")
            col1, col2 = st.columns(2)
            if eligible_30 and future_30 is not None:
                col1.metric("30-day Repeat", "Yes" if future_30 > 0 else "No")
            if eligible_90 and future_90 is not None:
                col2.metric("90-day Repeat", "Yes" if future_90 > 0 else "No")
        else:
            st.caption("Future outcomes: Not eligible for follow-up in requested horizons.")
    else:
        st.caption("Future outcomes: Unavailable (coverage not confirmed). Re-run analysis with coverage confirmation to enable.")


def render_customer_timeline(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Customer Timeline - Narrative purchase history with rhythm and category evolution."""
    from retail_analytics.visualization.chart_components import (
        ChartEvidence,
        ChartCard,
        create_chart_card,
    )
    
    st.subheader("📅 Customer Timeline")
    st.caption(
        "**Question:** What is this customer's purchase history over time? "
        "Shows timeline of baskets, spend, cadence, and category evolution with rhythm analysis."
    )

    # Customer selector with masked IDs
    customer_ids = collect(rows.select("customer_id").unique().sort("customer_id"))["customer_id"].to_list()
    if not customer_ids:
        st.info("No customers in filtered population.")
        return

    mask_ids = st.checkbox("Mask customer IDs", value=True, key="mask_customer_ids_timeline")
    display_ids = [f"Cust_{i+1:04d}" for i in range(len(customer_ids))] if mask_ids else customer_ids
    id_map = dict(zip(display_ids, customer_ids))

    selected_display = st.selectbox("Customer", display_ids, key="customer_timeline_id")
    selected_id = id_map[selected_display]
    if not selected_id:
        return

    # Load validated transactions for this customer
    validated_path = root / "validated.parquet"
    if not validated_path.exists():
        st.info("No validated transaction data available.")
        return

    validated = pl.scan_parquet(validated_path)
    customer_txns = collect(
        validated.filter(pl.col("customer_id") == selected_id)
        .sort("date")
        .select("date", "transaction_id", "product_id", "category", "department", "value", "quantity")
    )

    if customer_txns.is_empty():
        st.info("No transactions found for this customer.")
        return

    # ============================================================
    # Timeline Summary
    # ============================================================
    first_date = customer_txns["date"].min()
    last_date = customer_txns["date"].max()
    total_days = (last_date - first_date).days + 1 if first_date and last_date else 0
    total_txns = customer_txns.height
    total_spend = customer_txns["value"].sum()
    
    # Aggregate to basket level
    baskets = customer_txns.group_by(["date", "transaction_id"]).agg(
        pl.col("value").sum().alias("basket_spend"),
        pl.col("product_id").n_unique().alias("n_products"),
        pl.col("category").unique().alias("categories"),
        pl.col("department").unique().alias("departments"),
    ).sort("date")
    
    n_baskets = baskets.height
    median_basket = baskets["basket_spend"].median()
    
    # Calculate gaps
    gap_days = []
    prev_date = None
    for d in baskets["date"].to_list():
        if prev_date is not None:
            gap_days.append((d - prev_date).days)
        prev_date = d
    
    median_gap = np.median(gap_days) if gap_days else None
    gap_cv = (np.std(gap_days) / np.mean(gap_days)) if gap_days and np.mean(gap_days) > 0 else None
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("First Purchase", str(first_date))
    col2.metric("Last Purchase", str(last_date))
    col3.metric("Total Span", f"{total_days:,} days")
    col4.metric("Total Baskets", f"{n_baskets:,}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"{total_spend:,.2f}")
    col2.metric("Median Basket", f"{median_basket:,.2f}" if median_basket is not None else "N/A")
    col3.metric("Median Gap", f"{median_gap:.1f} days" if median_gap is not None else "N/A")
    col4.metric("Gap CV", f"{gap_cv:.2f}" if gap_cv is not None else "N/A")

    # Rhythm interpretation
    if median_gap is not None and median_gap > 0:
        # Get recency from snapshot
        customer_snapshot = collect(rows.filter(pl.col("customer_id") == selected_id))
        if not customer_snapshot.is_empty():
            recency = customer_snapshot["recency"][0]
            lateness = recency / median_gap if recency and median_gap > 0 else None
            
            if lateness is not None:
                if lateness < 1.0:
                    rhythm_status = "🟢 On cadence"
                elif lateness < 2.0:
                    rhythm_status = "🟡 1-2× overdue"
                else:
                    rhythm_status = "🔴 ≥2× overdue"
                st.caption(f"Current rhythm: **{rhythm_status}** (lateness = {lateness:.2f}×, recency = {recency}d, median gap = {median_gap:.1f}d)")

    # ============================================================
    # Basket Spend Timeline (ChartCard)
    # ============================================================
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=baskets["date"].to_list(),
        y=baskets["basket_spend"].to_list(),
        mode="lines+markers",
        name="Basket spend",
        line=dict(color="#0072B2"),
        marker=dict(size=8),
        hovertemplate="Date: %{x}<br>Basket spend: %{y:,.2f}<br>Products: %{customdata[0]}<extra></extra>",
        customdata=list(zip(baskets["n_products"].to_list())),
    ))
    
    # Add cumulative spend line
    cum_spend = baskets["basket_spend"].cum_sum().to_list()
    fig.add_trace(go.Scatter(
        x=baskets["date"].to_list(),
        y=cum_spend,
        mode="lines",
        name="Cumulative spend",
        line=dict(color="#009E73", dash="dot"),
        yaxis="y2",
        hovertemplate="Date: %{x}<br>Cumulative: %{y:,.2f}<extra></extra>",
    ))
    
    fig.update_layout(
        title="Purchase Timeline: Basket Spend & Cumulative",
        xaxis_title="Date",
        yaxis=dict(title="Basket Spend", side="left"),
        yaxis2=dict(title="Cumulative Spend", side="right", overlaying="y", showgrid=False),
        height=400,
        template="plotly_white",
        hovermode="x unified",
    )
    
    # Render with ChartCard
    card = create_chart_card(
        key="customer_timeline_spend",
        title="Purchase Timeline",
        figure=fig,
        question="When did this customer shop and how much did they spend per basket?",
        takeaway=f"{n_baskets} baskets over {total_days} days (median {median_basket:,.2f}/basket, median gap {median_gap:.1f}d)" if median_basket else f"{n_baskets} baskets over {total_days} days",
        evidence=ChartEvidence(
            numerator=f"{n_baskets} baskets",
            denominator=f"{total_days} days span",
            population_size=1,
            comparison_period=f"{first_date} to {last_date}",
            confidence_interval="Not applicable",
            interval_type="Descriptive",
            eligibility_criteria="Customer has ≥1 transaction",
        ),
        how_to_read=(
            "Blue line+markers: individual basket spend per purchase date. "
            "Green dotted line: cumulative spend (right axis). "
            "Hover shows basket spend, product count, and cumulative total."
        ),
        caveat="Shows observed transactions only. Returns/refunds not tracked. Cumulative includes all historical purchase days through cutoff.",
        next_step="Check 'Customer Brief' for narrative profile with rhythm and category analysis.",
        alt_description=f"Timeline showing {n_baskets} purchase events with basket spend and cumulative spend over {total_days} days.",
    )
    card.render()

    # ============================================================
    # Basket Details Table
    # ============================================================
    st.markdown("**Basket Details**")
    display_baskets = baskets.with_columns(
        pl.col("categories").list.join(", ").alias("categories"),
        pl.col("departments").list.join(", ").alias("departments"),
    ).select("date", "transaction_id", "basket_spend", "n_products", "categories", "departments")
    table(display_baskets)

    # ============================================================
    # Category Evolution Over Time (ChartCard)
    # ============================================================
    st.markdown("**Category Evolution**")
    cat_over_time = customer_txns.group_by(["date", "category"]).agg(
        pl.col("value").sum().alias("spend")
    ).sort(["date", "category"])

    if not cat_over_time.is_empty():
        fig2 = go.Figure()
        for cat in sorted(cat_over_time["category"].unique().to_list()):
            cat_data = cat_over_time.filter(pl.col("category") == cat)
            fig2.add_trace(go.Scatter(
                x=cat_data["date"].to_list(),
                y=cat_data["spend"].to_list(),
                mode="lines+markers",
                name=cat,
                line=dict(width=2),
                marker=dict(size=6),
            ))
        fig2.update_layout(
            title="Spend by Category Over Time",
            xaxis_title="Date",
            yaxis_title="Spend",
            height=350,
            template="plotly_white",
        )
        
        card2 = create_chart_card(
            key="customer_timeline_category",
            title="Category Evolution",
            figure=fig2,
            question="How has this customer's category spend evolved over time?",
            takeaway=f"{cat_over_time['category'].n_unique()} categories observed; dominant shifts visible over time",
            evidence=ChartEvidence(
                numerator="Category spend per date",
                denominator="Total spend per date",
                population_size=1,
                comparison_period=f"{first_date} to {last_date}",
                confidence_interval="Not applicable",
                interval_type="Descriptive",
                eligibility_criteria="Customer has ≥1 transaction",
            ),
            how_to_read=(
                "Each line: spend in a category over time. "
                "Peaks = high spend periods in that category. "
                "Multiple lines = category switching or diversification."
            ),
            caveat="Category = department/category from source data. Spend = recorded amount, not profit.",
            next_step="Check 'Customer Brief' for breadth/concentration metrics and dominant category.",
            alt_description=f"Line chart showing spend by category over time for {cat_over_time['category'].n_unique()} categories.",
        )
        card2.render()

    # ============================================================
    # Rhythm Analysis
    # ============================================================
    if gap_days:
        st.markdown("**Purchase Rhythm Analysis**")
        
        # Gap histogram
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=gap_days,
            nbinsx=min(20, len(set(gap_days))),
            marker_color="#0072B2",
            opacity=0.7,
            name="Inter-purchase gaps",
        ))
        if median_gap:
            fig3.add_vline(x=median_gap, line_dash="dash", line_color="green", 
                          annotation_text=f"Median: {median_gap:.1f}d")
        fig3.update_layout(
            title="Inter-Purchase Gap Distribution",
            xaxis_title="Days between purchases",
            yaxis_title="Frequency",
            height=300,
            template="plotly_white",
        )
        card3 = create_chart_card(
            key="customer_timeline_gaps",
            title="Gap Distribution",
            figure=fig3,
            question="What is the distribution of days between this customer's purchases?",
            takeaway=f"{len(gap_days)} gaps; median {median_gap:.1f}d, CV {gap_cv:.2f}" if median_gap and gap_cv else f"{len(gap_days)} gaps observed",
            evidence=ChartEvidence(
                numerator=f"{len(gap_days)} gaps",
                denominator="Consecutive purchase pairs",
                population_size=1,
                comparison_period=f"{first_date} to {last_date}",
                confidence_interval="Not applicable",
                interval_type="Descriptive",
                eligibility_criteria="Customer has ≥2 purchase days",
            ),
            how_to_read="Histogram of days between consecutive purchase dates. Median line (green) shows typical cadence.",
            caveat="Only shows observed gaps. Gaps may be censored at start/end of observation window.",
            next_step="Check 'Purchase Cadence' view for population-level cadence analysis.",
            alt_description=f"Histogram of {len(gap_days)} inter-purchase gaps with median line.",
        )
        card3.render()


def format_evidence_badge(state: EvidenceState) -> str:
    """Format evidence state as a colored badge for UI."""
    from retail_analytics.visualization.chart_components import format_evidence_badge as _format
    return _format(state)


__all__ = [
    "render_customer_brief",
    "render_customer_timeline",
    "format_evidence_badge",
]