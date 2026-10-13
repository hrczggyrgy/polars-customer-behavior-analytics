"""Overview views: Data Trust & Quality, Customer Base Summary, Key Signals."""

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from retail_analytics.visualization.chart_components import (
    ChartEvidence,
    collect,
    create_chart_card,
    get_color,
    read_json,
    table,
)


def render_data_trust_quality(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Data Trust & Quality view."""

    st.subheader("Data Trust & Quality")
    st.caption(
        "Upload-wide quality metrics. These reflect the raw data before filtering. "
        "Future outcomes require coverage confirmation."
    )

    inspection = read_json(root.parent / "inspection.json")

    # Quality flags
    labels = [
        "invalid_rows",
        "invalid_dates",
        "unknown_department_rows",
        "unknown_category_rows",
        "null_price",
        "malformed_price",
        "nonfinite_price",
        "negative_price",
        "null_quantity",
        "malformed_quantity",
        "nonfinite_quantity",
        "nonpositive_quantity",
        "nonfinite_value",
        "missing_price_raw",
        "missing_quantity_raw",
    ]
    counts = pl.DataFrame({"issue": labels, "rows": [inspection.get(k, 0) for k in labels]})

    # Bar chart with ChartCard
    fig = go.Figure(
        go.Bar(
            x=counts["rows"].to_list(),
            y=counts["issue"].to_list(),
            orientation="h",
            marker_color="#0072B2",
        )
    )

    card = create_chart_card(
        key="quality_flags",
        title="Upload-Wide Quality Flags",
        figure=fig,
        question="What data quality issues were found during upload validation?",
        takeaway=f"Total rows: {inspection.get('rows', 0):,}; Invalid: {inspection.get('invalid_rows', 0):,} ({inspection.get('rejection_rate', 0):.2%})",
        evidence=ChartEvidence(
            numerator="Rejected rows by reason",
            denominator="Source rows",
            population_size=inspection.get("rows", 0),
            comparison_period="Full upload",
            eligibility_criteria="Row-level validation rules",
        ),
        how_to_read="Horizontal bars show count of rows failing each validation rule. Flags can overlap; sums may exceed total rejections.",
        caveat="These counts cover the upload, not the filtered customer population. Future outcomes unavailable without coverage confirmation.",
        next_step="Check 'Data Quality Report' for per-snapshot quality metrics.",
        alt_description="Horizontal bar chart showing data quality flag counts by rejection reason.",
    )
    card.render()

    # Coverage status
    if coverage:
        st.success("✅ Coverage confirmed — future outcomes available")
    else:
        st.warning("⚠ Coverage NOT confirmed — future outcomes disabled")

    # Inspection details
    with st.expander("Full Inspection Details", expanded=False):
        st.json(inspection)

    # Quality.json if exists
    quality_path = root / "quality.json"
    if quality_path.exists():
        with st.expander("Snapshot Quality Metrics", expanded=False):
            st.json(read_json(quality_path))

    # Product variation
    pv_path = root / "product_variation.parquet"
    if pv_path.exists():
        with st.expander("Product Variation Audit", expanded=False):
            st.caption("Products with multiple department/category assignments or descriptions.")
            table = pl.read_parquet(pv_path)
            st.dataframe(table, hide_index=True, use_container_width=True)


def render_customer_base_summary(rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Customer Base Summary view."""
    st.subheader("Customer Base Summary")
    st.caption("Summary metrics for the current filtered population at the selected cutoff.")

    metrics = collect(
        rows.select(
            pl.len().alias("customers"),
            pl.col("spend_90d").sum().alias("sales"),
            pl.col("purchases_90d").sum().alias("baskets"),
            (pl.col("purchases_90d") > 0).sum().alias("active"),
        )
    ).row(0, named=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filtered Customers", f"{metrics['customers']:,}")
    col2.metric("Recorded Spend: 90d", f"{metrics['sales']:,.2f}")
    col3.metric("Baskets: 90d", f"{metrics['baskets']:,}")
    col4.metric("Active Customers: 90d", f"{metrics['active']:,}")

    st.caption("Recorded amounts are currency units, not profit. Windows end at the cutoff; inactive observed customers remain included.")

    # Segment composition
    DIMS = ["behavior", "trajectory", "lifecycle", "value_tier", "evidence"]
    dim = st.selectbox("Segmentation layer", DIMS, key="cbs_dim")

    data = collect(
        rows.group_by(dim).agg(
            pl.len().alias("customers"),
            pl.col("spend_90d").sum().alias("spend_90d"),
        )
    )

    fig = go.Figure(
        go.Bar(
            x=data["customers"].to_list(),
            y=data[dim].to_list(),
            orientation="h",
            marker_color=[get_color(v, color_map) for v in data[dim].to_list()],
        )
    )

    card = create_chart_card(
        key="cbs_segment_customers",
        title=f"Customers by {dim}",
        figure=fig,
        question=f"How many customers fall into each {dim} segment?",
        takeaway=f"{data.height} segments; largest: {data.sort('customers', descending=True)[dim][0]} ({data['customers'].max():,} customers)",
        evidence=ChartEvidence(
            numerator="Customers per segment",
            denominator="Filtered population",
            population_size=metrics["customers"],
            comparison_period="90-day window ending at cutoff",
            eligibility_criteria="Current filtered population",
        ),
        how_to_read="Horizontal bars show customer count per segment. Hover for exact numbers.",
        caveat="Segmentation based on behavioral rules; not validated predictions. 'reference_unavailable' means thresholds not calibrated.",
        next_step="Check 'Segments' view for repeat rates by segment.",
        alt_description=f"Horizontal bar chart of customer counts by {dim} segment.",
    )
    card.render()

    # Spend by segment
    fig2 = go.Figure(
        go.Bar(
            x=data["spend_90d"].to_list(),
            y=data[dim].to_list(),
            orientation="h",
            marker_color=[get_color(v, color_map) for v in data[dim].to_list()],
        )
    )

    card2 = create_chart_card(
        key="cbs_segment_spend",
        title=f"Recorded Spend by {dim}",
        figure=fig2,
        question=f"How does 90-day spend distribute across {dim} segments?",
        takeaway=f"Total spend: {metrics['sales']:,.2f}; top segment: {data.sort('spend_90d', descending=True)[dim][0]} ({data['spend_90d'].max():,.2f})",
        evidence=ChartEvidence(
            numerator="Spend per segment",
            denominator="Total spend",
            population_size=metrics["customers"],
            comparison_period="90-day window",
            eligibility_criteria="Current filtered population",
        ),
        how_to_read="Horizontal bars show total recorded spend per segment.",
        caveat="Spend is recorded amount, not profit. Segment definitions depend on threshold calibration.",
        next_step="Check 'Value & Risk' section for RFM response surface and value-at-risk analysis.",
        alt_description=f"Horizontal bar chart of 90-day spend by {dim} segment.",
    )
    card2.render()

    table(data)


def render_key_signals(rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Key Signals view - actionable insights."""
    from retail_analytics.visualization.visual_insights_core import concentration

    st.subheader("Key Signals")
    st.caption("Automatically derived insights from the current population. Each signal links to a detailed view.")

    metrics = collect(
        rows.select(
            pl.len().alias("customers"),
            pl.col("spend_90d").sum().alias("sales"),
            pl.col("purchases_90d").sum().alias("baskets"),
            (pl.col("purchases_90d") > 0).sum().alias("active"),
        )
    ).row(0, named=True)

    if not metrics["customers"]:
        st.info("No customers in filtered population.")
        return

    # Signal 1: Concentration
    conc = concentration(rows)
    if conc["n"] > 0:
        top_10 = conc["top"]
        top_10_share = next((t["spend_share"] for t in top_10 if t["requested_fraction"] == 0.1), 0)

        st.markdown("### 📊 Spend Concentration")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Top 10% Spend Share", f"{top_10_share:.1%}")
            st.caption(f"Gini: {conc['gini']:.3f}" if conc.get("gini") else "")
        with col2:
            # Mini Lorenz
            curve = conc["curve"]
            x_vals, y_vals = zip(*curve) if curve else ([], [])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", line=dict(color="#0072B2")))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(color="gray", dash="dash")))
            fig.update_layout(height=150, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🔍 Explore Concentration", key="ks_conc"):
            st.session_state["nav_view"] = "Spend Concentration"
            st.session_state["nav_section"] = "03 · Value & Risk"
            st.rerun()

    # Signal 2: Lifecycle distribution
    lifecycle_counts = collect(rows.group_by("lifecycle").agg(pl.len().alias("n")).sort("n", descending=True))
    if lifecycle_counts.height > 0:
        st.markdown("### 🔄 Lifecycle Distribution")
        top_lifecycle = lifecycle_counts["lifecycle"][0]
        top_count = lifecycle_counts["n"][0]
        st.write(f"Largest group: **{top_lifecycle}** ({top_count:,} customers, {top_count/metrics['customers']:.1%})")

        if st.button("🔍 Explore Lifecycle", key="ks_lifecycle"):
            st.session_state["nav_view"] = "Segments"
            st.session_state["nav_section"] = "02 · Lifecycle & Repeat"
            st.rerun()

    # Signal 3: Cadence alert
    eligible_cadence = collect(
        rows.filter(pl.col("median_gap").is_not_null() & (pl.col("median_gap") > 0))
        .select(pl.len().alias("n"))
    ).item()

    if eligible_cadence > 0:
        lateness_data = collect(
            rows.filter(pl.col("lateness").is_not_null())
            .select(pl.col("lateness").mean().alias("avg_lateness"), pl.col("lateness").quantile(0.9).alias("p90_lateness"))
        ).row(0, named=True)

        st.markdown("### ⏱️ Purchase Cadence")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Eligible for Cadence", f"{eligible_cadence:,}")
        with col2:
            avg_lat = lateness_data.get("avg_lateness", 0)
            st.metric("Avg Lateness", f"{avg_lat:.2f}×" if avg_lat else "N/A")

        if st.button("🔍 Explore Cadence", key="ks_cadence"):
            st.session_state["nav_view"] = "Purchase Cadence"
            st.session_state["nav_section"] = "02 · Lifecycle & Repeat"
            st.rerun()

    # Signal 4: Future outcomes (if coverage)
    if coverage:
        future_30 = collect(
            rows.filter(pl.col("eligible_followup_30d"))
            .select((pl.col("future_purchases_30d") > 0).mean().alias("rate"), pl.len().alias("n"))
        ).row(0, named=True)

        st.markdown("### 🔮 Future 30-Day Repeat")
        if future_30["n"] > 0:
            st.metric("Observed Repeat Rate", f"{future_30['rate']:.1%}")
            st.caption(f"Based on {future_30['n']:,} eligible customers")

            if st.button("🔍 Explore Future Validation", key="ks_fv"):
                st.session_state["nav_view"] = "Future Validation"
                st.session_state["nav_section"] = "02 · Lifecycle & Repeat"
                st.rerun()
    else:
        st.info("Future outcomes unavailable — confirm coverage to enable")

    st.markdown("---")
    st.caption("These signals are descriptive observations, not predictions or recommendations. Click 'Explore' to dive deeper.")
