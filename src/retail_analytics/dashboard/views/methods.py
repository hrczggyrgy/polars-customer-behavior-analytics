"""Methods & Data Quality views: Metric Definitions, Thresholds & Calibration, Data Quality Report, Analysis Settings."""

from datetime import date
from pathlib import Path

import polars as pl
import streamlit as st

from retail_analytics.visualization.chart_components import (
    ChartEvidence,
    ChartCard,
    create_chart_card,
)
from retail_analytics.visualization.visual_insights_core import METRICS
from retail_analytics.dashboard.app import read_json, table


def render_metric_definitions():
    """Metric Definitions - All metric formulas and caveats."""
    st.subheader("Metric Definitions")
    st.caption(
        "Question: How is each metric calculated? "
        "Shows formula, numerator, denominator, window, minimum data requirements, and limitations."
    )
    
    # Load METRICS from visual_insights_core
    for metric_name, (description, unit) in METRICS.items():
        with st.expander(f"{metric_name} — {description} ({unit})"):
            st.code(f"Metric: {metric_name}\nDescription: {description}\nUnit: {unit}")
    
    st.markdown("---")
    st.markdown("**Full metric specifications available in [METRICS.md](METRICS.md)**")
    st.markdown("""
    Key conventions:
    - **Snapshot cutoff (C)**: Analysis date. Historical windows include C. Future windows start at C+1.
    - **Null semantics**: Metric is `null` (not zero) when denominator is zero, inputs absent, or sample below threshold.
    - **Historical windows**: 30d [C-29, C], 90d [C-89, C], 365d [C-364, C], Previous 90d [C-179, C-90]
    - **Future windows**: 30d (C, C+30], 90d (C, C+90] — only when coverage confirmed
    - **Change metrics**: Require `tenure ≥ 179` (comparable_windows = true)
    """)


def render_thresholds_calibration(root: Path):
    """Thresholds & Calibration - Show calibrated thresholds and reference info."""
    st.subheader("Thresholds & Calibration")
    st.caption(
        "Question: What thresholds were used for segmentation, and how were they calibrated? "
        "Shows reference snapshot, population sizes, confidence intervals, and stability status."
    )
    
    thresholds = read_json(root / "thresholds.json")
    
    if thresholds.get("reference_selection", {}).get("selected"):
        ref_sel = thresholds["reference_selection"]
        st.success(
            f"✓ Thresholds auto-calibrated from snapshot **{ref_sel['reference_date']}** "
            f"({ref_sel['style_population']:,} style-eligible, {ref_sel['value_population']:,} value-eligible customers)"
        )
        
        with st.expander("Calibrated Thresholds", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Value tier low (P33)", f"{thresholds.get('value_low', 0):,.0f}")
                st.metric("Value tier high (P80)", f"{thresholds.get('value_high', 0):,.0f}")
                st.metric("Basket low (P33)", f"{thresholds.get('basket_low', 0):,.0f}")
                st.metric("Basket high (P80)", f"{thresholds.get('basket_high', 0):,.0f}")
            with col2:
                st.metric("Gap low (P33)", f"{thresholds.get('gap_low', 0):.1f} days")
                st.metric("Gap high (P80)", f"{thresholds.get('gap_high', 0):.1f} days")
                lateness = thresholds.get("lateness_overdue_min")
                if lateness:
                    st.metric("Lateness threshold", f"{lateness:.2f}")
                freq_thresh = thresholds.get("frequency_change_threshold")
                if freq_thresh:
                    st.metric("Frequency change threshold", f"±{freq_thresh:.2f}")
            
            # Show per-threshold details
            if "threshold_details" in thresholds:
                st.write("**Per-Threshold Metadata**")
                for name, details in thresholds["threshold_details"].items():
                    with st.expander(f"{name}: {details['status']}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Value", f"{details['value']:,.2f}" if details['value'] is not None else "N/A")
                            st.metric("Raw", f"{details['raw']:,.2f}" if details['raw'] is not None else "N/A")
                        with col2:
                            st.metric("CI Low", f"{details['ci'][0]:,.2f}" if details['ci'][0] is not None else "N/A")
                            st.metric("CI High", f"{details['ci'][1]:,.2f}" if details['ci'][1] is not None else "N/A")
                        with col3:
                            st.metric("N", f"{details['n']:,}")
                            st.metric("Method", details.get('method', 'N/A'))
                        
                        if details.get('clamp_bounds'):
                            st.caption(f"Clamp bounds: {details['clamp_bounds']}; Clamped value: {details.get('clamped_value', 'N/A')}")
                        if details.get('policy'):
                            st.caption(f"Policy: {details['policy']}")
        
        # Show warnings from data profile
        data_profile = thresholds.get("data_profile", {})
        warnings = data_profile.get("warnings", [])
        if warnings:
            st.markdown("**Data Profile Warnings**")
            for w in warnings:
                st.warning(w)
        
        # Show label share warnings
        label_shares = thresholds.get("label_shares", {})
        ls_warnings = label_shares.get("warnings", [])
        if ls_warnings:
            for w in ls_warnings:
                st.info(w)
                
        # Quantile interpolation method
        if "quantile_interpolation" in thresholds:
            st.caption(f"Quantile interpolation method: **{thresholds['quantile_interpolation']}**")
        
    elif (
        thresholds.get("reference_date") is None
        and thresholds.get("reference_selection", {}).get("selected") is False
    ):
        st.warning(
            "⚠ Thresholds not calibrated: no snapshot met minimum population criteria. "
            "Labels using quantile thresholds (large_basket_episodic, small_basket_frequent, value tiers) "
            "will show as 'reference_unavailable'."
        )
        
        if "reference_selection" in thresholds:
            with st.expander("Calibration Attempt Details", expanded=False):
                st.json(thresholds["reference_selection"])
    else:
        st.info("Thresholds not yet calibrated or no data available.")
    
    # Show reference selection details
    if "reference_selection" in thresholds:
        with st.expander("Reference Selection Details", expanded=False):
            st.json(thresholds["reference_selection"])


def render_data_quality_report(root: Path):
    """Data Quality Report - Detailed quality metrics per snapshot."""
    st.subheader("Data Quality Report")
    st.caption(
        "Question: What is the data quality at each snapshot? "
        "Shows rejection rates, eligibility rates, and data profile diagnostics per cutoff."
    )
    
    # Upload-wide quality
    inspection = read_json(root.parent / "inspection.json")
    
    st.write("**Upload-Wide Quality**")
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
    
    # Bar chart
    import plotly.graph_objects as go
    from retail_analytics.visualization.chart_components import build_color_map
    
    color_map = build_color_map(counts["issue"].to_list())
    
    fig = go.Figure(
        go.Bar(
            x=counts["rows"].to_list(),
            y=counts["issue"].to_list(),
            orientation="h",
            marker_color=[color_map[v] for v in counts["issue"].to_list()],
        )
    )
    fig.update_layout(
        title="Upload-Wide Quality Flags",
        xaxis_title="Row Count",
        yaxis_title="Issue",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Flags can overlap; these counts cover the upload, not the filtered customer population.")
    
    # Inspection details
    with st.expander("Full Inspection JSON", expanded=False):
        st.json(inspection)
    
    # Quality.json if exists
    quality_path = root / "quality.json"
    if quality_path.exists():
        st.write("**Per-Snapshot Quality**")
        with st.expander("Quality JSON", expanded=False):
            st.json(read_json(quality_path))
    
    # Product variation
    pv_path = root / "product_variation.parquet"
    if pv_path.exists():
        with st.expander("Product Variation Audit", expanded=False):
            st.caption("Products with multiple department/category assignments or descriptions.")
            table = pl.read_parquet(pv_path)
            st.dataframe(table, hide_index=True, use_container_width=True)
    
    # Per-snapshot data profiles (from thresholds)
    thresholds = read_json(root / "thresholds.json")
    if "data_profile" in thresholds:
        dp = thresholds["data_profile"]
        st.write("**Reference Snapshot Data Profile**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Customers", f"{dp.get('n_customers', 0):,}")
            st.metric("Style Eligible", f"{dp.get('style_eligible_n', 0):,} ({dp.get('style_eligible_rate', 0):.1%})")
        with col2:
            st.metric("Value Eligible", f"{dp.get('value_eligible_n', 0):,} ({dp.get('value_eligible_rate', 0):.1%})")
            st.metric("Zero Spend", f"{dp.get('zero_spend_n', 0):,} ({dp.get('zero_spend_rate', 0):.1%})")
        with col3:
            if "observation_span_days" in dp:
                st.metric("Observation Span", f"{dp['observation_span_days']:,} days")
            
            mg = dp.get("median_gap", {})
            if mg:
                st.metric("Median Gap", f"{mg.get('median', 0):.1f} days")
        
        # Distributions
        if "tenure" in dp:
            t = dp["tenure"]
            st.write("**Tenure Distribution**")
            st.write(f"Median: {t.get('median', 0):.0f} | Q25: {t.get('q25', 0):.0f} | Q75: {t.get('q75', 0):.0f}")
        
        if "days_365d" in dp:
            d = dp["days_365d"]
            st.write("**Shopping Days (365d) Distribution**")
            st.write(f"Median: {d.get('median', 0):.0f} | Q25: {d.get('q25', 0):.0f} | Q75: {d.get('q75', 0):.0f}")
        
        if "gap_count" in dp:
            g = dp["gap_count"]
            st.write("**Gap Count Distribution**")
            st.write(f"Median: {g.get('median', 0):.0f} | Q25: {g.get('q25', 0):.0f} | Q75: {g.get('q75', 0):.0f}")
        
        if mg:
            mg = dp["median_gap"]
            st.write("**Median Gap Distribution**")
            st.write(f"Median: {mg.get('median', 0):.1f} | Q25: {mg.get('q25', 0):.1f} | Q75: {mg.get('q75', 0):.1f} | Min: {mg.get('min', 0):.1f} | Max: {mg.get('max', 0):.1f}")


def render_analysis_settings(root: Path):
    """Analysis Settings - Show current pipeline configuration."""
    st.subheader("Analysis Settings")
    st.caption(
        "Question: How was this analysis configured? "
        "Shows pipeline configuration, coverage state, and key parameters."
    )
    
    # Run manifest
    manifest = read_json(root / "run.json")
    
    st.write("**Pipeline Configuration**")
    config_cols = ["price_mode", "polars_version", "source_sha256", "invalid_rows_excluded"]
    config_data = {k: manifest.get(k) for k in config_cols if k in manifest}
    st.json(config_data)
    
    # Coverage state
    st.write("**Coverage State**")
    cov_cols = ["coverage_confirmed", "coverage_complete_through", "latest_observed_transaction_date", 
                "observed_through", "coverage_scope_note", "coverage_confirmation_source"]
    cov_data = {k: manifest.get(k) for k in cov_cols if k in manifest}
    st.json(cov_data)
    
    # Snapshots
    st.write("**Analysis Cutoffs**")
    snapshots = manifest.get("snapshots", [])
    st.write(f"Cutoffs ({len(snapshots)}): {', '.join(snapshots)}")
    
    # Thresholds reference
    thresholds = read_json(root / "thresholds.json")
    if "reference_selection" in thresholds and thresholds["reference_selection"].get("selected"):
        ref = thresholds["reference_selection"]
        st.write("**Threshold Calibration Reference**")
        st.write(f"Reference date: {ref.get('reference_date', 'N/A')}")
        st.write(f"Style population: {ref.get('style_population', 'N/A'):,}")
        st.write(f"Value population: {ref.get('value_population', 'N/A'):,}")
        st.write(f"Checked snapshots: {ref.get('checked_snapshots', [])}")
    
    # Quantile interpolation
    if "quantile_interpolation" in thresholds:
        st.write(f"**Quantile Interpolation:** {thresholds['quantile_interpolation']}")
    
    # Calibration policy
    if "calibration_policy" in thresholds:
        with st.expander("Calibration Policy", expanded=False):
            st.json(thresholds["calibration_policy"])
    
    # Threshold details
    if "threshold_details" in thresholds:
        with st.expander("Threshold Details", expanded=False):
            for name, details in thresholds["threshold_details"].items():
                st.write(f"**{name}**: {details['status']} (n={details['n']}, method={details['method']})")
                if details.get('value') is not None:
                    st.write(f"  Value: {details['value']:,.2f} | Raw: {details['raw']:,.2f} | CI: [{details['ci'][0]:,.2f}, {details['ci'][1]:,.2f}]")
                if details.get('clamp_bounds'):
                    st.write(f"  Clamped: {details['clamped_value']:,.2f} (bounds: {details['clamp_bounds']})")


__all__ = [
    "render_metric_definitions",
    "render_thresholds_calibration",
    "render_data_quality_report",
    "render_analysis_settings",
]