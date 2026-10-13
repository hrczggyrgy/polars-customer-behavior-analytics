"""Segments view module."""

from datetime import date
from pathlib import Path

import polars as pl
import streamlit as st

from retail_analytics.core.stats import wilson
from retail_analytics.visualization.chart_components import (
    EvidenceState,
    SupportThresholds,
    check_horizon_eligibility,
    collect,
    evaluate_evidence,
    format_evidence_badge,
    get_color,
    multiplicity_warning,
    read_json,
    table,
)


def render_segments(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Segments view with future-purchase segment analysis."""
    dim = st.selectbox("Segmentation layer", ["behavior", "trajectory", "lifecycle", "value_tier", "evidence"])
    horizon = st.selectbox("Future-purchase horizon (days)", [30, 90])

    # Per-horizon eligibility check (P0 fix)
    manifest = read_json(root / "run.json")
    chosen_date = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
    eligible, reason = check_horizon_eligibility(manifest, horizon, chosen_date)

    if not coverage:
        st.warning(
            "⚠ **Coverage not confirmed** — Future-purchase segment analysis is unavailable.\n\n"
            "To enable segment repeat rates, re-run analysis with coverage confirmation "
            "and a complete-through date."
        )
    elif not eligible:
        st.error(f"❌ **{horizon}-day horizon ineligible:** {reason}")
    else:
        st.success(f"✅ **{horizon}-day horizon:** {reason}")

    # Sparse-data safeguards: minimum support threshold
    thresholds = SupportThresholds()
    minimum = st.number_input(
        "Minimum eligible customers per outcome estimate",
        1, 10000, thresholds.min_segment_rate_denominator,
        help=f"Recommended minimum: {thresholds.min_segment_rate_denominator} (per support thresholds policy)"
    )
    outcome, eligible_col = f"future_purchases_{horizon}d", f"eligible_followup_{horizon}d"
    data = collect(
        rows.group_by(dim)
        .agg(
            pl.len().alias("customers"),
            pl.col("spend_90d").sum().alias("spend_90d"),
            pl.col("median_gap").median().alias("median_gap"),
            pl.col(outcome).filter(pl.col(eligible_col)).count().alias("followup_n"),
            (pl.col(outcome) > 0).filter(pl.col(eligible_col)).sum().alias("future_buyers"),
        )
        .sort("customers", descending=True)
    )
    records = []
    for record in data.to_dicts():
        n, buyers = record["followup_n"], record["future_buyers"]

        # Evaluate evidence quality
        evidence_state = evaluate_evidence(
            n=record["customers"],
            numerator=buyers,
            denominator=n,
            thresholds=thresholds,
            analysis_type="segment",
        )

        allowed = coverage and n >= minimum
        lo, hi = wilson(buyers, n) if allowed else (None, None)  # noqa: F823
        record.update(
            rate=buyers / n if allowed else None,
            low=lo,
            high=hi,
            evidence_level=evidence_state.evidence_level,
            evidence_message=evidence_state.message,
        )
        if not coverage:
            record["followup_n"], record["future_buyers"] = None, None
        records.append(record)
    data = pl.DataFrame(records)

    # Segment bar chart
    bars(
        data,
        dim,
        st.selectbox("Segment bar measure", ["customers", "spend_90d"]),
        "Segment comparison",
        "segment_bars",
        color_map,
    )

    # Evidence badges for each segment
    st.markdown("**Evidence Quality by Segment**")
    for record in records:
        badge = format_evidence_badge(EvidenceState(
            has_evidence=record.get("evidence_level") in ["strong", "limited"],
            evidence_level=record.get("evidence_level", "insufficient"),
            n=record["customers"],
            message=record.get("evidence_message", ""),
        ))
        st.write(f"{record[dim]}: {badge}")

    # Multiplicity warning
    n_valid = len([r for r in records if r["rate"] is not None])
    mult_warn = multiplicity_warning(n_valid, thresholds.max_exploratory_comparisons)
    if mult_warn:
        st.warning(mult_warn)

    valid = [r for r in records if r["rate"] is not None][:25]
    if valid:
        import plotly.graph_objects as go
        fig = go.Figure(
            go.Scatter(
                x=[r["rate"] for r in valid],
                y=[r["dim"] for r in valid],
                mode="markers",
                marker=dict(size=10, color=[get_color(r["dim"], color_map) for r in valid]),
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
        "Intervals assume independent customers at one cutoff, not causal effects. "
        f"Up to {SupportThresholds().max_exploratory_comparisons} estimates shown; "
        "exploratory comparisons are not multiplicity-adjusted."
    )
    table(data)
    st.download_button(
        "Export filtered segment summary", export_csv(data), "segments.csv", "text/csv"
    )


__all__ = ["render_segments"]
