"""Shared chart components and specifications for consistent, explainable visualizations.

This module provides the ChartCard system for standardized chart rendering
with consistent metadata: question, takeaway, evidence, how-to-read, caveat, next-step.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from retail_analytics.core.coverage import check_horizon_eligibility, get_latest_observed_transaction_date
from retail_analytics.io.json_files import read_json


@dataclass
class ChartEvidence:
    """Evidence metadata for a chart."""
    numerator: str | None = None
    denominator: str | None = None
    population_size: int | None = None
    comparison_period: str | None = None
    confidence_interval: str | None = None
    interval_type: str | None = None  # e.g., "Wilson 95%", "bootstrap"
    eligibility_criteria: str | None = None
    sample_policy: str | None = None  # e.g., "hash-sampled 2000", "full population"
    additional: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartSpec:
    """
    Specification for a chart - defines the metadata contract.
    
    Per improvement plan Section 6: every chart should have question, takeaway,
    metric/unit, population, denominator, time_window, uncertainty, caveat, next_step.
    """
    # Core identification
    key: str
    title: str

    # Explanation metadata (required for primary views)
    question: str = ""           # What decision/customer question does this answer?
    takeaway: str = ""           # One concise factual sentence from displayed aggregates
    evidence: ChartEvidence = field(default_factory=ChartEvidence)
    how_to_read: str = ""        # Short interpretation of axes, bins, color, reference lines
    caveat: str = ""             # Sparse history, incomplete coverage, sampling, etc.
    next_step: str = ""          # One relevant drill-down, filter, or adjacent chart

    # Accessibility
    alt_description: str = ""    # Screen-reader description for Plotly

    # Chart object
    figure: go.Figure | None = None

    # Display options
    height: int = 430
    show_evidence: bool = True
    show_how_to_read: bool = True
    show_caveat: bool = True
    show_next_step: bool = True


@dataclass
class ChartCard:
    """
    Standardized chart component with consistent explanation metadata.
    
    Per improvement plan: every chart should show question, takeaway, evidence,
    how-to-read, caveat, and next-step.
    """
    spec: ChartSpec

    def render(self) -> None:
        """Render the chart card with standardized layout."""
        # Title and question
        st.subheader(self.spec.title)
        if self.spec.question:
            st.caption(f"**Question:** {self.spec.question}")

        # Takeaway (prominent)
        if self.spec.takeaway:
            st.info(f"**Takeaway:** {self.spec.takeaway}")

        # Evidence panel
        if self.spec.show_evidence and self._has_evidence():
            with st.expander("📊 Evidence & Methodology", expanded=False):
                self._render_evidence()

        # How to read
        if self.spec.show_how_to_read and self.spec.how_to_read:
            with st.expander("📖 How to read this chart", expanded=False):
                st.markdown(self.spec.how_to_read)

        # Render the Plotly chart
        if self.spec.figure is not None:
            self._render_chart()

        # Caveat
        if self.spec.show_caveat and self.spec.caveat:
            st.warning(f"⚠ **Caveat:** {self.spec.caveat}")

        # Next step
        if self.spec.show_next_step and self.spec.next_step:
            st.success(f"🔍 **Next step:** {self.spec.next_step}")

    def _has_evidence(self) -> bool:
        e = self.spec.evidence
        return any([
            e.numerator, e.denominator, e.population_size,
            e.comparison_period, e.confidence_interval,
            e.interval_type, e.eligibility_criteria, e.sample_policy
        ])

    def _render_evidence(self) -> None:
        e = self.spec.evidence
        cols = st.columns(3)
        idx = 0

        if e.population_size is not None:
            with cols[idx % 3]:
                st.metric("Population (n)", f"{e.population_size:,}")
            idx += 1

        if e.numerator:
            with cols[idx % 3]:
                st.metric("Numerator", e.numerator)
            idx += 1

        if e.denominator:
            with cols[idx % 3]:
                st.metric("Denominator", e.denominator)
            idx += 1

        if e.comparison_period:
            with cols[idx % 3]:
                st.metric("Period", e.comparison_period)
            idx += 1

        if e.confidence_interval:
            with cols[idx % 3]:
                st.metric("Confidence Interval", e.confidence_interval)
            idx += 1

        if e.interval_type:
            with cols[idx % 3]:
                st.metric("Interval Type", e.interval_type)
            idx += 1

        if e.eligibility_criteria:
            with cols[idx % 3]:
                st.metric("Eligibility", e.eligibility_criteria)
            idx += 1

        if e.sample_policy:
            with cols[idx % 3]:
                st.metric("Sample Policy", e.sample_policy)
            idx += 1

        # Additional evidence fields
        for key, value in e.additional.items():
            with cols[idx % 3]:
                st.metric(key.replace("_", " ").title(), str(value))
            idx += 1

    def _render_chart(self) -> None:
        """Render the Plotly chart with accessibility."""
        try:
            self.spec.figure.update_layout(
                title=self.spec.title,
                template="plotly_white",
                height=self.spec.height,
                margin=dict(l=20, r=20, t=65, b=50),
                uirevision=self.spec.key,
            )
            st.plotly_chart(
                self.spec.figure,
                use_container_width=True,
                key=self.spec.key,
                config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}},
                # Accessibility
                alt=self.spec.alt_description or f"Chart: {self.spec.title}",
            )
        except Exception as exc:
            st.error(f"Chart rendering failed for '{self.spec.title}': {type(exc).__name__}: {exc}")
            st.caption("This may be due to empty data, invalid data types, or missing columns.")


def create_chart_card(
    key: str,
    title: str,
    figure: go.Figure,
    question: str = "",
    takeaway: str = "",
    evidence: ChartEvidence | None = None,
    how_to_read: str = "",
    caveat: str = "",
    next_step: str = "",
    alt_description: str = "",
    **kwargs,
) -> ChartCard:
    """Factory function to create a ChartCard with all metadata."""
    spec = ChartSpec(
        key=key,
        title=title,
        question=question,
        takeaway=takeaway,
        evidence=evidence or ChartEvidence(),
        how_to_read=how_to_read,
        caveat=caveat,
        next_step=next_step,
        alt_description=alt_description,
        figure=figure,
        **kwargs,
    )
    return ChartCard(spec=spec)


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


def build_color_map(labels) -> dict[str, str]:
    """Build deterministic color map from sorted labels using Okabe-Ito palette."""
    sorted_labels = sorted(set(str(label) for label in labels))
    return {label: OKABE_ITO[i % len(OKABE_ITO)] for i, label in enumerate(sorted_labels)}


def get_color(label: str, color_map: dict[str, str]) -> str:
    """Get color for label from color map, fallback to hash."""
    return color_map.get(str(label), "#64748b")


def format_axis_title(metric_name: str, METRICS: dict[str, tuple] = None) -> str:
    """Convert metric column name to human-readable title with units."""
    if METRICS and metric_name in METRICS:
        return f"{METRICS[metric_name][0]} ({METRICS[metric_name][1]})"
    # Fallback: title case with underscores replaced
    return metric_name.replace("_", " ").title()


def safe_list(series) -> list:
    """Safely convert a Polars Series to a Python list, handling nulls and type conversion."""
    import math
    try:
        return [
            None if v is None or (isinstance(v, float) and math.isnan(v)) else v
            for v in series.to_list()
        ]
    except Exception:
        return []


# ============================================================================
# Sparse-Data Safeguards (Phase 3: Strengthen evidence for limited data)
# ============================================================================


@dataclass
class SupportThresholds:
    """Documented minimum-support policy by analysis type."""
    # Segment comparisons
    min_segment_size: int = 20
    min_segment_rate_denominator: int = 30

    # Cohort analysis
    min_cohort_size: int = 30
    min_cohort_periods: int = 3

    # Product associations
    min_basket_support: float = 0.001  # 0.1% of baskets
    min_pair_confidence: float = 0.10  # 10%
    min_pair_lift: float = 1.0
    min_pair_count: int = 5

    # Scatter plots / sampling
    max_scatter_points: int = 2000
    min_stratified_per_segment: int = 50

    # Bootstrap
    bootstrap_iterations: int = 500
    bootstrap_min_sample: int = 20

    # Multiplicity
    max_exploratory_comparisons: int = 20

    # Threshold stability
    threshold_min_population: int = 100
    threshold_max_ci_width_ratio: float = 0.5


@dataclass
class EvidenceState:
    """Represents the evidence quality for a metric/group."""
    has_evidence: bool
    evidence_level: str  # "strong", "limited", "insufficient", "ineligible"
    n: int
    message: str
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def evaluate_evidence(
    n: int,
    numerator: int = None,
    denominator: int = None,
    thresholds: SupportThresholds = None,
    analysis_type: str = "segment",
) -> EvidenceState:
    """
    Evaluate evidence quality for a metric/group.
    
    Returns EvidenceState with level and actionable message.
    """
    if thresholds is None:
        thresholds = SupportThresholds()

    if n == 0:
        return EvidenceState(
            has_evidence=False,
            evidence_level="insufficient",
            n=0,
            message="No observations in this group.",
            details={"reason": "zero_count"}
        )

    if analysis_type == "segment":
        if n < thresholds.min_segment_size:
            return EvidenceState(
                has_evidence=False,
                evidence_level="insufficient",
                n=n,
                message=f"Insufficient evidence: {n} customers (minimum {thresholds.min_segment_size} for segment comparisons).",
                details={"reason": "below_min_segment", "minimum": thresholds.min_segment_size}
            )
        if denominator is not None and denominator < thresholds.min_segment_rate_denominator:
            return EvidenceState(
                has_evidence=True,
                evidence_level="limited",
                n=n,
                message=f"Limited evidence: {denominator} eligible for rate calculation (minimum {thresholds.min_segment_rate_denominator}).",
                details={"reason": "low_denominator", "denominator": denominator, "minimum": thresholds.min_segment_rate_denominator}
            )
        return EvidenceState(
            has_evidence=True,
            evidence_level="strong",
            n=n,
            message=f"Adequate evidence: {n} customers.",
            details={"reason": "adequate"}
        )

    elif analysis_type == "cohort":
        if n < thresholds.min_cohort_size:
            return EvidenceState(
                has_evidence=False,
                evidence_level="insufficient",
                n=n,
                message=f"Cohort too small: {n} customers (minimum {thresholds.min_cohort_size}).",
                details={"reason": "below_min_cohort", "minimum": thresholds.min_cohort_size}
            )
        return EvidenceState(
            has_evidence=True,
            evidence_level="strong" if n >= thresholds.min_cohort_size * 2 else "limited",
            n=n,
            message=f"Cohort size: {n} customers.",
            details={"reason": "adequate"}
        )

    elif analysis_type == "product_pair":
        if numerator is not None and numerator < thresholds.min_pair_count:
            return EvidenceState(
                has_evidence=False,
                evidence_level="insufficient",
                n=numerator,
                message=f"Pair count too low: {numerator} co-occurrences (minimum {thresholds.min_pair_count}).",
                details={"reason": "low_pair_count", "minimum": thresholds.min_pair_count}
            )
        return EvidenceState(
            has_evidence=True,
            evidence_level="strong",
            n=numerator or n,
            message=f"Product pair count: {numerator or n}.",
            details={"reason": "adequate"}
        )

    # Default
    return EvidenceState(
        has_evidence=n >= 5,
        evidence_level="strong" if n >= 30 else "limited",
        n=n,
        message=f"Observations: {n}.",
        details={"reason": "default"}
    )


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.median,
    n_iterations: int = 500,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Array of values
        statistic: Function to compute (default: median)
        n_iterations: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)
        random_seed: Seed for reproducibility
    
    Returns:
        Tuple of (lower_ci, upper_ci)
    """
    if len(data) < 2:
        return (np.nan, np.nan)

    np.random.seed(random_seed)
    boot_stats = []
    n = len(data)

    for _ in range(n_iterations):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)

    return (float(lower), float(upper))


def format_evidence_badge(state: EvidenceState) -> str:
    """Format evidence state as a colored badge for UI."""
    badges = {
        "strong": "🟢 **Strong evidence**",
        "limited": "🟡 **Limited evidence**",
        "insufficient": "🔴 **Insufficient evidence**",
        "ineligible": "⚪ **Not eligible**",
    }
    return f"{badges.get(state.evidence_level, state.evidence_level)} — {state.message}"


def multiplicity_warning(n_comparisons: int, threshold: int = 20) -> str:
    """Generate multiplicity warning for exploratory comparisons."""
    if n_comparisons > threshold:
        return (
            f"⚠ **Multiplicity warning:** {n_comparisons} exploratory comparisons shown. "
            f"These are not multiplicity-adjusted. Treat as hypothesis-generating only."
        )
    return ""


def sampling_disclosure(
    sampled_n: int,
    population_n: int,
    method: str = "deterministic hash",
    stratified: bool = False,
    strata: list[str] = None,
) -> str:
    """Generate sampling disclosure text."""
    pct = (sampled_n / population_n * 100) if population_n > 0 else 0
    base = (
        f"Chart shows {sampled_n:,} of {population_n:,} customers ({pct:.1f}%) "
        f"using {method} sampling."
    )
    if stratified and strata:
        base += f" Stratified by: {', '.join(strata)}."
    base += " This is a sample, not the full population."
    return base


def threshold_sensitivity_note(
    reference_date: str,
    reference_population: int,
    current_snapshot: str,
    current_population: int,
) -> str:
    """Generate threshold stability note."""
    if reference_population == 0 or current_population == 0:
        return ""

    change_pct = (current_population - reference_population) / reference_population * 100

    return (
        f"📌 **Threshold stability:** Quantile thresholds calibrated on "
        f"{reference_date} (n={reference_population:,}). "
        f"Current snapshot {current_snapshot} has n={current_population:,} "
        f"({change_pct:+.1f}%). "
        f"Labels are relative to reference population, not absolute benchmarks."
    )


def bounded_grid(records, row, column, value, limit=24):
    """Create a bounded grid for heatmap visualization."""
    if not 1 <= limit <= 36:
        raise ValueError("Grid limit must be 1..36")
    ys = sorted({str(r[row]) for r in records})[-limit:]
    xs = sorted({str(r[column]) for r in records})[-limit:]
    lookup = {(str(r[row]), str(r[column])): r[value] for r in records}
    return xs, ys, [[lookup.get((y, x)) for x in xs] for y in ys]


PAGE_ROWS = 2000

def collect(frame):
    """Collect a LazyFrame using the streaming engine."""
    return frame.collect(engine="streaming")

def table(frame):
    """Display a Polars DataFrame or LazyFrame as a Streamlit table."""
    import streamlit as st
    data = (
        collect(frame.limit(PAGE_ROWS))
        if hasattr(frame, "collect")
        else frame.head(PAGE_ROWS)
    )
    st.dataframe(data, hide_index=True, use_container_width=True)
    st.caption(f"Tables display at most {PAGE_ROWS:,} rows.")


def apply_filters(rows, filters):
    """Apply filter selections to a LazyFrame."""
    for dimension, values in filters.items():
        if values:
            rows = rows.filter(pl.col(dimension).is_in(values))
    return rows


def chart(figure, title, key, x=None, y=None):
    """Render a Plotly chart with error handling."""
    import streamlit as st
    try:
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
    except Exception as e:
        st.error(f"Chart rendering failed for '{title}': {type(e).__name__}: {e}")
        st.caption(
            "This may be due to empty data, invalid data types, or missing columns. Check the data and filters."
        )


def bars(data, group, metric, title, key, color_map=None):
    """Render horizontal bar chart with color mapping."""
    import plotly.graph_objects as go
    import streamlit as st
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


def export_csv(data):
    """Export DataFrame to CSV with formula injection protection."""
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


__all__ = [
    "ChartSpec", "ChartEvidence", "ChartCard", "create_chart_card",
    "SupportThresholds", "EvidenceState", "evaluate_evidence",
    "bootstrap_ci", "format_evidence_badge", "multiplicity_warning",
    "sampling_disclosure", "threshold_sensitivity_note",
    "build_color_map", "get_color", "format_axis_title", "OKABE_ITO", "safe_list",
    "check_horizon_eligibility", "get_latest_observed_transaction_date",
    "read_json",
    "bounded_grid",
    "collect", "table", "PAGE_ROWS",
    "apply_filters", "chart", "bars", "export_csv",
]
