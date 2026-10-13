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

from retail_analytics.core.constants import (
    QUANTILE_STYLE_HIGH,
    QUANTILE_STYLE_LOW,
    QUANTILE_VALUE_HIGH,
    QUANTILE_VALUE_LOW,
)
from retail_analytics.core.retail_polars_workflow import COLS
from retail_analytics.core.stats import wilson
from retail_analytics.dashboard.dashboard_backend import cutoffs
from retail_analytics.visualization.product_affinity import (
    ProductAffinityConfig,
    basket_composition_by_segment,
    cross_sell_recommendations,
    new_to_repeat_pathways,
    product_affinity_analysis,
)
from retail_analytics.visualization.visual_insights_core import (
    METRICS,
    basket_composition_slope,
    behavior_trajectory_matrix,
    cadence_map,
    cadence_return_validation,
    calendar_period_revenue_trend,
    category_opportunity_map,
    category_participation_comparison,
    category_penetration_growth_portfolio,
    cohort_age_retention_heatmap,
    cohort_summary,
    collect,
    concentration,
    full_population_distribution,
    future_outcome_validation,
    momentum_map,
    multi_horizon_cohort_repeat,
    product_association_network,
    purchase_rhythm_panel,
    revenue_bridge,
    revenue_waterfall,
    rfm_response_surface,
    segment_alluvial_flow,
    time_to_second_purchase,
    transition_summary,
    trial_conversion_interval_plot,
    trial_to_repeat_map,
    value_at_risk_scatter,
    value_tier_transition_waterfall,
)
# Import view modules
from retail_analytics.dashboard.views import (
    render_data_trust_quality,
    render_customer_base_summary,
    render_key_signals,
    render_cohort_age_retention,
    render_time_to_repeat,
    render_multi_horizon_cohorts,
    render_purchase_cadence,
    render_cadence_validation,
    render_rfm_response_surface,
    render_value_at_risk_scatter,
    render_revenue_bridge,
    render_customer_momentum,
    render_spend_concentration,
    render_category_penetration_growth,
    render_category_opportunity_map,
    render_category_participation,
    render_product_association_network,
    render_trial_conversion_intervals,
    render_product_affinity,
    render_customer_brief,
    render_customer_timeline,
    render_metric_definitions,
    render_thresholds_calibration,
    render_data_quality_report,
    render_analysis_settings,
)


def safe_list(series):
    """Safely convert a Polars Series to a Python list, handling nulls and type conversion."""
    try:
        # Convert to Python native types, replacing nulls with None
        return [
            None if v is None or (isinstance(v, float) and math.isnan(v)) else v
            for v in series.to_list()
        ]
    except Exception:
        return []


def safe_to_python(val):
    """Convert a single value to Python native type."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if hasattr(val, "item"):  # numpy/polars scalar
        return val.item()
    return val


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


# ============================================================================
# Standardized Chart Component (ChartCard)
# ============================================================================

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ChartEvidence:
    """Evidence metadata for a chart."""
    numerator: Optional[str] = None
    denominator: Optional[str] = None
    population_size: Optional[int] = None
    comparison_period: Optional[str] = None
    confidence_interval: Optional[str] = None
    interval_type: Optional[str] = None  # e.g., "Wilson 95%", "bootstrap"
    eligibility_criteria: Optional[str] = None
    sample_policy: Optional[str] = None  # e.g., "hash-sampled 2000", "full population"
    additional: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartCard:
    """
    Standardized chart component with consistent explanation metadata.
    
    Per improvement plan: every chart should show question, takeaway, evidence,
    how-to-read, caveat, and next-step.
    """
    # Core identification
    key: str  # Unique key for Streamlit
    title: str  # Chart title
    
    # Explanation metadata (per improvement plan)
    question: str = ""  # What decision/customer question does this answer?
    takeaway: str = ""  # One concise factual sentence from displayed aggregates
    evidence: ChartEvidence = field(default_factory=ChartEvidence)
    how_to_read: str = ""  # Short interpretation of axes, bins, color, reference lines
    caveat: str = ""  # Sparse history, incomplete coverage, sampling, etc.
    next_step: str = ""  # One relevant drill-down, filter, or adjacent chart
    
    # Accessibility
    alt_description: str = ""  # Screen-reader description for Plotly
    
    # Chart object
    figure: Optional[go.Figure] = None
    
    # Display options
    height: int = 430
    show_evidence: bool = True
    show_how_to_read: bool = True
    show_caveat: bool = True
    show_next_step: bool = True
    
    def render(self) -> None:
        """Render the chart card with standardized layout."""
        # Title and question
        st.subheader(self.title)
        if self.question:
            st.caption(f"**Question:** {self.question}")
        
        # Takeaway (prominent)
        if self.takeaway:
            st.info(f"**Takeaway:** {self.takeaway}")
        
        # Evidence panel
        if self.show_evidence and self._has_evidence():
            with st.expander("📊 Evidence & Methodology", expanded=False):
                self._render_evidence()
        
        # How to read
        if self.show_how_to_read and self.how_to_read:
            with st.expander("📖 How to read this chart", expanded=False):
                st.markdown(self.how_to_read)
        
        # Render the Plotly chart
        if self.figure is not None:
            self._render_chart()
        
        # Caveat
        if self.show_caveat and self.caveat:
            st.warning(f"⚠ **Caveat:** {self.caveat}")
        
        # Next step
        if self.show_next_step and self.next_step:
            st.success(f"🔍 **Next step:** {self.next_step}")
    
    def _has_evidence(self) -> bool:
        e = self.evidence
        return any([
            e.numerator, e.denominator, e.population_size,
            e.comparison_period, e.confidence_interval,
            e.interval_type, e.eligibility_criteria, e.sample_policy
        ])
    
    def _render_evidence(self) -> None:
        e = self.evidence
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
            self.figure.update_layout(
                title=self.title,
                template="plotly_white",
                height=self.height,
                margin=dict(l=20, r=20, t=65, b=50),
                uirevision=self.key,
            )
            st.plotly_chart(
                self.figure,
                use_container_width=True,
                key=self.key,
                config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}},
                # Accessibility
                alt=self.alt_description or f"Chart: {self.title}",
            )
        except Exception as exc:
            st.error(f"Chart rendering failed for '{self.title}': {type(exc).__name__}: {exc}")
            st.caption("This may be due to empty data, invalid data types, or missing columns.")


def create_chart_card(
    key: str,
    title: str,
    figure: go.Figure,
    question: str = "",
    takeaway: str = "",
    evidence: Optional[ChartEvidence] = None,
    how_to_read: str = "",
    caveat: str = "",
    next_step: str = "",
    alt_description: str = "",
    **kwargs,
) -> ChartCard:
    """Factory function to create a ChartCard with all metadata."""
    return ChartCard(
        key=key,
        title=title,
        figure=figure,
        question=question,
        takeaway=takeaway,
        evidence=evidence or ChartEvidence(),
        how_to_read=how_to_read,
        caveat=caveat,
        next_step=next_step,
        alt_description=alt_description,
        **kwargs,
    )

# ============================================================================
# Sparse-Data Safeguards (Phase 3: Strengthen evidence for limited data)
# ============================================================================

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


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
) -> Tuple[float, float]:
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
    strata: List[str] = None,
) -> str:
    """Generate sampling disclosure text."""
    pct = (sampled_n / population_n * 100) if population_n > 0 else 0
    base = (
        f"Chart shows {sampled_n:,} of {population_n:,} customers ({pct:.1}%) "
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


def check_horizon_eligibility(manifest: dict, horizon_days: int, cutoff_date: date = None) -> tuple[bool, str]:
    """
    Check if a future-purchase horizon is eligible given coverage confirmation.
    
    Returns:
        (eligible: bool, reason: str)
    """
    coverage_confirmed = manifest.get("coverage_confirmed", False)
    if not coverage_confirmed:
        return False, "Coverage not confirmed — future outcomes unavailable"
    
    coverage_complete_through = manifest.get("coverage_complete_through")
    if not coverage_complete_through:
        return False, "Complete-through date not set"
    
    complete_date = date.fromisoformat(coverage_complete_through)
    
    if cutoff_date is None:
        # Use latest cutoff from manifest
        cutoff_date = date.fromisoformat(manifest["snapshots"][-1])
    
    horizon_end = cutoff_date + timedelta(days=horizon_days)
    
    if horizon_end > complete_date:
        return False, (
            f"Insufficient observed follow-up: cutoff {cutoff_date} + {horizon_days}d "
            f"= {horizon_end} exceeds complete-through {complete_date}"
        )
    
    return True, f"Eligible: complete-through {complete_date} covers horizon end {horizon_end}"


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
    """Render a Plotly chart with error handling."""
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
    
    # Read coverage fields from manifest (P0 fix)
    coverage_confirmed = manifest.get("coverage_confirmed", False)
    coverage_complete_through = manifest.get("coverage_complete_through")
    coverage_scope_note = manifest.get("coverage_scope_note")
    coverage_confirmation_source = manifest.get("coverage_confirmation_source")
    latest_observed_transaction_date = manifest.get("latest_observed_transaction_date")
    observed_through = manifest.get("observed_through")
    
    # Effective coverage for future outcomes
    coverage = bool(coverage_confirmed)

    # ============================================================
    # Coverage Status Banner (P0: explicit eligibility display)
    # ============================================================
    st.markdown("---")
    st.subheader("🔬 Observation Coverage Status")
    
    if coverage_confirmed:
        st.success(
            f"✅ **Coverage confirmed** (source: {coverage_confirmation_source or 'user_assertion'})"
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest transaction in file", latest_observed_transaction_date or "N/A")
        with col2:
            st.metric("Data complete through", coverage_complete_through or "N/A")
        with col3:
            st.metric("Analysis observed through", observed_through or "N/A")
        
        if coverage_scope_note and coverage_scope_note != "unknown":
            st.info(f"📍 **Scope covered:** {coverage_scope_note}")
        
        # Check for horizon eligibility
        chosen_for_check = manifest["snapshots"][-1]  # Latest cutoff
        chosen_date = date.fromisoformat(chosen_for_check)
        if coverage_complete_through:
            complete_date = date.fromisoformat(coverage_complete_through)
            for horizon in [30, 60, 90]:
                horizon_end = chosen_date + timedelta(days=horizon)
                if horizon_end > complete_date:
                    st.warning(
                        f"⚠ **{horizon}-day horizon incomplete:** "
                        f"Cutoff {chosen_for_check} + {horizon}d = {horizon_end} > "
                        f"complete-through {coverage_complete_through}. "
                        f"This horizon will show as unavailable."
                    )
    else:
        st.error(
            "❌ **Coverage NOT confirmed** — Future outcome validation is disabled.\n\n"
            "Historical descriptive metrics are available. "
            "To enable repeat-purchase rates, cohort repeat, RFM response surface, "
            "and other future-outcome views, re-run analysis with coverage confirmation "
            "and a complete-through date."
        )
        st.metric("Latest transaction in file", latest_observed_transaction_date or "N/A")
        st.caption(
            "The latest observed transaction date does NOT prove complete capture. "
            "Exports can omit stores, channels, customers, or delayed events."
        )
    
    st.markdown("---")
    
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
                lateness = thresholds.get("lateness_overdue_min")
                if lateness:
                    st.metric("Lateness threshold", f"{lateness:.2f}")
                freq_thresh = thresholds.get("frequency_change_threshold")
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
    elif (
        thresholds.get("reference_date") is None
        and thresholds.get("reference_selection", {}).get("selected") is False
    ):
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

    # Question-based navigation structure (per improvement plan)
    NAV_GROUPS = {
        "01 · Start Here": [
            "Data Trust & Quality",
            "Customer Base Summary",
            "Key Signals",
        ],
        "02 · Lifecycle & Repeat": [
            "Cohort-Age Retention",
            "Time to Repeat",
            "Multi-Horizon Cohorts",
            "Purchase Cadence",
        ],
        "03 · Value & Risk": [
            "RFM Response Surface",
            "Value-at-Risk Scatter",
            "Revenue Bridge",
            "Customer Momentum",
            "Spend Concentration",
        ],
        "04 · Products & Categories": [
            "Category Penetration × Growth",
            "Category Opportunity Map",
            "Category Participation",
            "Product Association Network",
            "Trial Conversion Intervals",
            "Product Affinity",
        ],
        "05 · Explore Customers": [
            "Customer Brief",
            "Customer Timeline",
        ],
        "06 · Methods & Data Quality": [
            "Metric Definitions",
            "Thresholds & Calibration",
            "Data Quality Report",
            "Analysis Settings",
        ],
        "07 · Predictive Modeling": [
            "Model Evaluation",
            "CLV Predictions",
        ],
        "Advanced Analytics": [
            # Segment Migration & Quality (legacy group)
            "Segment Alluvial Flow",
            "Behavior × Trajectory",
            "Transitions",
            "Basket Evolution",
            "Future Validation",
            "Quality",
            # Exploration (legacy group)
            "Overview",
            "Segments",
            "Features",
            "Cohorts",
            "Customer",
            "Methods",
            "Full-Population Distributions",
            # Research-backed priority views (legacy)
            "Cadence Validation",
            "Category Sankey",
            "Value-Tier Transitions",
            "Purchase Rhythm",
            "Revenue Waterfall",
        ],
    }

    # Two-level navigation: section selector + view selector
    section = st.selectbox(
        "Section",
        list(NAV_GROUPS.keys()),
        index=0,
        key="nav_section",
    )
    view = st.radio(
        "View",
        NAV_GROUPS[section],
        horizontal=True,
        key="nav_view",
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
        render_segments(root, rows, coverage, color_map)
    elif view == "Data Trust & Quality":
        render_data_trust_quality(root, rows, coverage, color_map)
    elif view == "Customer Base Summary":
        render_customer_base_summary(rows, coverage, color_map)
    elif view == "Key Signals":
        render_key_signals(rows, coverage, color_map)
    elif view == "Customer Brief":
        render_customer_brief(root, rows, coverage, color_map)
    elif view == "Customer Timeline":
        render_customer_timeline(root, rows, coverage, color_map)
    elif view == "Metric Definitions":
        render_metric_definitions()
    elif view == "Thresholds & Calibration":
        render_thresholds_calibration(root)
    elif view == "Data Quality Report":
        render_data_quality_report(root)
    elif view == "Analysis Settings":
        render_analysis_settings(root)
    elif view == "Model Evaluation":
        render_model_evaluation(root, rows, coverage, color_map)
    elif view == "CLV Predictions":
        render_clv_predictions(root, rows, coverage, color_map)
    elif view == "Features":
        render_features(root, rows, coverage, color_map)
    elif view == "Customer Momentum":
        render_customer_momentum(root, rows, coverage, color_map)

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

            # Reference lines: recency = median_gap and recency = 2 * median_gap
            # We need to add these as traces since they're diagonal lines
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

            # Highlight overdue region
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

            chart(
                fig,
                "Purchase cadence map",
                "cadence_scatter",
                "Median gap (days)",
                "Recency (days)",
            )

            st.caption(
                f"Shows up to {PAGE_ROWS:,} hash-sampled customers with finite median_gap > 0 and recency. "
                f"Grey line: recency = median gap (on cadence). Red line: recency = 2× median gap (overdue threshold). "
                f"Coloured regions: green = on cadence, light orange = 1-2× overdue, darker orange = ≥2× overdue. "
                f"Lateness = recency / median_gap. Only style_eligible customers have meaningful median_gap."
            )
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

    elif view == "Category Participation":
        render_category_participation(root, rows, coverage, color_map)

    elif view == "Trial Conversion Intervals":
        render_trial_conversion_intervals(root, rows, coverage, color_map)

    elif view == "Product Affinity":
        st.subheader("Product affinity & market-basket analysis")
        st.caption(
            "Lightweight association mining using sparse basket-level co-occurrence. "
            "No dense customer×product matrices. "
            "Support = fraction of baskets containing itemset. "
            "Confidence = P(B|A). Lift = P(A∪B)/(P(A)P(B)). Conviction = (1-P(B))/(1-P(B|A)). "
            "Minimum support and top-K limits keep computation tractable."
        )

        if not coverage:
            st.warning(
                "Coverage unconfirmed: segment-specific affinity requires confirmed follow-up. "
                "Enable coverage confirmation in sidebar to analyze by segment."
            )
        else:
            # Configuration
            col1, col2, col3 = st.columns(3)
            with col1:
                min_support = st.number_input(
                    "Minimum support",
                    0.0001,
                    0.1,
                    0.001,
                    format="%.4f",
                    help="Fraction of baskets containing itemset. Lower = more pairs, more compute.",
                )
            with col2:
                min_confidence = st.number_input(
                    "Minimum confidence",
                    0.01,
                    1.0,
                    0.10,
                    format="%.2f",
                    help="P(B|A) threshold for association rules.",
                )
            with col3:
                min_lift = st.number_input(
                    "Minimum lift",
                    0.5,
                    5.0,
                    1.0,
                    format="%.2f",
                    help="Lift = P(A∪B)/(P(A)P(B)). >1 means positive association.",
                )

            col4, col5, col6 = st.columns(3)
            with col4:
                top_k_items = st.number_input(
                    "Top-K items", 50, 2000, 500, step=50, help="Max frequent items to consider."
                )
            with col5:
                top_k_pairs = st.number_input(
                    "Top-K pairs", 100, 10000, 2000, step=100, help="Max pairs to emit by support."
                )
            with col6:
                max_items_per_basket = st.number_input(
                    "Max items/basket", 10, 100, 50, help="Cap basket size to limit pair explosion."
                )

            category_level = st.checkbox("Include category-level analysis", value=True)

            # Segment filter
            segment_options = ["All customers"] + sorted(
                set(
                    str(v)
                    for dim in ["behavior", "trajectory", "value_tier", "lifecycle", "evidence"]
                    for v in collect(rows.select(dim).unique().sort(dim))[dim].to_list()
                )
            )
            segment_filter = st.selectbox("Segment filter (optional)", segment_options)
            segment = None if segment_filter == "All customers" else segment_filter

            # Load validated data for basket-level analysis
            validated_path = root / "validated.parquet"
            if not validated_path.exists():
                st.info("No validated data available for this analysis.")
            else:
                try:
                    validated = pl.scan_parquet(validated_path)

                    with st.spinner("Computing product affinities..."):
                        config = ProductAffinityConfig(
                            min_support=min_support,
                            min_confidence=min_confidence,
                            min_lift=min_lift,
                            top_k_items=top_k_items,
                            top_k_pairs=top_k_pairs,
                            max_items_per_basket=max_items_per_basket,
                            category_level=category_level,
                        )
                        result = product_affinity_analysis(validated, rows, chosen, config, segment)

                    # Frequent products
                    st.write("**Frequent products**")
                    pf = result["product_frequent"]
                    if pf.is_empty():
                        st.info("No products meet the minimum support threshold.")
                    else:
                        display_pf = pf.head(50)
                        bars(
                            display_pf,
                            "product_id",
                            "basket_count",
                            "Top 50 products by basket count",
                            "freq_products",
                            color_map,
                        )
                        st.caption(
                            f"Total frequent products: {pf.height:,} (min support = {min_support:.3%} of {pf['total_baskets'][0]:,} baskets)"
                        )
                        table(display_pf.select("product_id", "basket_count", "support"))

                    # Product association rules
                    st.write("**Product association rules**")
                    pp = result["product_pairs"]
                    if pp.is_empty():
                        st.info(
                            "No product pairs meet the minimum support/confidence/lift thresholds."
                        )
                    else:
                        display_pp = pp.head(50)
                        # Format for display
                        bars(
                            display_pp,
                            "item_a",
                            "support",
                            "Top 50 product pairs by support",
                            "prod_pairs_support",
                            color_map,
                        )
                        table(
                            display_pp.select(
                                "item_a",
                                "item_b",
                                "cooccurrence_count",
                                "support",
                                "confidence_a_to_b",
                                "confidence_b_to_a",
                                "lift",
                                "conviction_a_to_b",
                                "conviction_b_to_a",
                            )
                        )

                    # Cross-sell recommendations
                    st.write("**Cross-sell recommendations**")
                    recs = cross_sell_recommendations(
                        result["product_pairs"],
                        result.get("category_pairs"),
                        top_n=10,
                        min_lift=min_lift,
                        min_confidence=min_confidence,
                    )
                    prod_recs = recs["product_recommendations"]
                    if prod_recs.is_empty():
                        st.info("No recommendations meet the quality thresholds.")
                    else:
                        table(
                            prod_recs.head(100).select(
                                "source_item",
                                "target_item",
                                "support",
                                "confidence",
                                "lift",
                                "conviction",
                                "direction",
                            )
                        )

                    # Category-level analysis
                    if category_level:
                        st.write("**Frequent categories**")
                        cf = result.get("category_frequent", pl.DataFrame())
                        if cf.is_empty():
                            st.info("No categories meet the minimum support threshold.")
                        else:
                            display_cf = cf.head(50)
                            bars(
                                display_cf,
                                "category_key",
                                "basket_count",
                                "Top 50 categories by basket count",
                                "freq_categories",
                                color_map,
                            )
                            st.caption(f"Total frequent categories: {cf.height:,}")
                            table(display_cf.select("category_key", "basket_count", "support"))

                        st.write("**Category association rules**")
                        cp = result.get("category_pairs", pl.DataFrame())
                        if cp.is_empty():
                            st.info(
                                "No category pairs meet the minimum support/confidence/lift thresholds."
                            )
                        else:
                            display_cp = cp.head(50)
                            bars(
                                display_cp,
                                "item_a",
                                "support",
                                "Top 50 category pairs by support",
                                "cat_pairs_support",
                                color_map,
                            )
                            table(
                                display_cp.select(
                                    "item_a",
                                    "item_b",
                                    "cooccurrence_count",
                                    "support",
                                    "confidence_a_to_b",
                                    "confidence_b_to_a",
                                    "lift",
                                    "conviction_a_to_b",
                                    "conviction_b_to_a",
                                )
                            )

                        cat_recs = recs["category_recommendations"]
                        if not cat_recs.is_empty():
                            st.write("**Category cross-sell recommendations**")
                            table(
                                cat_recs.head(100).select(
                                    "source_item",
                                    "target_item",
                                    "support",
                                    "confidence",
                                    "lift",
                                    "conviction",
                                    "direction",
                                )
                            )

                    # New-to-repeat pathways
                    st.write("**New-to-repeat pathways**")
                    st.caption(
                        "Products/categories customers tend to buy AFTER their first purchase "
                        "of a given product/category. Shows sequential purchase patterns."
                    )
                    pathways = new_to_repeat_pathways(validated, rows, chosen, min_trials=5)
                    prod_paths = pathways["product_transitions"]
                    cat_paths = pathways["category_transitions"]

                    if not prod_paths.is_empty():
                        st.write("Product → next product transitions")
                        table(
                            prod_paths.head(50).select(
                                "prev_product", "product_id", "transition_count"
                            )
                        )
                    else:
                        st.info("No product transitions meet minimum threshold.")

                    if not cat_paths.is_empty():
                        st.write("Category → next category transitions")
                        table(
                            cat_paths.head(50).select(
                                "prev_category", "category_key", "transition_count"
                            )
                        )
                    else:
                        st.info("No category transitions meet minimum threshold.")

                    # Basket composition by segment
                    if segment_filter == "All customers":
                        st.write("**Basket composition by segment**")
                        st.caption(
                            "Spend share of products/categories within each segment. "
                            "Shows what different customer segments actually buy."
                        )
                        comp = basket_composition_by_segment(
                            rows,
                            validated,
                            chosen,
                            dimensions=["behavior", "value_tier", "lifecycle"],
                        )
                        for dim, data in comp.items():
                            st.write(f"By {dim}")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write("Product composition (top 20)")
                                table(
                                    data["product_composition"]
                                    .head(20)
                                    .select(dim, "product_id", "spend_share", "buyers")
                                )
                            with col_b:
                                st.write("Category composition (top 20)")
                                table(
                                    data["category_composition"]
                                    .head(20)
                                    .select(dim, "category_key", "spend_share", "buyers")
                                )

                except Exception as e:
                    st.error(f"Product affinity analysis failed: {type(e).__name__}: {e}")
                    st.caption(
                        "Check that validated.parquet exists and has required columns. Large datasets may need adjusted thresholds."
                    )

    elif view == "Revenue Waterfall":
        render_revenue_waterfall(root, rows, coverage, color_map)

    elif view == "Value-Tier Transitions":
        render_value_tier_transitions(root, rows, coverage, color_map)

    elif view == "Behavior × Trajectory":
        render_behavior_trajectory(root, rows, coverage, color_map)

    elif view == "Purchase Rhythm":
        render_purchase_rhythm(root, rows, coverage, color_map)

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

            chart(
                fig,
                f"Mean basket composition shift (n={comp_n:,})",
                "basket_slope",
                "Period",
                "Value",
            )

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

            chart(
                fig2,
                f"Median basket composition shift (n={comp_n:,})",
                "basket_dumbbell",
                "Value",
                "Metric",
            )

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

                chart(
                    fig3,
                    f"Customer-level: {x_title} vs {y_title}",
                    "basket_scatter",
                    x_title,
                    y_title,
                )

            st.caption(
                f"Comparable customers: {comp_n:,} (tenure ≥ 179 days, non-zero previous purchases). "
                "Slope chart shows mean shifts. Dumbbell shows median shifts. "
                "Identity line: points above = increase, below = decrease. "
                "Missing/recent/previous values excluded from calculations."
            )

    elif view == "Category Sankey":
        st.subheader("Category participation comparison")
        st.caption(
            "Shows category participation change between two snapshots using 365-day windows. "
            "Participation = customers buying from category / eligible population for each snapshot. "
            "Uses validated transaction data for both periods. "
            "Categories appearing in only one period are shown with zero participation in the other."
        )

        if len(history) < 2:
            st.info("Need at least two cutoffs. Select an earlier snapshot in the sidebar.")
        else:
            origin = st.selectbox(
                "Previous cutoff", history[:-1], index=len(history) - 2, key="cat_origin"
            )

            # Load validated data
            validated_path = root / "validated.parquet"
            if not validated_path.exists():
                st.info("No validated data available.")
            else:
                validated = pl.scan_parquet(validated_path)

                # Get customer populations for both snapshots
                curr_ids = rows.select("customer_id").unique()
                curr_n = collect(curr_ids.select(pl.len())).item()

                prev_rows = apply_filters(
                    pl.scan_parquet(
                        [str(p) for p in (root / "snapshots" / origin).glob("*.parquet")]
                    ),
                    filters,
                )
                prev_ids = prev_rows.select("customer_id").unique()
                prev_n = collect(prev_ids.select(pl.len())).item()

                # Current cutoff date ranges (365-day window)
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
                    chart(
                        fig,
                        f"Category participation change: {origin} → {chosen} (top {top_n})",
                        "cat_sankey",
                        "Participation (share of period population)",
                        "Category",
                    )

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

    elif view == "Future Validation":
        render_future_validation(root, rows, coverage, color_map)

    elif view == "Cohorts":
        historical = apply_filters(
            pl.scan_parquet(
                [str(p) for d in history for p in (root / "snapshots" / d).glob("*.parquet")]
            ),
            filters,
        )

        # Calendar activity heatmap (existing)
        try:
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
        except Exception as e:
            st.error(f"Cohort heatmap failed: {type(e).__name__}: {e}")

        # Age-aligned repeat purchase chart using cohort_summary (Wilson intervals)
        st.subheader("Age-aligned repeat purchase (cohort_summary)")
        
        # Per-horizon eligibility check (P0 fix) - 30-day horizon
        manifest = read_json(root / "run.json")
        chosen_date = date.fromisoformat(chosen)
        eligible, reason = check_horizon_eligibility(manifest, 30, chosen_date)
        
        if not coverage:
            st.warning(
                "⚠ **Coverage not confirmed** — Age-aligned repeat purchase is unavailable.\n\n"
                "To enable this view, re-run analysis with coverage confirmation "
                "and a complete-through date."
            )
        elif not eligible:
            st.error(f"❌ **30-day horizon ineligible:** {reason}")
        else:
            st.success(f"✅ **30-day horizon:** {reason}")
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
            except Exception as e:
                st.error(f"Age-aligned repeat purchase analysis failed: {type(e).__name__}: {e}")
                st.caption("This may be due to missing baskets data, or insufficient cohort data.")

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

        if dim != "value_tier":
            st.info(
                f"Showing {dim} transitions as Sankey diagram. For value-tier waterfall chart, use the 'Value-Tier Transitions' view."
            )

        try:
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
                        target=[
                            len(sources) + targets.index(r["to_label"]) for r in flows.to_dicts()
                        ],
                        value=safe_list(flows["len"]),
                    ),
                )
            )
            chart(fig, f"{dim}: {origin} to {chosen}", "transition_sankey")
            # Heatmap
            xs, ys, z = bounded_grid(flows.to_dicts(), "from_label", "to_label", "rate", limit=20)
        except Exception as e:
            st.error(f"Transitions analysis failed: {type(e).__name__}: {e}")
            st.caption(
                "This may be due to missing transition data, or no matched customers between cutoffs."
            )
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

    # Research-backed priority views
    elif view == "Time to Repeat":
        st.subheader("Time to second purchase: Kaplan–Meier survival curve")
        st.caption(
            "Non-contractual customer activity model (Schmittlein, Morrison & Colombo, 1987). "
            "Duration starts at first observed shopping date; ends at second distinct shopping date. "
            "Customers without a second purchase by cutoff are right-censored. "
            "Shaded bands: 95% Greenwood confidence intervals."
        )

        try:
            baskets_paths = list((root / "events" / chosen).glob("*/baskets.parquet"))
            if not baskets_paths:
                st.info("No basket data available for this cutoff.")
            else:
                baskets = pl.scan_parquet(baskets_paths)
                cutoff_date = date.fromisoformat(chosen)
                km = time_to_second_purchase(baskets, rows, cutoff_date, max_days=180)

                if km.is_empty():
                    st.info("Insufficient data for survival curve.")
                else:
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
                    chart(
                        fig,
                        "Time to second observed shopping day (Kaplan–Meier)",
                        "km_survival",
                        "Days since first observed purchase",
                        "Cumulative repeat probability",
                    )

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

                    st.caption(
                        f"Right-censored at cutoff ({cutoff_date}). First observed purchase ≠ true acquisition. "
                        f"Greenwood 95% CI. Research: Schmittlein, Morrison & Colombo (1987), Management Science."
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

    elif view == "Cadence Validation":
        st.subheader("Cadence vs observed return probability")
        st.caption(
            "Tests whether the lateness = recency/median_gap boundary (2×) corresponds to "
            "actual changes in 30/90-day return probability. "
            "Bins customers by lateness; plots Wilson 95% CI for future purchase rates."
        )

        # Per-horizon eligibility check (P0 fix)
        manifest = read_json(root / "run.json")
        chosen_date = date.fromisoformat(chosen)
        
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
        else:
            # Show eligibility for each horizon
            for h in horizon_options:
                eligible, reason = horizon_eligibility[h]
                if eligible:
                    st.success(f"✅ **{h}-day horizon:** {reason}")
                else:
                    st.error(f"❌ **{h}-day horizon:** {reason}")
            
            eligible_horizons = [h for h in horizon_options if horizon_eligibility[h][0]]
            if not eligible_horizons:
                st.info("No horizons are eligible for cadence validation with the current coverage settings.")
            else:
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
                            # Check if CI data exists
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
                            chart(
                                fig,
                                f"{horizon}-day return rate by lateness bin",
                                f"cadence_val_{horizon}",
                                "Lateness (recency / median_gap)",
                                "Future purchase rate (Wilson 95% CI)",
                            )

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

    elif view == "Multi-Horizon Cohorts":
        st.subheader("Multi-horizon repeat curves by cohort")
        st.caption(
            "Age-aligned repeat purchase rates at 30, 60, and 90 days by first-observed month cohort. "
            "Wilson 95% confidence intervals. First cohort is left-censored (first purchase = observation start)."
        )

        # Per-horizon eligibility check (P0 fix)
        manifest = read_json(root / "run.json")
        chosen_date = date.fromisoformat(chosen)
        
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
        else:
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
            else:
                try:
                    min_cohort = st.number_input("Minimum cohort size", 5, 1000, 20)

                    baskets_paths = list((root / "events" / chosen).glob("*/baskets.parquet"))
                    if not baskets_paths:
                        st.info("No basket data available for this cutoff.")
                    else:
                        baskets = pl.scan_parquet(baskets_paths)
                        cutoff_date = date.fromisoformat(chosen)
                        cohort_data = multi_horizon_cohort_repeat(
                            baskets, rows, cutoff_date, coverage, eligible_horizons, min_cohort
                        )

                        if cohort_data.is_empty():
                            st.info("No eligible cohorts.")
                        else:
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
                            chart(
                                fig,
                                "Age-aligned repeat purchase rates by cohort (30/60/90d)",
                                "multi_horizon_cohorts",
                                "Cohort month",
                                "Repeat rate (Wilson 95% CI)",
                            )

                            table(cohort_data.sort("cohort", "horizon"))
                            st.caption(
                                "Age-aligned: repeat within X days of first purchase. Wilson 95% CI for eligible cohorts. "
                                "First cohort left-censored. Caveat: first observed ≠ true acquisition."
                            )

                except Exception as e:
                    st.error(f"Multi-horizon cohort analysis failed: {type(e).__name__}: {e}")

    elif view == "Category Opportunity Map":
        st.subheader("Category opportunity map: penetration × intensity")
        st.caption(
            "X: Penetration (% of customers buying category). Y: Purchase occasions per category buyer. "
            "Bubble: Category spend share. Color: Repeat buyer share. "
            "Research basis: Dirichlet model for repeat buying (Ehrenberg, 1988)."
        )

        try:
            validated_path = root / "validated.parquet"
            if not validated_path.exists():
                st.info("No validated data available.")
            else:
                validated = pl.scan_parquet(validated_path)
                cutoff_date = date.fromisoformat(chosen)
                opp_map = category_opportunity_map(validated, rows, cutoff_date, top_categories=30)

                if opp_map.is_empty():
                    st.info("No category data for this population.")
                else:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=opp_map["penetration"].to_list(),
                            y=opp_map["occasions_per_buyer"].to_list(),
                            mode="markers+text",
                            text=opp_map["label"].to_list(),
                            textposition="top center",
                            marker=dict(
                                size=[
                                    max(10, min(80, s * 200))
                                    for s in opp_map["spend_share"].to_list()
                                ],
                                color=opp_map["repeat_buyer_share"].to_list(),
                                colorscale="Viridis",
                                showscale=True,
                                colorbar=dict(title="Repeat buyer share", thickness=15),
                                opacity=0.7,
                            ),
                            customdata=list(
                                zip(
                                    opp_map["buyers"].to_list(),
                                    opp_map["spend_share"].to_list(),
                                    opp_map["repeat_buyer_share"].to_list(),
                                    opp_map["total_spend"].to_list(),
                                )
                            ),
                            hovertemplate=(
                                "%{text}<br>"
                                "Penetration: %{x:.1%}<br>"
                                "Occasions/buyer: %{y:.2f}<br>"
                                "Buyers: %{customdata[0]:,}<br>"
                                "Spend share: %{customdata[1]:.1%}<br>"
                                "Repeat buyer share: %{customdata[2]:.1%}<br>"
                                "Total spend: %{customdata[3]:,.2f}<extra></extra>"
                            ),
                        )
                    )
                    chart(
                        fig,
                        "Category opportunity map",
                        "cat_opp_map",
                        "Penetration (customer %)",
                        "Purchase occasions per category buyer",
                    )

                    table(
                        opp_map.select(
                            "label",
                            "buyers",
                            "penetration",
                            "occasions_per_buyer",
                            "spend_share",
                            "repeat_buyer_share",
                            "total_spend",
                        )
                    )
                    st.caption(
                        "Penetration = category buyers / population. Occasions/buyer = mean shopping days per buyer. "
                        "Bubble size ∝ spend share. Color = repeat buyer share. "
                        "Research: Ehrenberg (1988), Repeat Buying; Dirichlet model benchmarks."
                    )

        except Exception as e:
            st.error(f"Category opportunity map failed: {type(e).__name__}: {e}")

    elif view == "RFM Response Surface":
        # Per-horizon eligibility check (P0 fix) - RFM uses 30-day horizon
        manifest = read_json(root / "run.json")
        chosen_date = date.fromisoformat(chosen)
        eligible, reason = check_horizon_eligibility(manifest, 30, chosen_date)
        
        if not coverage:
            st.warning(
                "⚠ **Coverage not confirmed** — Future outcome validation is unavailable.\n\n"
                "To enable RFM response surface, re-run analysis with coverage confirmation "
                "and a complete-through date."
            )
        elif not eligible:
            st.error(f"❌ **30-day horizon ineligible:** {reason}")
        else:
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

                    # Wrap in ChartCard
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

    # ============================================================
    # NEW PRIORITY VIEWS (Phase 1-4 implementation)
    # ============================================================

    elif view == "Cohort-Age Retention":
        try:
            validated_path = root / "validated.parquet"
            if not validated_path.exists():
                st.info("No validated data available for cohort-age retention.")
            else:
                validated = pl.scan_parquet(validated_path)
                cutoff_date = date.fromisoformat(chosen)
                min_cohort = st.number_input(
                    "Minimum cohort size", 5, 1000, 20, key="car_min_cohort"
                )

                heatmap_data = cohort_age_retention_heatmap(
                    validated, rows, cutoff_date, min_cohort
                )

                if heatmap_data.is_empty():
                    st.info("Insufficient data for cohort-age retention heatmap.")
                else:
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

                    # Wrap in ChartCard
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

    elif view == "Value-at-Risk Scatter":
        st.subheader("Value-at-risk: high-spend customers overdue on cadence")
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
            else:
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

                color_map = build_color_map(sample[color_by].unique().to_list())

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
                                    max(6, min(40, math.sqrt(p) * 1.5))
                                    for p in part["purchases_365d"].to_list()
                                ],
                                opacity=0.6,
                                color=get_color(name, color_map),
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

                chart(
                    fig,
                    f"Value-at-risk: lateness vs 365d spend (n={sample.height:,} sampled)",
                    "var_scatter",
                    "Lateness (recency / median_gap)",
                    "Recorded spend 365d (currency units)",
                )

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

    elif view == "Revenue Bridge":
        st.subheader("Revenue bridge: calendar-period decomposition")
        st.caption(
            "Compares two calendar periods (non-overlapping) with matched-population option. "
            "Decomposes spend change into: purchasing customers, baskets per customer, spend per basket (AOV). "
            "Also shows calendar-period trend with independent monthly windows."
        )

        try:
            validated_path = root / "validated.parquet"
            if not validated_path.exists():
                st.info("No validated data available for calendar-period trend.")
            else:
                validated = pl.scan_parquet(validated_path)

                # Calendar-period trend
                st.write("**Calendar-period revenue trend**")
                first_date = collect(rows.select(pl.col("first_purchase").min())).item()
                if first_date:
                    trend_data = calendar_period_revenue_trend(
                        validated, rows, first_date, date.fromisoformat(chosen), months=12
                    )

                    if not trend_data.is_empty():
                        # Multi-metric chart
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
                        chart(
                            fig,
                            "Calendar-period revenue components (12 months)",
                            "cal_trend",
                            "Month",
                            "Value",
                        )

                        st.caption(
                            "Non-overlapping monthly windows. Fixed population (current filtered customers). Partial months marked."
                        )

                # Revenue bridge
                st.write("**Revenue decomposition bridge**")
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
                        filters,
                    )
                    cutoff_date = date.fromisoformat(chosen)
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

                        chart(
                            fig,
                            f"Revenue bridge: {origin} to {chosen} ({'matched' if match_pop else 'unmatched'} population)",
                            "revenue_bridge",
                            "Component",
                            "Spend change (currency units)",
                        )

                        table(bridge)
                        st.caption(
                            f"Fixed-order bridge: customers → frequency → AOV. {'Matched population' if match_pop else 'Full snapshot populations'}. "
                            f"Residual captures interaction effects. Order-dependent; not causal attribution."
                        )

        except Exception as e:
            st.error(f"Revenue bridge failed: {type(e).__name__}: {e}")

    elif view == "Product Association Network":
        st.subheader("Product association network")
        st.caption(
            "Network visualization of product co-occurrence associations. "
            "Nodes = products, edges = associations exceeding minimum lift/confidence. "
            "Node size = degree (connections), color = max lift. "
            "Association ≠ causation; does not imply incremental cross-sell lift."
        )

        if not coverage:
            st.warning(
                "Coverage unconfirmed: segment-specific affinity requires confirmed follow-up."
            )
        else:
            try:
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_support = st.number_input(
                        "Min support", 0.0001, 0.1, 0.001, format="%.4f", key="net_support"
                    )
                with col2:
                    min_conf = st.number_input(
                        "Min confidence", 0.01, 1.0, 0.15, format="%.2f", key="net_conf"
                    )
                with col3:
                    min_lift = st.number_input(
                        "Min lift", 0.5, 5.0, 1.2, format="%.2f", key="net_lift"
                    )

                top_k = st.number_input("Top-K pairs", 50, 2000, 200, step=50, key="net_topk")

                validated_path = root / "validated.parquet"
                if not validated_path.exists():
                    st.info("No validated data available.")
                else:
                    validated = pl.scan_parquet(validated_path)

                    with st.spinner("Computing product affinities..."):
                        config = ProductAffinityConfig(
                            min_support=min_support,
                            min_confidence=min_conf,
                            min_lift=min_lift,
                            top_k_items=500,
                            top_k_pairs=top_k,
                            max_items_per_basket=50,
                            category_level=False,
                        )
                        result = product_affinity_analysis(validated, rows, chosen, config, None)

                    pairs = result["product_pairs"]
                    if pairs.is_empty():
                        st.info("No product pairs meet the quality thresholds.")
                    else:
                        # Build network
                        network = product_association_network(
                            pairs, min_lift=min_lift, min_confidence=min_conf, top_n=top_k
                        )

                        nodes = network["nodes"]
                        edges = network["edges"]

                        if nodes.is_empty():
                            st.info("No network to display after filtering.")
                        else:
                            # Create network layout using spring layout
                            try:
                                import networkx as nx

                                G = nx.Graph()
                                for _, node in nodes.iter_rows(named=True):
                                    G.add_node(
                                        node["product_id"],
                                        size=node["degree"],
                                        lift=node["max_lift"],
                                    )
                                for _, edge in edges.iter_rows(named=True):
                                    G.add_edge(
                                        edge["item_a"], edge["item_b"], weight=edge["lift"] or 1.0
                                    )

                                pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

                                # Build edge traces
                                edge_x, edge_y = [], []
                                for edge in G.edges():
                                    x0, y0 = pos[edge[0]]
                                    x1, y1 = pos[edge[1]]
                                    edge_x.extend([x0, x1, None])
                                    edge_y.extend([y0, y1, None])

                                edge_trace = go.Scatter(
                                    x=edge_x,
                                    y=edge_y,
                                    line=dict(width=0.5, color="#888"),
                                    hoverinfo="none",
                                    mode="lines",
                                    showlegend=False,
                                )

                                # Build node traces
                                node_x = [pos[node][0] for node in G.nodes()]
                                node_y = [pos[node][1] for node in G.nodes()]
                                node_size = [
                                    max(10, min(50, G.nodes()[node].get("size", 1) * 5))
                                    for node in G.nodes()
                                ]
                                node_color = [
                                    G.nodes()[node].get("lift", 1.0) for node in G.nodes()
                                ]
                                node_text = list(G.nodes())

                                node_trace = go.Scatter(
                                    x=node_x,
                                    y=node_y,
                                    mode="markers+text",
                                    text=node_text,
                                    textposition="top center",
                                    textfont=dict(size=8),
                                    marker=dict(
                                        size=node_size,
                                        color=node_color,
                                        colorscale="Viridis",
                                        showscale=True,
                                        colorbar=dict(title="Max lift", thickness=15),
                                        line=dict(width=1, color="white"),
                                    ),
                                    hovertemplate="Product: %{text}<br>Connections: %{marker.size}<br>Max lift: %{marker.color:.2f}<extra></extra>",
                                    showlegend=False,
                                )

                                fig = go.Figure(data=[edge_trace, node_trace])
                                fig.update_layout(
                                    showlegend=False,
                                    hovermode="closest",
                                    xaxis=dict(
                                        showgrid=False, zeroline=False, showticklabels=False
                                    ),
                                    yaxis=dict(
                                        showgrid=False, zeroline=False, showticklabels=False
                                    ),
                                    height=600,
                                )
                                chart(fig, "Product association network", "prod_network", "", "")

                            except ImportError:
                                st.warning("networkx not installed; showing edge table instead.")
                                table(edges.head(100))

                            st.caption(
                                f"Network from {edges.height:,} associations (min support={min_support:.3%}, "
                                f"min confidence={min_conf:.2f}, min lift={min_lift:.2f}). "
                                f"Node size = degree, color = max lift. Association ≠ causation."
                            )

            except Exception as e:
                st.error(f"Product association network failed: {type(e).__name__}: {e}")

    elif view == "Trial Conversion Intervals":
        st.subheader("Trial-to-repeat conversion with confidence intervals")
        st.caption(
            "Dot-and-interval plot: each product's repeat conversion rate with 95% Wilson CI. "
            "Dot size/label = eligible trial count. "
            "Filters by minimum eligible trials. Identifies products with consistently high repeat behavior."
        )

        try:
            min_trials = st.number_input(
                "Minimum eligible trials per product", 1, 1000, 10, key="tci_min"
            )
            show_category = st.checkbox(
                "Show category-level aggregation", value=False, key="tci_cat"
            )

            prod_history_paths = list((root / "events" / chosen).glob("*/product_history.parquet"))
            if not prod_history_paths:
                st.info("No product history available for this cutoff.")
            else:
                prod_history = pl.scan_parquet(prod_history_paths)
                cutoff_date = date.fromisoformat(chosen)

                # Category lookup if needed
                category_lookup = None
                if show_category:
                    validated_path = root / "validated.parquet"
                    if validated_path.exists():
                        validated = pl.scan_parquet(validated_path)
                        category_lookup = collect(
                            validated.select("product_id", "department", "category")
                            .unique()
                            .with_columns(
                                pl.concat_str(
                                    [pl.col("department"), pl.lit(" / "), pl.col("category")]
                                ).alias("category_label")
                            )
                        )

                trial_data = trial_conversion_interval_plot(
                    prod_history, rows, cutoff_date, min_trials, category_lookup
                )

                if trial_data.is_empty():
                    st.info(f"No products with ≥ {min_trials} eligible trials.")
                else:
                    # Dot-and-interval plot
                    fig = go.Figure()

                    products = trial_data.sort("trial_conversion", descending=True).head(
                        50
                    )  # Top 50 for readability

                    for _, row in products.iter_rows(named=True):
                        prod_id = row["product_id"]
                        rate = row["trial_conversion"]
                        lo = row["lower_ci"]
                        hi = row["upper_ci"]
                        n = row["eligible_trials"]
                        buyers = row["repeated_trials"]
                        customers = row["trialling_customers"]

                        # Determine y-position (product rank)
                        # We'll use product_id as categorical
                        fig.add_trace(
                            go.Scatter(
                                x=[rate],
                                y=[prod_id],
                                mode="markers",
                                marker=dict(
                                    size=max(8, min(30, math.sqrt(n) * 1.5)),
                                    color="#0072B2",
                                    opacity=0.8,
                                ),
                                name=prod_id,
                                error_x=dict(
                                    type="data",
                                    symmetric=False,
                                    array=[hi - rate] if hi else None,
                                    arrayminus=[rate - lo] if lo else None,
                                    visible=True,
                                    color="gray",
                                    thickness=2,
                                )
                                if lo is not None
                                else None,
                                customdata=[[n, buyers, customers]],
                                hovertemplate=f"Product: {prod_id}<br>Conversion: {rate:.1%}<br>Trials: {n}<br>Repeats: {buyers}<br>Customers: {customers}<br>CI: [{lo:.1% if lo else 'N/A'}, {hi:.1% if hi else 'N/A'}]<extra></extra>",
                                showlegend=False,
                            )
                        )

                    fig.update_xaxes(
                        range=[0, 1],
                        tickformat=".0%",
                        title="Trial conversion rate (95% Wilson CI)",
                    )
                    fig.update_yaxes(autorange="reversed", title="Product ID")
                    fig.update_layout(height=max(400, products.height * 20))

                    chart(
                        fig,
                        f"Trial-to-repeat conversion (min {min_trials} trials)",
                        "trial_conv_intervals",
                        "Trial conversion rate",
                        "Product",
                    )

                    st.caption(
                        "Wilson 95% CI shown for products with ≥20 eligible trials. "
                        "Products with fewer trials have no CI (high uncertainty). "
                        "Pooled conversion = total repeats / total eligible trials across customers."
                    )

                    display_cols = [
                        "product_id",
                        "eligible_trials",
                        "repeated_trials",
                        "trialling_customers",
                        "trial_conversion",
                        "lower_ci",
                        "upper_ci",
                    ]
                    if "category_label" in trial_data.columns:
                        display_cols.insert(1, "category_label")
                    table(trial_data.select(display_cols))

        except Exception as e:
            st.error(f"Trial conversion intervals failed: {type(e).__name__}: {e}")

    elif view == "Category Penetration × Growth":
        st.subheader("Category penetration vs growth portfolio")
        st.caption(
            "X: Current category penetration (% of customers buying category). "
            "Y: Change in participation vs previous window (percentage points). "
            "Bubble size: Category spend share. Color: Repeat buyer share. "
            "Identifies expanding/declining categories and their commercial weight."
        )

        try:
            if len(history) < 2:
                st.info("Need at least two cutoffs for period comparison.")
            else:
                origin = st.selectbox(
                    "Previous cutoff", history[:-1], index=len(history) - 2, key="cpg_origin"
                )
                top_n = st.slider("Top categories", 10, 50, 30, key="cpg_top")

                validated_path = root / "validated.parquet"
                if not validated_path.exists():
                    st.info("No validated data available.")
                else:
                    validated = pl.scan_parquet(validated_path)
                    current_cutoff = date.fromisoformat(chosen)
                    previous_cutoff = date.fromisoformat(origin)

                    portfolio = category_penetration_growth_portfolio(
                        validated, rows, current_cutoff, previous_cutoff, top_n
                    )

                    if portfolio.is_empty():
                        st.info("No category data for this comparison.")
                    else:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=portfolio["curr_penetration"].to_list(),
                                y=portfolio["penetration_change_pp"].to_list(),
                                mode="markers+text",
                                text=portfolio["label"].to_list(),
                                textposition="top center",
                                textfont=dict(size=9),
                                marker=dict(
                                    size=[
                                        max(10, min(80, s * 300))
                                        for s in portfolio["spend_share"].to_list()
                                    ],
                                    color=portfolio["repeat_buyer_share"].to_list(),
                                    colorscale="Viridis",
                                    showscale=True,
                                    colorbar=dict(title="Repeat buyer share", thickness=15),
                                    opacity=0.7,
                                    line=dict(width=1, color="white"),
                                ),
                                customdata=list(
                                    zip(
                                        portfolio["curr_buyers"].to_list(),
                                        portfolio["prev_buyers"].to_list(),
                                        portfolio["curr_spend"].to_list(),
                                        portfolio["prev_spend"].to_list(),
                                        portfolio["spend_share"].to_list(),
                                        portfolio["repeat_buyer_share"].to_list(),
                                    )
                                ),
                                hovertemplate=(
                                    "%{text}<br>"
                                    "Current penetration: %{x:.1%}<br>"
                                    "Change (pp): %{y:.2f}<br>"
                                    "Current buyers: %{customdata[0]:,}<br>"
                                    "Previous buyers: %{customdata[1]:,}<br>"
                                    "Current spend: %{customdata[2]:,.2f}<br>"
                                    "Previous spend: %{customdata[3]:,.2f}<br>"
                                    "Spend share: %{customdata[4]:.1%}<br>"
                                    "Repeat buyer share: %{customdata[5]:.1%}<extra></extra>"
                                ),
                            )
                        )

                        # Add quadrant reference lines
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                        median_pen = portfolio["curr_penetration"].median()
                        fig.add_vline(
                            x=median_pen, line_dash="dash", line_color="gray", line_width=1
                        )

                        # Quadrant annotations
                        fig.add_annotation(
                            x=0.9,
                            y=portfolio["penetration_change_pp"].max() * 0.8,
                            text="High penetration<br>Growing",
                            showarrow=False,
                            font=dict(size=10, color="#009E73"),
                            opacity=0.7,
                        )
                        fig.add_annotation(
                            x=0.1,
                            y=portfolio["penetration_change_pp"].max() * 0.8,
                            text="Low penetration<br>Growing",
                            showarrow=False,
                            font=dict(size=10, color="#56B4E9"),
                            opacity=0.7,
                        )
                        fig.add_annotation(
                            x=0.9,
                            y=portfolio["penetration_change_pp"].min() * 0.8,
                            text="High penetration<br>Declining",
                            showarrow=False,
                            font=dict(size=10, color="#D55E00"),
                            opacity=0.7,
                        )
                        fig.add_annotation(
                            x=0.1,
                            y=portfolio["penetration_change_pp"].min() * 0.8,
                            text="Low penetration<br>Declining",
                            showarrow=False,
                            font=dict(size=10, color="#CC79A7"),
                            opacity=0.7,
                        )

                        chart(
                            fig,
                            "Category penetration vs growth portfolio",
                            "cat_pen_growth",
                            "Current penetration (customer %)",
                            "Penetration change vs previous window (pp)",
                        )

                        table(
                            portfolio.select(
                                "label",
                                "curr_buyers",
                                "prev_buyers",
                                "curr_penetration",
                                "prev_penetration",
                                "penetration_change_pp",
                                "curr_spend",
                                "prev_spend",
                                "spend_share",
                                "repeat_buyer_share",
                            )
                        )

                        st.caption(
                            "Fixed population (current filtered customers) for both periods. "
                            "Penetration = category buyers / population. Change in percentage points. "
                            "Category buyer populations overlap; not mutually exclusive market shares. "
                            "Descriptive signal, not demand forecast."
                        )

        except Exception as e:
            st.error(f"Category penetration × growth failed: {type(e).__name__}: {e}")

    elif view == "Segment Alluvial Flow":
        st.subheader("Segment migration alluvial flow")
        st.caption(
            "Multi-snapshot customer flow between behavioral states (value_tier, lifecycle, behavior). "
            "Shows genuine observed transitions; unmatched origin/destination shown separately. "
            "Does not label unmatched as churn/acquisition."
        )

        try:
            if len(history) < 2:
                st.info("Need at least two cutoffs for segment flow.")
            else:
                dim = st.selectbox(
                    "Segment dimension",
                    ["value_tier", "lifecycle", "behavior", "trajectory", "evidence"],
                    key="saf_dim",
                )
                num_snapshots = st.slider(
                    "Number of snapshots",
                    2,
                    min(5, len(history)),
                    min(3, len(history)),
                    key="saf_snaps",
                )

                # Build history of filtered rows
                rows_history = []
                for d in history[-num_snapshots:]:
                    r = apply_filters(
                        pl.scan_parquet(
                            [str(p) for p in (root / "snapshots" / d).glob("*.parquet")]
                        ),
                        filters,
                    )
                    rows_history.append(r)

                flow_data = segment_alluvial_flow(rows_history, dimensions=[dim])

                flows = flow_data["flows"]
                snapshots = flow_data["snapshots"]

                if flows.is_empty():
                    st.info("No matched customers across selected snapshots.")
                else:
                    # Build Sankey across multiple snapshots
                    # For multi-period, we chain transitions: Period1→Period2, Period2→Period3, etc.
                    all_sources = []
                    all_targets = []
                    all_values = []

                    # Collect all unique states with period prefix
                    for i, (from_snap, to_snap) in enumerate(snapshots):
                        period_flows = flows.filter(
                            (pl.col("from_snapshot") == from_snap)
                            & (pl.col("to_snapshot") == to_snap)
                        )
                        if period_flows.is_empty():
                            continue

                        for row in period_flows.to_dicts():
                            source = f"{from_snap}: {row['from_state']}"
                            target = f"{to_snap}: {row['to_state']}"
                            all_sources.append(source)
                            all_targets.append(target)
                            all_values.append(row["customer_count"])

                    # Create unique node list
                    all_nodes = list(dict.fromkeys(all_sources + all_targets))
                    node_colors = []
                    for node in all_nodes:
                        # Extract state label (after ": ")
                        state = node.split(": ", 1)[-1] if ": " in node else node
                        node_colors.append(get_color(state, color_map))

                    source_indices = [all_nodes.index(s) for s in all_sources]
                    target_indices = [all_nodes.index(t) for t in all_targets]

                    fig = go.Figure(
                        go.Sankey(
                            node=dict(
                                label=all_nodes,
                                color=node_colors,
                                pad=15,
                                thickness=20,
                            ),
                            link=dict(
                                source=source_indices,
                                target=target_indices,
                                value=all_values,
                                color="rgba(0,114,178,0.3)",
                            ),
                        )
                    )

                    chart(
                        fig,
                        f"{dim} migration across {num_snapshots} snapshots",
                        "segment_alluvial",
                        "",
                        "",
                    )

                    # Show reconciliation per period
                    st.write("**Reconciliation per transition**")
                    for from_snap, to_snap in snapshots:
                        period_flows = flows.filter(
                            (pl.col("from_snapshot") == from_snap)
                            & (pl.col("to_snapshot") == to_snap)
                        )
                        if period_flows.is_empty():
                            continue
                        matched = period_flows["customer_count"].sum()
                        st.caption(f"{from_snap} → {to_snap}: {matched:,} matched customers")

                    st.caption(
                        "Flow width = customer count. Only matched customers (present in both snapshots) shown. "
                        "Unmatched origin customers are not churn; unmatched destination are new arrivals. "
                        "States prefixed with snapshot date for clarity."
                    )

        except Exception as e:
            st.error(f"Segment alluvial flow failed: {type(e).__name__}: {e}")

    elif view == "Spend Concentration":
        st.subheader("Customer spend concentration (Lorenz curve)")
        st.caption(
            "Lorenz curve of 90-day spend distribution. "
            "Shows concentration of revenue among customers. "
            "Equality line (diagonal) represents perfectly even distribution."
        )

        conc = concentration(rows)
        if conc["n"] > 0:
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

    elif view == "Full-Population Distributions":
        st.subheader("Full-population feature distributions")
        st.caption(
            "Aggregated distribution summaries computed across the entire eligible population, not sampled. "
            "Shows histogram bins, quantiles, and summary statistics for key behavioral features. "
            "Avoids sampling bias that can conceal tail behavior and small high-value groups."
        )

        try:
            dist_data = full_population_distribution(rows)

            if not dist_data:
                st.info("No numeric features available for distribution analysis.")
            else:
                feature = st.selectbox("Feature", list(dist_data.keys()), key="fpd_feature")
                data = dist_data[feature]

                summary = data["summary"]
                hist = data["histogram"]
                n = data["n"]

                st.write(f"**{feature}** — {n:,} eligible customers")
                st.dataframe(summary, hide_index=True, use_container_width=True)

                if hist.height > 0:
                    fig = go.Figure(
                        go.Bar(
                            x=hist["bin"].to_list(),
                            y=hist["len"].to_list(),
                            marker_color="#0072B2",
                            opacity=0.7,
                        )
                    )
                    chart(
                        fig,
                        f"Full-population histogram: {feature}",
                        f"fpd_hist_{feature}",
                        "Bin",
                        "Customer count",
                    )

                st.caption(
                    "Computed on full eligible population (finite values only). Not a 2,000-row sample."
                )

        except Exception as e:
            st.error(f"Full-population distributions failed: {type(e).__name__}: {e}")

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


# ============================================================================
# New view render functions (question-based navigation)
# ============================================================================

def render_data_trust_quality(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Data Trust & Quality - Can these records support the analysis being requested?"""
    st.subheader("Data Trust & Quality")
    st.caption(
        "Question: Can these records support the analysis being requested? "
        "Shows quality funnel, date coverage, missingness, and source limitations."
    )

    # Load quality.json
    quality_path = root / "quality.json"
    if quality_path.exists():
        quality = read_json(quality_path)
    else:
        st.warning("Quality report not found.")
        return

    # Quality funnel
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Source rows", f"{quality.get('source_rows', 0):,}")
    col2.metric("Valid rows", f"{quality.get('valid_rows', 0):,}")
    col3.metric("Rejected rows", f"{quality.get('rejected_rows', 0):,}")
    col4.metric("Rejection rate", f"{quality.get('rejection_rate', 0):.2%}")

    # Date coverage
    st.markdown("**Date Coverage**")
    col1, col2, col3 = st.columns(3)
    col1.metric("First observed date", str(quality.get('first_date', 'N/A')))
    col2.metric("Last observed date", str(quality.get('last_date', 'N/A')))
    col3.metric("Stated complete through", str(quality.get('observed_through', 'N/A')))

    # Coverage confirmation status
    coverage_confirmed = quality.get('coverage_confirmed', False)
    if coverage_confirmed:
        st.success("✓ Observation coverage confirmed by user")
    else:
        st.warning("⚠ Observation coverage NOT confirmed — future-purchase metrics unavailable")
    st.caption(quality.get('coverage_interpretation', ''))

    # Rejection breakdown
    if quality.get('rejection_breakdown'):
        st.markdown("**Rejection Breakdown**")
        breakdown_df = pl.DataFrame([
            {"Reason": k, "Count": v} for k, v in quality['rejection_breakdown'].items()
        ])
        table(breakdown_df)

    # Data scope limitations (from METRICS.md)
    st.markdown("**Data Scope Limitations**")
    st.markdown("""
    - **Purchases only**: No return/refund handling. All valid rows treated as purchases.
    - **Currency opaque**: Amounts in input data's currency; no conversion or symbol assumed.
    - **First observed ≠ acquisition**: First recorded purchase may not be true customer acquisition.
    - **Basket grain**: Assumes `(customer_id, transaction_id, date)` uniquely identifies a basket.
    - **Relative segmentation**: Quantile thresholds depend on snapshot population; not absolute benchmarks.
    - **No causal claims**: Descriptive associations only. Segment differences ≠ treatment effects.
    """)

    # Missingness matrix (sample of features)
    st.markdown("**Feature Missingness (sample)**")
    features = [f for f in FEATURES if f in rows.collect_schema().names()]
    expressions = [pl.len().alias("n")]
    for f in features[:15]:  # Limit to first 15
        finite = pl.col(f).filter(pl.col(f).is_finite())
        expressions += [finite.count().alias(f + "_n")]
    stats = collect(rows.select(expressions)).row(0, named=True)
    missingness = pl.DataFrame([
        dict(feature=f, missing=stats["n"] - stats[f + "_n"], missing_pct=(stats["n"] - stats[f + "_n"]) / stats["n"] if stats["n"] else 0)
        for f in features[:15]
    ]).sort("missing", descending=True)
    table(missingness)


def render_customer_base_summary(rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Customer Base Summary - Is customer activity and recorded sales changing?"""
    st.subheader("Customer Base Summary")
    st.caption(
        "Question: Is customer activity and recorded sales changing? "
        "Shows active customers, baskets, recorded sales over time with calendar-month alignment."
    )

    # Overall metrics
    metrics = collect(
        rows.select(
            pl.len().alias("customers"),
            pl.col("spend_90d").sum().alias("sales"),
            pl.col("purchases_90d").sum().alias("baskets"),
            (pl.col("purchases_90d") > 0).sum().alias("active"),
        )
    ).row(0, named=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total customers", f"{metrics['customers']:,}")
    col2.metric("Active (90d)", f"{metrics['active']:,}")
    col3.metric("Baskets (90d)", f"{metrics['baskets']:,}")
    col4.metric("Recorded sales (90d)", f"{metrics['sales']:,.2f}")

    st.caption("Recorded amounts are currency units, not profit. Windows end at the cutoff; inactive observed customers remain included.")

    # Segment composition
    st.markdown("**Population Composition**")
    for dim in DIMS:
        data = collect(
            rows.group_by(dim).agg(
                pl.len().alias("customers"), pl.col("spend_90d").sum().alias("spend_90d")
            ).sort("customers", descending=True)
        )
        if not data.is_empty():
            with st.expander(f"By {dim}", expanded=(dim == "lifecycle")):
                bars(data, dim, "customers", f"Customers by {dim}", f"comp_cust_{dim}", color_map)
                bars(data, dim, "spend_90d", f"Spend by {dim}", f"comp_spend_{dim}", color_map)


def render_key_signals(rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Key Signals - The 3-5 most useful signals from the data."""
    st.subheader("Key Signals")
    st.caption(
        "Question: What are the most important patterns in the data right now? "
        "Highlights: repeat rates, overdue customers, category concentration, momentum."
    )

    # Signal 1: Repeat rate (if coverage confirmed)
    if coverage:
        horizon = 90
        outcome, eligible = f"future_purchases_{horizon}d", f"eligible_followup_{horizon}d"
        repeat_data = collect(
            rows.filter(pl.col(eligible))
            .select(
                pl.len().alias("eligible_n"),
                (pl.col(outcome) > 0).sum().alias("repeat_n"),
            )
        )
        if repeat_data["eligible_n"][0] > 0:
            rate = repeat_data["repeat_n"][0] / repeat_data["eligible_n"][0]
            lo, hi = wilson(repeat_data["repeat_n"][0], repeat_data["eligible_n"][0])
            st.metric(
                f"Next-{horizon}-day repeat rate",
                f"{rate:.1%}",
                help=f"Wilson 95% CI: [{lo:.1%}, {hi:.1%}] | n={repeat_data['eligible_n'][0]:,}"
            )
        else:
            st.info("No eligible customers for repeat rate calculation.")

    # Signal 2: Overdue customers (lateness >= 2)
    overdue = collect(
        rows.filter(pl.col("lateness").is_not_null() & (pl.col("lateness") >= 2))
        .select(
            pl.len().alias("overdue_n"),
            pl.col("spend_90d").sum().alias("overdue_spend"),
        )
    ).row(0, named=True)
    total = collect(rows.select(pl.len().alias("total"))).row(0)[0]
    if total > 0:
        overdue_pct = overdue["overdue_n"] / total
        st.metric(
            "Overdue customers (lateness ≥ 2)",
            f"{overdue['overdue_n']:,} ({overdue_pct:.1%})",
            help=f"Representing {overdue['overdue_spend']:,.2f} in 90d spend"
        )

    # Signal 3: Category concentration (HHI)
    hhi_data = collect(
        rows.filter(pl.col("category_hhi").is_not_null())
        .select(pl.col("category_hhi").median().alias("median_hhi"))
    ).row(0, named=True)
    if hhi_data["median_hhi"] is not None:
        st.metric(
            "Median category concentration (HHI)",
            f"{hhi_data['median_hhi']:.3f}",
            help="Higher = more concentrated spend in few categories. Range: [1/N, 1]"
        )

    # Signal 4: Customer momentum (frequency + basket change)
    momentum = collect(
        rows.filter(
            pl.col("frequency_change").is_not_null() &
            pl.col("basket_change").is_not_null()
        )
        .select(
            pl.len().alias("comparable_n"),
            (pl.col("frequency_change") > 0).sum().alias("freq_up"),
            (pl.col("frequency_change") < 0).sum().alias("freq_down"),
            (pl.col("basket_change") > 0).sum().alias("basket_up"),
            (pl.col("basket_change") < 0).sum().alias("basket_down"),
        )
    ).row(0, named=True)
    if momentum["comparable_n"] > 0:
        col1, col2 = st.columns(2)
        col1.metric(
            "Frequency trend",
            f"↑ {momentum['freq_up']:,} · ↓ {momentum['freq_down']:,}",
            help=f"Of {momentum['comparable_n']:,} customers with comparable windows"
        )
        col2.metric(
            "Basket size trend",
            f"↑ {momentum['basket_up']:,} · ↓ {momentum['basket_down']:,}",
            help="Recent 90d vs previous 90d AOV change"
        )

    # Signal 5: Evidence distribution
    evidence_data = collect(
        rows.group_by("evidence").agg(pl.len().alias("count")).sort("count", descending=True)
    )
    if not evidence_data.is_empty():
        # Create bar chart figure
        fig = go.Figure(
            go.Bar(
                x=evidence_data["count"].to_list(),
                y=evidence_data["evidence"].to_list(),
                orientation="h",
                marker_color=[get_color(v, color_map) for v in evidence_data["evidence"].to_list()],
            )
        )
        
        # Wrap in ChartCard
        card = create_chart_card(
            key="evidence_dist_card",
            title="Customers by Evidence Level",
            figure=fig,
            question="How much evidence supports each customer's behavioral classification?",
            takeaway=f"Population split: {evidence_data['count'].to_list()} across {evidence_data.height} evidence levels",
            evidence=ChartEvidence(
                numerator="Customers in each evidence level",
                denominator="Total filtered customers",
                population_size=int(evidence_data["count"].sum()),
                interval_type="Exact counts (no sampling)",
            ),
            how_to_read=(
                "Horizontal bars show customer count per evidence level. "
                "**Stronger** = style-eligible with ≥12 usable gaps; "
                "**Limited** = style-eligible with <12 gaps; "
                "**Insufficient** = not style-eligible. "
                "Colors are deterministic per label."
            ),
            caveat=(
                "Evidence levels are based on observed history only. "
                "First observed purchase may not be true acquisition. "
                "Left-censoring affects customers near observation start."
            ),
            next_step="Drill into 'Segments' view to see behavioral tags by evidence level.",
            alt_description="Horizontal bar chart showing customer counts by evidence level: stronger, limited, insufficient.",
        )
        card.render()


def render_customer_brief(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Customer Brief - Narrative profile with evidence and limitations.
    
    Per improvement plan CA-10: Answer 5 questions without requiring feature table interpretation:
    1. Observed relationship (first/last purchase, tenure, recency, purchase days)
    2. Shopping rhythm (median gap, usable gaps, eligibility)
    3. Basket/value pattern (recent spend, median basket, peer comparison)
    4. Breadth & preference (leading categories, concentration, repeat products)
    5. Change & evidence (recent vs prior, counts, limitations)
    """
    from retail_analytics.visualization.chart_components import (
        ChartSpec, ChartEvidence, ChartCard, create_chart_card,
        evaluate_evidence, SupportThresholds,
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
    """Customer Timeline - Narrative purchase history with rhythm and category evolution.
    
    Per improvement plan CA-10: Timeline with cadence, rhythm, and category evolution.
    Identifiers masked by default.
    """
    from retail_analytics.visualization.chart_components import (
        ChartSpec, ChartEvidence, ChartCard, create_chart_card,
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
    """Thresholds & Calibration - Quantile thresholds and calibration metadata."""
    st.subheader("Thresholds & Calibration")
    st.caption(
        "Question: How were the segmentation thresholds determined? "
        "Shows reference snapshot, populations, confidence intervals, and calibration policy."
    )

    thresholds_path = root / "thresholds.json"
    if not thresholds_path.exists():
        st.info("Thresholds not yet computed.")
        return

    thresholds = read_json(thresholds_path)

    # Reference selection
    ref_sel = thresholds.get("reference_selection", {})
    if ref_sel.get("selected"):
        st.success(
            f"✓ Thresholds calibrated from snapshot **{ref_sel.get('reference_date', 'N/A')}** "
            f"({ref_sel.get('style_population', 0):,} style-eligible, "
            f"{ref_sel.get('value_population', 0):,} value-eligible)"
        )
    else:
        st.warning("⚠ Thresholds not calibrated: no snapshot met minimum population criteria.")

    # Threshold stability note
    manifest_path = root / "run.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        chosen = manifest.get("snapshots", [])[-1] if manifest.get("snapshots") else None
        if chosen and ref_sel.get("selected"):
            ref_date = ref_sel.get("reference_date")
            ref_pop = ref_sel.get("style_population", 0)
            # Get current population from data profile
            data_profile = thresholds.get("data_profile", {})
            curr_pop = data_profile.get("style_eligible_n", 0)
            if ref_date and ref_pop > 0 and curr_pop > 0:
                sensitivity_note = threshold_sensitivity_note(
                    reference_date=ref_date,
                    reference_population=ref_pop,
                    current_snapshot=chosen,
                    current_population=curr_pop,
                )
                st.info(sensitivity_note)

    # Threshold values with CIs
    st.markdown("**Calibrated Thresholds**")
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

    # Threshold details with CIs
    st.markdown("**Threshold Details (with Confidence Intervals)**")
    details = thresholds.get("threshold_details", {})
    for name, info in details.items():
        with st.expander(f"{name}: {info.get('value', 'N/A')} (status: {info.get('status', 'N/A')})"):
            st.json(info)

    # Data profile
    data_profile = thresholds.get("data_profile", {})
    if data_profile:
        st.markdown("**Data Profile at Reference Snapshot**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total customers", f"{data_profile.get('n_customers', 0):,}")
        col2.metric("Style-eligible", f"{data_profile.get('style_eligible_n', 0):,} ({data_profile.get('style_eligible_rate', 0):.1%})")
        col3.metric("Value-eligible", f"{data_profile.get('value_eligible_n', 0):,} ({data_profile.get('value_eligible_rate', 0):.1%})")

        warnings = data_profile.get("warnings", [])
        if warnings:
            for w in warnings:
                st.warning(w)

    # Calibration policy
    st.markdown("**Calibration Policy**")
    policy = thresholds.get("calibration_policy", {})
    st.json(policy)


def render_data_quality_report(root: Path):
    """Data Quality Report - Detailed validation results."""
    st.subheader("Data Quality Report")
    st.caption(
        "Question: What data quality issues were found during ingestion? "
        "Shows rejected rows, rejection reasons, duplicate counts, and left-censoring."
    )

    quality_path = root / "quality.json"
    if not quality_path.exists():
        st.info("Quality report not found.")
        return

    quality = read_json(quality_path)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Source rows", f"{quality.get('source_rows', 0):,}")
    col2.metric("Valid rows", f"{quality.get('valid_rows', 0):,}")
    col3.metric("Rejected rows", f"{quality.get('rejected_rows', 0):,}")
    col4.metric("Rejection rate", f"{quality.get('rejection_rate', 0):.2%}")

    # Detailed rejection counters
    st.markdown("**Rejection Counters**")
    rejection_fields = [
        "null_customer_id", "null_transaction_id", "null_product_id", "null_date",
        "null_price", "nonfinite_price", "negative_price",
        "null_quantity", "nonfinite_quantity", "nonpositive_quantity",
        "nonfinite_value", "missing_price_raw", "missing_quantity_raw"
    ]
    rejection_data = {k: quality.get(k, 0) for k in rejection_fields if quality.get(k, 0) > 0}
    if rejection_data:
        rej_df = pl.DataFrame([{"Field": k, "Count": v} for k, v in rejection_data.items()])
        table(rej_df)
    else:
        st.success("No rejections recorded.")

    # Duplicate analysis
    st.markdown("**Duplicate Analysis**")
    col1, col2 = st.columns(2)
    col1.metric("Exact duplicate rows", f"{quality.get('exact_duplicate_rows', 0):,}")
    col1.metric("Exact duplicate value", f"{quality.get('exact_duplicate_value', 0):,.2f}")
    col2.metric("Normalized duplicate rows", f"{quality.get('normalized_duplicate_rows', 0):,}")
    col2.metric("Normalized duplicate value", f"{quality.get('normalized_duplicate_value', 0):,.2f}")

    # Left-censoring
    st.markdown("**Left-Censoring (customers first seen near observation start)**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Left-censored customers", f"{quality.get('left_censored_customers', 0):,}")
    col2.metric("Total customers at start", f"{quality.get('total_customers_at_start', 0):,}")
    col3.metric("Censor threshold (days)", f"{quality.get('censor_threshold_days', 0):,}")

    # Baskets with rejected lines
    st.metric("Baskets with rejected lines", f"{quality.get('baskets_with_rejected_lines', 0):,}")

    # Negative quantity (potential returns)
    st.markdown("**Negative Quantity Rows (potential returns/refunds)**")
    col1, col2 = st.columns(2)
    col1.metric("Rows with negative quantity", f"{quality.get('negative_quantity_rows', 0):,}")
    col2.metric("Negative quantity value", f"{quality.get('negative_quantity_value', 0):,.2f}")

    # Product variation
    st.markdown("**Product Variation Audit**")
    variation_path = root / "product_variation.parquet"
    if variation_path.exists():
        variation = pl.read_parquet(variation_path)
        if not variation.is_empty():
            st.warning(f"Found {variation.height} products with multiple department/category/description combinations")
            table(variation.head(50))
        else:
            st.success("No product variation issues found.")
    else:
        st.info("Product variation audit not available.")


def render_analysis_settings(root: Path):
    """Analysis Settings - Run configuration and parameters."""
    st.subheader("Analysis Settings")
    st.caption(
        "Question: How was this analysis configured? "
        "Shows snapshots, price mode, bucketing, threshold policy, and version."
    )

    config_path = root / "configuration.json"
    if config_path.exists():
        config = read_json(config_path)
        st.json(config)
    else:
        st.info("Configuration not found.")

    # Run manifest
    manifest_path = root / "run.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        st.markdown("**Run Manifest**")
        st.json(manifest)


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
            "Lower = more snapshots eligible; higher = more stable thresholds.",
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

    # Coverage confirmation - P0 fix: explicit user assertion required
    st.sidebar.markdown("**🔬 Observation coverage (required for future outcomes)**")
    
    # Show latest observed transaction date (computed, read-only)
    st.sidebar.info(f"📅 Latest transaction in file: **{last}**")
    
    # Coverage confirmation checkbox (default OFF)
    coverage_confirmed = st.sidebar.checkbox(
        "I confirm data is complete through a specific date",
        value=False,
        help="Check only if you can assert that ALL in-scope transactions are captured through the date below. "
             "Exports may omit stores, channels, or delayed events. When unchecked, future-purchase validation is disabled."
    )
    
    # Complete-through date input (only enabled when coverage confirmed)
    if coverage_confirmed:
        complete_through = st.sidebar.date_input(
            "Data complete through (inclusive)",
            value=last,
            min_value=first,
            max_value=last,
            help="Latest date for which you assert complete capture of all in-scope activity. "
                 "Future-purchase horizons extending past this date will be unavailable."
        )
        coverage_scope_note = st.sidebar.text_input(
            "Scope covered (stores/channels/business units, or 'unknown')",
            value="unknown",
            help="Optional: describe which stores, channels, or business units this export covers."
        )
        coverage_warning = (
            "⚠ **Warning:** Future repeat-purchase validation will use this date. "
            "If the export omits any in-scope activity through this date, "
            "observed repeat rates will be invalid."
        )
        st.sidebar.warning(coverage_warning)
        end = complete_through
    else:
        complete_through = None
        coverage_scope_note = None
        st.sidebar.warning(
            "⚠ **Coverage unconfirmed:** Historical descriptions available; "
            "future outcome validation (repeat rates, cohort repeat, RFM response surface, etc.) is disabled. "
            "Enable above only if you can confirm complete capture."
        )
        end = last  # Historical analysis still works; use latest date for cutoff bounds

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
        max_value=end - timedelta(days=1) if end > first + timedelta(days=30) else first + timedelta(days=30),
        help="Cutoff date for feature computation. Future windows start after this date.",
    )

    # Auto-calculated cutoffs
    dates = cutoffs(chosen, first, default_count)
    st.sidebar.caption(f"Cutoffs (monthly, {len(dates)}): " + ", ".join(map(str, dates)))

    # Override option for advanced users
    with st.sidebar.expander("Advanced: override auto settings", expanded=False):
        count = st.slider("Historical cutoffs", 1, 12, default_count)
        dates = cutoffs(chosen, first, count)
        partitions = st.slider("Customer partitions", 1, 32, default_partitions)
        st.caption(
            f"Partitions: {partitions} (≈{valid_customers // max(1, partitions)} customers each)"
        )

    job = dict(
        base,
        action="analyze",
        snapshots=[str(x) for x in dates],
        buckets=partitions,
        exclude_invalid=exclude,
        coverage_confirmed=coverage_confirmed,
        coverage_complete_through=str(complete_through) if complete_through else None,
        coverage_scope_note=coverage_scope_note,
        coverage_confirmation_source="user_assertion" if coverage_confirmed else None,
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
    if st.session_state.get("result"):
        try:
            display_results(Path(st.session_state["result"]), coverage_confirmed)
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
