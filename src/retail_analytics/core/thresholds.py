"""Data-driven threshold calibration with uncertainty quantification.

This module replaces hard-coded thresholds with robust, data-driven estimates
computed from the reference population at the earliest eligible cutoff.
"""

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from retail_analytics.core.constants import (
    GAP_CV_ROUTINE_MAX,
    MIN_DAYS_STYLE,
    MIN_GAP_COUNT_STYLE,
    MIN_REFERENCE_POPULATION,
    MIN_TENURE_STYLE,
    MIN_TENURE_VALUE_TIER,
    QUANTILE_STYLE_HIGH,
    QUANTILE_STYLE_LOW,
    QUANTILE_VALUE_HIGH,
    QUANTILE_VALUE_LOW,
    WILSON_Z,
)

# ──────────────────────────────────────────────────────────────────────
# Policy
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThresholdPolicy:
    """Configuration for threshold calibration."""
    value_quantile_low: float = QUANTILE_VALUE_LOW
    value_quantile_high: float = QUANTILE_VALUE_HIGH
    style_quantile_low: float = QUANTILE_STYLE_LOW
    style_quantile_high: float = QUANTILE_STYLE_HIGH
    min_n: int = MIN_REFERENCE_POPULATION
    z: float = WILSON_Z
    max_rel_ci: float = 0.5   # Reject if CI width > 50% of value

    def __post_init__(self):
        # Validate quantile pairs
        if not (0 < self.value_quantile_low < self.value_quantile_high < 1):
            raise ValueError("value_quantile_low must be < value_quantile_high and both in (0, 1)")
        if not (0 < self.style_quantile_low < self.style_quantile_high < 1):
            raise ValueError("style_quantile_low must be < style_quantile_high and both in (0, 1)")

        # Validate min_n - must be positive integer
        if not isinstance(self.min_n, int) or isinstance(self.min_n, bool) or self.min_n <= 0:
            raise ValueError("min_n must be a positive integer")

        # Validate z
        if not isinstance(self.z, (int, float)) or not math.isfinite(self.z) or self.z <= 0:
            raise ValueError("z must be a finite positive number")

        # Validate max_rel_ci
        if not isinstance(self.max_rel_ci, (int, float)) or not math.isfinite(self.max_rel_ci) or self.max_rel_ci < 0:
            raise ValueError("max_rel_ci must be a finite non-negative number")

    # Backward compatibility: support old constructor pattern
    @classmethod
    def from_legacy(
        cls,
        low_p: float | None = None,
        high_p: float | None = None,
        min_n: int | None = None,
        z: float | None = None,
        max_rel_ci: float | None = None,
        **kwargs,
    ):
        """Create policy from legacy low_p/high_p parameters.
        
        Legacy low_p maps to both value_quantile_low and style_quantile_low.
        Legacy high_p maps to both value_quantile_high and style_quantile_high.
        If value_quantile_* or style_quantile_* are explicitly provided, they take precedence.
        """
        v_low = kwargs.get("value_quantile_low", QUANTILE_VALUE_LOW)
        v_high = kwargs.get("value_quantile_high", QUANTILE_VALUE_HIGH)
        s_low = kwargs.get("style_quantile_low", QUANTILE_STYLE_LOW)
        s_high = kwargs.get("style_quantile_high", QUANTILE_STYLE_HIGH)

        if low_p is not None:
            v_low = low_p
            s_low = low_p
        if high_p is not None:
            v_high = high_p
            s_high = high_p

        return cls(
            value_quantile_low=v_low,
            value_quantile_high=v_high,
            style_quantile_low=s_low,
            style_quantile_high=s_high,
            min_n=min_n if min_n is not None else MIN_REFERENCE_POPULATION,
            z=z if z is not None else WILSON_Z,
            max_rel_ci=max_rel_ci if max_rel_ci is not None else 0.5,
        )

    # Backward compatibility properties
    @property
    def low_p(self) -> float:
        return min(self.value_quantile_low, self.style_quantile_low)

    @property
    def high_p(self) -> float:
        return max(self.value_quantile_high, self.style_quantile_high)


@dataclass(frozen=True)
class ThresholdResult:
    """Result of a single threshold calibration."""
    name: str
    value: float | None
    raw: float | None
    ci: list[float | None]  # [low, high] - unconstrained statistical CI
    n: int
    status: str  # "ok", "insufficient_n", "unstable", "reference_unavailable", "fixed"
    method: str
    policy: dict
    reference_date: str | None = None
    clamp_bounds: list[float] | None = None  # [min, max] applied to raw value
    clamped_value: float | None = None  # value after clamping


# ──────────────────────────────────────────────────────────────────────
# Core quantile estimation with distribution-free CI
# ──────────────────────────────────────────────────────────────────────

def _quantile_with_ci(
    s: pl.Series,
    p: float,
    policy: ThresholdPolicy,
) -> dict:
    """
    Compute quantile with distribution-free order-statistic confidence interval.

    The CI uses the normal approximation to the binomial:
        rank bounds = n·p ± z·√(n·p·(1-p))
    """
    x = s.drop_nulls()
    x = x.filter(x.is_finite()).sort()
    n = x.len()

    if n == 0:
        return dict(
            value=None, raw=None, ci=[None, None],
            n=n, status="insufficient_n", method="linear"
        )

    if n < policy.min_n:
        return dict(
            value=None, raw=None, ci=[None, None],
            n=n, status="insufficient_n", method="linear"
        )

    # Point estimate (linear interpolation)
    quantile_result = x.quantile(p, interpolation="linear")
    raw_value = float(quantile_result) if quantile_result is not None else 0.0

    # Order-statistic CI bounds (rank-based, no bootstrap)
    half = policy.z * math.sqrt(n * p * (1 - p))
    lo_idx = max(0, math.floor(n * p - half))
    hi_idx = min(n - 1, math.ceil(n * p + half))

    lo_val = x[lo_idx] if lo_idx < n else None
    hi_val = x[hi_idx] if hi_idx < n else None
    lo = float(lo_val) if lo_val is not None else None
    hi = float(hi_val) if hi_val is not None else None

    # Stability gate: reject if CI is too wide relative to estimate
    unstable = False
    if lo is not None and hi is not None:
        if raw_value == 0:
            # Zero estimate: any non-zero width indicates instability
            if hi - lo > 0:
                unstable = True
        else:
            rel_width = (hi - lo) / abs(raw_value)
            if rel_width > policy.max_rel_ci:
                unstable = True

    return dict(
        value=None if unstable else raw_value,
        raw=raw_value,
        ci=[lo, hi],
        n=n,
        status="unstable" if unstable else "ok",
        method="linear",
    )


# ──────────────────────────────────────────────────────────────────────
# Reference population eligibility
# ──────────────────────────────────────────────────────────────────────

def _eligible_for_value_tier() -> pl.Expr:
    """Customers eligible for value tier calibration."""
    return (
        (pl.col("tenure") >= MIN_TENURE_VALUE_TIER)
        & (pl.col("spend_90d") > 0)
        & (pl.col("spend_90d").is_finite())
    )


def _eligible_for_style() -> pl.Expr:
    """Customers eligible for style (basket/gap) calibration."""
    return (
        (pl.col("days_365d") >= MIN_DAYS_STYLE)
        & (pl.col("tenure") >= MIN_TENURE_STYLE)
        & (pl.col("gap_count") >= MIN_GAP_COUNT_STYLE)
        & (pl.col("median_basket").is_finite())
        & (pl.col("median_gap").is_finite())
    )


def _eligible_for_lateness() -> pl.Expr:
    """Customers eligible for lateness calibration."""
    return (
        (pl.col("tenure") >= MIN_TENURE_STYLE)
        & (pl.col("gap_count") >= MIN_GAP_COUNT_STYLE)
        & (pl.col("lateness").is_finite())
    )


# ──────────────────────────────────────────────────────────────────────
# Threshold calibration
# ──────────────────────────────────────────────────────────────────────

def calibrate_thresholds(
    ref: pl.LazyFrame,
    policy: ThresholdPolicy | None = None,
    reference_date: date | None = None,
) -> dict[str, Any]:
    """
    Calibrate all relative thresholds from the reference frame.

    Args:
        ref: LazyFrame at the reference cutoff (earliest snapshot).
        policy: Calibration policy (uses defaults if None).
        reference_date: Date of the reference snapshot.

    Returns:
        Dictionary with calibrated thresholds and metadata.
    """
    if policy is None:
        policy = ThresholdPolicy()

    # Materialize reference columns we need
    ref_cols = ref.select([
        "spend_90d", "median_basket", "median_gap",
        "tenure", "days_365d", "gap_count", "lateness",
        "frequency_change", "basket_change", "breadth_change",
    ]).collect(engine="streaming")

    results: dict[str, Any] = {}
    policy_dict = asdict(policy)

    # ─── Value tier thresholds (spend_90d) ───
    value_eligible = ref_cols.filter(_eligible_for_value_tier())
    value_n = value_eligible.height

    if value_n >= policy.min_n:
        for name, p in [("value_low", policy.value_quantile_low), ("value_high", policy.value_quantile_high)]:
            res = _quantile_with_ci(value_eligible["spend_90d"], p, policy)
            results[name] = ThresholdResult(
                name=name, reference_date=str(reference_date) if reference_date else None,
                policy=policy_dict, **res
            )
    else:
        for name in ["value_low", "value_high"]:
            results[name] = ThresholdResult(
                name=name, value=None, raw=None, ci=[None, None],
                n=value_n, status="insufficient_n", method="linear",
                policy=policy_dict, reference_date=str(reference_date) if reference_date else None
            )

    # ─── Style thresholds (median_basket, median_gap) ───
    style_eligible = ref_cols.filter(_eligible_for_style())
    style_n = style_eligible.height

    if style_n >= policy.min_n:
        for name, p in [("basket_low", policy.style_quantile_low), ("basket_high", policy.style_quantile_high)]:
            res = _quantile_with_ci(style_eligible["median_basket"], p, policy)
            results[name] = ThresholdResult(
                name=name, reference_date=str(reference_date) if reference_date else None,
                policy=policy_dict, **res
            )
        for name, p in [("gap_low", policy.style_quantile_low), ("gap_high", policy.style_quantile_high)]:
            res = _quantile_with_ci(style_eligible["median_gap"], p, policy)
            results[name] = ThresholdResult(
                name=name, reference_date=str(reference_date) if reference_date else None,
                policy=policy_dict, **res
            )
    else:
        for name in ["basket_low", "basket_high", "gap_low", "gap_high"]:
            results[name] = ThresholdResult(
                name=name, value=None, raw=None, ci=[None, None],
                n=style_n, status="insufficient_n", method="linear",
                policy=policy_dict, reference_date=str(reference_date) if reference_date else None
            )

    # ─── Lateness threshold (Q95 of lateness, clamped to [1.5, 4]) ───
    # Lateness = recency / median_gap. High lateness = overdue.
    # We calibrate the cutoff from the population distribution.
    late_eligible = ref_cols.filter(_eligible_for_lateness())
    late_n = late_eligible.height

    if late_n >= policy.min_n:
        # Use high quantile (0.95) for "clearly overdue"
        res = _quantile_with_ci(late_eligible["lateness"], 0.95, policy)
        # Clamp to reasonable bounds [1.5, 4]
        raw_val = res["raw"]
        if raw_val is not None:
            clamped = max(1.5, min(4.0, raw_val))
            # Preserve unconstrained CI separately; record clamp metadata
            unconstrained_lo, unconstrained_hi = res["ci"]
            # Ensure ordered endpoints after clamping
            lo = max(1.5, unconstrained_lo) if unconstrained_lo is not None else 1.5
            hi = min(4.0, unconstrained_hi) if unconstrained_hi is not None else 4.0
            if lo > hi:
                lo, hi = hi, lo
            res["value"] = None if res["status"] == "unstable" else clamped
            res["raw"] = raw_val
            res["ci"] = [unconstrained_lo, unconstrained_hi]
            res["clamp_bounds"] = [1.5, 4.0]
            res["clamped_value"] = clamped
        results["lateness_overdue_min"] = ThresholdResult(
            name="lateness_overdue_min", reference_date=str(reference_date) if reference_date else None,
            policy=policy_dict, **res
        )
    else:
        results["lateness_overdue_min"] = ThresholdResult(
            name="lateness_overdue_min", value=None, raw=None, ci=[None, None],
            n=late_n, status="insufficient_n", method="linear",
            policy=policy_dict, reference_date=str(reference_date) if reference_date else None
        )

    # ─── Change thresholds (empirical reference-population change heuristics) ───
    # For frequency_change, basket_change:
    # Use the reference frame's existing frequency_change and basket_change,
    # filtered by tenure, then take absolute changes and calibrate Q90.
    # These are empirical heuristics, not placebo tests or a null distribution.
    change_eligible = ref_cols.filter(
        (pl.col("tenure") >= MIN_TENURE_STYLE + 180)  # Need ~2 prior windows
        & (pl.col("frequency_change").is_finite())
        & (pl.col("basket_change").is_finite())
    )
    change_n = change_eligible.height

    if change_n >= policy.min_n:
        for col_name, threshold_name in [
            ("frequency_change", "frequency_change_threshold"),
            ("basket_change", "basket_change_threshold"),
        ]:
            # Use absolute values, take Q90
            abs_vals = change_eligible[col_name].abs()
            res = _quantile_with_ci(abs_vals, 0.90, policy)
            raw_val = res["raw"]
            # Clamp to reasonable bounds [0.10, 0.50]
            if raw_val is not None:
                clamped = max(0.10, min(0.50, raw_val))
                # Preserve unconstrained CI separately; record clamp metadata
                unconstrained_lo, unconstrained_hi = res["ci"]
                # Ensure ordered endpoints after clamping
                lo = max(0.10, unconstrained_lo) if unconstrained_lo is not None else 0.10
                hi = min(0.50, unconstrained_hi) if unconstrained_hi is not None else 0.50
                if lo > hi:
                    lo, hi = hi, lo
                res["value"] = None if res["status"] == "unstable" else clamped
                res["raw"] = raw_val
                res["ci"] = [unconstrained_lo, unconstrained_hi]
                res["clamp_bounds"] = [0.10, 0.50]
                res["clamped_value"] = clamped
            results[threshold_name] = ThresholdResult(
                name=threshold_name, reference_date=str(reference_date) if reference_date else None,
                policy=policy_dict, **res
            )
    else:
        for threshold_name in ["frequency_change_threshold", "basket_change_threshold"]:
            results[threshold_name] = ThresholdResult(
                name=threshold_name, value=None, raw=None, ci=[None, None],
                n=change_n, status="insufficient_n", method="linear",
                policy=policy_dict, reference_date=str(reference_date) if reference_date else None
            )

    # ─── Fixed behavioral ratios (not calibrated, but documented) ───
    # These are kept as constants with explicit documentation.
    # They could be calibrated in future if domain expertise supports it.
    # Fixed rules: ci is null (not a statistical interval), clamp_bounds indicates the fixed value
    results["gap_cv_routine_max"] = ThresholdResult(
        name="gap_cv_routine_max", value=GAP_CV_ROUTINE_MAX, raw=GAP_CV_ROUTINE_MAX,
        ci=[None, None], clamp_bounds=[GAP_CV_ROUTINE_MAX, GAP_CV_ROUTINE_MAX], n=style_n,
        status="fixed", method="fixed",
        policy={"note": "Poisson arrival has CV≈1; 0.75 means clearly more regular"},
        reference_date=str(reference_date) if reference_date else None
    )
    results["established_share_routine_min"] = ThresholdResult(
        name="established_share_routine_min", value=0.70, raw=0.70,
        ci=[None, None], clamp_bounds=[0.70, 0.70], n=style_n, status="fixed", method="fixed",
        policy={"note": "High established share = routine repeat buyer"},
        reference_date=str(reference_date) if reference_date else None
    )
    results["dominant_share_focused_min"] = ThresholdResult(
        name="dominant_share_focused_min", value=0.70, raw=0.70,
        ci=[None, None], clamp_bounds=[0.70, 0.70], n=style_n, status="fixed", method="fixed",
        policy={}, reference_date=str(reference_date) if reference_date else None
    )
    results["established_share_focused_min"] = ThresholdResult(
        name="established_share_focused_min", value=0.70, raw=0.70,
        ci=[None, None], clamp_bounds=[0.70, 0.70], n=style_n, status="fixed", method="fixed",
        policy={}, reference_date=str(reference_date) if reference_date else None
    )
    results["median_breadth_broad_min"] = ThresholdResult(
        name="median_breadth_broad_min", value=3, raw=3,
        ci=[None, None], clamp_bounds=[3, 3], n=style_n, status="fixed", method="fixed",
        policy={}, reference_date=str(reference_date) if reference_date else None
    )
    results["new_share_exploration_min"] = ThresholdResult(
        name="new_share_exploration_min", value=0.40, raw=0.40,
        ci=[None, None], clamp_bounds=[0.40, 0.40], n=style_n, status="fixed", method="fixed",
        policy={}, reference_date=str(reference_date) if reference_date else None
    )
    results["breadth_change_narrowing_max"] = ThresholdResult(
        name="breadth_change_narrowing_max", value=-0.5, raw=-0.5,
        ci=[None, None], clamp_bounds=[-0.5, -0.5], n=change_n, status="fixed", method="fixed",
        policy={}, reference_date=str(reference_date) if reference_date else None
    )
    results["overdue_categories_narrowing_min"] = ThresholdResult(
        name="overdue_categories_narrowing_min", value=2, raw=2,
        ci=[None, None], clamp_bounds=[2, 2], n=change_n, status="fixed", method="fixed",
        policy={}, reference_date=str(reference_date) if reference_date else None
    )
    results["frequency_change_expanding_min"] = ThresholdResult(
        name="frequency_change_expanding_min", value=0.25, raw=0.25,
        ci=[None, None], clamp_bounds=[0.25, 0.25], n=change_n, status="fixed", method="fixed",
        policy={}, reference_date=str(reference_date) if reference_date else None
    )
    results["basket_change_expanding_min"] = ThresholdResult(
        name="basket_change_expanding_min", value=0.25, raw=0.25,
        ci=[None, None], clamp_bounds=[0.25, 0.25], n=change_n, status="fixed", method="fixed",
        policy={}, reference_date=str(reference_date) if reference_date else None
    )
    results["reactivation_events_min"] = ThresholdResult(
        name="reactivation_events_min", value=1, raw=1,
        ci=[None, None], clamp_bounds=[1, 1], n=style_n, status="fixed", method="fixed",
        policy={}, reference_date=str(reference_date) if reference_date else None
    )

    # ─── Package into output dict (backward compatible flat keys + rich metadata) ───
    output = {
        # Flat keys for backward compatibility with label()
        "value_low": results["value_low"].value,
        "value_high": results["value_high"].value,
        "basket_low": results["basket_low"].value,
        "basket_high": results["basket_high"].value,
        "gap_low": results["gap_low"].value,
        "gap_high": results["gap_high"].value,
        # Calibrated behavioral thresholds
        "lateness_overdue_min": results["lateness_overdue_min"].value,
        "frequency_change_threshold": results["frequency_change_threshold"].value,
        "basket_change_threshold": results["basket_change_threshold"].value,
        # Fixed thresholds (for reference)
        "gap_cv_routine_max": GAP_CV_ROUTINE_MAX,
        "established_share_routine_min": 0.70,
        "dominant_share_focused_min": 0.70,
        "established_share_focused_min": 0.70,
        "median_breadth_broad_min": 3,
        "established_share_broad_min": 0.70,
        "new_share_exploration_min": 0.40,
        "breadth_change_narrowing_max": -0.5,
        "overdue_categories_narrowing_min": 2,
        "frequency_change_expanding_min": 0.25,
        "basket_change_expanding_min": 0.25,
        "reactivation_events_min": 1,
        # Metadata
        "reference_date": str(reference_date) if reference_date else None,
        "reference_population": style_n,
        "value_population": value_n,
        "style_population": style_n,
        "calibration_policy": policy_dict,
        # Rich per-threshold metadata
        "threshold_details": {
            name: {
                "value": r.value,
                "raw": r.raw,
                "ci": r.ci,
                "clamp_bounds": r.clamp_bounds,
                "clamped_value": r.clamped_value,
                "n": r.n,
                "status": r.status,
                "method": r.method,
                "policy": r.policy,
            }
            for name, r in results.items()
        }
    }

    return output


# ──────────────────────────────────────────────────────────────────────
# Data profile for manifest
# ──────────────────────────────────────────────────────────────────────

def compute_data_profile(
    ref: pl.LazyFrame,
    observation_start: date | None = None,
    observation_end: date | None = None,
) -> dict[str, Any]:
    """
    Compute data profile diagnostics for the manifest.

    Returns observation span, eligibility rates, and distributional summaries.
    """
    # Materialize needed columns
    cols = ref.select([
        "customer_id", "first_purchase", "last_purchase",
        "tenure", "days_365d", "gap_count", "median_gap",
        "median_basket", "spend_90d", "purchases_90d",
    ]).collect(engine="streaming")

    n_customers = cols.height
    if n_customers == 0:
        return {}

    # Observation span
    if observation_start and observation_end:
        obs_span_days = (observation_end - observation_start).days
    else:
        obs_span_days = None

    # Eligibility rates
    style_eligible_n = cols.filter(_eligible_for_style()).height
    value_eligible_n = cols.filter(_eligible_for_value_tier()).height
    zero_spend_n = cols.filter((pl.col("spend_90d") == 0) | (pl.col("spend_90d").is_null())).height

    # Median gap distribution
    median_gap_stats = cols.select(
        pl.col("median_gap").median().alias("median"),
        pl.col("median_gap").quantile(0.25).alias("q25"),
        pl.col("median_gap").quantile(0.75).alias("q75"),
        pl.col("median_gap").min().alias("min"),
        pl.col("median_gap").max().alias("max"),
    ).to_dicts()[0] if cols["median_gap"].is_not_null().any() else {}

    # Tenure distribution
    tenure_stats = cols.select(
        pl.col("tenure").median().alias("median"),
        pl.col("tenure").quantile(0.25).alias("q25"),
        pl.col("tenure").quantile(0.75).alias("q75"),
    ).to_dicts()[0]

    # Days_365d distribution
    days_stats = cols.select(
        pl.col("days_365d").median().alias("median"),
        pl.col("days_365d").quantile(0.25).alias("q25"),
        pl.col("days_365d").quantile(0.75).alias("q75"),
    ).to_dicts()[0]

    # Gap count distribution
    gap_stats = cols.select(
        pl.col("gap_count").median().alias("median"),
        pl.col("gap_count").quantile(0.25).alias("q25"),
        pl.col("gap_count").quantile(0.75).alias("q75"),
    ).to_dicts()[0]

    # Warning flags
    warnings = []
    if style_eligible_n / n_customers < 0.1:
        warnings.append(f"Low style eligibility: {style_eligible_n}/{n_customers} ({style_eligible_n/n_customers:.1%})")
    if median_gap_stats.get("median", 0) > 90:
        warnings.append(f"Median gap ({median_gap_stats['median']:.0f}d) exceeds 90-day window; many customers may have <2 visits/window")
    if zero_spend_n / n_customers > 0.5:
        warnings.append(f"High zero-spend share: {zero_spend_n}/{n_customers} ({zero_spend_n/n_customers:.1%})")

    return {
        "n_customers": n_customers,
        "observation_span_days": obs_span_days,
        "style_eligible_n": style_eligible_n,
        "style_eligible_rate": style_eligible_n / n_customers if n_customers else 0,
        "value_eligible_n": value_eligible_n,
        "value_eligible_rate": value_eligible_n / n_customers if n_customers else 0,
        "zero_spend_n": zero_spend_n,
        "zero_spend_rate": zero_spend_n / n_customers if n_customers else 0,
        "median_gap": median_gap_stats,
        "tenure": tenure_stats,
        "days_365d": days_stats,
        "gap_count": gap_stats,
        "warnings": warnings,
    }


# ──────────────────────────────────────────────────────────────────────
# Label share diagnostics
# ──────────────────────────────────────────────────────────────────────

def compute_label_shares(
    frame: pl.LazyFrame,
) -> dict[str, Any]:
    """Compute share of customers per label for diagnostics."""
    labels = frame.select(["behavior", "value_tier", "lifecycle", "evidence"]).collect()
    n = labels.height
    if n == 0:
        return {}

    out: dict[str, Any] = {}
    for col in ["behavior", "value_tier", "lifecycle", "evidence"]:
        vc = labels[col].value_counts().sort("count", descending=True)
        out[col] = [
            {"label": row[col], "count": row["count"], "share": row["count"] / n}
            for row in vc.to_dicts()
        ]
        # Flag sparse or dominant labels
        warnings_list: list[str] = []
        for row in vc.to_dicts():
            share = row["count"] / n
            if share < 0.01:
                warnings_list.append(
                    f"Label '{col}={row[col]}' covers only {share:.1%} of customers"
                )
            elif share > 0.60:
                warnings_list.append(
                    f"Label '{col}={row[col]}' covers {share:.1%} of customers (dominant)"
                )
        if warnings_list:
            out["warnings"] = warnings_list
    return out


# ──────────────────────────────────────────────────────────────────────
# Select reference cutoff
# ──────────────────────────────────────────────────────────────────────

def select_reference_cutoff(
    numeric_paths: list[Path],
    snapshots: list[date],
    root: Path,
    policy: ThresholdPolicy | None = None,
) -> tuple[date | None, pl.LazyFrame | None, dict[str, Any]]:
    """
    Select the earliest cutoff whose reference population meets min_n and stability.

    Iterates through snapshots in chronological order until finding one with
    sufficient and stable reference population. Returns (reference_date, ref_frame, info).

    If none meet the criteria, returns (None, None, info) and earlier snapshots
    will get "reference_unavailable" labels.

    Args:
        numeric_paths: List of paths to numeric parquet files (pre-label features).
        snapshots: List of snapshot dates.
        root: Output root directory.
        policy: Calibration policy.
    """
    if policy is None:
        policy = ThresholdPolicy()

    # Map snapshot date -> list of numeric paths for that date
    numeric_by_date: dict[date, list[Path]] = {}
    for p in numeric_paths:
        # Path format: root / "numeric" / "YYYY-MM-DD" / "bN.parquet"
        try:
            d = date.fromisoformat(p.parent.name)
            numeric_by_date.setdefault(d, []).append(p)
        except ValueError:
            continue

    checked: list[dict[str, Any]] = []
    for d in sorted(snapshots):
        if d not in numeric_by_date:
            checked.append({"date": str(d), "status": "no_numeric_data"})
            continue
        try:
            ref = pl.scan_parquet([str(p) for p in numeric_by_date[d]])
            # Quick check: does this snapshot have enough style-eligible customers?
            style_n = ref.filter(_eligible_for_style()).select(pl.len()).collect().item()
            value_n = ref.filter(_eligible_for_value_tier()).select(pl.len()).collect().item()

            if style_n < policy.min_n or value_n < policy.min_n:
                checked.append({
                    "date": str(d),
                    "status": "insufficient_population",
                    "style_population": style_n,
                    "value_population": value_n,
                    "min_n": policy.min_n,
                })
                continue

            # Also check stability by doing a quick calibration
            thresholds = calibrate_thresholds(ref, policy, d)
            # Check if key thresholds are stable
            key_statuses = {
                k: thresholds["threshold_details"][k]["status"]
                for k in ["value_low", "value_high", "basket_low", "basket_high", "gap_low", "gap_high"]
            }
            key_stable = all(v == "ok" for v in key_statuses.values())

            checked.append({
                "date": str(d),
                "status": "selected" if key_stable else "unstable_thresholds",
                "style_population": style_n,
                "value_population": value_n,
                "threshold_statuses": key_statuses,
            })

            if key_stable:
                return d, ref, {
                    "selected": True,
                    "reference_date": str(d),
                    "style_population": style_n,
                    "value_population": value_n,
                    "checked_snapshots": checked,
                }
        except (ValueError, KeyError, IndexError) as e:
            # Expected errors during calibration (insufficient data, missing columns, etc.)
            # Record and continue to next snapshot
            checked.append({
                "date": str(d),
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)[:500],
            })
            continue
        except Exception as e:
            # Unexpected errors (I/O, programming errors, schema corruption) - propagate with context
            raise RuntimeError(f"Reference selection failed at cutoff {d}: {type(e).__name__}: {e}") from e

    # No cutoff met criteria
    return None, None, {
        "selected": False,
        "reference_date": None,
        "style_population": 0,
        "value_population": 0,
        "checked_snapshots": checked,
    }


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────

def write_thresholds(root: Path, thresholds: dict[str, Any]) -> None:
    """Write thresholds.json with both flat keys and rich metadata."""
    (root / "thresholds.json").write_text(json.dumps(thresholds, indent=2, default=str))


def read_thresholds(root: Path) -> dict[str, Any]:
    """Read thresholds.json."""
    import json as _json
    return _json.loads((root / "thresholds.json").read_text())  # type: ignore[no-any-return]


# ──────────────────────────────────────────────────────────────────────
# Default policy factory (for CLI/dashboard)
# ──────────────────────────────────────────────────────────────────────

def default_policy(
    value_quantile_low: float = QUANTILE_VALUE_LOW,
    value_quantile_high: float = QUANTILE_VALUE_HIGH,
    style_quantile_low: float = QUANTILE_STYLE_LOW,
    style_quantile_high: float = QUANTILE_STYLE_HIGH,
    min_reference_population: int = MIN_REFERENCE_POPULATION,
) -> ThresholdPolicy:
    """Create policy from CLI/dashboard parameters."""
    if not (0 < value_quantile_low < value_quantile_high < 1):
        raise ValueError("value_quantile_low must be < value_quantile_high and both in (0, 1)")
    if not (0 < style_quantile_low < style_quantile_high < 1):
        raise ValueError("style_quantile_low must be < style_quantile_high and both in (0, 1)")
    if min_reference_population < 0:
        raise ValueError("min_reference_population must be non-negative")
    # Allow 0 to mean "use default"
    effective_min_n = min_reference_population if min_reference_population > 0 else MIN_REFERENCE_POPULATION
    return ThresholdPolicy(
        value_quantile_low=value_quantile_low,
        value_quantile_high=value_quantile_high,
        style_quantile_low=style_quantile_low,
        style_quantile_high=style_quantile_high,
        min_n=effective_min_n,
    )
