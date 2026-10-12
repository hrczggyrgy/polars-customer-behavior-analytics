"""Behavioral and trajectory labeling for the retail analytics pipeline.

This module applies multi-label classifications to customer snapshots:
- Behavioral style tags (routine, focused, broad, exploration)
- Trajectory tags (slowing, expanding, overdue, reactivated, stable)
- Lifecycle stages (first_observed, early_repeat, established_repeat)
- Evidence levels (stronger, limited, insufficient)
- Value tiers (high, middle, low, insufficient_exposure, no_recent_spend)
"""

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from retail_analytics.core.constants import (
    BREADTH_CHANGE_NARROWING_MAX,
    CENSOR_DAYS,
    DOMINANT_SHARE_FOCUSED_MIN,
    ESTABLISHED_SHARE_BROAD_MIN,
    ESTABLISHED_SHARE_FOCUSED_MIN,
    ESTABLISHED_SHARE_ROUTINE_MIN,
    GAP_CV_ROUTINE_MAX,
    LATENESS_OVERDUE_MIN,
    MEDIAN_BREADTH_BROAD_MIN,
    MIN_DAYS_STYLE,
    MIN_GAP_COUNT_STRONGER,
    MIN_GAP_COUNT_STYLE,
    MIN_TENURE_COMPARABLE,
    MIN_TENURE_ONE_SHOPPING_DAY,
    MIN_TENURE_STYLE,
    MIN_TENURE_VALUE_EXPOSURE,
    NEW_SHARE_EXPLORATION_MIN,
    OVERDUE_CATEGORIES_NARROWING_MIN,
    QUANTILE_STYLE_HIGH,
    QUANTILE_STYLE_LOW,
    QUANTILE_VALUE_HIGH,
    QUANTILE_VALUE_LOW,
    REACTIVATION_EVENTS_MIN,
    WINDOW_PREVIOUS_START_OFFSET,
    WINDOW_RECENT,
    WINDOW_RECENT_30,
    WINDOW_YEAR,
)
from retail_analytics.core.pipeline.prepare import _taglist
from retail_analytics.core.thresholds import default_policy


def apply_labels(
    frame: pl.LazyFrame,
    thresholds: dict[str, Any],
    snapshot_date: date | None = None,
    reference_date: date | None = None,
) -> pl.LazyFrame:
    """Apply labels to a snapshot frame.

    Args:
        frame: LazyFrame with customer features.
        thresholds: Dict of calibrated thresholds.
        snapshot_date: Date of the snapshot being labeled.
        reference_date: Date of the reference snapshot used for calibration.

    Returns:
        Frame with behavior, trajectory, lifecycle, evidence, and value_tier labels.
    """
    # Determine if reference thresholds are available for this snapshot
    ref_available = True
    if snapshot_date is not None and reference_date is not None:
        ref_available = snapshot_date >= reference_date

    # Style eligibility: sufficient history for behavioral classification
    enough = (
        (pl.col("days_365d") >= MIN_DAYS_STYLE)
        & (pl.col("tenure") >= MIN_TENURE_STYLE)
        & (pl.col("gap_count") >= MIN_GAP_COUNT_STYLE)
    )

    # Behavioral Style Tags (multiple can apply; all matching tags collected in order)
    routine = (pl.col("gap_cv") <= GAP_CV_ROUTINE_MAX) & (
        pl.col("established_share") >= ESTABLISHED_SHARE_ROUTINE_MIN
    )
    focused = (pl.col("dominant_share") >= DOMINANT_SHARE_FOCUSED_MIN) & (
        pl.col("established_share") >= ESTABLISHED_SHARE_FOCUSED_MIN
    )
    broad = (pl.col("median_breadth") >= MEDIAN_BREADTH_BROAD_MIN) & (
        pl.col("established_share") >= ESTABLISHED_SHARE_BROAD_MIN
    )
    explorer = pl.col("new_share") >= NEW_SHARE_EXPLORATION_MIN

    style = [
        ("routine_repeat", enough & routine),
        ("focused_repeat", enough & focused),
        ("broad_repeat", enough & broad),
        ("exploration_heavy", enough & explorer),
    ]

    # Quantile-based style tags (require thresholds from reference snapshot)
    if ref_available and (
        all(
            thresholds.get(x) is not None
            for x in ["basket_low", "basket_high", "gap_low", "gap_high"]
        )
        and thresholds["basket_high"] > thresholds["basket_low"]
        and thresholds["gap_high"] > thresholds["gap_low"]
    ):
        style += [
            (
                "large_basket_episodic",
                enough
                & (pl.col("median_basket") >= thresholds["basket_high"])
                & (pl.col("median_gap") >= thresholds["gap_high"]),
            ),
            (
                "small_basket_frequent",
                enough
                & (pl.col("median_basket") <= thresholds["basket_low"])
                & (pl.col("median_gap") <= thresholds["gap_low"]),
            ),
        ]

    # Calibrated thresholds (with fallback to constants)
    if ref_available:
        lateness_threshold = thresholds.get("lateness_overdue_min") or LATENESS_OVERDUE_MIN
        freq_change_threshold = thresholds.get("frequency_change_threshold") or 0.25
        basket_change_threshold = thresholds.get("basket_change_threshold") or 0.25
    else:
        lateness_threshold = LATENESS_OVERDUE_MIN
        freq_change_threshold = 0.25
        basket_change_threshold = 0.25

    # Trajectory Tags (multiple can apply; not mutually exclusive)
    slow = pl.col("frequency_change") <= -freq_change_threshold
    small = pl.col("basket_change") <= -basket_change_threshold
    narrow = pl.col("comparable_windows") & (
        (pl.col("breadth_change") <= BREADTH_CHANGE_NARROWING_MAX)
        | (pl.col("overdue_categories") >= OVERDUE_CATEGORIES_NARROWING_MIN)
    )
    grow = (pl.col("frequency_change") >= freq_change_threshold) | (
        pl.col("basket_change") >= basket_change_threshold
    )
    late = enough & (pl.col("lateness") >= lateness_threshold)
    back = pl.col("reactivation_events") >= REACTIVATION_EVENTS_MIN
    stable = (
        enough
        & pl.col("comparable_windows")
        & pl.col("frequency_change").is_not_null()
        & ~(
            slow.fill_null(False)
            | small.fill_null(False)
            | narrow.fill_null(False)
            | grow.fill_null(False)
            | late.fill_null(False)
            | back.fill_null(False)
        )
    )

    trajectory_pairs = [
        ("slowing", slow),
        ("smaller_baskets", small),
        ("narrowing", narrow),
        ("broad_decline", slow & small),
        ("expanding", grow),
        ("overdue", late),
        ("reactivated", back),
        ("stable", stable),
    ]

    # Lifecycle Stages (mutually exclusive) - threshold-independent
    lifecycle = (
        pl.when(pl.col("days_history") == 1)
        .then(
            pl.when(pl.col("tenure") >= MIN_TENURE_ONE_SHOPPING_DAY)
            .then(pl.lit("one_shopping_day"))
            .otherwise(pl.lit("first_observed"))
        )
        .when((pl.col("days_history") >= MIN_DAYS_STYLE) & (pl.col("tenure") >= MIN_TENURE_STYLE))
        .then(pl.lit("established_repeat"))
        .otherwise(pl.lit("early_repeat"))
    )

    # Evidence Level (mutually exclusive) - threshold-independent
    evidence = (
        pl.when(enough & (pl.col("gap_count") >= MIN_GAP_COUNT_STRONGER))
        .then(pl.lit("stronger"))
        .when(enough)
        .then(pl.lit("limited"))
        .otherwise(pl.lit("insufficient"))
    )

    frame = frame.with_columns(
        enough.alias("style_eligible"),
        _taglist(style).alias("behavioral_tags"),
        _taglist(trajectory_pairs).alias("trajectory_tags"),
        lifecycle.alias("lifecycle"),
        evidence.alias("evidence"),
    )

    # Behavior label: joined tags or status
    frame = frame.with_columns(
        pl.when(~pl.col("style_eligible"))
        .then(pl.lit("insufficient_history"))
        .when(pl.col("behavioral_tags").list.len() == 0)
        .then(pl.lit("mixed_other"))
        .otherwise(pl.col("behavioral_tags").list.join("+"))
        .alias("behavior"),
        pl.when(pl.col("trajectory_tags").list.len() == 0)
        .then(pl.lit("not_classified"))
        .otherwise(pl.col("trajectory_tags").list.join("+"))
        .alias("trajectory"),
    )

    # Value Tier (mutually exclusive, requires thresholds)
    if not ref_available or thresholds.get("value_low") is None or thresholds.get("value_high") is None:
        return frame.with_columns(pl.lit("reference_unavailable").alias("value_tier"))

    return frame.with_columns(
        pl.when(pl.col("tenure") < MIN_TENURE_VALUE_EXPOSURE)
        .then(pl.lit("insufficient_exposure"))
        .when(pl.col("spend_90d").is_null() | (pl.col("spend_90d") == 0))
        .then(pl.lit("no_recent_spend"))
        .when(pl.col("spend_90d") >= thresholds["value_high"])
        .then(pl.lit("high"))
        .when(pl.col("spend_90d") <= thresholds["value_low"])
        .then(pl.lit("low"))
        .otherwise(pl.lit("middle"))
        .alias("value_tier")
    )