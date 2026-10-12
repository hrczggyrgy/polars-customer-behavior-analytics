"""Point-in-time snapshot generation for the retail analytics pipeline.

This module generates customer feature snapshots at specific cutoff dates,
ensuring no temporal leakage and proper feature computation.
"""

from datetime import date
from pathlib import Path

import polars as pl

from retail_analytics.core.pipeline.features import (
    _compute_baskets,
    add_derived_features,
    add_future_outcomes,
    compute_base_features,
    compute_breadth_last6days,
    compute_category_aggregations,
    compute_category_features,
    compute_day_features,
    compute_habit_features,
    compute_price_features,
    compute_product_features,
    compute_quantity_features,
    compute_rhythm_features,
    compute_weekday_entropy,
    merge_features,
)
from retail_analytics.core.pipeline.utils import C, G, K, P, _sink


def generate_snapshot(
    path: Path,
    d: date,
    end: date,
    root: Path,
    coverage_confirmed: bool = False,
    observation_start: date | None = None,
) -> Path | None:
    """Generate a point-in-time snapshot of customer features.

    This function computes all customer behavioral features as of cutoff date `d`,
    using only transactions on or before `d`. Future outcomes are added only
    if `coverage_confirmed` is True and the follow-up window is fully observed.

    Args:
        path: Path to partition parquet file.
        d: Snapshot cutoff date.
        end: Observation end date.
        root: Output root directory.
        coverage_confirmed: Whether future follow-up is available.
        observation_start: First observed transaction date.

    Returns:
        Path to numeric output file, or None if no customers.
    """
    out = root / "events" / str(d) / path.stem
    target = root / "numeric" / str(d) / path.name
    line = pl.scan_parquet(path).filter(pl.col("date") <= d)

    # Temporal boundary assertion - prevent lookahead bias
    # Note: This check is performed after filtering in prepare.py, but validates
    # that no upstream leakage occurred. Could be combined with other scalar collections
    # in a benchmarked optimization pass (see Phase 9.2 plan).
    max_date_check = line.select(pl.col("date").max()).collect().item()
    if max_date_check is not None and max_date_check > d:
        raise AssertionError(
            f"Temporal leakage: descriptive data contains date {max_date_check} > cutoff {d}"
        )

    # Baskets
    basket = _compute_baskets(line, out)

    # Revenue reconciliation
    # Note: Separate scalar collection. Could be combined with other checks
    # in a benchmarked optimization pass (see Phase 9.2 plan).
    totals = line.select(pl.col("value").sum()).collect().item() or 0
    if abs((basket.select(pl.col("spend").sum()).collect().item() or 0) - totals) > 1e-7 * max(
        1, abs(totals)
    ):
        raise AssertionError("Revenue reconciliation failed")

    # Base features
    base, recent, _, year, s30, s90, s365 = compute_base_features(
        basket, d, observation_start
    )

    # Check if base has any customers
    # Note: Separate scalar collection. Could be combined with other checks
    # in a benchmarked optimization pass (see Phase 9.2 plan).
    base_count = base.select(pl.len()).collect()
    if base_count.height == 0 or base_count.item() == 0:
        return None

    # Day features
    day = compute_day_features(line, out, C)

    # Rhythm features
    usable = year & (pl.col("prev_date") >= s365)
    rhythm = compute_rhythm_features(day, C, year, usable, recent, s30)

    # Weekday entropy
    weekday_stats = compute_weekday_entropy(day, C, year)

    # Breadth last 6 days
    breadth = compute_breadth_last6days(line, day, C)

    # Product features
    trials = compute_product_features(line, out, P, year, d)

    # Habit features
    pd = pl.scan_parquet(out / "product_days.parquet")
    habits = compute_habit_features(line, pd, K, P, year, C)

    # Quantity features
    quantity = compute_quantity_features(pd, C, year)

    # Category features
    ch = compute_category_features(line, G, C, year, d, out)

    categories, dominant = compute_category_aggregations(ch, C)

    # Price features
    prices = compute_price_features(line, root, d, C, year)

    # Merge all features
    frame = merge_features(
        base,
        rhythm,
        weekday_stats,
        breadth,
        trials,
        habits,
        quantity,
        categories,
        dominant,
        prices,
    )

    # Post-join integrity check
    # Note: Validates 1:1 cardinality after feature merges. Could be combined
    # with other scalar collections in a benchmarked optimization pass (see Phase 9.2 plan).
    dup_check = frame.group_by(C).len().filter(pl.col("len") > 1)
    has_dups = dup_check.select(pl.len()).collect().item()
    if has_dups:
        raise ValueError("Duplicate customer_ids after feature merge")

    # Derived features
    frame = add_derived_features(frame, d)

    # Future outcomes
    frame = add_future_outcomes(frame, path, d, end, coverage_confirmed)

    _sink(frame, target)
    return target
