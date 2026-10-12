"""Shared pipeline stages for price references and threshold calibration.

This module extracts stage-specific functions used by both CLI pipeline
and dashboard backend to avoid duplication while preserving differences
in orchestration, reporting, and manifest generation.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from retail_analytics.core.pipeline.utils import _sink
from retail_analytics.core.thresholds import (
    calibrate_thresholds,
    compute_data_profile,
    default_policy,
    select_reference_cutoff,
    write_thresholds,
)


def build_price_reference(
    validated_path: Path,
    cutoff_date: date,
    output_dir: Path,
    *,
    sort_by_product: bool = False,
) -> None:
    """Build price reference for a given cutoff date.

    Args:
        validated_path: Path to validated.parquet
        cutoff_date: Snapshot cutoff date
        output_dir: Output directory (will create price_references/ subdir)
        sort_by_product: If True, sort output by product_id (CLI behavior)
    """
    past = pl.scan_parquet(validated_path).filter(
        pl.col("date").is_between(cutoff_date - timedelta(days=364), cutoff_date)
    )
    ref_frame = past.group_by("product_id").agg(
        pl.col("unit_price").median().alias("reference_price"),
        pl.len().alias("reference_count"),
    )

    ref_output = output_dir / "price_references" / f"{cutoff_date}.parquet"
    sort_by = "product_id" if sort_by_product else None
    _sink(ref_frame, ref_output, sort_by=sort_by)


def build_unavailable_reference_thresholds(
    policy,
    ref_info: dict[str, Any],
) -> dict[str, Any]:
    """Construct threshold dictionary for unavailable reference case."""
    return {
        "value_low": None,
        "value_high": None,
        "basket_low": None,
        "basket_high": None,
        "gap_low": None,
        "gap_high": None,
        "reference_date": None,
        "reference_population": 0,
        "value_population": 0,
        "style_population": 0,
        "quantile_interpolation": "linear",
        "calibration_policy": asdict(policy),
        "reference_selection": ref_info,
    }


def select_reference_and_calibrate(
    numeric: list[Path],
    dates: list[date],
    root: Path,
    policy,
    reference_date: str | None = None,
) -> tuple[date | None, pl.LazyFrame | None, dict[str, Any], dict[str, Any] | None]:
    """Select reference cutoff and calibrate thresholds.

    Args:
        numeric: List of snapshot paths
        dates: List of cutoff dates
        root: Root output directory
        policy: Calibration policy
        reference_date: Optional explicit reference date (ISO format)

    Returns:
        Tuple of (ref_date, ref_frame, ref_info, thresholds_or_None)
        If thresholds is None, reference was unavailable.
    """
    if reference_date is not None:
        ref_date: date | None = date.fromisoformat(reference_date)
        ref: pl.LazyFrame | None = pl.scan_parquet(
            [str(p) for p in numeric if p.parent.name == str(ref_date)]
        )
        ref_info = {"selected": True, "reference_date": reference_date}
    else:
        ref_date, ref, ref_info = select_reference_cutoff(numeric, dates, root, policy)

    if ref_date is None or ref is None:
        # No cutoff met criteria
        thresholds = build_unavailable_reference_thresholds(policy, ref_info)
        write_thresholds(root, thresholds)
        return None, None, ref_info, thresholds

    # Calibrate thresholds from reference frame
    thresholds = calibrate_thresholds(ref, policy, ref_date)

    # Compute data profile for manifest
    data_profile = compute_data_profile(ref, None, None)  # observation_start/end handled by caller
    thresholds["data_profile"] = data_profile
    thresholds["reference_selection"] = ref_info
    thresholds["quantile_interpolation"] = "linear"

    write_thresholds(root, thresholds)
    return ref_date, ref, ref_info, thresholds


def label_snapshots(
    numeric: list[Path],
    thresholds: dict[str, Any],
    ref_date: date | None,
    output_root: Path,
    sink_func,
) -> None:
    """Label all snapshots with calibrated thresholds.

    Args:
        numeric: List of snapshot paths
        thresholds: Calibrated threshold dictionary
        ref_date: Reference date (or None for unavailable)
        output_root: Root output directory
        sink_func: Sink function (_sink for CLI, sink for dashboard)
    """
    for path in numeric:
        snap_date = date.fromisoformat(path.parent.name)
        frame = pl.scan_parquet(path)
        # Apply labels (this will be patched by caller to include _mask_zero_spend_cols)
        labeled = frame  # Placeholder - caller must apply labels
        sink_func(labeled, output_root / "snapshots" / path.parent.name / path.name)