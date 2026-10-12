"""Main pipeline orchestration and CLI entry point.

This module coordinates the complete analytics pipeline:
1. Data preparation and validation
2. Point-in-time snapshot generation
3. Threshold calibration from reference population
4. Labeling with behavioral and trajectory tags
5. Report generation (segments, cohorts, transitions)
"""

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from retail_analytics.core.constants import (
    MAX_BUCKETS,
    MAX_CUTOFFS,
    MIN_REFERENCE_POPULATION,
    QUANTILE_STYLE_HIGH,
    QUANTILE_STYLE_LOW,
    QUANTILE_VALUE_HIGH,
    QUANTILE_VALUE_LOW,
)
from retail_analytics.core.pipeline.prepare import prepare
from retail_analytics.core.pipeline.snapshot import generate_snapshot
from retail_analytics.core.pipeline.labeling import apply_labels
from retail_analytics.core.pipeline.reporting import generate_reports
from retail_analytics.core.thresholds import (
    calibrate_thresholds,
    compute_data_profile,
    default_policy,
    select_reference_cutoff,
    write_thresholds,
)

VERSION = "reconstructed-1.3"
PriceMode = str  # "unit" or "line_total"


def run_pipeline(
    source: str,
    output: str,
    price_mode: PriceMode,
    snapshots: list[str],
    buckets: int = 8,
    exclude_invalid: bool = False,
    date_format: str | None = None,
    observed_through: str | None = None,
    coverage_confirmed: bool = False,
    drop_exact_duplicates: bool = False,
    overwrite: bool = False,
    timezone: str | None = None,
    value_quantile_low: float = QUANTILE_VALUE_LOW,
    value_quantile_high: float = QUANTILE_VALUE_HIGH,
    style_quantile_low: float = QUANTILE_STYLE_LOW,
    style_quantile_high: float = QUANTILE_STYLE_HIGH,
    min_reference_population: int = MIN_REFERENCE_POPULATION,
    reference_date: str | None = None,
) -> None:
    """Run the complete analytics pipeline.

    Args:
        source: Path to input Parquet file.
        output: Output directory.
        price_mode: "unit" (price * quantity) or "line_total" (price is line total).
        snapshots: List of cutoff dates (ISO format).
        buckets: Number of hash partitions (1-32).
        exclude_invalid: Exclude invalid rows instead of raising.
        date_format: Date format string for parsing.
        observed_through: Effective observation end date.
        coverage_confirmed: Whether future follow-up data is complete.
        drop_exact_duplicates: Drop exact duplicate rows.
        overwrite: Overwrite existing output directory.
        timezone: Timezone for datetime conversion.
        value_quantile_low: Low quantile for value tier.
        value_quantile_high: High quantile for value tier.
        style_quantile_low: Low quantile for style thresholds.
        style_quantile_high: High quantile for style thresholds.
        min_reference_population: Minimum population for threshold calibration.
        reference_date: Explicit reference date for threshold calibration.
    """
    if buckets < 1 or buckets > MAX_BUCKETS or price_mode not in ["unit", "line_total"]:
        raise ValueError("Invalid buckets or price_mode")

    root = Path(output)
    if root.exists() and any(root.iterdir()):
        if overwrite:
            shutil.rmtree(root)
        else:
            raise ValueError("Output directory must be empty (use --overwrite to replace)")

    root.mkdir(parents=True, exist_ok=True)
    dates = sorted({date.fromisoformat(x) for x in snapshots})
    if not dates:
        raise ValueError("At least one cutoff required")
    if len(dates) > MAX_CUTOFFS:
        raise ValueError(f"Maximum {MAX_CUTOFFS} cutoffs allowed")

    # Write configuration
    (root / "configuration.json").write_text(
        json.dumps(
            dict(
                source=source,
                price_mode=price_mode,
                snapshots=[str(x) for x in dates],
                buckets=buckets,
                exclude_invalid=exclude_invalid,
                date_format=date_format,
                observed_through=observed_through,
                coverage_confirmed=bool(coverage_confirmed),
                timezone=timezone,
                value_quantile_low=value_quantile_low,
                value_quantile_high=value_quantile_high,
                style_quantile_low=style_quantile_low,
                style_quantile_high=style_quantile_high,
                min_reference_population=min_reference_population,
                reference_date=reference_date,
                version=VERSION,
            ),
            indent=2,
        )
    )

    # Step 1: Prepare validated data
    paths, end, observation_start = prepare(
        source,
        root,
        price_mode,
        buckets,
        exclude_invalid,
        date_format,
        observed_through,
        coverage_confirmed,
        drop_exact_duplicates,
        timezone,
    )

    if dates[-1] > end:
        raise ValueError("Cutoff exceeds observed_through")

    # Step 2: Generate snapshots for each cutoff
    numeric = []
    for d in dates:
        # Build price reference for this cutoff
        past = pl.scan_parquet(root / "validated.parquet").filter(
            pl.col("date").is_between(d - timedelta(days=364), d)
        )
        _sink(
            past.group_by("product_id").agg(
                pl.col("unit_price").median().alias("reference_price"),
                pl.len().alias("reference_count"),
            ),
            root / "price_references" / f"{d}.parquet",
            sort_by="product_id",
        )
        for path in paths:
            if pl.scan_parquet(path).filter(pl.col("date") <= d).select(pl.len()).collect().item() > 0:
                result = generate_snapshot(path, d, end, root, coverage_confirmed, observation_start)
                if result is not None:
                    numeric.append(result)

    if not numeric:
        # No customers at any cutoff; write empty thresholds and skip labeling/reports
        thresholds = {
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
            "calibration_policy": asdict(default_policy()),
        }
        write_thresholds(root, thresholds)
        return

    # Step 3: Select reference cutoff and calibrate thresholds
    policy = default_policy(
        value_quantile_low=value_quantile_low,
        value_quantile_high=value_quantile_high,
        style_quantile_low=style_quantile_low,
        style_quantile_high=style_quantile_high,
        min_reference_population=min_reference_population,
    )

    # Select reference cutoff: earliest with sufficient & stable population
    if reference_date is not None:
        ref_date: date | None = date.fromisoformat(reference_date)
        ref: pl.LazyFrame | None = pl.scan_parquet([str(p) for p in numeric if p.parent.name == str(ref_date)])
        ref_info = {"selected": True, "reference_date": reference_date}
    else:
        ref_date, ref, ref_info = select_reference_cutoff(numeric, dates, root, policy)

    if ref_date is None or ref is None:
        # No cutoff met criteria - write thresholds with reference_unavailable status
        thresholds = {
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
            "calibration_policy": asdict(policy),
            "reference_selection": ref_info,
        }
        write_thresholds(root, thresholds)
        # Label all snapshots with reference_unavailable
        for path in numeric:
            snap_date = date.fromisoformat(path.parent.name)
            frame = apply_labels(
                pl.scan_parquet(path),
                thresholds,
                snapshot_date=snap_date,
                reference_date=None,
            ).with_columns(
                pl.when(pl.col("spend_365d") > 0)
                .then(pl.col("category_hhi"))
                .otherwise(None)
                .alias("category_hhi"),
                pl.when(pl.col("spend_365d") > 0)
                .then(pl.col("category_entropy"))
                .otherwise(None)
                .alias("category_entropy"),
            )
            _sink(frame, root / "snapshots" / path.parent.name / path.name)

        # Generate reports even without reference calibration
        generate_reports(root, buckets)

        (root / "run.json").write_text(
            json.dumps(
                {
                    "snapshots": [str(x) for x in dates],
                    "coverage_confirmed": coverage_confirmed,
                    "observed_through": str(end),
                    "price_mode": price_mode,
                    "polars_version": pl.__version__,
                    "reference_date": None,
                },
                indent=2,
                default=str,
            )
        )
        return

    # Calibrate thresholds from reference frame
    thresholds = calibrate_thresholds(ref, policy, ref_date)

    # Compute data profile for manifest
    data_profile = compute_data_profile(ref, observation_start, end)
    thresholds["data_profile"] = data_profile

    # Add reference selection info
    thresholds["reference_selection"] = ref_info

    write_thresholds(root, thresholds)

    # Step 4: Label all snapshots
    for path in numeric:
        snap_date = date.fromisoformat(path.parent.name)
        frame = apply_labels(
            pl.scan_parquet(path), thresholds, snapshot_date=snap_date, reference_date=ref_date
        ).with_columns(
            pl.when(pl.col("spend_365d") > 0)
            .then(pl.col("category_hhi"))
            .otherwise(None)
            .alias("category_hhi"),
            pl.when(pl.col("spend_365d") > 0)
            .then(pl.col("category_entropy"))
            .otherwise(None)
            .alias("category_entropy"),
        )
        _sink(frame, root / "snapshots" / path.parent.name / path.name)

    # Step 5: Generate reports
    generate_reports(root, buckets)

    # Write run manifest
    (root / "run.json").write_text(
        json.dumps(
            {
                "snapshots": [str(x) for x in dates],
                "coverage_confirmed": coverage_confirmed,
                "observed_through": str(end),
                "price_mode": price_mode,
                "polars_version": pl.__version__,
                "reference_date": str(ref_date),
            },
            indent=2,
            default=str,
        )
    )


def _sink(
    frame: pl.LazyFrame, path: Path, sort_by: list[str] | str | None = None
) -> None:
    """Write a LazyFrame to Parquet with ZSTD compression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sort_by is not None:
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        frame = frame.sort(sort_by)
    frame.collect(engine="streaming").write_parquet(path, compression="zstd")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Retail customer behavior analytics pipeline")
    parser.add_argument("--input", required=True, help="Source Parquet file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--price-mode", required=True, choices=["unit", "line_total"], help="Price interpretation"
    )
    parser.add_argument("--snapshots", required=True, nargs="+", help="Cutoff dates (YYYY-MM-DD)")
    parser.add_argument("--buckets", type=int, default=8, help="Hash partitions (1-32)")
    parser.add_argument("--exclude-invalid", action="store_true", help="Exclude invalid rows")
    parser.add_argument("--date-format", help="Date format (e.g., %%Y-%%m-%%d)")
    parser.add_argument("--observed-through", help="Effective end date (YYYY-MM-DD)")
    parser.add_argument("--coverage-confirmed", action="store_true", help="Future data complete")
    parser.add_argument("--drop-exact-duplicates", action="store_true", help="Drop exact duplicates")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory")
    parser.add_argument("--timezone", help="Timezone for datetime conversion")
    parser.add_argument("--value-quantile-low", type=float, default=QUANTILE_VALUE_LOW)
    parser.add_argument("--value-quantile-high", type=float, default=QUANTILE_VALUE_HIGH)
    parser.add_argument("--style-quantile-low", type=float, default=QUANTILE_STYLE_LOW)
    parser.add_argument("--style-quantile-high", type=float, default=QUANTILE_STYLE_HIGH)
    parser.add_argument("--min-reference-population", type=int, default=MIN_REFERENCE_POPULATION)
    parser.add_argument("--reference-date", help="Explicit reference date (YYYY-MM-DD)")

    args = parser.parse_args()
    run_pipeline(
        args.input,
        args.output,
        args.price_mode,
        args.snapshots,
        args.buckets,
        args.exclude_invalid,
        args.date_format,
        args.observed_through,
        args.coverage_confirmed,
        args.drop_exact_duplicates,
        args.overwrite,
        args.timezone,
        args.value_quantile_low,
        args.value_quantile_high,
        args.style_quantile_low,
        args.style_quantile_high,
        args.min_reference_population,
        args.reference_date,
    )


if __name__ == "__main__":
    main()