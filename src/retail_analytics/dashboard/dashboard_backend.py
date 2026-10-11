"""Validated, process-isolated adapter for the existing Polars workflow."""

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from retail_analytics.core.constants import (
    MAX_CUTOFFS,
    MIN_REFERENCE_POPULATION,
    QUANTILE_STYLE_HIGH,
    QUANTILE_STYLE_LOW,
    QUANTILE_VALUE_HIGH,
    QUANTILE_VALUE_LOW,
)
from retail_analytics.core.retail_polars_workflow import COLS, label, prepare, sink, snapshot
from retail_analytics.core.thresholds import (
    calibrate_thresholds,
    compute_data_profile,
    default_policy,
    select_reference_cutoff,
    write_thresholds,
)

MAX_ROWS = 3_000_000


def cutoffs(last, first, count):
    if not 1 <= count <= MAX_CUTOFFS or last < first:
        raise ValueError("Invalid cutoff range")
    result, cursor = [last], last
    for _ in range(count - 1):
        cursor = cursor.replace(day=1) - timedelta(days=1)
        if cursor < first:
            break
        result.append(cursor)
    return sorted(result)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=str, indent=2))


def inspect(job):
    root, source = Path(job["workspace"]), Path(job["source"])
    if job["kind"] == "parquet":
        with source.open("rb") as handle:
            start = handle.read(4)
            handle.seek(-4, 2)
            tail = handle.read(4)
        if start != b"PAR1" or tail != b"PAR1":
            raise ValueError("Not a supported, unencrypted Parquet file")
        raw = pl.scan_parquet(source)
    elif job["kind"] == "csv":
        raw = pl.scan_csv(
            source, separator=job.get("separator", ","), infer_schema=False, encoding="utf8"
        )
    else:
        raise ValueError("Only Parquet and UTF-8 CSV are supported")
    schema = raw.collect_schema()
    names = {x: x.lstrip("\ufeff").strip().lower() for x in schema.names()}
    if len(set(names.values())) != len(names):
        raise ValueError("Column names collide after trimming/case normalization")
    raw = raw.rename(names)
    schema = raw.collect_schema()

    # Apply column mapping if provided
    column_mapping = job.get("column_mapping", {})
    if column_mapping:
        # Validate mapping: all target columns must be mapped, source columns must exist
        # Check that all required COLS have a mapping
        mapped_targets = set(column_mapping.keys())
        required_targets = set(COLS)
        missing_targets = required_targets - mapped_targets
        if missing_targets:
            raise ValueError(
                f"Missing column mappings for required targets: {', '.join(sorted(missing_targets))}"
            )

        # Check that all source columns are distinct (no duplicate assignments)
        source_cols = [v for v in column_mapping.values() if v]
        if len(source_cols) != len(set(source_cols)):
            duplicates = [c for c in source_cols if source_cols.count(c) > 1]
            raise ValueError(f"Duplicate source column assignments: {', '.join(set(duplicates))}")

        # Check that all source columns exist in schema
        for target, source_col in column_mapping.items():
            if source_col and source_col not in schema.names():
                raise ValueError(f"Mapped source column '{source_col}' not found in data")

        # Check for target-name collisions (mapping creates duplicate target names)
        # After rename, check if any non-mapped columns would collide with mapped targets
        rename_map = {v: k for k, v in column_mapping.items() if v}
        # Check if any non-target columns would be renamed to existing target names
        non_target_cols = set(schema.names()) - set(rename_map.keys())
        mapped_target_names = set(rename_map.values())
        collisions = non_target_cols & mapped_target_names
        if collisions:
            raise ValueError(
                f"Column mapping creates target name collisions: {', '.join(collisions)}. Consider renaming source columns first."
            )

        # Apply rename
        raw = raw.rename(rename_map)
        schema = raw.collect_schema()

    missing = sorted(set(COLS) - set(schema.names()))
    if missing:
        # Return available columns for UI mapping
        available = sorted(
            [c for c in schema.names() if not schema[c].is_nested() and schema[c] != pl.Binary]
        )
        raise ValueError(
            f"Missing columns: {', '.join(missing)}. Available: {', '.join(available)}"
        )
    if any(schema[x].is_nested() or schema[x] == pl.Binary for x in COLS):
        raise ValueError("Required columns must be scalar, not nested or binary")
    count = raw.select(pl.len()).collect(engine="streaming").item()
    if not 0 < count <= MAX_ROWS:
        raise ValueError(f"Expected 1..{MAX_ROWS:,} rows; got {count:,}")
    dtype = schema["transaction_date"]
    timezone = job.get("timezone")
    if dtype == pl.String:
        text = pl.col("transaction_date").str.strip_chars()
        fmt = job.get("date_format")
        parsed = (text if fmt else text.str.slice(0, 10)).str.to_date(
            fmt or "%Y-%m-%d", strict=False
        )
    elif dtype == pl.Date:
        parsed = pl.col("transaction_date").cast(pl.Date)
    elif isinstance(dtype, pl.Datetime):
        # Preserve timezone through precision conversion
        dt_col = pl.col("transaction_date")
        if dtype.time_unit != "us":
            dt_col = dt_col.cast(pl.Datetime("us", time_zone=dtype.time_zone))
        if dtype.time_zone is not None:
            if timezone is None:
                raise ValueError("Timezone-aware Datetime input requires timezone parameter")
            dt_col = dt_col.dt.convert_time_zone(timezone)
        parsed = dt_col.cast(pl.Date)
    else:
        raise ValueError("Numeric epoch dates need conversion before upload")
    numeric = []
    for x in ["price", "quantity"]:
        value = pl.col(x).cast(pl.String).str.strip_chars()
        if job.get("decimal_comma"):
            value = value.str.replace_all(",", ".", literal=True)
        # Preserve raw string before casting to distinguish missing vs malformed
        # Apply same normalization as core: empty string -> null
        raw_col = value.replace("", None).alias(f"{x}_raw")
        casted = value.cast(pl.Float64, strict=False).alias(x)
        numeric.append(raw_col)
        numeric.append(casted)
    # Preserve original transaction_date for provenance
    raw = raw.select(COLS).with_columns(
        parsed.alias("date"),
        pl.col("transaction_date").alias("transaction_date_original"),
        *numeric,
        *[
            pl.col(x).cast(pl.String).str.strip_chars().replace("", None)
            for x in ["customer_id", "transaction_id", "product_id"]
        ],
        *[
            pl.col(x).cast(pl.String).str.strip_chars().replace("", None).fill_null("__UNKNOWN__")
            for x in ["department", "category"]
        ],
        pl.col("product_description").cast(pl.String).fill_null(""),
    )
    value = pl.col("price") * pl.col("quantity") if job["price_mode"] == "unit" else pl.col("price")
    valid = (
        pl.all_horizontal(
            [
                pl.col(x).is_not_null()
                for x in ["customer_id", "transaction_id", "product_id", "date"]
            ]
        )
        & pl.col("price").is_finite()
        & (pl.col("price") >= 0)
        & pl.col("quantity").is_finite()
        & (pl.col("quantity") > 0)
        & value.is_finite()
    ).fill_null(False)
    summary = (
        raw.select(
            pl.len().alias("rows"),
            (~valid).sum().alias("invalid_rows"),
            pl.col("date").null_count().alias("invalid_dates"),
            (pl.col("department") == "__UNKNOWN__").sum().alias("unknown_department_rows"),
            (pl.col("category") == "__UNKNOWN__").sum().alias("unknown_category_rows"),
            # Distinguish missing source from malformed token
            pl.col("price").is_null().sum().alias("null_price"),
            (pl.col("price").is_null() & pl.col("price_raw").is_not_null()).sum().alias("malformed_price"),
            (~pl.col("price").is_finite()).sum().alias("nonfinite_price"),
            (pl.col("price") < 0).sum().alias("negative_price"),
            pl.col("quantity").is_null().sum().alias("null_quantity"),
            (pl.col("quantity").is_null() & pl.col("quantity_raw").is_not_null()).sum().alias("malformed_quantity"),
            (~pl.col("quantity").is_finite()).sum().alias("nonfinite_quantity"),
            (pl.col("quantity") <= 0).sum().alias("nonpositive_quantity"),
            (~value.is_finite()).sum().alias("nonfinite_value"),
            pl.col("price_raw").is_null().sum().alias("missing_price_raw"),
            pl.col("quantity_raw").is_null().sum().alias("missing_quantity_raw"),
        )
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    bounds = (
        raw.filter(valid)
        .select(
            pl.col("date").min().alias("minimum_date"),
            pl.col("date").max().alias("maximum_date"),
            pl.col("customer_id").n_unique().alias("valid_customers"),
        )
        .collect()
        .to_dicts()[0]
    )
    summary.update(bounds)
    summary["ignored_columns"] = sorted(set(schema.names()) - set(COLS))
    summary["source_schema"] = {x: str(schema[x]) for x in COLS}
    write_json(root / "inspection.json", summary)
    if bounds["minimum_date"] is None:
        raise ValueError("No valid purchases after normalization")
    conflicts = (
        raw.filter(valid)
        .group_by(["customer_id", "transaction_id"])
        .agg(pl.col("date").n_unique().alias("dates"))
        .filter(pl.col("dates") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    summary["basket_key_conflicts"] = conflicts
    write_json(root / "inspection.json", summary)
    # Keep raw columns for rejection provenance
    sink(raw, root / "canonical.parquet")
    if conflicts:
        raise ValueError("Customer/transaction keys span dates; resolve the transaction grain")
    return summary


def preview(job):
    """Read file schema without validation for column mapping UI."""
    root, source = Path(job["workspace"]), Path(job["source"])
    if job["kind"] == "parquet":
        with source.open("rb") as handle:
            start = handle.read(4)
            handle.seek(-4, 2)
            tail = handle.read(4)
        if start != b"PAR1" or tail != b"PAR1":
            raise ValueError("Not a supported, unencrypted Parquet file")
        raw = pl.scan_parquet(source)
    elif job["kind"] == "csv":
        raw = pl.scan_csv(
            source, separator=job.get("separator", ","), infer_schema=False, encoding="utf8"
        )
    else:
        raise ValueError("Only Parquet and UTF-8 CSV are supported")
    schema = raw.collect_schema()
    names = {x: x.lstrip("\ufeff").strip().lower() for x in schema.names()}
    if len(set(names.values())) != len(names):
        raise ValueError("Column names collide after trimming/case normalization")
    # Apply the same normalization to get a consistent schema
    raw = raw.rename(names)
    schema = raw.collect_schema()

    # Return available scalar columns
    available = sorted(
        [c for c in schema.names() if not schema[c].is_nested() and schema[c] != pl.Binary]
    )
    write_json(
        root / "preview.json",
        {"columns": available, "dtypes": {c: str(schema[c]) for c in available}},
    )


def analyze(job):
    workspace, root = Path(job["workspace"]), Path(job["output"])
    metadata = json.loads((workspace / "inspection.json").read_text())
    dates = sorted({date.fromisoformat(x) for x in job["snapshots"]})
    first, last = (
        date.fromisoformat(metadata["minimum_date"]),
        date.fromisoformat(metadata["maximum_date"]),
    )
    end = date.fromisoformat(job["observed_through"]) if job.get("coverage_confirmed") else last
    if not dates or len(dates) > MAX_CUTOFFS or dates[0] < first or dates[-1] > end or end < last:
        raise ValueError("Invalid cutoff or coverage dates")
    if not 1 <= int(job["buckets"]) <= 32:
        raise ValueError("Bucket count must be 1..32")
    required_free = max(
        256 * 1024 * 1024, (workspace / "canonical.parquet").stat().st_size * (3 + 3 * len(dates))
    )
    if shutil.disk_usage(workspace).free < required_free:
        raise ValueError("Insufficient temporary disk space; reduce cutoffs or upload size")
    root.mkdir(parents=True, exist_ok=False)
    coverage = bool(job.get("coverage_confirmed"))
    timezone = job.get("timezone")
    date_format = job.get("date_format")
    paths, end, observation_start = prepare(
        str(workspace / "canonical.parquet"),
        root,
        job["price_mode"],
        job["buckets"],
        job["exclude_invalid"],
        date_format,
        str(end),
        coverage,
        False,  # drop_exact_duplicates
        timezone,
    )
    numeric = []
    for d in dates:
        past = pl.scan_parquet(root / "validated.parquet").filter(
            pl.col("date").is_between(d - timedelta(days=364), d)
        )
        sink(
            past.group_by("product_id").agg(
                pl.col("unit_price").median().alias("reference_price"),
                pl.len().alias("reference_count"),
            ),
            root / "price_references" / f"{d}.parquet",
        )
        for path in paths:
            available = (
                pl.scan_parquet(path).filter(pl.col("date") <= d).select(pl.len()).collect().item()
            )
            if available:
                numeric.append(snapshot(path, d, end, root, coverage, observation_start))
    if not numeric:
        raise ValueError("No customers at the selected cutoffs")

    # Get threshold parameters from job or use defaults
    value_quantile_low = job.get("value_quantile_low", QUANTILE_VALUE_LOW)
    value_quantile_high = job.get("value_quantile_high", QUANTILE_VALUE_HIGH)
    style_quantile_low = job.get("style_quantile_low", QUANTILE_STYLE_LOW)
    style_quantile_high = job.get("style_quantile_high", QUANTILE_STYLE_HIGH)
    min_reference_population = job.get("min_reference_population", MIN_REFERENCE_POPULATION)
    reference_date = job.get("reference_date")

    # Build policy from job parameters
    policy = default_policy(
        value_quantile_low=value_quantile_low,
        value_quantile_high=value_quantile_high,
        style_quantile_low=style_quantile_low,
        style_quantile_high=style_quantile_high,
        min_reference_population=min_reference_population,
    )

    # Select reference cutoff: earliest with sufficient & stable population
    if reference_date is not None:
        ref_date = date.fromisoformat(reference_date)
        ref = pl.scan_parquet([str(p) for p in numeric if p.parent.name == str(ref_date)])
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
            frame = label(pl.scan_parquet(path), thresholds, snapshot_date=snap_date, reference_date=None).with_columns(
                pl.when(pl.col("spend_365d") > 0)
                .then(pl.col("category_hhi"))
                .otherwise(None)
                .alias("category_hhi"),
                pl.when(pl.col("spend_365d") > 0)
                .then(pl.col("category_entropy"))
                .otherwise(None)
                .alias("category_entropy"),
            )
            sink(frame, root / "snapshots" / path.parent.name / path.name)
        write_json(
            root / "run.json",
            {
                "snapshots": [str(x) for x in dates],
                "coverage_confirmed": coverage,
                "observed_through": str(end),
                "price_mode": job["price_mode"],
                "polars_version": pl.__version__,
                "source_sha256": job["sha256"],
                "invalid_rows_excluded": bool(job["exclude_invalid"]),
                "reference_date": None,
            },
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
    for path in numeric:
        snap_date = date.fromisoformat(path.parent.name)
        frame = label(pl.scan_parquet(path), thresholds, snapshot_date=snap_date, reference_date=ref_date).with_columns(
            pl.when(pl.col("spend_365d") > 0)
            .then(pl.col("category_hhi"))
            .otherwise(None)
            .alias("category_hhi"),
            pl.when(pl.col("spend_365d") > 0)
            .then(pl.col("category_entropy"))
            .otherwise(None)
            .alias("category_entropy"),
        )
        sink(frame, root / "snapshots" / path.parent.name / path.name)
    write_json(
        root / "run.json",
        {
            "snapshots": [str(x) for x in dates],
            "coverage_confirmed": coverage,
            "observed_through": str(end),
            "price_mode": job["price_mode"],
            "polars_version": pl.__version__,
            "source_sha256": job["sha256"],
            "invalid_rows_excluded": bool(job["exclude_invalid"]),
            "reference_date": str(ref_date),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    result = Path(job["workspace"]) / "worker_status.json"
    try:
        if job["action"] == "inspect":
            inspect(job)
        elif job["action"] == "analyze":
            analyze(job)
        elif job["action"] == "preview":
            preview(job)
        else:
            raise ValueError(f"Unknown action: {job['action']}")
        write_json(result, {"ok": True})
    except Exception as exc:
        write_json(
            result, {"ok": False, "error_type": type(exc).__name__, "message": str(exc)[:1500]}
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
