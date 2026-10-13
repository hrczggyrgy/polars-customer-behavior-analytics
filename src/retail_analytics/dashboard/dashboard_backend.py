"""Validated, process-isolated adapter for the existing Polars workflow."""

import argparse
import json
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from retail_analytics.core.constants import (
    MAX_CUTOFFS,
    MIN_REFERENCE_POPULATION,
    QUANTILE_STYLE_HIGH,
    QUANTILE_STYLE_LOW,
    QUANTILE_VALUE_HIGH,
    QUANTILE_VALUE_LOW,
)
from retail_analytics.core.pipeline import (
    COLS,
    apply_labels,
    generate_snapshot,
    prepare,
    sink,
)
from retail_analytics.core.pipeline.preparation import (
    normalize_strings,
    parse_numeric_columns,
    validity_mask,
)
from retail_analytics.core.pipeline.stages import (
    build_price_reference,
    select_reference_and_calibrate,
)
from retail_analytics.core.pipeline.utils import _mask_zero_spend_cols
from retail_analytics.core.thresholds import (
    compute_data_profile,
    default_policy,
    write_thresholds,
)

MAX_ROWS = 3_000_000


def cutoffs(last: date, first: date, count: int) -> list[date]:
    """Generate monthly cutoffs going backward from last date."""
    if not 1 <= count <= MAX_CUTOFFS or last < first:
        raise ValueError("Invalid cutoff range")
    result, cursor = [last], last
    for _ in range(count - 1):
        cursor = cursor.replace(day=1) - timedelta(days=1)
        if cursor < first:
            break
        result.append(cursor)
    return sorted(result)


def write_json(path: Path | str, value: Any) -> None:
    """Write JSON with default string conversion."""
    Path(path).write_text(json.dumps(value, default=str, indent=2))


def _load_normalized_source(job: dict) -> tuple[pl.LazyFrame, pl.Schema]:
    """Load source file, validate format, normalize column names.
    
    Returns:
        Tuple of (LazyFrame with normalized column names, Schema after normalization)
    """
    source = Path(job["source"])

    # Load and validate file format
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
            source,
            separator=job.get("separator", ","),
            infer_schema=False,
            encoding="utf8",
        )
    else:
        raise ValueError("Only Parquet and UTF-8 CSV are supported")

    schema = raw.collect_schema()
    names = {x: x.lstrip("\ufeff").strip().lower() for x in schema.names()}
    if len(set(names.values())) != len(names):
        raise ValueError("Column names collide after trimming/case normalization")
    raw = raw.rename(names)
    schema = raw.collect_schema()
    return raw, schema


def inspect(job: dict) -> dict:
    """Inspect source data: validate schema, count rows, compute bounds."""
    root = Path(job["workspace"])

    # Load and normalize source
    raw, schema = _load_normalized_source(job)

    # Apply column mapping if provided
    column_mapping = job.get("column_mapping", {})
    if column_mapping:
        mapped_targets = set(column_mapping.keys())
        required_targets = set(COLS)
        missing_targets = required_targets - mapped_targets
        if missing_targets:
            raise ValueError(
                f"Missing column mappings for required targets: {', '.join(sorted(missing_targets))}"
            )

        source_cols = [v for v in column_mapping.values() if v]
        if len(source_cols) != len(set(source_cols)):
            duplicates = [c for c in source_cols if source_cols.count(c) > 1]
            raise ValueError(f"Duplicate source column assignments: {', '.join(set(duplicates))}")

        for target, source_col in column_mapping.items():
            if source_col and source_col not in schema.names():
                raise ValueError(f"Mapped source column '{source_col}' not found in data")

        rename_map = {v: k for k, v in column_mapping.items() if v}
        non_target_cols = set(schema.names()) - set(rename_map.keys())
        mapped_target_names = set(rename_map.values())
        collisions = non_target_cols & mapped_target_names
        if collisions:
            raise ValueError(
                f"Column mapping creates target name collisions: {', '.join(collisions)}. "
                "Consider renaming source columns first."
            )

        raw = raw.rename(rename_map)
        schema = raw.collect_schema()

    missing = sorted(set(COLS) - set(schema.names()))
    if missing:
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

    # Parse numeric columns with dashboard-specific options (decimal comma support)
    raw = raw.select(COLS).with_columns(
        parsed.alias("date"),
        pl.col("transaction_date").alias("transaction_date_original"),
    )
    raw = parse_numeric_columns(
        raw, schema, decimal_comma=job.get("decimal_comma", False), mode=job["price_mode"]
    )
    # Normalize strings with dashboard-specific options
    raw = normalize_strings(raw, unknown_dept_category=True, null_product_description=True)

    # Compute value expression for validity check
    value = pl.col("price") * pl.col("quantity") if job["price_mode"] == "unit" else pl.col("price")
    valid = validity_mask(raw, value_expr=value)

    summary = (
        raw.select(
            pl.len().alias("rows"),
            (~valid).sum().alias("invalid_rows"),
            pl.col("date").null_count().alias("invalid_dates"),
            (pl.col("department") == "__UNKNOWN__").sum().alias("unknown_department_rows"),
            (pl.col("category") == "__UNKNOWN__").sum().alias("unknown_category_rows"),
            pl.col("price").is_null().sum().alias("null_price"),
            (pl.col("price").is_null() & pl.col("price_raw").is_not_null())
            .sum()
            .alias("malformed_price"),
            (~pl.col("price").is_finite()).sum().alias("nonfinite_price"),
            (pl.col("price") < 0).sum().alias("negative_price"),
            pl.col("quantity").is_null().sum().alias("null_quantity"),
            (pl.col("quantity").is_null() & pl.col("quantity_raw").is_not_null())
            .sum()
            .alias("malformed_quantity"),
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

    sink(raw, root / "canonical.parquet")

    if conflicts:
        raise ValueError("Customer/transaction keys span dates; resolve the transaction grain")

    return summary


def preview(job: dict) -> dict:
    """Read file schema without validation for column mapping UI."""
    root = Path(job["workspace"])

    # Load and normalize source
    raw, schema = _load_normalized_source(job)

    available = sorted(
        [c for c in schema.names() if not schema[c].is_nested() and schema[c] != pl.Binary]
    )
    write_json(
        root / "preview.json",
        {"columns": available, "dtypes": {c: str(schema[c]) for c in available}},
    )


def analyze(job: dict) -> None:
    """Run full analysis pipeline in worker process."""
    workspace, root = Path(job["workspace"]), Path(job["output"])
    metadata = json.loads((workspace / "inspection.json").read_text())
    dates = sorted({date.fromisoformat(x) for x in job["snapshots"]})
    first, last = (
        date.fromisoformat(metadata["minimum_date"]),
        date.fromisoformat(metadata["maximum_date"]),
    )
    # Use coverage_complete_through for the effective observation end when coverage is confirmed
    # This can be later than the latest transaction date (no transactions = observable zeros if source is complete)
    coverage_complete_through = job.get("coverage_complete_through")
    if coverage_complete_through and job.get("coverage_confirmed"):
        end = date.fromisoformat(coverage_complete_through)
    else:
        end = date.fromisoformat(job["observed_through"]) if job.get("coverage_confirmed") else last

    if not dates or len(dates) > MAX_CUTOFFS or dates[0] < first or dates[-1] > end:
        raise ValueError("Invalid cutoff or coverage dates")
    # end can be >= last (complete_through can be later than latest transaction)
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
        # Build price reference (dashboard doesn't sort by product_id)
        build_price_reference(root / "validated.parquet", d, root, sort_by_product=False)
        for path in paths:
            available = (
                pl.scan_parquet(path).filter(pl.col("date") <= d).select(pl.len()).collect().item()
            )
            if available:
                numeric.append(generate_snapshot(path, d, end, root, coverage, observation_start))

    if not numeric:
        raise ValueError("No customers at the selected cutoffs")

    value_quantile_low = job.get("value_quantile_low", QUANTILE_VALUE_LOW)
    value_quantile_high = job.get("value_quantile_high", QUANTILE_VALUE_HIGH)
    style_quantile_low = job.get("style_quantile_low", QUANTILE_STYLE_LOW)
    style_quantile_high = job.get("style_quantile_high", QUANTILE_STYLE_HIGH)
    min_reference_population = job.get("min_reference_population", MIN_REFERENCE_POPULATION)
    reference_date = job.get("reference_date")

    policy = default_policy(
        value_quantile_low=value_quantile_low,
        value_quantile_high=value_quantile_high,
        style_quantile_low=style_quantile_low,
        style_quantile_high=style_quantile_high,
        min_reference_population=min_reference_population,
    )

    ref_date, ref, ref_info, thresholds = select_reference_and_calibrate(
        numeric, dates, root, policy, reference_date
    )

    if ref_date is None or ref is None:
        # thresholds already written by select_reference_and_calibrate
        for path in numeric:
            snap_date = date.fromisoformat(path.parent.name)
            frame = apply_labels(
                pl.scan_parquet(path), thresholds, snapshot_date=snap_date, reference_date=None
            ).pipe(_mask_zero_spend_cols)
            sink(frame, root / "snapshots" / path.parent.name / path.name)
        write_json(
            root / "run.json",
            {
                "snapshots": [str(x) for x in dates],
                "coverage_confirmed": coverage,
                "coverage_complete_through": job.get("coverage_complete_through"),
                "coverage_scope_note": job.get("coverage_scope_note"),
                "coverage_confirmation_source": job.get("coverage_confirmation_source"),
                "latest_observed_transaction_date": str(last),
                "observed_through": str(end),
                "price_mode": job["price_mode"],
                "polars_version": pl.__version__,
                "source_sha256": job["sha256"],
                "invalid_rows_excluded": bool(job["exclude_invalid"]),
                "reference_date": None,
            },
        )
        return

    # compute_data_profile requires observation_start and end - add them to thresholds
    data_profile = compute_data_profile(ref, observation_start, end)
    thresholds["data_profile"] = data_profile
    thresholds["reference_selection"] = ref_info
    write_thresholds(root, thresholds)

    for path in numeric:
        snap_date = date.fromisoformat(path.parent.name)
        frame = apply_labels(
            pl.scan_parquet(path), thresholds, snapshot_date=snap_date, reference_date=ref_date
        ).pipe(_mask_zero_spend_cols)
        sink(frame, root / "snapshots" / path.parent.name / path.name)

    write_json(
        root / "run.json",
        {
            "snapshots": [str(x) for x in dates],
            "coverage_confirmed": coverage,
            "coverage_complete_through": job.get("coverage_complete_through"),
            "coverage_scope_note": job.get("coverage_scope_note"),
            "coverage_confirmation_source": job.get("coverage_confirmation_source"),
            "latest_observed_transaction_date": str(last),
            "observed_through": str(end),
            "price_mode": job["price_mode"],
            "polars_version": pl.__version__,
            "source_sha256": job["sha256"],
            "invalid_rows_excluded": bool(job["exclude_invalid"]),
            "reference_date": str(ref_date),
        },
    )


def main() -> None:
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
