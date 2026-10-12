"""Data preparation, validation, and partitioning for the retail analytics pipeline.

This module handles:
- Input schema validation
- Date parsing and normalization
- String column normalization
- Price/quantity parsing and value computation
- Row-level validation with rejection reasons
- Quality audit metrics
- Hash-based partitioning for parallel processing
"""

import json
import argparse
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import polars as pl

from retail_analytics.core.constants import (
    BREADTH_CHANGE_NARROWING_MAX,
    CENSOR_DAYS,
    DOMINANT_SHARE_FOCUSED_MIN,
    ESTABLISHED_SHARE_BROAD_MIN,
    ESTABLISHED_SHARE_FOCUSED_MIN,
    ESTABLISHED_SHARE_ROUTINE_MIN,
    GAP_CV_ROUTINE_MAX,
    HASH_SEED_BUCKETING,
    LATENESS_OVERDUE_MIN,
    MEDIAN_BREADTH_BROAD_MIN,
    MIN_DAYS_STYLE,
    MIN_GAP_COUNT_STRONGER,
    MIN_GAP_COUNT_STYLE,
    MIN_REFERENCE_POPULATION,
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
from retail_analytics.core.thresholds import (
    calibrate_thresholds,
    compute_data_profile,
    compute_label_shares,
    default_policy,
    select_reference_cutoff,
    write_thresholds,
)
from retail_analytics.core.stats import ratio as safe_ratio

# Column definitions
REQUIRED_COLS = COLS = [
    "customer_id",
    "transaction_date",
    "transaction_id",
    "product_id",
    "product_description",
    "department",
    "category",
    "price",
    "quantity",
]

# Key aliases for readability
C = "customer_id"
K = [C, "transaction_id", "date"]
P = [C, "product_id"]
G = [C, "department", "category"]

VERSION = "reconstructed-1.3"

# Type aliases
PriceMode = Literal["unit", "line_total"]


def _safe_ratio(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Safe division: returns null when denominator is null or <= 0."""
    return pl.when(b.is_not_null() & (b > 0)).then(a / b).otherwise(None)


# Backward compatibility alias
ratio = _safe_ratio


def _sink(
    frame: pl.LazyFrame, path: Path, sort_by: list[str] | str | None = None
) -> None:
    """Write a LazyFrame to Parquet with ZSTD compression (internal)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sort_by is not None:
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        frame = frame.sort(sort_by)
    # Use non-streaming engine for reliability; streaming can skip empty frames
    frame.collect(engine="streaming").write_parquet(path, compression="zstd")


def sink(frame: pl.LazyFrame, path: Path, sort_by: list[str] | str | None = None) -> None:
    """Write a LazyFrame to Parquet with ZSTD compression (public API)."""
    _sink(frame, path, sort_by)


def _taglist(pairs: list[tuple[str, pl.Expr]]) -> pl.Expr:
    """Create a list of tags from (name, boolean_expression) pairs.

    ALL matching tags are collected in order (not first-match exclusive).
    The resulting list preserves the order of the input pairs.
    """
    return pl.concat_list(
        [
            pl.when(test.fill_null(False))
            .then(pl.lit(name))
            .otherwise(pl.lit(None, dtype=pl.String))
            for name, test in pairs
        ]
    ).list.drop_nulls()


def _rejection_reasons(raw: pl.LazyFrame) -> pl.Expr:
    """Compute per-row rejection reason codes.

    Returns the first matching reason in priority order.
    """
    reasons = []

    # Required string fields non-null and non-empty after trim
    for col in [C, "transaction_id", "product_id"]:
        reasons.append(
            pl.when(pl.col(col).is_null() | (pl.col(col).cast(pl.String).str.strip_chars() == ""))
            .then(pl.lit(f"missing_{col}"))
            .otherwise(pl.lit(None))
        )

    # Date parse failure (date becomes null)
    reasons.append(
        pl.when(pl.col("date").is_null()).then(pl.lit("invalid_date")).otherwise(pl.lit(None))
    )

    # Price checks - distinguish missing source (null after strip) from malformed token (null after cast)
    reasons.append(
        pl.when(pl.col("price").is_null() & pl.col("price_raw").is_not_null())
        .then(pl.lit("malformed_price"))
        .otherwise(pl.lit(None))
    )
    reasons.append(
        pl.when(pl.col("price").is_null() & pl.col("price_raw").is_null())
        .then(pl.lit("null_price"))
        .otherwise(pl.lit(None))
    )
    reasons.append(
        pl.when(~pl.col("price").is_finite())
        .then(pl.lit("nonfinite_price"))
        .otherwise(pl.lit(None))
    )
    reasons.append(
        pl.when(pl.col("price") < 0).then(pl.lit("negative_price")).otherwise(pl.lit(None))
    )

    # Quantity checks - distinguish missing source from malformed token
    reasons.append(
        pl.when(pl.col("quantity").is_null() & pl.col("quantity_raw").is_not_null())
        .then(pl.lit("malformed_quantity"))
        .otherwise(pl.lit(None))
    )
    reasons.append(
        pl.when(pl.col("quantity").is_null() & pl.col("quantity_raw").is_null())
        .then(pl.lit("null_quantity"))
        .otherwise(pl.lit(None))
    )
    reasons.append(
        pl.when(~pl.col("quantity").is_finite())
        .then(pl.lit("nonfinite_quantity"))
        .otherwise(pl.lit(None))
    )
    reasons.append(
        pl.when(pl.col("quantity") <= 0)
        .then(pl.lit("nonpositive_quantity"))
        .otherwise(pl.lit(None))
    )

    # Value computation
    reasons.append(
        pl.when(~pl.col("value").is_finite())
        .then(pl.lit("nonfinite_value"))
        .otherwise(pl.lit(None))
    )

    # Combine: first non-null reason wins (priority order above)
    combined = pl.coalesce(*reasons)
    return combined.alias("rejection_reason")


def _parse_dates(
    src: pl.LazyFrame,
    fmt: str | None,
    timezone: str | None,
) -> pl.Expr:
    """Parse transaction_date column to Date based on its dtype."""
    dtype = src.collect_schema()["transaction_date"]

    if dtype == pl.String:
        text = pl.col("transaction_date") if fmt else pl.col("transaction_date").str.slice(0, 10)
        return text.str.to_date(fmt or "%Y-%m-%d", strict=False)
    elif dtype == pl.Date:
        return pl.col("transaction_date").cast(pl.Date)
    elif isinstance(dtype, pl.Datetime):
        dt_col = pl.col("transaction_date")
        if dtype.time_unit != "us":
            dt_col = dt_col.cast(pl.Datetime("us", time_zone=dtype.time_zone))
        if dtype.time_zone is not None:
            if timezone is None:
                raise ValueError("Timezone-aware Datetime input requires --timezone parameter")
            dt_col = dt_col.dt.convert_time_zone(timezone)
        return dt_col.cast(pl.Date)
    else:
        raise ValueError("Dates must be Date, Datetime, or strings; epoch units are not guessed")


def _normalize_strings(raw: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize string columns: trim, replace empty with null/unknown."""
    return raw.with_columns(
        *[
            pl.col(x).cast(pl.String).str.strip_chars().replace("", None)
            for x in [C, "transaction_id", "product_id"]
        ],
        *[
            pl.col(x).cast(pl.String).str.strip_chars().replace("", None).fill_null("__UNKNOWN__")
            for x in ["department", "category"]
        ],
        pl.col("product_description").cast(pl.String).fill_null(""),
    )


def _handle_price_quantity(
    raw: pl.LazyFrame,
    mode: PriceMode,
    schema: dict[str, pl.DataType],
) -> pl.LazyFrame:
    """Handle price/quantity parsing and value computation."""
    # Handle price/quantity: if raw columns already exist (from dashboard canonical), use them
    if "price_raw" in schema and "quantity_raw" in schema:
        raw = raw.with_columns(
            pl.col("price").cast(pl.Float64, strict=False),
            pl.col("quantity").cast(pl.Float64, strict=False),
        )
    else:
        raw = raw.with_columns(
            # Preserve raw string values before casting to distinguish missing vs malformed
            pl.col("price")
            .cast(pl.String)
            .str.strip_chars()
            .replace("", None)
            .alias("price_raw"),
            pl.col("quantity")
            .cast(pl.String)
            .str.strip_chars()
            .replace("", None)
            .alias("quantity_raw"),
            *[pl.col(x).cast(pl.Float64, strict=False) for x in ["price", "quantity"]],
        )

    raw = raw.with_columns(
        (pl.col("price") * pl.col("quantity") if mode == "unit" else pl.col("price")).alias("value")
    )
    return raw


def _compute_validity_mask(raw: pl.LazyFrame) -> pl.Expr:
    """Compute validity mask for rows."""
    return (
        pl.all_horizontal(
            [pl.col(x).is_not_null() for x in [C, "transaction_id", "product_id", "date"]]
        )
        & pl.col("price").is_finite()
        & (pl.col("price") >= 0)
        & pl.col("quantity").is_finite()
        & (pl.col("quantity") > 0)
        & pl.col("value").is_finite()
    ).fill_null(False)


def _compute_quality_audit(
    raw: pl.LazyFrame,
    valid: pl.LazyFrame,
    rejected: pl.LazyFrame,
    observation_start: date | None,
    censor_days: int,
) -> dict[str, Any]:
    """Compute comprehensive quality audit metrics."""
    total_rows = raw.select(pl.len()).collect().item()
    rejected_count = rejected.select(pl.len()).collect().item()
    valid_count = total_rows - rejected_count

    # Left-censoring: customers whose first purchase is within CENSOR_DAYS of observation_start
    if observation_start is not None and valid_count > 0:
        first_purchase_per_customer = valid.group_by(C).agg(
            pl.col("date").min().alias("first_purchase")
        )
        left_censored_count = (
            first_purchase_per_customer.filter(
                (pl.col("first_purchase") - pl.lit(observation_start)).dt.total_days() < censor_days
            )
            .select(pl.len())
            .collect()
            .item()
        )
        total_customers = first_purchase_per_customer.select(pl.len()).collect().item()
    else:
        left_censored_count = 0
        total_customers = 0

    # Negative quantity metrics (for returns/refunds detection)
    neg_qty_rows = (
        raw.filter((pl.col("quantity") < 0) & pl.col("quantity").is_finite())
        .select(pl.len())
        .collect()
        .item()
    )
    neg_qty_value = (
        raw.filter((pl.col("quantity") < 0) & pl.col("quantity").is_finite())
        .select(pl.col("value").sum())
        .collect()
        .item()
        or 0.0
    )

    # Exact duplicate detection - track both source-exact and normalized same-day
    exact_source_cols = [
        C,
        "transaction_date",
        "transaction_id",
        "product_id",
        "product_description",
        "department",
        "category",
        "price",
        "quantity",
    ]
    exact_dup_count = (
        raw.group_by(exact_source_cols)
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum())
        .collect()
        .item()
        or 0
    )
    valid_raw = raw.filter(pl.col("valid"))
    exact_dup_value = (
        valid_raw.group_by(exact_source_cols)
        .agg(pl.col("value").first().alias("line_value"), pl.len().alias("dup_count"))
        .filter(pl.col("dup_count") > 1)
        .select((pl.col("line_value") * (pl.col("dup_count") - 1)).sum())
        .collect()
        .item()
        or 0.0
    )

    # Normalized same-day duplicate detection (for audit only)
    norm_source_cols = [
        C,
        "transaction_id",
        "product_id",
        "date",
        "product_description",
        "department",
        "category",
        "price",
        "quantity",
    ]
    norm_dup_count = (
        raw.group_by(norm_source_cols)
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum())
        .collect()
        .item()
        or 0
    )
    norm_dup_value = (
        valid_raw.group_by(norm_source_cols)
        .agg(pl.col("value").first().alias("line_value"), pl.len().alias("dup_count"))
        .filter(pl.col("dup_count") > 1)
        .select((pl.col("line_value") * (pl.col("dup_count") - 1)).sum())
        .collect()
        .item()
        or 0.0
    )

    # Baskets with rejected lines
    baskets_with_rejected = (
        raw.group_by([C, "transaction_id"])
        .agg(
            pl.col("valid").sum().alias("valid_count"),
            (pl.col("valid").not_()).sum().alias("rejected_count"),
        )
        .filter((pl.col("valid_count") > 0) & (pl.col("rejected_count") > 0))
        .select(pl.len())
        .collect()
        .item()
    )

    # Rejection breakdown
    if rejected_count > 0:
        breakdown_df = rejected.select(pl.col("rejection_reason").value_counts()).collect()
        breakdown = {}
        for row in breakdown_df.to_dicts():
            struct_val = row["rejection_reason"]
            breakdown[struct_val["rejection_reason"]] = struct_val["count"]
    else:
        breakdown = {}

    audit_base = (
        raw.select(
            pl.lit(total_rows).alias("source_rows"),
            pl.lit(rejected_count).alias("rejected_rows"),
            pl.lit(valid_count).alias("valid_rows"),
            pl.col("date").filter(pl.col("valid")).min().alias("first_date"),
            pl.col("date").filter(pl.col("valid")).max().alias("last_date"),
            # Null required fields (on raw)
            pl.col(C).is_null().sum().alias("null_customer_id"),
            pl.col("transaction_id").is_null().sum().alias("null_transaction_id"),
            pl.col("product_id").is_null().sum().alias("null_product_id"),
            pl.col("date").is_null().sum().alias("null_date"),
            # Invalid numerics
            pl.col("price").is_null().sum().alias("null_price"),
            (~pl.col("price").is_finite()).sum().alias("nonfinite_price"),
            (pl.col("price") < 0).sum().alias("negative_price"),
            pl.col("quantity").is_null().sum().alias("null_quantity"),
            (~pl.col("quantity").is_finite()).sum().alias("nonfinite_quantity"),
            (pl.col("quantity") <= 0).sum().alias("nonpositive_quantity"),
            (~pl.col("value").is_finite()).sum().alias("nonfinite_value"),
            # Raw string presence for malformed detection
            pl.col("price_raw").is_null().sum().alias("missing_price_raw"),
            pl.col("quantity_raw").is_null().sum().alias("missing_quantity_raw"),
            # New audit fields
            pl.lit(observation_start).alias("observation_start"),
            pl.lit(neg_qty_rows).alias("negative_quantity_rows"),
            pl.lit(neg_qty_value).alias("negative_quantity_value"),
            pl.lit(exact_dup_count).alias("exact_duplicate_rows"),
            pl.lit(exact_dup_value).alias("exact_duplicate_value"),
            pl.lit(norm_dup_count).alias("normalized_duplicate_rows"),
            pl.lit(norm_dup_value).alias("normalized_duplicate_value"),
            pl.lit(baskets_with_rejected).alias("baskets_with_rejected_lines"),
            # Left-censoring fields
            pl.lit(left_censored_count).alias("left_censored_customers"),
            pl.lit(total_customers).alias("total_customers_at_start"),
            pl.lit(censor_days).alias("censor_threshold_days"),
        )
        .collect(engine="streaming")
        .to_dicts()[0]
    )

    audit_base["rejection_breakdown"] = breakdown
    audit_base["rejection_rate"] = (
        round(audit_base["rejected_rows"] / audit_base["source_rows"], 4)
        if audit_base["source_rows"] > 0
        else 0.0
    )
    return audit_base


def _write_partitions(
    clean: pl.LazyFrame,
    root: Path,
    buckets: int,
    audit: dict[str, Any],
    clean_schema: pl.Schema,
) -> list[Path]:
    """Write partitioned data and check for basket conflicts."""
    paths = []
    # Remove bucket column from schema for empty partition handling
    partition_schema = pl.Schema({k: v for k, v in clean_schema.items() if k != "bucket"})
    for b in range(buckets):
        path = root / "partitions" / f"b{b}.parquet"
        partition_frame = clean.filter(pl.col("bucket") == b).drop("bucket")
        _sink(partition_frame, path, sort_by=C)
        # Ensure file exists even for empty partitions (streaming sink may skip empty)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(schema=partition_schema).write_parquet(path)
        part = pl.scan_parquet(path)
        conflicts = (
            part.group_by([C, "transaction_id"])
            .agg(pl.col("date").n_unique().alias("dates"))
            .filter(pl.col("dates") > 1)
        )
        if conflicts.select(pl.len()).collect().item():
            _sink(conflicts, root / "basket_conflicts.parquet", sort_by=[C, "transaction_id"])
            raise ValueError("Customer/transaction key spans dates; resolve its grain")
        audit["preserved_duplicate_extras"] += (
            part.group_by(part.collect_schema().names())
            .len()
            .select((pl.col("len") - 1).sum())
            .collect()
            .item()
            or 0
        )
        paths.append(path)
    return paths


def validate_input_schema(schema: dict[str, pl.DataType]) -> None:
    """Validate that input schema contains all required columns.

    Args:
        schema: Input DataFrame schema.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = set(REQUIRED_COLS) - set(schema.keys())
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def prepare(
    source: str,
    root: Path,
    mode: PriceMode,
    buckets: int,
    exclude: bool,
    fmt: str | None,
    through: str | None,
    coverage_confirmed: bool = False,
    drop_exact_duplicates: bool = False,
    timezone: str | None = None,
) -> tuple[list[Path], date, date | None]:
    """Prepare validated data from source Parquet.

    Args:
        source: Path to source Parquet file.
        root: Output root directory.
        mode: Price mode ("unit" or "line_total").
        buckets: Number of hash partitions.
        exclude: Whether to exclude invalid rows instead of raising.
        fmt: Date format string (e.g., "%Y-%m-%d").
        through: Observed through date (ISO format).
        coverage_confirmed: Whether future follow-up is available.
        drop_exact_duplicates: Whether to drop exact duplicate rows.
        timezone: Timezone for datetime conversion.

    Returns:
        Tuple of (partition paths, end date, observation start date).
    """
    src = pl.scan_parquet(source)
    schema = src.collect_schema()

    validate_input_schema(schema)

    dates_expr = _parse_dates(src, fmt, timezone)

    # Include price_raw/quantity_raw in select if they exist (from dashboard canonical)
    select_cols = REQUIRED_COLS + [c for c in ["price_raw", "quantity_raw"] if c in schema.keys()]

    # Preserve original source columns for exact deduplication (before normalization)
    raw = src.select(select_cols).with_columns(
        dates_expr.alias("date"),
        # Keep original source columns for exact deduplication
        pl.col(C).alias("customer_id_source"),
        pl.col("transaction_date").alias("transaction_date_source"),
        pl.col("transaction_id").alias("transaction_id_source"),
        pl.col("product_id").alias("product_id_source"),
        pl.col("product_description").alias("product_description_source"),
        pl.col("department").alias("department_source"),
        pl.col("category").alias("category_source"),
        pl.col("price").alias("price_source"),
        pl.col("quantity").alias("quantity_source"),
    )

    raw = _normalize_strings(raw)
    raw = _handle_price_quantity(raw, mode, schema)

    # Compute rejection reasons before filtering
    raw = raw.with_columns(_rejection_reasons(raw))

    good = _compute_validity_mask(raw)
    raw = raw.with_columns(good.alias("valid"))

    rejected = raw.filter(~pl.col("valid"))
    rejected_count = rejected.select(pl.len()).collect().item()
    valid = raw.filter(pl.col("valid"))

    # Enhanced quality audit
    observation_start = valid.select(pl.col("date").min()).collect().item()
    audit = _compute_quality_audit(raw, valid, rejected, observation_start, CENSOR_DAYS)

    (root / "quality.json").write_text(json.dumps(audit, default=str, indent=2))

    if not audit["source_rows"]:
        raise ValueError("Empty input")
    if audit["rejected_rows"]:
        # Write rejected rows with rejection_reason and raw values for diagnostics
        _sink(rejected.drop("valid"), root / "rejected.parquet")
        if not exclude:
            raise ValueError("Invalid rows: inspect quality.json and rejected.parquet")

    clean = valid.drop("valid", "rejection_reason", "price_raw", "quantity_raw")

    maximum_date = clean.select(pl.col("date").max()).collect().item()
    if maximum_date is None:
        raise ValueError("No valid records")

    end = date.fromisoformat(through) if through else maximum_date
    if end < maximum_date:
        raise ValueError(
            "observed_through precedes the latest valid transaction date; "
            "explicitly truncate the source before analysis"
        )

    clean = clean.filter(pl.col("date") <= end).with_columns(
        _safe_ratio(pl.col("value"), pl.col("quantity")).alias("unit_price"),
        pl.concat_str(
            [
                pl.col("department").str.len_bytes().cast(pl.String),
                pl.lit(":"),
                pl.col("department"),
                pl.col("category").str.len_bytes().cast(pl.String),
                pl.lit(":"),
                pl.col("category"),
            ]
        ).alias("category_key"),
        (pl.col(C).hash(seed=HASH_SEED_BUCKETING) % buckets).alias("bucket"),
    )

    # Drop exact duplicates if requested
    if drop_exact_duplicates:
        exact_source_cols = [
            "customer_id_source",
            "transaction_date_source",
            "transaction_id_source",
            "product_id_source",
            "product_description_source",
            "department_source",
            "category_source",
            "price_source",
            "quantity_source",
        ]
        clean = clean.unique(subset=exact_source_cols, maintain_order=True)

    # Drop source columns after exact duplicate detection
    source_cols_to_drop = [
        "customer_id_source",
        "transaction_date_source",
        "transaction_id_source",
        "product_id_source",
        "product_description_source",
        "department_source",
        "category_source",
        "price_source",
        "quantity_source",
        "transaction_date",
    ]
    clean = clean.drop(source_cols_to_drop)

    # Get schema before writing to file (for empty partition handling)
    clean_schema = clean.collect_schema()

    _sink(clean, root / "validated.parquet")
    clean = pl.scan_parquet(root / "validated.parquet")

    audit["observed_through"] = str(end)
    audit["coverage_confirmed"] = bool(coverage_confirmed)
    audit["coverage_interpretation"] = (
        "Future outcomes are unavailable unless coverage_confirmed is true; "
        "a maximum observed date alone does not establish complete capture."
    )
    audit["preserved_duplicate_extras"] = 0

    paths = _write_partitions(clean, root, buckets, audit, clean_schema)

    (root / "quality.json").write_text(json.dumps(audit, default=str, indent=2))

    _sink(
        clean.group_by("product_id").agg(
            pl.struct(["department", "category"]).n_unique().alias("classifications"),
            pl.col("product_description").n_unique().alias("descriptions"),
        ).filter((pl.col("classifications") > 1) | (pl.col("descriptions") > 1)),
        root / "product_variation.parquet",
        sort_by="product_id",
    )

    return paths, end, observation_start