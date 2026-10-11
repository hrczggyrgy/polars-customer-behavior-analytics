"""Purchase-only customer analytics from a nine-column Parquet source."""

import argparse
import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from retail_analytics.core.constants import (
    BASKET_CHANGE_EXPANDING_MIN,
    BASKET_CHANGE_SMALLER_MAX,
    BREADTH_CHANGE_NARROWING_MAX,
    CENSOR_DAYS,
    DOMINANT_SHARE_FOCUSED_MIN,
    ESTABLISHED_SHARE_BROAD_MIN,
    ESTABLISHED_SHARE_FOCUSED_MIN,
    ESTABLISHED_SHARE_ROUTINE_MIN,
    FREQUENCY_CHANGE_EXPANDING_MIN,
    FREQUENCY_CHANGE_SLOWING_MAX,
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
    MIN_TENURE_VALUE_TIER,
    NEW_SHARE_EXPLORATION_MIN,
    OVERDUE_CATEGORIES_NARROWING_MIN,
    QUANTILE_STYLE_HIGH,
    QUANTILE_STYLE_LOW,
    QUANTILE_VALUE_HIGH,
    QUANTILE_VALUE_LOW,
    REACTIVATION_EVENTS_MIN,
    WINDOW_PREVIOUS_START_OFFSET,
    WINDOW_RECENT,
    WINDOW_YEAR,
)
from retail_analytics.core.thresholds import (
    ThresholdPolicy,
    calibrate_thresholds,
    compute_data_profile,
    compute_label_shares,
    default_policy,
    select_reference_cutoff,
    write_thresholds,
)

COLS = [
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
C = "customer_id"
K = [C, "transaction_id", "date"]
P = [C, "product_id"]
G = [C, "department", "category"]
VERSION = "reconstructed-1.3"


def ratio(a, b):
    """Safe division: returns null when denominator is null or <= 0."""
    return pl.when(b.is_not_null() & (b > 0)).then(a / b).otherwise(None)


def sink(frame, path, sort_by=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sort_by is not None:
        frame = frame.sort(sort_by)
    frame.sink_parquet(path, compression="zstd", engine="streaming")


def taglist(pairs):
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


def _rejection_reasons(raw):
    """Compute per-row rejection reason codes."""
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


def prepare(
    source,
    root,
    mode,
    buckets,
    exclude,
    fmt,
    through,
    coverage_confirmed=False,
    drop_exact_duplicates=False,
    timezone=None,
):
    src = pl.scan_parquet(source)
    schema = src.collect_schema()
    if set(COLS) - set(schema.names()):
        raise ValueError(f"Missing columns: {set(COLS) - set(schema.names())}")
    dtype = schema["transaction_date"]
    if dtype == pl.String:
        text = pl.col("transaction_date") if fmt else pl.col("transaction_date").str.slice(0, 10)
        dates = text.str.to_date(fmt or "%Y-%m-%d", strict=False)
    elif dtype == pl.Date:
        dates = pl.col("transaction_date").cast(pl.Date)
    elif isinstance(dtype, pl.Datetime):
        if dtype.time_unit != "us":
            dt_col = pl.col("transaction_date").cast(pl.Datetime("us"))
        else:
            dt_col = pl.col("transaction_date")
        if dtype.time_zone is not None:
            if timezone is None:
                raise ValueError("Timezone-aware Datetime input requires --timezone parameter")
            dt_col = dt_col.dt.convert_time_zone(timezone)
        dates = dt_col.cast(pl.Date)
    else:
        raise ValueError("Dates must be Date, Datetime, or strings; epoch units are not guessed")
    raw = src.select(COLS).with_columns(
        dates.alias("date"),
        *[
            pl.col(x).cast(pl.String).str.strip_chars().replace("", None)
            for x in [C, "transaction_id", "product_id"]
        ],
        *[
            pl.col(x).cast(pl.String).str.strip_chars().replace("", None).fill_null("__UNKNOWN__")
            for x in ["department", "category"]
        ],
        pl.col("product_description").cast(pl.String).fill_null(""),
        # Preserve raw string values before casting to distinguish missing vs malformed
        pl.col("price").cast(pl.String).str.strip_chars().replace("", None).alias("price_raw"),
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

    # Compute rejection reasons before filtering
    raw = raw.with_columns(_rejection_reasons(raw))

    good = (
        pl.all_horizontal(
            [pl.col(x).is_not_null() for x in [C, "transaction_id", "product_id", "date"]]
        )
        & pl.col("price").is_finite()
        & (pl.col("price") >= 0)
        & pl.col("quantity").is_finite()
        & (pl.col("quantity") > 0)
        & pl.col("value").is_finite()
    ).fill_null(False)
    raw = raw.with_columns(good.alias("valid"))

    # Enhanced quality audit - compute ALL counts from raw frame (before filtering)
    # This ensures null/invalid counts are non-zero when problems exist
    total_rows = raw.select(pl.len()).collect().item()
    rejected = raw.filter(~pl.col("valid"))
    rejected_count = rejected.select(pl.len()).collect().item()
    valid = raw.filter(pl.col("valid"))
    valid_count = total_rows - rejected_count

    # Observation start (min valid date)
    observation_start = valid.select(pl.col("date").min()).collect().item()

    # Left-censoring: customers whose first purchase is within CENSOR_DAYS of observation_start
    # We compute this per customer and add it to the quality audit
    if observation_start is not None and valid_count > 0:
        first_purchase_per_customer = valid.group_by(C).agg(
            pl.col("date").min().alias("first_purchase")
        )
        left_censored_count = (
            first_purchase_per_customer.filter(
                (pl.col("first_purchase") - pl.lit(observation_start)).dt.total_days() < CENSOR_DAYS
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

    # Exact duplicate detection (on all 9 source columns after cleaning)
    raw_source_cols = [
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
    exact_dup_count = (
        raw.group_by(raw_source_cols)
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum())
        .collect()
        .item()
        or 0
    )
    exact_dup_value = (
        raw.group_by(raw_source_cols)
        .agg(pl.col("value").sum().alias("dup_value"), pl.len().alias("dup_count"))
        .filter(pl.col("dup_count") > 1)
        .select((pl.col("dup_value") * (pl.col("dup_count") - 1)).sum())
        .collect()
        .item()
        or 0.0
    )

    # Baskets with rejected lines (customer+transaction has >=1 rejected AND >=1 valid row)
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

    # Base audit metrics - ALL counts from raw
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
            # Invalid numerics - separate missing source from malformed and non-finite
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
            pl.lit(baskets_with_rejected).alias("baskets_with_rejected_lines"),
            # Left-censoring fields
            pl.lit(left_censored_count).alias("left_censored_customers"),
            pl.lit(total_customers).alias("total_customers_at_start"),
            pl.lit(CENSOR_DAYS).alias("censor_threshold_days"),
        )
        .collect(engine="streaming")
        .to_dicts()[0]
    )

    # Rejection breakdown (only if there are rejected rows)
    if rejected_count > 0:
        breakdown_df = rejected.select(pl.col("rejection_reason").value_counts()).collect()
        breakdown = {}
        for row in breakdown_df.to_dicts():
            struct_val = row["rejection_reason"]
            breakdown[struct_val["rejection_reason"]] = struct_val["count"]
    else:
        breakdown = {}

    audit_base["rejection_breakdown"] = breakdown
    audit_base["rejection_rate"] = (
        round(audit_base["rejected_rows"] / audit_base["source_rows"], 4)
        if audit_base["source_rows"] > 0
        else 0.0
    )
    audit = audit_base

    (root / "quality.json").write_text(json.dumps(audit, default=str, indent=2))

    if not audit["source_rows"]:
        raise ValueError("Empty input")
    if audit["rejected_rows"]:
        # Write rejected rows with rejection_reason
        sink(rejected.drop("valid", "price_raw", "quantity_raw"), root / "rejected.parquet")
        if not exclude:
            raise ValueError("Invalid rows: inspect quality.json and rejected.parquet")
    clean = raw.filter(pl.col("valid")).drop(
        "valid", "transaction_date", "rejection_reason", "price_raw", "quantity_raw"
    )
    maximum_date = clean.select(pl.col("date").max()).collect().item()
    if maximum_date is None:
        raise ValueError("No valid records")
    end = date.fromisoformat(through) if through else maximum_date
    if end < maximum_date:
        raise ValueError(
            "observed_through precedes the latest valid transaction date; explicitly truncate the source before analysis"
        )
    clean = clean.filter(pl.col("date") <= end).with_columns(
        ratio(pl.col("value"), pl.col("quantity")).alias("unit_price"),
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
        raw_source_cols = [
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
        clean = clean.unique(subset=raw_source_cols, maintain_order=True)
    sink(clean, root / "validated.parquet")
    clean = pl.scan_parquet(root / "validated.parquet")
    audit["observed_through"] = str(end)
    audit["coverage_confirmed"] = bool(coverage_confirmed)
    audit["coverage_interpretation"] = (
        "Future outcomes are unavailable unless coverage_confirmed is true; a maximum observed date alone does not establish complete capture."
    )
    audit["preserved_duplicate_extras"] = 0
    paths = []
    for b in range(buckets):
        path = root / "partitions" / f"b{b}.parquet"
        sink(clean.filter(pl.col("bucket") == b).drop("bucket"), path, sort_by=C)
        part = pl.scan_parquet(path)
        conflicts = (
            part.group_by([C, "transaction_id"])
            .agg(pl.col("date").n_unique().alias("dates"))
            .filter(pl.col("dates") > 1)
        )
        if conflicts.select(pl.len()).collect().item():
            sink(conflicts, root / "basket_conflicts.parquet", sort_by=[C, "transaction_id"])
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
    (root / "quality.json").write_text(json.dumps(audit, default=str, indent=2))
    sink(
        clean.group_by("product_id")
        .agg(
            pl.struct(["department", "category"]).n_unique().alias("classifications"),
            pl.col("product_description").n_unique().alias("descriptions"),
        )
        .filter((pl.col("classifications") > 1) | (pl.col("descriptions") > 1)),
        root / "product_variation.parquet",
        sort_by="product_id",
    )
    return paths, end, observation_start


def snapshot(path, d, end, root, coverage_confirmed=False, observation_start=None):
    out = root / "events" / str(d) / path.stem
    target = root / "numeric" / str(d) / path.name
    line = pl.scan_parquet(path).filter(pl.col("date") <= d)

    # Temporal boundary assertion: verify no post-cutoff dates in descriptive data
    max_date_check = line.select(pl.col("date").max()).collect().item()
    if max_date_check is not None and max_date_check > d:
        raise AssertionError(
            f"Temporal leakage: descriptive data contains date {max_date_check} > cutoff {d}"
        )

    s30 = d - timedelta(days=WINDOW_RECENT - 1)
    s90 = d - timedelta(days=WINDOW_RECENT - 1)
    s365 = d - timedelta(days=WINDOW_YEAR - 1)
    old0 = d - timedelta(days=WINDOW_PREVIOUS_START_OFFSET)
    old1 = d - timedelta(days=WINDOW_RECENT)
    recent, old, year = (
        pl.col("date") >= s90,
        pl.col("date").is_between(old0, old1),
        pl.col("date") >= s365,
    )
    sink(
        line.group_by(K).agg(
            pl.col("value").sum().alias("spend"),
            pl.col("product_id").n_unique().alias("products"),
            pl.col("category_key").n_unique().alias("breadth"),
        ),
        out / "baskets.parquet",
        sort_by=[C, "date", "transaction_id"],
    )
    basket = pl.scan_parquet(out / "baskets.parquet")
    totals = line.select(pl.col("value").sum()).collect().item() or 0
    if abs((basket.select(pl.col("spend").sum()).collect().item() or 0) - totals) > 1e-7 * max(
        1, abs(totals)
    ):
        raise AssertionError("Revenue reconciliation failed")
    base = basket.group_by(C).agg(
        pl.col("date").min().alias("first_purchase"),
        pl.col("date").max().alias("last_purchase"),
        pl.len().alias("purchases_history"),
        pl.col("spend").sum().alias("spend_history"),
        *[
            pl.col("date")
            .filter(pl.col("date") >= d - timedelta(days=w - 1))
            .len()
            .alias(f"purchases_{w}d")
            for w in [30, 90, 365]
        ],
        *[
            pl.col("spend")
            .filter(pl.col("date") >= d - timedelta(days=w - 1))
            .sum()
            .alias(f"spend_{w}d")
            for w in [30, 90, 365]
        ],
        pl.col("date").filter(old).len().alias("purchases_previous90d"),
        pl.col("spend").filter(old).sum().alias("spend_previous90d"),
        pl.col("spend").filter(year).mean().alias("aov_365d"),
        pl.col("spend").filter(year).median().alias("median_basket"),
        ratio(pl.col("spend").filter(year).std(), pl.col("spend").filter(year).mean()).alias(
            "basket_cv"
        ),
        pl.col("breadth").filter(year).median().alias("median_breadth"),
        pl.col("breadth").filter(recent).mean().alias("breadth_recent"),
        pl.col("breadth").filter(old).mean().alias("breadth_previous"),
    )
    # Left-censoring flag: customer's first purchase within CENSOR_DAYS of observation_start
    if observation_start is not None:
        base = base.with_columns(
            (
                (pl.col("first_purchase") - pl.lit(observation_start)).dt.total_days() < CENSOR_DAYS
            ).alias("left_censored"),
        )
    else:
        base = base.with_columns(pl.lit(None).alias("left_censored"))

    # Check if base has any customers before proceeding
    base_count = base.select(pl.len()).collect()
    if base_count.height == 0 or base_count.item() == 0:
        # No customers with transactions before cutoff, don't write numeric file
        return None
    day = line.group_by([C, "date"]).agg(
        pl.col("value").sum().alias("day_spend"),
        pl.col("product_id").unique().alias("products"),
        pl.col("category_key").unique().alias("categories"),
    )
    day = day.with_columns(
        *[
            pl.col(x).shift(1).over(C, order_by="date").alias("prev_" + x)
            for x in ["date", "products", "categories"]
        ],
        pl.col("day_spend")
        .shift(1)
        .rolling_median(5, min_samples=3)
        .over(C, order_by="date")
        .alias("baseline_spend"),
    )
    day = day.with_columns(
        (pl.col("date") - pl.col("prev_date")).dt.total_days().alias("gap"),
        ratio(pl.col("day_spend"), pl.col("baseline_spend")).alias("relative_spend"),
        *[
            ratio(
                pl.col(x).list.set_intersection(pl.col("prev_" + x)).list.len(),
                pl.col(x).list.set_union(pl.col("prev_" + x)).list.len(),
            ).alias(x + "_overlap")
            for x in ["products", "categories"]
        ],
    )
    day = day.with_columns(
        pl.col("gap")
        .shift(1)
        .rolling_median(5, min_samples=3)
        .over(C, order_by="date")
        .alias("baseline_gap")
    )
    sink(day, out / "days.parquet", sort_by=[C, "date"])
    day = pl.scan_parquet(out / "days.parquet")
    usable = year & (pl.col("prev_date") >= s365)
    rhythm = day.group_by(C).agg(
        pl.len().alias("days_history"),
        pl.col("date").filter(year).len().alias("days_365d"),
        pl.col("gap").filter(usable).count().alias("gap_count"),
        pl.col("gap").filter(usable).median().alias("median_gap"),
        ratio(pl.col("gap").filter(usable).std(), pl.col("gap").filter(usable).mean()).alias(
            "gap_cv"
        ),
        pl.col("products_overlap").filter(usable).median().alias("product_overlap"),
        pl.col("categories_overlap").filter(usable).median().alias("category_overlap"),
        (pl.col("date").dt.weekday() >= 6).filter(year).mean().alias("weekend_share"),
        pl.col("relative_spend").filter(recent).median().alias("personal_spend_ratio"),
        ((pl.col("date") >= s30) & (ratio(pl.col("gap"), pl.col("baseline_gap")) >= 2))
        .fill_null(False)
        .sum()
        .alias("reactivation_events"),
    )
    weekdays = (
        day.filter(year)
        .group_by([C, pl.col("date").dt.weekday().alias("weekday")])
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum().over(C)).alias("p"))
    )
    weekday_stats = weekdays.group_by(C).agg(
        (-(pl.col("p") * pl.col("p").log())).filter(pl.col("p") > 0).sum().alias("weekday_entropy")
    )
    last6 = (
        day.with_columns(
            pl.col("date").cum_count().over(C, order_by="date", descending=True).alias("rank")
        )
        .filter(pl.col("rank") <= 6)
        .select(C, "date")
    )
    breadth = (
        line.join(last6, on=[C, "date"], how="inner", validate="m:1")
        .group_by(C)
        .agg(
            pl.col("product_id").n_unique().alias("products_last6days"),
            pl.col("category_key").n_unique().alias("categories_last6days"),
        )
    )
    pd = line.group_by(P + ["date"]).agg(
        pl.col("quantity").sum().alias("qty"), pl.col("value").sum().alias("spend")
    )
    pd = pd.with_columns(
        (pl.col("date").cum_count().over(P, order_by="date") - 1).alias("prior_days"),
        pl.col("date").shift(-1).over(P, order_by="date").alias("next_date"),
        pl.col("qty")
        .shift(1)
        .rolling_median(5, min_samples=3)
        .over(P, order_by="date")
        .alias("baseline_qty"),
    )
    pd = pd.with_columns(
        (pl.col("next_date") - pl.col("date")).dt.total_days().alias("next_gap"),
        ratio(pl.col("qty"), pl.col("baseline_qty")).alias("relative_qty"),
    )
    sink(pd, out / "product_days.parquet", sort_by=[C, "product_id", "date"])
    pd = pl.scan_parquet(out / "product_days.parquet")
    ph = pd.group_by(P).agg(
        pl.col("date").min().alias("trial"),
        pl.col("date").filter(pl.col("prior_days") == 1).min().alias("repeat"),
        pl.len().alias("purchase_days"),
        pl.col("spend").filter(year).sum().alias("year_spend"),
    )
    sink(ph, out / "product_history.parquet", sort_by=P)
    eligible = pl.col("trial").is_between(s365, d - timedelta(days=90))
    success = (
        eligible
        & pl.col("repeat").is_not_null()
        & (pl.col("repeat") <= pl.col("trial") + pl.duration(days=90))
    )
    trials = ph.group_by(C).agg(
        eligible.sum().alias("eligible_trials"),
        success.fill_null(False).sum().alias("repeated_trials"),
        (pl.col("purchase_days") >= 3).sum().alias("core_size"),
        pl.col("year_spend").filter(pl.col("purchase_days") >= 3).sum().alias("core_spend"),
    )
    product_baskets = (
        line.select(K + ["product_id"])
        .unique()
        .join(pd.select(P + ["date", "prior_days"]), on=P + ["date"], how="left", validate="m:1")
        .group_by(K)
        .agg(
            (pl.col("prior_days") == 0).mean().alias("new_share"),
            (pl.col("prior_days") >= 1).mean().alias("repeat_share"),
            (pl.col("prior_days") >= 2).mean().alias("established_share"),
        )
    )
    habits = product_baskets.group_by(C).agg(
        pl.col("new_share").filter(year).mean().alias("new_share"),
        pl.col("repeat_share").filter(year).mean().alias("repeat_share"),
        pl.col("established_share").filter(year).mean().alias("established_share"),
    )
    quantity = pd.group_by(C).agg(
        pl.col("next_gap")
        .filter(year & (pl.col("relative_qty") >= 1.5))
        .median()
        .alias("gap_after_large_qty"),
        pl.col("next_gap")
        .filter(year & pl.col("relative_qty").is_between(0.75, 1.25))
        .median()
        .alias("gap_after_typical_qty"),
        pl.col("next_gap")
        .filter(year & (pl.col("relative_qty") >= 1.5))
        .count()
        .alias("large_qty_gap_count"),
    )
    cd = (
        line.group_by(G + ["date"])
        .agg(pl.col("value").sum().alias("spend"))
        .with_columns(pl.col("date").shift(1).over(G, order_by="date").alias("prev"))
    )
    cd = cd.with_columns((pl.col("date") - pl.col("prev")).dt.total_days().alias("gap"))
    ch = cd.group_by(G).agg(
        pl.col("spend").filter(year).sum().alias("year_spend"),
        pl.len().alias("days"),
        pl.col("date").filter(year).len().alias("year_days"),
        pl.col("date").max().alias("last"),
        pl.col("gap").median().alias("median_gap"),
    )
    ch = ch.with_columns(
        ratio(pl.col("year_spend"), pl.col("year_spend").sum().over(C)).alias("share"),
        ratio((pl.lit(d) - pl.col("last")).dt.total_days(), pl.col("median_gap")).alias("lateness"),
    )
    sink(ch, out / "category_history.parquet", sort_by=G)
    categories = ch.group_by(C).agg(
        (pl.col("year_days") > 0).sum().alias("category_count"),
        (pl.col("share") ** 2).sum().alias("category_hhi"),
        (-(pl.col("share") * pl.col("share").log()))
        .filter(pl.col("share") > 0)
        .sum()
        .alias("category_entropy"),
        pl.col("share").max().alias("dominant_share"),
        ((pl.col("days") >= 3) & (pl.col("lateness") >= 2))
        .fill_null(False)
        .sum()
        .alias("overdue_categories"),
    )
    dominant = (
        ch.filter(pl.col("year_days") > 0)
        .sort([C, "year_spend", "department", "category"], descending=[False, True, False, False])
        .group_by(C, maintain_order=True)
        .agg(
            pl.col("department").first().alias("dominant_department"),
            pl.col("category").first().alias("dominant_category"),
        )
    )
    prices = (
        line.filter(year)
        .join(
            pl.scan_parquet(root / "price_references" / f"{d}.parquet"),
            on="product_id",
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.when(pl.col("reference_count") >= 5)
            .then(ratio(pl.col("unit_price"), pl.col("reference_price")))
            .otherwise(None)
            .alias("relative_price")
        )
    )
    prices = prices.group_by(C).agg(
        pl.col("relative_price").median().alias("relative_paid_price"),
        pl.col("relative_price").count().alias("comparable_price_lines"),
    )
    frame = base
    for extra in [
        rhythm,
        weekday_stats,
        breadth,
        trials,
        habits,
        quantity,
        categories,
        dominant,
        prices,
    ]:
        frame = frame.join(extra, on=C, how="left", validate="1:1")

    # Post-join integrity check: no duplicate customers (lazy, no materialization)
    dup_check = frame.group_by(C).len().filter(pl.col("len") > 1)
    has_dups = dup_check.select(pl.len()).collect().item()
    if has_dups:
        raise ValueError("Duplicate customer_ids after feature merge")

    frame = frame.with_columns(
        pl.lit(d).alias("snapshot"),
        (pl.lit(d) - pl.col("first_purchase")).dt.total_days().alias("tenure"),
        (pl.lit(d) - pl.col("last_purchase")).dt.total_days().alias("recency"),
        ratio(pl.col("spend_90d"), pl.col("purchases_90d")).alias("aov_recent"),
        ratio(pl.col("spend_previous90d"), pl.col("purchases_previous90d")).alias("aov_previous"),
        ratio(pl.col("repeated_trials"), pl.col("eligible_trials")).alias("trial_conversion"),
        ratio(pl.col("core_spend"), pl.col("spend_365d")).alias("core_spend_share"),
    )
    frame = frame.with_columns(
        ratio(pl.col("recency"), pl.col("median_gap")).alias("lateness"),
        (pl.col("tenure") >= MIN_TENURE_COMPARABLE).alias("comparable_windows"),
        (pl.col("breadth_recent") - pl.col("breadth_previous")).alias("breadth_change"),
        *[
            pl.when(pl.col("days_history") >= 6).then(pl.col(x)).otherwise(None).alias(x)
            for x in ["products_last6days", "categories_last6days"]
        ],
    )
    frame = frame.with_columns(
        *[
            pl.when(pl.col("comparable_windows"))
            .then(ratio(pl.col(a), pl.col(b)) - 1)
            .otherwise(None)
            .alias(name)
            for a, b, name in [
                ("purchases_90d", "purchases_previous90d", "frequency_change"),
                ("aov_recent", "aov_previous", "basket_change"),
                ("spend_90d", "spend_previous90d", "spend_change"),
            ]
        ],
        pl.lit(VERSION).alias("version"),
    )
    for h in [30, 90]:
        horizon_end = d + timedelta(days=h)
        followup_eligible = bool(coverage_confirmed) and horizon_end <= end
        future = (
            pl.scan_parquet(path)
            .filter(pl.col("date").is_between(d + timedelta(days=1), horizon_end))
            .group_by(C)
            .agg(
                pl.col("transaction_id").n_unique().alias(f"future_purchases_{h}d"),
                pl.col("value").sum().alias(f"future_spend_{h}d"),
            )
        )
        frame = frame.join(future, on=C, how="left", validate="1:1").with_columns(
            pl.lit(followup_eligible).alias(f"eligible_followup_{h}d"),
            *[
                pl.when(pl.lit(followup_eligible))
                .then(pl.col(x).fill_null(0))
                .otherwise(None)
                .alias(x)
                for x in [f"future_purchases_{h}d", f"future_spend_{h}d"]
            ],
        )

    # Final temporal assertion: descriptive features must not use post-cutoff data
    # (future columns are explicitly allowed to be post-cutoff)
    [
        c
        for c in frame.collect_schema().names()
        if not c.startswith(("future_", "eligible_followup_"))
    ]
    # This is implicitly enforced by the initial `line.filter(pl.col("date") <= d)`
    # but we document the invariant here.

    target = root / "numeric" / str(d) / path.name
    sink(frame, target)
    return target


def label(frame, thresholds):
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
    if (
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
    lateness_threshold = thresholds.get("lateness_overdue_min") or LATENESS_OVERDUE_MIN
    freq_change_threshold = thresholds.get("frequency_change_threshold") or 0.25
    basket_change_threshold = thresholds.get("basket_change_threshold") or 0.25

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

    # Lifecycle Stages (mutually exclusive)
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

    # Evidence Level (mutually exclusive)
    evidence = (
        pl.when(enough & (pl.col("gap_count") >= MIN_GAP_COUNT_STRONGER))
        .then(pl.lit("stronger"))
        .when(enough)
        .then(pl.lit("limited"))
        .otherwise(pl.lit("insufficient"))
    )

    frame = frame.with_columns(
        enough.alias("style_eligible"),
        taglist(style).alias("behavioral_tags"),
        taglist(trajectory_pairs).alias("trajectory_tags"),
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
    if thresholds.get("value_low") is None or thresholds.get("value_high") is None:
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


def reports(root, buckets):
    allrows = pl.scan_parquet(str(root / "snapshots" / "*" / "*.parquet"))
    profile = allrows.group_by(["snapshot", "behavior"]).agg(
        pl.len().alias("customers"),
        pl.col("spend_90d").sum().alias("revenue"),
        pl.col("spend_90d").median().alias("median_spend"),
        pl.col("median_gap").median().alias("median_gap"),
        pl.col("new_share").mean().alias("exploration"),
        pl.col("trial_conversion").mean().alias("mean_customer_trial_conversion"),
        *[
            (pl.col(f"future_purchases_{h}d") > 0)
            .filter(pl.col(f"eligible_followup_{h}d"))
            .mean()
            .alias(f"future_purchase_rate_{h}d")
            for h in [30, 90]
        ],
        *[
            pl.col(f"future_spend_{h}d")
            .filter(pl.col(f"eligible_followup_{h}d"))
            .mean()
            .alias(f"future_mean_spend_{h}d")
            for h in [30, 90]
        ],
        *[
            pl.col(f"eligible_followup_{h}d").sum().alias(f"eligible_customers_{h}d")
            for h in [30, 90]
        ],
    )
    profile = profile.with_columns(
        ratio(pl.col("customers"), pl.col("customers").sum().over("snapshot")).alias(
            "customer_share"
        ),
        ratio(pl.col("revenue"), pl.col("revenue").sum().over("snapshot")).alias("revenue_share"),
    )
    sink(profile, root / "segment_profiles.parquet", sort_by=["snapshot", "behavior"])

    # Cohort profiles with exposure tracking
    cohort_data = allrows.with_columns(pl.col("first_purchase").dt.truncate("1mo").alias("cohort"))
    cohort_profile = cohort_data.group_by(["snapshot", "cohort"]).agg(
        pl.len().alias("customers"),
        pl.col("spend_history").mean().alias("mean_cumulative_spend"),
        (pl.col("purchases_30d") > 0).mean().alias("rolling30d_purchase_rate"),
        (pl.col("future_purchases_30d") > 0)
        .filter(pl.col("eligible_followup_30d"))
        .mean()
        .alias("future30d_purchase_rate"),
        # Exposure: months between cohort start and snapshot
        (
            (pl.col("snapshot").dt.year() - pl.col("cohort").dt.year()) * 12
            + (pl.col("snapshot").dt.month() - pl.col("cohort").dt.month())
        )
        .first()
        .alias("observation_months"),
        # Minimum tenure in cohort
        pl.col("tenure").min().alias("min_tenure_days"),
        pl.col("tenure").max().alias("max_tenure_days"),
    )
    sink(cohort_profile, root / "cohort_profiles.parquet", sort_by=["snapshot", "cohort"])

    paths = []
    for b in range(buckets):
        try:
            rows = pl.scan_parquet(str(root / "snapshots" / "*" / f"b{b}.parquet"))
            if rows.select(pl.len()).collect().item() == 0:
                continue
        except Exception:
            continue
        for dim in ["lifecycle", "behavior", "trajectory", "value_tier"]:
            prev = rows.with_columns(
                pl.col(dim).shift(1).over(C, order_by="snapshot").alias("from_label"),
                pl.col("snapshot").shift(1).over(C, order_by="snapshot").alias("from_date"),
            ).filter(pl.col("from_label").is_not_null())
            trans = (
                prev.group_by(["from_date", "snapshot", "from_label", dim])
                .len()
                .rename({dim: "to_label", "len": "customers"})
                .with_columns(pl.lit(dim).alias("dimension"))
            )
            target = root / "transition_parts" / f"b{b}_{dim}.parquet"
            sink(
                trans,
                target,
                sort_by=["from_date", "snapshot", "dimension", "from_label", "to_label"],
            )
            paths.append(str(target))
    if paths:
        trans = (
            pl.scan_parquet(paths)
            .group_by(["from_date", "snapshot", "dimension", "from_label", "to_label"])
            .agg(pl.col("customers").sum())
        )
        trans = trans.with_columns(
            pl.col("customers")
            .sum()
            .over(["from_date", "snapshot", "dimension", "from_label"])
            .alias("eligible_origin")
        )
        sink(
            trans.with_columns(
                ratio(pl.col("customers"), pl.col("eligible_origin")).alias("transition_rate")
            ),
            root / "segment_transitions.parquet",
            sort_by=["from_date", "snapshot", "dimension", "from_label", "to_label"],
        )


def run(
    source,
    output,
    price_mode,
    snapshots,
    buckets=8,
    exclude_invalid=False,
    date_format=None,
    observed_through=None,
    coverage_confirmed=False,
    drop_exact_duplicates=False,
    overwrite=False,
    timezone=None,
    value_quantile_low=QUANTILE_VALUE_LOW,
    value_quantile_high=QUANTILE_VALUE_HIGH,
    style_quantile_low=QUANTILE_STYLE_LOW,
    style_quantile_high=QUANTILE_STYLE_HIGH,
    min_reference_population=MIN_REFERENCE_POPULATION,
    reference_date=None,
):
    if buckets < 1 or price_mode not in ["unit", "line_total"]:
        raise ValueError("Invalid buckets or price_mode")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        if overwrite:
            import shutil

            shutil.rmtree(root)
        else:
            raise ValueError("Output directory must be empty (use --overwrite to replace)")
    root.mkdir(parents=True, exist_ok=True)
    dates = sorted({date.fromisoformat(x) for x in snapshots})
    if not dates:
        raise ValueError("At least one cutoff required")
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
            sort_by="product_id",
        )
        for path in paths:
            if pl.scan_parquet(path).select(pl.len()).collect().item() > 0:
                result = snapshot(path, d, end, root, coverage_confirmed, observation_start)
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
            "reference_date": str(dates[0]),
            "reference_population": 0,
            "value_population": 0,
            "style_population": 0,
        }
        write_thresholds(root, thresholds)
        return

    # Build policy from CLI parameters
    policy = ThresholdPolicy(
        low_p=min(value_quantile_low, style_quantile_low),
        high_p=max(value_quantile_high, style_quantile_high),
        min_n=min_reference_population,
    )

    # Select reference cutoff: earliest with sufficient & stable population
    # If reference_date is provided, use that specific cutoff; otherwise auto-select
    if reference_date is not None:
        ref_date = date.fromisoformat(reference_date)
        ref = pl.scan_parquet([str(x) for x in numeric if x.parent.name == str(ref_date)])
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
            sink(
                label(pl.scan_parquet(path), thresholds),
                root / "snapshots" / path.parent.name / path.name,
                sort_by=C,
            )
        reports(root, buckets)
        print(f"Completed {len(dates)} cutoffs: {root} (no reference cutoff met criteria)")
        return

    # Calibrate thresholds from reference frame
    thresholds = calibrate_thresholds(ref, policy, ref_date)

    # Compute data profile for manifest
    data_profile = compute_data_profile(ref, observation_start, end)
    thresholds["data_profile"] = data_profile

    # Add reference selection info
    thresholds["reference_selection"] = ref_info

    write_thresholds(root, thresholds)

    # Label all snapshots
    for path in numeric:
        sink(
            label(pl.scan_parquet(path), thresholds),
            root / "snapshots" / path.parent.name / path.name,
            sort_by=C,
        )

    # Compute label shares for diagnostics (on first snapshot)
    first_snapshot_path = root / "snapshots" / str(dates[0]) / "*.parquet"
    try:
        first_frame = pl.scan_parquet(str(first_snapshot_path))
        label_shares = compute_label_shares(first_frame)
        if label_shares:
            thresholds["label_shares"] = label_shares
            write_thresholds(root, thresholds)
    except Exception:
        pass

    reports(root, buckets)
    print(f"Completed {len(dates)} cutoffs: {root} (reference: {ref_date})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, dest="source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--price-mode", required=True, choices=["unit", "line_total"])
    parser.add_argument("--snapshots", nargs="+", required=True)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--exclude-invalid", action="store_true")
    parser.add_argument("--date-format")
    parser.add_argument("--observed-through")
    parser.add_argument("--coverage-confirmed", action="store_true")
    parser.add_argument("--drop-exact-duplicates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--timezone", help="Timezone for timezone-aware Datetime input (e.g., 'UTC', 'US/Eastern')"
    )
    # Threshold parameters
    parser.add_argument(
        "--value-quantile-low",
        type=float,
        default=QUANTILE_VALUE_LOW,
        help="Value tier low quantile (default 0.33)",
    )
    parser.add_argument(
        "--value-quantile-high",
        type=float,
        default=QUANTILE_VALUE_HIGH,
        help="Value tier high quantile (default 0.80)",
    )
    parser.add_argument(
        "--style-quantile-low",
        type=float,
        default=QUANTILE_STYLE_LOW,
        help="Style tier low quantile (default 0.33)",
    )
    parser.add_argument(
        "--style-quantile-high",
        type=float,
        default=QUANTILE_STYLE_HIGH,
        help="Style tier high quantile (default 0.80)",
    )
    parser.add_argument(
        "--min-reference-population",
        type=int,
        default=MIN_REFERENCE_POPULATION,
        help="Minimum reference population for thresholds (default 100)",
    )
    parser.add_argument(
        "--reference-date",
        help="Override reference date for threshold computation (default: first snapshot)",
    )
    run(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
