"""Customer feature computation for the retail analytics pipeline.

This module computes all customer-level behavioral features from transaction data:
- Base features: purchase history, spend, frequency, recency
- Day-level features: gaps, overlaps, spend ratios
- Rhythm features: cadence regularity, reactivation events
- Weekday entropy: temporal purchase patterns
- Breadth in last 6 shopping days
- Product-level features: trial, repeat, conversion
- Habit features: new/repeat/established product shares
- Quantity-based gap features
- Category-level features: spend, gaps, overdue
- Price features: relative price paid vs reference
"""

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from retail_analytics.core.constants import (
    CENSOR_DAYS,
    MIN_TENURE_COMPARABLE,
    WINDOW_PREVIOUS_START_OFFSET,
    WINDOW_RECENT,
    WINDOW_RECENT_30,
    WINDOW_YEAR,
)
from retail_analytics.core.stats import safe_ratio

# Key aliases
C = "customer_id"
K = [C, "transaction_id", "date"]
P = [C, "product_id"]
G = [C, "department", "category"]


def _compute_baskets(line: pl.LazyFrame, out: Path) -> pl.LazyFrame:
    """Compute basket-level aggregations per customer per transaction.

    Args:
        line: Line-level transaction data.
        out: Output directory for intermediate files.

    Returns:
        LazyFrame with basket aggregations (spend, products, breadth).
    """
    _sink(
        line.group_by(K).agg(
            pl.col("value").sum().alias("spend"),
            pl.col("product_id").n_unique().alias("products"),
            pl.col("category_key").n_unique().alias("breadth"),
        ),
        out / "baskets.parquet",
        sort_by=[C, "date", "transaction_id"],
    )
    return pl.scan_parquet(out / "baskets.parquet")


def _sink(frame: pl.LazyFrame, path: Path, sort_by: list[str] | str | None = None) -> None:
    """Write a LazyFrame to Parquet with ZSTD compression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sort_by is not None:
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        frame = frame.sort(sort_by)
    frame.collect(engine="streaming").write_parquet(path, compression="zstd")


def compute_base_features(
    basket: pl.LazyFrame, d: date, end: date, observation_start: date | None
) -> tuple[pl.LazyFrame, pl.Expr, pl.Expr, pl.Expr, date, date, date]:
    """Compute base customer-level features from baskets.

    Args:
        basket: Basket-level LazyFrame.
        d: Snapshot cutoff date.
        end: Observation end date.
        observation_start: First observed transaction date.

    Returns:
        Tuple of (base features frame, recent expr, old expr, year expr,
                 30-day start, 90-day start, 365-day start).
    """
    s30 = d - timedelta(days=WINDOW_RECENT_30 - 1)
    s90 = d - timedelta(days=WINDOW_RECENT - 1)
    s365 = d - timedelta(days=WINDOW_YEAR - 1)
    old0 = d - timedelta(days=WINDOW_PREVIOUS_START_OFFSET)
    old1 = d - timedelta(days=WINDOW_RECENT)

    recent: pl.Expr = pl.col("date") >= s90
    old: pl.Expr = pl.col("date").is_between(old0, old1)
    year: pl.Expr = pl.col("date") >= s365

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
        pl.col("spend").filter(old).len().alias("purchases_previous90d"),
        pl.col("spend").filter(old).sum().alias("spend_previous90d"),
        pl.col("spend").filter(year).mean().alias("aov_365d"),
        pl.col("spend").filter(year).median().alias("median_basket"),
        safe_ratio(pl.col("spend").filter(year).std(), pl.col("spend").filter(year).mean()).alias(
            "basket_cv"
        ),
        pl.col("breadth").filter(year).median().alias("median_breadth"),
        pl.col("breadth").filter(recent).mean().alias("breadth_recent"),
        pl.col("breadth").filter(old).mean().alias("breadth_previous"),
    )

    # Left-censoring flag
    if observation_start is not None:
        base = base.with_columns(
            (
                (pl.col("first_purchase") - pl.lit(observation_start)).dt.total_days() < CENSOR_DAYS
            ).alias("left_censored"),
        )
    else:
        base = base.with_columns(pl.lit(None).alias("left_censored"))

    return base, recent, old, year, s30, s90, s365


def compute_day_features(line: pl.LazyFrame, out: Path, customer_col: str = C) -> pl.LazyFrame:
    """Compute day-level features (gaps, overlaps, spend ratios).

    Args:
        line: Line-level transaction data.
        out: Output directory for intermediate files.
        customer_col: Customer ID column name.

    Returns:
        LazyFrame with day-level features.
    """
    day = line.group_by([customer_col, "date"]).agg(
        pl.col("value").sum().alias("day_spend"),
        pl.col("product_id").unique().alias("products"),
        pl.col("category_key").unique().alias("categories"),
    )
    day = day.with_columns(
        *[
            pl.col(x).shift(1).over(customer_col, order_by="date").alias("prev_" + x)
            for x in ["date", "products", "categories"]
        ],
        pl.col("day_spend")
        .shift(1)
        .rolling_median(5, min_samples=3)
        .over(customer_col, order_by="date")
        .alias("baseline_spend"),
    )
    day = day.with_columns(
        (pl.col("date") - pl.col("prev_date")).dt.total_days().alias("gap"),
        safe_ratio(pl.col("day_spend"), pl.col("baseline_spend")).alias("relative_spend"),
        *[
            safe_ratio(
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
        .over(customer_col, order_by="date")
        .alias("baseline_gap")
    )
    _sink(day, out / "days.parquet", sort_by=[customer_col, "date"])
    return pl.scan_parquet(out / "days.parquet")


def compute_rhythm_features(
    day: pl.LazyFrame,
    customer_col: str,
    year: pl.Expr,
    usable: pl.Expr,
    recent: pl.Expr,
    s30: date,
) -> pl.LazyFrame:
    """Compute rhythm features (gaps, overlaps, reactivation).

    Args:
        day: Day-level LazyFrame.
        customer_col: Customer ID column name.
        year: Year window expression.
        usable: Usable gaps expression.
        recent: Recent window expression.
        s30: 30-day window start date.

    Returns:
        LazyFrame with rhythm features per customer.
    """
    rhythm = day.group_by(customer_col).agg(
        pl.len().alias("days_history"),
        pl.col("date").filter(year).len().alias("days_365d"),
        pl.col("gap").filter(usable).count().alias("gap_count"),
        pl.col("gap").filter(usable).median().alias("median_gap"),
        safe_ratio(pl.col("gap").filter(usable).std(), pl.col("gap").filter(usable).mean()).alias(
            "gap_cv"
        ),
        pl.col("products_overlap").filter(usable).median().alias("product_overlap"),
        pl.col("categories_overlap").filter(usable).median().alias("category_overlap"),
        (pl.col("date").dt.weekday() >= 6).filter(year).mean().alias("weekend_share"),
        pl.col("relative_spend").filter(recent).median().alias("personal_spend_ratio"),
        ((pl.col("date") >= s30) & (safe_ratio(pl.col("gap"), pl.col("baseline_gap")) >= 2))
        .fill_null(False)
        .sum()
        .alias("reactivation_events"),
    )
    return rhythm


def compute_weekday_entropy(day: pl.LazyFrame, customer_col: str, year: pl.Expr) -> pl.LazyFrame:
    """Compute weekday entropy for purchase timing regularity.

    Args:
        day: Day-level LazyFrame.
        customer_col: Customer ID column name.
        year: Year window expression.

    Returns:
        LazyFrame with weekday entropy per customer.
    """
    weekdays = (
        day.filter(year)
        .group_by([customer_col, pl.col("date").dt.weekday().alias("weekday")])
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum().over(customer_col)).alias("p"))
    )
    return weekdays.group_by(customer_col).agg(
        (-(pl.col("p") * pl.col("p").log())).filter(pl.col("p") > 0).sum().alias("weekday_entropy")
    )


def compute_breadth_last6days(
    line: pl.LazyFrame, day: pl.LazyFrame, customer_col: str
) -> pl.LazyFrame:
    """Compute product/category breadth in last 6 shopping days.

    Args:
        line: Line-level transaction data.
        day: Day-level LazyFrame.
        customer_col: Customer ID column name.

    Returns:
        LazyFrame with breadth features per customer.
    """
    last6 = (
        day.with_columns(
            pl.col("date")
            .cum_count()
            .over(customer_col, order_by="date", descending=True)
            .alias("rank")
        )
        .filter(pl.col("rank") <= 6)
        .select(customer_col, "date")
    )
    return (
        line.join(last6, on=[customer_col, "date"], how="inner", validate="m:1")
        .group_by(customer_col)
        .agg(
            pl.col("product_id").n_unique().alias("products_last6days"),
            pl.col("category_key").n_unique().alias("categories_last6days"),
        )
    )


def compute_product_features(
    line: pl.LazyFrame, out: Path, product_keys: list[str], year: pl.Expr, d: date
) -> pl.LazyFrame:
    """Compute product-level features (trial, repeat, conversion).

    Args:
        line: Line-level transaction data.
        out: Output directory for intermediate files.
        product_keys: Product grouping keys [customer_id, product_id].
        year: Year window expression.
        d: Snapshot cutoff date.

    Returns:
        LazyFrame with trial conversion features per customer.
    """
    pd = line.group_by(product_keys + ["date"]).agg(
        pl.col("quantity").sum().alias("qty"), pl.col("value").sum().alias("spend")
    )
    pd = pd.with_columns(
        (pl.col("date").cum_count().over(product_keys, order_by="date") - 1).alias("prior_days"),
        pl.col("date").shift(-1).over(product_keys, order_by="date").alias("next_date"),
        pl.col("qty")
        .shift(1)
        .rolling_median(5, min_samples=3)
        .over(product_keys, order_by="date")
        .alias("baseline_qty"),
    )
    pd = pd.with_columns(
        (pl.col("next_date") - pl.col("date")).dt.total_days().alias("next_gap"),
        safe_ratio(pl.col("qty"), pl.col("baseline_qty")).alias("relative_qty"),
    )
    _sink(pd, out / "product_days.parquet", sort_by=[C, "product_id", "date"])
    pd = pl.scan_parquet(out / "product_days.parquet")

    ph = pd.group_by(product_keys).agg(
        pl.col("date").min().alias("trial"),
        pl.col("date").filter(pl.col("prior_days") == 1).min().alias("repeat"),
        pl.len().alias("purchase_days"),
        pl.col("spend").filter(year).sum().alias("year_spend"),
    )
    _sink(ph, out / "product_history.parquet", sort_by=product_keys)

    eligible = pl.col("trial").is_between(
        d - timedelta(days=WINDOW_YEAR - 1), d - timedelta(days=90)
    )
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
    return trials


def compute_habit_features(
    line: pl.LazyFrame,
    pd: pl.LazyFrame,
    basket_keys: list[str],
    product_keys: list[str],
    year: pl.Expr,
    customer_col: str,
) -> pl.LazyFrame:
    """Compute habit features (new/repeat/established share).

    Args:
        line: Line-level transaction data.
        pd: Product-day LazyFrame.
        basket_keys: Basket grouping keys [customer_id, transaction_id, date].
        product_keys: Product grouping keys [customer_id, product_id].
        year: Year window expression.
        customer_col: Customer ID column name.

    Returns:
        LazyFrame with habit features per customer.
    """
    product_baskets = (
        line.select(basket_keys + ["product_id"])
        .unique()
        .join(
            pd.select(product_keys + ["date", "prior_days"]),
            on=product_keys + ["date"],
            how="left",
            validate="m:1",
        )
        .group_by(basket_keys)
        .agg(
            (pl.col("prior_days") == 0).mean().alias("new_share"),
            (pl.col("prior_days") >= 1).mean().alias("repeat_share"),
            (pl.col("prior_days") >= 2).mean().alias("established_share"),
        )
    )
    return product_baskets.group_by(customer_col).agg(
        pl.col("new_share").filter(year).mean().alias("new_share"),
        pl.col("repeat_share").filter(year).mean().alias("repeat_share"),
        pl.col("established_share").filter(year).mean().alias("established_share"),
    )


def compute_quantity_features(pd: pl.LazyFrame, customer_col: str, year: pl.Expr) -> pl.LazyFrame:
    """Compute quantity-based gap features.

    Args:
        pd: Product-day LazyFrame.
        customer_col: Customer ID column name.
        year: Year window expression.

    Returns:
        LazyFrame with quantity gap features per customer.
    """
    return pd.group_by(customer_col).agg(
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


def compute_category_features(
    line: pl.LazyFrame,
    category_keys: list[str],
    customer_col: str,
    year: pl.Expr,
    d: date,
    out: Path,
) -> pl.LazyFrame:
    """Compute category-level features (spend, gaps, overdue).

    Args:
        line: Line-level transaction data.
        category_keys: Category grouping keys [customer_id, department, category].
        customer_col: Customer ID column name.
        year: Year window expression.
        d: Snapshot cutoff date.
        out: Output directory for intermediate files.

    Returns:
        LazyFrame with category history per customer-department-category.
    """
    cd = (
        line.group_by(category_keys + ["date"])
        .agg(pl.col("value").sum().alias("spend"))
        .with_columns(pl.col("date").shift(1).over(category_keys, order_by="date").alias("prev"))
    )
    cd = cd.with_columns((pl.col("date") - pl.col("prev")).dt.total_days().alias("gap"))

    ch = cd.group_by(category_keys).agg(
        pl.col("spend").filter(year).sum().alias("year_spend"),
        pl.len().alias("days"),
        pl.col("date").filter(year).len().alias("year_days"),
        pl.col("date").max().alias("last"),
        pl.col("gap").median().alias("median_gap"),
    )
    ch = ch.with_columns(
        safe_ratio(pl.col("year_spend"), pl.col("year_spend").sum().over(customer_col)).alias(
            "share"
        ),
        safe_ratio((pl.lit(d) - pl.col("last")).dt.total_days(), pl.col("median_gap")).alias(
            "lateness"
        ),
    )
    _sink(ch, out / "category_history.parquet", sort_by=category_keys)

    return ch


def compute_category_aggregations(
    ch: pl.LazyFrame, customer_col: str
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Compute category-level aggregations per customer.

    Args:
        ch: Category history LazyFrame.
        customer_col: Customer ID column name.

    Returns:
        Tuple of (category aggregations, dominant category).
    """
    categories = ch.group_by(customer_col).agg(
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
        .sort(
            [customer_col, "year_spend", "department", "category"],
            descending=[False, True, False, False],
        )
        .group_by(customer_col, maintain_order=True)
        .agg(
            pl.col("department").first().alias("dominant_department"),
            pl.col("category").first().alias("dominant_category"),
        )
    )
    return categories, dominant


def compute_price_features(
    line: pl.LazyFrame, root: Path, d: date, customer_col: str, year: pl.Expr
) -> pl.LazyFrame:
    """Compute relative price features vs reference prices.

    Args:
        line: Line-level transaction data.
        root: Root output directory.
        d: Snapshot cutoff date.
        customer_col: Customer ID column name.
        year: Year window expression.

    Returns:
        LazyFrame with price features per customer.
    """
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
            .then(safe_ratio(pl.col("unit_price"), pl.col("reference_price")))
            .otherwise(None)
            .alias("relative_price")
        )
    )
    return prices.group_by(customer_col).agg(
        pl.col("relative_price").median().alias("relative_paid_price"),
        pl.col("relative_price").count().alias("comparable_price_lines"),
    )


def merge_features(
    base: pl.LazyFrame,
    *extras: pl.LazyFrame,
    customer_col: str = C,
) -> pl.LazyFrame:
    """Merge all feature frames on customer_id.

    Args:
        base: Base features frame.
        *extras: Additional feature frames to merge.
        customer_col: Customer ID column name.

    Returns:
        Merged LazyFrame.
    """
    frame = base
    for extra in extras:
        frame = frame.join(extra, on=customer_col, how="left", validate="1:1")
    return frame


def add_derived_features(frame: pl.LazyFrame, d: date, end: date) -> pl.LazyFrame:
    """Add derived features: tenure, recency, AOV, changes, etc.

    Args:
        frame: Merged features frame.
        d: Snapshot cutoff date.
        end: Observation end date.

    Returns:
        Frame with derived features added.
    """
    frame = frame.with_columns(
        pl.lit(d).alias("snapshot"),
        (pl.lit(d) - pl.col("first_purchase")).dt.total_days().alias("tenure"),
        (pl.lit(d) - pl.col("last_purchase")).dt.total_days().alias("recency"),
        safe_ratio(pl.col("spend_90d"), pl.col("purchases_90d")).alias("aov_recent"),
        safe_ratio(pl.col("spend_previous90d"), pl.col("purchases_previous90d")).alias(
            "aov_previous"
        ),
        safe_ratio(pl.col("repeated_trials"), pl.col("eligible_trials")).alias("trial_conversion"),
        safe_ratio(pl.col("core_spend"), pl.col("spend_365d")).alias("core_spend_share"),
    )
    frame = frame.with_columns(
        safe_ratio(pl.col("recency"), pl.col("median_gap")).alias("lateness"),
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
            .then(safe_ratio(pl.col(a), pl.col(b)) - 1)
            .otherwise(None)
            .alias(name)
            for a, b, name in [
                ("purchases_90d", "purchases_previous90d", "frequency_change"),
                ("aov_recent", "aov_previous", "basket_change"),
                ("spend_90d", "spend_previous90d", "spend_change"),
            ]
        ],
        pl.lit("reconstructed-1.3").alias("version"),
    )
    return frame


def add_future_outcomes(
    frame: pl.LazyFrame,
    path: Path,
    d: date,
    end: date,
    coverage_confirmed: bool,
) -> pl.LazyFrame:
    """Add future outcome columns if coverage confirmed and window observed.

    Args:
        frame: Features frame.
        path: Path to validated parquet data.
        d: Snapshot cutoff date.
        end: Observation end date.
        coverage_confirmed: Whether future follow-up data is complete.

    Returns:
        Frame with future outcome columns added.
    """
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
    return frame
