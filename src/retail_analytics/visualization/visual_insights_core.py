"""Transaction-only aggregate views. No product-pair matrix or association mining."""

import math
from datetime import timedelta

from retail_analytics.core.stats import wilson

METRICS = {
    "spend_90d": ("Recorded spend, last 90 days", "currency units"),
    "purchases_90d": ("Baskets, last 90 days", "baskets"),
    "recency": ("Days since last recorded purchase", "days"),
    "median_basket": ("Median basket amount, last 365 days", "currency units"),
    "median_gap": ("Median completed shopping-day gap, last 365 days", "days"),
    "median_breadth": ("Median category breadth per basket, last 365 days", "categories"),
    "new_share": ("Mean unfamiliar-product share per basket, last 365 days", "%"),
    "repeat_share": ("Mean repeated-product share per basket, last 365 days", "%"),
    "established_share": ("Mean established-product share per basket, last 365 days", "%"),
    "category_hhi": ("Category spend concentration, last 365 days", "index"),
    "frequency_change": ("Relative change in purchase frequency (recent vs previous 90d)", "ratio"),
    "basket_change": ("Relative change in AOV (recent vs previous 90d)", "ratio"),
    "lateness": ("Recency / median gap (how overdue relative to cadence)", "ratio"),
}


def share(a, b):
    return a / b if b > 0 else None


def top_count(n, fraction):
    if n < 0 or not 0 < fraction <= 1:
        raise ValueError("Invalid population or fraction")
    return min(n, math.ceil(n * fraction))


def month_windows(first, last):
    cursor, result = first.replace(day=1), []
    while first <= last and cursor <= last:
        following = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        end = following - timedelta(days=1)
        lo, hi = max(first, cursor), min(last, end)
        result.append(dict(month=cursor, start=lo, end=hi, partial=lo != cursor or hi != end))
        cursor = following
    return result


def collect(frame):
    return frame.collect(engine="streaming")


def segment_summary(rows, dim, metric):
    import polars as pl

    if metric not in METRICS:
        raise ValueError("Unsupported metric")
    finite = pl.col(metric).filter(pl.col(metric).is_finite())
    data = rows.group_by(dim).agg(
        pl.len().alias("customers"),
        finite.count().alias("finite_n"),
        finite.quantile(0.25, interpolation="linear").alias("q25"),
        finite.median().alias("median"),
        finite.quantile(0.75, interpolation="linear").alias("q75"),
        pl.col("spend_90d").sum().alias("spend"),
        (pl.col("purchases_90d") > 0).sum().alias("active_n"),
        (pl.col("tenure") < 89).sum().alias("short_90d_history_n"),
        (pl.col("tenure") < 364).sum().alias("short_365d_history_n"),
    )
    return collect(
        data.with_columns(
            (pl.col("customers") - pl.col("finite_n")).alias("unavailable_n"),
            (pl.col("customers") / pl.col("customers").sum()).alias("customer_share"),
            pl.when(pl.col("spend").sum() > 0)
            .then(pl.col("spend") / pl.col("spend").sum())
            .otherwise(None)
            .alias("spend_share"),
            (pl.col("active_n") / pl.col("customers")).alias("active_share"),
        ).sort("customers", descending=True)
    )


def concentration(rows):
    import polars as pl

    values = rows.select(pl.col("spend_90d").cast(pl.Float64).alias("amount"))
    info = collect(
        values.select(
            pl.len().alias("n"),
            pl.col("amount").is_finite().sum().alias("finite"),
            (pl.col("amount") < 0).sum().alias("negative"),
        )
    ).row(0, named=True)
    n = info["n"]
    if info["finite"] != n or info["negative"]:
        raise ValueError("Concentration requires finite nonnegative spend")
    if not n:
        return dict(n=0, total=0.0, positive_n=0, curve=[], top=[])
    ranked = values.sort("amount", descending=True).with_row_index("rank", offset=1)
    fractions = [0.01, 0.05, 0.10, 0.20]
    stats = collect(
        ranked.select(
            pl.col("amount").sum().alias("total"),
            (pl.col("amount") > 0).sum().alias("positive_n"),
            *[
                pl.col("amount").filter(pl.col("rank") <= top_count(n, f)).sum().alias(f"top{i}")
                for i, f in enumerate(fractions)
            ],
        )
    ).row(0, named=True)
    top = [
        dict(
            requested_fraction=f,
            customers=top_count(n, f),
            actual_fraction=top_count(n, f) / n,
            spend_share=share(stats[f"top{i}"], stats["total"]),
        )
        for i, f in enumerate(fractions)
    ]
    curve = []
    if stats["total"] > 0:
        endpoint = ranked.with_columns(
            pl.col("amount").cum_sum().alias("cumulative"),
            (pl.col("rank") * 100 / n).ceil().cast(pl.Int32).alias("bin"),
        )
        data = collect(
            endpoint.group_by("bin")
            .agg(pl.col("rank").max(), pl.col("cumulative").max())
            .sort("rank")
        )
        curve = [(0.0, 0.0)] + [
            (r["rank"] / n, r["cumulative"] / stats["total"]) for r in data.to_dicts()
        ]
    return dict(n=n, total=stats["total"], positive_n=stats["positive_n"], curve=curve, top=top)


def monthly_summary(baskets, rows, first, cutoff, months=12):
    import polars as pl

    if not 1 <= months <= 36:
        raise ValueError("Months must be 1..36")
    windows = month_windows(first, cutoff)[-months:]
    calendar = pl.DataFrame(
        windows, schema={"month": pl.Date, "start": pl.Date, "end": pl.Date, "partial": pl.Boolean}
    )
    if not windows:
        return calendar
    ids = rows.select("customer_id").unique()
    n = collect(ids.select(pl.len())).item()
    selected = baskets.filter(pl.col("date").is_between(windows[0]["start"], cutoff)).join(
        ids, on="customer_id", how="semi"
    )
    summary = (
        selected.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(
            pl.col("spend").sum().alias("spend"),
            pl.len().alias("baskets"),
            pl.col("customer_id").n_unique().alias("buyers"),
        )
    )
    result = (
        calendar.lazy()
        .join(summary, on="month", how="left")
        .with_columns(
            pl.col("spend").fill_null(0.0),
            pl.col("baskets").fill_null(0),
            pl.col("buyers").fill_null(0),
            pl.lit(n).alias("population_n"),
        )
    )
    return collect(
        result.with_columns(
            pl.when(pl.col("baskets") > 0)
            .then(pl.col("spend") / pl.col("baskets"))
            .otherwise(None)
            .alias("amount_per_basket"),
            pl.when(pl.col("buyers") > 0)
            .then(pl.col("baskets") / pl.col("buyers"))
            .otherwise(None)
            .alias("baskets_per_buyer"),
            pl.when(pl.lit(n) > 0)
            .then(pl.col("buyers") / pl.lit(n))
            .otherwise(None)
            .alias("buying_share"),
        ).sort("month")
    )


def category_summary(history, rows):
    import polars as pl

    ids = rows.select("customer_id").unique()
    n = collect(ids.select(pl.len())).item()
    data = (
        history.filter(pl.col("year_days") > 0)
        .join(ids, on="customer_id", how="semi")
        .group_by("department", "category")
        .agg(
            pl.col("customer_id").n_unique().alias("customers"),
            pl.col("year_spend").sum().alias("spend"),
            pl.col("customer_id")
            .filter(pl.col("year_days") >= 3)
            .n_unique()
            .alias("recurring_customers"),
        )
    )
    return collect(
        data.with_columns(
            pl.lit(n).alias("population_n"),
            pl.when(pl.lit(n) > 0)
            .then(pl.col("customers") / pl.lit(n))
            .otherwise(None)
            .alias("participation"),
            pl.when(pl.col("spend").sum() > 0)
            .then(pl.col("spend") / pl.col("spend").sum())
            .otherwise(None)
            .alias("spend_share"),
            pl.concat_str([pl.col("department"), pl.lit(" / "), pl.col("category")]).alias("label"),
        ).sort("customers", descending=True)
    )


def cohort_summary(baskets, rows, cutoff, coverage, minimum=20):
    import polars as pl

    if minimum < 1:
        raise ValueError("Minimum must be positive")
    ids = (
        rows.select("customer_id", "first_purchase")
        .unique()
        .with_columns(pl.col("first_purchase").dt.truncate("1mo").alias("cohort"))
    )
    repeat = baskets.filter(pl.col("date") <= cutoff).join(
        ids.select("customer_id", "first_purchase"), on="customer_id", how="inner", validate="m:1"
    )
    repeat = (
        repeat.filter(pl.col("date") > pl.col("first_purchase"))
        .group_by("customer_id")
        .agg(pl.col("date").min().alias("repeat_date"))
    )
    profiles = ids.join(repeat, on="customer_id", how="left", validate="1:1")
    records = []
    for horizon in [30, 60, 90]:
        eligible = pl.col("first_purchase") + pl.duration(days=horizon) <= pl.lit(cutoff)
        success = (
            eligible
            & pl.col("repeat_date").is_not_null()
            & (pl.col("repeat_date") <= pl.col("first_purchase") + pl.duration(days=horizon))
        )
        data = collect(
            profiles.group_by("cohort").agg(
                pl.len().alias("cohort_n"),
                eligible.sum().alias("eligible_n"),
                success.fill_null(False).sum().alias("repeat_n"),
            )
        )
        for r in data.to_dicts():
            allowed = coverage and r["eligible_n"] >= minimum
            lo, hi = wilson(r["repeat_n"], r["eligible_n"]) if allowed else (None, None)
            r.update(
                horizon=horizon,
                rate=share(r["repeat_n"], r["eligible_n"]) if allowed else None,
                low=lo,
                high=hi,
                status="eligible"
                if allowed
                else "capture_unconfirmed"
                if not coverage
                else "insufficient_followup_or_n",
            )
            if not coverage:
                r["eligible_n"], r["repeat_n"] = None, None
            records.append(r)
    return pl.DataFrame(records).sort("cohort", "horizon") if records else pl.DataFrame()


def transition_summary(before, after, dim):
    import polars as pl

    origin = before.select("customer_id", pl.col(dim).alias("from_label"))
    destination = after.select("customer_id", pl.col(dim).alias("to_label"))
    joined = origin.join(destination, on="customer_id", how="inner", validate="1:1")
    n0, n1, matched = [collect(f.select(pl.len())).item() for f in [origin, destination, joined]]
    flows = collect(
        joined.group_by("from_label", "to_label")
        .len()
        .with_columns(pl.col("len").sum().over("from_label").alias("origin_matched_n"))
        .with_columns((pl.col("len") / pl.col("origin_matched_n")).alias("rate"))
        .sort("len", descending=True)
    )
    return flows, dict(
        origin_n=n0,
        destination_n=n1,
        matched_n=matched,
        origin_unmatched=n0 - matched,
        destination_unmatched=n1 - matched,
    )


def momentum_map(rows, sample_limit=2000):
    """Customer momentum map: frequency_change vs basket_change, colored by value_tier.

    Returns a DataFrame with customer-level data for scatter rendering, plus quadrant summary counts.
    Customers with unavailable (null) changes are excluded from scatter but counted separately.
    """
    import polars as pl

    # Filter to customers with finite frequency_change and basket_change
    finite_rows = rows.filter(
        pl.col("frequency_change").is_finite() & pl.col("basket_change").is_finite()
    )
    total_n = collect(rows.select(pl.len())).item()
    finite_n = collect(finite_rows.select(pl.len())).item()
    not_comparable_n = total_n - finite_n

    # Sample for rendering (deterministic hash-based sampling)
    sample = collect(
        finite_rows.sort(pl.col("customer_id").hash(seed=29))
        .limit(sample_limit)
        .select("customer_id", "frequency_change", "basket_change", "value_tier", "lifecycle", "evidence",
               "spend_90d", "purchases_90d", "purchases_previous90d", "aov_recent", "aov_previous")
    )

    # Quadrant classification for all finite rows
    quadrant_expr = (
        pl.when(pl.col("frequency_change") >= 0)
        .then(
            pl.when(pl.col("basket_change") >= 0)
            .then(pl.lit("growing_both"))
            .otherwise(pl.lit("more_frequent_smaller_baskets"))
        )
        .otherwise(
            pl.when(pl.col("basket_change") >= 0)
            .then(pl.lit("less_frequent_larger_baskets"))
            .otherwise(pl.lit("declining_both"))
        )
    )

    quadrant_summary = collect(
        finite_rows.with_columns(quadrant_expr.alias("quadrant"))
        .group_by("quadrant")
        .agg(
            pl.len().alias("customers"),
            pl.col("spend_90d").sum().alias("spend_90d"),
        )
        .with_columns(
            (pl.col("customers") / pl.col("customers").sum()).alias("customer_share"),
            pl.when(pl.col("spend_90d").sum() > 0)
            .then(pl.col("spend_90d") / pl.col("spend_90d").sum())
            .otherwise(None)
            .alias("spend_share"),
        )
    )

    return {
        "sample": sample,
        "quadrant_summary": quadrant_summary,
        "not_comparable_n": not_comparable_n,
        "finite_n": finite_n,
    }


def cadence_map(rows, sample_limit=2000, evidence_filter=None, value_tier_filter=None):
    """Purchase cadence map: median_gap vs recency with reference lines.

    Returns sampled customer data for 2D grid rendering, plus eligibility summary.
    """
    import polars as pl

    filtered = rows.filter(
        pl.col("median_gap").is_finite() &
        pl.col("recency").is_finite() &
        (pl.col("median_gap") > 0)
    )

    if evidence_filter:
        filtered = filtered.filter(pl.col("evidence").is_in(evidence_filter))
    if value_tier_filter:
        filtered = filtered.filter(pl.col("value_tier").is_in(value_tier_filter))

    # Eligibility: style_eligible customers (have enough history for median_gap)
    eligible_row = collect(filtered.select(
        pl.col("style_eligible").sum().alias("eligible_n"),
        pl.len().alias("total_n"),
    )).row(0, named=True)

    # Sample for rendering
    sample = collect(
        filtered.sort(pl.col("customer_id").hash(seed=29))
        .limit(sample_limit)
        .select("customer_id", "median_gap", "recency", "lateness", "value_tier", "evidence", "lifecycle",
               "gap_cv", "days_365d")
    )

    return {
        "sample": sample,
        "eligible_n": eligible_row["eligible_n"],
        "total_n": eligible_row["total_n"],
    }


def category_participation_comparison(history, rows):
    """Category participation comparison: recent 90d vs previous 90d.

    Participation = customers purchasing in category during period / fixed comparison population.
    Returns DataFrame with category-level participation in both periods, changes, and spend.
    """
    import polars as pl

    # Fixed population: all customers in the filtered population
    ids = rows.select("customer_id").unique()
    n = collect(ids.select(pl.len())).item()

    # Get cutoff from snapshot (assume latest snapshot in rows)
    cutoff = collect(rows.select(pl.col("snapshot").max())).item()

    if cutoff is None:
        return pl.DataFrame()

    # Recent 90d: [C-89, C], Previous 90d: [C-179, C-90]
    from retail_analytics.core.constants import WINDOW_RECENT, WINDOW_PREVIOUS_START_OFFSET
    recent_start = cutoff - timedelta(days=WINDOW_RECENT - 1)
    recent_end = cutoff
    prev_start = cutoff - timedelta(days=WINDOW_PREVIOUS_START_OFFSET)
    prev_end = cutoff - timedelta(days=WINDOW_RECENT)

    # Need category-history at day level to compute participation per period
    # history has per-category per-day spend; we need to count distinct customers per category per period
    # We'll use the original line data (but we only have category_history which is aggregated)

    # category_history has: customer_id, department, category, date (per day), spend
    # We can filter by date range and count distinct customers

    recent_cats = (
        history.filter(
            pl.col("date").is_between(recent_start, recent_end)
        )
        .join(ids, on="customer_id", how="semi")
        .group_by("department", "category")
        .agg(
            pl.col("customer_id").n_unique().alias("recent_buyers"),
            pl.col("spend").sum().alias("recent_spend"),
        )
    )

    prev_cats = (
        history.filter(
            pl.col("date").is_between(prev_start, prev_end)
        )
        .join(ids, on="customer_id", how="semi")
        .group_by("department", "category")
        .agg(
            pl.col("customer_id").n_unique().alias("previous_buyers"),
            pl.col("spend").sum().alias("previous_spend"),
        )
    )

    result = collect(
        recent_cats.join(prev_cats, on=["department", "category"], how="full")
        .with_columns(
            pl.lit(n).alias("population_n"),
            pl.when(pl.lit(n) > 0)
            .then(pl.col("recent_buyers").fill_null(0) / pl.lit(n))
            .otherwise(None)
            .alias("recent_participation"),
            pl.when(pl.lit(n) > 0)
            .then(pl.col("previous_buyers").fill_null(0) / pl.lit(n))
            .otherwise(None)
            .alias("previous_participation"),
            (pl.col("recent_buyers").fill_null(0) - pl.col("previous_buyers").fill_null(0)).alias("buyer_change"),
            pl.concat_str([pl.col("department"), pl.lit(" / "), pl.col("category")]).alias("label"),
        )
        .with_columns(
            (pl.col("recent_participation") - pl.col("previous_participation")).alias("participation_change_pp"),
        )
        .sort("recent_participation", descending=True)
    )

    return result


def trial_to_repeat_map(product_history, rows):
    """Product discovery-to-repeat map: category-level trial conversion.

    X: eligible trials (sum), Y: pooled trial conversion rate, bubble size: distinct trialling customers.
    """
    import polars as pl

    # product_history has: customer_id, product_id, trial, repeat, purchase_days, year_spend
    # eligible trials: trial in [C-364, C-90], repeat within 90d of trial
    cutoff = collect(rows.select(pl.col("snapshot").max())).item()

    if cutoff is None:
        return pl.DataFrame()

    from retail_analytics.core.constants import WINDOW_YEAR
    trial_start = cutoff - timedelta(days=WINDOW_YEAR - 1)
    trial_end = cutoff - timedelta(days=90)

    ids = rows.select("customer_id").unique()

    # Filter to eligible trials with enough observation time
    trials = (
        product_history.filter(
            pl.col("trial").is_between(trial_start, trial_end)
        )
        .join(ids, on="customer_id", how="semi")
    )

    # Join with product metadata to get category
    # We need department/category for each product - get from validated data or product_history doesn't have it
    # For now, we'll use a simplified approach: we need product -> category mapping
    # This would require joining with the original line data or a product lookup
    # Since product_history doesn't have category, we'll note this limitation
    # In practice, this needs a product_category lookup table

    # For now, return aggregated by product (could be extended with category lookup)
    result = collect(
        trials.group_by("product_id")
        .agg(
            pl.col("customer_id").n_unique().alias("trialling_customers"),
            pl.len().alias("eligible_trials"),
            (pl.col("repeat").is_not_null() & (pl.col("repeat") <= pl.col("trial") + pl.duration(days=90)))
            .fill_null(False)
            .sum()
            .alias("repeated_trials"),
        )
        .with_columns(
            (pl.col("repeated_trials") / pl.col("eligible_trials")).alias("trial_conversion"),
        )
        .filter(pl.col("eligible_trials") > 0)
        .sort("eligible_trials", descending=True)
    )

    return result


def revenue_waterfall(rows, previous_rows, cutoff):
    """Month-to-month revenue waterfall decomposition.

    Decomposes spend change into:
    - Purchasing customers
    - Baskets per purchasing customer
    - Spend per basket (AOV)
    """
    import polars as pl

    # This requires historical month data; for now, implement using recent vs previous 90d
    # as a two-period decomposition
    current = rows
    previous = previous_rows

    if current is None or previous is None:
        return pl.DataFrame()

    n_current = collect(current.select(pl.len())).item()
    n_previous = collect(previous.select(pl.len())).item()

    # Compute components for current period
    current_stats = collect(
        current.select(
            (pl.col("purchases_90d") > 0).sum().alias("purchasing_customers"),
            pl.col("purchases_90d").filter(pl.col("purchases_90d") > 0).mean().alias("baskets_per_buyer"),
            pl.col("spend_90d").filter(pl.col("purchases_90d") > 0).mean().alias("spend_per_basket"),
            pl.col("spend_90d").sum().alias("total_spend"),
        )
    ).row(0, named=True)

    previous_stats = collect(
        previous.select(
            (pl.col("purchases_90d") > 0).sum().alias("purchasing_customers"),
            pl.col("purchases_90d").filter(pl.col("purchases_90d") > 0).mean().alias("baskets_per_buyer"),
            pl.col("spend_90d").filter(pl.col("purchases_90d") > 0).mean().alias("spend_per_basket"),
            pl.col("spend_90d").sum().alias("total_spend"),
        )
    ).row(0, named=True)

    # Fixed-order bridge decomposition (customer count -> frequency -> AOV)
    pc_curr = current_stats["purchasing_customers"] or 0
    bpb_curr = current_stats["baskets_per_buyer"] or 0
    spb_curr = current_stats["spend_per_basket"] or 0
    pc_prev = previous_stats["purchasing_customers"] or 0
    bpb_prev = previous_stats["baskets_per_buyer"] or 0
    spb_prev = previous_stats["spend_per_basket"] or 0

    # Decomposition: spend = purchasing_customers * baskets_per_buyer * spend_per_basket
    # Contribution of each factor (holding others at previous level)
    spend_prev = previous_stats["total_spend"] or 0
    spend_curr = current_stats["total_spend"] or 0

    delta_pc = (pc_curr - pc_prev) * bpb_prev * spb_prev
    delta_bpb = pc_curr * (bpb_curr - bpb_prev) * spb_prev
    delta_spb = pc_curr * bpb_curr * (spb_curr - spb_prev)

    # Residual (interaction effects)
    residual = spend_curr - spend_prev - delta_pc - delta_bpb - delta_spb

    return pl.DataFrame([
        dict(component="Previous period spend", value=spend_prev, order=0),
        dict(component="Purchasing customers", value=delta_pc, order=1),
        dict(component="Baskets per customer", value=delta_bpb, order=2),
        dict(component="Spend per basket", value=delta_spb, order=3),
        dict(component="Residual (interactions)", value=residual, order=4),
        dict(component="Current period spend", value=spend_curr, order=5),
    ])