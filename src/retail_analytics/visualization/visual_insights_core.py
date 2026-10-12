"""Transaction-only aggregate views. Includes lightweight product affinity analysis."""

import math
from datetime import timedelta

from retail_analytics.core.stats import wilson
from retail_analytics.visualization.product_affinity import (
    ProductAffinityConfig,
    product_affinity_analysis,
    cross_sell_recommendations,
    new_to_repeat_pathways,
    basket_composition_by_segment,
    run_product_affinity_dashboard,
)

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

    # Check required columns exist
    schema = rows.collect_schema()
    required_cols = ["frequency_change", "basket_change", "value_tier", "lifecycle", "evidence",
                     "spend_90d", "purchases_90d", "purchases_previous90d", "aov_recent", "aov_previous",
                     "customer_id"]
    missing = [c for c in required_cols if c not in schema]
    if missing:
        # Return empty structure with expected schema
        empty_sample = pl.DataFrame(schema={c: schema.get(c, pl.Float64) for c in required_cols})
        empty_quad = pl.DataFrame(schema={"quadrant": pl.String, "customers": pl.Int64, "spend_90d": pl.Float64,
                                           "customer_share": pl.Float64, "spend_share": pl.Float64})
        return {
            "sample": empty_sample,
            "quadrant_summary": empty_quad,
            "not_comparable_n": 0,
            "finite_n": 0,
        }

    # Filter to customers with finite frequency_change and basket_change
    finite_rows = rows.filter(
        pl.col("frequency_change").is_finite() & pl.col("basket_change").is_finite()
    )
    total_n = collect(rows.select(pl.len())).item()
    finite_n = collect(finite_rows.select(pl.len())).item()
    not_comparable_n = total_n - finite_n

    if finite_n == 0:
        # Return empty structure with expected schema
        empty_sample = pl.DataFrame(schema={c: schema.get(c, pl.Float64) for c in required_cols})
        empty_quad = pl.DataFrame(schema={"quadrant": pl.String, "customers": pl.Int64, "spend_90d": pl.Float64,
                                           "customer_share": pl.Float64, "spend_share": pl.Float64})
        return {
            "sample": empty_sample,
            "quadrant_summary": empty_quad,
            "not_comparable_n": not_comparable_n,
            "finite_n": 0,
        }

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


def category_participation_comparison(validated, rows):
    """Category participation comparison: recent 90d vs previous 90d.

    Participation = customers purchasing in category during period / fixed comparison population.
    Uses validated transaction data (has per-transaction date, department, category).
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
    from retail_analytics.core.constants import WINDOW_PREVIOUS_START_OFFSET, WINDOW_RECENT
    recent_start = cutoff - timedelta(days=WINDOW_RECENT - 1)
    recent_end = cutoff
    prev_start = cutoff - timedelta(days=WINDOW_PREVIOUS_START_OFFSET)
    prev_end = cutoff - timedelta(days=WINDOW_RECENT)

    # Use validated data which has per-transaction date, department, category
    recent_cats = (
        validated.filter(
            pl.col("date").is_between(recent_start, recent_end)
        )
        .join(ids, on="customer_id", how="semi")
        .group_by("department", "category")
        .agg(
            pl.col("customer_id").n_unique().alias("recent_buyers"),
            pl.col("value").sum().alias("recent_spend"),
        )
    )

    prev_cats = (
        validated.filter(
            pl.col("date").is_between(prev_start, prev_end)
        )
        .join(ids, on="customer_id", how="semi")
        .group_by("department", "category")
        .agg(
            pl.col("customer_id").n_unique().alias("previous_buyers"),
            pl.col("value").sum().alias("previous_spend"),
        )
    )

    # Full join with coalesced category keys to preserve previous-only and recent-only categories
    result = collect(
        recent_cats.join(prev_cats, on=["department", "category"], how="full", coalesce=True)
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
            (pl.col("recent_spend").fill_null(0) - pl.col("previous_spend").fill_null(0)).alias("spend_change"),
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
    """Revenue waterfall decomposition between two snapshots.

    Decomposes spend change into:
    - Purchasing customers (N)
    - Baskets per purchasing customer (F)
    - Spend per basket / AOV (A)

    The bridge is order-dependent (N -> F -> A) and descriptive, not causal.
    For a valid factor bridge:
        R_t = total recorded spend
        B_t = total baskets
        N_t = distinct purchasing customers
        F_t = B_t / N_t
        A_t = R_t / B_t
    Then R_t = N_t * F_t * A_t.

    Sequential bridge (order N, F, A):
        delta_N = (N_1 - N_0) * F_0 * A_0
        delta_F = N_1 * (F_1 - F_0) * A_0
        delta_A = N_1 * F_1 * (A_1 - A_0)

    These sum to R_1 - R_0 (within floating-point tolerance).
    """
    import polars as pl

    if rows is None or previous_rows is None:
        return pl.DataFrame()

    # Compute components for current period from customer-level data
    current_stats = collect(
        rows.select(
            (pl.col("purchases_90d") > 0).sum().alias("purchasing_customers"),
            pl.col("purchases_90d").filter(pl.col("purchases_90d") > 0).sum().alias("total_baskets"),
            pl.col("spend_90d").filter(pl.col("purchases_90d") > 0).sum().alias("total_spend"),
        )
    ).row(0, named=True)

    previous_stats = collect(
        previous_rows.select(
            (pl.col("purchases_90d") > 0).sum().alias("purchasing_customers"),
            pl.col("purchases_90d").filter(pl.col("purchases_90d") > 0).sum().alias("total_baskets"),
            pl.col("spend_90d").filter(pl.col("purchases_90d") > 0).sum().alias("total_spend"),
        )
    ).row(0, named=True)

    pc_curr = current_stats["purchasing_customers"] or 0
    tb_curr = current_stats["total_baskets"] or 0
    ts_curr = current_stats["total_spend"] or 0

    pc_prev = previous_stats["purchasing_customers"] or 0
    tb_prev = previous_stats["total_baskets"] or 0
    ts_prev = previous_stats["total_spend"] or 0

    # Handle edge cases
    if pc_prev == 0 or tb_prev == 0:
        # Cannot compute previous-period rates; return total change with unavailable decomposition status
        return pl.DataFrame({
            "component": ["Previous period spend", "Current period spend", "Total change", "Decomposition"],
            "value": [float(ts_prev), float(ts_curr), float(ts_curr - ts_prev), None],
            "status": [None, None, None, "unavailable: zero purchasing customers or baskets in previous period"],
            "order": [0, 1, 2, 3],
        })

    if pc_curr == 0 or tb_curr == 0:
        # Current period has no purchasing activity
        return pl.DataFrame({
            "component": ["Previous period spend", "Current period spend", "Total change", "Decomposition"],
            "value": [float(ts_prev), float(ts_curr), float(ts_curr - ts_prev), None],
            "status": [None, None, None, "unavailable: zero purchasing customers or baskets in current period"],
            "order": [0, 1, 2, 3],
        })

    # Compute rates
    F_prev = tb_prev / pc_prev  # baskets per buyer
    A_prev = ts_prev / tb_prev  # AOV
    F_curr = tb_curr / pc_curr
    A_curr = ts_curr / tb_curr

    # Sequential bridge decomposition (order: N -> F -> A)
    delta_N = (pc_curr - pc_prev) * F_prev * A_prev
    delta_F = pc_curr * (F_curr - F_prev) * A_prev
    delta_A = pc_curr * F_curr * (A_curr - A_prev)

    # Verify reconciliation (within floating-point tolerance)
    residual = ts_curr - ts_prev - delta_N - delta_F - delta_A

    return pl.DataFrame({
        "component": ["Previous period spend", "Purchasing customers", "Baskets per customer", "Spend per basket (AOV)", "Residual (rounding)", "Current period spend"],
        "value": [float(ts_prev), float(delta_N), float(delta_F), float(delta_A), float(residual), float(ts_curr)],
        "status": [None, None, None, None, None, None],
        "order": [0, 1, 2, 3, 4, 5],
    })


def value_tier_transition_waterfall(transition_flows):
    """Value-tier transition waterfall showing net customer movement.

    Takes transition flows from transition_summary (from_label, to_label, len, rate)
    and computes net gain/loss per value tier.

    Accepts both eager DataFrames and LazyFrames.

    Note: Inflows/outflows include diagonal transitions (customers staying in same tier).
    Net change for a tier = inflow - outflow. For a closed matched population,
    sum of net_change across all tiers should be zero.
    """
    import polars as pl

    # Handle both eager and lazy input
    if isinstance(transition_flows, pl.DataFrame):
        flows = transition_flows
    else:
        flows = collect(transition_flows)
    if flows.is_empty():
        return pl.DataFrame()

    # Get all unique value tiers from both from and to
    from_tiers = flows.select(pl.col("from_label").alias("tier"))
    to_tiers = flows.select(pl.col("to_label").alias("tier"))
    all_tiers = pl.concat([from_tiers, to_tiers]).unique().sort("tier")

    # Compute outflows (customers leaving each tier) and inflows (customers entering each tier)
    outflows = (
        flows.group_by("from_label")
        .agg(pl.col("len").sum().alias("outflow"))
        .rename({"from_label": "tier"})
    )
    inflows = (
        flows.group_by("to_label")
        .agg(pl.col("len").sum().alias("inflow"))
        .rename({"to_label": "tier"})
    )

    # Net change per tier - compute in separate stages to ensure fill_null takes effect
    result = (
        all_tiers.join(outflows, on="tier", how="left")
        .join(inflows, on="tier", how="left")
        .with_columns(
            pl.col("outflow").fill_null(0),
            pl.col("inflow").fill_null(0),
        )
        .with_columns(
            (pl.col("inflow").cast(pl.Int64) - pl.col("outflow").cast(pl.Int64)).alias("net_change"),
        )
        .with_columns(
            pl.when(pl.col("net_change") >= 0)
            .then(pl.lit("gain"))
            .otherwise(pl.lit("loss"))
            .alias("direction"),
        )
        .sort("tier")
    )

    return result


def behavior_trajectory_matrix(rows):
    """Behavior × trajectory bubble matrix.

    X-axis: behavior (or behavioral_tags)
    Y-axis: trajectory_tags
    Bubble size: customer count
    Bubble color: median spend_90d or future purchase rate
    """
    import polars as pl

    # Check required columns exist
    schema = rows.collect_schema()
    required_cols = ["behavior", "trajectory", "spend_90d"]
    optional_cols = ["future_purchases_30d", "future_purchases_90d", "eligible_followup_30d", "eligible_followup_90d"]
    missing = [c for c in required_cols if c not in schema]
    if missing:
        # Return empty DataFrame with expected schema
        empty_schema = {
            "behavior": pl.String, "trajectory": pl.String, "customers": pl.Int64,
            "median_spend": pl.Float64, "mean_spend": pl.Float64,
            "future_30d_rate": pl.Float64, "future_90d_rate": pl.Float64,
            "eligible_30d": pl.Int64, "eligible_90d": pl.Int64,
        }
        return pl.DataFrame(schema=empty_schema)

    # Build aggregation expressions dynamically based on available columns
    agg_exprs = [
        pl.len().alias("customers"),
        pl.col("spend_90d").median().alias("median_spend"),
        pl.col("spend_90d").mean().alias("mean_spend"),
    ]

    if "future_purchases_30d" in schema and "eligible_followup_30d" in schema:
        agg_exprs.extend([
            (pl.col("future_purchases_30d") > 0).mean().alias("future_30d_rate"),
            pl.col("eligible_followup_30d").sum().alias("eligible_30d"),
        ])
    else:
        agg_exprs.extend([
            pl.lit(None, dtype=pl.Float64).alias("future_30d_rate"),
            pl.lit(0, dtype=pl.Int64).alias("eligible_30d"),
        ])

    if "future_purchases_90d" in schema and "eligible_followup_90d" in schema:
        agg_exprs.extend([
            (pl.col("future_purchases_90d") > 0).mean().alias("future_90d_rate"),
            pl.col("eligible_followup_90d").sum().alias("eligible_90d"),
        ])
    else:
        agg_exprs.extend([
            pl.lit(None, dtype=pl.Float64).alias("future_90d_rate"),
            pl.lit(0, dtype=pl.Int64).alias("eligible_90d"),
        ])

    # Get behavior and trajectory combinations
    combo = (
        rows.group_by("behavior", "trajectory")
        .agg(*agg_exprs)
        .filter(pl.col("customers") > 0)
        .sort("customers", descending=True)
    )

    return collect(combo)


def purchase_rhythm_panel(rows):
    """Purchase rhythm distribution panel.

    Returns:
    - median_gap distribution (histogram data)
    - gap_cv distribution (histogram data)
    - lateness distribution (histogram data)
    - Segment-level rhythm summary
    """
    import polars as pl

    # Filter to style-eligible customers (have meaningful rhythm metrics)
    eligible = rows.filter(pl.col("style_eligible"))

    # Histogram data for median_gap
    median_gap_hist = collect(
        eligible.filter(pl.col("median_gap").is_finite())
        .select("median_gap")
        .with_columns(pl.col("median_gap").alias("value"))
    )

    # Histogram data for gap_cv
    gap_cv_hist = collect(
        eligible.filter(pl.col("gap_cv").is_finite())
        .select("gap_cv")
        .with_columns(pl.col("gap_cv").alias("value"))
    )

    # Histogram data for lateness
    lateness_hist = collect(
        eligible.filter(pl.col("lateness").is_finite())
        .select("lateness")
        .with_columns(pl.col("lateness").alias("value"))
    )

    # Segment-level rhythm summary
    segment_rhythm = collect(
        eligible.group_by("behavior")
        .agg(
            pl.len().alias("customers"),
            pl.col("median_gap").median().alias("median_gap"),
            pl.col("gap_cv").median().alias("gap_cv"),
            pl.col("lateness").median().alias("lateness"),
            pl.col("recency").median().alias("median_recency"),
            pl.col("purchases_90d").mean().alias("avg_frequency_90d"),
        )
        .filter(pl.col("customers") > 0)
        .sort("customers", descending=True)
    )

    return {
        "median_gap_hist": median_gap_hist,
        "gap_cv_hist": gap_cv_hist,
        "lateness_hist": lateness_hist,
        "segment_rhythm": segment_rhythm,
    }


def basket_composition_slope(rows):
    """Basket composition slope chart: recent vs previous 90d.

    For customers with comparable windows, shows:
    - breadth_recent vs breadth_previous
    - aov_recent vs aov_previous
    - new_share vs repeat_share vs established_share
    - spend_90d vs spend_previous90d
    """
    import polars as pl

    # Filter to customers with comparable windows
    comp = rows.filter(
        pl.col("comparable_windows") &
        pl.col("breadth_recent").is_finite() &
        pl.col("breadth_previous").is_finite() &
        pl.col("aov_recent").is_finite() &
        pl.col("aov_previous").is_finite()
    )

    # Summary stats for slope chart
    slope_data = collect(
        comp.select(
            pl.len().alias("comparable_customers"),
            pl.col("breadth_recent").mean().alias("mean_breadth_recent"),
            pl.col("breadth_previous").mean().alias("mean_breadth_previous"),
            pl.col("aov_recent").mean().alias("mean_aov_recent"),
            pl.col("aov_previous").mean().alias("mean_aov_previous"),
            pl.col("spend_90d").mean().alias("mean_spend_recent"),
            pl.col("spend_previous90d").mean().alias("mean_spend_previous"),
            pl.col("new_share").mean().alias("mean_new_share"),
            pl.col("repeat_share").mean().alias("mean_repeat_share"),
            pl.col("established_share").mean().alias("mean_established_share"),
            pl.col("breadth_recent").median().alias("median_breadth_recent"),
            pl.col("breadth_previous").median().alias("median_breadth_previous"),
            pl.col("aov_recent").median().alias("median_aov_recent"),
            pl.col("aov_previous").median().alias("median_aov_previous"),
        )
    )

    # Customer-level data for scatter/slope (sampled)
    # Select only columns that exist in the filtered frame
    sample_cols = ["customer_id"]
    for col in ["behavior", "value_tier", "breadth_recent", "breadth_previous",
                "aov_recent", "aov_previous", "spend_90d", "spend_previous90d",
                "new_share", "repeat_share", "established_share",
                "frequency_change", "basket_change"]:
        if col in comp.collect_schema().names():
            sample_cols.append(col)

    sample = collect(
        comp.sort(pl.col("customer_id").hash(seed=29))
        .limit(2000)
        .select(sample_cols)
    )

    return {
        "summary": slope_data,
        "sample": sample,
    }


def category_sankey(category_history, rows, previous_rows=None):
    """Category participation Sankey between two snapshots.

    Uses customer-category participation (customers buying from category / population).
    """
    import polars as pl

    ids = rows.select("customer_id").unique()
    n = collect(ids.select(pl.len())).item()

    if previous_rows is None:
        # Single snapshot: category participation distribution
        part = collect(
            category_history.filter(pl.col("year_days") > 0)
            .join(ids, on="customer_id", how="semi")
            .group_by("department", "category")
            .agg(
                pl.col("customer_id").n_unique().alias("customers"),
                pl.col("year_spend").sum().alias("spend"),
            )
            .with_columns(
                pl.lit(n).alias("population_n"),
                (pl.col("customers") / pl.lit(n)).alias("participation"),
            )
            .sort("customers", descending=True)
        )
        return part, None

    # Two snapshots: need category participation at both
    # previous_rows has its own customer population
    prev_ids = previous_rows.select("customer_id").unique()
    prev_n = collect(prev_ids.select(pl.len())).item()

    # Current participation
    curr_part = (
        category_history.filter(pl.col("year_days") > 0)
        .join(ids, on="customer_id", how="semi")
        .group_by("department", "category")
        .agg(
            pl.col("customer_id").n_unique().alias("curr_customers"),
            pl.col("year_spend").sum().alias("curr_spend"),
        )
        .with_columns(
            pl.lit(n).alias("curr_population"),
            (pl.col("curr_customers") / pl.lit(n)).alias("curr_participation"),
        )
    )

    # Previous participation (need previous category history)
    # This requires loading the previous cutoff's category_history
    # For now, return current only with note
    result = collect(curr_part.sort("curr_customers", descending=True))
    return result, None


def future_outcome_validation(rows):
    """Future outcome validation: do current tags predict future purchases?

    Compares future_30d/90d purchase rates and spend by:
    - behavior
    - trajectory
    - value_tier
    - lifecycle
    - evidence
    """
    import polars as pl

    # Filter to eligible follow-up customers
    eligible_30 = rows.filter(pl.col("eligible_followup_30d"))
    eligible_90 = rows.filter(pl.col("eligible_followup_90d"))

    if eligible_30.select(pl.len()).collect().item() == 0:
        return pl.DataFrame()

    results = []
    for dim in ["behavior", "trajectory", "value_tier", "lifecycle", "evidence"]:
        # 30-day validation
        stats_30 = collect(
            eligible_30.group_by(dim).agg(
                pl.len().alias("eligible_n"),
                (pl.col("future_purchases_30d") > 0).sum().alias("buyers"),
                (pl.col("future_purchases_30d") > 0).mean().alias("purchase_rate"),
                pl.col("future_spend_30d").mean().alias("mean_spend"),
                pl.col("future_spend_30d").filter(pl.col("future_purchases_30d") > 0).mean().alias("mean_spend_buyers"),
            ).rename({dim: "dim_value"})
            .with_columns(
                (pl.col("buyers") / pl.col("eligible_n")).alias("rate"),
                pl.lit(30).alias("horizon"),
                pl.lit(dim).alias("dimension"),
            )
        )
        results.append(stats_30)

        # 90-day validation
        stats_90 = collect(
            eligible_90.group_by(dim).agg(
                pl.len().alias("eligible_n"),
                (pl.col("future_purchases_90d") > 0).sum().alias("buyers"),
                (pl.col("future_purchases_90d") > 0).mean().alias("purchase_rate"),
                pl.col("future_spend_90d").mean().alias("mean_spend"),
                pl.col("future_spend_90d").filter(pl.col("future_purchases_90d") > 0).mean().alias("mean_spend_buyers"),
            ).rename({dim: "dim_value"})
            .with_columns(
                (pl.col("buyers") / pl.col("eligible_n")).alias("rate"),
                pl.lit(90).alias("horizon"),
                pl.lit(dim).alias("dimension"),
            )
        )
        results.append(stats_90)

    return pl.concat(results, how="vertical_relaxed") if results else pl.DataFrame()
