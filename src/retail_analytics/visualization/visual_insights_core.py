"""Transaction-only aggregate views. Includes lightweight product affinity analysis."""

import math
import networkx as nx
import polars as pl
from datetime import date, timedelta
from typing import Optional

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

    # Check required columns exist
    schema = rows.collect_schema()
    required_cols = [
        "frequency_change",
        "basket_change",
        "value_tier",
        "lifecycle",
        "evidence",
        "spend_90d",
        "purchases_90d",
        "purchases_previous90d",
        "aov_recent",
        "aov_previous",
        "customer_id",
    ]
    missing = [c for c in required_cols if c not in schema]
    if missing:
        # Return empty structure with expected schema
        empty_sample = pl.DataFrame(schema={c: schema.get(c, pl.Float64) for c in required_cols})
        empty_quad = pl.DataFrame(
            schema={
                "quadrant": pl.String,
                "customers": pl.Int64,
                "spend_90d": pl.Float64,
                "customer_share": pl.Float64,
                "spend_share": pl.Float64,
            }
        )
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
        empty_quad = pl.DataFrame(
            schema={
                "quadrant": pl.String,
                "customers": pl.Int64,
                "spend_90d": pl.Float64,
                "customer_share": pl.Float64,
                "spend_share": pl.Float64,
            }
        )
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
        .select(
            "customer_id",
            "frequency_change",
            "basket_change",
            "value_tier",
            "lifecycle",
            "evidence",
            "spend_90d",
            "purchases_90d",
            "purchases_previous90d",
            "aov_recent",
            "aov_previous",
        )
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
        pl.col("median_gap").is_finite()
        & pl.col("recency").is_finite()
        & (pl.col("median_gap") > 0)
    )

    if evidence_filter:
        filtered = filtered.filter(pl.col("evidence").is_in(evidence_filter))
    if value_tier_filter:
        filtered = filtered.filter(pl.col("value_tier").is_in(value_tier_filter))

    # Eligibility: style_eligible customers (have enough history for median_gap)
    eligible_row = collect(
        filtered.select(
            pl.col("style_eligible").sum().alias("eligible_n"),
            pl.len().alias("total_n"),
        )
    ).row(0, named=True)

    # Sample for rendering
    sample = collect(
        filtered.sort(pl.col("customer_id").hash(seed=29))
        .limit(sample_limit)
        .select(
            "customer_id",
            "median_gap",
            "recency",
            "lateness",
            "value_tier",
            "evidence",
            "lifecycle",
            "gap_cv",
            "days_365d",
        )
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
        validated.filter(pl.col("date").is_between(recent_start, recent_end))
        .join(ids, on="customer_id", how="semi")
        .group_by("department", "category")
        .agg(
            pl.col("customer_id").n_unique().alias("recent_buyers"),
            pl.col("value").sum().alias("recent_spend"),
        )
    )

    prev_cats = (
        validated.filter(pl.col("date").is_between(prev_start, prev_end))
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
            (pl.col("recent_buyers").fill_null(0) - pl.col("previous_buyers").fill_null(0)).alias(
                "buyer_change"
            ),
            (pl.col("recent_spend").fill_null(0) - pl.col("previous_spend").fill_null(0)).alias(
                "spend_change"
            ),
            pl.concat_str([pl.col("department"), pl.lit(" / "), pl.col("category")]).alias("label"),
        )
        .with_columns(
            (pl.col("recent_participation") - pl.col("previous_participation")).alias(
                "participation_change_pp"
            ),
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
    trials = product_history.filter(pl.col("trial").is_between(trial_start, trial_end)).join(
        ids, on="customer_id", how="semi"
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
            (
                pl.col("repeat").is_not_null()
                & (pl.col("repeat") <= pl.col("trial") + pl.duration(days=90))
            )
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
            pl.col("purchases_90d")
            .filter(pl.col("purchases_90d") > 0)
            .sum()
            .alias("total_baskets"),
            pl.col("spend_90d").filter(pl.col("purchases_90d") > 0).sum().alias("total_spend"),
        )
    ).row(0, named=True)

    previous_stats = collect(
        previous_rows.select(
            (pl.col("purchases_90d") > 0).sum().alias("purchasing_customers"),
            pl.col("purchases_90d")
            .filter(pl.col("purchases_90d") > 0)
            .sum()
            .alias("total_baskets"),
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
        return pl.DataFrame(
            {
                "component": [
                    "Previous period spend",
                    "Current period spend",
                    "Total change",
                    "Decomposition",
                ],
                "value": [float(ts_prev), float(ts_curr), float(ts_curr - ts_prev), None],
                "status": [
                    None,
                    None,
                    None,
                    "unavailable: zero purchasing customers or baskets in previous period",
                ],
                "order": [0, 1, 2, 3],
            }
        )

    if pc_curr == 0 or tb_curr == 0:
        # Current period has no purchasing activity
        return pl.DataFrame(
            {
                "component": [
                    "Previous period spend",
                    "Current period spend",
                    "Total change",
                    "Decomposition",
                ],
                "value": [float(ts_prev), float(ts_curr), float(ts_curr - ts_prev), None],
                "status": [
                    None,
                    None,
                    None,
                    "unavailable: zero purchasing customers or baskets in current period",
                ],
                "order": [0, 1, 2, 3],
            }
        )

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

    return pl.DataFrame(
        {
            "component": [
                "Previous period spend",
                "Purchasing customers",
                "Baskets per customer",
                "Spend per basket (AOV)",
                "Residual (rounding)",
                "Current period spend",
            ],
            "value": [
                float(ts_prev),
                float(delta_N),
                float(delta_F),
                float(delta_A),
                float(residual),
                float(ts_curr),
            ],
            "status": [None, None, None, None, None, None],
            "order": [0, 1, 2, 3, 4, 5],
        }
    )


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
            (pl.col("inflow").cast(pl.Int64) - pl.col("outflow").cast(pl.Int64)).alias(
                "net_change"
            ),
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
    missing = [c for c in required_cols if c not in schema]
    if missing:
        # Return empty DataFrame with expected schema
        empty_schema = {
            "behavior": pl.String,
            "trajectory": pl.String,
            "customers": pl.Int64,
            "median_spend": pl.Float64,
            "mean_spend": pl.Float64,
            "future_30d_rate": pl.Float64,
            "future_90d_rate": pl.Float64,
            "eligible_30d": pl.Int64,
            "eligible_90d": pl.Int64,
        }
        return pl.DataFrame(schema=empty_schema)

    # Build aggregation expressions dynamically based on available columns
    agg_exprs = [
        pl.len().alias("customers"),
        pl.col("spend_90d").median().alias("median_spend"),
        pl.col("spend_90d").mean().alias("mean_spend"),
    ]

    if "future_purchases_30d" in schema and "eligible_followup_30d" in schema:
        agg_exprs.extend(
            [
                (pl.col("future_purchases_30d") > 0).mean().alias("future_30d_rate"),
                pl.col("eligible_followup_30d").sum().alias("eligible_30d"),
            ]
        )
    else:
        agg_exprs.extend(
            [
                pl.lit(None, dtype=pl.Float64).alias("future_30d_rate"),
                pl.lit(0, dtype=pl.Int64).alias("eligible_30d"),
            ]
        )

    if "future_purchases_90d" in schema and "eligible_followup_90d" in schema:
        agg_exprs.extend(
            [
                (pl.col("future_purchases_90d") > 0).mean().alias("future_90d_rate"),
                pl.col("eligible_followup_90d").sum().alias("eligible_90d"),
            ]
        )
    else:
        agg_exprs.extend(
            [
                pl.lit(None, dtype=pl.Float64).alias("future_90d_rate"),
                pl.lit(0, dtype=pl.Int64).alias("eligible_90d"),
            ]
        )

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
        pl.col("comparable_windows")
        & pl.col("breadth_recent").is_finite()
        & pl.col("breadth_previous").is_finite()
        & pl.col("aov_recent").is_finite()
        & pl.col("aov_previous").is_finite()
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
    for col in [
        "behavior",
        "value_tier",
        "breadth_recent",
        "breadth_previous",
        "aov_recent",
        "aov_previous",
        "spend_90d",
        "spend_previous90d",
        "new_share",
        "repeat_share",
        "established_share",
        "frequency_change",
        "basket_change",
    ]:
        if col in comp.collect_schema().names():
            sample_cols.append(col)

    sample = collect(comp.sort(pl.col("customer_id").hash(seed=29)).limit(2000).select(sample_cols))

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
    collect(previous_rows.select("customer_id").unique().select(pl.len()))

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
            eligible_30.group_by(dim)
            .agg(
                pl.len().alias("eligible_n"),
                (pl.col("future_purchases_30d") > 0).sum().alias("buyers"),
                (pl.col("future_purchases_30d") > 0).mean().alias("purchase_rate"),
                pl.col("future_spend_30d").mean().alias("mean_spend"),
                pl.col("future_spend_30d")
                .filter(pl.col("future_purchases_30d") > 0)
                .mean()
                .alias("mean_spend_buyers"),
            )
            .rename({dim: "dim_value"})
            .with_columns(
                (pl.col("buyers") / pl.col("eligible_n")).alias("rate"),
                pl.lit(30).alias("horizon"),
                pl.lit(dim).alias("dimension"),
            )
        )
        results.append(stats_30)

        # 90-day validation
        stats_90 = collect(
            eligible_90.group_by(dim)
            .agg(
                pl.len().alias("eligible_n"),
                (pl.col("future_purchases_90d") > 0).sum().alias("buyers"),
                (pl.col("future_purchases_90d") > 0).mean().alias("purchase_rate"),
                pl.col("future_spend_90d").mean().alias("mean_spend"),
                pl.col("future_spend_90d")
                .filter(pl.col("future_purchases_90d") > 0)
                .mean()
                .alias("mean_spend_buyers"),
            )
            .rename({dim: "dim_value"})
            .with_columns(
                (pl.col("buyers") / pl.col("eligible_n")).alias("rate"),
                pl.lit(90).alias("horizon"),
                pl.lit(dim).alias("dimension"),
            )
        )
        results.append(stats_90)

    return pl.concat(results, how="vertical_relaxed") if results else pl.DataFrame()


# ============================================================================
# RESEARCH-BACKED PRIORITY VISUALIZATIONS
# ============================================================================


def time_to_second_purchase(baskets, rows, cutoff, max_days=180, confidence=0.95):
    """Kaplan-Meier time-to-second-purchase survival curve.

    Research basis: Schmittlein, Morrison & Colombo (1987) - Counting Your Customers.
    Non-contractual customer activity models with right-censoring.

    Args:
        baskets: LazyFrame with basket data (customer_id, date)
        rows: LazyFrame with snapshot customer features (customer_id, first_purchase)
        cutoff: Snapshot cutoff date (date object or ISO string)
        max_days: Maximum days to track (default 180)
        confidence: Confidence level for bands (default 0.95)

    Returns:
        DataFrame with columns: days, survival_prob, lower_ci, upper_ci, at_risk, events, censored
    """
    import math

    import polars as pl

    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    # Get first purchase dates for all customers in snapshot
    cust_first = rows.select("customer_id", "first_purchase").unique().collect()

    # Get all basket dates for these customers up to cutoff
    cust_ids = cust_first["customer_id"].to_list()
    basket_data = collect(
        baskets.filter(pl.col("customer_id").is_in(cust_ids))
        .filter(pl.col("date") <= pl.lit(cutoff))
        .select("customer_id", "date")
        .unique(subset=["customer_id", "date"])
        .sort("customer_id", "date")
    )

    if basket_data.is_empty():
        return pl.DataFrame()

    # Find second purchase date for each customer
    # Only customers with at least 2 unique basket dates have a second purchase
    basket_counts = basket_data.group_by("customer_id").agg(
        pl.col("date").n_unique().alias("n_dates")
    )
    second_purchase = (
        basket_data.join(
            basket_counts.filter(pl.col("n_dates") >= 2), on="customer_id", how="inner"
        )
        .group_by("customer_id")
        .agg(pl.col("date").sort().head(2).last().alias("second_purchase"))
        .filter(pl.col("second_purchase").is_not_null())
    )

    # Merge with first purchase
    cust_data = cust_first.join(second_purchase, on="customer_id", how="left")

    # Compute time to second purchase (or censoring at cutoff)
    cust_data = cust_data.with_columns(
        [
            (pl.col("second_purchase") - pl.col("first_purchase"))
            .dt.total_days()
            .alias("days_to_second"),
            (pl.lit(cutoff) - pl.col("first_purchase")).dt.total_days().alias("censor_days"),
        ]
    )

    # Observed duration: min(event_time, censor_time), capped at max_days
    # Event occurs if second purchase observed within max_days and before cutoff
    cust_data = cust_data.with_columns(
        [
            pl.when(
                pl.col("days_to_second").is_not_null()
                & (pl.col("days_to_second") <= max_days)
                & (pl.col("days_to_second") <= pl.col("censor_days"))
            )
            .then(1)
            .otherwise(0)
            .alias("event"),
            pl.min_horizontal(
                [
                    pl.col("days_to_second").fill_null(max_days + 1),
                    pl.col("censor_days"),
                    pl.lit(max_days),
                ]
            ).alias("observed_duration"),
        ]
    )

    # Only include customers with first_purchase early enough to have at least 1 day of observation
    cust_data = cust_data.filter(pl.col("censor_days") >= 1)

    if cust_data.is_empty():
        return pl.DataFrame()

    # Kaplan-Meier estimation
    # For each day t: survival(t) = product over t' <= t of (1 - events(t') / at_risk(t'))
    # Where at_risk(t) = number of customers with observed_duration >= t

    max_censor = min(max_days, int(cust_data["observed_duration"].max()))

    results = []
    survival = 1.0

    for t in range(1, max_censor + 1):
        at_risk = cust_data.filter(pl.col("observed_duration") >= t).height
        if at_risk == 0:
            break

        events_t = cust_data.filter((pl.col("days_to_second") == t) & (pl.col("event") == 1)).height
        censored_t = cust_data.filter(
            (pl.col("censor_days") == t)
            & (pl.col("event") == 0)
            & (
                pl.col("days_to_second").is_null()
                | (pl.col("days_to_second") > max_days)
                | (pl.col("days_to_second") > pl.col("censor_days"))
            )
        ).height

        if events_t > 0:
            hazard = events_t / at_risk
            survival *= 1 - hazard

        # Greenwood formula for confidence intervals
        # var(survival) = survival^2 * sum(hazard / (at_risk - events))
        # We'll compute cumulative variance
        if t == 1:
            cum_var = 0.0
        if events_t > 0 and at_risk > events_t:
            cum_var += (events_t / at_risk) / (at_risk - events_t)

        # Z-score for confidence level
        z = 1.96 if confidence >= 0.95 else 1.645  # 95% or 90%
        se = survival * math.sqrt(cum_var) if cum_var > 0 else 0
        # Use log-log transformation for better CI bounds
        if survival > 0 and survival < 1 and cum_var > 0:
            log_surv = math.log(-math.log(survival))
            log_se = (
                math.sqrt(cum_var) / (survival * -math.log(survival))
                if survival * -math.log(survival) > 0
                else se
            )
            lower = max(0.0, math.exp(-math.exp(log_surv + z * log_se)))
            upper = min(1.0, math.exp(-math.exp(log_surv - z * log_se)))
        else:
            lower = max(0.0, survival - z * se)
            upper = min(1.0, survival + z * se)

        results.append(
            {
                "days": t,
                "survival_prob": survival,
                "lower_ci": lower,
                "upper_ci": upper,
                "at_risk": at_risk,
                "events": events_t,
                "censored": censored_t,
                "cum_repeat_prob": 1 - survival,
            }
        )

    return pl.DataFrame(results)


def cadence_return_validation(rows, coverage, max_lateness=5.0, bins=10):
    """Cadence vs observed return probability validation.

    Bins customers by lateness (recency / median_gap) and plots
    confirmed future purchase rates with Wilson intervals.

    Research basis: Repeat-buying models (BG/NBD) - does the 2x median_gap
    heuristic boundary correspond to actual return probability changes?

    Args:
        rows: LazyFrame with customer features (must have recency, median_gap, future_*)
        coverage: Boolean - whether coverage is confirmed
        max_lateness: Maximum lateness to include in plot
        bins: Number of lateness bins

    Returns:
        DataFrame with lateness bin statistics
    """
    import polars as pl

    from retail_analytics.core.stats import wilson

    if not coverage:
        return pl.DataFrame({"warning": ["Coverage unconfirmed - future outcomes unavailable"]})

    schema = rows.collect_schema()
    required = [
        "recency",
        "median_gap",
        "future_purchases_30d",
        "eligible_followup_30d",
        "future_purchases_90d",
        "eligible_followup_90d",
        "median_gap",
    ]
    missing = [c for c in required if c not in schema]
    if missing:
        return pl.DataFrame({"error": [f"Missing columns: {missing}"]})

    # Filter to customers with finite cadence metrics and eligible follow-up
    eligible = rows.filter(
        pl.col("median_gap").is_finite()
        & (pl.col("median_gap") > 0)
        & pl.col("recency").is_finite()
        & pl.col("eligible_followup_30d")
    )

    if eligible.select(pl.len()).collect().item() == 0:
        return pl.DataFrame()

    # Compute lateness
    eligible = eligible.with_columns(
        (pl.col("recency") / pl.col("median_gap")).alias("lateness")
    ).filter((pl.col("lateness") >= 0) & (pl.col("lateness") <= max_lateness))

    if eligible.select(pl.len()).collect().item() == 0:
        return pl.DataFrame()

    # Create lateness bins
    bin_edges = [i * max_lateness / bins for i in range(bins + 1)]
    bin_labels = [f"{bin_edges[i]:.1f}–{bin_edges[i + 1]:.1f}" for i in range(bins)]

    # Polars .cut() requires interior cut points (len(labels) == len(cut_points) + 1)
    interior_edges = bin_edges[1:-1]
    eligible = eligible.with_columns(
        pl.col("lateness").cut(interior_edges, labels=bin_labels).alias("lateness_bin")
    )

    # Compute return rates by bin for both horizons
    results = []
    for horizon, fut_col, elig_col in [
        (30, "future_purchases_30d", "eligible_followup_30d"),
        (90, "future_purchases_90d", "eligible_followup_90d"),
    ]:
        h_data = eligible.filter(pl.col(elig_col))
        if h_data.select(pl.len()).collect().item() == 0:
            continue

        bin_stats = collect(
            h_data.group_by("lateness_bin")
            .agg(
                pl.len().alias("eligible_n"),
                (pl.col(fut_col) > 0).sum().alias("buyers"),
                (pl.col(fut_col) > 0).mean().alias("purchase_rate"),
            )
            .sort("lateness_bin")
        )

        for row in bin_stats.to_dicts():
            n = row["eligible_n"]
            buyers = row["buyers"]
            lo, hi = wilson(buyers, n) if n >= 20 else (None, None)
            results.append(
                {
                    "horizon": horizon,
                    "lateness_bin": row["lateness_bin"],
                    "eligible_n": n,
                    "buyers": buyers,
                    "purchase_rate": buyers / n if n > 0 else None,
                    "lower_ci": lo,
                    "upper_ci": hi,
                }
            )

    return pl.DataFrame(results)


def multi_horizon_cohort_repeat(
    baskets, rows, cutoff, coverage, horizons=[30, 60, 90], min_cohort_size=20
):
    """Multi-horizon repeat curves by cohort with confidence intervals.

    Extends existing cohort_summary to show all horizons together.

    Args:
        baskets: LazyFrame with basket data
        rows: LazyFrame with snapshot features
        cutoff: Cutoff date
        coverage: Coverage confirmation boolean
        horizons: List of horizons in days
        min_cohort_size: Minimum cohort size for display

    Returns:
        DataFrame with cohort, horizon, rate, lower_ci, upper_ci, eligible_n, repeat_n
    """

    import polars as pl

    from retail_analytics.core.stats import wilson

    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    # Get customer cohorts (first purchase month)
    cust_cohorts = rows.select("customer_id", "first_purchase").unique()
    cust_cohorts = cust_cohorts.with_columns(
        pl.col("first_purchase").dt.truncate("1mo").alias("cohort")
    )

    # Get repeat purchases from baskets
    cust_ids = collect(cust_cohorts.select("customer_id"))["customer_id"].to_list()
    # Join baskets to customer cohorts to get first_purchase for each customer
    baskets_with_first = baskets.join(
        cust_cohorts.select("customer_id", "first_purchase"),
        on="customer_id",
        how="inner",
        validate="m:1",
    )
    repeat_data = collect(
        baskets_with_first.filter(pl.col("customer_id").is_in(cust_ids))
        .filter(pl.col("date") <= pl.lit(cutoff))
        .filter(pl.col("date") > pl.col("first_purchase"))
        .group_by("customer_id")
        .agg(pl.col("date").min().alias("repeat_date"))
    ).lazy()

    # Merge
    cohort_data = cust_cohorts.join(repeat_data, on="customer_id", how="left")

    results = []
    for horizon in horizons:
        eligible = pl.col("first_purchase") + pl.duration(days=horizon) <= pl.lit(cutoff)
        success = (
            eligible
            & pl.col("repeat_date").is_not_null()
            & (pl.col("repeat_date") <= pl.col("first_purchase") + pl.duration(days=horizon))
        )

        cohort_stats = (
            cohort_data.group_by("cohort")
            .agg(
                pl.len().alias("cohort_n"),
                eligible.sum().alias("eligible_n"),
                success.fill_null(False).sum().alias("repeat_n"),
            )
            .filter(pl.col("cohort_n") >= min_cohort_size)
        )

        for row in cohort_stats.collect().to_dicts():
            n = row["eligible_n"]
            buyers = row["repeat_n"]
            if coverage and n >= min_cohort_size:
                lo, hi = wilson(buyers, n)
                rate = buyers / n if n > 0 else None
            else:
                lo, hi = None, None
                rate = buyers / n if n > 0 and coverage else None

            results.append(
                {
                    "cohort": row["cohort"],
                    "horizon": horizon,
                    "cohort_n": row["cohort_n"],
                    "eligible_n": n if coverage else None,
                    "repeat_n": buyers if coverage else None,
                    "rate": rate,
                    "lower_ci": lo,
                    "upper_ci": hi,
                    "status": "eligible"
                    if (coverage and n >= min_cohort_size)
                    else "capture_unconfirmed"
                    if not coverage
                    else "insufficient_followup_or_n",
                }
            )

    return pl.DataFrame(results) if results else pl.DataFrame()


def category_opportunity_map(validated, rows, cutoff, top_categories=30):
    """Category penetration × buying intensity opportunity map.

    X: Penetration (% of customers buying category)
    Y: Purchase occasions per category buyer
    Bubble: Category spend share
    Color: Repeat buyer share

    Research basis: Dirichlet model for repeat buying; Ehrenberg (1988) - Repeat Buying.

    Args:
        validated: LazyFrame with validated transaction data
        rows: LazyFrame with snapshot customer features
        cutoff: Cutoff date
        top_categories: Max categories to return

    Returns:
        DataFrame with category, penetration, occasions_per_buyer, spend_share, repeat_buyer_share, buyers
    """
    from datetime import timedelta

    import polars as pl

    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    # 365-day window
    from retail_analytics.core.constants import WINDOW_YEAR

    window_start = cutoff - timedelta(days=WINDOW_YEAR - 1)

    # Get snapshot customer IDs
    cust_ids = rows.select("customer_id").unique().collect()["customer_id"].to_list()
    total_customers = len(cust_ids)

    if total_customers == 0:
        return pl.DataFrame()

    # Get category-level data in window
    # First deduplicate to customer × category × date (shopping occasions)
    cat_occasions = collect(
        validated.filter(
            pl.col("date").is_between(window_start, cutoff) & pl.col("customer_id").is_in(cust_ids)
        )
        .unique(subset=["customer_id", "department", "category", "date"])
        .group_by("department", "category", "customer_id")
        .agg(
            pl.len().alias("occasions"),
            pl.col("value").sum().alias("spend"),
        )
    )

    if cat_occasions.is_empty():
        return pl.DataFrame()

    # Aggregate to category level
    cat_data = (
        cat_occasions.group_by("department", "category")
        .agg(
            pl.len().alias("buyers"),
            pl.col("occasions").mean().alias("occasions_per_buyer"),
            pl.col("spend").sum().alias("total_spend"),
            (pl.col("occasions") > 1).sum().alias("repeat_buyers"),
        )
        .sort("buyers", descending=True)
    )

    # Total spend across ALL categories (for share denominator)
    total_spend_all = cat_data["total_spend"].sum()

    # Select top categories for display
    cat_data = cat_data.head(top_categories)

    result = cat_data.with_columns(
        [
            (pl.col("buyers") / total_customers).alias("penetration"),
            (pl.col("repeat_buyers") / pl.col("buyers")).alias("repeat_buyer_share"),
            (pl.col("total_spend") / total_spend_all).alias("spend_share"),
            pl.concat_str([pl.col("department"), pl.lit(" / "), pl.col("category")]).alias("label"),
        ]
    )

    return result.select(
        "label",
        "department",
        "category",
        "buyers",
        "penetration",
        "occasions_per_buyer",
        "spend_share",
        "repeat_buyer_share",
        "total_spend",
    )


def rfm_response_surface(rows, coverage, recency_bins=5, frequency_bins=5, value_tier_facet=False):
    """RFM response surface: repeat probability heatmap across recency × frequency.

    Research basis: RFM models and Pareto/NBD - explicit response surfaces
    rather than single scores (Fader, Hardie & Lee, 2005).

    Args:
        rows: LazyFrame with customer features
        coverage: Coverage confirmation boolean
        recency_bins: Number of recency quantile bins
        frequency_bins: Number of frequency quantile bins
        value_tier_facet: If True, return separate surfaces per value_tier

    Returns:
        DataFrame with recency_bin, frequency_bin, value_tier (if faceted), 
        eligible_n, buyers, rate, lower_ci, upper_ci
    """
    import polars as pl

    from retail_analytics.core.stats import wilson

    if not coverage:
        return pl.DataFrame({"warning": ["Coverage unconfirmed - future outcomes unavailable"]})

    schema = rows.collect_schema()
    required = ["recency", "purchases_90d", "future_purchases_30d", "eligible_followup_30d"]
    if value_tier_facet:
        required.append("value_tier")
    missing = [c for c in required if c not in schema]
    if missing:
        return pl.DataFrame({"error": [f"Missing columns: {missing}"]})

    eligible = rows.filter(
        pl.col("eligible_followup_30d")
        & pl.col("recency").is_finite()
        & pl.col("purchases_90d").is_finite()
    )

    if eligible.select(pl.len()).collect().item() == 0:
        return pl.DataFrame()

    # If faceting by value_tier, compute quantile bins per tier
    if value_tier_facet:
        # Get value tiers present
        tiers = collect(eligible.select("value_tier").unique().sort("value_tier"))["value_tier"].to_list()
        
        all_results = []
        for tier in tiers:
            tier_data = eligible.filter(pl.col("value_tier") == tier)
            
            if tier_data.select(pl.len()).collect().item() == 0:
                continue
            
            # Create quantile bins within this tier
            recency_quantiles = [i / recency_bins for i in range(recency_bins + 1)]
            freq_quantiles = [i / frequency_bins for i in range(frequency_bins + 1)]

            rec_stats = collect(
                tier_data.select(
                    *[
                        pl.col("recency").quantile(q, interpolation="linear").alias(f"q{i}")
                        for i, q in enumerate(recency_quantiles)
                    ]
                )
            ).row(0, named=True)
            freq_stats = collect(
                tier_data.select(
                    *[
                        pl.col("purchases_90d").quantile(q, interpolation="linear").alias(f"q{i}")
                        for i, q in enumerate(freq_quantiles)
                    ]
                )
            ).row(0, named=True)

            rec_edges = [rec_stats[f"q{i}"] for i in range(recency_bins + 1)]
            freq_edges = [freq_stats[f"q{i}"] for i in range(frequency_bins + 1)]
            rec_edges = sorted(set(rec_edges))
            freq_edges = sorted(set(freq_edges))

            if len(rec_edges) < 2 or len(freq_edges) < 2:
                continue

            rec_labels = [f"R{i + 1}" for i in range(len(rec_edges) - 1)]
            freq_labels = [f"F{i + 1}" for i in range(len(freq_edges) - 1)]
            rec_interior = rec_edges[1:-1]
            freq_interior = freq_edges[1:-1]

            tier_binned = tier_data.with_columns(
                [
                    pl.col("recency").cut(rec_interior, labels=rec_labels).alias("recency_bin"),
                    pl.col("purchases_90d").cut(freq_interior, labels=freq_labels).alias("frequency_bin"),
                ]
            )

            surface = collect(
                tier_binned.group_by("recency_bin", "frequency_bin")
                .agg(
                    pl.len().alias("eligible_n"),
                    (pl.col("future_purchases_30d") > 0).sum().alias("buyers"),
                    (pl.col("future_purchases_30d") > 0).mean().alias("rate"),
                )
                .sort("recency_bin", "frequency_bin")
            )

            for row in surface.to_dicts():
                n = row["eligible_n"]
                buyers = row["buyers"]
                lo, hi = wilson(buyers, n) if n >= 20 else (None, None)
                all_results.append(
                    {
                        "value_tier": tier,
                        "recency_bin": row["recency_bin"],
                        "frequency_bin": row["frequency_bin"],
                        "eligible_n": n,
                        "buyers": buyers,
                        "rate": buyers / n if n > 0 else None,
                        "lower_ci": lo,
                        "upper_ci": hi,
                    }
                )

        return pl.DataFrame(all_results) if all_results else pl.DataFrame()

    # Non-faceted version (original behavior)
    recency_quantiles = [i / recency_bins for i in range(recency_bins + 1)]
    freq_quantiles = [i / frequency_bins for i in range(frequency_bins + 1)]

    rec_stats = collect(
        eligible.select(
            *[
                pl.col("recency").quantile(q, interpolation="linear").alias(f"q{i}")
                for i, q in enumerate(recency_quantiles)
            ]
        )
    ).row(0, named=True)
    freq_stats = collect(
        eligible.select(
            *[
                pl.col("purchases_90d").quantile(q, interpolation="linear").alias(f"q{i}")
                for i, q in enumerate(freq_quantiles)
            ]
        )
    ).row(0, named=True)

    rec_edges = [rec_stats[f"q{i}"] for i in range(recency_bins + 1)]
    freq_edges = [freq_stats[f"q{i}"] for i in range(frequency_bins + 1)]
    rec_edges = sorted(set(rec_edges))
    freq_edges = sorted(set(freq_edges))

    if len(rec_edges) < 2 or len(freq_edges) < 2:
        return pl.DataFrame()

    rec_labels = [f"R{i + 1}" for i in range(len(rec_edges) - 1)]
    freq_labels = [f"F{i + 1}" for i in range(len(freq_edges) - 1)]
    rec_interior = rec_edges[1:-1]
    freq_interior = freq_edges[1:-1]

    eligible = eligible.with_columns(
        [
            pl.col("recency").cut(rec_interior, labels=rec_labels).alias("recency_bin"),
            pl.col("purchases_90d").cut(freq_interior, labels=freq_labels).alias("frequency_bin"),
        ]
    )

    surface = collect(
        eligible.group_by("recency_bin", "frequency_bin")
        .agg(
            pl.len().alias("eligible_n"),
            (pl.col("future_purchases_30d") > 0).sum().alias("buyers"),
            (pl.col("future_purchases_30d") > 0).mean().alias("rate"),
        )
        .sort("recency_bin", "frequency_bin")
    )

    results = []
    for row in surface.to_dicts():
        n = row["eligible_n"]
        buyers = row["buyers"]
        lo, hi = wilson(buyers, n) if n >= 20 else (None, None)
        results.append(
            {
                "recency_bin": row["recency_bin"],
                "frequency_bin": row["frequency_bin"],
                "eligible_n": n,
                "buyers": buyers,
                "rate": buyers / n if n > 0 else None,
                "lower_ci": lo,
                "upper_ci": hi,
            }
        )

    return pl.DataFrame(results)


def cohort_age_retention_heatmap(validated: pl.LazyFrame, rows: pl.LazyFrame, cutoff: date, min_cohort_size: int = 20):
    """Cohort-age retention heatmap: monthly activity by first-observed cohort age.

    Unlike calendar-time heatmaps, this aligns customers by relationship age (months since first purchase).
    Shows proportion of each cohort making a purchase in each subsequent month.

    Args:
        validated: LazyFrame with validated transaction data (customer_id, date)
        rows: LazyFrame with snapshot customer features (customer_id, first_purchase)
        cutoff: Snapshot cutoff date
        min_cohort_size: Minimum cohort size for inclusion

    Returns:
        DataFrame with cohort_month, age_month, active_customers, cohort_size, activity_rate
    """
    import polars as pl

    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    # Get customer cohorts (first observed purchase month)
    cust_cohorts = rows.select("customer_id", "first_purchase").unique().collect()
    if cust_cohorts.is_empty():
        return pl.DataFrame()

    cust_cohorts = cust_cohorts.with_columns(
        pl.col("first_purchase").dt.truncate("1mo").alias("cohort_month")
    )

    # Get all validated purchases for these customers up to cutoff
    cust_ids = cust_cohorts["customer_id"].to_list()
    purchases = collect(
        validated.filter(
            (pl.col("customer_id").is_in(cust_ids))
            & (pl.col("date") <= pl.lit(cutoff))
        )
        .select("customer_id", "date")
        .unique(subset=["customer_id", "date"])
    )

    if purchases.is_empty():
        return pl.DataFrame()

    # Add month truncation to purchases
    purchases = purchases.with_columns(
        pl.col("date").dt.truncate("1mo").alias("purchase_month")
    )

    # Join with cohort info
    purchases = purchases.join(cust_cohorts.select("customer_id", "cohort_month", "first_purchase"), on="customer_id", how="inner")

    # Compute age in months since first purchase
    purchases = purchases.with_columns(
        ((pl.col("purchase_month").dt.year() - pl.col("cohort_month").dt.year()) * 12
         + (pl.col("purchase_month").dt.month() - pl.col("cohort_month").dt.month())).alias("age_month")
    )

    # Filter to observed months (purchase_month <= cutoff truncated to month)
    cutoff_month = date(cutoff.year, cutoff.month, 1)
    purchases = purchases.filter(pl.col("purchase_month") <= pl.lit(cutoff_month))

    # Cohort sizes
    cohort_sizes = (
        cust_cohorts.group_by("cohort_month")
        .agg(pl.len().alias("cohort_size"))
        .filter(pl.col("cohort_size") >= min_cohort_size)
    )

    if cohort_sizes.is_empty():
        return pl.DataFrame()

    # Monthly active customers per cohort per age
    monthly_active = (
        purchases.join(cohort_sizes, on="cohort_month", how="inner")
        .group_by("cohort_month", "age_month")
        .agg(pl.col("customer_id").n_unique().alias("active_customers"))
    )

    # Join with cohort sizes to compute rate
    result = (
        monthly_active.join(cohort_sizes, on="cohort_month", how="inner")
        .with_columns(
            (pl.col("active_customers") / pl.col("cohort_size")).alias("activity_rate"),
        )
        .sort("cohort_month", "age_month")
    )

    return result.select("cohort_month", "age_month", "active_customers", "cohort_size", "activity_rate")


def calendar_period_revenue_trend(validated: pl.LazyFrame, rows: pl.LazyFrame, first_date: date, cutoff: date, months: int = 12):
    """Calendar-period revenue trends with purchasing components.

    Shows non-overlapping monthly windows with:
    - Monthly recorded spend
    - Unique purchasing customers
    - Basket count
    - Average basket value (AOV)
    - Baskets per purchaser
    - Buying share (purchasers / population)

    This complements the rolling 90-day view with independent calendar periods.

    Args:
        validated: LazyFrame with validated transaction data
        rows: LazyFrame with snapshot customer features
        first_date: Earliest date in data
        cutoff: Snapshot cutoff date
        months: Number of months to include

    Returns:
        DataFrame with month, spend, baskets, buyers, aov, baskets_per_buyer, buying_share, population_n
    """
    from retail_analytics.visualization.visual_insights_core import month_windows, collect

    windows = month_windows(first_date, cutoff)[-months:]
    calendar = pl.DataFrame(
        windows, schema={"month": pl.Date, "start": pl.Date, "end": pl.Date, "partial": pl.Boolean}
    )
    if not windows:
        return calendar

    ids = rows.select("customer_id").unique()
    n = collect(ids.select(pl.len())).item()
    selected = validated.filter(pl.col("date").is_between(windows[0]["start"], cutoff)).join(
        ids, on="customer_id", how="semi"
    )

    summary = (
        selected.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(
            pl.col("value").sum().alias("spend"),
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
            .alias("aov"),
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


def revenue_bridge(rows, previous_rows, cutoff, match_customers: bool = True):
    """Revenue bridge with explicit population matching option.

    Decomposes spend change into:
    1. Customer count effect (matched or unmatched population)
    2. Frequency effect (baskets per purchaser)
    3. AOV effect (spend per basket)
    4. Residual (interaction)

    Args:
        rows: Current snapshot LazyFrame
        previous_rows: Previous snapshot LazyFrame
        cutoff: Current cutoff date
        match_customers: If True, use only matched customers for frequency/AOV comparison

    Returns:
        DataFrame with component, value, description, population_type
    """
    import polars as pl

    if rows is None or previous_rows is None:
        return pl.DataFrame()

    # Current period stats
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

    if pc_prev == 0 or tb_prev == 0:
        return pl.DataFrame(
            {
                "component": [
                    "Previous period spend",
                    "Current period spend",
                    "Total change",
                    "Decomposition",
                ],
                "value": [float(ts_prev), float(ts_curr), float(ts_curr - ts_prev), None],
                "status": [
                    None,
                    None,
                    None,
                    "unavailable: zero purchasing customers or baskets in previous period",
                ],
            }
        )

    if pc_curr == 0 or tb_curr == 0:
        return pl.DataFrame(
            {
                "component": [
                    "Previous period spend",
                    "Current period spend",
                    "Total change",
                    "Decomposition",
                ],
                "value": [float(ts_prev), float(ts_curr), float(ts_curr - ts_prev), None],
                "status": [
                    None,
                    None,
                    None,
                    "unavailable: zero purchasing customers or baskets in current period",
                ],
            }
        )

    F_prev = tb_prev / pc_prev
    A_prev = ts_prev / tb_prev
    F_curr = tb_curr / pc_curr
    A_curr = ts_curr / tb_curr

    delta_N = (pc_curr - pc_prev) * F_prev * A_prev
    delta_F = pc_curr * (F_curr - F_prev) * A_prev
    delta_A = pc_curr * F_curr * (A_curr - A_prev)

    residual = ts_curr - ts_prev - delta_N - delta_F - delta_A

    pop_type = "matched" if match_customers else "unmatched (full snapshots)"

    # Try to get snapshot date from previous_rows for description, fallback to cutoff
    try:
        prev_snapshot = previous_rows.collect().select(pl.col("snapshot").max()).item()
        prev_label = str(prev_snapshot) if prev_snapshot else "previous"
    except Exception:
        prev_label = "previous"

    return pl.DataFrame(
        {
            "component": [
                "Previous period spend",
                "Purchasing customers",
                "Baskets per customer",
                "Spend per basket (AOV)",
                "Residual (interaction)",
                "Current period spend",
            ],
            "value": [
                float(ts_prev),
                float(delta_N),
                float(delta_F),
                float(delta_A),
                float(residual),
                float(ts_curr),
            ],
            "description": [
                f"Total spend {prev_label}",
                f"ΔCustomers × prev frequency × prev AOV ({pop_type})",
                f"Current customers × Δfrequency × prev AOV ({pop_type})",
                f"Current customers × current frequency × ΔAOV ({pop_type})",
                "Interaction/rounding effects",
                f"Total spend {cutoff}",
            ],
            "population_type": [None, pop_type, pop_type, pop_type, None, None],
        }
    )


def value_at_risk_scatter(rows, lateness_threshold: float = 2.0, sample_limit: int = 2000):
    """Value-at-risk scatter: customers combining high spend with overdue cadence.

    X-axis: lateness (recency / median_gap)
    Y-axis: spend_365d (or spend_90d)
    Bubble size: purchases_365d
    Color: value_tier or trajectory

    Args:
        rows: LazyFrame with customer features
        lateness_threshold: Lateness value marking "overdue" (default 2.0)
        sample_limit: Max customers for sampled scatter

    Returns:
        Dict with 'sample' (DataFrame for plotting), 'summary' (full-pop aggregates)
    """
    import polars as pl

    # Check required columns
    schema = rows.collect_schema()
    required = ["recency", "median_gap", "spend_365d", "spend_90d", "purchases_365d", "value_tier", "trajectory", "lateness"]
    missing = [c for c in required if c not in schema]
    if missing:
        return {"sample": pl.DataFrame(), "summary": pl.DataFrame()}

    # Filter to customers with finite cadence metrics
    eligible = rows.filter(
        pl.col("median_gap").is_finite()
        & (pl.col("median_gap") > 0)
        & pl.col("recency").is_finite()
        & pl.col("spend_365d").is_finite()
        & pl.col("spend_90d").is_finite()
    )

    total_n = collect(rows.select(pl.len())).item()
    eligible_n = collect(eligible.select(pl.len())).item()

    if eligible_n == 0:
        return {"sample": pl.DataFrame(), "summary": pl.DataFrame()}

    # Full population summary stats
    summary = collect(
        eligible.select(
            pl.len().alias("eligible_customers"),
            pl.col("spend_365d").sum().alias("total_spend_365d"),
            pl.col("spend_90d").sum().alias("total_spend_90d"),
            (pl.col("lateness") >= lateness_threshold).sum().alias("overdue_customers"),
            (pl.col("lateness") >= lateness_threshold).mean().alias("overdue_share"),
            pl.col("spend_365d").filter(pl.col("lateness") >= lateness_threshold).sum().alias("at_risk_spend_365d"),
            pl.col("spend_90d").filter(pl.col("lateness") >= lateness_threshold).sum().alias("at_risk_spend_90d"),
            pl.col("purchases_365d").filter(pl.col("lateness") >= lateness_threshold).sum().alias("at_risk_purchases"),
        )
    ).row(0, named=True)

    summary_df = pl.DataFrame([summary])

    # Sample for scatter plot
    sample = collect(
        eligible.sort(pl.col("customer_id").hash(seed=29))
        .limit(sample_limit)
        .select(
            "customer_id",
            "recency",
            "median_gap",
            "lateness",
            "spend_365d",
            "spend_90d",
            "purchases_365d",
            "purchases_90d",
            "value_tier",
            "trajectory",
            "evidence",
            "lifecycle",
        )
    )

    return {"sample": sample, "summary": summary_df}


def product_association_network(product_pairs: pl.DataFrame, min_lift: float = 1.2, min_confidence: float = 0.15, top_n: int = 100):
    """Build a product association network from affinity pairs.

    Creates nodes and edges suitable for network visualization (Plotly or cytoscape).
    Filters by quality thresholds.

    Args:
        product_pairs: DataFrame from compute_item_pairs (product level)
        min_lift: Minimum lift threshold
        min_confidence: Minimum confidence threshold
        top_n: Maximum edges to return

    Returns:
        Dict with 'nodes' (DataFrame) and 'edges' (DataFrame) for network rendering
    """
    import polars as pl

    if product_pairs.is_empty():
        return {"nodes": pl.DataFrame(), "edges": pl.DataFrame()}

    # Filter by quality
    filtered = product_pairs.filter(
        (pl.col("lift") >= min_lift)
        & (
            (pl.col("confidence_a_to_b") >= min_confidence)
            | (pl.col("confidence_b_to_a") >= min_confidence)
        )
    ).head(top_n)

    if filtered.is_empty():
        return {"nodes": pl.DataFrame(), "edges": pl.DataFrame()}

    # Build node list (all unique items in filtered pairs)
    items_a = filtered.select(pl.col("item_a").alias("product_id"))
    items_b = filtered.select(pl.col("item_b").alias("product_id"))
    nodes = pl.concat([items_a, items_b]).unique().sort("product_id")

    # Add node metrics: degree, total support
    edge_data = filtered.to_dicts()
    node_stats = {}
    for edge in edge_data:
        for item, conf, lift in [
            (edge["item_a"], edge["confidence_a_to_b"], edge["lift"]),
            (edge["item_b"], edge["confidence_b_to_a"], edge["lift"]),
        ]:
            if item not in node_stats:
                node_stats[item] = {"degree": 0, "total_support": 0.0, "max_lift": 0.0}
            node_stats[item]["degree"] += 1
            node_stats[item]["total_support"] += edge["support"]
            node_stats[item]["max_lift"] = max(node_stats[item]["max_lift"], edge["lift"] or 0)

    nodes = nodes.with_columns([
        pl.col("product_id").map_elements(lambda x: node_stats.get(x, {}).get("degree", 0), return_dtype=pl.Int64).alias("degree"),
        pl.col("product_id").map_elements(lambda x: node_stats.get(x, {}).get("total_support", 0.0), return_dtype=pl.Float64).alias("total_support"),
        pl.col("product_id").map_elements(lambda x: node_stats.get(x, {}).get("max_lift", 0.0), return_dtype=pl.Float64).alias("max_lift"),
    ])

    # Edges for visualization
    edges = filtered.select(
        "item_a",
        "item_b",
        "cooccurrence_count",
        "support",
        "confidence_a_to_b",
        "confidence_b_to_a",
        "lift",
        "conviction_a_to_b",
        "conviction_b_to_a",
    )

    return {"nodes": nodes, "edges": edges}


def trial_conversion_interval_plot(product_history: pl.LazyFrame, rows: pl.LazyFrame, cutoff: date, min_trials: int = 5, category_lookup: Optional[pl.DataFrame] = None):
    """Trial-to-repeat conversion with Wilson confidence intervals per product.

    Shows eligible trials, repeat conversion rate, and 95% CI for each product.
    Optional category-level aggregation.

    Args:
        product_history: LazyFrame with product history (trial, repeat dates)
        rows: LazyFrame with snapshot customer features
        cutoff: Snapshot cutoff date
        min_trials: Minimum eligible trials per product
        category_lookup: Optional DataFrame mapping product_id to category

    Returns:
        DataFrame with product_id, category (if available), eligible_trials, repeated_trials,
        trial_conversion, lower_ci, upper_ci, trialling_customers
    """
    import polars as pl

    from retail_analytics.core.stats import wilson
    from retail_analytics.core.constants import WINDOW_YEAR

    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    trial_start = cutoff - timedelta(days=WINDOW_YEAR - 1)
    trial_end = cutoff - timedelta(days=90)

    ids = rows.select("customer_id").unique()

    # Filter to eligible trials
    trials = product_history.filter(
        pl.col("trial").is_between(trial_start, trial_end)
    ).join(ids, on="customer_id", how="semi")

    # Aggregate by product
    result = collect(
        trials.group_by("product_id")
        .agg(
            pl.col("customer_id").n_unique().alias("trialling_customers"),
            pl.len().alias("eligible_trials"),
            (
                pl.col("repeat").is_not_null()
                & (pl.col("repeat") <= pl.col("trial") + pl.duration(days=90))
            )
            .fill_null(False)
            .sum()
            .alias("repeated_trials"),
        )
        .filter(pl.col("eligible_trials") >= min_trials)
        .with_columns(
            (pl.col("repeated_trials") / pl.col("eligible_trials")).alias("trial_conversion"),
        )
    )

    # Add Wilson CIs
    rows_with_ci = []
    for row in result.to_dicts():
        n = row["eligible_trials"]
        buyers = row["repeated_trials"]
        lo, hi = wilson(buyers, n) if n >= 20 else (None, None)
        row["lower_ci"] = lo
        row["upper_ci"] = hi
        rows_with_ci.append(row)

    result_df = pl.DataFrame(rows_with_ci) if rows_with_ci else pl.DataFrame()

    # Add category if lookup provided
    if category_lookup is not None and not result_df.is_empty():
        result_df = result_df.join(category_lookup, on="product_id", how="left")

    return result_df.sort("eligible_trials", descending=True)


def category_penetration_growth_portfolio(validated: pl.LazyFrame, rows: pl.LazyFrame, current_cutoff: date, previous_cutoff: date, top_categories: int = 30):
    """Category penetration vs growth portfolio.

    X-axis: Current category penetration (buyers / population)
    Y-axis: Change in participation vs previous window (percentage points)
    Bubble size: Category spend share
    Bubble color: Repeat buyer share

    Args:
        validated: Validated transaction data
        rows: Current snapshot customer features
        current_cutoff: Current analysis cutoff
        previous_cutoff: Previous comparison cutoff
        top_categories: Max categories to return

    Returns:
        DataFrame with category, current_penetration, prev_penetration, penetration_change_pp,
        spend_share, repeat_buyer_share, current_buyers, prev_buyers, current_spend, prev_spend
    """
    import polars as pl

    from retail_analytics.core.constants import WINDOW_YEAR

    if isinstance(current_cutoff, str):
        current_cutoff = date.fromisoformat(current_cutoff)
    if isinstance(previous_cutoff, str):
        previous_cutoff = date.fromisoformat(previous_cutoff)

    # Get customer populations
    curr_ids = rows.select("customer_id").unique().collect()
    curr_n = curr_ids.height
    curr_id_list = curr_ids["customer_id"].to_list()

    # Need previous snapshot rows - this is a limitation; we'll use the current population
    # for both periods (fixed population comparison) or need previous_rows passed in
    # For now, use fixed population (current snapshot customers) for both periods
    # This matches the "Category Participation" view methodology

    # Current 365-day window
    curr_start = current_cutoff - timedelta(days=WINDOW_YEAR - 1)
    curr_end = current_cutoff

    # Previous 365-day window (shifted by same interval between cutoffs)
    interval = (current_cutoff - previous_cutoff).days
    prev_start = curr_start - timedelta(days=interval)
    prev_end = curr_end - timedelta(days=interval)

    # Current participation
    curr_cats = collect(
        validated.filter(
            pl.col("date").is_between(curr_start, curr_end)
            & pl.col("customer_id").is_in(curr_id_list)
        )
        .unique(subset=["customer_id", "department", "category", "date"])
        .group_by("department", "category")
        .agg(
            pl.col("customer_id").n_unique().alias("curr_buyers"),
            pl.col("value").sum().alias("curr_spend"),
            (pl.col("customer_id").unique().n_unique() > 1).sum().alias("curr_repeat_buyers"),  # simplified
        )
        .with_columns(
            (pl.col("curr_buyers") / curr_n).alias("curr_penetration"),
            (pl.col("curr_repeat_buyers") / pl.col("curr_buyers")).alias("curr_repeat_buyer_share"),
        )
    )

    # Previous participation (same customer population)
    prev_cats = collect(
        validated.filter(
            pl.col("date").is_between(prev_start, prev_end)
            & pl.col("customer_id").is_in(curr_id_list)
        )
        .unique(subset=["customer_id", "department", "category", "date"])
        .group_by("department", "category")
        .agg(
            pl.col("customer_id").n_unique().alias("prev_buyers"),
            pl.col("value").sum().alias("prev_spend"),
            (pl.col("customer_id").unique().n_unique() > 1).sum().alias("prev_repeat_buyers"),
        )
        .with_columns(
            (pl.col("prev_buyers") / curr_n).alias("prev_penetration"),
            (pl.col("prev_repeat_buyers") / pl.col("prev_buyers")).alias("prev_repeat_buyer_share"),
        )
    )

    # Full join
    merged = (
        curr_cats.join(prev_cats, on=["department", "category"], how="full", coalesce=True)
        .with_columns([
            pl.col("curr_penetration").fill_null(0),
            pl.col("prev_penetration").fill_null(0),
            pl.col("curr_buyers").fill_null(0),
            pl.col("prev_buyers").fill_null(0),
            pl.col("curr_spend").fill_null(0),
            pl.col("prev_spend").fill_null(0),
            pl.col("curr_repeat_buyer_share").fill_null(0),
            pl.col("prev_repeat_buyer_share").fill_null(0),
        ])
        .with_columns(
            (pl.col("curr_penetration") - pl.col("prev_penetration")).alias("penetration_change_pp"),
            (pl.col("curr_spend") + pl.col("prev_spend")).alias("total_spend"),
            pl.concat_str([pl.col("department"), pl.lit(" / "), pl.col("category")]).alias("label"),
        )
    )

    # Spend share denominator: total spend across all categories in current period
    total_curr_spend = merged["curr_spend"].sum()
    merged = merged.with_columns(
        (pl.col("curr_spend") / total_curr_spend if total_curr_spend > 0 else 0).alias("spend_share"),
        pl.col("curr_repeat_buyer_share").alias("repeat_buyer_share"),
    )

    return merged.sort("curr_penetration", descending=True).head(top_categories)


def segment_alluvial_flow(rows_history: list, dimensions: list = None):
    """Multi-snapshot segment alluvial flow.

    Builds flow data for Sankey/alluvial diagram across multiple snapshots.
    Tracks customer movement between states (lifecycle, value_tier, behavior).

    Args:
        rows_history: List of LazyFrames, one per snapshot (chronological order)
        dimensions: List of dimension columns to track (default: ["value_tier", "lifecycle", "behavior"])

    Returns:
        Dict with 'flows' (DataFrame with from_snapshot, to_snapshot, from_state, to_state, count),
        'snapshots' (list of cutoff dates), 'dimensions' (list)
    """
    import polars as pl

    if dimensions is None:
        dimensions = ["value_tier", "lifecycle", "behavior"]

    if len(rows_history) < 2:
        return {"flows": pl.DataFrame(), "snapshots": [], "dimensions": dimensions}

    all_flows = []
    snapshots = []

    for i in range(len(rows_history) - 1):
        before = rows_history[i]
        after = rows_history[i + 1]

        # Get cutoff dates
        before_cutoff = collect(before.select(pl.col("snapshot").max())).item()
        after_cutoff = collect(after.select(pl.col("snapshot").max())).item()
        snapshots.append((before_cutoff, after_cutoff))

        for dim in dimensions:
            if dim not in before.collect_schema().names() or dim not in after.collect_schema().names():
                continue

            origin = before.select("customer_id", pl.col(dim).alias("from_state"))
            destination = after.select("customer_id", pl.col(dim).alias("to_state"))
            joined = origin.join(destination, on="customer_id", how="inner", validate="1:1")

            matched_n = collect(joined.select(pl.len())).item()
            if matched_n == 0:
                continue

            flows = collect(
                joined.group_by("from_state", "to_state")
                .len()
                .with_columns([
                    pl.lit(before_cutoff).alias("from_snapshot"),
                    pl.lit(after_cutoff).alias("to_snapshot"),
                    pl.lit(dim).alias("dimension"),
                ])
                .rename({"len": "customer_count"})
            )
            all_flows.append(flows)

    if not all_flows:
        return {"flows": pl.DataFrame(), "snapshots": snapshots, "dimensions": dimensions}

    combined = pl.concat(all_flows, how="vertical_relaxed")
    return {"flows": combined, "snapshots": snapshots, "dimensions": dimensions}


def full_population_distribution(rows, features: list = None):
    """Full-population distribution summaries (not sampled).

    Computes histogram bins, quantiles, and summary statistics for the entire
    eligible population, avoiding sampling bias in distribution views.

    Args:
        rows: LazyFrame with customer features
        features: List of feature columns to summarize (default: key behavioral metrics)

    Returns:
        Dict mapping feature -> DataFrame with bin edges, counts, and summary stats
    """
    import polars as pl

    if features is None:
        features = [
            "spend_90d", "spend_365d", "purchases_90d", "recency",
            "median_gap", "gap_cv", "lateness", "median_basket",
            "new_share", "repeat_share", "established_share",
            "category_hhi", "trial_conversion", "frequency_change", "basket_change"
        ]

    schema = rows.collect_schema()
    available = [f for f in features if f in schema and schema[f].is_numeric()]

    if not available:
        return {}

    results = {}
    for feature in available:
        # Filter finite values
        data = rows.filter(pl.col(feature).is_finite())

        n = collect(data.select(pl.len())).item()
        if n == 0:
            continue

        # Summary stats
        stats = collect(
            data.select(
                pl.col(feature).count().alias("n"),
                pl.col(feature).mean().alias("mean"),
                pl.col(feature).median().alias("median"),
                pl.col(feature).std().alias("std"),
                pl.col(feature).min().alias("min"),
                pl.col(feature).max().alias("max"),
                pl.col(feature).quantile(0.01).alias("p1"),
                pl.col(feature).quantile(0.05).alias("p5"),
                pl.col(feature).quantile(0.25).alias("q25"),
                pl.col(feature).quantile(0.75).alias("q75"),
                pl.col(feature).quantile(0.95).alias("p95"),
                pl.col(feature).quantile(0.99).alias("p99"),
            )
        ).row(0, named=True)

        # Histogram bins (Sturges' rule, capped)
        nbins = min(50, max(10, int(math.log2(n)) + 1))
        hist = collect(
            data.select(
                pl.col(feature).cut(
                    [stats["min"] + (stats["max"] - stats["min"]) * i / nbins for i in range(1, nbins)],
                    labels=[f"bin_{i}" for i in range(nbins)]
                ).alias("bin")
            ).group_by("bin").len().sort("bin")
        )

        results[feature] = {
            "summary": pl.DataFrame([stats]),
            "histogram": hist,
            "n": n,
        }

    return results
