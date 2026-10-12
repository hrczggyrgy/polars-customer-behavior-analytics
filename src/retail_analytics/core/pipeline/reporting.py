"""Reporting: segment profiles, cohort profiles, and transition analysis.

This module generates aggregated reports from labeled snapshots:
- Segment profiles: customer counts, revenue, future purchase rates by segment
- Cohort profiles: customer behavior by acquisition cohort with exposure tracking
- Segment transitions: transition rates between segments over time
"""

from pathlib import Path

import polars as pl

from retail_analytics.core.pipeline.prepare import C, _sink
from retail_analytics.core.stats import safe_ratio


def generate_reports(root: Path, buckets: int) -> None:
    """Generate segment profiles, cohort profiles, and transitions.

    Args:
        root: Output root directory.
        buckets: Number of hash partitions.
    """
    allrows = pl.scan_parquet(str(root / "snapshots" / "*" / "*.parquet"))

    # Segment profiles
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
        safe_ratio(pl.col("customers"), pl.col("customers").sum().over("snapshot")).alias(
            "customer_share"
        ),
        safe_ratio(pl.col("revenue"), pl.col("revenue").sum().over("snapshot")).alias(
            "revenue_share"
        ),
    )
    _sink(profile, root / "segment_profiles.parquet", sort_by=["snapshot", "behavior"])

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
    _sink(cohort_profile, root / "cohort_profiles.parquet", sort_by=["snapshot", "cohort"])

    # Transitions
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
            _sink(
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
        _sink(
            trans.with_columns(
                safe_ratio(pl.col("customers"), pl.col("eligible_origin")).alias("transition_rate")
            ),
            root / "segment_transitions.parquet",
            sort_by=["from_date", "snapshot", "dimension", "from_label", "to_label"],
        )
