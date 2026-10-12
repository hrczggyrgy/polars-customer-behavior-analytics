"""Lightweight product affinity and market-basket analysis.

This module implements sparse, streaming-friendly association mining that avoids
dense customer-by-product matrices. It uses basket-level aggregation with
support thresholds and top-k candidate pruning.

Key design principles:
- No dense matrices: operates on basket (transaction) level only
- Streaming-friendly: uses Polars lazy evaluation with collect() only at final steps
- Configurable support thresholds: filters infrequent items early
- Sparse pair generation: only emits pairs exceeding minimum support
- Category and product level: supports both granularities
"""

import math
from datetime import timedelta
from typing import Optional

import polars as pl

from retail_analytics.core.constants import WINDOW_YEAR, WINDOW_RECENT


def _support_from_counts(count: int, total_baskets: int) -> float:
    """Compute support as fraction of total baskets."""
    return count / total_baskets if total_baskets > 0 else 0.0


def _confidence(support_ab: float, support_a: float) -> Optional[float]:
    """Compute confidence P(B|A) = support(A∪B) / support(A)."""
    return support_ab / support_a if support_a > 0 else None


def _lift(support_ab: float, support_a: float, support_b: float) -> Optional[float]:
    """Compute lift = P(A∪B) / (P(A) * P(B))."""
    denom = support_a * support_b
    return support_ab / denom if denom > 0 else None


def _conviction(support_a: float, support_b: float, confidence_ab: float) -> Optional[float]:
    """Compute conviction = (1 - P(B)) / (1 - P(B|A))."""
    if confidence_ab is None or confidence_ab >= 1.0:
        return None
    return (1.0 - support_b) / (1.0 - confidence_ab)


class ProductAffinityConfig:
    """Configuration for product affinity analysis."""

    def __init__(
        self,
        min_support: float = 0.001,           # Minimum support for itemsets (fraction of baskets)
        min_confidence: float = 0.1,          # Minimum confidence for rules
        min_lift: float = 1.0,                # Minimum lift for rules
        max_items_per_basket: int = 50,       # Cap basket size to limit pair explosion
        top_k_items: int = 500,               # Consider only top-K frequent items
        top_k_pairs: int = 2000,              # Emit only top-K pairs by support
        category_level: bool = True,          # Also compute category-level affinities
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_items_per_basket = max_items_per_basket
        self.top_k_items = top_k_items
        self.top_k_pairs = top_k_pairs
        self.category_level = category_level


def compute_frequent_items(
    baskets: pl.LazyFrame,
    item_col: str,
    config: ProductAffinityConfig,
    cutoff_date: Optional[str] = None,
    window_days: int = WINDOW_YEAR,
) -> pl.DataFrame:
    """Compute frequent items (products or categories) from basket data.

    Args:
        baskets: LazyFrame with basket-level data (must have 'date' and item_col)
        item_col: Column name for items ('product_id' or 'category_key')
        config: ProductAffinityConfig with thresholds
        cutoff_date: Optional snapshot cutoff date (ISO format)
        window_days: Lookback window in days

    Returns:
        DataFrame with frequent items: item, basket_count, support
    """
    if cutoff_date:
        cutoff = pl.lit(cutoff_date).str.to_date()
        start = cutoff - pl.duration(days=window_days - 1)
        filtered = baskets.filter(pl.col("date").is_between(start, cutoff))
    else:
        filtered = baskets

    # Total DISTINCT baskets in window (denominator for support)
    total_baskets = collect(
        filtered.select("date", "customer_id", "transaction_id")
        .unique()
        .select(pl.len())
    ).item()

    # Get distinct items per basket (deduplicate within basket)
    items_per_basket = (
        filtered.select("date", "customer_id", "transaction_id", item_col)
        .unique(subset=["date", "customer_id", "transaction_id", item_col])
    )

    # Count baskets per item
    item_counts = (
        items_per_basket.group_by(item_col)
        .agg(pl.len().alias("basket_count"))
        .sort("basket_count", descending=True)
    )

    # Filter by minimum support
    min_baskets = max(1, math.ceil(config.min_support * total_baskets))
    frequent = item_counts.filter(pl.col("basket_count") >= min_baskets)

    # Limit to top-K
    frequent = frequent.head(config.top_k_items)

    # Add support
    return collect(
        frequent.with_columns(
            (pl.col("basket_count") / pl.lit(total_baskets)).alias("support"),
            pl.lit(total_baskets).alias("total_baskets"),
        )
    )


def compute_item_pairs(
    baskets: pl.LazyFrame,
    item_col: str,
    frequent_items: pl.DataFrame,
    config: ProductAffinityConfig,
    cutoff_date: Optional[str] = None,
    window_days: int = WINDOW_YEAR,
) -> pl.DataFrame:
    """Compute co-occurrence pairs for frequent items.

    Uses a sparse approach: only generates pairs from baskets containing
    at least 2 frequent items, and only emits pairs exceeding min_support.

    Args:
        baskets: LazyFrame with basket-level data
        item_col: Column name for items
        frequent_items: DataFrame from compute_frequent_items
        config: ProductAffinityConfig
        cutoff_date: Optional snapshot cutoff date
        window_days: Lookback window in days

    Returns:
        DataFrame with pairs: item_a, item_b, cooccurrence_count, support, confidence_a, confidence_b, lift, conviction
    """
    # Get frequent item list as a Python list for filtering
    frequent_set = set(frequent_items[item_col].to_list())
    if len(frequent_set) < 2:
        return pl.DataFrame(schema={
            "item_a": pl.String, "item_b": pl.String,
            "cooccurrence_count": pl.Int64, "support": pl.Float64,
            "confidence_a_to_b": pl.Float64, "confidence_b_to_a": pl.Float64,
            "lift": pl.Float64, "conviction_a_to_b": pl.Float64, "conviction_b_to_a": pl.Float64,
        })

    # Map items to their individual supports
    item_supports = dict(zip(
        frequent_items[item_col].to_list(),
        frequent_items["support"].to_list()
    ))
    # Use correct total_baskets from frequent_items (which now has correct denominator)
    total_baskets = frequent_items["total_baskets"][0] if len(frequent_items) > 0 else 0
    min_cooccurrence = max(1, math.ceil(config.min_support * total_baskets))

    if cutoff_date:
        cutoff = pl.lit(cutoff_date).str.to_date()
        start = cutoff - pl.duration(days=window_days - 1)
        filtered = baskets.filter(pl.col("date").is_between(start, cutoff))
    else:
        filtered = baskets

    # Get distinct items per basket, filter to frequent items only
    basket_items = (
        filtered.select("date", "customer_id", "transaction_id", item_col)
        .unique(subset=["date", "customer_id", "transaction_id", item_col])
        .filter(pl.col(item_col).is_in(frequent_set))
    )

    # Count items per basket and filter to baskets with 2+ frequent items
    basket_item_counts = basket_items.group_by(["date", "customer_id", "transaction_id"]).agg(
        pl.col(item_col).alias("items"),
        pl.len().alias("item_count"),
    ).filter(pl.col("item_count") >= 2)

    # Cap basket size to limit pair explosion
    basket_item_counts = basket_item_counts.with_columns(
        pl.when(pl.col("item_count") > config.max_items_per_basket)
        .then(pl.col("items").list.slice(0, config.max_items_per_basket))
        .otherwise(pl.col("items"))
        .alias("items_capped"),
    ).drop("item_count", "items")

    # Self-join to generate pairs within each basket
    # This is the key sparse operation - only generates pairs from actual co-occurrences
    pairs = (
        basket_item_counts
        .explode("items_capped")
        .rename({"items_capped": "item_a"})
        .join(
            basket_item_counts.explode("items_capped").rename({"items_capped": "item_b"}),
            on=["date", "customer_id", "transaction_id"],
            how="inner",
        )
        .filter(pl.col("item_a") < pl.col("item_b"))  # Deduplicate: only A < B
    )

    # Count co-occurrences
    pair_counts = (
        pairs.group_by("item_a", "item_b")
        .agg(pl.len().alias("cooccurrence_count"))
        .filter(pl.col("cooccurrence_count") >= min_cooccurrence)
        .sort("cooccurrence_count", descending=True)
        .head(config.top_k_pairs)
    )

    # Compute metrics
    return collect(
        pair_counts.with_columns([
            (pl.col("cooccurrence_count") / pl.lit(total_baskets)).alias("support"),
        ])
        .with_columns([
            # Confidence A -> B
            pl.when(pl.col("item_a").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64) > 0)
            .then(pl.col("support") / pl.col("item_a").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64))
            .otherwise(None)
            .alias("confidence_a_to_b"),
            # Confidence B -> A
            pl.when(pl.col("item_b").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64) > 0)
            .then(pl.col("support") / pl.col("item_b").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64))
            .otherwise(None)
            .alias("confidence_b_to_a"),
        ])
        .with_columns([
            # Lift
            pl.when(
                (pl.col("item_a").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64) > 0) &
                (pl.col("item_b").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64) > 0)
            )
            .then(
                pl.col("support") /
                (pl.col("item_a").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64) *
                 pl.col("item_b").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64))
            )
            .otherwise(None)
            .alias("lift"),
        ])
        .with_columns([
            # Conviction A -> B
            pl.when(
                (pl.col("confidence_a_to_b").is_not_null()) &
                (pl.col("confidence_a_to_b") < 1.0) &
                (pl.col("item_b").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64) < 1.0)
            )
            .then(
                (1.0 - pl.col("item_b").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64)) /
                (1.0 - pl.col("confidence_a_to_b"))
            )
            .otherwise(None)
            .alias("conviction_a_to_b"),
            # Conviction B -> A
            pl.when(
                (pl.col("confidence_b_to_a").is_not_null()) &
                (pl.col("confidence_b_to_a") < 1.0) &
                (pl.col("item_a").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64) < 1.0)
            )
            .then(
                (1.0 - pl.col("item_a").map_elements(lambda x: item_supports.get(x, 0.0), return_dtype=pl.Float64)) /
                (1.0 - pl.col("confidence_b_to_a"))
            )
            .otherwise(None)
            .alias("conviction_b_to_a"),
        ])
        .filter(
            (pl.col("confidence_a_to_b") >= config.min_confidence) |
            (pl.col("confidence_b_to_a") >= config.min_confidence)
        )
        .filter(
            (pl.col("lift") >= config.min_lift) | (pl.col("lift").is_null())
        )
        .sort("support", descending=True)
    )


def product_affinity_analysis(
    validated: pl.LazyFrame,
    snapshots: pl.LazyFrame,
    cutoff: str,
    config: Optional[ProductAffinityConfig] = None,
    segment_filter: Optional[str] = None,
) -> dict:
    """Run complete product affinity analysis for a snapshot.

    Args:
        validated: Validated transaction data (lazy)
        snapshots: Labeled snapshot data for the cutoff (lazy)
        cutoff: Snapshot cutoff date (ISO format)
        config: ProductAffinityConfig (uses defaults if None)
        segment_filter: Optional behavior/trajectory/value_tier/lifecycle/evidence value to filter

    Returns:
        Dict with keys:
        - 'product_frequent': DataFrame of frequent products
        - 'product_pairs': DataFrame of product association rules
        - 'category_frequent': DataFrame of frequent categories (if config.category_level)
        - 'category_pairs': DataFrame of category association rules (if config.category_level)
        - 'config': Config used
        - 'segment': Segment filter applied (if any)
    """
    if config is None:
        config = ProductAffinityConfig()

    # Filter snapshot to segment if requested
    if segment_filter:
        snapshots = snapshots.filter(pl.any_horizontal([
            pl.col("behavior") == segment_filter,
            pl.col("trajectory") == segment_filter,
            pl.col("value_tier") == segment_filter,
            pl.col("lifecycle") == segment_filter,
            pl.col("evidence") == segment_filter,
        ]))

    # Get customer IDs in segment
    segment_customers = collect(snapshots.select("customer_id").unique())
    segment_ids = segment_customers["customer_id"].to_list()

    if not segment_ids:
        return {
            "product_frequent": pl.DataFrame(),
            "product_pairs": pl.DataFrame(),
            "category_frequent": pl.DataFrame() if config.category_level else None,
            "category_pairs": pl.DataFrame() if config.category_level else None,
            "config": config,
            "segment": segment_filter,
            "customer_count": 0,
        }

    # Filter validated data to segment customers and date window
    cutoff_date = pl.lit(cutoff).str.to_date()
    window_start = cutoff_date - pl.duration(days=WINDOW_YEAR - 1)

    segment_validated = validated.filter(
        pl.col("customer_id").is_in(segment_ids) &
        pl.col("date").is_between(window_start, cutoff_date)
    )

    # Build baskets with product_id and category_key
    baskets = (
        segment_validated.group_by(["customer_id", "transaction_id", "date"])
        .agg(
            pl.col("product_id").unique().alias("products"),
            pl.col("category_key").unique().alias("categories"),
            pl.col("value").sum().alias("spend"),
        )
        .with_columns(
            pl.col("products").list.len().alias("product_count"),
            pl.col("categories").list.len().alias("category_count"),
        )
        .filter(
            (pl.col("product_count") > 0) & (pl.col("category_count") > 0)
        )
    )

    # Product-level analysis
    # Explode products for product-level basket view
    product_baskets = baskets.explode("products").rename({"products": "product_id"})
    product_frequent = compute_frequent_items(
        product_baskets, "product_id", config, cutoff, WINDOW_YEAR
    )
    product_pairs = compute_item_pairs(
        product_baskets, "product_id", product_frequent, config, cutoff, WINDOW_YEAR
    )

    result = {
        "product_frequent": product_frequent,
        "product_pairs": product_pairs,
        "config": config,
        "segment": segment_filter,
        "customer_count": len(segment_ids),
    }

    # Category-level analysis
    if config.category_level:
        category_baskets = baskets.explode("categories").rename({"categories": "category_key"})
        category_frequent = compute_frequent_items(
            category_baskets, "category_key", config, cutoff, WINDOW_YEAR
        )
        category_pairs = compute_item_pairs(
            category_baskets, "category_key", category_frequent, config, cutoff, WINDOW_YEAR
        )
        result["category_frequent"] = category_frequent
        result["category_pairs"] = category_pairs

    return result


def cross_sell_recommendations(
    product_pairs: pl.DataFrame,
    category_pairs: Optional[pl.DataFrame] = None,
    top_n: int = 20,
    min_lift: float = 1.2,
    min_confidence: float = 0.15,
) -> dict:
    """Generate cross-sell recommendations from association rules.

    Args:
        product_pairs: DataFrame from compute_item_pairs (product level)
        category_pairs: Optional DataFrame from compute_item_pairs (category level)
        top_n: Number of top recommendations per item
        min_lift: Minimum lift threshold
        min_confidence: Minimum confidence threshold

    Returns:
        Dict with 'product_recommendations' and 'category_recommendations' DataFrames
    """
    def _get_recs(pairs: pl.DataFrame) -> pl.DataFrame:
        if pairs.is_empty():
            return pl.DataFrame()

        # Filter by quality thresholds
        filtered = pairs.filter(
            (pl.col("lift") >= min_lift) &
            ((pl.col("confidence_a_to_b") >= min_confidence) |
             (pl.col("confidence_b_to_a") >= min_confidence))
        )

        if filtered.is_empty():
            return pl.DataFrame()

        # Generate directional recommendations
        recs_a_to_b = filtered.select([
            pl.col("item_a").alias("source_item"),
            pl.col("item_b").alias("target_item"),
            pl.col("support"),
            pl.col("confidence_a_to_b").alias("confidence"),
            pl.col("lift"),
            pl.col("conviction_a_to_b").alias("conviction"),
            pl.lit("A_to_B").alias("direction"),
        ])
        recs_b_to_a = filtered.select([
            pl.col("item_b").alias("source_item"),
            pl.col("item_a").alias("target_item"),
            pl.col("support"),
            pl.col("confidence_b_to_a").alias("confidence"),
            pl.col("lift"),
            pl.col("conviction_b_to_a").alias("conviction"),
            pl.lit("B_to_A").alias("direction"),
        ])

        all_recs = pl.concat([recs_a_to_b, recs_b_to_a])
        return collect(
            all_recs.filter(pl.col("confidence").is_not_null())
            .sort(["source_item", "confidence"], descending=[False, True])
            .group_by("source_item")
            .head(top_n)
            .sort("source_item", "confidence", descending=[False, True])
        )

    return {
        "product_recommendations": _get_recs(product_pairs),
        "category_recommendations": _get_recs(category_pairs) if category_pairs is not None else pl.DataFrame(),
    }


def new_to_repeat_pathways(
    validated: pl.LazyFrame,
    snapshots: pl.LazyFrame,
    cutoff: str,
    min_trials: int = 5,
) -> pl.DataFrame:
    """Analyze new-to-repeat product pathways.

    Identifies products that customers tend to buy AFTER their first purchase
    of a given product/category. Useful for cross-sell sequencing.

    Args:
        validated: Validated transaction data
        snapshots: Labeled snapshot data for the cutoff
        cutoff: Snapshot cutoff date
        min_trials: Minimum trials for a product to be considered

    Returns:
        DataFrame with pathway analysis
    """
    cutoff_date = pl.lit(cutoff).str.to_date()
    window_start = cutoff_date - pl.duration(days=WINDOW_YEAR - 1)

    # Get all transactions in window for snapshot customers
    cust_ids = collect(snapshots.select("customer_id").unique())["customer_id"].to_list()
    if not cust_ids:
        return pl.DataFrame()

    txns = validated.filter(
        pl.col("customer_id").is_in(cust_ids) &
        pl.col("date").is_between(window_start, cutoff_date)
    ).select("customer_id", "date", "product_id", "category_key")

    # For each customer, order their transactions
    ordered = txns.sort(["customer_id", "date", "transaction_id"]).with_columns(
        pl.col("product_id").shift(1).over("customer_id", order_by="date").alias("prev_product"),
        pl.col("category_key").shift(1).over("customer_id", order_by="date").alias("prev_category"),
    ).filter(pl.col("prev_product").is_not_null())

    # Product -> next product transitions
    product_paths = collect(
        ordered.group_by("prev_product", "product_id")
        .agg(pl.len().alias("transition_count"))
        .sort("transition_count", descending=True)
    )

    # Category -> next category transitions
    category_paths = collect(
        ordered.group_by("prev_category", "category_key")
        .agg(pl.len().alias("transition_count"))
        .sort("transition_count", descending=True)
    )

    # Filter by minimum trials
    product_paths = product_paths.filter(pl.col("transition_count") >= min_trials)
    category_paths = category_paths.filter(pl.col("transition_count") >= min_trials)

    return {
        "product_transitions": product_paths,
        "category_transitions": category_paths,
    }


def basket_composition_by_segment(
    snapshots: pl.LazyFrame,
    validated: pl.LazyFrame,
    cutoff: str,
    dimensions: list[str] = ["behavior", "value_tier", "lifecycle"],
) -> dict:
    """Analyze basket composition (product/category mix) by customer segment.

    Args:
        snapshots: Labeled snapshot data
        validated: Validated transaction data
        cutoff: Snapshot cutoff date
        dimensions: Segment dimensions to analyze

    Returns:
        Dict mapping dimension -> DataFrame with composition stats
    """
    cutoff_date = pl.lit(cutoff).str.to_date()
    window_start = cutoff_date - pl.duration(days=WINDOW_RECENT - 1)

    # Get recent transactions for snapshot customers
    recent_txns = validated.filter(
        pl.col("date").is_between(window_start, cutoff_date)
    ).join(
        snapshots.select("customer_id"), on="customer_id", how="inner"
    )

    results = {}
    for dim in dimensions:
        if dim not in snapshots.collect_schema().names():
            continue

        # Join segment labels
        txns_with_segment = recent_txns.join(
            snapshots.select("customer_id", dim), on="customer_id", how="inner"
        )

        # Product composition per segment
        product_comp = collect(
            txns_with_segment.group_by(dim, "product_id")
            .agg(
                pl.col("customer_id").n_unique().alias("buyers"),
                pl.col("quantity").sum().alias("units"),
                pl.col("value").sum().alias("spend"),
            )
            .with_columns(
                pl.col("spend").sum().over(dim).alias("segment_spend"),
                (pl.col("spend") / pl.col("segment_spend")).alias("spend_share"),
            )
            .sort([dim, "spend_share"], descending=[False, True])
        )

        # Category composition per segment
        category_comp = collect(
            txns_with_segment.group_by(dim, "category_key")
            .agg(
                pl.col("customer_id").n_unique().alias("buyers"),
                pl.col("quantity").sum().alias("units"),
                pl.col("value").sum().alias("spend"),
            )
            .with_columns(
                pl.col("spend").sum().over(dim).alias("segment_spend"),
                (pl.col("spend") / pl.col("segment_spend")).alias("spend_share"),
            )
            .sort([dim, "spend_share"], descending=[False, True])
        )

        results[dim] = {
            "product_composition": product_comp,
            "category_composition": category_comp,
        }

    return results


# Convenience function for dashboard integration
def run_product_affinity_dashboard(
    validated: pl.LazyFrame,
    snapshots: pl.LazyFrame,
    cutoff: str,
    min_support: float = 0.001,
    min_confidence: float = 0.1,
    min_lift: float = 1.0,
    top_k_pairs: int = 2000,
    segment: Optional[str] = None,
) -> dict:
    """Dashboard-friendly product affinity analysis.

    Uses sensible defaults for interactive exploration.
    """
    config = ProductAffinityConfig(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        top_k_pairs=top_k_pairs,
        category_level=True,
    )
    affinity_result = product_affinity_analysis(
        validated, snapshots, cutoff, config, segment
    )
    recommendations = cross_sell_recommendations(
        affinity_result["product_pairs"],
        affinity_result.get("category_pairs"),
    )
    pathways = new_to_repeat_pathways(validated, snapshots, cutoff)

    return {
        **affinity_result,
        "recommendations": recommendations,
        "pathways": pathways,
    }


def collect(frame: pl.LazyFrame) -> pl.DataFrame:
    """Collect with streaming engine."""
    return frame.collect(engine="streaming")