"""Pipeline modules for the retail analytics workflow.

This package decomposes the monolithic workflow into focused modules:
- prepare: Data validation, normalization, and partitioning
- features: Customer feature computation (baskets, rhythm, products, categories)
- snapshot: Point-in-time feature snapshot generation
- labeling: Behavioral and trajectory classification
- reporting: Segment profiles, cohorts, and transition analysis
- run: Main pipeline orchestration and CLI entry point
"""

from retail_analytics.core.pipeline.prepare import (
    prepare,
    validate_input_schema,
    COLS,
    REQUIRED_COLS,
    C,
    K,
    P,
    G,
    VERSION,
    PriceMode,
    _sink,
    sink,
    _taglist,
    _rejection_reasons,
    _parse_dates,
    _normalize_strings,
    _handle_price_quantity,
    _compute_validity_mask,
    _compute_quality_audit,
    _write_partitions,
)
from retail_analytics.core.pipeline.features import (
    _compute_baskets,
    compute_base_features,
    compute_day_features,
    compute_rhythm_features,
    compute_weekday_entropy,
    compute_breadth_last6days,
    compute_product_features,
    compute_habit_features,
    compute_quantity_features,
    compute_category_features,
    compute_category_aggregations,
    compute_price_features,
    merge_features,
    add_derived_features,
    add_future_outcomes,
)
from retail_analytics.core.pipeline.snapshot import generate_snapshot
from retail_analytics.core.pipeline.labeling import apply_labels
from retail_analytics.core.pipeline.reporting import generate_reports
from retail_analytics.core.pipeline.run import run_pipeline, main
from retail_analytics.core.stats import safe_ratio

# Backward compatibility aliases
ratio = safe_ratio

__all__ = [
    "prepare",
    "validate_input_schema",
    "COLS",
    "REQUIRED_COLS",
    "C",
    "K",
    "P",
    "G",
    "VERSION",
    "PriceMode",
    "_sink",
    "sink",
    "_taglist",
    "_rejection_reasons",
    "_parse_dates",
    "_normalize_strings",
    "_handle_price_quantity",
    "_compute_validity_mask",
    "_compute_quality_audit",
    "_write_partitions",
    "_compute_baskets",
    "compute_base_features",
    "compute_day_features",
    "compute_rhythm_features",
    "compute_weekday_entropy",
    "compute_breadth_last6days",
    "compute_product_features",
    "compute_habit_features",
    "compute_quantity_features",
    "compute_category_features",
    "compute_category_aggregations",
    "compute_price_features",
    "merge_features",
    "add_derived_features",
    "add_future_outcomes",
    "generate_snapshot",
    "apply_labels",
    "generate_reports",
    "run_pipeline",
    "main",
    "safe_ratio",
    "ratio",
]