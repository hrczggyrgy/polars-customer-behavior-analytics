"""Purchase-only customer analytics from a nine-column Parquet source.

This module implements a complete customer behavior analytics pipeline:
- Data preparation and validation from 9-column transaction data
- Point-in-time snapshot generation with rich behavioral features
- Data-driven threshold calibration with uncertainty quantification
- Multi-label behavioral and trajectory classification
- Segment profiling and transition analysis

This is a backward-compatible facade that re-exports from the decomposed pipeline modules.
For new code, prefer importing directly from retail_analytics.core.pipeline.*
"""

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import polars as pl

# Re-export from pipeline modules for backward compatibility
from retail_analytics.core.pipeline.prepare import (
    prepare,
    validate_input_schema,
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
from retail_analytics.core.pipeline.snapshot import generate_snapshot as snapshot
from retail_analytics.core.pipeline.labeling import apply_labels as label
from retail_analytics.core.pipeline.reporting import generate_reports as reports
from retail_analytics.core.pipeline.run import run_pipeline as run, main

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
from retail_analytics.core.stats import safe_ratio

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

# Backward compatibility aliases
ratio = _safe_ratio = safe_ratio

__all__ = [
    "prepare",
    "validate_input_schema",
    "snapshot",
    "label",
    "reports",
    "run",
    "main",
    "C",
    "K",
    "P",
    "G",
    "COLS",
    "REQUIRED_COLS",
    "VERSION",
    "PriceMode",
    "safe_ratio",
    "ratio",
    "sink",
    "_sink",
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
]