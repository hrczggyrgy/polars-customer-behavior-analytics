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

from typing import Literal

from retail_analytics.core.pipeline.features import (
    _compute_baskets,
    add_derived_features,
    add_future_outcomes,
    compute_base_features,
    compute_breadth_last6days,
    compute_category_aggregations,
    compute_category_features,
    compute_day_features,
    compute_habit_features,
    compute_price_features,
    compute_product_features,
    compute_quantity_features,
    compute_rhythm_features,
    compute_weekday_entropy,
    merge_features,
)
from retail_analytics.core.pipeline.labeling import apply_labels as label

# Re-export from pipeline modules for backward compatibility
from retail_analytics.core.pipeline.prepare import (
    _compute_quality_audit,
    _compute_validity_mask,
    _handle_price_quantity,
    _normalize_strings,
    _parse_dates,
    _rejection_reasons,
    _sink,
    _taglist,
    _write_partitions,
    prepare,
    sink,
    validate_input_schema,
)
from retail_analytics.core.pipeline.reporting import generate_reports as reports
from retail_analytics.core.pipeline.run import main
from retail_analytics.core.pipeline.run import run_pipeline as run
from retail_analytics.core.pipeline.snapshot import generate_snapshot as snapshot
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
