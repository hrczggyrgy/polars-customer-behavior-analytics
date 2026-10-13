"""Shared preparation utilities for normalization and validity.

This module consolidates string normalization and validity expressions
shared between the pipeline and dashboard paths.

Key design: preserve deliberate differences between paths via parameters:
- Dashboard string-date parsing: strips whitespace
- Pipeline string-date parsing: does NOT strip whitespace
- Dashboard numeric parsing: supports decimal commas
- Pipeline: preserves/casts existing numeric/raw columns
- Validity: accept computed value expression rather than forcing materialization
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from retail_analytics.core.pipeline.utils import C, _safe_ratio

PriceMode = Literal["unit", "line_total"]


def normalize_strings(
    frame: pl.LazyFrame,
    *,
    unknown_dept_category: bool = True,
    null_product_description: bool = True,
) -> pl.LazyFrame:
    """Normalize string columns: trim, replace empty with null/unknown.

    Args:
        frame: Input LazyFrame with raw string columns
        unknown_dept_category: Replace empty department/category with "__UNKNOWN__"
        null_product_description: Replace null product_description with ""

    Returns:
        LazyFrame with normalized string columns
    """
    return frame.with_columns(
        *[
            pl.col(x).cast(pl.String).str.strip_chars().replace("", None)
            for x in [C, "transaction_id", "product_id"]
        ],
        *[
            pl.col(x).cast(pl.String).str.strip_chars().replace("", None).fill_null("__UNKNOWN__")
            for x in ["department", "category"]
        ]
        if unknown_dept_category
        else [
            pl.col(x).cast(pl.String).str.strip_chars().replace("", None)
            for x in ["department", "category"]
        ],
        pl.col("product_description").cast(pl.String).fill_null("")
        if null_product_description
        else pl.col("product_description").cast(pl.String),
    )


def parse_numeric_columns(
    frame: pl.LazyFrame,
    schema: dict[str, pl.DataType],
    *,
    decimal_comma: bool = False,
    mode: PriceMode = "unit",
) -> pl.LazyFrame:
    """Parse price/quantity columns and compute value.

    Args:
        frame: Input LazyFrame
        schema: Schema dict for checking existing columns
        decimal_comma: If True, replace ',' with '.' before casting (dashboard path)
        mode: "unit" for price*quantity, "line_total" for price as line total

    Returns:
        LazyFrame with price, quantity, value, and raw columns
    """
    # Handle price/quantity: if raw columns already exist (from dashboard canonical), use them
    if "price_raw" in schema and "quantity_raw" in schema:
        frame = frame.with_columns(
            pl.col("price").cast(pl.Float64, strict=False),
            pl.col("quantity").cast(pl.Float64, strict=False),
        )
    else:
        price_expr = pl.col("price").cast(pl.String).str.strip_chars().replace("", None)
        qty_expr = pl.col("quantity").cast(pl.String).str.strip_chars().replace("", None)
        if decimal_comma:
            price_expr = price_expr.str.replace_all(",", ".", literal=True)
            qty_expr = qty_expr.str.replace_all(",", ".", literal=True)

        frame = frame.with_columns(
            # Preserve raw string values before casting to distinguish missing vs malformed
            price_expr.alias("price_raw"),
            qty_expr.alias("quantity_raw"),
            price_expr.cast(pl.Float64, strict=False).alias("price"),
            qty_expr.cast(pl.Float64, strict=False).alias("quantity"),
        )

    frame = frame.with_columns(
        (pl.col("price") * pl.col("quantity") if mode == "unit" else pl.col("price")).alias("value")
    )
    return frame


def validity_mask(
    frame: pl.LazyFrame,
    value_expr: pl.Expr | None = None,
) -> pl.Expr:
    """Compute validity mask for rows.

    Args:
        frame: Input LazyFrame (used for column presence checks)
        value_expr: Optional pre-computed value expression. If None, uses frame["value"].

    Returns:
        Boolean expression for valid rows
    """
    value = value_expr if value_expr is not None else pl.col("value")

    required_not_null = pl.all_horizontal(
        [pl.col(x).is_not_null() for x in [C, "transaction_id", "product_id", "date"]]
    )

    return (
        required_not_null
        & pl.col("price").is_finite()
        & (pl.col("price") >= 0)
        & pl.col("quantity").is_finite()
        & (pl.col("quantity") > 0)
        & value.is_finite()
    ).fill_null(False)


def compute_quality_audit(
    raw: pl.LazyFrame,
    valid: pl.LazyFrame,
    rejected: pl.LazyFrame,
    observation_start: str | None,
    censor_days: int,
) -> dict[str, any]:
    """Compute comprehensive quality audit metrics (simplified for shared use)."""
    # This is a placeholder - the full audit is in prepare.py
    # Dashboard uses a different summary approach
    total_rows = raw.select(pl.len()).collect().item()
    rejected_count = rejected.select(pl.len()).collect().item()
    valid_count = total_rows - rejected_count

    return {
        "source_rows": total_rows,
        "rejected_rows": rejected_count,
        "valid_rows": valid_count,
    }


def audit_duplicate_counts(
    frame: pl.LazyFrame,
    group_cols: list[str],
    *,
    value_col: str | None = None,
    valid_frame: pl.LazyFrame | None = None,
) -> tuple[int, float]:
    """Audit duplicate counts and optionally attributed value.

    Args:
        frame: Frame to audit (raw for counts, valid for value attribution)
        group_cols: Columns to group by for duplicate detection
        value_col: If provided, attribute value from this column (uses valid_frame)
        valid_frame: Frame with valid rows for value attribution

    Returns:
        Tuple of (duplicate_row_count, duplicate_value_sum)
    """
    # Count duplicate rows (rows beyond first in each group)
    dup_count = (
        frame.group_by(group_cols)
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum())
        .collect()
        .item()
        or 0
    )

    # Attribute value if requested (uses valid rows only)
    dup_value = 0.0
    if value_col is not None and valid_frame is not None:
        dup_value = (
            valid_frame.group_by(group_cols)
            .agg(pl.col(value_col).first().alias("line_value"), pl.len().alias("dup_count"))
            .filter(pl.col("dup_count") > 1)
            .select((pl.col("line_value") * (pl.col("dup_count") - 1)).sum())
            .collect()
            .item()
            or 0.0
        )

    return dup_count, dup_value