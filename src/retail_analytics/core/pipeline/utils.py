"""Shared utilities for the retail analytics pipeline.

This module consolidates common utilities used across pipeline modules
to eliminate duplication and provide a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from retail_analytics.core.stats import safe_ratio

# ──────────────────────────────────────────────────────────────────────
# Version
# ──────────────────────────────────────────────────────────────────────

VERSION = "reconstructed-1.3"


# ──────────────────────────────────────────────────────────────────────
# Column key aliases (single source of truth)
# ──────────────────────────────────────────────────────────────────────

C = "customer_id"
K = [C, "transaction_id", "date"]
P = [C, "product_id"]
G = [C, "department", "category"]

# Required input columns
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


# ──────────────────────────────────────────────────────────────────────
# I/O utilities
# ──────────────────────────────────────────────────────────────────────


def _sink(frame: pl.LazyFrame, path: Path, sort_by: list[str] | str | None = None) -> None:
    """Write a LazyFrame to Parquet with ZSTD compression (internal)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sort_by is not None:
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        frame = frame.sort(sort_by)
    # Use streaming engine for reliability; non-streaming can skip empty frames
    frame.collect(engine="streaming").write_parquet(path, compression="zstd")


def sink(frame: pl.LazyFrame, path: Path, sort_by: list[str] | str | None = None) -> None:
    """Write a LazyFrame to Parquet with ZSTD compression (public API)."""
    _sink(frame, path, sort_by)


def collect(frame: pl.LazyFrame) -> pl.DataFrame:
    """Collect a LazyFrame using the streaming engine."""
    return frame.collect(engine="streaming")


# ──────────────────────────────────────────────────────────────────────
# Tag list utility
# ──────────────────────────────────────────────────────────────────────


def _taglist(pairs: list[tuple[str, pl.Expr]]) -> pl.Expr:
    """Create a list of tags from (name, boolean_expression) pairs.

    ALL matching tags are collected in order (not first-match exclusive).
    The resulting list preserves the order of the input pairs.
    """
    return pl.concat_list(
        [
            pl.when(test.fill_null(False))
            .then(pl.lit(name))
            .otherwise(pl.lit(None, dtype=pl.String))
            for name, test in pairs
        ]
    ).list.drop_nulls()


# ──────────────────────────────────────────────────────────────────────
# Safe ratio utility
# ──────────────────────────────────────────────────────────────────────


# Backward compatibility alias
_safe_ratio = safe_ratio


def _mask_zero_spend_cols(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Set category_hhi and category_entropy to null when spend_365d <= 0.
    
    This masking is applied after labeling to ensure zero-spend customers
    don't have misleading concentration/entropy values.
    """
    return frame.with_columns(
        pl.when(pl.col("spend_365d") > 0)
        .then(pl.col("category_hhi"))
        .otherwise(None)
        .alias("category_hhi"),
        pl.when(pl.col("spend_365d") > 0)
        .then(pl.col("category_entropy"))
        .otherwise(None)
        .alias("category_entropy"),
    )
