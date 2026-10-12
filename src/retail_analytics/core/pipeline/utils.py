"""Shared utilities for the retail analytics pipeline.

This module consolidates common utilities used across pipeline modules
to eliminate duplication and provide a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

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


def _safe_ratio(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Safe division: returns null when denominator is null or <= 0."""
    return pl.when(b.is_not_null() & (b > 0)).then(a / b).otherwise(None)
