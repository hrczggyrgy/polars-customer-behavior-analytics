"""Statistical helpers for the retail analytics pipeline."""

from __future__ import annotations

import math

WILSON_Z = 1.959963984540054


def wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion (95% confidence).

    Args:
        successes: Number of successes (must be >= 0)
        total: Total number of trials (must be >= 0)

    Returns:
        Tuple of (lower_bound, upper_bound) as floats in [0, 1],
        or (None, None) if total <= 0.

    Raises:
        ValueError: If successes < 0 or successes > total.
    """
    if total <= 0:
        return None, None
    if not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts: successes must be in [0, total]")

    p = successes / total
    denom = 1 + WILSON_Z * WILSON_Z / total
    center = (p + WILSON_Z * WILSON_Z / (2 * total)) / denom
    half = (
        WILSON_Z
        * math.sqrt(p * (1 - p) / total + WILSON_Z * WILSON_Z / (4 * total * total))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Safe division: returns None when denominator is None or <= 0.

    Args:
        numerator: Numerator value
        denominator: Denominator value

    Returns:
        numerator / denominator if denominator > 0, otherwise None
    """
    if denominator is None or denominator <= 0:
        return None
    if numerator is None:
        return None
    return numerator / denominator
