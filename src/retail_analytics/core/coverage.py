"""Coverage and horizon eligibility checks."""

from datetime import date, timedelta


def check_horizon_eligibility(
    manifest: dict, horizon_days: int, cutoff_date: date = None
) -> tuple[bool, str]:
    """
    Check if a future-purchase horizon is eligible given coverage confirmation.

    Returns:
        (eligible: bool, reason: str)
    """
    coverage_confirmed = manifest.get("coverage_confirmed", False)
    if not coverage_confirmed:
        return False, "Coverage not confirmed — future outcomes unavailable"

    coverage_complete_through = manifest.get("coverage_complete_through")
    if not coverage_complete_through:
        return False, "Complete-through date not set"

    complete_date = date.fromisoformat(coverage_complete_through)

    if cutoff_date is None:
        # Use latest cutoff from manifest
        cutoff_date = date.fromisoformat(manifest["snapshots"][-1])

    horizon_end = cutoff_date + timedelta(days=horizon_days)

    if horizon_end > complete_date:
        return False, (
            f"Insufficient observed follow-up: cutoff {cutoff_date} + {horizon_days}d "
            f"= {horizon_end} exceeds complete-through {complete_date}"
        )

    return True, f"Eligible: complete-through {complete_date} covers horizon end {horizon_end}"
