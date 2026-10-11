"""Centralized constants for the retail analytics pipeline.

All window sizes, eligibility thresholds, and quantile definitions live here
to ensure consistency across the core pipeline, dashboard backend, and patronage layer.
"""

from __future__ import annotations

# Window sizes (in days)
WINDOW_RECENT = 90
WINDOW_PREVIOUS = 90
WINDOW_YEAR = 365
WINDOW_TWO_YEARS = 730

# Window boundaries (inclusive): N-day window = [cutoff - (N-1), cutoff]
# Previous 90d window = [cutoff - 179, cutoff - 90]
WINDOW_PREVIOUS_START_OFFSET = WINDOW_RECENT + WINDOW_PREVIOUS - 1  # 179

# Minimum tenure for various eligibility checks
MIN_TENURE_STYLE = 90  # style_eligible: tenure >= 90
MIN_TENURE_VALUE_TIER = 89  # value_tier reference population: tenure >= 89
MIN_TENURE_COMPARABLE = WINDOW_PREVIOUS_START_OFFSET  # comparable_windows: tenure >= 179

# Minimum history requirements
MIN_DAYS_STYLE = 6  # style_eligible: days_365d >= 6
MIN_GAP_COUNT_STYLE = 4  # style_eligible: gap_count >= 4

# Quantile thresholds (computed at reference snapshot)
QUANTILE_VALUE_LOW = 0.33  # 33rd percentile for value_tier low
QUANTILE_VALUE_HIGH = 0.80  # 80th percentile for value_tier high
QUANTILE_STYLE_LOW = 0.33  # 33rd percentile for basket/gap low
QUANTILE_STYLE_HIGH = 0.80  # 80th percentile for basket/gap high

# Minimum reference population size for threshold computation
MIN_REFERENCE_POPULATION = 100

# Behavioral style tag thresholds
GAP_CV_ROUTINE_MAX = 0.75
ESTABLISHED_SHARE_ROUTINE_MIN = 0.70
DOMINANT_SHARE_FOCUSED_MIN = 0.70
ESTABLISHED_SHARE_FOCUSED_MIN = 0.70
MEDIAN_BREADTH_BROAD_MIN = 3
ESTABLISHED_SHARE_BROAD_MIN = 0.70
NEW_SHARE_EXPLORATION_MIN = 0.40

# Trajectory tag thresholds
FREQUENCY_CHANGE_SLOWING_MAX = -0.25
BASKET_CHANGE_SMALLER_MAX = -0.25
BREADTH_CHANGE_NARROWING_MAX = -0.5
OVERDUE_CATEGORIES_NARROWING_MIN = 2
FREQUENCY_CHANGE_EXPANDING_MIN = 0.25
BASKET_CHANGE_EXPANDING_MIN = 0.25
LATENESS_OVERDUE_MIN = 2
REACTIVATION_EVENTS_MIN = 1

# Lifecycle thresholds
MIN_TENURE_ONE_SHOPPING_DAY = 90

# Evidence thresholds
MIN_GAP_COUNT_STRONGER = 12

# Value tier thresholds
MIN_TENURE_VALUE_EXPOSURE = 89  # insufficient_exposure if tenure < 89

# Wilson interval
WILSON_Z = 1.959963984540054  # 95% confidence

# Default hash seeds for deterministic partitioning/sampling
HASH_SEED_BUCKETING = 17
HASH_SEED_SAMPLING = 29

# Censoring threshold (days between observation start and first purchase)
CENSOR_DAYS = 90

# Default minimum eligible customers for outcome estimates
DEFAULT_MIN_ELIGIBLE = 20

# Default maximum cutoffs
MAX_CUTOFFS = 12

# Default maximum buckets
MAX_BUCKETS = 32

# Default maximum upload size (bytes)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# Default dashboard page rows
DASHBOARD_PAGE_ROWS = 2000

# Default worker timeout (seconds)
WORKER_TIMEOUT = 600
