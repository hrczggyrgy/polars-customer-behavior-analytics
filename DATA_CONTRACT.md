# Data Contract

This document specifies the input requirements, output schemas, and validation rules for the pipeline.

---

## Input Specification

### Required Columns (9 columns, exact names)

| Column | Type | Description | Nullable | Validation |
|--------|------|-------------|----------|------------|
| `customer_id` | String | Customer identifier | No | Non-null, non-empty after trim |
| `transaction_date` | Date / Datetime / String | Transaction timestamp | No | Parsable to Date |
| `transaction_id` | String | Transaction/basket identifier | No | Non-null, non-empty after trim |
| `product_id` | String | Product identifier | No | Non-null, non-empty after trim |
| `product_description` | String | Product description | Yes | Empty string if null |
| `department` | String | Department name | No | Filled with `__UNKNOWN__` if null/empty |
| `category` | String | Category name | No | Filled with `__UNKNOWN__` if null/empty |
| `price` | Numeric (Float64) | Unit price or line total | No | Finite, ≥ 0 |
| `quantity` | Numeric (Float64) | Quantity sold | No | Finite, > 0 |

### Date Handling

- **Accepted formats**: `Date`, `Datetime` (any time unit, timezone-naive), or `String`
- **String parsing**: If `String`, uses `--date-format` if provided; otherwise tries ISO `YYYY-MM-DD` (first 10 chars)
- **Epoch timestamps**: NOT accepted (no automatic unit detection)
- **Timezone**: Discarded; only date component used
- **Output date column**: `date` (Date type, derived from `transaction_date`)

### Price Mode Semantics

| Mode | Interpretation | Line Value Formula |
|------|----------------|-------------------|
| `unit` | `price` = unit selling price | `price × quantity` |
| `line_total` | `price` = line total amount | `price` |

**Critical**: The pipeline does not detect or validate which mode matches your data. You must know your data's price meaning.

### Row Validation Rules

A row is **valid** only if ALL pass:
1. `customer_id`, `transaction_id`, `product_id` non-null and non-empty after whitespace trim
2. `transaction_date` parses to a valid `Date`
3. `price` is finite and `≥ 0`
4. `quantity` is finite and `> 0`
5. Computed `value` (per price mode) is finite

Rows failing any rule are **rejected** and written to `rejected.parquet`.

### Duplicate Handling

- **Basket key**: `(customer_id, transaction_id, date)` — must be unique per basket
- **Validation**: After cleaning, pipeline checks that each basket key maps to exactly one date
- **Conflict**: If same `(customer_id, transaction_id)` has multiple dates → **error** (pipeline stops)
- **Exact duplicates**: Rows identical on all columns (including `product_id`) are preserved and counted in `quality.json` as `preserved_duplicate_extras`
- **No automatic deduplication**: User must preprocess if deduplication needed

### Product Variation Audit

- Checks for `product_id` with multiple `(department, category)` combinations or multiple `product_description` values
- Written to `product_variation.parquet` (informational only; does not stop pipeline)

---

## Output Specification

### 1. quality.json

```json
{
  "source_rows": 2595732,
  "rejected_rows": 15330,
  "first_date": "2000-01-01",
  "last_date": "2001-12-11",
  "observed_through": "2001-12-11",
  "coverage_confirmed": false,
  "coverage_interpretation": "Future outcomes are unavailable unless coverage_confirmed is true; a maximum observed date alone does not establish complete capture.",
  "preserved_duplicate_extras": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `source_rows` | Integer | Total input rows |
| `rejected_rows` | Integer | Rows failing validation |
| `first_date` | Date (string) | Min valid transaction date |
| `last_date` | Date (string) | Max valid transaction date |
| `observed_through` | Date (string) | Effective analysis end date |
| `coverage_confirmed` | Boolean | CLI `--coverage-confirmed` value |
| `coverage_interpretation` | String | Explanatory note |
| `preserved_duplicate_extras` | Integer | Duplicate row count beyond first per unique key |

**Invariant**: `source_rows = valid_rows + rejected_rows` (where `valid_rows` = rows in `validated.parquet`)

---

### 2. rejected.parquet

Schema: All 9 input columns + derived columns

| Column | Type | Notes |
|--------|------|-------|
| `customer_id` | String | Original (trimmed) |
| `transaction_date` | Datetime | Original |
| `transaction_id` | String | Original (trimmed) |
| `product_id` | String | Original (trimmed) |
| `product_description` | String | Original |
| `department` | String | Original (with `__UNKNOWN__` fill) |
| `category` | String | Original (with `__UNKNOWN__` fill) |
| `price` | Float64 | Original |
| `quantity` | Float64 | Original |
| `date` | Date | Parsed date (null if parse failed) |
| `value` | Float64 | Computed value (null if computation failed) |
| `valid` | Boolean | Always `false` |

**Known gap**: No `rejection_reason` column. Each rejected row's specific failure reason is not recorded.

---

### 3. validated.parquet

Cleaned, validated transactions ready for analysis.

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | String | Trimmed, non-null |
| `transaction_id` | String | Trimmed, non-null |
| `product_id` | String | Trimmed, non-null |
| `product_description` | String | Non-null (empty string if was null) |
| `department` | String | Non-null (`__UNKNOWN__` if was null/empty) |
| `category` | String | Non-null (`__UNKNOWN__` if was null/empty) |
| `price` | Float64 | Finite, ≥ 0 |
| `quantity` | Float64 | Finite, > 0 |
| `date` | Date | Parsed transaction date |
| `value` | Float64 | Line value per price mode |
| `unit_price` | Float64 | `value / quantity` (for `unit` mode: equals `price`; for `line_total`: derived) |
| `category_key` | String | Composite key: `len(dept):deptlen(cat):cat` |
| `bucket` | Int64 | Hash partition: `hash(customer_id) % buckets` |

Partitioned by `bucket` into `partitions/b{N}.parquet`.

---

### 4. price_references/{snapshot_date}.parquet

Per-snapshot reference prices for price feature computation.

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | String | Product identifier |
| `reference_price` | Float64 | Median `unit_price` in [C-364, C] |
| `reference_count` | Int64 | Number of observations in window |

Used for `relative_paid_price` feature (requires `reference_count ≥ 5`).

---

### 5. snapshots/{snapshot_date}/b{N}.parquet

Customer feature snapshots (78 columns). One file per bucket per snapshot.

#### Identity & Temporal
| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | String | Customer identifier |
| `snapshot` | Date | Cutoff date C |
| `first_purchase` | Date | First purchase date ≤ C |
| `last_purchase` | Date | Last purchase date ≤ C |
| `tenure` | Int64 | C - first_purchase (days) |
| `recency` | Int64 | C - last_purchase (days) |
| `version` | String | Pipeline version |

#### Historical Counts & Spend (windows include C)
| Column | Type | Window |
|--------|------|--------|
| `purchases_history` | UInt32 | All time ≤ C |
| `spend_history` | Float64 | All time ≤ C |
| `purchases_30d` | UInt32 | [C-29, C] |
| `spend_30d` | Float64 | [C-29, C] |
| `purchases_90d` | UInt32 | [C-89, C] |
| `spend_90d` | Float64 | [C-89, C] |
| `purchases_365d` | UInt32 | [C-364, C] |
| `spend_365d` | Float64 | [C-364, C] |
| `purchases_previous90d` | UInt32 | [C-179, C-90] |
| `spend_previous90d` | Float64 | [C-179, C-90] |

#### Basket Metrics (365d window)
| Column | Type | Description |
|--------|------|-------------|
| `aov_365d` | Float64 | Mean basket value |
| `median_basket` | Float64 | Median basket value |
| `basket_cv` | Float64 | CV of basket values |
| `median_breadth` | Float64 | Median distinct products/basket |
| `breadth_recent` | Float64 | Mean breadth in 90d |
| `breadth_previous` | Float64 | Mean breadth in previous 90d |

#### Daily Rhythm Features (365d window with usability filter)
| Column | Type | Description |
|--------|------|-------------|
| `days_history` | UInt32 | Total shopping days |
| `days_365d` | UInt32 | Shopping days in 365d |
| `gap_count` | UInt32 | Usable gaps (prior ≥ C-365) |
| `median_gap` | Float64 | Median gap (days) |
| `gap_cv` | Float64 | CV of gaps |
| `product_overlap` | Float64 | Median product Jaccard |
| `category_overlap` | Float64 | Median category Jaccard |
| `weekend_share` | Float64 | Weekend shopping day proportion |
| `personal_spend_ratio` | Float64 | Median day_spend / baseline_spend |
| `reactivation_events` | UInt32 | Gaps ≥ 2× baseline in last 30d |

#### Weekday & Breadth
| Column | Type | Description |
|--------|------|-------------|
| `weekday_entropy` | Float64 | Shannon entropy of weekday distribution |
| `products_last6days` | UInt32 | Distinct products in last 6 shopping days |
| `categories_last6days` | UInt32 | Distinct categories in last 6 shopping days |

#### Trial & Product Habits (365d window)
| Column | Type | Description |
|--------|------|-------------|
| `eligible_trials` | UInt32 | Products with trial in [C-365, C-90] |
| `repeated_trials` | UInt32 | Eligible trials with repeat ≤ 90d |
| `trial_conversion` | Float64 | `repeated_trials / eligible_trials` |
| `core_size` | UInt32 | Products with ≥ 3 purchase days |
| `core_spend` | Float64 | Spend on core products |
| `core_spend_share` | Float64 | `core_spend / spend_365d` |
| `new_share` | Float64 | Basket proportion with new products |
| `repeat_share` | Float64 | Basket proportion all repeat |
| `established_share` | Float64 | Basket proportion all established |

#### Quantity Rhythm
| Column | Type | Description |
|--------|------|-------------|
| `gap_after_large_qty` | Float64 | Median gap after qty ≥ 1.5× baseline |
| `gap_after_typical_qty` | Float64 | Median gap after qty 0.75–1.25× baseline |
| `large_qty_gap_count` | UInt32 | Count of large-qty gaps |

#### Category Features (365d window)
| Column | Type | Description |
|--------|------|-------------|
| `category_count` | UInt32 | Categories with ≥ 1 shopping day |
| `category_hhi` | Float64 | Sum of squared spend shares |
| `category_entropy` | Float64 | Shannon entropy (natural log) |
| `dominant_share` | Float64 | Max category spend share |
| `overdue_categories` | UInt32 | Categories with lateness ≥ 2 and days ≥ 3 |
| `dominant_department` | String | Dept of top-spend category |
| `dominant_category` | String | Top-spend category |

#### Price Features
| Column | Type | Description |
|--------|------|-------------|
| `relative_paid_price` | Float64 | Median paid/reference price (ref_count ≥ 5) |
| `comparable_price_lines` | UInt32 | Lines with sufficient reference data |

#### Derived Change Metrics (require `comparable_windows`)
| Column | Type | Description |
|--------|------|-------------|
| `aov_recent` | Float64 | `spend_90d / purchases_90d` |
| `aov_previous` | Float64 | `spend_previous90d / purchases_previous90d` |
| `frequency_change` | Float64 | `purchases_90d / purchases_previous90d - 1` |
| `basket_change` | Float64 | `aov_recent / aov_previous - 1` |
| `spend_change` | Float64 | `spend_90d / spend_previous90d - 1` |
| `lateness` | Float64 | `recency / median_gap` |
| `comparable_windows` | Boolean | `tenure ≥ 179` |
| `breadth_change` | Float64 | `breadth_recent - breadth_previous` |

#### Future Follow-up (Post-Cutoff, Exclusive of C)
| Column | Type | Description |
|--------|------|-------------|
| `future_purchases_30d` | UInt32 | Baskets in (C, C+30] |
| `future_spend_30d` | Float64 | Spend in (C, C+30] |
| `eligible_followup_30d` | Boolean | Coverage confirmed AND C+30 ≤ observed_through |
| `future_purchases_90d` | UInt32 | Baskets in (C, C+90] |
| `future_spend_90d` | Float64 | Spend in (C, C+90] |
| `eligible_followup_90d` | Boolean | Coverage confirmed AND C+90 ≤ observed_through |

**When not eligible**: Future columns are `null`, eligible flags are `false`.

#### Labels
| Column | Type | Description |
|--------|------|-------------|
| `style_eligible` | Boolean | `days_365d ≥ 6 AND tenure ≥ 90 AND gap_count ≥ 4` |
| `behavioral_tags` | List(String) | Applicable style tags (precedence order) |
| `trajectory_tags` | List(String) | Applicable trajectory tags |
| `lifecycle` | String | `first_observed` / `one_shopping_day` / `early_repeat` / `established_repeat` |
| `evidence` | String | `stronger` / `limited` / `insufficient` |
| `behavior` | String | Joined `behavioral_tags` or `insufficient_history` / `mixed_other` |
| `trajectory` | String | Joined `trajectory_tags` or `not_classified` |
| `value_tier` | String | `high` / `middle` / `low` / `insufficient_exposure` / `no_recent_spend` / `reference_unavailable` |

---

### 6. segment_profiles.parquet

Per-snapshot, per-behavior segment aggregates.

| Column | Type | Description |
|--------|------|-------------|
| `snapshot` | Date | Cutoff date |
| `behavior` | String | Behavior label |
| `customers` | UInt32 | Customer count in segment |
| `revenue` | Float64 | Sum of `spend_90d` |
| `median_spend` | Float64 | Median `spend_90d` |
| `median_gap` | Float64 | Median `median_gap` |
| `exploration` | Float64 | Mean `new_share` |
| `mean_customer_trial_conversion` | Float64 | Mean `trial_conversion` (null-excluding) |
| `future_purchase_rate_30d` | Float64 | Proportion with future purchase (eligible only) |
| `future_purchase_rate_90d` | Float64 | Proportion with future purchase (eligible only) |
| `future_mean_spend_30d` | Float64 | Mean future spend (eligible only) |
| `future_mean_spend_90d` | Float64 | Mean future spend (eligible only) |
| `eligible_customers_30d` | UInt32 | Count with `eligible_followup_30d` |
| `eligible_customers_90d` | UInt32 | Count with `eligible_followup_90d` |
| `customer_share` | Float64 | `customers / total_customers` |
| `revenue_share` | Float64 | `revenue / total_revenue` |

---

### 7. cohort_profiles.parquet

Per-snapshot, per-first-purchase-month aggregates.

| Column | Type | Description |
|--------|------|-------------|
| `snapshot` | Date | Cutoff date |
| `cohort` | Date | First purchase month (truncated to 1st) |
| `customers` | UInt32 | Customers in cohort at snapshot |
| `mean_cumulative_spend` | Float64 | Mean `spend_history` |
| `rolling30d_purchase_rate` | Float64 | Proportion with `purchases_30d > 0` |
| `future30d_purchase_rate` | Float64 | Proportion with `future_purchases_30d > 0` (eligible only; null if not) |

**Critical**: Cohorts have unequal observation windows. `cohort` date vs `snapshot` date determines exposure. Do not compare raw rates across cohorts with different tenures.

---

### 8. segment_transitions.parquet

Transition counts and rates between snapshots.

| Column | Type | Description |
|--------|------|-------------|
| `from_date` | Date | Previous snapshot |
| `snapshot` | Date | Current snapshot |
| `dimension` | String | `behavior` / `trajectory` / `lifecycle` / `value_tier` |
| `from_label` | String | Label at `from_date` |
| `to_label` | String | Label at `snapshot` |
| `customers` | UInt32 | Transition count |
| `eligible_origin` | UInt32 | Total with `from_label` at `from_date` |
| `transition_rate` | Float64 | `customers / eligible_origin` |

---

### 9. thresholds.json

Quantile thresholds computed at first snapshot.

```json
{
  "value_low": 149.33,
  "value_high": 701.44,
  "basket_low": 12.65,
  "basket_high": 32.05,
  "gap_low": 4.0,
  "gap_high": 9.0,
  "reference_date": "2000-06-30"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `value_low` | Float64 | 33rd percentile of `spend_90d` (tenure ≥ 89, spend_90d > 0) |
| `value_high` | Float64 | 80th percentile of `spend_90d` (same population) |
| `basket_low` | Float64 | 33rd percentile of `median_basket` (style-eligible) |
| `basket_high` | Float64 | 80th percentile of `median_basket` (style-eligible) |
| `gap_low` | Float64 | 33rd percentile of `median_gap` (style-eligible) |
| `gap_high` | Float64 | 80th percentile of `median_gap` (style-eligible) |
| `reference_date` | Date (string) | Snapshot where thresholds computed |

---

### 10. configuration.json

Pipeline run configuration.

```json
{
  "source": "customer_transactions.parquet",
  "price_mode": "unit",
  "snapshots": ["2000-06-30", "2001-06-30", "2001-12-11"],
  "buckets": 4,
  "exclude_invalid": true,
  "date_format": null,
  "observed_through": null,
  "version": "reconstructed-1.1"
}
```

---

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | Yes | — | Input Parquet file path |
| `--output` | Yes | — | Output directory (must be empty or not exist) |
| `--price-mode` | Yes | — | `unit` or `line_total` |
| `--snapshots` | Yes | — | One or more cutoff dates (YYYY-MM-DD) |
| `--buckets` | No | 8 | Hash partitions (1–32) |
| `--exclude-invalid` | No | false | Continue despite rejected rows |
| `--date-format` | No | null | Strptime format for string dates |
| `--observed-through` | No | max date | Analysis end date (≤ max valid date) |
| `--coverage-confirmed` | No | false | Enable future follow-up metrics |

---

## Error Conditions

| Condition | Exit Code | Message Pattern |
|-----------|-----------|-----------------|
| Missing input file | 1 | File not found |
| Missing required columns | 1 | "Missing columns: ..." |
| Invalid date format | 1 | "Dates must be Date, Datetime, or strings..." |
| Empty input | 1 | "Empty input" |
| No valid records | 1 | "No valid records" |
| Rejected rows (without `--exclude-invalid`) | 1 | "Invalid rows: inspect quality.json and rejected.parquet" |
| Basket key conflicts | 1 | "Customer/transaction key spans dates..." |
| Cutoff > observed_through | 1 | "Cutoff exceeds observed_through" |
| Invalid buckets/price_mode | 1 | "Invalid buckets or price_mode" |
| Output directory not empty | 1 | "Output directory must be empty" |

---

## Reproducibility Requirements

1. **Deterministic output ordering**: All outputs sorted by `customer_id` (and `snapshot` where applicable) before write
2. **Fixed hash seed**: `hash(seed=17)` for bucketing; `hash(seed=29)` for dashboard sampling
3. **Quantile interpolation**: Polars default (nearest) — thresholds reproducible for fixed input
4. **Version tracking**: `version` field in outputs and `configuration.json`
5. **No random state**: No stochastic algorithms used

---

## Methodological Limitations (Must Acknowledge)

1. **Purchase-only assumption**: No return/refund/cancellation detection. All valid rows = purchases.
2. **Relative segmentation**: Quantile thresholds are population-relative; not absolute benchmarks.
3. **Cohort exposure bias**: Cohort metrics not age-adjusted; immature cohorts appear worse.
4. **No causal inference**: Descriptive associations only. Segment differences ≠ treatment effects.
5. **Future outcomes conditional**: Require explicit `coverage_confirmed`; otherwise null.
6. **Basket grain assumption**: `(customer_id, transaction_id, date)` = one shopping occasion.
7. **Currency opaque**: No currency metadata; amounts in input units.
8. **Timezone naive**: Datetimes truncated to date; no TZ conversion.