# Polars Customer Behavior Analytics

A customer behavior analytics pipeline built with [Polars](https://pola.rs) for retail transaction data.

## Documentation

| Document | Description |
|--------|-------------|
| [METRICS.md](../METRICS.md) | Complete metric specifications with formulas, denominators, windows, and null semantics |
| [DATA_CONTRACT.md](../DATA_CONTRACT.md) | Input/output schemas, validation rules, and CLI reference |
| [workflow.md](workflow.md) | Pipeline architecture diagram and data flow |
| [DASHBOARD.md](DASHBOARD.md) | Streamlit dashboard usage and limitations |
| [VISUAL_INSIGHTS.md](VISUAL_INSIGHTS.md) | Visual analytics module documentation |
| [PATRONAGE.md](PATRONAGE.md) | Observed patronage layer documentation |

## Data Workflow

```mermaid
flowchart TD
    %% =============================================
    %% HIGH-LEVEL DATA PIPELINE
    %% =============================================
    
    subgraph INPUT["INPUT"]
        SRC["customer_transactions.parquet\n(2.6M rows x 9 cols)"]
    end

    subgraph PREP["PREPARATION & VALIDATION"]
        SCAN["Scan & Validate Schema"]
        PARSE["Parse Dates\n(String/Datetime to Date)"]
        CLEAN["Clean & Cast Types\n(Strip, null handling, Float64)"]
        VALUE["Compute Value\n(unit: price x qty or line_total: price)"]
        VALIDATE["Row Validation\n(Required fields, price>=0, qty>0, finite)"]
        AUDIT["Quality Audit\nto quality.json"]
        REJECT["Rejected Rows\nto rejected.parquet (with rejection_reason)"]
        ENRICH["Enrich\n(unit_price, category_key, bucket)"]
        PARTITION["Partition by Bucket\nto partitions/bN.parquet"]
        CHECKS["Integrity Checks\n- Basket conflicts\n- Product variation"]
    end

    subgraph SNAPSHOT["SNAPSHOT GENERATION (per cutoff date)"]
        PRICE_REF["Price References\n(Median unit_price per product)"]
        BASKETS["Basket Aggregation\n(customer_id, txn_id, date)\nto spend, products, breadth"]
        BASE["Customer Base Features\n- History: first/last purchase\n- Counts: purchases_30d, 90d, 365d\n- Spend: spend_30d, 90d, 365d\n- AOV, median_basket, breadth"]
        DAILY["Daily Features\n- day_spend, products list, categories list\n- gap, relative_spend\n- Jaccard overlap (products/categories)"]
        RHYTHM["Rhythm Features\n- median_gap, gap_cv\n- product/category overlap\n- weekend_share, reactivation_events"]
        WEEKDAY["Weekday Entropy (Shannon, natural log)"]
        BREADTH["Last 6 Days Breadth\n(requires ≥6 shopping days)"]
        PROD["Product-Level Features\n- trial/repeat dates\n- prior_days, next_gap\n- relative_qty"]
        TRIALS["Trial Conversion\n- eligible_trials, repeated_trials\n- core_size, core_spend"]
        CAT["Category History\n- share, HHI, entropy\n- dominant category\n- overdue_categories"]
        PRICE_FEAT["Price Features\n- relative_paid_price\n- comparable_price_lines"]
        MERGE["Merge All Features\nto 79 columns"]
        DERIVED["Derived Metrics\n- tenure, recency\n- aov_recent/previous\n- frequency/basket/spend_change\n- lateness, breadth_change"]
        FOLLOWUP["Future Followup (no lookahead)\n- future_purchases_30d, 90d\n- future_spend_30d, 90d\n- Requires coverage_confirmed"]
    end

    subgraph LABEL["LABELING & CLASSIFICATION"]
        THRESH["Quantile Thresholds\n(value/basket/gap low/high)\nComputed at first snapshot"]
        STYLE["Behavioral Style Tags (all matching tags collected in order)\n- routine_repeat (gap_cv≤.75, established≥.7)\n- focused_repeat (dominant≥.7, established≥.7)\n- broad_repeat (breadth≥3, established≥.7)\n- exploration_heavy (new_share≥.4)\n- large_basket_episodic (quantile-based)\n- small_basket_frequent (quantile-based)"]
        TRAJ["Trajectory Tags (multi-label)\n- slowing / smaller_baskets\n- narrowing / broad_decline\n- expanding / overdue / reactivated / stable"]
        LIFECYCLE["Lifecycle (mutually exclusive)\n- first_observed / one_shopping_day\n- early_repeat / established_repeat"]
        EVIDENCE["Evidence Level (mutually exclusive)\n- stronger / limited / insufficient"]
        BEHAVIOR["Behavior Label\n(joined style tags or status)"]
        TRAJ_L["Trajectory Label\n(joined trajectory tags)"]
        VALUE_TIER["Value Tier (mutually exclusive)\n- high / middle / low\n- insufficient_exposure / no_recent_spend\n- reference_unavailable"]
    end

    subgraph REPORTS["REPORTING"]
        SEG["Segment Profiles\n(customers, revenue, rates by behavior)\nIncludes customer_share, revenue_share"]
        COHORT["Cohort Profiles\n(mean spend, purchase rates by cohort)\nIncludes observation_months, tenure range"]
        TRANS["Segment Transitions\n(transition rates by dimension)\nSum to 1 per origin label"]
    end

    %% Connections
    SRC --> SCAN --> PARSE --> CLEAN --> VALUE --> VALIDATE --> AUDIT
    AUDIT --> REJECT
    AUDIT --> ENRICH --> PARTITION --> CHECKS
    
    CHECKS --> PRICE_REF
    CHECKS --> BASKETS
    BASKETS --> BASE
    BASE --> DAILY --> RHYTHM
    DAILY --> WEEKDAY
    DAILY --> BREADTH
    DAILY --> PROD --> TRIALS
    PROD --> CAT
    BASE -.-> PRICE_FEAT
    PRICE_REF -.-> PRICE_FEAT
    
    BASE --> MERGE
    RHYTHM --> MERGE
    WEEKDAY --> MERGE
    BREADTH --> MERGE
    TRIALS --> MERGE
    CAT --> MERGE
    PRICE_FEAT --> MERGE
    
    MERGE --> DERIVED --> FOLLOWUP
    
    FOLLOWUP --> THRESH --> STYLE --> TRAJ --> LIFECYCLE --> EVIDENCE --> BEHAVIOR --> TRAJ_L --> VALUE_TIER
    
    VALUE_TIER --> SEG
    VALUE_TIER --> COHORT
    VALUE_TIER --> TRANS

    %% Styling
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef prep fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef snap fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef label fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef report fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class SRC input
    class SCAN,PARSE,CLEAN,VALUE,VALIDATE,AUDIT,REJECT,ENRICH,PARTITION,CHECKS prep
    class PRICE_REF,BASKETS,BASE,DAILY,RHYTHM,WEEKDAY,BREADTH,PROD,TRIALS,CAT,PRICE_FEAT,MERGE,DERIVED,FOLLOWUP snap
    class THRESH,STYLE,TRAJ,LIFECYCLE,EVIDENCE,BEHAVIOR,TRAJ_L,VALUE_TIER label
    class SEG,COHORT,TRANS report
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run pipeline
retail-analytics \
  --input customer_transactions.parquet \
  --output results \
  --price-mode unit \
  --snapshots 2000-06-30 2001-06-30 2001-12-11 \
  --buckets 4 \
  --exclude-invalid
```

## Outputs

| File | Description |
|------|-------------|
| `results/quality.json` | Data quality audit with rejection breakdown |
| `results/rejected.parquet` | Rejected rows with `rejection_reason` column |
| `results/validated.parquet` | Cleaned transactions |
| `results/snapshots/{date}/b{N}.parquet` | Customer features (78 cols) with labels |
| `results/segment_profiles.parquet` | Segment metrics per cutoff |
| `results/segment_transitions.parquet` | Transition rates between segments |
| `results/cohort_profiles.parquet` | Cohort analysis by first purchase month (includes observation_months) |
| `results/thresholds.json` | Quantile thresholds from first snapshot |
| `results/configuration.json` | Pipeline run configuration |
| `results/price_references/{date}.parquet` | Per-snapshot median prices per product |
| `results/product_variation.parquet` | Product classification audit |

## Tests

```bash
# Run all tests (discovers all test_*.py modules)
pytest -q

# Run individual test modules
pytest tests/test_retail_polars_workflow.py -v
pytest tests/test_dashboard.py -v
pytest tests/test_metrics_contract.py -v
pytest tests/test_patronage.py -v
pytest tests/test_visual_insights.py -v
```

## Key Methodological Notes

- **No lookahead**: All descriptive features use only transactions ≤ cutoff date. Future follow-up metrics require explicit `--coverage-confirmed`.
- **Purchase-only**: Pipeline treats all valid rows as purchases. Returns/refunds not supported (negative quantities rejected).
- **Relative segmentation**: Quantile thresholds are population-relative at first snapshot; not portable across populations.
- **Cohort exposure**: Cohort metrics include `observation_months`; raw rates not comparable across unequal tenures.
- **Null semantics**: Metrics return `null` (not zero) for zero denominators or insufficient data.
- **Currency**: Amounts in input data's currency; no conversion or symbol assumed.
- **No causal claims**: All associations descriptive. Segment differences ≠ intervention effects.

## Requirements

```
polars>=1.31
streamlit>=1.28.0
plotly>=5.17.0
```

Install with `pip install -e ".[dev]"`.