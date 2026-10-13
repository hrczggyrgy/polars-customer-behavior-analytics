# Polars Customer Behavior Analytics

A customer behavior analytics pipeline built with [Polars](https://pola.rs) for retail transaction data.

## Documentation

| Document | Description |
|--------|-------------|
| [METRICS.md](METRICS.md) | Complete metric specifications with formulas, denominators, windows, and null semantics |
| [DATA_CONTRACT.md](DATA_CONTRACT.md) | Input/output schemas, validation rules, and CLI reference |
| [docs/workflow.md](docs/workflow.md) | Pipeline architecture diagram and data flow |
| [docs/DASHBOARD.md](docs/DASHBOARD.md) | Streamlit dashboard usage and limitations |
| [docs/VISUAL_INSIGHTS.md](docs/VISUAL_INSIGHTS.md) | Visual analytics module documentation |
| [docs/PATRONAGE.md](docs/PATRONAGE.md) | Observed patronage layer documentation |
| [docs/customer_analytics_improvement_plan.md](docs/customer_analytics_improvement_plan.md) | Evidence-led improvement plan (P0-P2) |

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
| `results/snapshots/{date}/b{N}.parquet` | Customer features with labels |
| `results/segment_profiles.parquet` | Segment metrics per cutoff |
| `results/segment_transitions.parquet` | Transition rates between segments |
| `results/cohort_profiles.parquet` | Cohort analysis by first purchase month (includes observation_months) |
| `results/thresholds.json` | Quantile thresholds from first snapshot (includes `quantile_interpolation: linear`) |
| `results/configuration.json` | Pipeline run configuration |
| `results/price_references/{date}.parquet` | Per-snapshot median prices per product |
| `results/product_variation.parquet` | Product classification audit |
| `results/run.json` | Run manifest with coverage state and dates |

## Tests

```bash
# Run all tests
pytest -q

# Run with coverage
pytest --cov=retail_analytics --cov-report=term-missing

# Run individual test modules
pytest tests/test_core.py -v
pytest tests/test_visualizations.py -v
pytest tests/test_smoke.py -v
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

- Python ≥ 3.10
- polars ≥ 1.31
- streamlit ≥ 1.28.0
- plotly ≥ 5.17.0

Install with `pip install -e ".[dev]"`.

## Dashboard

```bash
streamlit run src/retail_analytics/dashboard/app.py
```

### Dashboard Navigation

The dashboard uses a **question-based two-level navigation**:

| Section | Views |
|---------|-------|
| **01 · Start Here** | Data Trust & Quality, Customer Base Summary, Key Signals |
| **02 · Lifecycle & Repeat** | Cohort-Age Retention, Time to Repeat, Multi-Horizon Cohorts, Purchase Cadence, Cadence Validation, Future Validation |
| **03 · Value & Risk** | RFM Response Surface, Value-at-Risk Scatter, Revenue Bridge, Revenue Waterfall, Customer Momentum, Spend Concentration, Value-Tier Transitions |
| **04 · Products & Categories** | Category Penetration × Growth, Category Opportunity Map, Category Participation, Product Association Network, Trial Conversion Intervals, Product Affinity |
| **05 · Explore Customers** | Customer Brief, Customer Timeline |
| **06 · Methods & Data Quality** | Metric Definitions, Thresholds & Calibration, Data Quality Report, Analysis Settings |
| **07 · Predictive Modeling** | Model Evaluation, CLV Predictions |
| **Advanced Analytics** | Behavior × Trajectory, Purchase Rhythm, Basket Evolution, Category Sankey, Transitions, Quality, Segments, Features, Cohorts |

### Dashboard Features

- **Coverage-aware**: Future outcomes require explicit coverage confirmation (P0 fix)
- **Evidence-based charts**: Every primary chart shows question, takeaway, evidence, how-to-read, caveat, next-step
- **Accessibility**: Colorblind-safe Okabe-Ito palette, `alt_description` for screen readers
- **Sparse-data safeguards**: Evidence badges (strong/limited/insufficient), Wilson 95% CI, multiplicity warnings
- **Masked customer IDs**: Customer explorer masks IDs by default

## Predictive Modeling (P2)

The dashboard includes probabilistic model integration:

| View | Description |
|------|-------------|
| **Model Evaluation** | Temporal holdout evaluation with configurable train/validation/test splits; BG/NBD + Gamma-Gamma |
| **CLV Predictions** | Customer Lifetime Value predictions combining BG/NBD (transaction count) + Gamma-Gamma (monetary) |

Models use temporal holdout evaluation with baseline comparisons. Only displayed after temporal validation passes.

## Architecture

```
src/retail_analytics/
├── core/                 # Pipeline core (pipeline, thresholds, stats, constants)
├── dashboard/
│   ├── app.py           # Streamlit app (modular view dispatch)
│   ├── dashboard_backend.py
│   └── views/           # Modular view modules
│       ├── overview.py
│       ├── lifecycle.py
│       ├── value_risk.py
│       ├── products.py
│       ├── customer_explorer.py
│       ├── methods.py
│       ├── advanced.py
│       ├── modeling.py
│       ├── segments.py
│       ├── features.py
│       ├── cohorts.py
│       └── transitions.py
├── modeling/
│   ├── models.py        # BG/NBD, Pareto/NBD, Gamma-Gamma
│   ├── evaluation.py    # Temporal holdout evaluation
│   └── __init__.py
├── visualization/
│   ├── chart_components.py   # ChartSpec, ChartCard, ChartEvidence, SupportThresholds
│   ├── chart_specs.py        # 23 typed ChartSpec factory functions
│   └── visual_insights_core.py
├── security.py           # Upload validation, ID masking, CSP, audit logging
└── patronage/            # Observed patronage layer
```

## Tests

```bash
# Run all tests (51 tests)
pytest -q

# Run with coverage
pytest --cov=retail_analytics --cov-report=term-missing

# Run individual test modules
pytest tests/test_core.py -v
pytest tests/test_visualizations.py -v
pytest tests/test_smoke.py -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.