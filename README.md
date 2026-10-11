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
| `results/thresholds.json` | Quantile thresholds from first snapshot |
| `results/configuration.json` | Pipeline run configuration |
| `results/price_references/{date}.parquet` | Per-snapshot median prices per product |
| `results/product_variation.parquet` | Product classification audit |

## Tests

```bash
# Run all tests
pytest -q

# Run with coverage
pytest --cov=retail_analytics --cov-report=term-missing

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

- Python ≥ 3.10
- polars ≥ 1.31
- streamlit ≥ 1.28.0
- plotly ≥ 5.17.0

Install with `pip install -e ".[dev]"`.

## Dashboard

```bash
streamlit run src/retail_analytics/dashboard/app.py
```

## License

MIT License - see [LICENSE](LICENSE) for details.