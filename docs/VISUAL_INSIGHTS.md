# Transaction-only Visual Analytics

This module provides aggregate calculations and regression tests for customer behavior analytics. The dashboard (`src/retail_analytics/dashboard/app.py`) integrates these functions for interactive visual exploration.

## Core APIs

- `segment_summary`: Full-population median/IQR, finite/missing counts, 90/365-day exposure and normalized shares
- `concentration`: All selected customers including zero spend, 90-day top fractions and at most 101 exact Lorenz curve points
- `monthly_summary`: Non-overlapping months from existing basket outputs, fixed cutoff population and purchasing components
- `category_summary`: 365-day category-history participation/spend share/recurrence without pairs
- `cohort_summary`: First-observed later-shopping-day repeats within 30/60/90 days, mature follow-up, capture gates and 95% Wilson intervals
- `transition_summary`: Matched label rates with origin/destination/unmatched reconciliation
- `wilson`: 95% Wilson score interval for binomial proportions

## Definitions

- **90-day bounds**: Cutoff minus 89 through cutoff (inclusive)
- **365-day bounds**: Cutoff minus 364 through cutoff (inclusive)
- **Basket outputs**: Use customer + transaction + date grain
- **Zero spend**: Yields undefined proportions (null, not zero)
- **Top fractions**: Round up to whole customers; ties may split
- **IQR**: Within-group variation, not confidence
- **Minimum history**: Changes denominators; rule-defining differences are not independent validation
- **Monthly buyers**: Not additive unique people; fixed populations include customers first observed after earlier months
- **Category participation**: Overlaps; all selected customers, including inactive, remain in denominator
- **Category spend shares**: Include all categories; recurrence is at least three in-window shopping days
- **Product established share**: Subset of repeat share, not a separate stacked part
- **HHI**: Depends on category breadth

## Limitations

- Repeat cohorts use first observation, not verified acquisition
- Same-day baskets are excluded from repeat calculations
- Eligible follow-up ends at the selected cutoff; incomplete horizons and unconfirmed capture suppress rates
- Denominators vary by horizon
- Wilson intervals assume independent customers and are not multiplicity-adjusted
- Origin transitions are unfiltered, destination filtered: unmatched customers are not churn and destination-only customers are not acquisitions
- No product matrix, full-catalog market basket mining, pair expansion, automatic seasonality adjustment, calibrated churn or anomaly alerts
- Currency unspecified; amounts are not profit or net revenue after refunds
- Capture is user-confirmed, not independently verified
- Source history/tenure do not verify acquisition; duplicates remain preserved
- No production dataset or commercial findings claimed

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/test_visual_insights.py -v
```

All tests pass without skips in the CI environment.