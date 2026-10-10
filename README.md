# Polars customer behavior analytics

Purchase-only analytics from exactly nine columns: customer_id, transaction_date,
transaction_id, product_id, product_description, department, category, price,
quantity. No customer data is included.

## Run
```bash
python -m pip install -r requirements.txt
python -m unittest test_retail_polars_workflow
python -m retail_polars_workflow --input transactions.parquet --output results_01 --price-mode unit --snapshots 2025-05-31 2025-06-30 --buckets 8
```
Example cutoffs must be replaced with dates in your data. Output must be empty.
Use unit for price * quantity or line_total when price is already the line value.
Non-ISO strings need --date-format. Dates are calendar dates; normalize timezone
before ingestion. Invalid rows fail by default; --exclude-invalid is explicit.
Duplicates are reported and preserved.

## Outputs
Validation reports, customer partitions, historical baskets/shopping days/product
and category histories, numeric and labelled snapshots, price references, frozen
thresholds, segment profiles, observed-cohort summaries and segment transitions.
Features cover value, cadence, personal baselines, repetition, trials, exploration,
category withdrawal, fixed-event breadth, adjacent-day overlap, comparable prices,
quantity/next-gap relationships and nonoverlapping 90-day changes. Segmentation
separates lifecycle, behavioral tags, trajectory, observed value and evidence.
See label() for all rule thresholds; they require validation, not blind adoption.

## Status and limitations
Reconstructed from the conversation, not byte-identical to the earlier bundle.
Syntax checked; Polars runtime tests NOT executed in the authoring environment.
Run tests locally and pin the tested Polars version. Descriptions are retained,
but automatic attribute parsing is not enabled. No profit CLV, elasticity,
permanent-churn probability, clustering, survival or causal models are computed.
Evidence labels are not calibrated confidence. Float64 is analytical, not audited
accounting. Cohorts use first observed purchase, not confirmed acquisition.

Future outcomes are separate from features and null for incomplete follow-up.
The latest purchase date is a provisional coverage endpoint, not proof of complete
capture. Use --observed-through only with confirmed complete data. Quantity-gap
summaries use completed intervals only and are not censoring-corrected. Seasonality
is not automatically adjusted. No dense matrices are built, but medians, ordered
windows, grouping and lists still hold state. Increase buckets and monitor RAM.
History is replayed per cutoff; this is not an incremental production service.
