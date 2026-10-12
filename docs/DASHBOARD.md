# Scientific dashboard

```bash
python -m pip install -r requirements.txt
python -m unittest test_retail_polars_workflow test_dashboard
streamlit run app.py
```

The app integrates the existing pipeline through dashboard_backend.py. Upload
Parquet or UTF-8 CSV and explicitly choose price meaning, date format, delimiter
and optional decimal comma (no thousands separators). Missing/nested columns,
header collisions, corrupt files and basket-key conflicts fail validation. Extras
are disclosed and ignored; invalid rows require explicit exclusion. Duplicates
remain preserved. Epoch units and timezones are not guessed.

Views: overview, quality, layered segments, feature eligibility/missingness,
exploratory scatter, first-observed cohorts, label transitions, opt-in customer
explorer and bounded aggregate exports. Outcome comparisons require confirmed
complete capture and sufficient follow-up. Wilson 95% intervals assume independent
customers at one cutoff; they do not correct exploratory multiple comparisons or
establish causality. Rule labels are not calibrated confidence, permanent churn,
profit CLV or elasticity. No automatic seasonal adjustment or attribute parsing.

Limits: 100 MiB upload, 3 million rows, 12 cutoffs, 32 buckets, 10-minute workers.
Disk estimates and streaming are not guarantees: compressed files expand and
ordered/grouped calculations retain state. Use deployment-level memory/CPU limits,
authentication and encrypted transport. Do not expose sensitive data in an open
public deployment; this is not a hostile-file sandbox.

Each session owns a temporary workspace. No uploads go to GitHub or shared caches.
Changed inputs invalidate/remove stale results; explicit clearing removes session
files. Failed result folders are removed. Disconnect cleanup is best-effort, not
secure erasure. History is replayed, not incrementally updated. Browser tables
show at most 2,000 rows; CSV text is escaped against common formula injection.

Verification: Python syntax and statistical/calendar helper checks passed.
Polars integration and Streamlit UI could NOT be runtime-tested in the authoring
environment. Run the included regressions locally and pin verified dependencies.
