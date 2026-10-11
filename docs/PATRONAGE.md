# Observed patronage layer

This additive extension preserves app.py, the core pipeline, existing labels and
future-outcome logic. Once the companion UI files are installed, run:

```bash
python -m pip install -r requirements.txt
python -m unittest test_patronage test_retail_polars_workflow test_dashboard
streamlit run loyalty_app.py
```

The companion entry point reuses the original upload/validation workflow and
switches between Visual analytics and Observed patronage. The original app and
CLI do not automatically gain these fields. The derived layer is explicit.

## Meaning and default precedence

These are proposed rules, not validated customer classes. Observed patronage is
not emotional loyalty, retailer exclusivity, share of wallet, profitability,
permanent churn or calibrated confidence. First purchase means first observed.
Product repetition, shopping style, trajectory and value tiers remain separate.

1. Unconfirmed capture -> capture_unconfirmed.
2. One purchase day -> first_observed below 90 days, otherwise long_observed_one_day.
3. Tenure below 90 days -> recent_repeat.
4. Fewer than 3 purchase days, 3 complete months or 2 active complete months -> insufficient_history.
5. Regular cadence and recency >=2 times median gap -> recurring_overdue.
6. Recent resumed purchasing -> reactivated_repeat.
7. Regular cadence and active-period share >=0.60 -> established_recurring.
8. Remaining sufficiently observed repeats -> established_occasional.

Regular cadence needs >=6 purchase days in 365 days, >=4 usable completed gaps,
positive finite median gap and finite gap CV between 0 and 0.75. Overdue status
precedes resumption. Resumption is a purchase in the last 30 days after a gap
>=2 times the preceding baseline median gap; enough preceding gaps are required.

Both original coverage confirmation and explicit history-capture confirmation
are required for classification. Descriptive features remain available otherwise.
Complete calendar months exclude the partial first and cutoff months; a boundary
month is included when first observation is day 1 or cutoff is month end. Calendar
completeness does not prove complete capture or establish acquisition. Zero
exposure yields null ratios. active_months_observed includes partial bins.

Category summaries cover the preceding 365 days inclusive. Recurring categories
have >=3 purchase days. Gaps have both endpoints inside the window; no zero or
censored gaps are invented. Participation divides by all segment customers;
spending share has a different denominator. No demographic personas are inferred.

## Companion views and safeguards

Profiles show counts, finite-value counts, medians and quartiles. Distributions
and cadence plots are capped at 2,000 points; style overlap and transition grids
are capped at 20 labels per axis. Category participation, frozen-rule transitions,
local overdue-threshold sensitivity, consent-gated details and definitions are
available. The original workspace supplies purchase-day journeys. Defining-feature
differences are not independent validation; stability is not business usefulness.
Sensitivity holds resume_events fixed; full sensitivity needs a rebuild.

Derived files live under the existing session result directory, keyed by source
run, version, rules and capture decision. A separate worker has a 10-minute limit.
The completion manifest is installed atomically. Failed builds are removed;
incomplete timed-out builds are removed on retry. Clearing the original workspace
removes the layer. No shared cache; use an access-controlled deployment.

Eight pure rule/calendar/adapter tests passed locally. Two Polars integration tests
were skipped because Polars is unavailable. UI is not runtime-tested. The adapter
test skips until its companion UI file is installed. Run the full suite before
claiming runtime verification or using the labels for decisions.

Scientific rationale: Dick and Basu (1994), DOI 10.1177/0092070394222001;
Platzer and Reutterer (2016), DOI 10.1287/mksc.2015.0963;
Jacobs, Fok and Donkers (2021), DOI 10.1287/mksc.2020.1279.
These motivate the design, not the numerical thresholds.
