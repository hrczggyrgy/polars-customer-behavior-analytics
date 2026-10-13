# Customer Analytics App: Full Evidence-Led Improvement Plan

**Repository:** [hrczggyrgy/polars-customer-behavior-analytics](https://github.com/hrczggyrgy/polars-customer-behavior-analytics)  
**Audit date:** 12 October 2026  
**Scope:** Repository source, contracts, documentation, test structure, and proposed product experience.  
**Product goal:** Help users understand customers through explainable, trustworthy, outstanding charts, even when transaction histories are limited.

> **Executive decision:** Do not start by adding a larger collection of machine-learning models. First fix evidence eligibility, then make the existing analytical capability discoverable, explain every important chart consistently, and handle small samples honestly. Add predictive methods only when they can be evaluated against future holdout data.

---

## Contents

1. [Executive recommendation](#1-executive-recommendation)
2. [What the repository already does well](#2-what-the-repository-already-does-well)
3. [Prioritized gap assessment](#3-prioritized-gap-assessment)
4. [P0: Fix observation-coverage eligibility](#4-p0-fix-observation-coverage-eligibility)
5. [Target customer-understanding experience](#5-target-customer-understanding-experience)
6. [A shared explainable-chart system](#6-a-shared-explainable-chart-system)
7. [Chart-by-chart improvement portfolio](#7-chart-by-chart-improvement-portfolio)
8. [Scientific approach for limited transaction data](#8-scientific-approach-for-limited-transaction-data)
9. [Model roadmap: what to add, and when](#9-model-roadmap-what-to-add-and-when)
10. [Data and metric contract improvements](#10-data-and-metric-contract-improvements)
11. [Implementation plan and code touchpoints](#11-implementation-plan-and-code-touchpoints)
12. [Testing and acceptance criteria](#12-testing-and-acceptance-criteria)
13. [Product success metrics](#13-product-success-metrics)
14. [Security, privacy, and deployment considerations](#14-security-privacy-and-deployment-considerations)
15. [Suggested delivery backlog](#15-suggested-delivery-backlog)
16. [Research and documentation references](#16-research-and-documentation-references)
17. [Audit boundary](#17-audit-boundary)

---

## 1. Executive recommendation

The repository already has a substantial transaction analytics engine and a broad set of research-inspired visualizations. It includes Polars-based transaction processing, explicit metric definitions, input-quality auditing, historical customer snapshots, interpretable behavioral rule labels, cohort analysis, RFM response surfaces, Kaplan–Meier time-to-second-purchase analysis, cadence validation, product-affinity analysis, category-opportunity maps, concentration curves, revenue bridges, and segment transitions.

The next stage should focus on making those capabilities trustworthy and usable as a single product. Priorities, in order:

1. **Scientific correctness:** future outcomes must be unavailable until the source's complete observation coverage is explicitly confirmed.
2. **User comprehension:** organize the UI around customer questions, not a long catalogue of analytical methods.
3. **Explainable charts:** use one contract for the question, takeaway, denominator, timeframe, uncertainty, caveat, and next step for every important chart.
4. **Sparse-data guardrails:** distinguish zero, missing, insufficient history, incomplete follow-up, and true observed non-repeat.
5. **Customer narrative:** let users understand an individual customer through a short profile, evidence and timeline instead of a wide table alone.
6. **Model discipline:** add predictive models only after temporal holdout validation, baseline comparison, calibration checks, and clear naming.
7. **Release confidence:** resolve documented contract/code discrepancies and verify the application end-to-end in the actual runtime environment.

### The most important issue found

In `src/retail_analytics/dashboard/app.py`, the upload workflow sets `coverage = True` automatically and sends that value as `coverage_confirmed`. The README and dashboard documentation state that future outcome calculations require explicit confirmation of complete follow-up. The most recent observed transaction date does **not** prove that the source captured all transactions through that date; exports can omit stores, channels, customers, or delayed events.

This is a scientific-validity defect, not a cosmetic UI issue. Resolve it before presenting observed future-purchase rates as evidence.

### Product positioning

Build an **evidence-led customer intelligence workspace for transaction-only data**. It should still be useful with sparse histories, and should say when the evidence is too weak to support a conclusion. It should not pretend transaction-only files can establish profit, emotional loyalty, permanent churn, marketing incrementality, or campaign causality.

---

## 2. What the repository already does well

### 2.1 Data contract and audit foundations

The repository documents input/output fields, row rejection reasons, basket construction, date-window boundaries, price interpretation, currency limitations, null semantics, and conditions for future follow-up metrics. This is a strong base for trustworthy analytics. Retain and extend the contracts rather than reimplementing metric definitions inside the UI.

### 2.2 Historical feature generation

Customer snapshots are built with explicit cutoff dates. The workflow documentation describes a broad feature set including recency, tenure, spend/frequency windows, average order/basket value, purchase-gap patterns, category mix, breadth, product/category overlap, trial-to-repeat behaviors, and change between comparable periods. The documented no-lookahead rule is valuable: historical features must use only transactions at or before the cutoff.

### 2.3 Interpretable behavioral labels

Rule-based lifecycle, evidence, style, trajectory and observed-value labels are more inspectable than a default black-box clustering system. The labels are not automatically validated truth, but they can provide a useful starting point when their rules, reference population and minimum evidence are visible.

### 2.4 Existing charts worth improving rather than replacing

The application already includes or documents functions and views for:

- Customer momentum: frequency versus basket change.
- Purchase cadence: recency versus median inter-purchase gap.
- Cohort-age retention heatmaps.
- Time to second purchase using a Kaplan–Meier estimator.
- Repeat-purchase probability over RFM bins.
- Category penetration and intensity/opportunity.
- Product trial-to-repeat and product association rules/networks.
- Customer spend concentration and Lorenz curves.
- Revenue waterfall/bridge views.
- Value-tier and other segment transitions, including alluvial flows.
- Behavior × trajectory matrices and purchase-rhythm summaries.
- Feature missingness, distributions, scatter plots, and customer timelines.

The goal is to improve **selection, comparison, interpretation and evidence quality**, not to maximize the number of visual types.

### 2.5 Existing testing foundation

Tests cover the core workflow, data/metric contracts, dashboard helpers, patronage rules, and several research visualization calculations. Keep the calculation functions testable outside Streamlit, and add integration tests for the full user flow and rendered-chart metadata.

### 2.6 Useful existing cautionary statements

The documentation already notes that:

- observed monetary spend is not profit;
- first observed purchase is not necessarily acquisition;
- descriptive segment differences are not causal effects;
- rule labels are not calibrated churn or CLV predictions;
- future outcomes require confirmed observation coverage;
- raw rates with different cohort exposure are not directly comparable;
- confidence intervals and sample caps have assumptions and limitations.

Turn these into visible product behavior rather than leaving them only in documentation.

### Primary repository files reviewed

- [`README.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/README.md)
- [`METRICS.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/METRICS.md)
- [`DATA_CONTRACT.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/DATA_CONTRACT.md)
- [`DECISIONS.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/DECISIONS.md)
- [`src/retail_analytics/dashboard/app.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/dashboard/app.py)
- [`src/retail_analytics/dashboard/dashboard_backend.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/dashboard/dashboard_backend.py)
- [`src/retail_analytics/visualization/visual_insights_core.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/visualization/visual_insights_core.py)
- [`src/retail_analytics/visualization/product_affinity.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/visualization/product_affinity.py)
- [`docs/DASHBOARD.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/DASHBOARD.md)
- [`docs/VISUAL_INSIGHTS.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/VISUAL_INSIGHTS.md)
- [`docs/PATRONAGE.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/PATRONAGE.md)

---

## 3. Prioritized gap assessment

| Priority | Gap | Evidence observed in repository | Recommended action | Why it matters |
|---|---|---|---|---|
| **P0 — correctness** | Future-outcome coverage is assumed instead of explicitly confirmed. | The dashboard workflow assigns `coverage = True` and forwards it as `coverage_confirmed`, while documentation says complete follow-up must be confirmed. | Default the gate to off. Ask for a source-level complete-through date and scope confirmation. Suppress follow-up estimates unless eligible. Record the decision in the run manifest. | Prevents invalid future repeat rates and false assurance about customer behavior. |
| **P1 — product UX** | Navigation is broad and effectively flat. | `NAV_GROUPS` groups view names in code, but the dashboard flattens them into one horizontal radio selector. There are many views, including specialist and legacy/exploration views. | Use a small primary navigation plus context-specific sub-navigation. Keep advanced views reachable in an Advanced area. | Users should ask customer questions without learning the names of analytical methods. |
| **P1 — explainability** | Chart interpretation is written individually for each view rather than enforced by one shared contract. | There are many thoughtful captions/hover templates, but the common chart wrapper does not pass the `alt` description to `st.plotly_chart`. | Define a chart spec: question, takeaway, metric/unit, population, denominator, period, eligible n, uncertainty, caveat, next step and accessible description. | Increases clarity and consistency; reduces the risk that users misread a percentage or chart sample. |
| **P1 — evidence quality** | Support thresholds and uncertainty are not uniform across all views. | Some views show Wilson intervals/counts; other exploratory charts provide point estimates, and documentation notes no multiple-comparison adjustment. | Put `n`, denominator, eligibility and horizon on outcome charts. Use intervals when meaningful and label low-support results as preliminary or insufficient evidence. | A 50% result from 1 of 2 customers is not the same evidence as 500 of 1,000. |
| **P1 — representativeness** | Bounded scatterplots can under-represent smaller groups. | Some plots deterministically hash-select up to 2,000 customers and disclose this in captions. | Show full-population aggregates separately; use segment-aware sampling or density/hexbin views where helpful; display population and sample sizes. | Prevents visual impressions from depending too strongly on which records were sampled. |
| **P1 — accessibility** | Charts rely heavily on color and hover. | The shared wrapper calls `st.plotly_chart` without its `alt` parameter; some conclusions live only in hover details. | Add specific `alt` text, a visible takeaway and data summary, tables/exports, keyboard-friendly controls, and palette checks. | Makes the findings understandable without color vision or hover interaction. |
| **P1 — data semantics** | Transaction-only measures can be mistaken for loyalty, profit or full lifecycle status. | Returns/refunds are unsupported; negative quantities are rejected; currency is opaque; first purchase is only first observed. | Show a persistent scope banner and label metrics as observed spend/behavior. Capture currency, return policy, ID scope, data source and coverage. | Prevents overclaiming about profit, customer acquisition, permanent churn or share of wallet. |
| **P1 — contract drift** | Known contract/documentation discrepancies may remain open. | `DECISIONS.md` lists work around duplicate behavior, timezone wording, quantile interpolation metadata and a broad exception in transition reporting. | Close the decision log with tests; align docs, manifests and actual implementation; do not hide unexpected reporting failures. | Users need reproducible metrics and clear failures rather than silent omissions. |
| **P2 — customer understanding** | Customer exploration is more feature-oriented than narrative-oriented. | The customer view offers exact-ID lookup, a feature table, and a purchase timeline gated by consent. | Add a readable customer brief, cadence and purchase timeline, recent/prior comparison, category mix, and evidence limitations; keep identifiers masked by default. | A non-technical user should understand the evidence without interpreting dozens of feature columns. |
| **P2 — modeling roadmap** | Predictive future value is not yet available; that is acceptable but a future extension. | The documentation correctly says labels are not calibrated churn, profit CLV or elasticity and the data contract lacks marketing exposure. | Create an evaluation layer first. Later test BG/NBD or Pareto/NBD for repeat counts and Gamma-Gamma for monetary value only if assumptions and holdout accuracy are appropriate. | Prevents complex methods from adding false precision to sparse or unsuitable data. |
| **P2 — maintainability** | Dashboard rendering has grown into a very large single module; runtime UI verification is limited in documentation. | `app.py` contains a large number of view branches; the docs note that UI runtime verification has not always been possible in the authoring environment. | Extract page/view renderers and shared chart components; preserve pure calculation functions; run a real Streamlit smoke test. | Reduces regression risk and enables consistent UI improvements. |
| **P2 — product validation** | Repository tests validate calculations but do not by themselves prove that users understand the UI or that visualizations render correctly in deployment. | Unit tests exist; documentation distinguishes helper tests from full runtime UI verification. | Add rendered-chart smoke tests, fixed fixtures, accessibility checks and short usability testing. | Analytical correctness and user comprehension are separate acceptance criteria. |

### 3.1 Additional contract and error-handling work

Review the open items in `DECISIONS.md` as a dedicated cleanup ticket:

1. **Duplicate handling:** documentation says duplicates are preserved unless users preprocess them, while the implementation has an explicit opt-in `drop_exact_duplicates` option. Clarify the default and the audit-count timing; never silently change the default behavior.
2. **Transition reporting:** the decision log identifies a broad `except Exception: continue` pattern that can hide unexpected transition errors. Catch known missing/empty-input conditions separately; log and surface schema/computation failures.
3. **Timezone wording:** align `DATA_CONTRACT.md` with the implementation for timezone-aware versus naive dates, and test the boundary around local midnight.
4. **Quantile interpolation:** document that the relevant quantile/CI calculation uses linear interpolation if that remains the implementation.
5. **Threshold metadata:** include the interpolation method and reference snapshot in threshold output so threshold values can be reproduced.

Do not treat every issue recorded in `DECISIONS.md` as definitely unresolved without checking current code and commit history during implementation. The point is to turn that file's decision items into explicitly verified acceptance tests.

---

## 4. P0: Fix observation-coverage eligibility

### 4.1 The problem

The current upload flow determines minimum/maximum dates from transactions and later sets `coverage = True`. This conflates two different concepts:

- **Latest observed transaction date:** the maximum event date in the file.
- **Known complete-through date:** the latest date for which the data owner asserts that all in-scope activity has been captured.

Those dates can differ. An export may omit one store, channel or payment source; records can arrive late; or a source system can have a delayed refresh. The event maximum is not proof of complete capture.

### 4.2 Desired UI and stored fields

During import/validation, show a short, explicit control:

- `Latest transaction date in file: YYYY-MM-DD` (computed; read-only).
- `I can confirm data is complete through: [date]` (user-entered/asserted).
- `Scope covered: [stores/channels/business units, or unknown]` (user assertion / optional metadata).
- `Returns/refunds present?` and `Currency` (if known).
- Clear warning: “Use future repeat-purchase validation only if this export is complete for the selected population and period.”

Persist, at minimum:

```json
{
  "coverage_confirmed": false,
  "coverage_complete_through": null,
  "latest_observed_transaction_date": "YYYY-MM-DD",
  "coverage_scope_note": null,
  "coverage_confirmation_source": "user_assertion"
}
```

This is a proposed manifest shape; adjust names to the existing schema but preserve the semantic distinction. A run manifest should also contain the source fingerprint, input mapping, price mode, timezone setting, date cutoffs, software version and threshold reference metadata.

### 4.3 Expected behavior

| Situation | Historical descriptive metrics | Future outcome validation | UI wording |
|---|---|---|---|
| Complete-through date is unknown or confirmation is off | Allowed, subject to ordinary data-quality checks | Disabled | “Historical descriptions available; future outcomes unavailable because coverage is unconfirmed.” |
| Coverage is asserted but a requested 30/90-day horizon extends past the complete-through date | Allowed | That horizon unavailable | “Insufficient observed follow-up through the requested horizon.” |
| Coverage is confirmed and the horizon is complete | Allowed | Eligible customers/rates may be computed | Show horizon, eligible n, numerator/denominator and uncertainty. |
| No eligible observations or denominator is zero | Allowed | Null / unavailable, not zero | Explain the exact eligibility or denominator reason. |

The UI must not convert incomplete follow-up to “no repeat purchase,” nor allow a chart to imply that an unconfirmed absence is a customer choice.

### 4.4 Implementation and tests

**Likely touchpoints:** `src/retail_analytics/dashboard/app.py`, `dashboard_backend.py`, the run/configuration manifest writer, metric contract documentation, and dashboard tests.

- Replace the hard-coded coverage assignment with explicit user input, default `False`.
- Pass both confirmation and complete-through date into the worker/analysis configuration.
- Validate that the requested horizon is covered for each cutoff.
- Keep historical snapshots usable without coverage confirmation.
- Show eligibility warnings inside every view whose result depends on future outcomes, not only once in the sidebar.
- Export the coverage state with the results so charts and CSV/JSON downloads preserve context.

Required regression cases:

1. Default upload without confirmation disables future-outcome charts.
2. An asserted complete-through date earlier than the end of the requested horizon suppresses that horizon.
3. A confirmed complete horizon reports the correct eligible customers and denominator.
4. Zero events with a valid complete observation period yields a real zero repeat count/rate; missing or incomplete observation yields unavailable/null.
5. Changing the coverage setting invalidates stale result files and cached UI state.
6. The coverage status and dates appear in the JSON manifest and exported summaries.

**Acceptance:** no future-purchase rate is displayed as evidence unless the user explicitly confirms complete capture for the relevant population and time horizon, and the output records the confirmation and dates.

---

## 5. Target customer-understanding experience

### 5.1 Navigation should follow business questions

Use a compact primary navigation and a second-level view picker, rather than placing every view in one horizontal radio control. Specialist views should remain available but should not dominate the default journey.

| Primary area | User question | Main content | Existing views to reuse |
|---|---|---|---|
| **Start here** | What do I know about this customer base, and how trustworthy is the data? | Data scope, quality, active customer counts, recorded spend/baskets trend, key changes, missingness and coverage state. | Overview, Quality, Features, concentration and monthly trends. |
| **Lifecycle & repeat** | Who comes back, and how long does it take? | First-observed cohorts, time to second purchase, observed return rates, purchase rhythm and cadence. | Cohort-age retention, Time to Repeat, Multi-Horizon Cohorts, Purchase Cadence and Cadence Validation. |
| **Value & behavior** | How does observed customer behavior vary and change? | RFM response surface, observed spend distribution, customer momentum, cadence relative to spend, and segment comparisons. | RFM Response Surface, Value-at-Risk Scatter, Customer Momentum, Spend Concentration, Basket Evolution. |
| **Products & categories** | What do people buy, repeat, or explore? | Category reach/intensity, product associations, category participation change and trial-to-repeat patterns. | Category Opportunity Map, Category Penetration × Growth, Product Affinity, Product Association Network, Trial Conversion Intervals. |
| **Customer explorer** | What have we observed about this customer? | Concise profile, purchase timeline, typical gap, recency, frequency, basket size, category mix and changes between comparable periods. | Existing Customer lookup and timeline with a narrative summary layer. |
| **Methods & data quality** | What do these numbers mean and when should I trust them? | Definitions, formulas, eligibility rules, coverage, thresholds, uncertainty, rejected rows, exports and known limitations. | Methods, Quality, threshold reference info and manifests. |
| **Advanced analytics** | Which specialist diagnostics are available? | Transitions, alluvial flows, network analysis, threshold sensitivity and deeper exploratory charts. | Preserve the existing specialist views. |

### 5.2 Example first-screen hierarchy

1. **Data scope strip:** date span, customer count, transaction count, row exclusions, currency/return policy and complete-through state.
2. **Four to six key summaries:** active customers for the selected period, observed baskets, recorded spend, repeat eligibility, category breadth or concentration—only when the definitions are valid.
3. **Trend:** preferably consistent calendar-month measures; if rolling 90-day windows are shown, label overlap prominently.
4. **Three to five factual insights:** e.g. “Category X represents the largest share of recorded category spend in the selected period.” Each insight should link to the chart and evidence behind it.
5. **Next questions:** buttons/links to lifecycle, value, category, and customer explorer views. Do not label a descriptive pattern as a causal action recommendation.

### 5.3 Customer brief design

An individual profile should answer five questions without requiring the user to read a long feature table:

- **Observed relationship:** first and last observed shopping days, tenure in the dataset, recency and number of purchase days.
- **Shopping rhythm:** median observed gap and number of usable gaps; clearly show when the rhythm estimate is not eligible.
- **Basket/value pattern:** recent spend, median basket size, and distribution compared with a properly defined peer group; state that monetary figures are not profit.
- **Breadth and preference:** leading recorded categories, concentration/diversity, products or categories repeated across shopping occasions.
- **Change and evidence:** recent versus previous comparable period, counts behind the change, data-history caveat and direct links to metric definitions.

The customer profile should distinguish “not enough history” from “low activity,” avoid inferring demographics/personas, and keep identifiable IDs masked unless the user intentionally enables detail viewing.

### 5.4 Carry context across the workflow

When the user selects a cutoff, lifecycle group, behavior label, category or cohort, keep the selection visible as they move into a related chart. Show a clear reset control. For every chart drill-down, expose the filtered population count so users can understand why the denominator changes.

---

## 6. A shared explainable-chart system

### 6.1 Required chart contract

Every important chart should have the following metadata available to the renderer and export process.

| Field | Required content | Example |
|---|---|---|
| `question` | The question the chart answers | “How does repeat purchase vary by recency and frequency?” |
| `title` | Plain-language chart title | “Observed 30-day repeat rate by recency and purchase frequency” |
| `takeaway` | Factual summary derived from the displayed data | “The newest-recency group has a higher observed repeat rate in this sample.” |
| `metric` and `unit` | Metric name and unit | “Repeat rate”, “percent of eligible customers” |
| `population` | Filtered/eligible population | “Customers with complete 30-day follow-up and usable recency/frequency” |
| `denominator` | Eligible n and numerator/denominator where meaningful | “43 / 120 customers purchased again” |
| `time_window` | Cutoff and horizon | “Cutoff 2025-06-30; follow-up 30 days” |
| `uncertainty` | Interval or why unavailable | “95% Wilson interval; independent-customer assumption” |
| `how_to_read` | Axes, bins, legend and reference-line interpretation | “Rows group recency; columns group prior purchase frequency” |
| `caveat` | Material limitations | “Observed activity only; first recorded purchase may not be acquisition” |
| `next_step` | Relevant adjacent investigation | “Open cohort-age retention” |
| `alt_text` | Short, specific chart description for screen readers | “Heatmap of observed 30-day repeat rates by recency and frequency bins.” |
| `data_version` | Reproducibility info | Source fingerprint/run id, cutoff and metric version |

The `takeaway` must be computed from the actual values shown; do not use generic, static copy that could contradict the chart after filters change. If no defensible takeaway can be generated, show a neutral descriptive status rather than manufacturing a narrative.

### 6.2 Reusable chart-card layout

**Header:** question-oriented title plus period and population badge/text.  
**Takeaway:** one short sentence, with the key magnitude and denominator when useful.  
**Chart:** accessible visual with clear units, direct labels where feasible, semantic colors, fixed comparison scales and deliberate tooltips.  
**Evidence footer:** numerator, denominator, interval/sample size and method.  
**Caveat:** concise statement of the biggest limitation.  
**Explore next:** one related view or drill-down.  
**Alternative:** an accessible chart description and data table/export for key views.

Do not solve explainability by adding a full paragraph to every chart. The card should be concise at first glance, with expandable “Method and limitations” detail for users who need it.

### 6.3 Visual design rules

- **Color:** use stable categorical colors for segment names, sequential scales for magnitude and diverging scales for signed differences. A color should have the same meaning across pages. Do not encode value tiers by colors that imply “good/bad” unless the metric truly warrants that judgment.
- **Scales:** comparable cohorts/segments need comparable axes and heatmap ranges. Add reference lines only where they represent a real threshold or population benchmark.
- **Sorting:** sorted horizontal bars are the default for category comparisons. Use small multiples for multiple groups. Avoid a pie chart for many categories.
- **Density:** for large point clouds, show a density/hexbin/contour option and keep the sample size visible; avoid implying all points were drawn if the chart is sampled.
- **Uncertainty:** show error bars or interval bands when they are statistically meaningful, and name the interval type. Do not infer statistical significance from simple interval overlap.
- **Empty states:** explain why no result exists: no rows after filter, insufficient history, no eligible customers, unconfirmed coverage, or model failure are different states.
- **Precision:** avoid false decimals. Round according to data quality and decision context.
- **Interactions:** hover reveals supporting values, not the only definition of the result. Downloads should retain filters, cutoff, denominators and caveat metadata.
- **Accessibility:** never communicate a critical distinction by color alone. Give each Plotly chart a short, specific `alt` description; provide a summary/table for complex plots.

Streamlit's current [`st.plotly_chart` API](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart) includes an `alt` parameter for screen-reader descriptions; the shared wrapper should supply it. See [Principles of Effective Data Visualization](https://pmc.ncbi.nlm.nih.gov/articles/PMC7733875/) for the importance of encoding, context and uncertainty.

### 6.4 Suggested reusable software boundary

Keep calculation functions pure where possible. A chart-building layer should receive a data summary and a chart specification; the Streamlit page should be responsible for layout and interactions rather than recomputing metrics inline.

Potential structure (adapt to current package conventions):

```text
src/retail_analytics/
  metrics/                 # canonical definitions and eligibility rules
  visualization/
    visual_insights_core.py
    product_affinity.py
    chart_specs.py          # question/takeaway/evidence/caveat metadata
    chart_components.py    # shared Plotly/Streamlit rendering
  dashboard/
    app.py                  # app shell and shared session workflow
    views/
      overview.py
      lifecycle.py
      value_behavior.py
      products_categories.py
      customer_explorer.py
      methods_quality.py
```

This is a target direction, not a requirement to move every file in one change. Refactor one primary view at a time and preserve the existing metric contract as the source of truth.

---

## 7. Chart-by-chart improvement portfolio

The repository already provides much of this portfolio. The recommendations below prioritize evidence, accessibility and interpretation over decorative novelty.

### 7.1 Data trust and observation coverage

**Question:** Can this dataset support the particular analysis being requested?  
**Recommended visual:** validation funnel or outcome counts by rejection reason; first/last observed date strip; source complete-through marker; missingness matrix by feature.  
**Show:** rows loaded, accepted and rejected; unique customer identifiers; date span; invalid-date counts; duplicates policy; currency and price mode; source/return assumptions; coverage state.  
**Key safeguard:** rejected issue counts can overlap; say so. Do not make the latest observed transaction date look like the source's completeness guarantee.

### 7.2 Customer-base trend

**Question:** Is observed shopping activity changing across calendar periods?  
**Recommended visual:** monthly lines or aligned small multiples for active customers, basket count and recorded spend.  
**Show:** period duration, customer base, missing periods, cutoff and whether the population is fixed or varies.  
**Key safeguard:** rolling 90-day windows overlap, so consecutive observations are not independent. If the goal is month-to-month change, use non-overlapping calendar-month metrics or show both with labels.

### 7.3 Cohort-age retention heatmap

**Question:** How does observed activity vary by time since first recorded purchase?  
**Encoding:** cohort month on rows, months since first observed shopping date on columns; one consistent 0–100% scale; explicitly blank/grey cells for not-yet-observable periods.  
**Show:** cohort size, eligible denominator, active customers, capture boundaries and the exact “active” definition.  
**Key safeguard:** first-observed cohort may be left-censored; it is not necessarily an acquisition cohort. Do not interpret missing future periods as zero retention.

### 7.4 Time-to-second-purchase survival curve

**Question:** How long does it take for a customer to make a second observed shopping event?  
**Encoding:** Kaplan–Meier step curve with confidence bands plus a number-at-risk row.  
**Show:** definition of event, time origin, end of follow-up, censoring count, eligible cohort size and date range.  
**Key safeguard:** right-censor customers without a second observed purchase by the analysis cutoff; do not mark them as permanent churn. Explain that the time origin is first observed purchase, not verified customer acquisition.

### 7.5 Repeat-rate response surface (RFM)

**Question:** How does an observed future repeat outcome vary across prior recency and frequency?  
**Encoding:** heatmap with fixed scale and visible count/denominator annotations or rich hover, plus explicit empty cells.  
**Show:** eligible `n`, repeat buyers, rate, interval, bins/thresholds and forecast horizon.  
**Key safeguard:** this is descriptive unless the design supports causal conclusions. Gate on confirmed complete follow-up and avoid interpreting sparse cells. Include an unbinned distribution view to reveal how quantile binning groups the data.

### 7.6 Category opportunity portfolio

**Question:** Which categories have broad reach but low intensity, or narrow reach but high intensity?  
**Encoding:** penetration (share of selected customers with category activity) versus spend/repeat intensity among a clearly defined eligible group; point size can represent count. Add medians/reference quadrants only if their definitions are stated.  
**Show:** unique customer denominator for penetration, active category customers for intensity, period, category count and any minimum-support threshold.  
**Key safeguard:** categories may overlap within customers, so category shares need not sum to 100%. Label spend as recorded spend, not margin.

### 7.7 Value-at-risk / purchase-cadence view

**Question:** Which customers have high historical spend and are overdue relative to their own typical purchase rhythm?  
**Encoding:** observed spend against lateness or recency, with categorical eligibility and a density option if points overlap.  
**Show:** cadence eligibility, number of usable gaps, threshold, customer count and spend window.  
**Key safeguard:** call this a **cadence-based monitoring signal**, not churn probability or revenue-at-risk in the financial sense. Spend may not be profit; the customer's past spend is not guaranteed future loss.

### 7.8 Product association network

**Question:** Which products occur together in the same defined basket often enough to warrant investigation?  
**Encoding:** nodes are products; edge width represents pair support or count; use a separate clear encoding for lift, and filter labels to the most supported relationships. Consider a sorted pair table as the default and the network as an exploration view.  
**Show:** number of baskets, support count/percentage, confidence, lift, direction where relevant, minimum support/confidence/lift, and segment/filter context.  
**Key safeguard:** rare pairs can have huge lift. Do not rank recommendations by lift alone. Report the raw pair count and support denominator prominently.

### 7.9 Product trial-to-repeat conversion

**Question:** Among products first observed within an eligible trial window, what share has a qualifying later repeat?  
**Encoding:** ordered dot plot with intervals; support counts next to the estimates.  
**Show:** eligible trials, repeated trials, repeat window, minimum product support and confidence-interval method.  
**Key safeguard:** suppress ineligible or immature trial windows, disclose that product availability and source capture affect repeat rates, and avoid equating repeat with satisfaction.

### 7.10 Segment comparison / forest plot

**Question:** Which segments have different observed outcome rates, and how uncertain are those differences?  
**Encoding:** dots for segment rates, horizontal confidence intervals, vertical baseline and labels sorted by a named metric. Pair with segment count bars only where useful.  
**Show:** segment definition, reference date, eligible `n`, numerator/denominator, interval type, and how many comparisons are displayed.  
**Key safeguard:** exploratory comparisons across many segments are subject to multiple-comparison issues. A point-estimate ranking is not proof that the top segment is truly different.

### 7.11 Customer momentum and basket evolution

**Question:** Did a customer's observed shopping frequency, basket size, breadth or spend change between comparable windows?  
**Encoding:** small multiples or slope charts using matched customer populations and explicit recent/prior windows. A quadrant chart can summarize frequency change versus basket change.  
**Show:** number of customers eligible in both windows, exact window boundaries, medians/distributions and the metric's units.  
**Key safeguard:** comparable-window logic must exclude insufficient tenures consistently. Do not compare different customer populations without showing the membership change.

### 7.12 Segment transitions / alluvial flow

**Question:** How did labels among matched customers change between two cutoffs?  
**Encoding:** Sankey/alluvial flow for a small set of stable labels; a transition matrix for the precise values.  
**Show:** origin and destination cutoffs, matched population, unmatched counts and per-origin transition proportions.  
**Key safeguard:** transition denominators must be explicit. Customers absent from a snapshot are not automatically churned, new or acquired; they may reflect source capture or eligibility changes.

### 7.13 Revenue bridge / waterfall

**Question:** How much of observed spend change comes from customers active in both periods versus customers entering/leaving observed activity?  
**Encoding:** waterfall with clearly reconciled components and a final net total.  
**Show:** exact period boundaries, matching rule, period totals, contribution definitions and the handling of zero activity.  
**Key safeguard:** the decomposition is arithmetic, not causal attribution. Call it a recorded-spend bridge and distinguish revenue from profit.

### 7.14 Customer concentration / Lorenz curve

**Question:** How concentrated is recorded spending among customers?  
**Encoding:** cumulative share of customers versus cumulative share of spend, plus equality reference line and top-share summaries.  
**Show:** number of customers, spend period, treatment of zero spend and share represented by the top 1/5/10% (where sample size supports it).  
**Key safeguard:** concentration is population-specific and depends on inclusion rules. Do not infer profitability or customer dependency without margin/cost context.

### 7.15 Feature distributions and scatterplots

**Question:** What does the entire customer feature distribution look like, and which measures appear associated?  
**Encoding:** histograms/density and box/violin plots for distributions; scatter/density plot for relationships. Use log scales only when labeled and appropriate.  
**Show:** missing/finite counts, sampled n, total eligible population, selection method, and descriptive summaries.  
**Key safeguard:** correlations in transaction features are descriptive and can be dominated by zeros, skewness and repeated observations. The plotted sample should not replace exact population summaries.

### 7.16 Minimum chart bar for release

A chart cannot be marked “ready” until its metric logic is tested, it reports a relevant population/denominator, its scale/labels have been reviewed, its caveat and accessible description exist, and a representative rendered example has been inspected. Passing a pure calculation test alone is not sufficient.

---

## 8. Scientific approach for limited transaction data

### 8.1 Use the simplest defensible method first

Limited transaction history does not automatically mean a complex model is necessary. In many cases the most useful first answer is a descriptive statistic with a correct denominator and clear eligibility rules.

| Evidence tier | Methods | Appropriate use | Required guardrails |
|---|---|---|---|
| **0 — Descriptive** | RFM; recency, frequency, observed spend; basket and category mix; purchase gaps; transparent behavioral rules. | Default for sparse files and exploratory customer understanding. | Label metrics as observed. Make source coverage, history length, missingness and exclusions visible. |
| **1 — Cohort / time-to-event** | Age-aligned cohort repeat curves; Kaplan–Meier time-to-second-purchase. | Enough trustworthy event dates and a known observation boundary. | Right-censor unresolved customers; distinguish first observation from acquisition; no permanent-churn claim from non-repeat alone. |
| **2 — Calibrated empirical rates** | Repeat rate by segment/horizon with Wilson intervals; optionally beta-binomial shrinkage for small groups. | A clear outcome window and an adequately eligible population. | Show raw numerator/denominator beside any shrinkage; disclose prior and interval; suppress unconfirmed follow-up. |
| **3 — Probabilistic purchase models** | BG/NBD or Pareto/NBD transaction counts; Gamma-Gamma monetary-value model where assumptions fit. | Non-contractual repeat purchasing with sufficient history, enough customers, suitable positive monetary values, and plausible model assumptions. | Time-based holdout, simple baselines, calibration checks, residual/fit diagnostics, uncertainty and segment-level error analysis. |
| **4 — Causal/action models** | Randomized experiments, uplift, difference-in-differences, synthetic control or Bayesian structural time-series designs when suitable. | Marketing exposure/treatment and credible comparison information exist. | Transaction histories alone do not establish campaign exposure or causality. State assumptions, diagnostics and treatment/comparison design. |

### 8.2 Small-sample and sparse-history safeguards

- Always show the denominator for a percentage. “50% repeated” must be interpretable as `1/2` or `500/1,000`.
- Keep missing and ineligible metrics as `null`; do not silently convert them to zero.
- Separate `zero observed purchase`, `no qualifying follow-up`, `missing data`, and `insufficient history` in both data and UI states.
- Use Wilson intervals for simple binomial rates where the independent-customer interpretation is defensible. State the assumption and avoid implying that the interval corrects selection bias, repeated comparisons or source-capture problems.
- For differences in aggregate statistics such as median spend, consider customer-level bootstrap intervals using the customer as the resampling unit. Avoid treating transaction lines as independent customers.
- For numerous exploratory segment comparisons, label them exploratory and consider multiplicity adjustment before formal claims.
- Set analysis-specific minimum support. There is no universal sample threshold that makes every estimate reliable; thresholds should reflect the metric, imbalance, decision risk and tolerated uncertainty.
- If using Bayesian or empirical shrinkage for sparse segment estimates, show raw rates/counts and the adjusted estimate separately, and explain the prior/pooling logic.
- Test whether segment and threshold labels are stable across reference cutoffs and resamples. A cluster can be separable in one snapshot but unstable or not commercially useful. Research on supermarket segment stability shows that stability should be evaluated rather than assumed ([Fonseca & Cardoso, 2007](https://doi.org/10.1057/palgrave.jt.5750052)).
- For product affinity, a high lift based on one or two baskets is not strong evidence of a useful cross-sell relationship. Filter and display support count as well as lift/confidence.
- Keep the existing rule-based labels as a transparent baseline. Do not introduce K-means solely because it is familiar. If clustering is later added, compare results with rules, examine feature scaling/skew, report separation metrics such as silhouette with caution, and evaluate internal/dynamic stability.

### 8.3 Use time correctly

- Preserve explicit historical cutoff boundaries and the no-lookahead rule.
- Future outcome windows must begin after the cutoff and lie inside a confirmed complete-through period.
- Cohort curves need age-aligned periods so recent cohorts are not penalized for having less follow-up.
- Rolling windows overlap; do not treat rolling values as independent monthly measurements.
- Segment transitions require matched customers and clear origin/destination population definitions.
- Distinguish calendar duration from number of shopping days; the existing metric contract makes this distinction for some measures, and the UI should preserve it.

---

## 9. Model roadmap: what to add, and when

### Stage 1 — Descriptive insight engine (now)

Use the current feature generation and rules to describe observed behavior. Make the main outputs interpretable:

- recency/frequency/spend distributions;
- number of purchase days and usable purchase gaps;
- median basket value alongside mean/AOV;
- recent versus previous windows with comparability eligibility;
- category spend shares, concentration and breadth;
- age-aligned observed repeat rates only when follow-up is eligible.

**Do not call these predictions, CLV, profitability, or churn.**

### Stage 2 — Empirical repeat outcomes (after the P0 coverage gate)

Estimate repeat within clearly defined horizons (e.g. 30/60/90 days) with:

- explicit eligible denominator;
- repeat count / eligible count;
- interval estimate;
- horizon and cutoff;
- source-coverage state;
- group size and minimum-support policy.

A beta-binomial hierarchical estimate can be tested later to stabilize sparse segments, but it must not hide the raw count or manufacture an estimate for incomplete follow-up.

### Stage 3 — Probabilistic future transaction count

Candidate models include BG/NBD or Pareto/NBD for non-contractual repeat purchases. The classical BG/NBD paper positions it as a practical alternative to Pareto/NBD, while customer-base modeling literature validates components using holdout periods rather than in-sample fit alone ([Fader, Hardie & Lee, 2005](https://doi.org/10.1287/mksc.1040.0098)).

Before implementation:

1. Check that the business process fits the non-contractual model assumptions.
2. Verify purchase event definition and data capture boundaries.
3. Build temporal train/validation/test cutoffs without leakage.
4. Compare against simple baselines such as historical average frequency, recency-frequency groups and cohort rates.
5. Evaluate count error/calibration overall and by tenure, frequency, cohort and value group.
6. Report uncertainty and the horizon; monitor drift and model version.

Do not equate low predicted activity with permanent churn. Models infer behavior from a process assumption, not direct observation of a customer's intention.

### Stage 4 — Monetary value / CLV

A Gamma-Gamma model can be evaluated only if the monetary data meet its assumptions and values represent a suitable monetary outcome. The classical Pareto/NBD + Gamma-Gamma framework has been tested on holdout data ([Fader, Hardie & Lee, 2005](https://doi.org/10.1509/jmkr.2005.42.4.415)).

Use precise labels:

- **Observed spend:** amount recorded in the selected historical window.
- **Expected future revenue:** a model-based forecast of monetary transactions.
- **Contribution profit / net CLV:** only available when gross margin, discounts, returns, variable costs, appropriate horizon and discounting are adequately represented.

Do not label spend or revenue as “profit CLV.” Validate predictions by time-based holdout and report errors, bias and calibration by group.

### Stage 5 — Causal marketing measurement (different data requirements)

Attribution, incrementality, media mix modeling, and uplift modeling need campaign exposure, treatment assignment, marketing spend, time/geography variation or a credible control design. Transactions alone cannot tell whether marketing caused a purchase.

If the product expands into this area, create a separate scope with its own data contract, estimands, identification assumptions, diagnostics and experiment support. Do not mix causal claims into current descriptive customer views.

---

## 10. Data and metric contract improvements

### 10.1 Import scope card

Before analysis, capture or explicitly mark unknown:

- source system and export timestamp;
- data time zone and date interpretation;
- latest transaction date present in file;
- source complete-through date and coverage scope;
- currency and whether the value field is unit price or line total;
- returns/refunds/cancellations handling;
- customer identifier scope/stability;
- transaction/basket definition and duplicate policy;
- exclusions applied after validation.

Avoid making an unknown field look known by filling it with a default. Present “unknown” as an honest state.

### 10.2 Data semantics and labels

Use stable definitions in metric code, tooltips, chart descriptions, exports and documentation. A metric label is part of the scientific contract.

| Avoid | Prefer | Reason |
|---|---|---|
| “Churned customers” based on non-repeat in the file | “No recorded repeat within the observed follow-up window” | Non-repeat can reflect censoring or incomplete capture. |
| “Loyal customers” from repeated purchases alone | “Observed repeat purchasers” or a clearly defined patronage rule label | Repeat purchase alone does not prove attitude or exclusivity. |
| “Most profitable customers” from spend | “Highest observed spend in the selected window” | Profit requires margin and cost data. |
| “CLV” for historical spend | “Observed spend” or, if validated, “expected future revenue” | CLV is forward-looking and typically economic, not merely historical sales. |
| “Campaign drove sales” from association | “Sales co-occurred with the campaign period” unless causal design supports more | Correlation is not incrementality. |
| Zero for every unavailable ratio | `null` / “insufficient evidence” with a reason | Missingness and zero are different states. |

### 10.3 Audit/manifest fields

Ensure every export can be traced to:

- source fingerprint/run ID and app/package version;
- price mode, date format/time zone and column mapping;
- validation counts and exclusions;
- snapshot cutoff(s), historical window definitions and future horizons;
- coverage confirmation and complete-through date;
- segment/rule/threshold version, calibration snapshot and interpolation method;
- sampling policy and chart population filters;
- metric definitions/version where useful.

### 10.4 Resolve contract/code differences explicitly

Use `DECISIONS.md` as an implementation checklist. For each decision, document the adopted behavior, update the data contract, add regression tests, and close the item in the decision log. Avoid silently changing a metric definition because a chart looks better with different numbers.

---

## 11. Implementation plan and code touchpoints

The phases below are intentionally incremental. Ship a small, testable change at each stage rather than doing one risky dashboard rewrite.

### Phase 0 — Correctness gate (P0)

**Likely files:**

- `src/retail_analytics/dashboard/app.py`
- `src/retail_analytics/dashboard/dashboard_backend.py`
- `src/retail_analytics/core/pipeline/run.py` (manifest/config propagation, if appropriate)
- `README.md`, `DATA_CONTRACT.md`, `METRICS.md`, `docs/DASHBOARD.md`
- `tests/test_dashboard.py`, `tests/test_metrics_contract.py`, relevant visualization tests

**Tasks:**

- Remove automatic `coverage = True` behavior.
- Capture explicit confirmation and complete-through date.
- Make per-horizon eligibility a tested contract.
- Display unconfirmed/incomplete states at every dependent chart.
- Invalidate stale workspaces/results if the relevant coverage configuration changes.
- Export the coverage state and dates.

**Acceptance:** all P0 cases in Section 4 pass.

### Phase 1 — User-question navigation (P1)

**Likely files:** `dashboard/app.py`; new `dashboard/views/` modules as incremental refactor; view tests.

**Tasks:**

- Replace flattened horizontal radio navigation with 5–7 top-level areas.
- Add a Start Here overview before specialist views.
- Move methods, transition diagnostics and sensitivity controls into the relevant page or Advanced analytics.
- Keep current cutoff, selected segment/category and filters visible as context.
- Add a persistent reset/clear-filter control and indicate active filters.
- Build the Customer Brief view from existing profile and timeline data before adding entirely new metrics.

**Acceptance:** a new non-technical user can find the six primary analytical jobs without knowing model names; advanced views still exist.

### Phase 2 — Shared chart contract and renderer (P1)

**Likely files:** `visualization/chart_specs.py` (new), `visualization/chart_components.py` (new), current chart helper in `dashboard/app.py`, per-view renderers.

**Tasks:**

- Define a typed chart specification or dataclass with the fields listed in Section 6.
- Require `title`, `question`, `alt_text`, metric/unit, population, period and caveat for primary views.
- Add a reusable takeaway/evidence/caveat layout, with optional expanded Method panel.
- Pass short, chart-specific `alt` descriptions to Streamlit Plotly charts.
- Standardize number formatting, tick formatting, semantic color maps, axis titles, direct labels, legends and downloadable filenames.
- Add textual summaries/tables for dense graphs.
- Keep calculation code outside Streamlit rendering functions.

**Acceptance:** the top five user journeys use the same explanation layout and cannot silently omit required evidence metadata.

### Phase 3 — Sparse-data and chart quality (P1)

**Likely files:** `visual_insights_core.py`, `product_affinity.py`, shared chart components, metric contract docs and visualization tests.

**Tasks:**

- Define per-analysis minimum evidence policies; avoid one global sample threshold for every method.
- Show exact numerator, denominator, eligible count and horizon for outcome estimates.
- Retain Wilson intervals where appropriate; add customer-level bootstrap or appropriate intervals for selected aggregate comparisons.
- Add “preliminary” and “insufficient evidence” states for small groups and empty heatmap cells.
- Show count/support with product-pair lift and confidence. Add warning for rare pairs.
- Add segment-stratified deterministic sampling, density options and visible full-population/sample counts.
- Check fixed heatmap scales and consistent semantic colors for multi-panel comparisons.
- Add threshold stability/sensitivity views and report the reference snapshot.

**Acceptance:** no chart turns missing/ineligible into zero, and all primary outcome charts show n/denominator/horizon or an explicit unavailable reason.

### Phase 4 — Validated predictive layer (P2, only after previous phases)

**Likely files:** new isolated model/evaluation module, saved model config/results, visualizations, tests and docs.

**Tasks:**

- Define temporal evaluation cutoffs and no-leakage train/validation/holdout construction.
- Establish simple repeat/count baselines first.
- Evaluate BG/NBD or Pareto/NBD only if the purchase process assumptions are credible.
- Evaluate Gamma-Gamma only if monetary-value assumptions and input data are appropriate.
- Report holdout metrics, bias/calibration by segment, horizon, uncertainty and baseline comparisons.
- Version the model, fitted cutoff, input features, data source and assumptions.

**Acceptance:** a model is not displayed as predictive unless it performs credibly out of time and the UI names its target precisely. Historical spend cannot be relabeled as CLV.

### Phase 5 — Maintainability, test, deployment (P1/P2 parallel work)

**Tasks:**

- Split the large dashboard into smaller domain view modules over several PRs.
- Preserve one canonical metric implementation; do not calculate different formulas in different pages.
- Fix broad exception handling that hides unexpected errors; log actionable failures while keeping user messages safe.
- Pin verified dependencies and run the full automated suite.
- Add a real upload → validation → analysis → chart-rendering smoke test in the target environment.
- Store and inspect screenshot baselines for key pages across representative datasets.
- Verify upload limits, worker timeout, authentication, transport encryption, access control and workspace cleanup.
- State exactly what is runtime-tested; never infer UI correctness from a successful pure-function test.

**Acceptance:** a known fixture completes the user flow without a runtime error; computed results match expected values; failures are visible and diagnosable; data access is restricted in the deployed environment.

---

## 12. Testing and acceptance criteria

### 12.1 Test dataset matrix

Maintain a small deterministic fixture suite that includes each of these conditions:

| Fixture | Why it matters | Expected behavior |
|---|---|---|
| One purchase per customer | Common sparse-history case | Cadence and repeat eligibility are unavailable, not fabricated. |
| Many repeat purchases with regular cadence | Reference case for gap metrics | Median gap and related measures match known expected values. |
| Irregular purchasing | Tests high gap variability | Rhythm labels follow documented rules and expose limited evidence where required. |
| Incomplete follow-up | Coverage defect regression | Future outcome rates are disabled for incomplete horizons. |
| Confirmed complete-through date | Positive coverage case | Only customers with full target horizon are eligible. |
| Customer with no purchase in a window | Zero versus missing semantics | Valid zero count is distinct from an undefined ratio. |
| Category appears in only one or two baskets | Rare association handling | Support and raw count are visible; low-support insight is not ranked as strong. |
| Strongly skewed spend with outliers | Robust summaries | Median/IQR and distribution views remain interpretable; means are clearly named. |
| Duplicate rows | Duplicate policy | Default and explicit deduplication are tested/documented. |
| Invalid/ambiguous dates or timezone-aware dates | Date logic | Parsing and cutoff boundaries are deterministic and documented. |
| Returns/refunds/negative quantities | Current data limitations | Invalid rows are surfaced; users cannot mistake the data for net revenue. |
| Changed source/cutoff/coverage setting | Cache/result safety | Stale result files are invalidated and cannot be reused silently. |
| Large file near limit and high-cardinality categories | Performance/UX | Limits and sampling are disclosed; UI remains responsive or fails with an actionable message. |

### 12.2 Automated test layers

1. **Metric unit tests:** formulas, date boundaries, null semantics, numerator/denominator and eligibility.
2. **Property/contract tests:** date-window invariants, metric schema/version, manifest fields, stable outputs, threshold metadata.
3. **Analysis tests:** segmentation, cohort denominators, survival censoring, product association support/confidence/lift and transitions.
4. **UI integration tests:** each primary page handles empty, sparse, normal and invalid inputs; coverage gates show the correct message.
5. **Visual regression checks:** render important charts on fixed data; inspect scales, empty cells, axes, labels, alt text and overflow. A screenshot test should supplement, not replace, metric tests.
6. **Performance smoke tests:** realistic near-limit row counts and category cardinality, on the actual deployment dependencies and resource constraints.
7. **Security tests:** access control, upload size/type/content checks, temporary file cleanup and export escaping.

### 12.3 Release checklist

- [ ] Future outcomes are disabled by default until capture completeness is confirmed.
- [ ] The complete-through date is separate from latest observed transaction date.
- [ ] Source scope, currency, returns/refunds, basket key and customer-ID semantics are visible.
- [ ] Each outcome chart has eligible population, numerator/denominator, horizon and uncertainty or an explicit unavailability reason.
- [ ] Every primary chart has a question, takeaway, caveat and accessible description.
- [ ] Missing, zero, ineligible and incomplete states are distinct in UI and exported data.
- [ ] Sample limits and sampling methods are visible; samples are not presented as full population.
- [ ] Category/product associations include support count and denominator, not just lift.
- [ ] Customer identifiers are not exposed in chart tooltips and remain hidden unless deliberately requested.
- [ ] Segment/threshold definitions and calibration reference periods can be inspected.
- [ ] Prediction models have temporal holdout results and baseline comparisons before release.
- [ ] No causal/attribution claims are made from transaction data alone.
- [ ] Full tests plus end-to-end UI runtime smoke tests pass in the release environment.
- [ ] Known limitations and testing scope are recorded in the release notes.

---

## 13. Product success metrics

The application itself should be evaluated, not only the customers represented by its data.

### 13.1 Usability and comprehension

- **Time to first useful insight:** elapsed time from valid upload to first understood finding.
- **Task completion:** proportion of users who can answer a set of core questions (e.g., “which customer group has the highest observed repeat rate?”) without help.
- **Comprehension accuracy:** whether users correctly identify the denominator, time horizon and key caveat of a chart.
- **Navigation success:** whether users find lifecycle, value, category and customer explorer views without searching the methods page.
- **Evidence drill-down rate:** whether a chart takeaway lets a user inspect the supporting counts/data instead of only seeing a headline.

### 13.2 Reliability and trust

- **Eligibility correctness:** proportion of future-outcome views with valid coverage and horizon gates.
- **Metric parity:** same metric has same value across chart, table, CSV and documentation for a known fixture.
- **Empty-state correctness:** number of missing/ineligible results incorrectly displayed as zero (target: zero).
- **Chart accessibility coverage:** proportion of primary charts with useful alt descriptions and a text/table alternative.
- **Render stability:** failures and out-of-bounds layouts across representative input sizes.

### 13.3 Model value (only if predictive methods are introduced)

- Temporal holdout performance versus simple baseline.
- Calibration by customer tenure/frequency/value group.
- Bias/error by forecast horizon and observation cohort.
- Stability/drift across repeated data refreshes.
- Decision utility measured against a predeclared business objective; do not optimize an offline score alone.

Set numerical targets after baseline usability and reliability measurement. Avoid inventing target improvements before measuring the current system.

---

## 14. Security, privacy, and deployment considerations

The dashboard documentation already warns that data are processed on the application server and that temporary-session cleanup is best-effort, not secure erasure. Treat the following as deployment requirements rather than promises provided by a temporary workspace.

- Restrict access to trusted users; do not expose an open public upload endpoint for identifiable transaction data.
- Use encrypted transport and appropriate authentication/authorization in the deployment environment.
- Do not log raw transaction rows, customer IDs or sensitive file contents unnecessarily.
- Keep customer identifiers out of chart hover text and exported aggregate files unless the user explicitly requests an authorized customer-level export.
- Confirm data retention, workspace cleanup, backups and incident-response policies for the actual host.
- Treat uploaded files as untrusted; apply size/type/parse limits and resource/time budgets. A worker subprocess is not, by itself, a security sandbox.
- Verify formula injection protections in CSV exports and review escaping for all supported export formats.
- Test access isolation and cleanup under disconnect, exception and timeout scenarios.
- Document whether results are stored only in the current session or retained, and for how long.

Do not claim secure deletion solely because temporary directories are removed; filesystem, host snapshots and backups can have separate retention semantics.

---

## 15. Suggested delivery backlog

Use this as a sequence of reviewable engineering tickets. Owners/estimates should be assigned after confirming the current branch state and deployment constraints.

| ID | Task | Priority | Main touchpoints | Definition of done |
|---|---|---|---|---|
| CA-01 | Remove implicit observation-coverage confirmation | **P0** | `dashboard/app.py`, `dashboard_backend.py`, manifests, tests | Coverage defaults off; horizon eligibility verified by regression tests. |
| CA-02 | Store asserted complete-through date and source scope | **P0** | import UI, config/run manifest, docs | Latest event date and complete-through date are separate and visible in exports. |
| CA-03 | Add per-horizon unavailable states | **P0** | future validation, cohort views, repeat charts | Each dependent view explains whether the blocker is incomplete coverage, insufficient follow-up or insufficient n. |
| CA-04 | Resolve decision-log contract drift | **P1** | `DECISIONS.md`, `DATA_CONTRACT.md`, code/tests | Duplicate/timezone/quantile metadata/error handling decisions match implementation and have regression tests. |
| CA-05 | Reorganize primary navigation | **P1** | `dashboard/app.py` and view modules | Question-based entry points; advanced views remain accessible; filters/cutoff are contextual. |
| CA-06 | Define chart specification contract | **P1** | new chart spec module and shared renderer | Required metadata is validated for every primary chart. |
| CA-07 | Add accessible chart descriptions and alternatives | **P1** | chart wrapper, key views | Chart `alt`, visible takeaway/evidence and table/download alternative for primary views. |
| CA-08 | Normalize uncertainty and sample reporting | **P1** | visual functions, view renderers, metric docs | Counts, eligible n, intervals and support warnings appear consistently. |
| CA-09 | Improve product-affinity interpretation | **P1** | `product_affinity.py`, product charts/tests | Support count, support denominator, confidence and lift are displayed; rare pairs are flagged. |
| CA-10 | Add the customer brief | **P2** | Customer view, profile features, timeline chart | Clear narrative plus drill-downs to underlying features; identifiers hidden by default. |
| CA-11 | Add temporal model-evaluation harness | **P2** | isolated modeling/evaluation package | Baselines and time-based holdouts run without leakage; report includes bias/calibration by group. |
| CA-12 | Prototype probabilistic repeat/value models | **P2** | model module and optional views | Models only ship after defensible assumptions and holdout performance. |
| CA-13 | Add UI smoke and visual regression tests | **P1/P2** | test suite/CI/release script | Major pages render on deterministic fixtures; defects are detected before release. |
| CA-14 | Confirm deployment security and data lifecycle | **P1** | deployment configuration, docs, tests | Access/retention/cleanup behavior verified in the target environment. |

### Recommended order of execution

1. Complete CA-01 through CA-03 first. They protect analytical validity.
2. Complete CA-04 and CA-05 next. They reduce semantic drift and make the existing capability usable.
3. Implement CA-06 through CA-09 together so the chart system is consistent instead of retrofitted independently into each page.
4. Add CA-10 after the shared evidence components exist.
5. Implement CA-13 and CA-14 as part of each release and before exposing data to broader users.
6. Only then pursue CA-11 and CA-12.

---

## 16. Research and documentation references

### 16.1 Repository evidence

- Project overview and documented limits: [README.md](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/README.md)
- Metric formulas, windows, denominators and null semantics: [METRICS.md](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/METRICS.md)
- Input/output schemas and data-quality contract: [DATA_CONTRACT.md](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/DATA_CONTRACT.md)
- Known implementation/documentation decisions: [DECISIONS.md](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/DECISIONS.md)
- Dashboard flow and implementation: [app.py](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/dashboard/app.py)
- Dashboard upload/backend workflow: [dashboard_backend.py](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/dashboard/dashboard_backend.py)
- Aggregate visualizations and declared limitations: [VISUAL_INSIGHTS.md](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/VISUAL_INSIGHTS.md)
- Observed patronage rules and limitations: [PATRONAGE.md](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/PATRONAGE.md)
- Pure visualization calculations: [visual_insights_core.py](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/visualization/visual_insights_core.py)
- Basket association logic: [product_affinity.py](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/visualization/product_affinity.py)

### 16.2 Customer-base modeling

1. **Fader, Hardie & Lee (2005). “Counting Your Customers the Easy Way: An Alternative to the Pareto/NBD Model.” _Marketing Science_.** Presents BG/NBD as a practical model for repeat transaction counts. It does not remove the need to verify assumptions and evaluate on holdout periods. [DOI: 10.1287/mksc.1040.0098](https://doi.org/10.1287/mksc.1040.0098)
2. **Fader, Hardie & Lee (2005). “RFM and CLV: Using Iso-Value Curves for Customer Base Analysis.” _Journal of Marketing Research_.** Combines Pareto/NBD and Gamma-Gamma components and reports holdout tests. [DOI: 10.1509/jmkr.2005.42.4.415](https://doi.org/10.1509/jmkr.2005.42.4.415)
3. **Fader, Hardie & Shang (2010). “Customer-Base Analysis in a Discrete-Time Noncontractual Setting.” _Marketing Science_.** Useful context for non-contractual repeat-purchase models in discrete time. [DOI: 10.1287/mksc.1100.0580](https://doi.org/10.1287/mksc.1100.0580)
4. **Fonseca & Cardoso (2007). “Supermarket customers segments stability.” _Journal of Targeting, Measurement and Analysis for Marketing_.** Discusses internal and dynamic stability of customer segments and illustrates that stability should be assessed rather than assumed. [DOI: 10.1057/palgrave.jt.5750052](https://doi.org/10.1057/palgrave.jt.5750052)

### 16.3 Causal inference, chart quality and technical APIs

5. **Brodersen et al. (2015). “Inferring causal impact using Bayesian structural time-series models.” _Annals of Applied Statistics_.** Counterfactual time-series causal inference requires a suitable time-series/covariate setup and assumptions. [Google Research publication page](https://research.google/pubs/inferring-causal-impact-using-bayesian-structural-time-series-models/)
6. **Midway (2020). “Principles of Effective Data Visualization.” _Patterns_.** Covers effective encoding and the importance of expressing uncertainty in scientific visuals. [Open-access article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7733875/)
7. **Streamlit. `st.plotly_chart` API documentation.** Documents Plotly rendering and the `alt` description parameter for assistive technologies. [API docs](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart)
8. **Plotly. Hover text and formatting.** Useful for building deliberate, metric-specific hover content. [Documentation](https://plotly.com/python/hover-text-and-formatting/)

### 16.4 Research hygiene

- Prefer primary papers, official API docs and code-level evidence over unsourced claims in blog posts.
- Treat preprints as provisional and label them as such.
- Keep the reference list current when a method is adopted; document the actual assumptions implemented in the code, not only the paper's general rationale.
- Research relevance does not prove that a method fits a particular retailer's data. Validate on the app's own data and decision context.

---

## 17. Audit boundary

This plan is based on the repository's public `main` branch and its visible source/documentation inspected on **12 October 2026**. The review covered the README, metrics/data contract, decision log, dashboard and backend structure, visualization modules and tests. It did **not** run the user's local or production deployment and did not validate behavior against private customer data. Runtime performance, deployment security, actual rendered UI and business usefulness must be verified in the target environment before release.

The plan intentionally does not claim that every suggested method should be implemented. Its order is evidence-first: fix eligibility, make current insights understandable, establish uncertainty and validation, then introduce more predictive sophistication only when the data justify it.
