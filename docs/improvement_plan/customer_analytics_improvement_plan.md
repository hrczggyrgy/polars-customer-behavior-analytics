# Customer Analytics App: Evidence-Led Improvement Plan

**Repository:** [hrczggyrgy/polars-customer-behavior-analytics](https://github.com/hrczggyrgy/polars-customer-behavior-analytics)  
**Audit date:** 12 October 2026  
**Product goal:** Help people understand customers through explainable, trustworthy, high-quality charts, even when transaction histories are sparse.

## Executive recommendation

The project already has a strong analytics engine and a surprisingly broad set of research-oriented visualizations. It includes Polars-based transaction processing, explicit metric definitions, data-quality auditing, snapshot features, behavioral rule labels, cohort analyses, RFM response surfaces, Kaplan-Meier time-to-repeat analysis, cadence validation, product-affinity analyses, category-opportunity maps, customer concentration, revenue bridges, and transition views.

The next step should **not** be to add a large number of algorithms. Prioritize scientific correctness, simpler navigation, a consistent explanation layer for every chart, and honest small-sample uncertainty. Add probabilistic customer-value predictions only after these foundations are reliable and predictions can be checked against later observed transactions.

The single highest-priority defect is in `src/retail_analytics/dashboard/app.py`: the upload workflow sets `coverage = True` automatically and passes this through as `coverage_confirmed`. Yet the README and dashboard documentation say future-outcome calculations require explicit confirmation of complete follow-up. Since the latest transaction date is not proof that the source captured every transaction through that date, this is a scientific-validity issue, not just a UI issue.

## 1. What is already strong

- **Data contract and audit:** documented input/output fields, row rejection reasons, basket grain, date windows, currency semantics, and null behavior.
- **Temporal discipline:** historical customer features are computed only through a chosen cutoff; future outcomes are meant to be gated by capture completeness.
- **Interpretable behavior layer:** relative value tiers, lifecycle/evidence labels, style and trajectory tags, and threshold calibration are more explainable than an opaque clustering model.
- **Useful research charts:** cohort-age heatmaps, time-to-second-purchase survival curves, RFM response surfaces, value-at-risk plots, category penetration/growth portfolios, product association networks, customer-concentration curves, transition/alluvial views, and revenue bridges already exist.
- **Testing foundation:** tests cover metric contracts, pipeline behavior, dashboard helpers, patronage rules, and research visualizations.
- **Good cautionary language:** the documentation already states that observed spend is not profit, first observed purchase is not confirmed acquisition, descriptive segment differences are not causal effects, and rule labels are not calibrated churn or CLV predictions.

Relevant implementation references: [README](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/README.md), [metric definitions](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/METRICS.md), [dashboard](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/dashboard/app.py), [visual analytics documentation](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/VISUAL_INSIGHTS.md), and [patronage documentation](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/PATRONAGE.md).

## 2. Prioritized gap assessment

| Priority | Gap | Evidence in the repository | Recommendation |
|---|---|---|---|
| **P0 — correctness** | Future-outcome coverage is assumed instead of explicitly confirmed. | The dashboard workflow sets `coverage = True` and passes it to the analysis job; documentation says follow-up outcomes require confirmed complete capture. | Make coverage confirmation explicit, default it to off, record the stated observation end date and source-scope limitations, and suppress all future-outcome results until the gate passes. Add regression tests. |
| **P1 — user experience** | Navigation is too broad and effectively flat. | `NAV_GROUPS` is built in the dashboard, but its lists are flattened into one horizontal radio control. There are many views and several legacy/exploration views. | Make the groups real in the UI. Use a primary workflow and a second-level view selector. Keep advanced/research charts available but out of the default path. |
| **P1 — explainability** | Chart explanations are not represented by one consistent contract. | The code has many thoughtful captions and some rich hover templates, but explanatory metadata is authored view by view. The common Plotly wrapper does not pass an accessible chart `alt` description. | Standardize chart cards: question, takeaway, evidence, denominator, period, sample size, uncertainty, caveat, and next investigation. Pass a specific accessible description to each chart. |
| **P1 — data semantics** | Important transaction-only limitations can be mistaken for customer value or loyalty. | Negative quantities are rejected; returns/refunds are unsupported; currency is opaque; spend is explicitly not profit; first observed purchase may not be acquisition. | Show a prominent data-scope banner. Call the current values *observed spend/value tiers*, not profit CLV. Record return handling, currency, identifier scope, source coverage and basket definition during import. |
| **P1 — evidence quality** | Uncertainty, sample support and multiplicity warnings are not uniform across every comparison. | Some views use Wilson intervals and expose denominators; documentation notes independent-customer assumptions, sample limits and no multiple-comparison correction. Other descriptive views are point estimates. | Require `n`, numerator/denominator where applicable, eligibility and the time window on every outcome chart. Use intervals where meaningful; label small samples as preliminary and warn when many exploratory comparisons are viewed. |
| **P1 — representativeness** | Bounded scatter plots can visually under-represent small groups. | Some feature scatter plots deterministically hash-select at most 2,000 customers; captions disclose this. | Use stratified samples when colour represents a segment, retain exact population summaries, show sampled/eligible counts, and offer density/hexbin views for large populations. Never imply a sample is the whole population. |
| **P1 — UX accessibility** | Charts depend heavily on colour/hover and do not set a screen-reader name through the shared wrapper. | `chart()` calls `st.plotly_chart` without its `alt` argument. Many plots rely on hover and colour-coded categories. | Add concise alt descriptions, visible data summaries, keyboard-friendly controls, semantic colour palettes, and tables or downloadable data for key figures. Test colour-blind contrast and narrow-screen layouts. |
| **P2 — customer understanding** | Customer exploration is more feature-oriented than narrative-oriented. | The customer view allows exact-ID lookup with a feature table and purchase timeline, gated by a consent checkbox. | Add a customer brief: observed relationship age, recent purchase pattern, typical gap, category mix, recent-vs-prior change, evidence limitations, and a linked timeline. Keep raw identifiers hidden until deliberately requested. |
| **P1 — contract drift** | Some code/documentation discrepancies are already recorded but not all are closed. | `DECISIONS.md` lists action items around duplicate handling, timezone wording, quantile interpolation and threshold metadata. | Resolve the decision log, align contract and behavior, and add contract checks so future drift is caught in review. |
| **P2 — model roadmap** | The product does not yet offer a validated probabilistic future-value model, which is appropriate for the current data limits but leaves a future extension opportunity. | Current documentation says labels are rules, not calibrated churn, profit CLV or elasticity; no marketing exposure data are present in the documented contract. | Do not bolt on black-box churn or attribution models now. First build time-based validation. Later evaluate BG/NBD or Pareto/NBD for repeat transactions and Gamma-Gamma for monetary value only when assumptions and holdout performance are defensible. |
| **P2 — maintainability and release confidence** | The dashboard UI has grown into a very large single module, and documentation notes limitations in runtime UI verification. | `app.py` contains thousands of lines of view rendering; docs say the dashboard UI was not runtime-tested in the authoring environment. | Extract domain views, view-model/calculation contracts and reusable chart components. Run a real Streamlit smoke test in CI or the release environment, not only helper-level tests. |

### Highest-risk defect: coverage gate

The code path in `app.py` currently:

1. infers the dataset's latest date from the observed rows;
2. sets `coverage = True` in the dashboard workflow;
3. passes this value as `coverage_confirmed` into the analysis job.

This allows a data file's latest date to be treated as proof of complete capture. That proof does not follow from transaction data alone. A source export can stop early, omit stores/channels, or contain delayed events.

**Required behavior:** coverage confirmation must be an explicit user decision with clear wording. Store both (a) the latest date that the source is known to be complete through and (b) the file's latest observed transaction date. If completeness is unknown, historical descriptions may still run, but future repeat/return validation must be marked **unavailable: observation coverage not confirmed**, not zero and not a prediction. The setting and stated coverage end must appear in the run manifest and exported report.

Repository evidence: [`app.py` coverage workflow](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/dashboard/app.py#L4838-L4841), [`DASHBOARD.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/DASHBOARD.md), and [`README.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/README.md).

## 3. Target user experience

Design around user questions rather than model names.

### Primary navigation

1. **Start here** — data trust, what is known, what is missing, customer base summary and the 3–5 most useful signals.
2. **Lifecycle & repeat** — observed cohorts, time to second purchase, repeat rates by horizon, and cadence patterns.
3. **Value & risk** — observed spending distribution, RFM response surface, concentration, momentum and customers who are unusually overdue relative to their own cadence.
4. **Products & categories** — category penetration, category spend intensity, product associations and trial-to-repeat behavior.
5. **Explore customers** — an understandable profile with a time line and supporting transactions/features.
6. **Methods & data quality** — definitions, formulas, coverage assumptions, threshold reference date, rejected-row summary and exports.

Retain an **Advanced analytics** area for the alluvial flows, transition matrices, sensitivity views and other specialist charts. Avoid putting every analysis option in one horizontal selector.

### Shared chart explanation pattern

Every chart should follow the same pattern:

- **Question:** What decision or customer question does this view answer?
- **Takeaway:** One concise, factual sentence generated from the displayed aggregates (not an unsupported causal claim).
- **Evidence:** numerator, denominator, population size, comparison period and applicable confidence interval.
- **How to read it:** short interpretation of axes, bins, colour and reference lines.
- **Caveat:** sparse history, incomplete coverage, changing population, sampling or other known limitation.
- **Next step:** one relevant drill-down, filter or adjacent chart.

Use neutral language such as “customers with no recorded repeat within 90 days” instead of “churned customers”; “high observed spend” instead of “most profitable”; “associated with” instead of “drives”. When the user opens a segment, customer or category, carry the selected filters into the next chart so the story remains coherent.

## 4. Recommended chart portfolio

The code already contains much of this portfolio. Focus first on discoverability, consistent explanation, visual quality and validation rather than adding more chart types.

| Chart | Main question | Recommended encoding | Evidence to show |
|---|---|---|---|
| **Data trust / coverage panel** | Can these records support the analysis being requested? | Quality funnel or stacked counts by validation outcome, date coverage bar, missingness matrix | Rows loaded/accepted/rejected, customer IDs available, first/last observed date, stated complete-through date, currency/return policy and exclusions |
| **Customer base trend** | Is customer activity and recorded sales changing? | Calendar-month line charts or aligned small multiples; avoid presenting overlapping rolling windows as independent periods | Active customers, baskets, recorded sales, period length, eligible base and any exclusions |
| **Cohort-age heatmap** | How does observed activity change after the first recorded purchase? | Cohort rows × relationship-age columns; grey/blank not-yet-observed cells; consistent 0–100% scale | Cohort size, active/eligible denominator, capture boundary, note that first observed is not acquisition |
| **Time-to-second-purchase survival** | How long do customers take to make a second recorded purchase? | Kaplan–Meier step curve with uncertainty band and number-at-risk row | Right-censoring, event definition, confidence band, date limits and first-observed caveat |
| **RFM response surface** | Which combinations of recency and frequency show different future repeat rates? | Heatmap with fixed colour scale, explicit missing cells and per-cell sample sizes | Observed future horizon, numerator/denominator, Wilson interval or equivalent, minimum cell support and capture gate |
| **Category opportunity portfolio** | Which categories have reach versus intensity opportunities? | Penetration on one axis, spend per active customer or repeat intensity on the other, point size for eligible customers; median/reference lines | Active-customer denominator, category support, chosen period and category definitions |
| **Value-at-risk / cadence scatter** | Which customers have high observed spend and are overdue relative to their personal rhythm? | Recency or lateness vs observed spend; semantic colours; density overlay if crowded | Eligibility rules, thresholds, sample size, observed spend not profit; no unsupported “churn probability” claim |
| **Product association network** | Which items occur together often enough to merit investigation? | Nodes for products; edge width for support; a separate or clearly encoded lift scale; filterable labels | Basket count, pair support, confidence, lift, minimum support, directionality and small-sample warning |
| **Segment comparison forest/dot plot** | Which segments differ, and how uncertain are the differences? | Dots with confidence intervals; show segment size; order by one declared metric | Exact denominator, absolute and relative rates, interval type, unadjusted exploratory-comparison warning |
| **Customer brief and timeline** | What has actually been observed about this customer? | Timeline of baskets/spend plus small multiples for cadence, recent trend and category breadth | Number of purchase days/baskets, time span, missing history caveat and direct links to metric definitions |

### Chart design rules

- Use one semantic encoding consistently: colour means the same thing on every page; use a sequential scale for magnitude, a diverging scale for signed change and categorical colours for named segments.
- Do not rely on hover alone. Put the key evidence and an accessible short description outside/in the chart metadata; provide a table or export for dense charts.
- Show denominators and uncertainty on outcomes. A bare rate of 50% is not informative without whether it is 1/2 or 500/1,000.
- Prefer direct labels, sensible axes and restrained gridlines over decorative effects. Use annotations to explain a real threshold or important change, not to make every point look significant.
- For more than a few categories, use sorted horizontal bars or small multiples rather than pie charts or a crowded network.
- Use fixed scales across comparable cohorts or segments. Do not change heatmap bounds independently per panel if users are meant to compare panels.
- Keep sample and population views distinct. Where a 2,000-customer sample is used, show sample method and population size; stratify by segment when appropriate.

## 5. Scientific approach for limited transaction data

### Use the simplest defensible method first

| Tier | Method | When to use | Guardrail |
|---|---|---|---|
| 0 — Descriptive | RFM, recency, frequency, observed spend, basket and category mix, purchase gaps, rule-based behavioral labels | Default for sparse transaction files | Label values as observed, not predicted. Make missingness and capture limits visible. |
| 1 — Cohort / time-to-event | Age-aligned cohort repeats and Kaplan–Meier time-to-second-purchase | When customer event dates and observation boundaries are sufficiently clear | Handle right-censoring; distinguish first observed purchase from acquisition; do not infer permanent churn. |
| 2 — Calibrated empirical rates | Repeat rates by horizon and segment with Wilson intervals; optionally beta-binomial shrinkage for sparse segment rates | Enough customers are eligible for an honest denominator; rates answer a clear question | Show raw counts alongside any shrinkage; disclose prior and interval; no estimates for unconfirmed follow-up. |
| 3 — Probability models | BG/NBD or Pareto/NBD for transaction counts; Gamma-Gamma for monetary value | Repeat-purchase process is plausible, there is adequate history/population, values are positive and model assumptions are checked | Compare against simple baselines on a time-based holdout. Check calibration by customer group and forecast horizon. Do not call revenue “profit CLV”. |
| 4 — Causal / action models | Randomized uplift, controlled experiments, difference-in-differences or time-series counterfactuals | Only after campaign/treatment exposure and credible control/time-series data exist | Purchase histories alone cannot tell whether a marketing action caused a purchase. Do not add attribution or uplift models to the current transaction-only contract. |

Do not add K-means just because it is popular. The existing rule labels are easier to explain. If unsupervised segments are added later, compare them to the rule-based baseline, standardize/skew-correct features where justified, evaluate silhouette or related separation measures, and assess stability across resamples, random seeds and later snapshots. A segment that looks clean in one plot may be unstable or commercially useless.

### Uncertainty and sparse groups

1. Put sample size and denominator beside each outcome rate.
2. Keep binomial intervals for proportions (the current Wilson intervals are a reasonable baseline under their stated independent-customer assumption).
3. Use customer-level bootstrap intervals for aggregate differences such as median spend or spend change, where the bootstrap unit matches the business unit.
4. For many simultaneous exploratory comparisons, flag the analysis as exploratory and consider multiplicity adjustment before formal claims.
5. Use explicit “insufficient evidence” states for cells with too few customers; do not convert missing or ineligible values to zero.
6. Re-check segment and threshold stability through time. Quantile labels calibrated on one reference snapshot are relative to that reference population and are not universal customer truths.
7. Association rules need minimum basket support and robust counts; large lift from a rare pair is not automatically a promising recommendation.

## 6. Implementation roadmap

### Phase 0 — Stop invalid conclusions (first)

- Replace the hard-coded coverage assumption with an unchecked, explicit control.
- Record the source's asserted complete-through date separately from the maximum observed event date.
- Gate all future outcomes and cadence-validation views on the saved confirmation and horizon completeness.
- Show a prominent “historical descriptions only” state when coverage is not confirmed.
- Add regression tests for unconfirmed coverage, insufficient follow-up and a source file that ends before the asserted observation period.

**Acceptance:** No future-outcome rate is calculated or shown when coverage is not explicitly confirmed; the exported manifest records the decision and dates.

### Phase 1 — Reorganize the UI around customer questions

- Replace the long horizontal view selector with real grouped navigation.
- Put a “Start here” page first, showing data scope, quality, observation period and a small set of high-value summaries.
- Move rarely used diagnostics and older/research views under Advanced analytics without deleting their functionality.
- Design the customer explorer as a readable profile plus a timeline, not primarily a wide feature table.
- Keep selected filters, time cutoff and cohort/segment context visible across a drill-down.

**Acceptance:** A new user can find lifecycle, value/risk, product/category, customer profile and methods without knowing the names of the underlying statistical methods.

### Phase 2 — Build a shared explainable-chart system

- Add a small reusable chart specification (question, title, metric, units, population/denominator, period, sample policy, caveat and alt description).
- Build shared components for insight summary, evidence, uncertainty, methodology and “explore next”.
- Standardize typography, axis titles, tick formatting, colour meaning, legends, empty states and download names.
- Add `alt` descriptions to Plotly charts and a text/table alternative for every key visualization.
- Add chart screenshots or visual regression tests for the main pages; do not rely only on unit tests of calculations.

**Acceptance:** Every primary chart explains its question, what the axes and encodings mean, how much data supports it, and what it cannot establish.

### Phase 3 — Strengthen evidence for limited data

- Display eligible counts and uncertainty consistently; establish documented minimum-support policy by analysis type.
- Add sampling disclosures and segment-aware sampling for bounded scatter plots.
- Add sensitivity checks for relative thresholds and segment migration across snapshots.
- Improve product-pair results with support, confidence, lift, counts and a prominent warning for rare pairs.
- Add compact population summaries so users can see distribution shape and outliers before interpreting a median or segment average.

**Acceptance:** Small groups do not produce unqualified rankings or “risk” assertions; the app distinguishes missingness, zero activity and ineligibility.

### Phase 4 — Add predictions only when validation passes

- Create a model evaluation module separate from UI code.
- Define time-based calibration/holdout splits and backtests for expected repeat purchases.
- Compare probabilistic purchase models against simple frequency/cohort baselines.
- Add a future-value estimate only if model assumptions, monetary values and out-of-time accuracy are defensible; call it expected revenue unless contribution margin/costs are included.
- Treat marketing attribution, uplift and media mix modeling as a separate future product scope requiring exposure/spend/treatment data.

**Acceptance:** Prediction pages show training period, holdout period, calibration, baseline comparison, uncertainty and eligible customer count. No model ships based only on in-sample fit.

### Phase 5 — Release reliability and privacy hardening

- Split the huge dashboard module into domain views and shared chart components without duplicating metric logic.
- Run the full test suite and a real Streamlit runtime smoke test in the deployed dependency environment.
- Maintain deterministic fixtures for sparse histories, one-time purchasers, missing IDs, invalid dates, duplicated rows, category sparsity, zero-price lines, rejected negative quantities and incomplete capture.
- Verify authentication, encrypted transport, access controls, size/time limits and session-workspace cleanup for the real deployment. Temporary-session cleanup is not equivalent to secure erasure.
- Document what has been runtime-tested and pin the dependency versions used for release.

**Acceptance:** A known fixture can be uploaded, validated, analyzed and explored end-to-end; key chart calculations and labels match expected values; private transaction files are not publicly accessible.

## 7. Recommended delivery sequence

| Order | Deliverable | Priority | Definition of done |
|---|---|---|---|
| 1 | Explicit observation coverage gate | P0 | Future outcomes stay unavailable by default; regression tests cover the gate. |
| 2 | Data-scope and quality summary | P1 | User sees data dates, row quality, currency/return assumptions and customer identity limitations before reading insights. |
| 3 | Question-based navigation | P1 | Main user journey has fewer, grouped entry points; advanced views remain accessible. |
| 4 | Shared chart explanation and accessibility contract | P1 | Key charts show question, takeaway, denominator, timeframe, uncertainty and accessible description. |
| 5 | Sparse-sample safeguards | P1 | Minimum support, uncertainty, multiplicity caveat and population/sample distinction are consistent. |
| 6 | Customer brief | P2 | A non-technical user can understand a profile and navigate to supporting evidence. |
| 7 | Validated repeat/value predictions | P2 | Out-of-time tests beat a baseline and outputs are appropriately named and caveated. |
| 8 | Experiment and causal analytics | Later | Only start when exposure/treatment and credible comparison data are available. |

## 8. Acceptance checklist

- [ ] Future-outcome analysis is disabled until complete observation coverage is explicitly confirmed.
- [ ] The source's complete-through date is stored separately from its latest observed transaction date.
- [ ] Purchases-only, returns/refunds, currency and first-observed-customer limitations are prominent in the UI.
- [ ] Every outcome metric has an eligible population, denominator, horizon and uncertainty or an explicit reason why none is shown.
- [ ] Every key chart has a question, a data-grounded interpretation, a caveat and an accessible text description.
- [ ] The same metric and colour semantics are consistent across pages and cutoffs.
- [ ] Sampling and truncated display limits are visible and never presented as the complete population.
- [ ] Customer profile detail is consent-gated and sensitive identifiers are hidden by default.
- [ ] Segment labels and value tiers expose their rules, reference period and stability limitations.
- [ ] Predictions are evaluated on a temporal holdout and compared with simple baselines.
- [ ] No causal/attribution claims are made from purchase transactions alone.
- [ ] The release workflow runs all regression tests plus an end-to-end UI smoke test.

## 9. Research and documentation references

### Repository audit evidence

- Project README and documented methodological notes: https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/README.md
- Metric formulas, denominators and window semantics: https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/METRICS.md
- Data contract and stated limitations: https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/DATA_CONTRACT.md
- Dashboard flow and view implementation: https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/dashboard/app.py
- Visual insight functions and limitations: https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/VISUAL_INSIGHTS.md
- Patronage definitions and safeguards: https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/PATRONAGE.md

### External research and technical documentation

1. Fader, Hardie & Lee (2005), “Counting Your Customers the Easy Way: An Alternative to the Pareto/NBD Model,” *Marketing Science*. BG/NBD is a candidate transaction-count model, not a reason to skip holdout validation: https://doi.org/10.1287/mksc.1040.0098
2. Fader, Hardie & Lee (2005), “RFM and CLV: Using Iso-Value Curves for Customer Base Analysis,” *Journal of Marketing Research*. Shows how recency, frequency and monetary value relate to modeled future customer value: https://doi.org/10.1509/jmkr.2005.42.4.415
3. Brodersen et al. (2015), “Inferring causal impact using Bayesian structural time-series models,” *Annals of Applied Statistics*. Counterfactual causal methods require suitable time-series/covariate structure and assumptions: https://research.google/pubs/inferring-causal-impact-using-bayesian-structural-time-series-models/
4. Fonseca & Cardoso (2007), “Supermarket customers segments stability,” *Journal of Targeting, Measurement and Analysis for Marketing*. Segment stability should be assessed rather than assumed: https://doi.org/10.1057/palgrave.jt.5750052
5. “Principles of Effective Data Visualization” (2020), available through PubMed Central. Discusses uncertainty, sample size, and the value of showing data points alongside summaries: https://pmc.ncbi.nlm.nih.gov/articles/PMC7733875/
6. Streamlit, `st.plotly_chart` API documentation. Includes its `alt` parameter for a short accessible chart description: https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
7. Plotly, hover text and formatting documentation. Useful for deliberate, metric-specific tooltips rather than default metadata dumps: https://plotly.com/python/hover-text-and-formatting/
8. Hassan (2026), “Customer Segmentation Analytics in Retail: An Empirical K-Means Clustering Study,” SSRN preprint. A recent applied example comparing clustering methods and seed stability; treat preprint findings as provisional: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6817639

## Audit boundary

This is a repository and methodology review based on the files visible on the `main` branch on 12 October 2026. I inspected the project documentation and source structure, but did not execute the user's local deployment or verify behavior against their private customer data. The priority findings above are grounded in repository code/documentation; runtime performance and rendered UI should be confirmed in the target environment before release.
