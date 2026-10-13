# Customer Analytics App — Current-Stage Audit and Detailed Improvement Plan

**Repository:** [hrczggyrgy/polars-customer-behavior-analytics](https://github.com/hrczggyrgy/polars-customer-behavior-analytics)  
**Review date:** 13 October 2026  
**Branch reviewed:** `main`  
**Latest commit reviewed:** [`aa668d38` — Implement customer analytics improvement plan (P0/P1 + CA-12)](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/commit/aa668d38cb20f2b490927a406c9dab00418fe45b)  
**Product goal:** Help a user understand customers from limited retail transaction data using explainable, scientifically honest, visually outstanding charts.

> **Executive recommendation:** The codebase has advanced meaningfully: coverage controls, question-oriented views, chart-evidence components, sparse-data safeguards, customer briefs, and probabilistic customer models now exist in the repository. Do not expand the model or chart catalogue yet. First (1) restore green CI, (2) remove placeholder predictive-evaluation results, (3) close the remaining observation-coverage contract gaps, (4) finish the dashboard/module consolidation, and (5) verify the actual deployed UI. The next release should be defined by trustworthy evidence and smooth interpretation—not by the number of new features.

---

## Contents

1. [Executive status](#1-executive-status)
2. [Scope and review method](#2-scope-and-review-method)
3. [Current product baseline](#3-current-product-baseline)
4. [Prioritized findings](#4-prioritized-findings)
5. [P0: Make CI green and meaningful](#5-p0-make-ci-green-and-meaningful)
6. [P0: Make model evaluation scientifically valid](#6-p0-make-model-evaluation-scientifically-valid)
7. [P0/P1: Finish observation-coverage semantics](#7-p0p1-finish-observation-coverage-semantics)
8. [P1: Consolidate the dashboard architecture](#8-p1-consolidate-the-dashboard-architecture)
9. [P1: Make the chart-evidence system consistent](#9-p1-make-the-chart-evidence-system-consistent)
10. [P1: Design for sparse transaction histories](#10-p1-design-for-sparse-transaction-histories)
11. [P1: Improve customer understanding in the UI](#11-p1-improve-customer-understanding-in-the-ui)
12. [Chart portfolio and per-view acceptance criteria](#12-chart-portfolio-and-per-view-acceptance-criteria)
13. [P1/P2: Integrate security and privacy utilities](#13-p1p2-integrate-security-and-privacy-utilities)
14. [P2: Predictive-model roadmap](#14-p2-predictive-model-roadmap)
15. [Data contract and documentation alignment](#15-data-contract-and-documentation-alignment)
16. [Detailed delivery phases](#16-detailed-delivery-phases)
17. [Test strategy and acceptance matrix](#17-test-strategy-and-acceptance-matrix)
18. [Release checklist and go/no-go rules](#18-release-checklist-and-gono-go-rules)
19. [Backlog with dependencies](#19-backlog-with-dependencies)
20. [Success measures](#20-success-measures)
21. [Risks and mitigations](#21-risks-and-mitigations)
22. [Research and technical references](#22-research-and-technical-references)
23. [Audit boundary and unresolved verification](#23-audit-boundary-and-unresolved-verification)

---

## 1. Executive status

### Overall judgement

**Stage: feature-rich prototype moving toward a product, but not release-verified.** The repository contains a solid transaction-processing and descriptive-analysis base and has implemented important parts of the earlier improvement plan. The principal release blocker is now not a missing chart; it is that parts of the predictive-evaluation implementation still return placeholder values that look like real measurements, while the latest automated CI run is failing.

### Status snapshot

| Area | Status from source review | Next decision |
|---|---|---|
| Transaction pipeline and feature snapshots | Existing, broad and documented | Preserve behavior; keep formulas in the metric contract/core layer |
| Observation coverage | Explicit confirmation checkbox and complete-through input added | Close date semantics and persist all confirmation metadata consistently |
| Question-oriented navigation | Two-level navigation added | Finish removing duplicate/legacy routes and test view routing |
| Shared chart evidence | `ChartCard`, `ChartSpec`, evidence thresholds and chart-spec catalogue added | Complete adoption; remove competing inline chart implementations |
| Sparse-data policy | Threshold and bootstrap helpers added | Apply uniformly; test boundary values and small/zero denominators |
| Customer brief/timeline | Dedicated views added | Validate interpretability, privacy masking and realistic edge cases |
| BG/NBD, Pareto/NBD, Gamma-Gamma | Fitting/prediction code exists | Gate results on convergence, assumptions and genuine holdout validation |
| Model evaluation | Placeholder measurements remain in `evaluation.py` | **Do not present as a successful evaluation until implemented** |
| CI | Latest run failed; nine preceding returned runs also reported failure | **Restore green CI before merging more features** |
| Security utilities | `security.py` added | Integrate or remove/defer unused utilities; test the live upload path |
| Live Streamlit/UI smoke testing | Not established by this review | Run against fixtures and representative data in target deployment |
| Documentation | Full improvement plan committed under `docs/` | Update its status to reflect the 13 October implementation state |

### Immediate release blockers

1. **Model-evaluation placeholders:** `run_temporal_holdout_evaluation()` returns hardcoded metrics rather than predictions scored against observed future outcomes.
2. **CI failure:** the latest GitHub Actions run has conclusion `failure`. Individual job logs were not accessible through the available GitHub tool in this review, so the recorded failing step remains unconfirmed. A concrete likely test-collection defect is present in `tests/test_smoke.py`: it annotates a function with `Tuple[bool, str]` without importing `Tuple` or enabling postponed annotations.
3. **Coverage metadata/date contract:** the UI now collects explicit confirmation, a complete-through date and a scope note. Verify these fields are persisted and used consistently end-to-end; the current date input caps the complete-through date at the latest transaction date, which can be too restrictive when a source is known to be complete after the most recent purchase.
4. **Duplicate dashboard implementations:** new view modules and chart components exist, but the main dashboard retains large inline implementations and defines functions with names already imported from `dashboard.views`. This can make fixes land in a module that is not actually used by the route.

---

## 2. Scope and review method

This is a repository source/documentation review of the publicly available `main` branch as of the review date. The review considered the README and metric/data contracts, recent commits, workflow configuration and Actions status, dashboard entry point, view modules, visualization components/specifications, model fitting/evaluation code, security utilities, and smoke-test code.

Key files:

- [`README.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/README.md)
- [`DATA_CONTRACT.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/DATA_CONTRACT.md)
- [`METRICS.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/METRICS.md)
- [`DECISIONS.md`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/DECISIONS.md)
- [`src/retail_analytics/dashboard/app.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/dashboard/app.py)
- [`src/retail_analytics/dashboard/dashboard_backend.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/dashboard/dashboard_backend.py)
- [`src/retail_analytics/dashboard/views/`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/tree/main/src/retail_analytics/dashboard/views)
- [`src/retail_analytics/visualization/chart_components.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/visualization/chart_components.py)
- [`src/retail_analytics/visualization/chart_specs.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/visualization/chart_specs.py)
- [`src/retail_analytics/visualization/visual_insights_core.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/visualization/visual_insights_core.py)
- [`src/retail_analytics/modeling/models.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/modeling/models.py)
- [`src/retail_analytics/modeling/evaluation.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/modeling/evaluation.py)
- [`src/retail_analytics/security.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/security.py)
- [`tests/test_smoke.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/tests/test_smoke.py)
- [`tests/test_visualizations.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/tests/test_visualizations.py)
- [CI workflow](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/.github/workflows/ci.yml)

### Evidence labels used in this plan

- **Verified in source:** the implementation or configuration is directly visible in repository files.
- **Reported by GitHub:** metadata such as the latest commit or workflow conclusion was obtained from GitHub’s API.
- **Needs runtime verification:** source review cannot prove the deployed app renders, performs or behaves correctly in an actual execution environment.
- **Recommendation:** a proposed change, not a claim that the feature already exists.

Do not interpret a code-level test or successful import as proof that a chart looks correct or a predictive method is statistically useful.

---

## 3. Current product baseline

### 3.1 What is already valuable and should be retained

The repository provides a transaction-only foundation that avoids requiring a large dense customer-by-product matrix for core customer analysis. Existing capabilities include:

- CSV/Parquet ingestion, explicit price semantics, row validation, rejection audit and output schemas.
- Customer features using historical cutoffs and no-lookahead conventions.
- Recency, frequency, monetary spend, basket value, purchasing gaps, category breadth/concentration, product/category repetition and recent-versus-prior changes.
- Behavioral, trajectory, lifecycle, evidence and observed-value labels based on documented rules.
- Cohort summaries and transitions across snapshots.
- Repeat timing, cadence, RFM response surfaces, spend concentration, revenue bridges/waterfalls and product/category analysis.
- A metric contract that defines windows, denominators, null handling and interpretation limits.
- A new chart component/specification layer with question, takeaway, evidence, how-to-read, caveat and next-step fields.
- New customer brief/timeline views, modeling module and security helper module.

These are useful building blocks. The plan should consolidate and test them rather than replace them all with a new framework.

### 3.2 The intended product experience

A non-technical retail analyst should be able to answer, without understanding implementation-specific labels:

1. **What can I trust in this data?** Quality, observation scope, time coverage, missing values, duplicate handling and known limitations.
2. **Who are the customers in this observed dataset?** Activity, frequency, spend distribution and the degree of heterogeneity.
3. **How do customers return?** First-observed cohorts, time to second purchase, repeat rates at stated horizons and cadence.
4. **How is behavior changing?** Comparable period changes, category evolution, segment migration and the reasons behind aggregate spend movement.
5. **What do they buy and repeat?** Category participation and concentration, product co-occurrence, trial-to-repeat patterns and basket composition.
6. **What does one customer’s history say?** A readable profile with links back to underlying events, denominators and definitions.
7. **How strong is the evidence?** Sample size, eligibility, uncertainty, threshold source and what cannot be concluded.

### 3.3 Non-goals unless the data schema expands

Do not represent transaction-only analysis as if it can independently establish:

- Profit or margin-based CLV without costs/margins and an appropriate financial definition.
- Emotional loyalty, exclusivity, or share of wallet without external context.
- Permanent churn in a non-contractual setting from an arbitrary inactivity threshold.
- Marketing incrementality, campaign attribution or causal response without exposure, treatment and/or suitable experimental/causal data.
- Demographic personas from purchase totals alone.
- Price elasticity from observed price/spend association alone.
- Verified customer acquisition when only the first *observed* transaction is known.

---

## 4. Prioritized findings

| ID | Priority | Finding | Current evidence | Required outcome |
|---|---|---|---|---|
| CA-CI-01 | P0 | Latest CI is failing | Current Actions run #31 on commit `aa668d38` reports `failure` | All CI matrix jobs pass and skipped tests are zero |
| CA-MODEL-01 | P0 | Temporal evaluation returns placeholder metrics | `evaluation.py` sets zero errors/perfect calibration without computing future actuals | Only real observed holdout metrics can be rendered as model results |
| CA-TEST-01 | P0 | Smoke test has undefined `Tuple` annotation | `test_smoke.py` references `Tuple[bool, str]` but lacks `typing.Tuple` import and future annotations | Test module collects cleanly on supported Python versions |
| CA-COVER-01 | P0/P1 | Coverage input semantics need final contract alignment | Checkbox, complete-through date and scope note entered in UI; end date and metadata flow through backend must be verified | Complete-through and confirmation scope are explicit, persisted, and used to gate each horizon |
| CA-ARCH-01 | P1 | New view modules coexist with legacy inline views | `app.py` retains large view branches and duplicate function names | Exactly one route and implementation per view |
| CA-CHART-01 | P1 | Shared chart system is not yet universal | Legacy `chart()`/Plotly calls coexist with `ChartCard`/`ChartSpec` | All core product views follow one evidence/alt-text contract |
| CA-SPARSE-01 | P1 | Evidence rules need universal adoption and validation | Threshold/bootstrap helpers exist, but views may still use point estimates or custom logic | Small samples are clearly labelled and no missing evidence is rendered as zero/confidence |
| CA-SEC-01 | P1 | Security module is not visibly wired through the dashboard entry point | Utilities are present; main app has no clear end-to-end import/use of the module | Actual uploads, exports, session cleanup, masks and access controls are tested |
| CA-MODEL-02 | P1 | Model assumptions and naming can outstrip available data | BG/NBD, Pareto/NBD and Gamma-Gamma fit/predict code exists | Gate predictions on data eligibility, convergence and validated performance; label spend-based outputs accurately |
| CA-DOC-01 | P1 | Improvement-plan status can drift from source | Existing plan records the earlier coverage issue as outstanding | Maintain a status ledger: done / partial / not done / runtime-unverified |
| CA-UX-01 | P1 | Feature breadth can still overwhelm a new user | Advanced and legacy views remain accessible within the same selector structure | Guided primary journey + advanced area; avoid naming the experience only after algorithms |
| CA-OPS-01 | P1 | Deployed behavior is not established by source audit | Docs previously noted limitations around runtime/UI verification; this review did not execute target deployment | End-to-end release evidence is documented |

---

## 5. P0: Make CI green and meaningful

### 5.1 Fix smoke-test collection

**File:** `tests/test_smoke.py`

Tasks:

1. Import `Tuple` from `typing`, or change the return annotation to `tuple[bool, str]` (supported by Python 3.10+) and standardize type annotations.
2. Run `pytest --collect-only -q` before executing the suite to isolate collection failures.
3. Fix package imports in tests to use the installed package path consistently, e.g. `retail_analytics.security`, rather than relying on repository-root namespace behavior such as `src.retail_analytics.security`.
4. Remove `assert True`, `assert converged or True`, and any similar unconditional acceptance checks.
5. Where an optimizer may legitimately fail on tiny synthetic data, assert that the failure is reported safely and predictions are not presented as validated. Have a separate deterministic fixture that should converge.
6. Add tests for non-convergence, NaN/infinite inputs, empty populations, all-zero monetary values, one-customer data and insufficient history.

### 5.2 Make CI execution deterministic and debuggable

**File:** `.github/workflows/ci.yml`

Tasks:

1. Keep supported Python versions in the matrix, but ensure that each CI step has a single responsibility and a clear failing output.
2. Run format/lint/test once in the main job. The current “fail if any tests skipped” step invokes pytest a second time and parses human-readable output using brittle string conditions. Replace that logic with a robust check or configure the test suite to fail on expected skips through explicit markers/policies.
3. Preserve `pytest -q -rs` output as an artifact or use the GitHub Actions logs for failure diagnosis; never hide the actual failing assertion behind a generic status.
4. Consider a separate optional integration/UI smoke job after fast unit tests pass.
5. Pin or constrain dependency ranges enough to keep the release environment reproducible, then test the deployed dependency set rather than assuming latest transitive packages are compatible.
6. Add a status badge only after the main branch is green consistently.

### 5.3 Required CI acceptance criteria

- Ruff check succeeds.
- Ruff format check succeeds.
- Test collection succeeds in Python 3.10 and 3.12.
- The complete required test suite passes in both matrix jobs.
- No tests are skipped unintentionally; any intentionally optional test has a documented reason and is not silently treated as passed.
- New statistical/model tests fail when their expected calculation is deliberately perturbed.
- The result from each run is reproducible for fixed synthetic fixtures.

**Do not proceed to a broader model/UI release until these pass.**

---

## 6. P0: Make model evaluation scientifically valid

**Primary file:** [`src/retail_analytics/modeling/evaluation.py`](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/src/retail_analytics/modeling/evaluation.py)  
**UI:** `src/retail_analytics/dashboard/views/modeling.py`  
**Models:** `src/retail_analytics/modeling/models.py`

### 6.1 Current issue

The temporal evaluation function currently prepares split data and fits at least BG/NBD, but then returns hardcoded `EvaluationMetrics` with zero MAE/RMSE/MAPE and perfect-looking calibration values. It returns only a BG/NBD result while comments stand in for Pareto/NBD and Gamma-Gamma. In `compute_evaluation_metrics()`, the concordance statistic is also a constant placeholder, and by-horizon outputs are currently initialized to NaN/zero counts rather than calculated horizon-wise results.

This is a **blocker** because the UI has a Model Evaluation view. A polished chart showing fabricated performance is more dangerous than a disabled chart.

### 6.2 Short-term safe behavior

Before the full evaluation exists:

- Disable the “validated performance” state and label the feature “Experimental — evaluation incomplete”.
- Do not show placeholder zeros or calibration slope/intercept as model performance.
- Surface optimizer convergence, log-likelihood and eligibility status without conflating those with predictive accuracy.
- If a prediction is shown for exploratory purposes, label it “unvalidated model output”, show horizon and model assumptions, and avoid ranking customers by it.

### 6.3 Implement correct temporal splits

Use three chronological intervals with clearly documented boundaries:

- Training: data through `train_end`.
- Validation: a subsequent complete interval for model selection/tuning.
- Test: a later, untouched interval used once for final reporting.

Prevent transaction leakage. Every feature for a cutoff must use data on or before that cutoff. Actual target outcomes must only be drawn from the relevant future window. Filter any record whose outcome horizon is not fully observed under the declared coverage contract.

Define and test boundary semantics for `C`, `C+1`, horizon end dates, and customers with no future purchase. Avoid overlapping training labels with validation/test target windows when estimating repeated outcomes.

### 6.4 Baselines first

Implement at least these honest baselines before model comparison:

1. **Zero-future-purchases baseline** where a zero forecast is a valid lower-information benchmark.
2. **Historical-rate baseline** learned only from training data for the chosen horizon and customer eligibility group.
3. **Frequency-rate baseline** using purchases per observed time, with a clearly defined exposure duration and no division-by-zero ambiguity.
4. If useful, a simple cohort/category baseline that is available from the training period alone.

The existing `naive_repeat_baseline()` describes a frequency-binned placeholder/proxy rather than true future repeat-rate evaluation. Replace it with a real baseline or rename it to reflect descriptive summary only.

### 6.5 Prediction targets and metrics

Use target-appropriate evaluation:

| Target | Primary metrics | Supporting diagnostics |
|---|---|---|
| Future transaction count | MAE, RMSE, mean actual vs predicted, horizon-level count | Error by frequency/tenure strata, calibration plot, zero-inflation diagnostics |
| Any purchase within horizon | Brier score, log loss, ROC-AUC/PR-AUC where appropriate | Reliability curve, calibration slope/intercept, observed vs predicted by bins, prevalence and confidence intervals |
| Time to next/second purchase | Survival-specific metrics where available, calibration at horizon, concordance calculated correctly | Kaplan–Meier observed curve vs model prediction, risk-set counts, censoring information |
| Expected transaction monetary amount | MAE/RMSE or deviance appropriate to target; compare with simple mean/basket baseline | Error by history length, sensitivity to extreme basket values, predictive intervals if provided |
| Customer value forecast | Horizon-specific monetary error and calibration | Compare with historical-spend and cohort baselines; explicitly state revenue vs profit definition |

Do not use MAPE as the headline metric when actual values frequently equal zero; the common epsilon-denominator form can yield huge, misleading values. Use MAE/RMSE and, where appropriate, a documented scaled error or count-model deviance. Any reported metric needs the number of eligible customers and the target window.

### 6.6 Model fitting and convergence

For BG/NBD, Pareto/NBD and Gamma-Gamma:

- Validate array lengths, finite values, valid frequency/recency/age ranges and monetary-value constraints before optimization.
- Verify that all parameter bounds and likelihood terms are numerically safe.
- Distinguish optimizer convergence from model adequacy.
- Record optimizer status/message, number of observations, fit date/cutoff, training interval, data/config hash, model version and parameter estimates.
- Treat failure to converge, non-finite likelihood or out-of-support input as a failed fit—not as a prediction-ready model.
- Add numerical tests against known/reference values or independently computed examples, including zero-frequency and very short histories.
- Check the assumptions of Gamma-Gamma monetary modeling; do not apply it blindly to refunds, negative values or data where the monetary assumptions are inappropriate.

### 6.7 Prediction report contract

Every evaluation output should contain:

- `model_name` and `model_version`;
- train/validation/test start and end dates;
- feature cutoff and target horizon;
- `coverage_confirmed`, source complete-through date and coverage scope;
- eligible population `n`, actual outcome totals and prediction totals;
- metrics with correct target/units;
- baseline metrics and delta vs baseline;
- optimizer convergence and warnings;
- calibration and uncertainty information where applicable;
- an explicit `evaluation_status` such as `valid`, `insufficient_data`, `fit_failed`, `coverage_unconfirmed` or `not_implemented`.

Never serialize or render a success-shaped zero value for “not computed”. Use null/missing plus a reason.

### 6.8 Acceptance criteria

- Tests prove that metrics change when predictions change and match hand-calculated toy data.
- Validation and test targets are computed from actual future transactions, not from RFM proxies.
- Baselines use only training data.
- No perfect metrics appear on fixtures unless the predictions truly match actuals.
- Convergence failures and insufficient coverage block the validated badge.
- The dashboard renders the evaluation status and caveats visibly.
- The CLV view distinguishes monetary spend prediction from profit and never calls a model validated until the evaluation contract passes.

---

## 7. P0/P1: Finish observation-coverage semantics

**Primary files:** `dashboard/app.py`, `dashboard/dashboard_backend.py`, `DATA_CONTRACT.md`, `METRICS.md`, cohort/repeat functions and tests.

### 7.1 What is improved already

The current UI now includes an explicit observation-coverage section:

- A checkbox defaults to false.
- An explanatory warning says all in-scope records must be captured.
- A complete-through date input appears after confirmation.
- A scope note can describe included stores/channels/business units.
- Coverage-related values are passed into the analysis job.
- The dashboard has messaging that future validation is unavailable when coverage is unconfirmed.

This addresses the earlier unsafe default. Preserve the default-off behavior.

### 7.2 Remaining semantic problem: latest event date is not the same as capture-complete-through date

The current UI sets the complete-through date picker’s `max_value` to the latest observed transaction date (`last`). This prevents a user from asserting that an export is complete through a later date on which no transaction happened.

Example: the last transaction row might be dated 30 June, but the point-of-sale export can be verified complete through 31 July. The lack of transactions in July is itself an observable zero only if the source is complete through July. A date field capped at 30 June cannot represent that fact.

**Fix:** distinguish at least three dates:

- `latest_observed_transaction_date`: maximum transaction event date in file;
- `coverage_complete_through`: the last date for which all in-scope events are known to be captured;
- `analysis_cutoff`: the feature snapshot reference date.

Do not infer the second from the first. Let the user supply it within a defensible bound declared for the export/source (e.g., a source-provided reporting end date), not necessarily `last`.

### 7.3 Persist the user's assertion and scope

The UI submits values such as `coverage_complete_through`, `coverage_scope_note` and `coverage_confirmation_source`. The backend currently uses `observed_through` as an end value and persists coverage fields, but the contract should be checked to ensure all separately named fields are preserved and not merely passed and then dropped.

Required manifest and quality fields:

```json
{
  "coverage_confirmed": false,
  "coverage_complete_through": null,
  "coverage_confirmation_source": null,
  "coverage_scope_note": null,
  "latest_observed_transaction_date": "YYYY-MM-DD",
  "analysis_cutoff": "YYYY-MM-DD",
  "coverage_policy_version": "1"
}
```

Keep `observed_through` only if its meaning is precisely specified; otherwise migrate away from a field whose name can be confused with observed transaction date or user-asserted capture date. Add backwards-compatible reads for old manifests and migrate documentation.

### 7.4 Gate every future outcome by horizon eligibility

For cutoff `C` and horizon `h`, future outcomes should be available only when:

- explicit coverage confirmation exists;
- the asserted complete-through date reaches at least `C + h`;
- the relevant customer is otherwise eligible for the analysis;
- the outcome window is formed with the documented boundary convention;
- any model-specific minimum history/support criteria are met.

A single global `coverage=True` is not enough if the chosen cutoff/horizon exceeds the asserted follow-up end. The gate must be calculated per cutoff and horizon.

### 7.5 Required coverage tests

- Default checkbox off results in all future-outcome views disabled.
- Complete coverage through `C+29` allows a 30-day window only if the project’s inclusive/exclusive convention makes that window complete; test the exact boundary explicitly.
- Coverage ending one day short blocks the horizon.
- Latest transaction date earlier than complete-through date is accepted when a separate source-complete-through date is present.
- A scope note and confirmation source survive into `run.json`, `quality.json` and exported analysis manifest.
- Changing coverage settings invalidates old result caches/results.
- The overview and chart card consistently distinguish “no observed repeat in a complete window” from “outcome unavailable because observation is incomplete”.

---

## 8. P1: Consolidate the dashboard architecture

### 8.1 Current issue

A dedicated `src/retail_analytics/dashboard/views/` package has been added, with modules such as `overview.py`, `lifecycle.py`, `value_risk.py`, `products.py`, `customer_explorer.py`, `methods.py`, `modeling.py` and other view modules. This is a good separation direction.

However, `app.py` remains a very large module and still has inline implementations for many views. Some names are imported from `dashboard.views` and then redefined in `app.py` later in the module. In Python, the later function definition replaces that imported name in the module namespace, so edits to the imported implementation may not affect the route selected by the app.

The main file also defines a ChartCard/chart rendering layer while `visualization/chart_components.py` now defines the reusable system. That creates two paths for styling, evidence and errors.

### 8.2 Target structure

Keep `app.py` responsible for application orchestration only:

- page configuration;
- upload/configuration flow;
- input validation and job submission;
- analysis selection/state;
- top-level navigation and view dispatch;
- shared context construction;
- global exception handling.

Move view logic into the relevant module. The view modules should receive an explicit context object or typed arguments, not import back into `app.py` for helper functions such as `collect`, `table`, `read_json` or `apply_filters`.

Suggested structure:

```text
src/retail_analytics/
  dashboard/
    app.py                     # entry point + orchestrator
    dashboard_backend.py       # isolated worker and pipeline adapter
    context.py                 # selected cutoff, filters, coverage, manifest
    ui_utils.py                # table/export/formatting shared helpers
    views/
      overview.py
      lifecycle.py
      segments.py
      value_risk.py
      products.py
      customer_explorer.py
      transitions.py
      features.py
      methods.py
      modeling.py
      advanced.py
  visualization/
    chart_components.py        # only reusable chart card/evidence system
    chart_specs.py             # per-chart metadata and interpretation
    visual_insights_core.py    # pure analytics functions
    product_affinity.py        # product-affinity calculations
  modeling/
    models.py
    evaluation.py
```

This is a target, not a mandate to create every file immediately. The key is one owner for each responsibility.

### 8.3 Refactor sequence

1. Create a route registry mapping each visible `view_id` to a renderer and user-facing title.
2. Add a test asserting every nav view has exactly one registered renderer.
3. Move reusable helpers out of `app.py` into `ui_utils.py` or an appropriate shared layer.
4. Ensure view modules do not import `app.py` to avoid circular dependencies.
5. Migrate one section at a time: Data Trust/Overview; Lifecycle; Value/Risk; Products; Customer Explorer; Methods; Modeling; Advanced.
6. For each section, compare old/new calculations on the same synthetic fixture and preserve approved output semantics.
7. Remove the old inline branch only after tests verify the new path.
8. Remove duplicate `ChartCard` and chart wrappers; keep one rendering system.
9. Run format/lint/tests after each migration group rather than waiting until the entire file is rewritten.

### 8.4 Acceptance criteria

- `app.py` contains orchestration and routing, not hundreds/thousands of lines of view rendering logic.
- Every registered route points to one implementation and is covered by a smoke test.
- No view module imports app implementation helpers back from `app.py`.
- There is one source of truth for evidence-state wording, colors, uncertainty display and alt-text behavior.
- A bug fix in a view module is demonstrably used by the deployed route.

---

## 9. P1: Make the chart-evidence system consistent

**Core files:** `visualization/chart_components.py`, `visualization/chart_specs.py`, view modules and chart tests.

### 9.1 Standard chart contract

Every major chart should carry, where applicable:

1. **Question:** the customer/business question answered.
2. **Takeaway:** a factual statement derived from the displayed aggregate, not a causal claim or unsupported narrative.
3. **Metric and unit:** such as recorded spend in source currency units, shopping days, repeat probability or purchase count.
4. **Population:** who is included in the chart, and the total filtered population.
5. **Numerator/denominator:** the exact count/eligibility base for rates.
6. **Time definition:** cutoff date, lookback window, horizon and whether windows are inclusive.
7. **Sample policy:** full population, deterministic sample, top-N display, or aggregation cap.
8. **Uncertainty:** interval type and level when statistically appropriate; otherwise a clear reason why it is not provided.
9. **Evidence state:** sufficient, limited, insufficient, unavailable or invalid—using rules appropriate to the chart.
10. **How to read:** axes, reference lines, bins, colors and meaningful visual encodings.
11. **Caveat:** the main interpretation boundary, especially for sparse histories and descriptive associations.
12. **Next step:** a relevant drill-down, not a generic recommendation.
13. **Accessible alternative:** specific alt description plus visible text/table of the main result.

A `ChartSpec` should ideally be a data object, not just presentation text. Define typed fields and validate required metadata for chart types. Avoid a proliferation of optional free-form strings that can make empty cards appear complete.

### 9.2 Evidence states must not imply probability/confidence unless calibrated

Use wording such as:

- **Sufficient sample for this descriptive comparison** — threshold met, not a guarantee that the finding is true.
- **Limited evidence** — estimate is available but denominator or history is limited.
- **Insufficient data** — suppress the estimate or show a clear unavailable state.
- **Outcome unavailable** — observation coverage or horizon eligibility not met.
- **Exploratory association** — the relationship is descriptive and not an intervention effect.
- **Model not validated** — output must not be presented as reliable prediction.

Avoid green “success” treatments for a generic “next step” or for a weak result. Visual styling should distinguish normal explanation from meaningful risk or validation status. Avoid confusing “evidence level” with confidence in a specific prediction.

### 9.3 Accessibility and visual excellence

- Add specific `alt` descriptions for every major Plotly chart and verify the installed Streamlit version supports the chosen API.
- Show an adjacent or expandable data table for the values that support the chart; do not rely exclusively on hover.
- Avoid using color as the only encoding. Combine color with position, shape, labels or line style where useful.
- Keep color meaning stable across views: the same segment label must not change color between charts within one run.
- Include readable axis labels and units, concise titles, sensible ordering, zero-baseline choices where appropriate, and explicit treatment of missing values.
- Avoid 3D charts and decorative visual complexity if they make comparisons harder.
- For dense customer scatter plots, consider density/hexbin summaries or segment-aware deterministic sampling, while making the sampling policy explicit.
- Keep plot samples and display limits in chart metadata, not only free text.

### 9.4 Statistical display rules

- Show `n` and denominator on all key rate estimates.
- Use Wilson intervals for binomial proportions where the assumptions are reasonable; use appropriate bootstrap/survival intervals only when their implementation and sampling assumptions are documented.
- Never show a 0% rate when data is missing, ineligible or not observed through the full horizon.
- Do not treat within-segment IQR as a confidence interval.
- Warn when users compare many groups; exploratory comparisons are not multiplicity-adjusted unless a correction is actually used.
- Show exact pair/basket support for association rules alongside lift/confidence; large lift based on tiny support should not be highlighted as a robust recommendation.
- Keep confidence intervals and thresholds derived independently from the labels being validated where feasible. A rule evaluated on the same feature it was defined with is not independent validation.

### 9.5 Implementation tasks

1. Audit all uses of `st.plotly_chart` and legacy `chart()`; classify each as core, advanced or internal.
2. Replace duplicated rendering wrappers with `ChartCard`/`ChartSpec` or one simple, explicitly justified alternative for tiny utility plots.
3. Build a `validate_chart_spec()` helper for required metadata by chart family.
4. Add tests for missing denominator, missing alt text, low support, unconfirmed coverage, top-N capping and sampled data disclosure.
5. Standardize chart title/axis/unit formatting through the metric registry.
6. Add accessible text summaries for charts with a single main takeaway or key rows.
7. Test in a browser/Streamlit environment; unit tests on the component class are not sufficient to validate screen-reader, layout or responsive behavior.

---

## 10. P1: Design for sparse transaction histories

Limited data is a core product constraint, not an edge case. The app should degrade gracefully rather than quietly changing what a metric means.

### 10.1 Separate the main sources of uncertainty

The UI should distinguish at least:

- **Sparse event history:** a customer has too few shopping occasions to estimate stable cadence.
- **Short relationship observation:** first observed transaction is recent; lifecycle estimates have short exposure.
- **Right censoring/incomplete follow-up:** the outcome window extends beyond complete source coverage.
- **Missing/invalid field:** a feature cannot be computed due to missing source data.
- **Zero activity:** a complete window was observed and no qualifying purchase occurred.
- **Sampling/aggregation:** the chart displays a sample or truncated set, even though aggregate calculations use broader data.
- **Model uncertainty:** estimated parameters/predictions are uncertain or the model fit did not converge.

These states should be represented separately in the data model, UI and exported metadata.

### 10.2 Minimum support policy

The repository now has a `SupportThresholds` utility. Convert it into a central policy registry with explicit per-analysis thresholds and rationale:

| Analysis | Minimum evidence to consider | Additional safeguards |
|---|---|---|
| Segment profile counts/spend | Descriptive profile can show any positive population, but mark tiny groups | Show `n`, median/IQR and contribution denominator; avoid precise ranking claims on tiny segments |
| Binomial repeat rate | Minimum eligible denominator, currently represented by a default of 30 in the helper | Wilson interval; expose events/eligible `n`; hide or label low-support intervals |
| Cohort repeat curves | Minimum cohort size and enough mature follow-up at each horizon | Do not count immature cohorts as failures; keep not-yet-observed cells blank |
| Cadence classification | Enough distinct shopping dates and usable gaps to estimate a stable median gap | Do not estimate personal cadence from one purchase day; show “history insufficient” |
| Product pair association | Minimum pair co-occurrence count plus support/confidence/lift floors | Display pair count and total baskets; test rare-item artifacts and near-duplicate products |
| Trial-to-repeat conversion | Minimum eligible trials with complete follow-up | Denominator must include all eligible trials; hide or flag immature trials |
| CLV/purchase models | Minimum fit-eligible population and identifiable variation; model-specific checks | Require convergence, diagnostics, temporal holdout, baseline comparison and explicit model limitations |

These are starting safeguards, not universal scientific constants. Validate thresholds with sensitivity analysis and document the cost of false claims vs suppressing a real pattern.

### 10.3 Confidence intervals and sensitivity

- Use Wilson intervals for straightforward binomial proportions when the customer-level independence assumption is acceptable.
- For clustered customers/stores or repeated periods, consider an interval method consistent with the sampling unit; do not automatically apply independent-customer intervals to every comparison.
- Bootstrap only with explicit resampling unit, deterministic random seed, minimum effective sample and failure behavior.
- Show how segment sizes, repeat rates and rule-defined labels change under reasonable threshold choices.
- Where changing a threshold changes a label, make that dependency explicit. If a label is defined by purchase-gap CV, do not claim that a gap-CV difference independently proves the label is meaningful.

### 10.4 Customer-level metrics need qualification

- `recency`: days since last observed purchase as of cutoff.
- `median_gap`: typical gap between distinct observed shopping dates; undefined/unstable without sufficient gaps.
- `lateness = recency / median_gap`: meaningful only if median gap is finite and positive and the customer has enough cadence evidence.
- `spend_90d`: recorded amount in the window; not profit and may include no refunds handling.
- `value_tier`: relative tier based on selected reference population, not a portable absolute value standard.
- `first_purchase`: first observed in this source/history, not necessarily real acquisition.
- `future_purchases_h`: outcome only when the entire horizon is observed and coverage is affirmed.
- `p_alive`: a model-based probability under model assumptions, not literal proof a customer is active/loyal.

### 10.5 Acceptance criteria

- Every sparse-data-sensitive view contains a tested small-n path.
- Empty, null, non-finite and insufficient-history cases do not crash and do not convert undefined values into zero.
- A single purchase cannot produce a confident cadence/CLV conclusion.
- Immature cohorts display unavailable cells instead of fabricated drop-off.
- Threshold adjustment changes are reproducible and disclosed.
- Charts remain readable when only one category/segment exists or when all points have identical values.

---

## 11. P1: Improve customer understanding in the UI

### 11.1 Organize the main navigation around user questions

Keep the new two-level navigation, but reduce the complexity of the primary path. A suggested structure:

1. **Data Trust & Overview** — can I trust this analysis; what is the observed customer base?
2. **Customer Lifecycle** — when do customers return; how do observed cohorts behave?
3. **Customer Value & Change** — what spending is recorded; which behaviors changed?
4. **Product & Category Behavior** — what products/categories are explored or repeated?
5. **Customer Explorer** — tell me about one customer’s observed history.
6. **Methods & Definitions** — explain the metrics, thresholds and assumptions.
7. **Predictive Modeling (Experimental)** — model fitting/evaluation only when real evaluation is implemented and valid.
8. **Advanced Analytics** — specialized charts, transitions and detailed exports.

The main labels should describe a customer question. Statistical names such as Kaplan–Meier, RFM and association lift can appear as explanatory sub-labels.

### 11.2 Overview page: useful first minute

The overview should answer “What is in this data, what should I trust, and what changed?”

Recommended top summary:

- Rows received / validated / rejected.
- Observed customer count and distinct purchase occasions.
- Observed date range and coverage-confirmed-through date (two separate fields).
- Most recent analysis cutoff and selected lookback window.
- Share of customers with enough history for cadence, repeat and model analyses.
- Missing/invalid feature count or a compact quality status.
- Recorded spend/frequency trends with a clear note when rolling windows overlap.

Follow with a limited number of high-value plots rather than every chart. Suggested top charts: activity/spend trend, customer spend distribution/concentration, repeat timing/cohort summary if coverage is valid, and one “not enough evidence” banner where appropriate.

### 11.3 “Key signals” need transparent generation

A key-signal card should be generated from an explicit rule (e.g., largest observed period-over-period change among sufficiently supported comparisons), not an LLM-like narrative guess. It should include:

- what changed;
- amount or percentage change;
- before/after window;
- affected population and denominator;
- uncertainty/support status;
- whether the change is descriptive or causally identified;
- a link to the source chart.

If no signal passes the threshold, say “No sufficiently supported signal found in the selected view” rather than creating a weaker claim to fill space.

### 11.4 Customer brief

The customer brief should translate features into a compact, inspectable narrative:

- Observed status: first observed/one shopping day/repeat history, with evidence level.
- Most recent observed shopping date and recency.
- Distinct shopping occasions and recorded spend in a selected window.
- Typical basket value and variability, only when enough baskets exist.
- Typical interval between shopping dates, only when enough gaps exist.
- Top observed categories and category concentration.
- Recent-vs-prior changes using comparable windows.
- Data limitations and explicit “we cannot infer” statements.
- Click-to-definition for each metric and click-to-timeline for supporting events.

Do not state “at risk of churn” as a fact solely because lateness exceeds a rule threshold. Use “purchase cadence is overdue relative to observed history” and show the rule.

### 11.5 Customer timeline

The timeline should:

- distinguish purchase occasions from transaction lines;
- show daily/basket recorded value and optionally cumulative recorded spend;
- highlight long intervals with no recorded purchase without claiming no purchase occurred outside the dataset;
- show category/product details only on demand;
- show how the date range and cutoff affect inclusion;
- mask customer IDs by default and use explicit reveal/consent for identifiable data;
- handle one purchase, many same-day lines, duplicate rows and multiple baskets correctly.

### 11.6 Filters and comparison semantics

- Display active filters in a compact summary.
- Make it clear whether a comparison uses a fixed matched population or a changing population at every cutoff.
- Default period comparisons to matched customers when the question is “what changed for the same customers?”; allow a separate “whole population at each period” mode.
- Ensure segment-transition tables explicitly reconcile matched, unmatched-origin and unmatched-destination customers. Unmatched does not automatically mean churn/acquisition.
- Keep export labels and manifest population filters in sync with chart titles.

---

## 12. Chart portfolio and per-view acceptance criteria

The app already has many chart types. Implement a systematic chart audit and improve the most useful views first. The list below gives each chart a job and a minimum evidence contract.

### 12.1 Data Trust & Quality

**Question:** Can the uploaded records support the analyses being requested?

**Recommended visuals:** Quality funnel (source/valid/rejected), rejection reasons, date coverage vs complete-through date, feature missingness, product/category identity variation summary.

**Must show:** source row count; valid/rejected counts and rates; whether flags overlap; date range vs capture end; missing values vs true zero; duplicate policy; price mode and currency assumptions.

**Acceptance:** Synthetic files with overlapping rejection issues demonstrate that stacked counts do not imply a false accounting identity. Coverage confirmation cannot be inferred from latest row date.

### 12.2 Customer Base Summary

**Question:** What does the observed customer base look like?

**Recommended visuals:** activity trend, count distributions, median/IQR summary, Lorenz curve or contribution curve, compact segment composition.

**Must show:** active population definition, transaction/basket grain, lookback windows, currency units, population filters, and whether spend includes returns (currently purchase-only inputs).

**Acceptance:** Customer counts are distinct at the declared grain and do not get added across overlapping windows or months without a warning.

### 12.3 Cohort-age retention heatmap

**Question:** How does observed repeat activity change with time since first observed purchase?

**Recommended visuals:** cohort × age heatmap; cell tooltip with active customers, cohort size, eligible denominator and interval if available.

**Must show:** age 0 definition; right-censoring / incomplete exposure; blank means not yet observed or insufficient data, not zero; first observed cohort is not necessarily acquisition cohort.

**Acceptance:** An immature cohort never shows false zeros in unobserved future months. Cohort denominator and window tests include boundary cases.

### 12.4 Time to second purchase (Kaplan–Meier)

**Question:** How long do customers take to make a second observed shopping occasion?

**Recommended visuals:** survival curve plus confidence band, at-risk counts beneath/alongside the curve, a summary of observed events and censoring.

**Must show:** start/end event definition, censoring date, coverage status, distinct shopping dates vs baskets and eligible population.

**Acceptance:** Customers with no second purchase by cutoff are right-censored, not labeled permanently lost. Test one event, no events, equal dates and empty data.

### 12.5 Purchase cadence / overdue analysis

**Question:** Is the customer returning later than their established observed rhythm?

**Recommended visuals:** recency vs median-gap scatter with reference lines; optional density/hexbin summary; distribution of lateness for eligible customers.

**Must show:** `median_gap` eligibility, the exact threshold (e.g., 2× median gap), population excluded for insufficient history, point sample/full population, and the descriptive nature of the label.

**Acceptance:** Customers with one shopping occasion or null median gap are excluded from the cadence classification and counted in an “insufficient evidence” explanation.

### 12.6 Cadence validation against repeat outcomes

**Question:** Does the chosen overdue threshold correspond to different *observed* future return rates in this dataset?

**Recommended visuals:** repeat rate by lateness band and horizon, with intervals, eligible `n` and observed events per band.

**Must show:** coverage complete through each horizon; bin edge rules; number of bins; event numerator; eligible denominator; comparison multiplicity caveat.

**Acceptance:** View is unavailable when follow-up is incomplete. Thresholds can be sensitivity-tested and are not called causal or universally valid.

### 12.7 RFM response surface

**Question:** How does the observed future purchase outcome differ across recency/frequency strata?

**Recommended visuals:** recency × frequency heatmap; optional value-tier facets only when cell sizes support them.

**Must show:** bin boundaries, `n` per cell, actual events/eligible denominator, interval, horizon, exact cutoff and coverage gate. Avoid implying monotonicity if bins are sparse/noisy.

**Acceptance:** Empty cells remain blank; low-n cells get an evidence state; category/value facets do not silently change denominators.

### 12.8 Customer momentum

**Question:** Which customers changed frequency or basket behavior across comparable periods?

**Recommended visuals:** frequency change × basket change scatter/quadrants, with quadrant counts and median movement; optional density for very large populations.

**Must show:** definitions of frequency and basket value; period boundaries; comparable-history eligibility; sample count and sample policy; axis units/normalization; no causal inference.

**Acceptance:** A zero-baseline case and constant-value population render without misleading quadrants or errors. Segment labels are stable across the plot.

### 12.9 Value-at-risk / high recorded spend with overdue cadence

**Question:** Which customers combine historically high observed spending with unusual purchase gaps?

**Recommended visuals:** recency/lateness vs spend scatter, bounded by a threshold with quadrants, plus a ranked customer list available only with appropriate consent.

**Must show:** “recorded spend at risk” rather than profit at risk; eligible customers and spend denominator; overdue-rule threshold; sample disclosure; exclusions and uncertainty.

**Acceptance:** Never equate all past spend of an overdue customer with likely lost revenue. Use neutral descriptive naming unless a separately validated outcome model justifies a risk estimate.

### 12.10 Segment migration and alluvial flows

**Question:** How did matched customers move between descriptive labels across snapshots?

**Recommended visuals:** transition matrix as a reliable fallback; alluvial/Sankey when label counts remain readable.

**Must show:** origin and destination cutoff, matched customer count, unmatched-origin and unmatched-destination count, filter semantics and label-rule version.

**Acceptance:** Flows reconcile to the matched population. Unmatched-origin is not called churn; destination-only is not called acquisition without acquisition data.

### 12.11 Spend concentration / Lorenz curve

**Question:** How concentrated is recorded spend among observed customers?

**Recommended visuals:** Lorenz curve against equality line; optional top-1%, top-5%, top-10% shares and Gini only if implemented/tested.

**Must show:** all selected customers including zero-spend/inactive ones; period; customer count; top-fraction rounding/tie policy; spend meaning and currency units.

**Acceptance:** Synthetic examples for equal spending, one dominant customer and zero spend match hand calculations. Undefined shares remain null where the denominator is zero.

### 12.12 Revenue bridge/waterfall

**Question:** What components account for the change in aggregate recorded spend between periods?

**Recommended visuals:** matched-customer bridge separating customer-level changes from entrants/exits only where the population definition permits it; clearly reconcile to total change.

**Must show:** matched-population mode vs all-period populations, period boundaries, contribution definitions and bridge residual.

**Acceptance:** Bridge components sum to the defined net change within tolerance. One-sided periods, no-spend periods and changed populations are tested and labeled.

### 12.13 Category participation and category opportunity

**Question:** Which categories are purchased by many customers, and which have higher or changing intensity?

**Recommended visuals:** participation vs spend/intensity portfolio; recent vs prior category participation; category mix/concentration.

**Must show:** participation denominator; recurrence definition; overlapping category memberships; spend share vs customer participation as separate metrics; minimum customer/category support.

**Acceptance:** Category participation can sum to more than 100% when categories overlap and states this clearly. Unknown/missing categories remain visible as data-quality context rather than disappearing without notice.

### 12.14 Product affinity / association network

**Question:** Which products co-occur within the defined basket?

**Recommended visuals:** ranked association table first, network as an optional exploratory view with bounded nodes/edges.

**Must show:** basket definition, number of baskets, pair count, support, confidence in each direction where applicable, lift and filtering thresholds.

**Acceptance:** Minimum pair count is enforced; rare products do not dominate the network purely by inflated lift; tests cover duplicate lines and multiple transactions per customer/day.

### 12.15 Trial-to-repeat / product repeat

**Question:** After first observed purchase of a product, how often is it purchased again within the allowed follow-up window?

**Recommended visuals:** product repeat-rate dot plot with intervals; eligible trial count; repeat count; optionally a ranked table.

**Must show:** trial-date eligibility, repeat window, censoring/coverage, product identity variation and minimum eligible trial count.

**Acceptance:** Trials without complete follow-up are censored or excluded from the rate denominator according to the defined estimand; never count them as non-repeat simply because the file ends.

### 12.16 Basket evolution and purchase rhythm

**Question:** Is customer purchasing becoming broader, narrower, more regular or more variable?

**Recommended visuals:** paired recent/prior summaries and distributions, not only mean slope lines; show median and spread.

**Must show:** matched customers, exact windows, changes in eligible sample size, and whether breadth measures categories or distinct products.

**Acceptance:** Single-basket customers do not receive misleading variability measures. Sample-composition changes are separated from within-customer changes.

### 12.17 Feature distribution and missingness

**Question:** Which features vary in this population, and where is evidence absent?

**Recommended visuals:** median/IQR/finite count, histogram/box plot or robust distribution view, scatter/density for selected relationships.

**Must show:** full population vs bounded sample, finite and unavailable counts, variable units, extreme-value behavior and whether the visualization uses a sample.

**Acceptance:** Deterministic sample is disclosed; identical values and outliers do not break axes or charts; data are not randomly sampled in a way that cannot be reproduced.

### 12.18 Predictive model view

**Question:** Do model predictions outperform simple baselines on a genuinely later observed interval?

**Recommended visuals:** observed vs predicted by horizon, calibration/reliability plot for probabilities, residual distributions, error by segment/history length, baseline comparison.

**Must show:** split dates, target definition, horizon, eligible `n`, actual vs predicted, baseline, model fit/convergence state, coverage and warnings.

**Acceptance:** View remains disabled/experimental until evaluation logic is correct. No placeholder metrics; no “validated” state without passing model-specific tests and baseline comparisons.

---

## 13. P1/P2: Integrate security and privacy utilities

**File:** `src/retail_analytics/security.py`

### 13.1 Integration audit

The repository now contains utility functions for upload checks, workspace cleanup, CSV escaping, filename sanitization, ID masking, PII scans, access-control checks, audit logging, CSP headers and deployment checks. The main dashboard file does not visibly import and consistently apply the module across the upload/session workflow in the reviewed source. Treat these utilities as **not fully integrated until verified**, rather than assuming their presence alone secures the app.

### 13.2 Required changes

1. Use the same upload-size/extension validation logic in the actual upload path as in the tested utilities. Validate file content with a real parser; a magic-header check alone is not a hostile-file sandbox.
2. Keep all uploaded files and derived outputs isolated by session/workspace with restrictive permissions where supported.
3. On upload change, analysis failure, explicit clear and session end, remove the applicable workspace/results. Document that deletion is best-effort and not guaranteed secure erasure on hosted storage.
4. Mask customer identifiers in default tables and charts; only reveal full IDs through explicit action and access policy. Avoid placing raw IDs in logs, chart metadata or exception messages.
5. Apply CSV-formula-injection escaping in every export path, not just one helper.
6. Sanitize download filenames and verify that exports contain only fields the user is permitted to access.
7. Do not treat an in-process `AccessControl` object as authentication. Configure platform-level authentication/authorization for any shared/public deployment.
8. Add operational limits for upload bytes, rows, memory/time, decompression expansion and concurrency; ensure users see actionable errors when limits are reached.
9. Validate deployment settings at startup and make insecure configurations fail or display a blocking warning.
10. Keep telemetry free of customer rows/IDs; audit events should record action/time/status without copying sensitive values.

### 13.3 Acceptance criteria

- Integration tests exercise the real dashboard upload code path and confirm invalid/corrupt/oversized inputs are rejected.
- Tests prove data from a prior upload cannot appear after a new upload or session reset.
- Customer IDs are masked by default and reveal is explicitly gated.
- Exports escape formula-like text and use safe filenames.
- Deployment documentation clearly covers authentication, storage lifetime, transport encryption, server trust and the limits of best-effort cleanup.

---

## 14. P2: Predictive-model roadmap

Do not lead with prediction until descriptive analysis, coverage, tests and UI semantics are reliable.

### 14.1 Phase A — validated descriptive methods

Ship reliable RFM summaries, cadence analysis, age-aligned cohorts, concentration, product/category aggregates and explainable rule labels. Benchmark rule thresholds with sensitivity analysis. These methods offer high interpretability and need fewer observations than complex predictive models.

### 14.2 Phase B — repeat-purchase probability and transaction count

After the temporal evaluation framework works:

- Compare historical-rate and frequency-recency baselines.
- Fit BG/NBD and/or Pareto/NBD only for data distributions and customer histories where the assumptions are appropriate.
- Validate future transaction count by horizon, segment and history length.
- Test sensitivity to short observation periods and event definition.
- Store the model version, training cutoff and fit status.

A `p_alive` output is model-dependent and should not be shown as a universal probability of loyalty. Clearly state the model’s non-contractual purchasing assumptions.

### 14.3 Phase C — monetary prediction

Gamma-Gamma style monetary models should only be considered when positive monetary values and assumptions are defensible. If the source is strictly purchase-only and amounts reflect recorded revenue, label output accordingly. Without costs and margins, do not call it profit CLV.

### 14.4 Phase D — calibrated customer decision support

Only after the predictive estimates are validated should the product present ranked lists or action-oriented “who to contact” recommendations. A model predicting future purchase is not the same as an uplift model estimating who changes behavior because of an intervention.

If later adding marketing response, incrementality or uplift:

- collect exposure/treatment and timing where possible;
- use randomized experiments or a defensible causal design;
- define outcome and estimand before training;
- evaluate calibration, overlap/positivity and treatment assignment issues;
- report incremental outcomes and uncertainty, not only propensity or AUC.

### 14.5 Model shipping gate

A model is marked **Validated** only when all are true:

- data/coverage eligible;
- fit converged and parameters finite/in-range;
- temporal validation and untouched test are complete;
- it beats or provides justified value over a baseline on the target metric;
- error/calibration is reported for appropriate horizons;
- performance and uncertainty are documented for relevant customer-history groups;
- edge cases and model output schema are tested;
- UI labels the target as spend, revenue or profit accurately;
- a maintainer has reviewed the assumptions and acceptance report.

Otherwise label it **Experimental**, **Insufficient data**, **Fit failed** or **Not evaluated**.

---

## 15. Data contract and documentation alignment

### 15.1 Coverage schema

Align UI, worker job, `run.json`, `quality.json`, output manifests and docs around one canonical set of fields. Do not store one field as a synonym for another date concept.

### 15.2 Metric registry

Create/maintain one central definition registry used by:

- the feature pipeline and analytical functions;
- tooltips/metric-definition view;
- chart titles/axis labels;
- export column descriptions;
- tests and generated documentation.

Every metric entry should define name, plain-English definition, formula, grain, time window, numerator/denominator, valid-data requirement, null semantics, interpretation caveat and version introduced.

### 15.3 Decision log

Review `DECISIONS.md` items against the current branch. For each item:

- mark `done`, `deferred` or `not done`;
- link the implementation commit and regression test;
- keep open items visible in the release checklist;
- remove contradictions only after code and tests verify the intended behavior.

In particular, keep duplicate handling, timezone conversion, quantile interpolation metadata and reporting exception behavior consistent between docs and code.

### 15.4 Documentation status ledger

Update `docs/customer_analytics_improvement_plan.md` to include a current-state table:

| Workstream | Status | Evidence | Remaining verification |
|---|---|---|---|
| Coverage default-off UI | Implemented in source | Checkbox default false | UI test + per-horizon boundary test |
| Coverage date semantics | Partial | UI and worker fields exist | Permit legitimate coverage end beyond latest transaction date; persist scope consistently |
| Chart evidence framework | Partial | Shared component/spec files | Adoption audit across every core/advanced route |
| Customer explorer | Implemented in source | Separate brief/timeline views | Usability and privacy-flow tests |
| Model fitting | Implemented in source | BG/NBD/Pareto-NBD/Gamma-Gamma functions | Numerical correctness, diagnostics and model assumptions |
| Temporal evaluation | Not ready | Placeholder values in evaluation function | Real future actuals and no fabricated metrics |
| CI | Failing as of review | Latest Actions run says failure | Diagnose and get both supported-Python jobs green |
| Security integration | Unverified/partial | Utility module exists | Show real app path applies it |
| UI runtime | Unverified | Source-level review only | Test deployed app and representative datasets |

### 15.5 README usability

The README should tell a first-time user:

1. how to install and start the dashboard;
2. the exact expected schema and price meaning;
3. how to confirm or not confirm capture coverage;
4. what analysis is available without complete follow-up;
5. what a behavioral rule label does and does not mean;
6. where tests and runtime verification instructions live;
7. whether predictive models are experimental or validated.

Remove examples implying commercial findings from sample outputs unless the provenance is explicit.

---

## 16. Detailed delivery phases

The sequence below is dependency-driven. Treat “days” only as relative effort sizing, not a commitment to a calendar release.

### Phase 0 — Stabilize the branch (blockers first)

**Scope:** CI, smoke-test collection, model evaluation gating.

Tasks:

- [ ] Fix `Tuple` annotation/import and standardize test imports.
- [ ] Run collection and repair all lint/format/test failures.
- [ ] Remove unconditional passing assertions.
- [ ] Hide/disable validation-shaped model metrics until evaluation is genuine.
- [ ] Add an `evaluation_status` field and render unimplemented/invalid states honestly.
- [ ] Capture and review actual GitHub Actions failure logs.
- [ ] Commit the smallest possible changes, run CI, and confirm green status before feature changes resume.

**Exit gate:** both Python CI jobs green, no fabricated evaluation metrics visible, tests meaningful enough to detect the known defect.

### Phase 1 — Finish coverage semantics

**Scope:** coverage UI, data contract, job schema, future outcome gates.

Tasks:

- [ ] Define `latest_observed_transaction_date`, `coverage_complete_through` and `analysis_cutoff` separately.
- [ ] Allow a complete-through date later than latest transaction date when supported by a user/source assertion and a documented upper bound.
- [ ] Preserve scope note, confirmation source, source-complete-through date and policy version through worker/output manifests.
- [ ] Apply horizon-level eligibility to every future-outcome function and chart.
- [ ] Add cutoff/horizon boundary tests.
- [ ] Add a test confirming that unchecked coverage leaves historical features usable but disables future outcome claims.
- [ ] Update UI messages and chart specifications.

**Exit gate:** end-to-end test proves every future metric is either eligible under the declared coverage contract or has an explicit unavailable reason.

### Phase 2 — Build valid model evaluation

**Scope:** temporal split, model predictions, metrics, baseline comparisons, reporting.

Tasks:

- [ ] Define target labels/horizons and split boundary contracts.
- [ ] Implement training-only baselines and actual test outcomes.
- [ ] Implement actual predictions for supported model family, one at a time.
- [ ] Replace placeholder metrics/concordance/by-horizon values.
- [ ] Handle MAPE/zero-target cases appropriately; prioritize MAE/RMSE/Brier/log loss according to target.
- [ ] Add convergence and parameter diagnostics.
- [ ] Store evaluation metadata and warnings.
- [ ] Add toy fixtures with hand-computed expected metrics.
- [ ] Keep unsupported model families marked not evaluated instead of returning mock results.

**Exit gate:** model view demonstrates real held-out metrics, beats or is honestly compared with baseline, and never uses future outcomes as features.

### Phase 3 — Finish dashboard consolidation

**Scope:** route registry, view modules, shared helpers, removal of duplicate views.

Tasks:

- [ ] Enumerate all views in a route registry and assert one renderer each.
- [ ] Identify duplicate names imported from view modules then defined in `app.py`.
- [ ] Select the canonical module for each view.
- [ ] Move common helpers to a dependency-safe module.
- [ ] Migrate one navigation section at a time.
- [ ] Remove legacy branches only after output parity tests pass.
- [ ] Remove the embedded chart/evidence implementation from `app.py` once all callers use the shared layer.
- [ ] Add route-level smoke tests for all primary views and selected advanced views.

**Exit gate:** no accidental name shadowing, no view-module circular import of `app.py`, all routes resolve correctly.

### Phase 4 — Standardize chart evidence and sparse-data behavior

**Scope:** ChartSpec validation, evidence thresholds, alt text, uncertainty, display rules.

Tasks:

- [ ] Create a chart audit inventory mapping every view to a spec and evidence fields.
- [ ] Implement spec validation by chart family.
- [ ] Add explicit sample-policy data to sampled/limited plots.
- [ ] Apply support thresholds uniformly to repeat rates, cohorts, product associations and model outcomes.
- [ ] Standardize “insufficient”, “limited”, “unavailable” and “not validated” states.
- [ ] Test interval computation and deterministic resampling.
- [ ] Add accessible visible summaries and specific alt descriptions.
- [ ] Apply consistent segment colors, axes, units and reference-line explanations.
- [ ] Ensure chart cards are not visually cluttered; show key context by default and detailed methodology on demand.

**Exit gate:** every priority chart has a valid spec and tests for normal, low-support, empty and coverage-disabled scenarios.

### Phase 5 — Improve user journey and customer explorer

**Scope:** navigation, overview, key signals, customer brief, timeline, filter semantics.

Tasks:

- [ ] Reduce the primary route list to user questions and move specialized plots under Advanced Analytics.
- [ ] Validate the Overview first-minute experience with users unfamiliar with RFM/Kaplan–Meier.
- [ ] Add context-specific definitions and drill-down links.
- [ ] Ensure key signals are rule-generated, auditable and never forced when evidence is weak.
- [ ] Add matched-population vs changing-population mode for comparisons.
- [ ] Validate timeline grain, same-day basket handling and masked IDs.
- [ ] Check mobile/narrow-window layout and keyboard accessibility in target environment.

**Exit gate:** user can answer the core customer questions without learning implementation jargon; chart interpretation matches displayed data.

### Phase 6 — Integrate security/privacy and operational safeguards

**Scope:** actual app path, session isolation, export safety, deployment requirements.

Tasks:

- [ ] Wire validation and sanitization into upload path.
- [ ] Test data isolation across upload changes, session resets and failures.
- [ ] Verify masking defaults and permissions for identifier reveal.
- [ ] Audit all export paths for formula injection, safe field selection and filename handling.
- [ ] Confirm authentication/access controls at deployment layer.
- [ ] Configure limits, logging, data retention and safe error messages.
- [ ] Add a short deployment hardening checklist.

**Exit gate:** real-app path passes security-focused tests and documentation accurately describes hosted-data handling.

### Phase 7 — Runtime and release verification

**Scope:** installed environment, deployment smoke, representative uploads and operational playbook.

Tasks:

- [ ] Install from clean environment using documented commands and pinned/resolved dependencies.
- [ ] Run unit/integration tests.
- [ ] Start Streamlit and test the complete upload-preview-map-validate-analyze-view-export workflow.
- [ ] Test representative sparse/dense datasets, CSV and Parquet, both price modes, invalid dates, duplicate lines, unknown categories, zero spend, insufficient history and unconfirmed coverage.
- [ ] Test workload limits and graceful failure with larger files.
- [ ] Verify chart clipping/overflow/labels in the actual browser.
- [ ] Verify exports, session cleanup, masking and deployment security behavior.
- [ ] Write release notes separating verified capabilities from experimental ones.

**Exit gate:** reproducible test report and documented runtime verification attached to the release/commit.

---

## 17. Test strategy and acceptance matrix

### 17.1 Unit tests

Test pure calculation functions for metric definitions, windows, transitions, cohort exposure, concentration, product association, threshold evaluation and model metrics. Include hand-calculated examples, boundary conditions and failures.

### 17.2 Contract tests

Assert that output JSON/Parquet schemas include required fields; metric definition changes update docs/specs; null semantics are stable; every manifest carries cutoff, configuration/version, and coverage status; and unexpected schema mismatch fails visibly.

### 17.3 Integration tests

Run a small generated transaction file through the real pipeline/backend and assert that:

- output snapshot and manifest are written;
- coverage and cutoff values survive exactly;
- invalid rows are accounted for correctly;
- results match expected counts and selected metrics;
- new upload/configuration invalidates previous results;
- no future outcome field is treated as valid when the horizon is incomplete;
- model evaluation either produces measured values from future actuals or reports that it is unavailable.

### 17.4 UI smoke tests

Run the actual Streamlit app against fixtures in an environment that supports it. Check route selection, empty-data messages, evidence cards, accessible labels, warnings, chart layout, data tables, download buttons and customer-mask defaults. A test that merely imports view functions is not a UI smoke test.

### 17.5 Suggested fixture matrix

| Fixture | What it tests | Expected behavior |
|---|---|---|
| One customer, one purchase occasion | Sparse history | No invented cadence, no confident CLV, clear insufficient-history state |
| Many customers, all single-purchase | Rate denominator/retention | Correct repeat outcome with coverage; cadence unavailable |
| Repeated regular buyers | Cadence/KM baseline | Correct distinct shopping dates and time-to-repeat event handling |
| Highly skewed spend | Concentration | Lorenz/concentration match hand calculation; robust axis labels |
| Zero recorded spend | Undefined shares | Null proportions where denominator is zero; not NaN crash/false zero |
| Product appearing once | Association support | Do not show unsupported pair as robust recommendation |
| Unequal cohort maturity | Censoring/exposure | Immature future cells blank/unavailable, not 0% repeat |
| Complete-through after last purchase | Coverage boundary | Complete capture end distinct from latest event date |
| Complete-through one day short | Horizon gate | Future rate disabled for the horizon extending past the end |
| Exact duplicate line items | Duplicate policy | Preserve or remove only per configured explicit policy; audit count reconciles |
| Multiple basket IDs same day | Event grain | No accidental collapse/overcount beyond documented basket definition |
| Mixed/invalid dates and prices | Input validation | Clear rejected-row reasons and quality totals |
| Large but bounded dataset | Performance | Bounded render and clear resource-limit feedback |
| Optimizer non-convergence | Model evaluation | Fit failure status; no valid-looking placeholder metrics |
| Perfect predictions on toy data | Metric correctness | Expected zero MAE/RMSE and correct calibration only when values truly match |
| Poor predictions on toy data | Metric correctness | Error metrics reflect error; test fails if output is hardcoded |

### 17.6 Statistical property tests

Where applicable test invariants:

- rates remain in [0,1]; denominators equal eligible customers;
- cohort counts cannot exceed source population under the declared grain;
- rate intervals are ordered and contain the point estimate under supported calculations;
- concentration shares reconcile to total spend under the stated policy;
- transition flows reconcile to matched customers and unmatched categories;
- revenue bridge components reconcile to net change within numeric tolerance;
- no future target is used in historical feature calculations;
- a model’s reported metric is computed from the same observations and horizon stated in the report.

---

## 18. Release checklist and go/no-go rules

### Scientific correctness

- [ ] Coverage defaults to unconfirmed.
- [ ] Latest transaction date and complete-through date are separate concepts.
- [ ] Every future target window verifies full coverage.
- [ ] Undefined/missing/insufficient outcomes are not displayed as zero.
- [ ] Model evaluation contains no placeholders.
- [ ] Model metrics use observed future actuals and fair baselines.
- [ ] Revenue/spend is not labeled profit.
- [ ] Behavior rules are not described as causal effects, permanent churn or emotional loyalty.

### UI and chart quality

- [ ] All primary charts have valid ChartSpec/ChartCard metadata.
- [ ] Axis labels, units, windows and population are explicit.
- [ ] Rates show numerator/denominator and eligible `n`.
- [ ] Sampled plots disclose sample and population counts.
- [ ] Low support and invalid coverage have visible status.
- [ ] Alt descriptions and visible data summaries exist.
- [ ] Color is not the only information channel.
- [ ] There is one implementation per route.

### Engineering and security

- [ ] CI green on supported Python versions.
- [ ] No unintended test skips or unconditional passing assertions.
- [ ] Clean environment install works.
- [ ] Upload/session isolation tests pass.
- [ ] Exports are formula-injection-aware and identifiers are handled intentionally.
- [ ] Deployment auth/access controls are documented and tested.
- [ ] Resource limit failures produce actionable messages.

### Decision

**No-go** if CI fails, model metrics remain fabricated, incomplete follow-up can be reported as a valid outcome, or the target deployment has not been checked for exposing sensitive customer data. Other advanced charts can be released progressively once the core user journey and evidence contract pass.

---

## 19. Backlog with dependencies

| Backlog ID | Task | Priority | Dependency | Definition of done |
|---|---|---|---|---|
| CA-CI-01 | Diagnose current workflow failure | P0 | None | Exact failing step documented; latest commit has green checks |
| CA-TEST-01 | Fix `Tuple` annotation and imports | P0 | None | Test collection succeeds on Python 3.10 and 3.12 |
| CA-TEST-02 | Replace vacuous assertions | P0 | CA-TEST-01 | Tests fail when expected computation is broken |
| CA-MODEL-01 | Remove placeholder evaluation metrics | P0 | None | UI never reports uncomputed scores as results |
| CA-MODEL-02 | Implement true temporal outcomes/baselines | P0 | CA-COVER-01 | Test results compare predictions to future actuals |
| CA-COVER-01 | Canonicalize coverage dates and scope fields | P0 | None | UI/job/output schemas align; boundary tests pass |
| CA-COVER-02 | Gate every future horizon independently | P0 | CA-COVER-01 | No outcome chart appears when window is incomplete |
| CA-ARCH-01 | Create a single view route registry | P1 | CI green | All nav views map to one renderer |
| CA-ARCH-02 | Remove duplicate/shadowed view functions | P1 | CA-ARCH-01 | Core UI modules own their implementation |
| CA-CHART-01 | Unify chart component path | P1 | CA-ARCH-02 | One chart/evidence implementation |
| CA-CHART-02 | Enforce chart-spec required fields | P1 | CA-CHART-01 | Contract tests for all primary charts pass |
| CA-SPARSE-01 | Centralize evidence policy | P1 | CA-CHART-02 | Shared thresholds and states, with tested boundaries |
| CA-SPARSE-02 | Add fixture matrix for sparse histories | P1 | CA-SPARSE-01 | All listed edge cases tested |
| CA-UX-01 | Finish question-oriented navigation | P1 | CA-ARCH-01 | New user reaches core questions without technical jargon |
| CA-UX-02 | Validate customer brief/timeline | P1 | CA-CHART-01 | Timeline/profile tested against known customer fixture |
| CA-SEC-01 | Integrate upload validation/session lifecycle | P1 | None | Tests cover actual app path, not utility functions only |
| CA-SEC-02 | Verify deployment access controls | P1 | CA-SEC-01 | Hosted deployment requires trusted access for customer data |
| CA-DOC-01 | Update implementation plan status ledger | P1 | Current implementation verified | Done/partial/unverified states link to evidence |
| CA-OPS-01 | End-to-end runtime smoke | P1 | CI green | Clean install and deployment workflow tested |
| CA-MODEL-03 | Promote model only after baseline gate | P2 | CA-MODEL-02 | Model has test report and validated UI status |
| CA-UX-03 | User comprehension/usability review | P2 | Core routes stable | Observed tasks completed without misinterpretation |

---

## 20. Success measures

Use product/quality measures that reflect customer understanding—not just the number of algorithms or charts shipped.

### Data trust

- 100% of future outcome reports record coverage confirmation, complete-through date and horizon eligibility.
- 0 charts convert unavailable outcomes into zero outcomes.
- 100% of model-evaluation reports include split dates, horizon, eligible `n`, actual outcome summary and baseline comparison.

### Chart consistency

- 100% of primary charts pass `ChartSpec` validation.
- 100% of rate charts show event count and denominator.
- 100% of sampled charts disclose the sampling policy.
- 100% of primary charts provide a visible summary and an accessible description.

### Engineering health

- Main-branch CI is green on supported Python versions.
- No known unconditional-success assertions in required tests.
- No duplicate route implementations for active core views.
- A clean environment can install and run the documented application.

### User understanding

Test a small representative group of users performing tasks such as “who returns?”, “which categories are repeated?”, “which customer pattern changed?”, and “how trustworthy is this result?”. Record whether they can identify the timeframe, denominator, evidence status and causal limitation without coaching.

The quality bar should be: users can explain what a chart means and what it does not mean—not merely that they find it attractive.

---

## 21. Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Placeholder evaluation survives refactor | Perfect-looking metrics can mislead decisions | P0 gate; remove placeholder outputs and test with deliberately incorrect predictions |
| Coverage is confirmed but scope/date is ambiguous | Outcome labels may be biased by missing stores/channels or incomplete time | Canonical metadata; required complete-through policy; visible scope note; per-horizon eligibility |
| One-off customers are classified with noisy cadence | Sparse history can look like churn/overdue behavior | Minimum usable gaps; explicit insufficient-evidence category; sensitivity views |
| Too many advanced plots overwhelm new users | Users cannot locate or interpret useful insights | Question-first navigation; advanced/experimental area; test real user tasks |
| Duplicate view code drifts | Fixes may not affect running UI; formulas may diverge | Route registry, single implementation, parity tests during migration |
| Chart card becomes verbose/cluttered | Users ignore explanations if every panel is expanded | Show question/takeaway/status by default; put formulas and extra detail in expandable evidence panel |
| CI “green” relies on weak assertions | Regression detection is illusory | Property tests, hand-computed fixtures, no unconditional pass logic |
| Security utilities create false confidence | Helpers alone are not authentication/sandboxing | Integrate in real flow; use platform auth; document threat boundary |
| Monetary forecast is mistaken for profit | Decisions are based on wrong business objective | Correct naming; require margin/costs for profit CLV |
| Rule-based label is marketed as a validated score | Labels may be circular or uncalibrated | Distinguish descriptive rules from predictive estimates and causal effects |
| Product association is driven by low counts | High lift can be unstable in sparse baskets | Minimum pair counts, uncertainty/sensitivity, support count always visible |
| Sampled scatter hides small segments | Visually dominant classes can crowd out minor patterns | Show full-population aggregates; consider stratified/density views; disclose sample counts |

---

## 22. Research and technical references

Use these references to ground method choices and reporting standards. Re-check the exact assumptions before implementation; references do not validate the repository’s current code by themselves.

### Customer-base models and CLV

- Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). “RFM and CLV: Using Iso-Value Curves for Customer Base Analysis.” *Journal of Marketing Research*. DOI: [10.1509/jmkr.2005.42.4.415](https://doi.org/10.1509/jmkr.2005.42.4.415).
- Fader, P. S., Hardie, B. G. S., & Lee, K. L. “Counting Your Customers the Easy Way: An Alternative to the Pareto/NBD Model.” BG/NBD background and related material from the authors’ customer-base analysis work; see also [Bruce Hardie’s research page](https://www.brucehardie.com/).
- Schmittlein, D. C., Morrison, D. G., & Colombo, R. (1987). “Counting Your Customers: Who Are They and What Will They Do Next?” *Management Science*. DOI: [10.1287/mnsc.33.1.1](https://doi.org/10.1287/mnsc.33.1.1).

### Survival analysis

- Kaplan, E. L., & Meier, P. (1958). “Nonparametric Estimation from Incomplete Observations.” *Journal of the American Statistical Association*. DOI: [10.1080/01621459.1958.10501452](https://doi.org/10.1080/01621459.1958.10501452).

### Experimentation and causal interpretation

- Hernán, M. A., & Robins, J. M. *Causal Inference: What If*. Free book: [https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/).
- Athey, S., & Imbens, G. W. (2017). “The State of Applied Econometrics: Causality and Policy Evaluation.” *Journal of Economic Perspectives*. DOI: [10.1257/jep.31.2.3](https://doi.org/10.1257/jep.31.2.3).

### Forecasting and evaluation

- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*. [Online textbook](https://otexts.com/fpp3/).
- Proper evaluation should use time-ordered splits, target-appropriate error metrics and comparison baselines. Treat MAE/RMSE, probability calibration and survival metrics as target-specific rather than interchangeable.

### Data visualization

- Munzner, T. *Visualization Analysis and Design*. CRC Press (framework for task/data/encoding design).
- Cleveland, W. S., & McGill, R. (1984). “Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods.” *Journal of the American Statistical Association*. DOI: [10.1080/01621459.1984.10478080](https://doi.org/10.1080/01621459.1984.10478080).
- For implementation-specific APIs and current support, consult the installed documentation for [Streamlit](https://docs.streamlit.io/), [Plotly](https://plotly.com/python/) and [Polars](https://docs.pola.rs/). A method’s appearance in documentation is not evidence that its current app integration was executed successfully.

### Repository-specific documentation

- [Metric definitions](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/METRICS.md)
- [Data contract](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/DATA_CONTRACT.md)
- [Dashboard limits and methodology](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/DASHBOARD.md)
- [Visual analytics definitions and limitations](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/VISUAL_INSIGHTS.md)
- [Observed patronage rules and caveats](https://github.com/hrczggyrgy/polars-customer-behavior-analytics/blob/main/docs/PATRONAGE.md)

---

## 23. Audit boundary and unresolved verification

This plan is based on source and documentation visible on GitHub, repository metadata and reported GitHub Actions status. In the current review:

- The latest CI run is confirmed as `failure` by GitHub metadata, but the exact run log/job output could not be retrieved through the available tool.
- The undefined `Tuple` annotation and vacuous assertions are verifiable source defects; they are strong repair candidates but are not asserted to be the only cause of the CI failure.
- Model-evaluation placeholders are directly visible in source and must not be treated as valid metrics.
- The coverage-default behavior has been improved in source; the remaining date/metadata contract requires end-to-end tests.
- A source-level inspection cannot prove the hosted Streamlit UI renders correctly, that every imported view is the path actually used, or that production data fits resource limits.
- Security helper functions are present, but the actual application path and deployment controls require integration/runtime verification.

### Final recommendation

**Ship the next milestone only after P0 correctness and CI work is complete.** Keep the current transaction-based analytical foundation. Finish the coverage contract, implement real time-ordered model evaluation or hide it, consolidate the UI around one view/chart framework, and verify the actual app against sparse as well as dense synthetic fixtures. Only then should predictive functionality be labelled validated or be used to guide operational decisions.

The central product promise should be simple: **every chart makes customer behavior easier to understand, and every conclusion communicates the limits of the data that supports it.**
