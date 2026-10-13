"""Chart specifications for primary views - defines the metadata contract for each chart.

This module provides typed specifications (ChartSpec) for all primary views,
ensuring consistent question/takeaway/evidence/caveat/next-step metadata.
"""

import plotly.graph_objects as go

from retail_analytics.visualization.chart_components import (
    OKABE_ITO,
    ChartCard,
    ChartEvidence,
    ChartSpec,
    EvidenceState,
    SupportThresholds,
    bootstrap_ci,
    build_color_map,
    create_chart_card,
    evaluate_evidence,
    format_axis_title,
    format_evidence_badge,
    get_color,
    multiplicity_warning,
    safe_list,
    sampling_disclosure,
    threshold_sensitivity_note,
)

# ============================================================================
# Primary View Chart Specifications
# ============================================================================

def get_cohort_age_retention_spec(figure: go.Figure, manifest: dict, chosen: str,
                                  total_cohorts: int, total_customers: int,
                                  observed_cells: int, min_cohort: int) -> ChartSpec:
    """Cohort-Age Retention Heatmap specification."""
    cutoff_date = chosen
    return ChartSpec(
        key="cohort_age_retention",
        title="Cohort-Age Retention Heatmap",
        question="How does observed activity change after the first recorded purchase?",
        takeaway=f"{total_cohorts} cohorts, {total_customers:,} total customers, {observed_cells} observed age-cohort cells",
        evidence=ChartEvidence(
            numerator="Active customers in cohort at each age",
            denominator="Original cohort size",
            population_size=total_customers,
            comparison_period="Monthly since first observed purchase",
            confidence_interval="Not shown (raw rates)",
            interval_type="Descriptive rates only",
            eligibility_criteria=f"Min cohort size: {min_cohort}",
        ),
        how_to_read=(
            "Rows: first-observed purchase month (cohort). Columns: months since first purchase (age). "
            "Color: proportion of original cohort making a purchase in each month (0-100%). "
            "Age-0 is the cohort month (baseline). Blank cells = not yet observed (cohort too young). "
            "Consistent 0-100% color scale across all cohorts for comparability."
        ),
        caveat=(
            "First-observed cohort may be left-censored (capture start ≠ acquisition). "
            "Cohorts have unequal observation windows - immature cohorts appear worse. "
            "Raw rates not comparable without exposure adjustment. "
            "This is observed activity retention, not contractual retention."
        ),
        next_step="Check 'Multi-Horizon Cohorts' for repeat rates with Wilson intervals by horizon.",
        alt_description="Heatmap showing cohort-age retention with cohorts as rows and months since first purchase as columns, color-coded by activity rate.",
        figure=figure,
    )


def get_rfm_response_surface_spec(figure: go.Figure, surface_data, rec_bins: int, freq_bins: int,
                                   total_eligible: int, cells_with_ci: int) -> ChartSpec:
    """RFM Response Surface specification."""
    return ChartSpec(
        key="rfm_surface",
        title="RFM Response Surface: 30-Day Repeat Probability",
        question="Which combinations of recency and frequency show different future repeat rates?",
        takeaway=f"Surface shows {cells_with_ci} cells with Wilson 95% CI across {rec_bins}×{freq_bins} bins; total eligible: {total_eligible:,}",
        evidence=ChartEvidence(
            numerator="Customers with repeat purchase within 30 days",
            denominator="Eligible customers per recency×frequency bin",
            population_size=total_eligible,
            comparison_period="30-day follow-up",
            confidence_interval="Wilson 95% per cell",
            interval_type="Binomial proportion CI",
            eligibility_criteria="Coverage confirmed + 30-day follow-up observed",
        ),
        how_to_read=(
            "Heatmap cells show 30-day repeat rate (color: green=high, red=low). "
            "X-axis: frequency quantile bins (purchases_90d). Y-axis: recency quantile bins. "
            "Hover shows exact rate, eligible N, buyers, and Wilson CI. "
            "Blank cells = insufficient eligible customers (<20)."
        ),
        caveat=(
            "Quantile bins on recency and frequency_90d. Wilson 95% CI for cells with ≥20 eligible. "
            "Descriptive association, not causal prediction or CLV forecast. "
            "Multiple comparisons not multiplicity-adjusted. "
            "First observed purchase ≠ true acquisition."
        ),
        next_step="Drill into 'Segments' view to see repeat rates by behavioral segment.",
        alt_description="Heatmap showing 30-day repeat probability across recency and frequency quantile bins with Wilson confidence intervals.",
        figure=figure,
    )


def get_cadence_validation_spec(figure: go.Figure, horizon: int, max_lateness: float, bins: int) -> ChartSpec:
    """Cadence Validation specification."""
    return ChartSpec(
        key=f"cadence_validation_{horizon}d",
        title=f"{horizon}-Day Return Rate by Lateness Bin",
        question="Does the lateness = recency/median_gap boundary correspond to actual return probability?",
        takeaway=f"Return rate by lateness bins (max {max_lateness}×, {bins} bins) with Wilson 95% CI",
        evidence=ChartEvidence(
            numerator="Customers with future purchase within horizon",
            denominator="Eligible customers per lateness bin",
            population_size=None,  # Filled dynamically
            comparison_period=f"{horizon}-day follow-up",
            confidence_interval="Wilson 95% per bin",
            interval_type="Binomial proportion CI",
            eligibility_criteria="Style-eligible customers with confirmed coverage",
        ),
        how_to_read=(
            "Bars show future purchase rate per lateness bin. "
            "Grey line: recency = median_gap (on cadence). "
            "Red line: recency = 2× median_gap (overdue threshold). "
            "Error bars: Wilson 95% CI (bins with ≥20 eligible)."
        ),
        caveat=(
            "Wilson 95% CI shown for bins with ≥20 eligible customers. "
            "Lateness = recency / median_gap. Research: BG/NBD repeat-buying models. "
            "Exploratory comparisons not multiplicity-adjusted."
        ),
        next_step="Check 'Purchase Cadence' for customer-level cadence map.",
        alt_description=f"Bar chart showing {horizon}-day return rate across lateness bins with Wilson confidence intervals.",
        figure=figure,
    )


def get_multi_horizon_cohorts_spec(figure: go.Figure, horizons: list[int],
                                    cohort_data, colors: dict) -> ChartSpec:
    """Multi-Horizon Cohorts specification."""
    return ChartSpec(
        key="multi_horizon_cohorts",
        title="Age-Aligned Repeat Purchase Rates by Cohort (30/60/90d)",
        question="How do repeat purchase rates vary by cohort age and horizon?",
        takeaway=f"Age-aligned repeat rates for {len(horizons)} horizons across {cohort_data['cohort'].n_unique()} cohorts",
        evidence=ChartEvidence(
            numerator="Customers with repeat purchase within horizon",
            denominator="Original cohort size",
            population_size=None,  # Filled dynamically
            comparison_period="30/60/90 days since first observed purchase",
            confidence_interval="Wilson 95% for eligible cohorts",
            interval_type="Binomial proportion CI",
            eligibility_criteria="Coverage confirmed + complete follow-up + min cohort size",
        ),
        how_to_read=(
            "Lines show age-aligned repeat rates by cohort month for each horizon. "
            "Markers with error bars: eligible cohorts with Wilson 95% CI. "
            "X markers: ineligible cohorts (insufficient follow-up or N). "
            "Red dashed line: first cohort (left-censored)."
        ),
        caveat=(
            "Age-aligned: repeat within X days of first purchase. Wilson 95% CI for eligible cohorts. "
            "First cohort left-censored. Caveat: first observed ≠ true acquisition."
        ),
        next_step="Check 'Cohort-Age Retention' for full retention heatmap.",
        alt_description="Line chart showing age-aligned repeat purchase rates by cohort month for 30, 60, and 90-day horizons with Wilson confidence intervals.",
        figure=figure,
    )


def get_future_validation_spec(figure: go.Figure, horizon: int, dim: str,
                                plot_data, total_eligible: int) -> ChartSpec:
    """Future Validation specification."""
    return ChartSpec(
        key=f"future_validation_{horizon}d_{dim}",
        title=f"Future {horizon}-Day Purchase Rate by {dim}",
        question="Do current behavioral tags predict future purchases?",
        takeaway=f"Future {horizon}-day purchase rate by {dim} with Wilson 95% CI (n={total_eligible:,} eligible)",
        evidence=ChartEvidence(
            numerator="Customers with future purchase within horizon",
            denominator="Eligible customers per segment",
            population_size=total_eligible,
            comparison_period=f"{horizon}-day follow-up",
            confidence_interval="Wilson 95% per segment",
            interval_type="Binomial proportion CI",
            eligibility_criteria="Coverage confirmed + eligible_followup flag",
        ),
        how_to_read=(
            "Bars show repeat rate per segment. Error bars: Wilson 95% CI (groups with ≥20 eligible). "
            "Hover shows buyers/eligible, mean future spend, and mean spend of buyers."
        ),
        caveat=(
            "Only customers with eligible_followup flag included. Wilson 95% CI for groups with ≥20 eligible. "
            "Descriptive association, not causal prediction. Multiple comparisons not multiplicity-adjusted."
        ),
        next_step="Check 'RFM Response Surface' for recency×frequency breakdown.",
        alt_description=f"Bar chart showing future {horizon}-day purchase rate by {dim} with Wilson confidence intervals.",
        figure=figure,
    )


def get_time_to_repeat_spec(figure: go.Figure, km_data, total_at_risk: int,
                            total_events: int, median_day: int, pct_30: float) -> ChartSpec:
    """Time to Repeat (Kaplan-Meier) specification."""
    return ChartSpec(
        key="km_survival",
        title="Time to Second Observed Shopping Day (Kaplan–Meier)",
        question="How long does it take for a customer to make a second observed shopping event?",
        takeaway=f"Median time to repeat: {median_day}d; {pct_30:.1%} repeat within 30 days (n={total_at_risk:,} at risk)",
        evidence=ChartEvidence(
            numerator="Observed second purchases",
            denominator="Customers at risk at day 1",
            population_size=total_at_risk,
            comparison_period="Up to 180 days since first observed purchase",
            confidence_interval="Greenwood 95% CI",
            interval_type="Kaplan-Meier survival CI",
            eligibility_criteria="Customers with first purchase before cutoff",
        ),
        how_to_read=(
            "Blue line: cumulative repeat probability (1 - survival). "
            "Shaded band: 95% Greenwood confidence interval. "
            "Vertical dotted lines: 25th, 50th, 75th percentile days. "
            "Right-censored at cutoff for customers without second purchase."
        ),
        caveat=(
            "Right-censored at cutoff. First observed purchase ≠ true acquisition. "
            "Greenwood 95% CI. Research: Schmittlein, Morrison & Colombo (1987), Management Science."
        ),
        next_step="Check 'Cadence Validation' for cadence-based return probability.",
        alt_description="Kaplan-Meier survival curve showing cumulative repeat probability over days since first observed purchase with 95% confidence bands.",
        figure=figure,
    )


def get_category_opportunity_map_spec(figure: go.Figure, opp_map, top_categories: int) -> ChartSpec:
    """Category Opportunity Map specification."""
    return ChartSpec(
        key="category_opportunity_map",
        title="Category Opportunity Map: Penetration × Intensity",
        question="Which categories have broad reach but low intensity, or narrow reach but high intensity?",
        takeaway=f"Top {top_categories} categories by spend share; bubble size=spend share, color=repeat buyer share",
        evidence=ChartEvidence(
            numerator="Category buyers",
            denominator="Population customers",
            population_size=None,  # Filled dynamically
            comparison_period="365-day window",
            confidence_interval="Not applicable",
            interval_type="Descriptive rates only",
            eligibility_criteria="Customers with ≥1 shopping day in window",
        ),
        how_to_read=(
            "X-axis: Penetration (% of customers buying category). "
            "Y-axis: Purchase occasions per category buyer. "
            "Bubble size: Category spend share. Color: Repeat buyer share. "
            "Top-right: high penetration + high intensity (core categories). "
            "Top-left: high penetration + low intensity (broad but shallow). "
            "Bottom-right: low penetration + high intensity (niche/loyal)."
        ),
        caveat=(
            "Categories may overlap within customers, so shares need not sum to 100%. "
            "Spend is recorded spend, not margin. Research: Ehrenberg (1988), Dirichlet model benchmarks."
        ),
        next_step="Check 'Category Penetration × Growth' for period-over-period changes.",
        alt_description="Scatter plot showing category penetration vs purchase occasions per buyer, bubble size by spend share, color by repeat buyer share.",
        figure=figure,
    )


def get_category_penetration_growth_spec(figure: go.Figure, portfolio, top_n: int,
                                          current_cutoff: str, previous_cutoff: str) -> ChartSpec:
    """Category Penetration × Growth Portfolio specification."""
    return ChartSpec(
        key="category_penetration_growth",
        title="Category Penetration vs Growth Portfolio",
        question="Which categories are growing/shrinking in customer reach?",
        takeaway=f"Top {top_n} categories: penetration change ({current_cutoff} vs {previous_cutoff}), bubble=spend share",
        evidence=ChartEvidence(
            numerator="Category buyers in current period",
            denominator="Population customers in each period",
            population_size=None,
            comparison_period=f"{previous_cutoff} → {current_cutoff}",
            confidence_interval="Not shown",
            interval_type="Descriptive change",
            eligibility_criteria="Customers with ≥1 shopping day in window",
        ),
        how_to_read=(
            "X-axis: Current penetration (% of customers buying category). "
            "Y-axis: Change in penetration (percentage points, current - previous). "
            "Bubble size: Current spend share. Color: Repeat buyer share. "
            "Top-right: growing reach + high penetration. Bottom-left: declining reach."
        ),
        caveat=(
            "Participation = category buyers / period population. Fixed populations per period. "
            "Categories with zero buyers in a period show 0% participation, not missing. "
            "Change in percentage points, not relative change."
        ),
        next_step="Check 'Category Opportunity Map' for intensity analysis.",
        alt_description=f"Scatter plot showing category penetration vs penetration change from {previous_cutoff} to {current_cutoff}.",
        figure=figure,
    )


def get_value_at_risk_spec(figure: go.Figure, summary, lateness_thresh: float,
                           eligible_n: int, overdue_n: int) -> ChartSpec:
    """Value-at-Risk Scatter specification."""
    return ChartSpec(
        key="var_scatter",
        title="Value-at-Risk: Lateness vs 365d Spend",
        question="Which high-spend customers are overdue relative to their typical cadence?",
        takeaway=f"{eligible_n:,} eligible customers; {overdue_n:,} overdue (≥{lateness_thresh}× median gap)",
        evidence=ChartEvidence(
            numerator="Overdue high-spend customers",
            denominator="Style-eligible customers with finite cadence",
            population_size=eligible_n,
            comparison_period="365-day window",
            confidence_interval="Not applicable",
            interval_type="Descriptive",
            eligibility_criteria="Finite median_gap > 0, finite recency, style-eligible",
        ),
        how_to_read=(
            "X-axis: Lateness = recency / median_gap (how overdue). "
            "Y-axis: Recorded spend in last 365 days. "
            "Bubble size: Purchase count in 365d. "
            "Grey dashed: on cadence (1×). Red dashed: overdue (2×). "
            "Orange dotted: user threshold. Green/Orange/Red regions: on cadence / 1-2× / ≥2× overdue."
        ),
        caveat=(
            "Not a churn prediction; describes observed inactivity relative to historical cadence. "
            "Spend is recorded amount, not profit. Customers may be inactive for many reasons."
        ),
        next_step="Check 'Customer Brief' for individual customer timeline.",
        alt_description="Scatter plot showing lateness (recency/median_gap) vs 365-day spend, bubble size by purchase count, colored regions for cadence status.",
        figure=figure,
    )


def get_product_affinity_spec(figure: go.Figure, min_support: float, min_conf: float,
                              min_lift: float, total_baskets: int, total_pairs: int) -> ChartSpec:
    """Product Affinity specification."""
    return ChartSpec(
        key="product_affinity",
        title="Product Affinity: Association Rules",
        question="Which products occur together in baskets often enough to warrant investigation?",
        takeaway=f"Top {total_pairs} pairs from {total_baskets:,} baskets (min support={min_support:.3%}, conf={min_conf:.2f}, lift={min_lift:.2f})",
        evidence=ChartEvidence(
            numerator="Co-occurrence count",
            denominator="Total baskets",
            population_size=total_baskets,
            comparison_period="Analysis cutoff window",
            confidence_interval="Not shown",
            interval_type="Descriptive",
            eligibility_criteria=f"Support ≥ {min_support:.3%}, Confidence ≥ {min_conf:.0%}, Lift ≥ {min_lift:.2f}",
            sample_policy="All qualifying pairs",
            additional={
                "Min Support": f"{min_support:.3%}",
                "Min Confidence": f"{min_conf:.0%}",
                "Min Lift": f"{min_lift:.2f}",
            }
        ),
        how_to_read=(
            "Support = fraction of baskets containing both products. "
            "Confidence = P(B|A). Lift = P(A∪B)/(P(A)P(B)). "
            "Conviction = (1-P(B))/(1-P(B|A)). "
            "Table shows all metrics; rare pairs with high lift flagged."
        ),
        caveat=(
            "Rare pairs can have huge lift. Do not rank recommendations by lift alone. "
            "Report raw pair count and support denominator prominently. "
            "Association ≠ causation; does not imply incremental cross-sell lift."
        ),
        next_step="Check 'Product Association Network' for network visualization.",
        alt_description="Bar chart and table of product association rules with support, confidence, lift, and conviction metrics.",
        figure=figure,
    )


def get_trial_conversion_spec(figure: go.Figure, min_trials: int, trial_data) -> ChartSpec:
    """Trial Conversion Intervals specification."""
    return ChartSpec(
        key="trial_conversion_intervals",
        title=f"Trial-to-Repeat Conversion (Min {min_trials} Trials)",
        question="Among products first observed in an eligible trial window, what share has a qualifying repeat?",
        takeaway=f"{trial_data.height} products with ≥{min_trials} eligible trials, Wilson 95% CI shown for ≥20 trials",
        evidence=ChartEvidence(
            numerator="Repeated trials",
            denominator="Eligible trials",
            population_size=None,
            comparison_period="90-day repeat window after trial",
            confidence_interval="Wilson 95% (products with ≥20 eligible trials)",
            interval_type="Binomial proportion CI",
            eligibility_criteria="First purchase in [C-364, C-90] with 90-day follow-up fully observed",
        ),
        how_to_read=(
            "Dots: pooled trial conversion rate per product (total repeats / total eligible trials). "
            "Horizontal lines: Wilson 95% CI (products with ≥20 trials). "
            "Dot size: number of eligible trials. "
            "Products without CI have high uncertainty."
        ),
        caveat=(
            "Suppresses ineligible/immature trial windows. Product availability and source capture affect repeat rates. "
            "Avoid equating repeat with satisfaction. Pooled conversion = Σ repeats / Σ eligible trials across customers."
        ),
        next_step="Check 'Product Association Network' for sequential purchase patterns.",
        alt_description="Dot-and-interval plot showing trial-to-repeat conversion rate by product with Wilson 95% confidence intervals.",
        figure=figure,
    )


def get_segment_summary_spec(figure: go.Figure, dim: str, horizon: int,
                             records: list, valid_count: int) -> ChartSpec:
    """Segment Summary specification."""
    return ChartSpec(
        key=f"segment_summary_{horizon}d_{dim}",
        title=f"Segment Comparison: {horizon}-Day Repeat Rate by {dim}",
        question="How does observed future repeat outcome vary across segments?",
        takeaway=f"{valid_count} segments with eligible data; Wilson 95% CI per segment",
        evidence=ChartEvidence(
            numerator="Customers with repeat purchase",
            denominator="Eligible customers per segment",
            population_size=None,
            comparison_period=f"{horizon}-day follow-up",
            confidence_interval="Wilson 95% per segment",
            interval_type="Binomial proportion CI",
            eligibility_criteria="Coverage confirmed + eligible_followup flag",
        ),
        how_to_read=(
            "Bars: segment size (customers or spend). "
            "Scatter with error bars: repeat rate with Wilson 95% CI. "
            "Evidence badges below show strong/limited/insufficient per segment."
        ),
        caveat=(
            "Intervals assume independent customers at one cutoff, not causal effects. "
            "Up to 20 exploratory comparisons shown; not multiplicity-adjusted. "
            "First observed purchase ≠ true acquisition."
        ),
        next_step="Check 'RFM Response Surface' for recency×frequency breakdown.",
        alt_description=f"Bar chart and scatter plot showing {horizon}-day repeat rate by {dim} segment with Wilson confidence intervals.",
        figure=figure,
    )


def get_revenue_bridge_spec(figure: go.Figure, origin: str, chosen: str,
                            bridge_data, match_pop: bool) -> ChartSpec:
    """Revenue Bridge specification."""
    return ChartSpec(
        key="revenue_bridge",
        title=f"Revenue Bridge: {origin} to {chosen}",
        question="How much of observed spend change comes from customers active in both periods vs entering/leaving?",
        takeaway=f"Spend change decomposed into customers, frequency, AOV, and residual ({'matched' if match_pop else 'unmatched'} population)",
        evidence=ChartEvidence(
            numerator="Component contributions to spend change",
            denominator="Previous period spend",
            population_size=None,
            comparison_period=f"{origin} → {chosen}",
            confidence_interval="Not applicable",
            interval_type="Arithmetic decomposition",
            eligibility_criteria="Fixed-order bridge: N → F → A",
        ),
        how_to_read=(
            "Waterfall bars: blue=start/end totals, green=positive contribution, red=negative. "
            "Components: (1) Purchasing customers change × prev freq × prev AOV, "
            "(2) Current customers × freq change × prev AOV, "
            "(3) Current customers × current freq × AOV change. "
            "Residual = interaction effects. Hover shows component description."
        ),
        caveat=(
            "Fixed-order bridge: attribution depends on decomposition order. "
            "Not causal attribution. Recorded spend, not profit. "
            f"{'Matched customers across periods' if match_pop else 'Full snapshot populations'}."
        ),
        next_step="Check 'Revenue Waterfall' for calendar-period decomposition.",
        alt_description="Waterfall chart decomposing spend change into customer count, frequency, AOV, and residual components.",
        figure=figure,
    )


def get_product_association_network_spec(figure: go.Figure, min_lift: float,
                                          min_conf: float, top_n: int,
                                          nodes_count: int, edges_count: int) -> ChartSpec:
    """Product Association Network specification."""
    return ChartSpec(
        key="product_association_network",
        title="Product Association Network",
        question="Which products co-occur in baskets with strong association?",
        takeaway=f"Network of {nodes_count} products with {edges_count} associations (min lift={min_lift:.2f}, conf={min_conf:.2f})",
        evidence=ChartEvidence(
            numerator="Association edges meeting thresholds",
            denominator="All possible product pairs",
            population_size=None,
            comparison_period="Analysis cutoff window",
            confidence_interval="Not applicable",
            interval_type="Descriptive",
            eligibility_criteria=f"Support ≥ threshold, Confidence ≥ {min_conf:.0%}, Lift ≥ {min_lift:.2f}",
        ),
        how_to_read=(
            "Nodes: products. Edges: associations exceeding thresholds. "
            "Node size = degree (connections). Node color = max lift. "
            "Layout: force-directed (spring) for clustering. "
            "Hover shows product, degree, max lift."
        ),
        caveat=(
            "Association ≠ causation; does not imply incremental cross-sell lift. "
            "Network layout is algorithmic, not semantically meaningful. "
            "Rare pairs with high lift may dominate visual attention."
        ),
        next_step="Check 'Product Affinity' for tabular association rules.",
        alt_description="Network graph showing product co-occurrence associations with node size by degree and color by max lift.",
        figure=figure,
    )


def get_revenue_waterfall_spec(figure: go.Figure, first_date: str, cutoff: str, months: int) -> ChartSpec:
    """Revenue Waterfall (Calendar Trend) specification."""
    return ChartSpec(
        key="revenue_waterfall_calendar",
        title="Calendar-Period Revenue Components",
        question="How do monthly spend, buyers, AOV, and baskets/buyer trend over non-overlapping calendar periods?",
        takeaway="12-month calendar trend (non-overlapping windows, fixed population)",
        evidence=ChartEvidence(
            numerator="Monthly metrics",
            denominator="Population customers",
            population_size=None,
            comparison_period=f"{first_date} to {cutoff}",
            confidence_interval="Not applicable",
            interval_type="Descriptive monthly series",
            eligibility_criteria="Fixed population (current filtered customers). Partial months marked.",
        ),
        how_to_read=(
            "Blue: Monthly spend. Green: Purchasing customers (right axis). "
            "Orange: AOV (right axis). Red: Baskets/buyer (right axis). "
            "Non-overlapping monthly windows. Independent observations per month."
        ),
        caveat=(
            "Non-overlapping windows are independent. Rolling 90-day windows overlap and are NOT independent. "
            "Fixed population = current filtered customers; partial months at edges may be incomplete."
        ),
        next_step="Check 'Revenue Bridge' for period-over-period decomposition.",
        alt_description="Multi-axis line chart showing monthly spend, buyers, AOV, and baskets per buyer over 12 calendar months.",
        figure=figure,
    )


def get_cohort_summary_spec(figure: go.Figure, horizon: int, horizon_data, first_cohort: str) -> ChartSpec:
    """Cohort Summary (Age-Aligned Repeat) specification."""
    return ChartSpec(
        key=f"cohort_repeat_rate_{horizon}d",
        title=f"Age-Aligned {horizon}-Day Repeat Purchase Rate by Cohort (Wilson 95% CI)",
        question="How does repeat purchase within X days of first purchase vary by first-observed cohort?",
        takeaway=f"Wilson 95% CI for eligible cohorts; first cohort left-censored at {first_cohort}",
        evidence=ChartEvidence(
            numerator="Customers with repeat purchase within horizon",
            denominator="Original cohort size",
            population_size=None,
            comparison_period=f"{horizon} days since first observed purchase",
            confidence_interval="Wilson 95% for eligible cohorts (min 20 customers)",
            interval_type="Binomial proportion CI",
            eligibility_criteria="Cohort size ≥ 20, complete follow-up for horizon",
        ),
        how_to_read=(
            "Lines+markers: eligible cohorts with Wilson 95% CI. "
            "X markers: ineligible cohorts (capture unconfirmed or insufficient N). "
            "Red dashed line: first cohort (left-censored). "
            "Color coding: blue=eligible, purple=capture unconfirmed, red=insufficient."
        ),
        caveat=(
            "First-observed cohort may be left-censored (capture start ≠ acquisition). "
            "Age-aligned: repeat within X days of first purchase. "
            "Raw rates not comparable without exposure adjustment."
        ),
        next_step="Check 'Multi-Horizon Cohorts' for 30/60/90d comparison.",
        alt_description=f"Line chart showing age-aligned {horizon}-day repeat rates by cohort with Wilson confidence intervals.",
        figure=figure,
    )


def get_customer_momentum_spec(figure: go.Figure, mom_data, finite_n: int,
                               not_comp_n: int, quad_data) -> ChartSpec:
    """Customer Momentum Map specification."""
    return ChartSpec(
        key="customer_momentum",
        title="Customer Momentum Map: Frequency vs. Basket Change",
        question="Did customers' observed shopping frequency and basket size change between comparable windows?",
        takeaway=f"{finite_n:,} comparable customers; {not_comp_n:,} not comparable (excluded)",
        evidence=ChartEvidence(
            numerator="Customers with positive/negative change",
            denominator="Comparable customers (tenure ≥ 179d, non-zero previous purchases)",
            population_size=finite_n,
            comparison_period="Recent 90d vs Previous 90d",
            confidence_interval="Not applicable",
            interval_type="Descriptive scatter",
            eligibility_criteria="Tenure ≥ 179 days, non-zero previous purchases, finite frequency_change & basket_change",
        ),
        how_to_read=(
            "X-axis: Frequency change (recent vs previous 90d purchases). "
            "Y-axis: Basket change (recent vs previous 90d AOV). "
            "Quadrants: Growing on both (top-right), More frequent/lighter (bottom-right), "
            "Less frequent/heavier (top-left), Declining on both (bottom-left). "
            "Color: value tier. Zero lines: no change reference."
        ),
        caveat=(
            "Only customers with comparable windows shown. "
            "Missing changes (not comparable) are not shown at zero. "
            "Observed spend, not profit. Quadrant labels are visual guides only."
        ),
        next_step="Check 'Basket Evolution' for slope/dumbbell comparison.",
        alt_description="Scatter plot showing frequency change vs basket change by value tier, with quadrant annotations and zero reference lines.",
        figure=figure,
    )


def get_purchase_cadence_spec(figure: go.Figure, cad_data, eligible_n: int, total_n: int) -> ChartSpec:
    """Purchase Cadence Map specification."""
    return ChartSpec(
        key="cadence_scatter",
        title="Purchase Cadence Map: Recency vs. Median Gap",
        question="Which customers are overdue relative to their typical purchase rhythm?",
        takeaway=f"{eligible_n:,} of {total_n:,} customers have finite cadence metrics",
        evidence=ChartEvidence(
            numerator="Customers with finite median_gap and recency",
            denominator="Style-eligible customers",
            population_size=eligible_n,
            comparison_period="Current snapshot",
            confidence_interval="Not applicable",
            interval_type="Descriptive scatter",
            eligibility_criteria="Style-eligible (days_365d≥6, tenure≥90, gap_count≥4)",
        ),
        how_to_read=(
            "X-axis: Median gap (typical inter-purchase interval in days). "
            "Y-axis: Recency (days since last purchase). "
            "Grey line: recency = median gap (on cadence). "
            "Red line: recency = 2× median gap (overdue threshold). "
            "Colored regions: green=on cadence, light orange=1-2× overdue, dark orange=≥2× overdue. "
            "Lateness = recency / median_gap."
        ),
        caveat=(
            "Only style-eligible customers have meaningful median_gap. "
            "Hash-sampled up to 2,000 customers. "
            "First observed purchase ≠ true acquisition."
        ),
        next_step="Check 'Cadence Validation' for return probability by lateness.",
        alt_description="Scatter plot showing recency vs median gap with cadence reference lines and overdue regions.",
        figure=figure,
    )


def get_category_participation_spec(figure: go.Figure, part_data, population_n: int,
                                     chosen: str, origin: str, top_n: int) -> ChartSpec:
    """Category Participation specification."""
    return ChartSpec(
        key="category_participation",
        title=f"Category Participation: Recent 90d vs Previous 90d (Top {top_n})",
        question="How has category reach changed between two periods?",
        takeaway=f"Population: {population_n:,} customers; participation change in pp (recent - previous)",
        evidence=ChartEvidence(
            numerator="Category buyers in period",
            denominator="Fixed population per period",
            population_size=population_n,
            comparison_period=f"Recent 90d ending {chosen} vs Previous 90d ending {origin}",
            confidence_interval="Not applicable",
            interval_type="Descriptive paired comparison",
            eligibility_criteria="Customers present in filtered population at both cutoffs",
        ),
        how_to_read=(
            "Paired horizontal bars: blue=recent 90d, orange=previous 90d. "
            "Bars grouped by category. Hover shows buyers, spend, and change (pp). "
            "Participation = category buyers / population (same denominator per period)."
        ),
        caveat=(
            "Categories with zero buyers in a period show 0% participation, not missing. "
            "Current period uses current filtered population; previous uses previous filtered population. "
            "For fixed-population comparison, use this view; for changing populations, use 'Category Sankey'."
        ),
        next_step="Check 'Category Opportunity Map' for penetration×intensity.",
        alt_description="Paired horizontal bar chart showing category participation in recent vs previous 90-day windows.",
        figure=figure,
    )


def get_segment_alluvial_spec(figure: go.Figure, dim: str, origin: str, chosen: str,
                               counts: dict, flows_count: int) -> ChartSpec:
    """Segment Alluvial Flow specification."""
    return ChartSpec(
        key=f"segment_alluvial_{dim}",
        title=f"{dim.title()} Transitions: {origin} to {chosen}",
        question="How did labels change between two snapshots?",
        takeaway=f"Matched: {counts['matched_n']:,}; Unmatched origin: {counts['origin_unmatched']:,}; Unmatched dest: {counts['destination_unmatched']:,}",
        evidence=ChartEvidence(
            numerator="Transition counts",
            denominator="Matched customers per origin label",
            population_size=counts['matched_n'],
            comparison_period=f"{origin} → {chosen}",
            confidence_interval="Not applicable",
            interval_type="Descriptive flow",
            eligibility_criteria="Customers present at both cutoffs",
        ),
        how_to_read=(
            "Sankey: flows from origin labels (left) to destination labels (right). "
            "Width = customer count. Colors = label colors. "
            "Heatmap below: origin-normalized transition proportions. "
            "Blank cells = no transitions."
        ),
        caveat=(
            "Unmatched origin customers are not churn; unmatched destination are new arrivals. "
            "Rates use matched, destination-filtered customers per origin. "
            "Displayed cells need not sum to 100% per origin."
        ),
        next_step="Check 'Value-Tier Transitions' for waterfall view of value tier flows.",
        alt_description=f"Sankey diagram showing {dim} label transitions from {origin} to {chosen}.",
        figure=figure,
    )


# ============================================================================
# Chart Card Factory Functions
# ============================================================================

def render_chart_card(spec: ChartSpec) -> None:
    """Render a ChartCard from a ChartSpec."""
    card = ChartCard(spec=spec)
    card.render()


def create_evidence_badge(state: EvidenceState) -> str:
    """Format evidence state as colored badge."""
    return format_evidence_badge(state)


# Export all
__all__ = [
    # Core classes
    "ChartSpec", "ChartEvidence", "ChartCard", "create_chart_card", "render_chart_card",
    # Support classes
    "SupportThresholds", "EvidenceState", "evaluate_evidence",
    "bootstrap_ci", "format_evidence_badge", "multiplicity_warning",
    "sampling_disclosure", "threshold_sensitivity_note",
    "build_color_map", "get_color", "format_axis_title", "OKABE_ITO", "safe_list",
    # View specs
    "get_cohort_age_retention_spec",
    "get_rfm_response_surface_spec",
    "get_cadence_validation_spec",
    "get_multi_horizon_cohorts_spec",
    "get_future_validation_spec",
    "get_time_to_repeat_spec",
    "get_category_opportunity_map_spec",
    "get_category_penetration_growth_spec",
    "get_value_at_risk_spec",
    "get_product_affinity_spec",
    "get_trial_conversion_spec",
    "get_segment_summary_spec",
    "get_revenue_bridge_spec",
    "get_product_association_network_spec",
    "get_revenue_waterfall_spec",
    "get_cohort_summary_spec",
    "get_customer_momentum_spec",
    "get_purchase_cadence_spec",
    "get_category_participation_spec",
    "get_segment_alluvial_spec",
]
