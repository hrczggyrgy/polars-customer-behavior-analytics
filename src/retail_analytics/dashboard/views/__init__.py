"""Dashboard view modules."""

from retail_analytics.dashboard.views.overview import render_data_trust_quality, render_customer_base_summary, render_key_signals
from retail_analytics.dashboard.views.lifecycle import (
    render_cohort_age_retention,
    render_time_to_repeat,
    render_multi_horizon_cohorts,
    render_purchase_cadence,
    render_cadence_validation,
    render_future_validation,
)
from retail_analytics.dashboard.views.value_risk import (
    render_rfm_response_surface,
    render_value_at_risk_scatter,
    render_revenue_bridge,
    render_revenue_waterfall,
    render_customer_momentum,
    render_spend_concentration,
    render_value_tier_transitions,
)
from retail_analytics.dashboard.views.products import (
    render_category_penetration_growth,
    render_category_opportunity_map,
    render_category_participation,
    render_product_association_network,
    render_trial_conversion_intervals,
    render_product_affinity,
)
from retail_analytics.dashboard.views.customer_explorer import (
    render_customer_brief,
    render_customer_timeline,
)
from retail_analytics.dashboard.views.methods import (
    render_metric_definitions,
    render_thresholds_calibration,
    render_data_quality_report,
    render_analysis_settings,
)
from retail_analytics.dashboard.views.segments import render_segments
from retail_analytics.dashboard.views.features import render_features
from retail_analytics.dashboard.views.cohorts import render_cohorts
from retail_analytics.dashboard.views.transitions import render_transitions
from retail_analytics.dashboard.views.advanced import (
    render_behavior_trajectory,
    render_purchase_rhythm,
    render_basket_evolution,
    render_category_sankey,
)
from retail_analytics.dashboard.views.modeling import (
    render_model_evaluation,
    render_clv_predictions,
)

__all__ = [
    # Overview
    "render_data_trust_quality",
    "render_customer_base_summary",
    "render_key_signals",
    # Lifecycle & Repeat
    "render_cohort_age_retention",
    "render_time_to_repeat",
    "render_multi_horizon_cohorts",
    "render_purchase_cadence",
    "render_cadence_validation",
    "render_future_validation",
    # Value & Risk
    "render_rfm_response_surface",
    "render_value_at_risk_scatter",
    "render_revenue_bridge",
    "render_revenue_waterfall",
    "render_customer_momentum",
    "render_spend_concentration",
    "render_value_tier_transitions",
    # Products & Categories
    "render_category_penetration_growth",
    "render_category_opportunity_map",
    "render_category_participation",
    "render_product_association_network",
    "render_trial_conversion_intervals",
    "render_product_affinity",
    # Customer Explorer
    "render_customer_brief",
    "render_customer_timeline",
    # Methods & Data Quality
    "render_metric_definitions",
    "render_thresholds_calibration",
    "render_data_quality_report",
    "render_analysis_settings",
    # Advanced Analytics
    "render_behavior_trajectory",
    "render_purchase_rhythm",
    "render_basket_evolution",
    "render_category_sankey",
    # Segments
    "render_segments",
    # Features
    "render_features",
    # Cohorts
    "render_cohorts",
    # Predictive Modeling
    "render_model_evaluation",
    "render_clv_predictions",
]