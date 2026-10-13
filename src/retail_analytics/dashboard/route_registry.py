"""Route registry for dashboard views - single source of truth for view routing."""

from typing import Callable, TypedDict
from pathlib import Path

import polars as pl
from retail_analytics.visualization.chart_components import build_color_map


class ViewRenderer(TypedDict):
    """Type for a view renderer function."""
    renderer: Callable
    title: str
    category: str  # "core", "advanced", "experimental"
    requires_coverage: bool


# Route registry: maps view_id to renderer info
# This is the single source of truth for all view routing
# Renderer names should match functions exported from views modules
# Views marked with "inline:app" are still implemented in app.py and need migration
ROUTE_REGISTRY: dict[str, ViewRenderer] = {
    # Core views - always available (Data Trust & Overview)
    "Data Trust & Quality": {
        "renderer": "render_data_trust_quality",
        "title": "Data Trust & Quality",
        "category": "core",
        "requires_coverage": False,
    },
    "Customer Base Summary": {
        "renderer": "render_customer_base_summary",
        "title": "Customer Base Summary",
        "category": "core",
        "requires_coverage": False,
    },
    "Key Signals": {
        "renderer": "render_key_signals",
        "title": "Key Signals",
        "category": "core",
        "requires_coverage": False,
    },
    "Customer Brief": {
        "renderer": "render_customer_brief",
        "title": "Customer Brief",
        "category": "core",
        "requires_coverage": False,
    },
    "Customer Timeline": {
        "renderer": "render_customer_timeline",
        "title": "Customer Timeline",
        "category": "core",
        "requires_coverage": False,
    },
    "Segments": {
        "renderer": "render_segments",
        "title": "Segments",
        "category": "core",
        "requires_coverage": False,
    },
    "Analysis Settings": {
        "renderer": "render_analysis_settings",
        "title": "Analysis Settings",
        "category": "core",
        "requires_coverage": False,
    },
    "Metric Definitions": {
        "renderer": "render_metric_definitions",
        "title": "Metric Definitions",
        "category": "core",
        "requires_coverage": False,
    },
    "Thresholds & Calibration": {
        "renderer": "render_thresholds_calibration",
        "title": "Thresholds & Calibration",
        "category": "core",
        "requires_coverage": False,
    },
    "Data Quality Report": {
        "renderer": "render_data_quality_report",
        "title": "Data Quality Report",
        "category": "core",
        "requires_coverage": False,
    },
    
    # Core views requiring coverage (Predictive Modeling)
    "Model Evaluation": {
        "renderer": "render_model_evaluation",
        "title": "Model Evaluation",
        "category": "core",
        "requires_coverage": True,
    },
    "CLV Predictions": {
        "renderer": "render_clv_predictions",
        "title": "CLV Predictions",
        "category": "core",
        "requires_coverage": True,
    },
    
    # Lifecycle & Repeat views
    "Cohort-Age Retention": {
        "renderer": "render_cohort_age_retention",
        "title": "Cohort-Age Retention",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Time to Repeat": {
        "renderer": "render_time_to_repeat",
        "title": "Time to Repeat",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Cadence Validation": {
        "renderer": "render_cadence_validation",
        "title": "Cadence Validation",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Multi-Horizon Cohorts": {
        "renderer": "render_multi_horizon_cohorts",
        "title": "Multi-Horizon Cohorts",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Purchase Cadence": {
        "renderer": "render_purchase_cadence",
        "title": "Purchase Cadence",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Cohorts": {
        "renderer": "render_cohorts",
        "title": "Cohorts",
        "category": "advanced",
        "requires_coverage": True,
    },
    
    # Value & Risk views
    "RFM Response Surface": {
        "renderer": "render_rfm_response_surface",
        "title": "RFM Response Surface",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Value-at-Risk Scatter": {
        "renderer": "render_value_at_risk_scatter",
        "title": "Value-at-Risk Scatter",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Revenue Bridge": {
        "renderer": "render_revenue_bridge",
        "title": "Revenue Bridge",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Revenue Waterfall": {
        "renderer": "render_revenue_waterfall",
        "title": "Revenue Waterfall",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Customer Momentum": {
        "renderer": "render_customer_momentum",
        "title": "Customer Momentum",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Spend Concentration": {
        "renderer": "render_spend_concentration",
        "title": "Spend Concentration",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Value-Tier Transitions": {
        "renderer": "render_value_tier_transitions",
        "title": "Value-Tier Transitions",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Behavior × Trajectory": {
        "renderer": "render_behavior_trajectory",
        "title": "Behavior × Trajectory",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Purchase Rhythm": {
        "renderer": "render_purchase_rhythm",
        "title": "Purchase Rhythm",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Basket Evolution": {
        "renderer": "render_basket_evolution",
        "title": "Basket Evolution",
        "category": "advanced",
        "requires_coverage": True,
    },
    
    # Products & Categories views
    "Category Opportunity Map": {
        "renderer": "render_category_opportunity_map",
        "title": "Category Opportunity Map",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Category Participation": {
        "renderer": "render_category_participation",
        "title": "Category Participation",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Trial Conversion Intervals": {
        "renderer": "render_trial_conversion_intervals",
        "title": "Trial Conversion Intervals",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Product Affinity": {
        "renderer": "render_product_affinity",
        "title": "Product Affinity",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Product Association Network": {
        "renderer": "render_product_association_network",
        "title": "Product Association Network",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Category Penetration × Growth": {
        "renderer": "render_category_penetration_growth",
        "title": "Category Penetration × Growth",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Category Sankey": {
        "renderer": "render_category_sankey",
        "title": "Category Sankey",
        "category": "advanced",
        "requires_coverage": False,
    },
    
    # Advanced Analytics views
    "Future Validation": {
        "renderer": "render_future_validation",
        "title": "Future Validation",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Transitions": {
        "renderer": "render_transitions",
        "title": "Transitions",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Segment Alluvial Flow": {
        "renderer": "render_segment_alluvial_flow",
        "title": "Segment Alluvial Flow",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Full-Population Distributions": {
        "renderer": "render_full_population_distributions",
        "title": "Full-Population Distributions",
        "category": "advanced",
        "requires_coverage": False,
    },
    
    # Legacy views (inline in app.py - need migration to view modules)
    "Overview": {
        "renderer": "render_overview",
        "title": "Overview",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Segments": {
        "renderer": "render_segments",
        "title": "Segments",
        "category": "core",
        "requires_coverage": False,
    },
    "Features": {
        "renderer": "render_features",
        "title": "Features",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Cohorts": {
        "renderer": "render_cohorts",
        "title": "Cohorts",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Customer": {
        "renderer": "render_customer_explorer",
        "title": "Customer Explorer",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Methods": {
        "renderer": "render_methods",
        "title": "Methods",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Quality": {
        "renderer": "render_quality",
        "title": "Quality",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Cadence Validation": {
        "renderer": "render_cadence_validation",
        "title": "Cadence Validation",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Category Sankey": {
        "renderer": "render_category_sankey",
        "title": "Category Sankey",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Value-Tier Transitions": {
        "renderer": "render_value_tier_transitions",
        "title": "Value-Tier Transitions",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Purchase Rhythm": {
        "renderer": "render_purchase_rhythm",
        "title": "Purchase Rhythm",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Revenue Waterfall": {
        "renderer": "render_revenue_waterfall",
        "title": "Revenue Waterfall",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Basket Evolution": {
        "renderer": "render_basket_evolution",
        "title": "Basket Evolution",
        "category": "advanced",
        "requires_coverage": True,
    },
    "Behavior × Trajectory": {
        "renderer": "render_behavior_trajectory",
        "title": "Behavior × Trajectory",
        "category": "advanced",
        "requires_coverage": False,
    },
    "Purchase Cadence": {
        "renderer": "render_purchase_cadence",
        "title": "Purchase Cadence",
        "category": "advanced",
        "requires_coverage": True,
    },
    
    # Features view
    "Features": {
        "renderer": "render_features",
        "title": "Features",
        "category": "advanced",
        "requires_coverage": False,
    },
}


def get_route_registry() -> dict[str, ViewRenderer]:
    """Get the route registry."""
    return ROUTE_REGISTRY


def get_core_views() -> list[str]:
    """Get list of core view IDs."""
    return [k for k, v in ROUTE_REGISTRY.items() if v["category"] == "core"]


def get_advanced_views() -> list[str]:
    """Get list of advanced view IDs."""
    return [k for k, v in ROUTE_REGISTRY.items() if v["category"] == "advanced"]


def get_experimental_views() -> list[str]:
    """Get list of experimental view IDs."""
    return [k for k, v in ROUTE_REGISTRY.items() if v["category"] == "experimental"]


def get_views_by_coverage(coverage: bool) -> list[str]:
    """Get list of view IDs available given coverage status."""
    return [k for k, v in ROUTE_REGISTRY.items() if not v["requires_coverage"] or coverage]


def validate_route_registry() -> list[str]:
    """
    Validate the route registry for consistency.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check for duplicate titles
    titles = {}
    for view_id, info in ROUTE_REGISTRY.items():
        if info["title"] in titles:
            errors.append(f"Duplicate title '{info['title']}' for views: {titles[info['title']]} and {view_id}")
        else:
            titles[info["title"]] = view_id
    
    # Check for empty renderer names
    for view_id, info in ROUTE_REGISTRY.items():
        if not info["renderer"]:
            errors.append(f"View '{view_id}' has empty renderer")
    
    # Check categories are valid
    valid_categories = {"core", "advanced", "experimental"}
    for view_id, info in ROUTE_REGISTRY.items():
        if info["category"] not in valid_categories:
            errors.append(f"View '{view_id}' has invalid category '{info['category']}'")
    
    return errors


# Test that registry is valid
if __name__ == "__main__":
    errors = validate_route_registry()
    if errors:
        print("Route registry validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Route registry is valid!")
        print(f"Core views: {len(get_core_views())}")
        print(f"Advanced views: {len(get_advanced_views())}")
        print(f"Experimental views: {len(get_experimental_views())}")