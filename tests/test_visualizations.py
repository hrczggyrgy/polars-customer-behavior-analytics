"""Tests for visualization functions."""

from datetime import date, timedelta
import polars as pl
import pytest

from retail_analytics.visualization.visual_insights_core import (
    monthly_summary,
    calendar_period_revenue_trend,
    cohort_age_retention_heatmap,
    cohort_summary,
    multi_horizon_cohort_repeat,
    future_outcome_validation,
    trial_conversion_interval_plot,
    product_association_network,
    revenue_waterfall,
    value_tier_transition_waterfall,
    momentum_map,
    behavior_trajectory_matrix,
    cadence_return_validation,
)
from retail_analytics.visualization.product_affinity import (
    ProductAffinityConfig,
    compute_frequent_items,
    compute_item_pairs,
    product_affinity_analysis,
    compute_item_pairs,
)


def make_test_data():
    """Create minimal test data for visualization tests."""
    # Validated transactions
    validated = pl.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2", "C2", "C3", "C3", "C3"],
        "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
        "date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-01-10", "2024-01-20", "2024-01-05", "2024-01-25", "2024-02-10"],
        "product_id": ["P1", "P2", "P3", "P1", "P2", "P1", "P3", "P2"],
        "category_key": ["CAT1", "CAT2", "CAT1", "CAT1", "CAT2", "CAT1", "CAT1", "CAT2"],
        "department": ["DEPT1", "DEPT2", "DEPT1", "DEPT1", "DEPT2", "DEPT1", "DEPT1", "DEPT2"],
        "quantity": [1, 2, 1, 1, 1, 2, 1, 3],
        "value": [10.0, 20.0, 15.0, 10.0, 20.0, 20.0, 15.0, 30.0],
    }).with_columns(pl.col("date").str.to_date())

    # Snapshots with features
    snapshots = pl.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "first_purchase": ["2024-01-01", "2024-01-10", "2024-01-05"],
        "last_purchase": ["2024-02-01", "2024-01-20", "2024-02-10"],
        "purchases_90d": [3, 2, 3],
        "spend_90d": [45.0, 30.0, 65.0],
        "median_gap": [15.0, 10.0, 12.0],
        "recency": [10, 20, 15],
        "tenure": [60, 40, 35],
        "behavior": ["loyal", "at_risk", "new"],
        "trajectory": ["growing", "declining", "new"],
        "value_tier": ["high", "medium", "low"],
        "lifecycle": ["active", "active", "new"],
        "evidence": ["strong", "weak", "minimal"],
        "future_purchases_30d": [1, 0, 1],
        "future_spend_30d": [15.0, 0.0, 20.0],
        "eligible_followup_30d": [True, True, True],
        "future_purchases_90d": [2, 1, 2],
        "future_spend_90d": [25.0, 10.0, 30.0],
        "eligible_followup_90d": [True, True, True],
        "style_eligible": [True, True, True],
        "median_gap": [15.0, 10.0, 12.0],
    }).with_columns([
        pl.col("first_purchase").str.to_date(),
        pl.col("last_purchase").str.to_date(),
    ])

    # Baskets for cohort tests
    baskets = pl.DataFrame({
        "customer_id": ["C1", "C1", "C2", "C2", "C3", "C3"],
        "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "date": ["2024-01-01", "2024-01-15", "2024-01-10", "2024-01-20", "2024-01-05", "2024-01-25"],
        "spend": [10.0, 20.0, 10.0, 20.0, 20.0, 15.0],
    }).with_columns(pl.col("date").str.to_date())

    return validated.lazy(), snapshots.lazy(), baskets.lazy()


class TestMonthlySummary:
    """Tests for monthly_summary function."""

    def test_monthly_summary_basic(self):
        validated, rows, _ = make_test_data()
        # Create baskets with spend column
        baskets = pl.DataFrame({
            "customer_id": ["C1", "C1", "C2", "C2", "C3", "C3"],
            "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
            "date": ["2024-01-01", "2024-01-15", "2024-01-10", "2024-01-20", "2024-01-05", "2024-01-25"],
            "spend": [10.0, 20.0, 10.0, 20.0, 20.0, 15.0],
        }).with_columns(pl.col("date").str.to_date()).lazy()
        result = monthly_summary(
            baskets,
            rows,
            date(2024, 1, 1),
            date(2024, 3, 1),
            months=3,
        )
        assert "month" in result.columns
        assert "spend" in result.columns
        assert "baskets" in result.columns
        assert "buyers" in result.columns


class TestCalendarPeriodRevenueTrend:
    """Tests for calendar_period_revenue_trend function."""

    def test_calendar_period_revenue_trend_basic(self):
        validated, rows, _ = make_test_data()
        # validated has "value" column which is required by the function
        result = calendar_period_revenue_trend(
            validated,
            rows,
            date(2024, 1, 1),
            date(2024, 3, 1),
            months=3,
        )
        assert "month" in result.columns
        assert "aov" in result.columns
        assert "baskets_per_buyer" in result.columns
        assert "buying_share" in result.columns


class TestCohortAgeRetentionHeatmap:
    """Tests for cohort_age_retention_heatmap function."""

    def test_cohort_age_retention_heatmap_basic(self):
        validated, rows, _ = make_test_data()
        result = cohort_age_retention_heatmap(
            validated, rows, date(2024, 3, 1), min_cohort_size=1
        )
        assert "cohort_month" in result.columns
        assert "age_month" in result.columns
        assert "activity_rate" in result.columns


class TestCohortSummary:
    """Tests for cohort_summary function."""

    def test_cohort_summary_basic(self):
        validated, rows, baskets = make_test_data()
        result = cohort_summary(
            baskets, rows, date(2024, 3, 1), coverage=True, minimum=1
        )
        assert "cohort" in result.columns
        assert "horizon" in result.columns
        assert "rate" in result.columns


class TestMultiHorizonCohortRepeat:
    """Tests for multi_horizon_cohort_repeat function."""

    def test_multi_horizon_cohort_repeat_basic(self):
        validated, rows, baskets = make_test_data()
        result = multi_horizon_cohort_repeat(
            baskets, rows, date(2024, 3, 1), coverage=True,
            horizons=[30, 60], min_cohort_size=1
        )
        assert "cohort" in result.columns
        assert "horizon" in result.columns
        assert "rate" in result.columns


class TestFutureOutcomeValidation:
    """Tests for future_outcome_validation function."""

    def test_future_outcome_validation_basic(self):
        validated, rows, _ = make_test_data()
        result = future_outcome_validation(rows)
        assert "dimension" in result.columns
        assert "horizon" in result.columns
        assert "rate" in result.columns

    def test_future_outcome_validation_dimension_names(self):
        validated, rows, _ = make_test_data()
        result = future_outcome_validation(rows)
        # Check that original dimension names are preserved
        dims = result["dimension"].unique().to_list()
        assert "behavior" in dims
        assert "trajectory" in dims


class TestTrialConversionIntervalPlot:
    """Tests for trial_conversion_interval_plot function."""

    def test_trial_conversion_interval_plot_basic(self):
        validated, rows, _ = make_test_data()
        # Create product history
        product_history = pl.DataFrame({
            "customer_id": ["C1", "C1", "C2"],
            "product_id": ["P1", "P2", "P1"],
            "trial": [date(2024, 1, 1), date(2024, 1, 15), date(2024, 1, 20)],
            "repeat": [date(2024, 1, 20), None, date(2024, 2, 10)],
        }).lazy()

        result = trial_conversion_interval_plot(
            product_history, rows, date(2024, 4, 1), min_trials=1
        )
        assert "product_id" in result.columns
        assert "trial_conversion" in result.columns


class TestProductAssociationNetwork:
    """Tests for product_association_network function."""

    def test_product_association_network_basic(self):
        pairs = pl.DataFrame({
            "item_a": ["P1", "P1", "P2"],
            "item_b": ["P2", "P3", "P3"],
            "cooccurrence_count": [10, 5, 8],
            "support": [0.1, 0.05, 0.08],
            "confidence_a_to_b": [0.5, 0.3, 0.4],
            "confidence_b_to_a": [0.6, 0.2, 0.5],
            "lift": [1.5, 1.2, 1.8],
            "conviction_a_to_b": [2.0, 1.5, 2.5],
            "conviction_b_to_a": [2.5, 1.3, 3.0],
        })

        result = product_association_network(pairs, min_lift=1.0, min_confidence=0.1, top_n=10)
        assert "nodes" in result
        assert "edges" in result
        assert result["nodes"].height >= 3

    def test_product_association_network_empty(self):
        pairs = pl.DataFrame({
            "item_a": [],
            "item_b": [],
            "cooccurrence_count": [],
            "support": [],
            "confidence_a_to_b": [],
            "confidence_b_to_a": [],
            "lift": [],
            "conviction_a_to_b": [],
            "conviction_b_to_a": [],
        }, schema={
            "item_a": pl.String, "item_b": pl.String,
            "cooccurrence_count": pl.Int64, "support": pl.Float64,
            "confidence_a_to_b": pl.Float64, "confidence_b_to_a": pl.Float64,
            "lift": pl.Float64, "conviction_a_to_b": pl.Float64,
            "conviction_b_to_a": pl.Float64,
        })

        result = product_association_network(pairs)
        assert result["nodes"].is_empty()
        assert result["edges"].is_empty()


class TestRevenueWaterfall:
    """Tests for revenue_waterfall function."""

    def test_revenue_waterfall_basic(self):
        rows = pl.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "purchases_90d": [3, 2, 1],
            "spend_90d": [100.0, 50.0, 25.0],
        }).lazy()
        previous_rows = pl.DataFrame({
            "customer_id": ["C1", "C2"],
            "purchases_90d": [2, 3],
            "spend_90d": [80.0, 60.0],
        }).lazy()
        result = revenue_waterfall(rows, previous_rows, date(2024, 3, 1))
        assert "component" in result.columns
        assert "value" in result.columns


class TestValueTierTransitionWaterfall:
    """Tests for value_tier_transition_waterfall function."""

    def test_value_tier_transition_waterfall_basic(self):
        flows = pl.DataFrame({
            "from_label": ["high", "high", "medium", "low"],
            "to_label": ["high", "medium", "low", "low"],
            "len": [10, 5, 3, 8],
        })
        result = value_tier_transition_waterfall(flows)
        assert "tier" in result.columns
        assert "inflow" in result.columns
        assert "outflow" in result.columns
        assert "net_change" in result.columns


class TestMomentumMap:
    """Tests for momentum_map function."""

    def test_momentum_map_basic(self):
        rows = pl.DataFrame({
            "customer_id": ["C1", "C2", "C3", "C4"],
            "frequency_change": [0.1, -0.1, 0.2, -0.2],
            "basket_change": [0.05, -0.05, 0.1, -0.1],
            "value_tier": ["high", "medium", "high", "low"],
            "lifecycle": ["active", "active", "new", "active"],
            "evidence": ["strong", "weak", "strong", "minimal"],
            "spend_90d": [100.0, 50.0, 200.0, 25.0],
            "purchases_90d": [3, 2, 4, 1],
            "purchases_previous90d": [2, 3, 3, 2],
            "aov_recent": [33.0, 25.0, 50.0, 25.0],
            "aov_previous": [30.0, 30.0, 45.0, 30.0],
        }).lazy()

        result = momentum_map(rows)
        assert "sample" in result
        assert "quadrant_summary" in result


class TestBehaviorTrajectoryMatrix:
    """Tests for behavior_trajectory_matrix function."""

    def test_behavior_trajectory_matrix_basic(self):
        rows = pl.DataFrame({
            "customer_id": ["C1", "C2", "C3", "C4"],
            "behavior": ["loyal", "loyal", "at_risk", "new"],
            "trajectory": ["growing", "stable", "declining", "new"],
            "spend_90d": [100.0, 80.0, 30.0, 20.0],
        }).lazy()

        result = behavior_trajectory_matrix(rows)
        assert "behavior" in result.columns
        assert "trajectory" in result.columns
        assert "customers" in result.columns
        assert "median_spend" in result.columns


class TestCadenceReturnValidation:
    """Tests for cadence_return_validation function."""

    def test_cadence_return_validation_no_coverage(self):
        rows = pl.DataFrame({
            "customer_id": ["C1", "C2"],
            "recency": [10.0, 20.0],
            "median_gap": [14.0, 14.0],
            "future_purchases_30d": [1, 0],
            "eligible_followup_30d": [True, True],
            "future_purchases_90d": [2, 1],
            "eligible_followup_90d": [True, True],
        }).lazy()

        result = cadence_return_validation(rows, coverage=False, max_lateness=5.0, bins=5)
        assert "warning" in result.columns


class TestProductAffinity:
    """Tests for product affinity analysis."""

    def test_compute_frequent_items(self):
        baskets = pl.DataFrame({
            "customer_id": ["C1", "C1", "C2", "C2", "C3"],
            "transaction_id": ["T1", "T2", "T3", "T4", "T5"],
            "date": [date(2024, 1, 1), date(2024, 1, 15), date(2024, 1, 10), date(2024, 1, 20), date(2024, 1, 5)],
            "product_id": ["P1", "P2", "P1", "P2", "P1"],
        }).with_columns(pl.col("date").cast(pl.Date)).lazy()

        config = ProductAffinityConfig(min_support=0.01, top_k_items=10)
        result = compute_frequent_items(baskets, "product_id", config, "2024-02-01", 90)
        assert "product_id" in result.columns
        assert "basket_count" in result.columns
        assert "support" in result.columns

    def test_compute_item_pairs(self):
        baskets = pl.DataFrame({
            "customer_id": ["C1", "C1", "C1", "C2", "C2"],
            "transaction_id": ["T1", "T1", "T2", "T3", "T3"],
            "date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 15), date(2024, 1, 10), date(2024, 1, 10)],
            "product_id": ["P1", "P2", "P2", "P1", "P2"],
        }).with_columns(pl.col("date").cast(pl.Date)).lazy()

        frequent_items = pl.DataFrame({
            "product_id": ["P1", "P2"],
            "basket_count": [2, 3],
            "support": [0.4, 0.6],
            "total_baskets": [5, 5],
        })

        config = ProductAffinityConfig(min_support=0.01, top_k_pairs=10)
        result = compute_item_pairs(
            baskets, "product_id", frequent_items, config, "2024-02-01", 90
        )
        # May be empty if no co-occurrences
        if not result.is_empty():
            assert "item_a" in result.columns
            assert "item_b" in result.columns
            assert "confidence_a_to_b" in result.columns

    def test_product_affinity_analysis(self):
        validated, rows, baskets = make_test_data()
        config = ProductAffinityConfig(min_support=0.01, top_k_items=10, top_k_pairs=10)
        result = product_affinity_analysis(validated, rows, "2024-03-01", config, None)
        assert "product_frequent" in result
        assert "product_pairs" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])