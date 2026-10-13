"""Core unit tests for retail analytics pipeline."""

from datetime import date

import polars as pl
import pytest

from retail_analytics.core.pipeline.preparation import (
    audit_duplicate_counts,
    normalize_strings,
    parse_numeric_columns,
    validity_mask,
)
from retail_analytics.core.pipeline.stages import (
    build_unavailable_reference_thresholds,
)
from retail_analytics.core.pipeline.utils import (
    VERSION,
    _mask_zero_spend_cols,
    collect,
)
from retail_analytics.core.stats import wilson
from retail_analytics.patronage.patronage import (
    Rules,
    classify,
    label_frame,
    month_bounds,
)


class TestStats:
    """Tests for statistical helpers."""

    def test_wilson_basic(self):
        assert wilson(0, 0) == (None, None)
        low, high = wilson(10, 20)
        assert 0 <= low <= 0.5 <= high <= 1

    def test_safe_ratio_scalar(self):
        from retail_analytics.core.stats import ratio
        assert ratio(10, 5) == 2.0
        assert ratio(10, 0) is None
        assert ratio(10, -1) is None
        assert ratio(None, 5) is None


class TestPipelineUtils:
    """Tests for pipeline utilities."""

    def test_version_constant(self):
        assert VERSION == "reconstructed-1.3"

    def test_mask_zero_spend_cols(self):
        df = pl.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "spend_365d": [100.0, 0.0, -10.0],
            "category_hhi": [0.5, 0.3, 0.2],
            "category_entropy": [1.0, 0.5, 0.3],
        }).lazy()
        result = collect(_mask_zero_spend_cols(df))
        assert result["category_hhi"][0] == 0.5
        assert result["category_hhi"][1] is None
        assert result["category_hhi"][2] is None

    def test_collect(self):
        df = pl.DataFrame({"a": [1, 2, 3]}).lazy()
        result = collect(df)
        assert result.shape == (3, 1)


class TestPreparation:
    """Tests for shared preparation utilities."""

    def test_normalize_strings(self):
        df = pl.DataFrame({
            "customer_id": [" C1 ", "C2", None],
            "transaction_id": [" T1 ", "T2", ""],
            "product_id": ["P1", " P2 ", ""],
            "department": [" DEPT1 ", "", None],
            "category": [" CAT1 ", "CAT2", ""],
            "product_description": ["desc", None, ""],
        }).lazy()
        result = collect(normalize_strings(df))
        assert result["customer_id"][0] == "C1"
        assert result["customer_id"][2] is None
        assert result["department"][1] == "__UNKNOWN__"
        assert result["department"][2] == "__UNKNOWN__"
        assert result["product_description"][1] == ""

    def test_parse_numeric_columns(self):
        schema = {"price": pl.String, "quantity": pl.String}
        df = pl.DataFrame({
            "price": ["10.5", "20.0", "abc"],
            "quantity": ["2", "3", "4"],
        }).lazy()
        result = collect(parse_numeric_columns(df, schema, decimal_comma=False, mode="unit"))
        assert result["price"][0] == 10.5
        assert result["quantity"][0] == 2.0
        assert result["value"][0] == 21.0

    def test_parse_numeric_decimal_comma(self):
        schema = {"price": pl.String, "quantity": pl.String}
        df = pl.DataFrame({
            "price": ["10,5", "20,0"],
            "quantity": ["2", "3"],
        }).lazy()
        result = collect(parse_numeric_columns(df, schema, decimal_comma=True, mode="unit"))
        assert result["price"][0] == 10.5
        assert result["value"][0] == 21.0

    def test_validity_mask(self):
        df = pl.DataFrame({
            "customer_id": ["C1", "C2", "C3", "C4", "C5"],
            "transaction_id": ["T1", "T2", "T3", "T4", "T5"],
            "product_id": ["P1", "P2", "P3", "P4", "P5"],
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
            "price": [10.0, 20.0, -5.0, 15.0, 25.0],
            "quantity": [1.0, 2.0, 0.0, -1.0, 2.0],
            "value": [10.0, 40.0, 0.0, -15.0, 50.0],
        }).lazy()
        mask = validity_mask(df)
        result = collect(df.with_columns(mask.alias("valid")))
        # Row 0: valid (price>0, qty>0, value finite)
        assert result["valid"][0] is True
        # Row 1: valid (price>0, qty>0, value finite)
        assert result["valid"][1] is True
        # Row 2: invalid (negative price, zero quantity)
        assert result["valid"][2] is False
        # Row 3: invalid (negative quantity)
        assert result["valid"][3] is False
        # Row 4: valid (price>0, qty>0, value finite)
        assert result["valid"][4] is True

    def test_audit_duplicate_counts(self):
        df = pl.DataFrame({
            "customer_id": ["C1", "C1", "C2"],
            "transaction_id": ["T1", "T1", "T2"],
            "product_id": ["P1", "P1", "P2"],
            "value": [10.0, 10.0, 20.0],
        }).lazy()
        valid_df = df.filter(pl.col("value").is_not_null())
        count, value = audit_duplicate_counts(
            df, ["customer_id", "transaction_id", "product_id"],
            value_col="value", valid_frame=valid_df
        )
        assert count == 1  # one duplicate row
        assert value == 10.0  # value of duplicate


class TestStages:
    """Tests for shared pipeline stages."""

    def test_build_unavailable_reference_thresholds(self):
        from retail_analytics.core.thresholds import default_policy
        policy = default_policy()
        ref_info = {"selected": False}
        thresholds = build_unavailable_reference_thresholds(policy, ref_info)
        assert thresholds["value_low"] is None
        assert thresholds["quantile_interpolation"] == "linear"
        assert thresholds["reference_selection"] == ref_info


class TestPatronage:
    """Tests for patronage module."""

    def test_month_bounds_basic(self):
        start, end, count = month_bounds(date(2024, 1, 15), date(2024, 6, 20))
        assert start == date(2024, 2, 1)
        assert end == date(2024, 5, 1)
        assert count == 4

    def test_month_bounds_first_day(self):
        start, end, count = month_bounds(date(2024, 1, 1), date(2024, 3, 1))
        assert start == date(2024, 1, 1)
        assert end == date(2024, 2, 1)
        assert count == 2

    def test_month_bounds_cutoff_last_day(self):
        start, end, count = month_bounds(date(2024, 1, 15), date(2024, 3, 31))
        assert start == date(2024, 2, 1)
        assert end == date(2024, 3, 1)
        assert count == 2

    def test_classify_capture_unconfirmed(self):
        rules = Rules()
        row = {"days_history": 10, "tenure": 200, "capture_confirmed": False}
        assert classify(row, rules, capture_confirmed=False) == "capture_unconfirmed"

    def test_classify_first_observed(self):
        rules = Rules()
        row = {"days_history": 1, "tenure": 50, "capture_confirmed": True}
        assert classify(row, rules, capture_confirmed=True) == "first_observed"

    def test_classify_long_observed_one_day(self):
        rules = Rules()
        row = {"days_history": 1, "tenure": 120, "capture_confirmed": True}
        assert classify(row, rules, capture_confirmed=True) == "long_observed_one_day"

    def test_classify_recent_repeat(self):
        rules = Rules()
        row = {"days_history": 10, "tenure": 50, "capture_confirmed": True}
        assert classify(row, rules, capture_confirmed=True) == "recent_repeat"

    def test_classify_insufficient_history(self):
        rules = Rules()
        row = {"days_history": 2, "tenure": 120, "capture_confirmed": True}
        assert classify(row, rules, capture_confirmed=True) == "insufficient_history"

    def test_classify_recurring_overdue(self):
        rules = Rules()
        row = {
            "days_history": 10, "tenure": 120,
            "recency": 30, "median_gap": 14.0,
            "gap_cv": 0.3, "gap_count": 8, "days_365d": 15,
            "complete_months": 6, "active_complete_months": 5,
            "active_period_share": 0.8, "resume_events": 0,
            "capture_confirmed": True
        }
        assert classify(row, rules, capture_confirmed=True) == "recurring_overdue"

    def test_label_frame_parity(self):
        """Test that label_frame produces same labels as classify."""
        rules = Rules()
        rows = [
            {"customer_id": "C1", "days_history": 10, "tenure": 200, "recency": 10,
             "gap_cv": 0.3, "median_gap": 14.0, "gap_count": 8, "days_365d": 15,
             "complete_months": 6, "active_complete_months": 5,
             "active_period_share": 0.8, "resume_events": 0,
             "observed_categories_365d": 3, "recurring_categories_365d": 2},
            {"customer_id": "C2", "days_history": 1, "tenure": 50, "recency": 10,
             "gap_cv": None, "median_gap": None, "gap_count": 0, "days_365d": 1,
             "complete_months": 0, "active_complete_months": 0,
             "active_period_share": 0.0, "resume_events": 0,
             "observed_categories_365d": 0, "recurring_categories_365d": 0},
            {"customer_id": "C3", "days_history": 1, "tenure": 120, "recency": 10,
             "gap_cv": None, "median_gap": None, "gap_count": 0, "days_365d": 1,
             "complete_months": 0, "active_complete_months": 0,
             "active_period_share": 0.0, "resume_events": 0,
             "observed_categories_365d": 0, "recurring_categories_365d": 0},
            {"customer_id": "C4", "days_history": 0, "tenure": 50, "recency": 10,
             "gap_cv": None, "median_gap": None, "gap_count": 0, "days_365d": 0,
             "complete_months": 0, "active_complete_months": 0,
             "active_period_share": 0.0, "resume_events": 0,
             "observed_categories_365d": 0, "recurring_categories_365d": 0},
            {"customer_id": "C5", "days_history": 2, "tenure": 120, "recency": 10,
             "gap_cv": 0.5, "median_gap": 15.0, "gap_count": 2, "days_365d": 2,
             "complete_months": 1, "active_complete_months": 1,
             "active_period_share": 0.5, "resume_events": 0,
             "observed_categories_365d": 1, "recurring_categories_365d": 0},
        ]
        df = pl.DataFrame(rows)
        labeled = label_frame(df, rules, capture_confirmed=True)
        scalar_labels = [classify(r, rules, True) for r in rows]
        vector_labels = labeled["observed_patronage"].to_list()
        assert scalar_labels == vector_labels


class TestPatronageEdgeCases:
    """Tests for edge cases in patronage."""

    @pytest.fixture
    def rules(self):
        return Rules()

    def make_frame(self, **kwargs) -> pl.DataFrame:
        defaults = {
            "customer_id": ["C1"],
            "days_history": [10],
            "tenure": [200],
            "recency": [10],
            "gap_cv": [0.3],
            "median_gap": [14.0],
            "gap_count": [8],
            "days_365d": [15],
            "complete_months": [6],
            "active_complete_months": [5],
            "active_period_share": [0.8],
            "resume_events": [0],
            "observed_categories_365d": [3],
            "recurring_categories_365d": [2],
        }
        defaults.update({k: [v] for k, v in kwargs.items()})
        return pl.DataFrame(defaults)

    def test_null_gap_cv(self, rules):
        df = self.make_frame(gap_cv=None)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_nan_gap_cv(self, rules):
        df = self.make_frame(gap_cv=float("nan"))
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_infinite_median_gap(self, rules):
        df = self.make_frame(median_gap=float("inf"))
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_boundary_active_share(self, rules):
        df = self.make_frame(active_period_share=rules.active_share,
                             gap_cv=0.3, median_gap=14.0, gap_count=8, days_365d=15)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_recurring"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
