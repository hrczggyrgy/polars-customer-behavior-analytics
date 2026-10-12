"""Parity tests for patronage scalar (classify) vs vectorized (label_frame) implementations."""

import math
from datetime import date, timedelta

import polars as pl
import pytest

from retail_analytics.patronage.patronage import (
    Rules,
    classify,
    label_frame,
    month_bounds,
)


class TestPatronageParity:
    """Ensure classify() and label_frame() produce identical labels for identical inputs."""

    @pytest.fixture
    def rules(self):
        return Rules()

    def make_row(self, **kwargs) -> dict:
        """Create a complete row with defaults, overridden by kwargs.
        
        Defaults are chosen to NOT trigger 'recurring_overdue' (recency/gap < 2.0):
        recency=10, median_gap=14.0 => 10/14 = 0.71 < 2.0
        """
        defaults = {
            "customer_id": "C1",
            "days_history": 10,
            "tenure": 200,
            "recency": 10,
            "gap_cv": 0.3,
            "median_gap": 14.0,
            "gap_count": 8,
            "days_365d": 15,
            "complete_months": 6,
            "active_complete_months": 5,
            "active_period_share": 0.8,
            "resume_events": 0,
            "observed_categories_365d": 3,
            "recurring_categories_365d": 2,
        }
        defaults.update(kwargs)
        return defaults

    def test_capture_unconfirmed(self, rules):
        row = self.make_row(capture_confirmed=False)
        assert classify(row, rules, capture_confirmed=False) == "capture_unconfirmed"

        df = pl.DataFrame([self.make_row(capture_confirmed=False)])
        labeled = label_frame(df, rules, capture_confirmed=False)
        assert labeled["observed_patronage"][0] == "capture_unconfirmed"

    def test_first_observed(self, rules):
        row = self.make_row(days_history=1, tenure=50, capture_confirmed=True)
        assert classify(row, rules, capture_confirmed=True) == "first_observed"

        df = pl.DataFrame([self.make_row(days_history=1, tenure=50, capture_confirmed=True)])
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "first_observed"

    def test_long_observed_one_day(self, rules):
        row = self.make_row(days_history=1, tenure=120, capture_confirmed=True)
        assert classify(row, rules, capture_confirmed=True) == "long_observed_one_day"

        df = pl.DataFrame([self.make_row(days_history=1, tenure=120, capture_confirmed=True)])
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "long_observed_one_day"

    def test_recent_repeat(self, rules):
        row = self.make_row(tenure=50, capture_confirmed=True)
        assert classify(row, rules, capture_confirmed=True) == "recent_repeat"

        df = pl.DataFrame([self.make_row(tenure=50, capture_confirmed=True)])
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "recent_repeat"

    def test_insufficient_history(self, rules):
        # Not enough days
        row = self.make_row(days_history=2, tenure=120, capture_confirmed=True)
        assert classify(row, rules, capture_confirmed=True) == "insufficient_history"

        # Not enough complete months
        row = self.make_row(complete_months=1, tenure=120, capture_confirmed=True)
        assert classify(row, rules, capture_confirmed=True) == "insufficient_history"

        # Not enough active complete months
        row = self.make_row(active_complete_months=1, tenure=120, capture_confirmed=True)
        assert classify(row, rules, capture_confirmed=True) == "insufficient_history"

    def test_recurring_overdue(self, rules):
        row = self.make_row(
            recency=30,  # recency/median_gap = 30/14 = 2.14 > 2.0
            median_gap=14.0,
            gap_cv=0.3,
            gap_count=8,
            days_365d=15,
            capture_confirmed=True,
        )
        assert classify(row, rules, capture_confirmed=True) == "recurring_overdue"

        df = pl.DataFrame([row])
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "recurring_overdue"

    def test_reactivated_repeat(self, rules):
        row = self.make_row(
            resume_events=2,
            recency=10,  # recency/gap = 10/14 = 0.71 < 2.0 (not overdue)
            capture_confirmed=True,
        )
        assert classify(row, rules, capture_confirmed=True) == "reactivated_repeat"

        df = pl.DataFrame([row])
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "reactivated_repeat"

    def test_established_recurring(self, rules):
        row = self.make_row(
            active_period_share=0.7,
            gap_cv=0.3,
            median_gap=14.0,
            gap_count=8,
            days_365d=15,
            capture_confirmed=True,
        )
        assert classify(row, rules, capture_confirmed=True) == "established_recurring"

        df = pl.DataFrame([row])
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_recurring"

    def test_established_occasional(self, rules):
        row = self.make_row(
            active_period_share=0.5,  # Below active_share threshold of 0.6
            gap_cv=0.3,
            median_gap=14.0,
            gap_count=8,
            days_365d=15,
            capture_confirmed=True,
        )
        assert classify(row, rules, capture_confirmed=True) == "established_occasional"

        df = pl.DataFrame([row])
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_batch_parity(self, rules):
        """Test multiple rows at once for full parity."""
        rows = [
            self.make_row(capture_confirmed=False),
            self.make_row(days_history=1, tenure=50, capture_confirmed=True),
            self.make_row(days_history=1, tenure=120, capture_confirmed=True),
            self.make_row(tenure=50, capture_confirmed=True),
            self.make_row(days_history=2, tenure=120, capture_confirmed=True),
            self.make_row(recency=30, median_gap=14.0, gap_cv=0.3, gap_count=8, days_365d=15, capture_confirmed=True),
            self.make_row(resume_events=2, recency=10, capture_confirmed=True),
            self.make_row(active_period_share=0.7, gap_cv=0.3, median_gap=14.0, gap_count=8, days_365d=15, capture_confirmed=True),
            self.make_row(active_period_share=0.5, gap_cv=0.3, median_gap=14.0, gap_count=8, days_365d=15, capture_confirmed=True),
        ]

        scalar_labels = [classify(r, rules, r.get("capture_confirmed", True)) for r in rows]
        df = pl.DataFrame(rows)
        labeled = label_frame(df, rules, capture_confirmed=True)
        vector_labels = labeled["observed_patronage"].to_list()

        # Note: capture_confirmed is a global parameter in label_frame
        # We need to handle the first row specially
        assert scalar_labels[1:] == vector_labels[1:]


class TestPatronageEdgeCases:
    """Test edge cases with missing, null, NaN, and infinite values."""

    @pytest.fixture
    def rules(self):
        return Rules()

    def make_frame(self, **kwargs) -> pl.DataFrame:
        """Create a frame with one row, using defaults for unspecified columns.
        
        Defaults chosen to NOT trigger 'recurring_overdue':
        recency=10, median_gap=14.0 => 10/14 = 0.71 < 2.0
        """
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
        """Null gap_cv should not be regular."""
        df = self.make_frame(gap_cv=None)
        labeled = label_frame(df, rules, capture_confirmed=True)
        # Should fall through to established_occasional (not regular)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_nan_gap_cv(self, rules):
        """NaN gap_cv should not be regular."""
        df = self.make_frame(gap_cv=float("nan"))
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_infinite_gap_cv(self, rules):
        """Infinite gap_cv should not be regular."""
        df = self.make_frame(gap_cv=float("inf"))
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_negative_gap_cv(self, rules):
        """Negative gap_cv should not be regular."""
        df = self.make_frame(gap_cv=-0.1)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_gap_cv_above_threshold(self, rules):
        """gap_cv above max_gap_cv should not be regular."""
        df = self.make_frame(gap_cv=0.8)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_null_median_gap(self, rules):
        """Null median_gap should not be regular."""
        df = self.make_frame(median_gap=None)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_nan_median_gap(self, rules):
        """NaN median_gap should not be regular."""
        df = self.make_frame(median_gap=float("nan"))
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_zero_median_gap(self, rules):
        """Zero median_gap should not be regular."""
        df = self.make_frame(median_gap=0.0)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_negative_median_gap(self, rules):
        """Negative median_gap should not be regular."""
        df = self.make_frame(median_gap=-1.0)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_infinite_median_gap(self, rules):
        """Infinite median_gap should not be regular."""
        df = self.make_frame(median_gap=float("inf"))
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_null_active_period_share(self, rules):
        """Null active_period_share should not be established_recurring."""
        df = self.make_frame(active_period_share=None)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_occasional"

    def test_insufficient_days_history(self, rules):
        """days_history < min_days should be insufficient_history."""
        df = self.make_frame(days_history=2)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "insufficient_history"

    def test_insufficient_complete_months(self, rules):
        """complete_months < min_complete_months should be insufficient_history."""
        df = self.make_frame(complete_months=2)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "insufficient_history"

    def test_insufficient_active_complete_months(self, rules):
        """active_complete_months < min_active_months should be insufficient_history."""
        df = self.make_frame(active_complete_months=1)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "insufficient_history"

    def test_boundary_tenure(self, rules):
        """Tenure exactly at min_tenure threshold."""
        df = self.make_frame(tenure=rules.min_tenure, days_history=10)
        labeled = label_frame(df, rules, capture_confirmed=True)
        # Should not be recent_repeat
        assert labeled["observed_patronage"][0] != "recent_repeat"

    def test_boundary_gap_cv(self, rules):
        """gap_cv exactly at max_gap_cv threshold."""
        df = self.make_frame(gap_cv=rules.max_gap_cv, median_gap=14.0, gap_count=8, days_365d=15)
        labeled = label_frame(df, rules, capture_confirmed=True)
        # Should be regular
        assert labeled["patronage_cadence_eligible"][0] == True

    def test_boundary_active_share(self, rules):
        """active_period_share exactly at active_share threshold."""
        df = self.make_frame(active_period_share=rules.active_share, gap_cv=0.3, median_gap=14.0, gap_count=8, days_365d=15)
        labeled = label_frame(df, rules, capture_confirmed=True)
        assert labeled["observed_patronage"][0] == "established_recurring"


class TestMonthBounds:
    """Tests for month_bounds function.
    
    month_bounds returns (start, end, count) where:
    - start = first day of month AFTER first's month (unless first.day == 1)
    - end = first day of cutoff's month (if cutoff is last day of month), 
             otherwise first day of PREVIOUS month
    - count = number of complete months between start and end (inclusive)
    """

    def test_basic(self):
        first = date(2024, 1, 15)
        cutoff = date(2024, 6, 20)
        start, end, count = month_bounds(first, cutoff)
        # first=Jan 15 -> start=Feb 1
        # cutoff=Jun 20 (not last day) -> end=May 1
        # months: Feb, Mar, Apr, May = 4
        assert start == date(2024, 2, 1)
        assert end == date(2024, 5, 1)
        assert count == 4

    def test_first_day_of_month(self):
        first = date(2024, 1, 1)
        cutoff = date(2024, 3, 1)
        start, end, count = month_bounds(first, cutoff)
        # first=Jan 1 -> start=Jan 1
        # cutoff=Mar 1 (not last day) -> end=Feb 1
        # months: Jan, Feb = 2
        assert start == date(2024, 1, 1)
        assert end == date(2024, 2, 1)
        assert count == 2

    def test_cutoff_last_day(self):
        first = date(2024, 1, 15)
        cutoff = date(2024, 3, 31)
        start, end, count = month_bounds(first, cutoff)
        # first=Jan 15 -> start=Feb 1
        # cutoff=Mar 31 (last day) -> end=Mar 1
        # months: Feb, Mar = 2
        assert start == date(2024, 2, 1)
        assert end == date(2024, 3, 1)
        assert count == 2

    def test_cutoff_not_last_day(self):
        first = date(2024, 1, 15)
        cutoff = date(2024, 3, 15)
        start, end, count = month_bounds(first, cutoff)
        # first=Jan 15 -> start=Feb 1
        # cutoff=Mar 15 (not last day) -> end=Feb 1
        # months: Feb = 1
        assert start == date(2024, 2, 1)
        assert end == date(2024, 2, 1)
        assert count == 1

    def test_cutoff_before_first(self):
        first = date(2024, 6, 1)
        cutoff = date(2024, 1, 15)
        with pytest.raises(ValueError):
            month_bounds(first, cutoff)

    def test_same_month(self):
        first = date(2024, 3, 15)
        cutoff = date(2024, 3, 20)
        start, end, count = month_bounds(first, cutoff)
        # first=Mar 15 -> start=Apr 1
        # cutoff=Mar 20 (not last day) -> end=Feb 1 (previous month)
        # start > end -> count = 0
        assert start == date(2024, 4, 1)
        assert end == date(2024, 2, 1)
        assert count == 0

    def test_feb_leap_year(self):
        first = date(2024, 2, 15)
        cutoff = date(2024, 3, 15)
        start, end, count = month_bounds(first, cutoff)
        # first=Feb 15 -> start=Mar 1
        # cutoff=Mar 15 (not last day) -> end=Feb 1
        # start > end -> count = 0
        assert start == date(2024, 3, 1)
        assert end == date(2024, 2, 1)
        assert count == 0

    def test_full_month_range(self):
        """Test when first is 1st and cutoff is last day."""
        first = date(2024, 1, 1)
        cutoff = date(2024, 3, 31)
        start, end, count = month_bounds(first, cutoff)
        # first=Jan 1 -> start=Jan 1
        # cutoff=Mar 31 (last day) -> end=Mar 1
        # months: Jan, Feb, Mar = 3
        assert start == date(2024, 1, 1)
        assert end == date(2024, 3, 1)
        assert count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])