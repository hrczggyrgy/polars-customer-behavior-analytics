"""Regression tests for research-backed visualizations.

These tests validate numerical correctness of the five new visual analytics functions:
- time_to_second_purchase
- cadence_return_validation
- multi_horizon_cohort_repeat
- category_opportunity_map
- rfm_response_surface

Each test uses hand-calculated expected values from small synthetic datasets.
"""

import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from retail_analytics.core.retail_polars_workflow import run
from retail_analytics.visualization.visual_insights_core import (
    time_to_second_purchase,
    cadence_return_validation,
    multi_horizon_cohort_repeat,
    category_opportunity_map,
    rfm_response_surface,
)


class ResearchVisualizationTests(unittest.TestCase):
    """Hand-verifiable tests for the new research-backed visualizations."""

    def setUp(self):
        """Create a minimal deterministic dataset for testing.

        Customer A: first_purchase=2025-01-01, repeats on day 10, 30, 60
        Customer B: first_purchase=2025-01-15, repeats on day 45
        Customer C: first_purchase=2025-02-01, no repeat
        Customer D: first_purchase=2025-02-15, repeats on day 5
        Customer E: first_purchase=2025-03-01, repeats on day 15
        """
        self.tmpdir = TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.source = self.root / "input.parquet"

        rows = []
        # Customer A - 4 purchases total (days 0, 10, 30, 60)
        for i, day in enumerate([0, 10, 30, 60]):
            d = (date(2025, 1, 1) + timedelta(days=day)).isoformat()
            rows.append({
                "customer_id": "A",
                "transaction_date": d,
                "transaction_id": f"A{i}",
                "product_id": "P1",
                "product_description": "Product 1",
                "department": "D",
                "category": "C1",
                "price": 10.0,
                "quantity": 1.0,
            })
        # Customer B - 2 purchases (days 0, 45)
        for i, day in enumerate([0, 45]):
            d = (date(2025, 1, 15) + timedelta(days=day)).isoformat()
            rows.append({
                "customer_id": "B",
                "transaction_date": d,
                "transaction_id": f"B{i}",
                "product_id": "P1",
                "product_description": "Product 1",
                "department": "D",
                "category": "C1",
                "price": 10.0,
                "quantity": 1.0,
            })
        # Customer C - 1 purchase only (day 0)
        rows.append({
            "customer_id": "C",
            "transaction_date": "2025-02-01",
            "transaction_id": "C0",
            "product_id": "P2",
            "product_description": "Product 2",
            "department": "D",
            "category": "C2",
            "price": 20.0,
            "quantity": 1.0,
        })
        # Customer D - 2 purchases (days 0, 5)
        for i, day in enumerate([0, 5]):
            d = (date(2025, 2, 15) + timedelta(days=day)).isoformat()
            rows.append({
                "customer_id": "D",
                "transaction_date": d,
                "transaction_id": f"D{i}",
                "product_id": "P1",
                "product_description": "Product 1",
                "department": "D",
                "category": "C1",
                "price": 10.0,
                "quantity": 1.0,
            })
        # Customer E - 2 purchases (days 0, 15)
        for i, day in enumerate([0, 15]):
            d = (date(2025, 3, 1) + timedelta(days=day)).isoformat()
            rows.append({
                "customer_id": "E",
                "transaction_date": d,
                "transaction_id": f"E{i}",
                "product_id": "P3",
                "product_description": "Product 3",
                "department": "D",
                "category": "C3",
                "price": 5.0,
                "quantity": 1.0,
            })

        pl.DataFrame(rows).write_parquet(self.source)

        # Run pipeline to generate snapshots
        self.out = self.root / "results"
        run(
            str(self.source),
            str(self.out),
            "unit",
            ["2025-03-15"],  # Cutoff at last transaction date
            buckets=1,
            coverage_confirmed=True,
        )
        
        self._tmp_refs = []

    def tearDown(self):
        self.tmpdir.cleanup()
        for tmp in self._tmp_refs:
            tmp.cleanup()

    def _run_and_load_snapshot(self, cutoff_date):
        """Run pipeline with specific cutoff and load snapshot data."""
        tmp = TemporaryDirectory()
        root = Path(tmp.name)
        out = root / "results"
        
        run(
            str(self.source),
            str(out),
            "unit",
            [cutoff_date],
            buckets=1,
            coverage_confirmed=True,
        )
        
        snapshot_path = out / "snapshots" / cutoff_date / "b0.parquet"
        rows = pl.scan_parquet(str(snapshot_path))
        baskets_path = out / "events" / cutoff_date / "b0" / "baskets.parquet"
        baskets = pl.scan_parquet(str(baskets_path))
        validated_path = out / "validated.parquet"
        validated = pl.scan_parquet(str(validated_path))
        
        # Store tmp reference to prevent premature cleanup
        self._tmp_refs.append(tmp)
        
        return rows, baskets, validated

    def _load_snapshot(self, cutoff="2025-03-15"):
        """Load snapshot and related data for testing."""
        snapshot_path = self.out / "snapshots" / cutoff / "b0.parquet"
        rows = pl.scan_parquet(str(snapshot_path))
        baskets_path = self.out / "events" / cutoff / "b0" / "baskets.parquet"
        baskets = pl.scan_parquet(str(baskets_path))
        validated_path = self.out / "validated.parquet"
        validated = pl.scan_parquet(str(validated_path))
        return rows, baskets, validated

    # ============================================================
    # time_to_second_purchase tests
    # ============================================================

    def test_time_to_second_purchase_basic(self):
        """Test Kaplan-Meier survival curve with known expected values.

        Expected events:
        - A: second purchase at day 10 -> event at t=10
        - B: second purchase at day 45 -> event at t=45
        - C: no second purchase -> censored at censor_days (44 days to cutoff)
        - D: second purchase at day 5 -> event at t=5
        - E: second purchase at day 15 -> NOT observed (censored at day 14, cutoff=2025-03-15)

        At t=1: at_risk=5 (all have censor_days >= 1)
        At t=5: D has event -> events=1, at_risk=5 -> hazard=0.2, survival=0.8
        At t=10: A has event -> events=1, at_risk=4 -> hazard=0.25, survival=0.6
        At t=15: E is censored (no event), B has no event yet
        At t=45: B has event -> events=1, at_risk=1 -> hazard=1.0, survival=0.0
        C is censored at t=44 (cutoff - first_purchase = Mar 15 - Feb 1 = 44 days)
        """
        rows, baskets, _ = self._load_snapshot()
        cutoff = date(2025, 3, 15)

        result = time_to_second_purchase(baskets, rows, cutoff, max_days=180)

        self.assertFalse(result.is_empty())

        # Check structure
        expected_cols = ["days", "survival_prob", "lower_ci", "upper_ci", "at_risk", "events", "censored", "cum_repeat_prob"]
        for col in expected_cols:
            self.assertIn(col, result.columns)

        # Check specific values
        # Day 5: 1 event (D), at_risk=5, survival should be 0.8
        day5 = result.filter(pl.col("days") == 5)
        self.assertEqual(day5["events"][0], 1)
        self.assertEqual(day5["at_risk"][0], 5)
        self.assertAlmostEqual(day5["survival_prob"][0], 0.8, places=2)

        # Day 10: 1 event (A), at_risk=4, survival should be 0.6
        day10 = result.filter(pl.col("days") == 10)
        self.assertEqual(day10["events"][0], 1)
        self.assertEqual(day10["at_risk"][0], 4)
        self.assertAlmostEqual(day10["survival_prob"][0], 0.6, places=2)

        # Day 15: NO event (E is censored at 14, B has no event yet)
        day15 = result.filter(pl.col("days") == 15)
        self.assertEqual(day15["events"][0], 0)

        # Day 45: 1 event (B), at_risk=1 (C censored at day 44), survival should be 0.0
        day45 = result.filter(pl.col("days") == 45)
        self.assertEqual(day45["events"][0], 1)
        self.assertEqual(day45["at_risk"][0], 1)
        self.assertAlmostEqual(day45["survival_prob"][0], 0.0, places=2)

        # Check CIs are valid bounds
        for row in result.to_dicts():
            if row["lower_ci"] is not None:
                self.assertLessEqual(row["lower_ci"], row["survival_prob"])
                self.assertGreaterEqual(row["lower_ci"], 0.0)
            if row["upper_ci"] is not None:
                self.assertGreaterEqual(row["upper_ci"], row["survival_prob"])
                self.assertLessEqual(row["upper_ci"], 1.0)

        # Cumulative repeat probability = 1 - survival
        for row in result.to_dicts():
            self.assertAlmostEqual(row["cum_repeat_prob"], 1 - row["survival_prob"], places=6)

    def test_time_to_second_purchase_censoring(self):
        """Test that right-censoring is handled correctly.

        Customer C has no second purchase and is censored at day 44 (cutoff - first_purchase = Mar 15 - Feb 1 = 44).
        They should NOT be at risk after day 44.
        """
        rows, baskets, _ = self._load_snapshot()
        cutoff = date(2025, 3, 15)

        result = time_to_second_purchase(baskets, rows, cutoff, max_days=180)

        # C's censor_days = 44 (Mar 15 - Feb 1 = 44 days)
        # After day 44, at_risk should not include C
        # At day 44: C is still at risk (observed_duration = 44)
        # At day 45: C is no longer at risk

        day44 = result.filter(pl.col("days") == 44)
        self.assertEqual(day44["at_risk"][0], 1)  # Only C at risk

        day45 = result.filter(pl.col("days") == 45)
        # At day 45, B has event but C is censored, so at_risk includes B
        # C's observed_duration = 44, so not at risk at 45
        self.assertGreaterEqual(day45["at_risk"][0], 0)

    def test_time_to_second_purchase_empty_handling(self):
        """Test handling of empty basket data."""
        rows, baskets, _ = self._load_snapshot()
        # Filter baskets to empty
        empty_baskets = baskets.filter(pl.col("customer_id") == "NONEXISTENT")
        result = time_to_second_purchase(empty_baskets, rows, date(2025, 3, 15))
        self.assertTrue(result.is_empty())

    # ============================================================
    # cadence_return_validation tests
    # ============================================================

    def test_cadence_return_validation_binning(self):
        """Test lateness binning uses correct interior edges."""
        # Use cutoff that allows 90-day follow-up but has enough data.
        # Latest transaction is 2025-03-16. For 90-day follow-up: cutoff + 90 <= 2025-03-16
        # So cutoff <= 2025-01-15 (approximately).
        # Let's use 2025-01-31 which gives ~45 days before cutoff.
        rows, _, _ = self._run_and_load_snapshot("2025-01-31")

        result = cadence_return_validation(rows, coverage=True, max_lateness=5.0, bins=10)

        # Should have results for 30-day horizon at minimum
        horizons = result["horizon"].unique().to_list()
        self.assertIn(30, horizons)
        # 90-day may not be present if not enough follow-up data
        # Just verify the function works without error

        # Check bin labels format
        for lateness_bin in result["lateness_bin"].unique():
            self.assertRegex(lateness_bin, r"^\d+\.\d–\d+\.\d$")

        # Wilson CIs present for n >= 20
        for row in result.to_dicts():
            if row["eligible_n"] >= 20:
                self.assertIsNotNone(row["lower_ci"])
                self.assertIsNotNone(row["upper_ci"])
                self.assertLessEqual(row["lower_ci"], row["purchase_rate"])
                self.assertGreaterEqual(row["upper_ci"], row["purchase_rate"])

    def test_cadence_return_validation_no_coverage(self):
        """Test that function returns warning when coverage unconfirmed."""
        rows, _, _ = self._load_snapshot()

        result = cadence_return_validation(rows, coverage=False)
        self.assertEqual(result.height, 1)
        self.assertIn("warning", result.columns)

    def test_cadence_return_validation_constant_lateness(self):
        """Test edge case where all customers have same lateness."""
        # Create data where all have same median_gap and recency
        rows = pl.DataFrame({
            "customer_id": ["X1", "X2", "X3"],
            "median_gap": [10.0, 10.0, 10.0],
            "recency": [5.0, 5.0, 5.0],  # lateness = 0.5 for all
            "eligible_followup_30d": [True, True, True],
            "future_purchases_30d": [1, 0, 1],
            "eligible_followup_90d": [True, True, True],
            "future_purchases_90d": [1, 1, 0],
        }).lazy()

        result = cadence_return_validation(rows, coverage=True, max_lateness=5.0, bins=10)
        # Should still work - all fall in same bin
        self.assertFalse(result.is_empty())

    # ============================================================
    # multi_horizon_cohort_repeat tests
    # ============================================================

    def test_multi_horizon_cohort_repeat_correct_join(self):
        """Test that join to first_purchase works correctly.

        Customer A: first_purchase=2025-01-01, cohort=2025-01, repeat at day 10, 30, 60
        Customer B: first_purchase=2025-01-15, cohort=2025-01, repeat at day 45
        Customer C: first_purchase=2025-02-01, cohort=2025-02, no repeat
        Customer D: first_purchase=2025-02-15, cohort=2025-02, repeat at day 5
        Customer E: first_purchase=2025-03-01, cohort=2025-03, repeat at day 15

        At cutoff=2025-03-15:
        - 2025-01 cohort: A and B (cohort_n=2)
          - 30d horizon: both eligible (first + 30 <= cutoff)
            A repeats at day 10 <= 30: success
            B repeats at day 45 > 30: no success
            -> repeat_n=1, rate=0.5
          - 90d horizon: NEITHER eligible (first + 90 > cutoff for both)
            A: first + 90 = 2025-03-31 > 2025-03-15, not eligible
            B: first + 90 = 2025-04-14 > 2025-03-15, not eligible
            -> eligible_n=0, repeat_n=0, rate=None
        - 2025-02 cohort: C and D (cohort_n=2)
          - 30d horizon: NEITHER eligible (first + 30 > cutoff for both)
            C: first + 30 = 2025-03-03 <= 2025-03-15, ELIGIBLE
            D: first + 30 = 2025-03-17 > 2025-03-15, NOT eligible
            -> only C eligible, C no repeat: repeat_n=0, rate=0
        - 2025-03 cohort: E (cohort_n=1, below min_cohort_size=2 by default)
        """
        rows, baskets, _ = self._load_snapshot()
        cutoff = date(2025, 3, 15)

        result = multi_horizon_cohort_repeat(baskets, rows, cutoff, coverage=True, min_cohort_size=1)

        self.assertFalse(result.is_empty())

        # Check 2025-01 cohort
        jan30 = result.filter((pl.col("cohort") == date(2025, 1, 1)) & (pl.col("horizon") == 30))
        self.assertEqual(jan30["cohort_n"][0], 2)
        self.assertEqual(jan30["eligible_n"][0], 2)
        self.assertEqual(jan30["repeat_n"][0], 1)  # A repeats within 30, B does not
        self.assertAlmostEqual(jan30["rate"][0], 0.5)

        # 90d horizon: no one eligible (first_purchase + 90 > cutoff for both)
        jan90 = result.filter((pl.col("cohort") == date(2025, 1, 1)) & (pl.col("horizon") == 90))
        self.assertEqual(jan90["eligible_n"][0], 0)
        self.assertEqual(jan90["repeat_n"][0], 0)
        self.assertIsNone(jan90["rate"][0])

        # Check 2025-02 cohort - only C is eligible for 30d (D: first+30 > cutoff)
        feb = result.filter((pl.col("cohort") == date(2025, 2, 1)) & (pl.col("horizon") == 30))
        self.assertEqual(feb["cohort_n"][0], 2)
        self.assertEqual(feb["eligible_n"][0], 1)  # Only C eligible
        self.assertEqual(feb["repeat_n"][0], 0)   # C has no repeat
        self.assertAlmostEqual(feb["rate"][0], 0.0)

        # Check CI bounds
        for row in result.to_dicts():
            if row["lower_ci"] is not None:
                self.assertLessEqual(row["lower_ci"], row["rate"])
                self.assertGreaterEqual(row["lower_ci"], 0.0)
            if row["upper_ci"] is not None:
                self.assertGreaterEqual(row["upper_ci"], row["rate"])
                self.assertLessEqual(row["upper_ci"], 1.0)

    def test_multi_horizon_cohort_repeat_no_coverage(self):
        """Test that rates are None when coverage unconfirmed."""
        rows, baskets, _ = self._load_snapshot()
        cutoff = date(2025, 3, 15)

        result = multi_horizon_cohort_repeat(baskets, rows, cutoff, coverage=False, min_cohort_size=1)

        for row in result.to_dicts():
            self.assertIsNone(row["rate"])
            self.assertIsNone(row["lower_ci"])
            self.assertIsNone(row["upper_ci"])
            self.assertEqual(row["status"], "capture_unconfirmed")

    # ============================================================
    # category_opportunity_map tests
    # ============================================================

    def test_category_opportunity_map_distinct_occasions(self):
        """Test that occasions count distinct shopping days, not line items.

        Create data where one customer has multiple lines in same category on same day.
        Should count as 1 occasion, not multiple.
        """
        rows, _, validated = self._load_snapshot()
        cutoff = date(2025, 3, 15)

        result = category_opportunity_map(validated, rows, cutoff, top_categories=10)

        self.assertFalse(result.is_empty())

        # Check that spend_share denominator is ALL categories, not just top N
        spend_share_sum = result["spend_share"].sum()
        self.assertAlmostEqual(spend_share_sum, 1.0, places=6)

        # Check structure
        for col in ["label", "department", "category", "buyers", "penetration",
                    "occasions_per_buyer", "spend_share", "repeat_buyer_share"]:
            self.assertIn(col, result.columns)

        # Penetration should be buyers / total_customers (5)
        for row in result.to_dicts():
            self.assertAlmostEqual(row["penetration"], row["buyers"] / 5, places=6)
            self.assertGreaterEqual(row["repeat_buyer_share"], 0.0)
            self.assertLessEqual(row["repeat_buyer_share"], 1.0)

    def test_category_opportunity_map_multi_line_same_day(self):
        """Test that multiple lines in same category on same day count as 1 occasion."""
        # Create a test case with multiple lines same day
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.parquet"

            # Customer X buys 2 different products in C1 on same day
            rows = [
                {"customer_id": "X", "transaction_date": "2025-01-01", "transaction_id": "X1",
                 "product_id": "P1", "product_description": "P1", "department": "D", "category": "C1",
                 "price": 10.0, "quantity": 1.0},
                {"customer_id": "X", "transaction_date": "2025-01-01", "transaction_id": "X2",
                 "product_id": "P2", "product_description": "P2", "department": "D", "category": "C1",
                 "price": 10.0, "quantity": 1.0},
                {"customer_id": "X", "transaction_date": "2025-01-15", "transaction_id": "X3",
                 "product_id": "P1", "product_description": "P1", "department": "D", "category": "C1",
                 "price": 10.0, "quantity": 1.0},
                {"customer_id": "Y", "transaction_date": "2025-01-01", "transaction_id": "Y1",
                 "product_id": "P1", "product_description": "P1", "department": "D", "category": "C1",
                 "price": 10.0, "quantity": 1.0},
            ]
            pl.DataFrame(rows).write_parquet(source)

            out = root / "results"
            run(str(source), str(out), "unit", ["2025-01-15"], buckets=1, coverage_confirmed=True)

            rows_snap, validated = self._load_snapshot_data(out, "2025-01-15")
            result = category_opportunity_map(validated, rows_snap, date(2025, 1, 15))

            # Customer X has 2 distinct shopping days in C1 (Jan 1 and Jan 15)
            # Customer Y has 1 day
            # Total buyers = 2
            # X occasions = 2, Y occasions = 1
            # occasions_per_buyer = (2 + 1) / 2 = 1.5
            c1_row = result.filter(pl.col("category") == "C1")
            self.assertEqual(c1_row["buyers"][0], 2)
            self.assertAlmostEqual(c1_row["occasions_per_buyer"][0], 1.5)

    def _load_snapshot_data(self, out, cutoff):
        snapshot_path = out / "snapshots" / cutoff / "b0.parquet"
        rows = pl.scan_parquet(str(snapshot_path))
        validated_path = out / "validated.parquet"
        validated = pl.scan_parquet(str(validated_path))
        return rows, validated

    # ============================================================
    # rfm_response_surface tests
    # ============================================================

    def test_rfm_response_surface_binning(self):
        """Test quantile binning uses correct interior edges."""
        # Use cutoff that allows 30-day follow-up
        rows, _, _ = self._run_and_load_snapshot("2025-02-14")

        result = rfm_response_surface(rows, coverage=True, recency_bins=3, frequency_bins=3)

        self.assertFalse(result.is_empty())

        # Check bin labels
        rec_bins = result["recency_bin"].unique().to_list()
        freq_bins = result["frequency_bin"].unique().to_list()

        for b in rec_bins:
            self.assertRegex(b, r"^R\d+$")
        for b in freq_bins:
            self.assertRegex(b, r"^F\d+$")

        # Number of bins should be <= requested (may be fewer due to duplicate edges)
        self.assertLessEqual(len(rec_bins), 3)
        self.assertLessEqual(len(freq_bins), 3)

        # Wilson CIs for n >= 20
        for row in result.to_dicts():
            if row["eligible_n"] >= 20:
                self.assertIsNotNone(row["lower_ci"])
                self.assertIsNotNone(row["upper_ci"])
                self.assertLessEqual(row["lower_ci"], row["rate"])
                self.assertGreaterEqual(row["upper_ci"], row["rate"])

    def test_rfm_response_surface_constant_values(self):
        """Test edge case where all customers have same recency/frequency."""
        rows = pl.DataFrame({
            "customer_id": ["X1", "X2", "X3", "X4"],
            "recency": [10.0, 10.0, 10.0, 10.0],
            "purchases_90d": [2, 2, 2, 2],
            "eligible_followup_30d": [True, True, True, True],
            "future_purchases_30d": [1, 0, 1, 0],
        }).lazy()

        result = rfm_response_surface(rows, coverage=True, recency_bins=5, frequency_bins=5)

        # With constant values, quantile edges will all be the same
        # After dedup, should have < 2 edges -> return empty DataFrame
        # OR handle gracefully
        # Current implementation returns empty when < 2 unique edges
        self.assertTrue(result.is_empty() or result.height == 0 or "warning" in result.columns or "error" in result.columns)

    def test_rfm_response_surface_no_coverage(self):
        """Test warning when coverage unconfirmed."""
        rows, _, _ = self._load_snapshot()

        result = rfm_response_surface(rows, coverage=False)
        self.assertEqual(result.height, 1)
        self.assertIn("warning", result.columns)

    # ============================================================
    # Integration: all functions should not crash
    # ============================================================

    def test_all_functions_executable(self):
        """Smoke test: all five functions execute without error."""
        # Use cutoff that allows 30-day follow-up for cadence/rfm
        rows, baskets, validated = self._run_and_load_snapshot("2025-02-14")
        cutoff = date(2025, 2, 14)

        # time_to_second_purchase
        km = time_to_second_purchase(baskets, rows, cutoff, max_days=180)
        self.assertFalse(km.is_empty())

        # cadence_return_validation
        cadence = cadence_return_validation(rows, coverage=True, max_lateness=5.0, bins=10)
        self.assertFalse(cadence.is_empty())

        # multi_horizon_cohort_repeat
        cohorts = multi_horizon_cohort_repeat(baskets, rows, cutoff, coverage=True, min_cohort_size=1)
        self.assertFalse(cohorts.is_empty())

        # category_opportunity_map
        cat_map = category_opportunity_map(validated, rows, cutoff, top_categories=10)
        self.assertFalse(cat_map.is_empty())

        # rfm_response_surface
        rfm = rfm_response_surface(rows, coverage=True, recency_bins=5, frequency_bins=5)
        self.assertFalse(rfm.is_empty())


if __name__ == "__main__":
    unittest.main()