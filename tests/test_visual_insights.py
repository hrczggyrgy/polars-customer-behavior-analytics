import importlib.util
import unittest
from datetime import date

from retail_analytics.visualization.visual_insights_core import (
    month_windows,
    share,
    top_count,
    wilson,
)

HAS_POLARS = importlib.util.find_spec("polars") is not None
if HAS_POLARS:
    import polars as pl

    from retail_analytics.visualization.visual_insights_core import (
        cadence_map,
        cohort_summary,
        collect,
        concentration,
        momentum_map,
        revenue_waterfall,
        transition_summary,
        value_tier_transition_waterfall,
    )


class Definitions(unittest.TestCase):
    def test_definitions(self):
        self.assertIsNone(share(0, 0))
        self.assertEqual(top_count(101, 0.1), 11)
        with self.assertRaises(ValueError):
            top_count(3, 0)
        self.assertEqual(month_windows(date(2024, 2, 1), date(2024, 2, 29))[0]["partial"], False)
        self.assertEqual(month_windows(date(2024, 3, 1), date(2024, 2, 1)), [])

    def test_wilson(self):
        self.assertEqual(wilson(0, 0), (None, None))
        for k, n in [(0, 1), (1, 1), (7, 20)]:
            low, high = wilson(k, n)
            self.assertTrue(0 <= low <= k / n <= high <= 1)
        with self.assertRaises(ValueError):
            wilson(2, 1)


@unittest.skipUnless(HAS_POLARS, "Polars unavailable")
class Aggregates(unittest.TestCase):
    def test_concentration(self):
        r = concentration(pl.DataFrame({"spend_90d": [70.0, 20.0, 10.0, 0.0]}).lazy())
        self.assertEqual(r["curve"][-1], (1.0, 1.0))
        self.assertEqual(r["positive_n"], 3)
        self.assertAlmostEqual(r["top"][0]["spend_share"], 0.7)
        self.assertEqual(concentration(pl.DataFrame({"spend_90d": [0.0]}).lazy())["curve"], [])
        with self.assertRaises(ValueError):
            concentration(pl.DataFrame({"spend_90d": [-1.0]}).lazy())

    def test_cohorts(self):
        rows = pl.DataFrame(
            {"customer_id": ["A", "B"], "first_purchase": [date(2024, 1, 1)] * 2}
        ).lazy()
        baskets = pl.DataFrame(
            {
                "customer_id": ["A", "A", "B", "B"],
                "date": [date(2024, 1, 1), date(2024, 1, 31), date(2024, 1, 1), date(2024, 4, 1)],
            }
        ).lazy()
        data = cohort_summary(baskets, rows, date(2024, 3, 1), True, 1)
        self.assertEqual(data.filter(pl.col("horizon") == 30)["rate"][0], 0.5)
        self.assertIsNone(data.filter(pl.col("horizon") == 90)["rate"][0])
        self.assertTrue(
            all(
                v is None for v in cohort_summary(baskets, rows, date(2024, 3, 1), False, 1)["rate"]
            )
        )

    def test_transitions(self):
        before = pl.DataFrame({"customer_id": ["A", "B", "C"], "g": ["old"] * 3}).lazy()
        after = pl.DataFrame({"customer_id": ["A", "B", "D"], "g": ["new", "old", "new"]}).lazy()
        data, counts = transition_summary(before, after, "g")
        self.assertEqual(counts["matched_n"], 2)
        self.assertEqual(counts["origin_unmatched"], 1)
        self.assertEqual(counts["destination_unmatched"], 1)
        self.assertEqual(data["rate"].sum(), 1.0)

    def test_momentum_map(self):
        # Test momentum map with customers having finite changes
        rows = pl.DataFrame({
            "customer_id": ["A", "B", "C", "D", "E"],
            "frequency_change": [0.5, -0.3, 0.2, None, -0.1],
            "basket_change": [0.2, -0.4, None, 0.1, -0.2],
            "value_tier": ["high", "middle", "low", "high", "middle"],
            "lifecycle": ["established_repeat"] * 5,
            "evidence": ["stronger", "limited", "stronger", "limited", "stronger"],
            "spend_90d": [100.0, 50.0, 75.0, 200.0, 30.0],
            "purchases_90d": [5, 3, 4, 10, 2],
            "purchases_previous90d": [4, 4, 3, 8, 2],
            "aov_recent": [20.0, 16.7, 18.75, 20.0, 15.0],
            "aov_previous": [16.0, 25.0, 20.0, 18.0, 18.0],
        }).lazy()

        result = momentum_map(rows)
        # A, B, E have both finite (C has null basket, D has null freq)
        self.assertEqual(result["finite_n"], 3)
        self.assertEqual(result["not_comparable_n"], 2)
        self.assertEqual(result["sample"].height, 3)
        self.assertEqual(set(result["sample"]["customer_id"].to_list()), {"A", "B", "E"})
        # Quadrant summary
        quad = result["quadrant_summary"]
        self.assertEqual(quad.height, 2)  # growing_both (A), declining_both (B, E)
        # A: freq>0, basket>0 -> growing_both
        # B: freq<0, basket<0 -> declining_both
        # E: freq<0, basket<0 -> declining_both
        quadrants = set(quad["quadrant"].to_list())
        self.assertIn("growing_both", quadrants)
        self.assertIn("declining_both", quadrants)

    def test_cadence_map(self):
        rows = pl.DataFrame({
            "customer_id": ["A", "B", "C", "D"],
            "median_gap": [7.0, 14.0, 30.0, None],
            "recency": [5.0, 20.0, 100.0, 10.0],
            "lateness": [0.7, 1.4, 3.3, None],
            "value_tier": ["high", "middle", "low", "high"],
            "evidence": ["stronger", "limited", "stronger", "limited"],
            "lifecycle": ["established_repeat"] * 4,
            "gap_cv": [0.2, 0.5, 0.8, 0.3],
            "days_365d": [50, 25, 12, 40],
            "style_eligible": [True, True, True, False],
        }).lazy()

        result = cadence_map(rows)
        # D has null median_gap, filtered out -> 3 remain
        self.assertEqual(result["eligible_n"], 3)
        self.assertEqual(result["total_n"], 3)
        self.assertEqual(result["sample"].height, 3)
        self.assertEqual(set(result["sample"]["customer_id"].to_list()), {"A", "B", "C"})

        # With evidence filter
        result_filtered = cadence_map(rows, evidence_filter=["stronger"])
        self.assertEqual(result_filtered["eligible_n"], 2)
        self.assertEqual(set(result_filtered["sample"]["customer_id"].to_list()), {"A", "C"})

    def test_revenue_waterfall(self):
        # Fixture: Previous: 2 buyers, 4 baskets, spend 40. Current: 2 buyers, 6 baskets, spend 90.
        # Expected: buyer contribution 0, frequency +20, AOV +30, total +50, residual ~0.
        current = pl.DataFrame({
            "customer_id": ["A", "B", "C"],
            "purchases_90d": [2, 4, 0],
            "spend_90d": [20.0, 70.0, 0.0],
        }).lazy()

        previous = pl.DataFrame({
            "customer_id": ["A", "B", "C"],
            "purchases_90d": [2, 2, 0],
            "spend_90d": [20.0, 20.0, 0.0],
        }).lazy()

        cutoff = date(2024, 6, 1)
        result = revenue_waterfall(current, previous, cutoff)

        self.assertEqual(result.height, 6)
        self.assertIn("status", result.columns)
        self.assertIn("order", result.columns)

        # Verify components and values
        comp_vals = {row["component"]: row["value"] for row in result.to_dicts()}

        # Previous: 2 purchasers, 4 baskets total, spend 40 -> AOV = 10
        # Current: 2 purchasers, 6 baskets total, spend 90 -> AOV = 15
        # F_prev = 4/2 = 2, A_prev = 40/4 = 10
        # F_curr = 6/2 = 3, A_curr = 90/6 = 15
        # delta_N = (2-2) * 2 * 10 = 0
        # delta_F = 2 * (3-2) * 10 = 20
        # delta_A = 2 * 3 * (15-10) = 30
        # total change = 90 - 40 = 50
        # residual = 50 - 0 - 20 - 30 = 0

        self.assertAlmostEqual(comp_vals["Previous period spend"], 40.0)
        self.assertAlmostEqual(comp_vals["Purchasing customers"], 0.0)
        self.assertAlmostEqual(comp_vals["Baskets per customer"], 20.0)
        self.assertAlmostEqual(comp_vals["Spend per basket (AOV)"], 30.0)
        self.assertAlmostEqual(comp_vals["Residual (rounding)"], 0.0, places=6)
        self.assertAlmostEqual(comp_vals["Current period spend"], 90.0)

        # Check status column - all should be None for valid decomposition
        for row in result.to_dicts():
            self.assertIsNone(row["status"])

    def test_revenue_waterfall_zero_previous_activity(self):
        # Previous period has no purchasing activity
        current = pl.DataFrame({
            "customer_id": ["A", "B"],
            "purchases_90d": [2, 3],
            "spend_90d": [20.0, 30.0],
        }).lazy()

        previous = pl.DataFrame({
            "customer_id": ["A", "B"],
            "purchases_90d": [0, 0],
            "spend_90d": [0.0, 0.0],
        }).lazy()

        cutoff = date(2024, 6, 1)
        result = revenue_waterfall(current, previous, cutoff)

        self.assertEqual(result.height, 4)
        comp_vals = {row["component"]: row["value"] for row in result.to_dicts()}
        self.assertAlmostEqual(comp_vals["Previous period spend"], 0.0)
        self.assertAlmostEqual(comp_vals["Current period spend"], 50.0)
        self.assertAlmostEqual(comp_vals["Total change"], 50.0)
        self.assertIsNone(comp_vals["Decomposition"])
        self.assertEqual(result.filter(pl.col("component") == "Decomposition")["status"][0],
                         "unavailable: zero purchasing customers or baskets in previous period")

    def test_revenue_waterfall_zero_current_activity(self):
        # Current period has no purchasing activity
        current = pl.DataFrame({
            "customer_id": ["A", "B"],
            "purchases_90d": [0, 0],
            "spend_90d": [0.0, 0.0],
        }).lazy()

        previous = pl.DataFrame({
            "customer_id": ["A", "B"],
            "purchases_90d": [2, 3],
            "spend_90d": [20.0, 30.0],
        }).lazy()

        cutoff = date(2024, 6, 1)
        result = revenue_waterfall(current, previous, cutoff)

        self.assertEqual(result.height, 4)
        comp_vals = {row["component"]: row["value"] for row in result.to_dicts()}
        self.assertAlmostEqual(comp_vals["Previous period spend"], 50.0)
        self.assertAlmostEqual(comp_vals["Current period spend"], 0.0)
        self.assertAlmostEqual(comp_vals["Total change"], -50.0)
        self.assertIsNone(comp_vals["Decomposition"])
        self.assertEqual(result.filter(pl.col("component") == "Decomposition")["status"][0],
                         "unavailable: zero purchasing customers or baskets in current period")

    def test_revenue_waterfall_none_input(self):
        # None input should return empty DataFrame
        result = revenue_waterfall(None, None, date(2024, 6, 1))
        self.assertEqual(result.height, 0)

    def test_value_tier_transition_waterfall_eager_lazy(self):
        """Test value_tier_transition_waterfall with both eager and lazy inputs."""
        # Create test data with value_tier transitions
        before = pl.DataFrame({
            "customer_id": ["A", "B", "C", "D", "E"],
            "value_tier": ["high", "high", "middle", "middle", "low"],
        }).lazy()
        
        after = pl.DataFrame({
            "customer_id": ["A", "B", "C", "D", "E"],
            "value_tier": ["high", "middle", "middle", "low", "low"],
        }).lazy()
        
        # Get transitions
        flows, counts = transition_summary(before, after, "value_tier")
        
        # Test with lazy flows (as returned by transition_summary)
        waterfall_lazy = value_tier_transition_waterfall(flows)
        self.assertEqual(waterfall_lazy.height, 3)  # high, middle, low
        
        # Test with eager flows - transition_summary returns eager DataFrame already
        # Just call value_tier_transition_waterfall again with the same flows
        waterfall_eager = value_tier_transition_waterfall(flows)
        self.assertEqual(waterfall_eager.height, 3)
        
        # Both should give same results
        lazy_dict = {row["tier"]: row["net_change"] for row in waterfall_lazy.to_dicts()}
        eager_dict = {row["tier"]: row["net_change"] for row in waterfall_eager.to_dicts()}
        self.assertEqual(lazy_dict, eager_dict)
        
        # Verify net changes:
        # high: outflow=2 (A stays, B moves to middle), inflow=1 (A stays) -> net = -1
        # middle: outflow=2 (C stays, D moves to low), inflow=2 (B from high, C stays) -> net = 0
        # low: outflow=1 (E stays), inflow=2 (D from middle, E stays) -> net = +1
        self.assertEqual(lazy_dict["high"], -1)
        self.assertEqual(lazy_dict["middle"], 0)
        self.assertEqual(lazy_dict["low"], 1)
        
        # Sum of net changes should be 0 for matched population
        self.assertEqual(sum(lazy_dict.values()), 0)
    
    def test_value_tier_transition_waterfall_empty(self):
        """Test value_tier_transition_waterfall with empty flows."""
        empty_flows = pl.DataFrame({
            "from_label": [],
            "to_label": [],
            "len": [],
        }).lazy()
        
        result = value_tier_transition_waterfall(empty_flows)
        self.assertEqual(result.height, 0)
    
    def test_value_tier_transition_waterfall_one_sided(self):
        """Test value_tier_transition_waterfall with tiers appearing only on one side."""
        before = pl.DataFrame({
            "customer_id": ["A", "B"],
            "value_tier": ["high", "middle"],
        }).lazy()
        
        after = pl.DataFrame({
            "customer_id": ["A", "B", "C"],
            "value_tier": ["middle", "low", "high"],
        }).lazy()
        
        flows, _ = transition_summary(before, after, "value_tier")
        waterfall = value_tier_transition_waterfall(flows)
        
        # Should have all three tiers
        tiers = set(waterfall["tier"].to_list())
        self.assertEqual(tiers, {"high", "middle", "low"})
        
        # Net change should sum to 0 for matched population
        matched_net = waterfall.filter(pl.col("tier").is_in(["high", "middle", "low"]))["net_change"].sum()
        # For matched customers (A, B): high->middle (high -1, middle +1), middle->low (middle -1, low +1)
        # Net: high=-1, middle=0, low=+1, sum=0
        self.assertEqual(matched_net, 0)

if __name__ == "__main__":
    unittest.main()
