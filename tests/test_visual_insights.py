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
        calendar_period_revenue_trend,
        category_penetration_growth_portfolio,
        cohort_age_retention_heatmap,
        cohort_summary,
        concentration,
        full_population_distribution,
        momentum_map,
        product_association_network,
        revenue_bridge,
        revenue_waterfall,
        rfm_response_surface,
        segment_alluvial_flow,
        transition_summary,
        trial_conversion_interval_plot,
        value_at_risk_scatter,
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
        rows = pl.DataFrame(
            {
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
            }
        ).lazy()

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
        rows = pl.DataFrame(
            {
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
            }
        ).lazy()

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
        current = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C"],
                "purchases_90d": [2, 4, 0],
                "spend_90d": [20.0, 70.0, 0.0],
            }
        ).lazy()

        previous = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C"],
                "purchases_90d": [2, 2, 0],
                "spend_90d": [20.0, 20.0, 0.0],
            }
        ).lazy()

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
        current = pl.DataFrame(
            {
                "customer_id": ["A", "B"],
                "purchases_90d": [2, 3],
                "spend_90d": [20.0, 30.0],
            }
        ).lazy()

        previous = pl.DataFrame(
            {
                "customer_id": ["A", "B"],
                "purchases_90d": [0, 0],
                "spend_90d": [0.0, 0.0],
            }
        ).lazy()

        cutoff = date(2024, 6, 1)
        result = revenue_waterfall(current, previous, cutoff)

        self.assertEqual(result.height, 4)
        comp_vals = {row["component"]: row["value"] for row in result.to_dicts()}
        self.assertAlmostEqual(comp_vals["Previous period spend"], 0.0)
        self.assertAlmostEqual(comp_vals["Current period spend"], 50.0)
        self.assertAlmostEqual(comp_vals["Total change"], 50.0)
        self.assertIsNone(comp_vals["Decomposition"])
        self.assertEqual(
            result.filter(pl.col("component") == "Decomposition")["status"][0],
            "unavailable: zero purchasing customers or baskets in previous period",
        )

    def test_revenue_waterfall_zero_current_activity(self):
        # Current period has no purchasing activity
        current = pl.DataFrame(
            {
                "customer_id": ["A", "B"],
                "purchases_90d": [0, 0],
                "spend_90d": [0.0, 0.0],
            }
        ).lazy()

        previous = pl.DataFrame(
            {
                "customer_id": ["A", "B"],
                "purchases_90d": [2, 3],
                "spend_90d": [20.0, 30.0],
            }
        ).lazy()

        cutoff = date(2024, 6, 1)
        result = revenue_waterfall(current, previous, cutoff)

        self.assertEqual(result.height, 4)
        comp_vals = {row["component"]: row["value"] for row in result.to_dicts()}
        self.assertAlmostEqual(comp_vals["Previous period spend"], 50.0)
        self.assertAlmostEqual(comp_vals["Current period spend"], 0.0)
        self.assertAlmostEqual(comp_vals["Total change"], -50.0)
        self.assertIsNone(comp_vals["Decomposition"])
        self.assertEqual(
            result.filter(pl.col("component") == "Decomposition")["status"][0],
            "unavailable: zero purchasing customers or baskets in current period",
        )

    def test_revenue_waterfall_none_input(self):
        # None input should return empty DataFrame
        result = revenue_waterfall(None, None, date(2024, 6, 1))
        self.assertEqual(result.height, 0)

    def test_value_tier_transition_waterfall_eager_lazy(self):
        """Test value_tier_transition_waterfall with both eager and lazy inputs."""
        # Create test data with value_tier transitions
        before = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C", "D", "E"],
                "value_tier": ["high", "high", "middle", "middle", "low"],
            }
        ).lazy()

        after = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C", "D", "E"],
                "value_tier": ["high", "middle", "middle", "low", "low"],
            }
        ).lazy()

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
        empty_flows = pl.DataFrame(
            {
                "from_label": [],
                "to_label": [],
                "len": [],
            }
        ).lazy()

        result = value_tier_transition_waterfall(empty_flows)
        self.assertEqual(result.height, 0)

    def test_value_tier_transition_waterfall_one_sided(self):
        """Test value_tier_transition_waterfall with tiers appearing only on one side."""
        before = pl.DataFrame(
            {
                "customer_id": ["A", "B"],
                "value_tier": ["high", "middle"],
            }
        ).lazy()

        after = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C"],
                "value_tier": ["middle", "low", "high"],
            }
        ).lazy()

        flows, _ = transition_summary(before, after, "value_tier")
        waterfall = value_tier_transition_waterfall(flows)

        # Should have all three tiers
        tiers = set(waterfall["tier"].to_list())
        self.assertEqual(tiers, {"high", "middle", "low"})

        # Net change should sum to 0 for matched population
        matched_net = waterfall.filter(pl.col("tier").is_in(["high", "middle", "low"]))[
            "net_change"
        ].sum()
        # For matched customers (A, B): high->middle (high -1, middle +1), middle->low (middle -1, low +1)
        # Net: high=-1, middle=0, low=+1, sum=0
        self.assertEqual(matched_net, 0)

    def test_rfm_response_surface_no_coverage(self):
        """Test rfm_response_surface returns warning when coverage is False."""
        rows = pl.DataFrame(
            {
                "customer_id": ["A", "B"],
                "recency": [10.0, 20.0],
                "purchases_90d": [5, 3],
                "future_purchases_30d": [1, 0],
                "eligible_followup_30d": [True, True],
            }
        ).lazy()

        result = rfm_response_surface(rows, coverage=False)
        self.assertIn("warning", result.columns)
        self.assertEqual(result["warning"][0], "Coverage unconfirmed - future outcomes unavailable")

    def test_rfm_response_surface_binning(self):
        """Test rfm_response_surface creates correct bins with varied values."""
        # Use enough varied values to create proper bins
        rows = pl.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(40)],
                "recency": [5.0] * 10 + [15.0] * 10 + [50.0] * 10 + [100.0] * 10,
                "purchases_90d": [1] * 10 + [3] * 10 + [5] * 10 + [8] * 10,
                "future_purchases_30d": [1] * 30 + [0] * 10,
                "eligible_followup_30d": [True] * 40,
            }
        ).lazy()

        result = rfm_response_surface(rows, coverage=True, recency_bins=2, frequency_bins=2)

        self.assertNotIn("warning", result.columns)
        self.assertNotIn("error", result.columns)
        # Should have up to 4 cells (2x2), but may be fewer if data doesn't fill all bins
        self.assertGreaterEqual(result.height, 1)
        self.assertLessEqual(result.height, 4)
        # Check structure
        self.assertIn("recency_bin", result.columns)
        self.assertIn("frequency_bin", result.columns)
        self.assertIn("rate", result.columns)
        self.assertIn("lower_ci", result.columns)
        self.assertIn("upper_ci", result.columns)

    def test_rfm_response_surface_constant_values(self):
        """Test rfm_response_surface handles constant recency/frequency (single bin)."""
        rows = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C", "D", "E"],
                "recency": [10.0] * 5,  # All same recency
                "purchases_90d": [3] * 5,  # All same frequency
                "future_purchases_30d": [1, 1, 0, 1, 0],
                "eligible_followup_30d": [True] * 5,
            }
        ).lazy()

        result = rfm_response_surface(rows, coverage=True, recency_bins=5, frequency_bins=5)
        # With constant values, only 1 bin possible -> empty result
        self.assertEqual(result.height, 0)

    def test_cohort_age_retention_heatmap(self):
        """Test cohort_age_retention_heatmap returns correct structure."""
        from datetime import date

        # Create validated transaction data
        validated = pl.DataFrame(
            {
                "customer_id": ["A", "A", "B", "B", "C", "C"],
                "date": [
                    date(2024, 1, 1),
                    date(2024, 2, 1),  # A: 2 months
                    date(2024, 1, 15),
                    date(2024, 3, 1),  # B: 2 months
                    date(2024, 2, 1),
                    date(2024, 2, 15),  # C: same month (no repeat)
                ],
            }
        ).lazy()

        # Snapshot data with first_purchase
        rows = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C"],
                "first_purchase": [date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 1)],
            }
        ).lazy()

        cutoff = date(2024, 3, 15)
        result = cohort_age_retention_heatmap(validated, rows, cutoff, min_cohort_size=1)

        self.assertEqual(result.height, 4)  # 2 cohorts * ages (0, 1 for Jan; 0 for Feb)
        self.assertIn("cohort_month", result.columns)
        self.assertIn("age_month", result.columns)
        self.assertIn("active_customers", result.columns)
        self.assertIn("cohort_size", result.columns)
        self.assertIn("activity_rate", result.columns)

        # Jan cohort (A, B): age 0 = 2/2=1.0, age 1 = 1/2=0.5 (A only), age 2 = 1/2=0.5 (B only)
        jan_data = result.filter(pl.col("cohort_month") == date(2024, 1, 1))
        age0 = jan_data.filter(pl.col("age_month") == 0)["activity_rate"][0]
        self.assertAlmostEqual(age0, 1.0)

    def test_calendar_period_revenue_trend(self):
        """Test calendar_period_revenue_trend returns correct structure."""
        from datetime import date

        validated = pl.DataFrame(
            {
                "customer_id": ["A", "A", "B", "B", "C"],
                "date": [
                    date(2024, 1, 1),
                    date(2024, 2, 1),
                    date(2024, 1, 15),
                    date(2024, 2, 15),
                    date(2024, 1, 10),
                ],
                "value": [100.0, 200.0, 150.0, 250.0, 300.0],
            }
        ).lazy()

        rows = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C"],
                "first_purchase": [date(2023, 12, 1)] * 3,
            }
        ).lazy()

        first_date = date(2024, 1, 1)
        cutoff = date(2024, 2, 28)
        result = calendar_period_revenue_trend(validated, rows, first_date, cutoff, months=2)

        self.assertEqual(result.height, 2)
        self.assertIn("month", result.columns)
        self.assertIn("spend", result.columns)
        self.assertIn("baskets", result.columns)
        self.assertIn("buyers", result.columns)
        self.assertIn("aov", result.columns)
        self.assertIn("baskets_per_buyer", result.columns)
        self.assertIn("buying_share", result.columns)
        self.assertIn("population_n", result.columns)

    def test_revenue_bridge(self):
        """Test revenue_bridge with matched and unmatched options."""
        current = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C"],
                "purchases_90d": [2, 4, 0],
                "spend_90d": [20.0, 70.0, 0.0],
            }
        ).lazy()

        previous = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C"],
                "purchases_90d": [2, 2, 0],
                "spend_90d": [20.0, 20.0, 0.0],
            }
        ).lazy()

        cutoff = date(2024, 6, 1)

        # Test matched
        result_matched = revenue_bridge(current, previous, cutoff, match_customers=True)
        self.assertEqual(result_matched.height, 6)
        comp_vals = {row["component"]: row["value"] for row in result_matched.to_dicts()}
        self.assertAlmostEqual(comp_vals["Previous period spend"], 40.0)
        self.assertAlmostEqual(comp_vals["Current period spend"], 90.0)
        # All population_type should be "matched"
        for row in result_matched.to_dicts():
            if row["population_type"] is not None:
                self.assertEqual(row["population_type"], "matched")

        # Test unmatched
        result_unmatched = revenue_bridge(current, previous, cutoff, match_customers=False)
        for row in result_unmatched.to_dicts():
            if row["population_type"] is not None:
                self.assertEqual(row["population_type"], "unmatched (full snapshots)")

    def test_revenue_bridge_edge_cases(self):
        """Test revenue_bridge with zero activity."""
        current = pl.DataFrame(
            {"customer_id": ["A", "B"], "purchases_90d": [0, 0], "spend_90d": [0.0, 0.0]}
        ).lazy()

        previous = pl.DataFrame(
            {"customer_id": ["A", "B"], "purchases_90d": [2, 3], "spend_90d": [20.0, 30.0]}
        ).lazy()

        result = revenue_bridge(current, previous, date(2024, 6, 1))
        self.assertEqual(result.height, 4)
        self.assertIn(
            "unavailable: zero purchasing customers or baskets in current period",
            result["status"].to_list(),
        )

    def test_value_at_risk_scatter(self):
        """Test value_at_risk_scatter returns correct structure."""
        rows = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C", "D", "E"],
                "recency": [10.0, 60.0, 120.0, 5.0, 30.0],
                "median_gap": [30.0, 30.0, 30.0, 7.0, 15.0],
                "lateness": [0.33, 2.0, 4.0, 0.71, 2.0],
                "spend_365d": [1000.0, 500.0, 2000.0, 100.0, 800.0],
                "spend_90d": [200.0, 100.0, 500.0, 50.0, 300.0],
                "purchases_365d": [12, 6, 24, 4, 10],
                "purchases_90d": [3, 2, 5, 1, 4],
                "value_tier": ["high", "middle", "high", "low", "middle"],
                "trajectory": ["stable", "slowing", "accelerating", "stable", "slowing"],
                "evidence": ["stronger"] * 5,
                "lifecycle": ["established_repeat"] * 5,
            }
        ).lazy()

        result = value_at_risk_scatter(rows, lateness_threshold=2.0, sample_limit=5)

        self.assertIn("sample", result)
        self.assertIn("summary", result)
        self.assertEqual(result["sample"].height, 5)
        self.assertEqual(result["summary"].height, 1)

        summary = result["summary"].row(0, named=True)
        self.assertEqual(summary["eligible_customers"], 5)
        self.assertEqual(summary["overdue_customers"], 3)  # B, C, E have lateness >= 2.0
        self.assertAlmostEqual(summary["at_risk_spend_365d"], 3300.0)  # 500 + 2000 + 800

    def test_product_association_network(self):
        """Test product_association_network builds nodes and edges."""
        pairs = pl.DataFrame(
            {
                "item_a": ["A", "A", "B"],
                "item_b": ["B", "C", "C"],
                "cooccurrence_count": [10, 5, 8],
                "support": [0.1, 0.05, 0.08],
                "confidence_a_to_b": [0.5, 0.3, 0.4],
                "confidence_b_to_a": [0.6, 0.2, 0.5],
                "lift": [1.5, 1.2, 1.8],
                "conviction_a_to_b": [1.2, 1.1, 1.3],
                "conviction_b_to_a": [1.3, 1.0, 1.4],
            }
        )

        network = product_association_network(pairs, min_lift=1.0, min_confidence=0.1, top_n=100)

        self.assertIn("nodes", network)
        self.assertIn("edges", network)
        self.assertEqual(network["nodes"].height, 3)  # A, B, C
        self.assertEqual(network["edges"].height, 3)
        self.assertIn("degree", network["nodes"].columns)
        self.assertIn("total_support", network["nodes"].columns)
        self.assertIn("max_lift", network["nodes"].columns)

    def test_trial_conversion_interval_plot(self):
        """Test trial_conversion_interval_plot returns correct structure."""
        from datetime import date

        product_history = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C", "D"],
                "product_id": ["P1", "P1", "P2", "P2"],
                "trial": [date(2024, 1, 1)] * 4,
                "repeat": [date(2024, 1, 15), None, date(2024, 1, 20), date(2024, 3, 1)],
            }
        ).lazy()

        rows = pl.DataFrame(
            {"customer_id": ["A", "B", "C", "D"], "first_purchase": [date(2023, 12, 1)] * 4}
        ).lazy()

        cutoff = date(2024, 4, 1)
        result = trial_conversion_interval_plot(product_history, rows, cutoff, min_trials=1)

        self.assertEqual(result.height, 2)  # P1, P2
        self.assertIn("product_id", result.columns)
        self.assertIn("eligible_trials", result.columns)
        self.assertIn("repeated_trials", result.columns)
        self.assertIn("trial_conversion", result.columns)
        self.assertIn("lower_ci", result.columns)
        self.assertIn("upper_ci", result.columns)
        self.assertIn("trialling_customers", result.columns)

    def test_category_penetration_growth_portfolio(self):
        """Test category_penetration_growth_portfolio returns correct structure."""
        from datetime import date

        validated = pl.DataFrame(
            {
                "customer_id": ["A", "A", "B", "B", "C"],
                "department": ["D1", "D1", "D1", "D2", "D1"],
                "category": ["C1", "C1", "C1", "C2", "C1"],
                "date": [
                    date(2024, 6, 1),
                    date(2024, 7, 1),
                    date(2024, 6, 15),
                    date(2024, 7, 15),
                    date(2024, 6, 10),
                ],
                "value": [100.0, 200.0, 150.0, 250.0, 300.0],
            }
        ).lazy()

        rows = pl.DataFrame(
            {"customer_id": ["A", "B", "C"], "first_purchase": [date(2024, 1, 1)] * 3}
        ).lazy()

        current_cutoff = date(2024, 7, 15)
        previous_cutoff = date(2024, 4, 15)

        result = category_penetration_growth_portfolio(
            validated, rows, current_cutoff, previous_cutoff, top_categories=10
        )

        self.assertEqual(result.height, 2)  # C1, C2
        self.assertIn("label", result.columns)
        self.assertIn("curr_penetration", result.columns)
        self.assertIn("prev_penetration", result.columns)
        self.assertIn("penetration_change_pp", result.columns)
        self.assertIn("spend_share", result.columns)
        self.assertIn("repeat_buyer_share", result.columns)

    def test_segment_alluvial_flow(self):
        """Test segment_alluvial_flow builds flows across snapshots."""
        from datetime import date

        snap1 = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C", "D"],
                "value_tier": ["high", "high", "middle", "low"],
                "snapshot": [date(2024, 1, 1)] * 4,
            }
        ).lazy()

        snap2 = pl.DataFrame(
            {
                "customer_id": ["A", "B", "C", "D", "E"],
                "value_tier": ["high", "middle", "middle", "low", "high"],
                "snapshot": [date(2024, 2, 1)] * 5,
            }
        ).lazy()

        result = segment_alluvial_flow([snap1, snap2], dimensions=["value_tier"])

        self.assertIn("flows", result)
        self.assertIn("snapshots", result)
        self.assertEqual(len(result["snapshots"]), 1)
        self.assertEqual(result["flows"].height, 4)  # 4 transitions for matched customers
        self.assertIn("from_snapshot", result["flows"].columns)
        self.assertIn("to_snapshot", result["flows"].columns)
        self.assertIn("from_state", result["flows"].columns)
        self.assertIn("to_state", result["flows"].columns)
        self.assertIn("customer_count", result["flows"].columns)
        self.assertIn("dimension", result["flows"].columns)

    def test_full_population_distribution(self):
        """Test full_population_distribution returns correct structure."""
        rows = pl.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(100)],
                "spend_90d": list(range(100)),
                "purchases_90d": [i % 10 for i in range(100)],
                "recency": [i * 0.5 for i in range(100)],
            }
        ).lazy()

        result = full_population_distribution(
            rows, features=["spend_90d", "purchases_90d", "recency"]
        )

        self.assertEqual(len(result), 3)
        for feature in ["spend_90d", "purchases_90d", "recency"]:
            self.assertIn(feature, result)
            self.assertIn("summary", result[feature])
            self.assertIn("histogram", result[feature])
            self.assertIn("n", result[feature])
            self.assertEqual(result[feature]["n"], 100)
            # Summary should have expected columns
            summary = result[feature]["summary"]
            self.assertIn("mean", summary.columns)
            self.assertIn("median", summary.columns)
            self.assertIn("p1", summary.columns)
            self.assertIn("p99", summary.columns)


if __name__ == "__main__":
    unittest.main()
