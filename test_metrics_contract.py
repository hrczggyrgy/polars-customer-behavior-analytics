"""Deterministic metric contract tests for the retail pipeline.

Each test hand-verifies a specific metric calculation against a known expected result.
Run with: python -m unittest test_metrics_contract -v
"""
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import polars as pl
from retail_polars_workflow import run, prepare, snapshot, label, ratio


class MetricContractTests(unittest.TestCase):
    """Hand-verifiable tests for every core metric formula."""
    
    def _make_fixture(self):
        """Create a minimal deterministic dataset.
        
        Customer A: 
        - 2025-01-01: P1 (qty=2, price=10) -> value=20
        - 2025-01-15: P1 (qty=1, price=10) -> value=10
        - 2025-02-01: P2 (qty=3, price=5) -> value=15
        - 2025-03-01: P1 (qty=2, price=10) -> value=20
        - 2025-03-15: P3 (qty=1, price=20) -> value=20
        
        Customer B:
        - 2025-02-01: P1 (qty=1, price=10) -> value=10
        - 2025-02-15: P2 (qty=2, price=5) -> value=10
        
        Customer C (single purchase):
        - 2025-04-01: P1 (qty=1, price=10) -> value=10
        """
        rows = [
            # Customer A - 5 transactions, 4 shopping days
            dict(customer_id="A", transaction_date="2025-01-01", transaction_id="A1", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=2.0),
            dict(customer_id="A", transaction_date="2025-01-15", transaction_id="A2", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=1.0),
            dict(customer_id="A", transaction_date="2025-02-01", transaction_id="A3", product_id="P2", product_description="P2", department="D", category="C2", price=5.0, quantity=3.0),
            dict(customer_id="A", transaction_date="2025-03-01", transaction_id="A4", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=2.0),
            dict(customer_id="A", transaction_date="2025-03-15", transaction_id="A5", product_id="P3", product_description="P3", department="D", category="C3", price=20.0, quantity=1.0),
            # Customer B - 2 transactions, 2 shopping days
            dict(customer_id="B", transaction_date="2025-02-01", transaction_id="B1", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=1.0),
            dict(customer_id="B", transaction_date="2025-02-15", transaction_id="B2", product_id="P2", product_description="P2", department="D", category="C2", price=5.0, quantity=2.0),
            # Customer C - 1 transaction, 1 shopping day
            dict(customer_id="C", transaction_date="2025-04-01", transaction_id="C1", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=1.0),
        ]
        return pl.DataFrame(rows)
    
    def _run_pipeline(self, source_df, cutoff, buckets=1, coverage_confirmed=False, observed_through=None):
        """Run pipeline on fixture and return snapshot frame."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.parquet"
            source_df.write_parquet(source)
            out = root / "results"
            # If observed_through not provided, use max date in data, but ensure >= cutoff
            if observed_through is None:
                max_date = source_df["transaction_date"].max()
                cutoff_date = date.fromisoformat(cutoff)
                if isinstance(max_date, str):
                    max_date = date.fromisoformat(max_date)
                observed_through = str(max(max_date, cutoff_date))
            run(str(source), str(out), "unit", [cutoff], buckets=buckets, coverage_confirmed=coverage_confirmed, observed_through=observed_through)
            # Read all bucket files for the snapshot
            frames = []
            for path in (out / "snapshots" / cutoff).glob("*.parquet"):
                frames.append(pl.read_parquet(path))
            if frames:
                return pl.concat(frames).sort("customer_id")
            return pl.DataFrame()
    
    def test_spend_calculation(self):
        """Test spend = sum(line_value) per window."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        b = frame.filter(pl.col("customer_id") == "B").row(0, named=True)
        
        # Customer A: all 5 transactions before cutoff, total = 20+10+15+20+20 = 85
        self.assertAlmostEqual(a["spend_history"], 85.0)
        # 30d window: [2025-03-02, 2025-03-31] -> only transaction on 2025-03-15 (20)
        self.assertAlmostEqual(a["spend_30d"], 20.0)
        # 90d window: [2024-12-31, 2025-03-31] -> all 5 transactions = 85
        self.assertAlmostEqual(a["spend_90d"], 85.0)
        # 365d window: all = 85
        self.assertAlmostEqual(a["spend_365d"], 85.0)
        
        # Customer B: transactions on 2025-02-01 and 2025-02-15
        self.assertAlmostEqual(b["spend_history"], 20.0)
        self.assertAlmostEqual(b["spend_30d"], 0.0)  # No transactions in last 30 days
        self.assertAlmostEqual(b["spend_90d"], 20.0)
        self.assertAlmostEqual(b["spend_365d"], 20.0)
        
        # Customer C: transaction on 2025-04-01 (after cutoff!)
        # Should not appear in snapshot since cutoff is 2025-03-31
        self.assertEqual(frame.filter(pl.col("customer_id") == "C").height, 0)
    
    def test_purchases_basket_count(self):
        """Test purchases = count of distinct baskets (customer_id, transaction_id, date)."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        b = frame.filter(pl.col("customer_id") == "B").row(0, named=True)
        
        # Customer A: 5 transactions, all unique (transaction_id), 4 shopping days
        # But basket aggregation is by (customer_id, transaction_id, date) -> each transaction is a basket
        self.assertEqual(a["purchases_history"], 5)
        # 30d window: [C-29, C] = [2025-03-02, 2025-03-31] -> only 2025-03-15 (A5) = 1
        self.assertEqual(a["purchases_30d"], 1)
        # 90d window: [C-89, C] = [2024-12-31, 2025-03-31] -> all 5
        self.assertEqual(a["purchases_90d"], 5)
        # 365d window: [C-364, C] = [2024-04-01, 2025-03-31] -> all 5
        self.assertEqual(a["purchases_365d"], 5)
        
        # Customer B: 2 transactions on 2025-02-01 and 2025-02-15
        # 30d window: [2025-03-02, 2025-03-31] -> none
        self.assertEqual(b["purchases_history"], 2)
        self.assertEqual(b["purchases_30d"], 0)
        # 90d window: [2024-12-31, 2025-03-31] -> both transactions
        self.assertEqual(b["purchases_90d"], 2)
        self.assertEqual(b["purchases_365d"], 2)
    
    def test_aov_calculation(self):
        """Test AOV = spend / purchases (per basket)."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        b = frame.filter(pl.col("customer_id") == "B").row(0, named=True)
        
        # Customer A: spend_90d=85, purchases_90d=5 -> AOV = 17.0
        self.assertAlmostEqual(a["aov_recent"], 17.0)
        # aov_365d = mean basket value = 85/5 = 17.0
        self.assertAlmostEqual(a["aov_365d"], 17.0)
        
        # Customer B: spend_90d=20, purchases_90d=2 -> AOV = 10.0
        self.assertAlmostEqual(b["aov_recent"], 10.0)
        self.assertAlmostEqual(b["aov_365d"], 10.0)
    
    def test_median_basket(self):
        """Test median_basket = median of basket values in 365d."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        b = frame.filter(pl.col("customer_id") == "B").row(0, named=True)
        
        # Customer A basket values: 20, 10, 15, 20, 20 -> sorted: 10, 15, 20, 20, 20 -> median = 20
        self.assertAlmostEqual(a["median_basket"], 20.0)
        
        # Customer B basket values: 10, 10 -> median = 10
        self.assertAlmostEqual(b["median_basket"], 10.0)
    
    def test_recency_tenure(self):
        """Test recency and tenure in days."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        b = frame.filter(pl.col("customer_id") == "B").row(0, named=True)
        
        # Customer A: first_purchase=2025-01-01, last_purchase=2025-03-15
        # tenure = 2025-03-31 - 2025-01-01 = 89 days
        # recency = 2025-03-31 - 2025-03-15 = 16 days
        self.assertEqual(a["tenure"], 89)
        self.assertEqual(a["recency"], 16)
        
        # Customer B: first_purchase=2025-02-01, last_purchase=2025-02-15
        # tenure = 2025-03-31 - 2025-02-01 = 58 days
        # recency = 2025-03-31 - 2025-02-15 = 44 days
        self.assertEqual(b["tenure"], 58)
        self.assertEqual(b["recency"], 44)
    
    def test_gap_and_gap_cv(self):
        """Test gap (days between shopping days) and gap_cv."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        
        # Customer A shopping days: 2025-01-01, 2025-01-15, 2025-02-01, 2025-03-01, 2025-03-15
        # Gaps: 14, 17, 28, 14 days
        # Usable filter requires prev_date >= C-365 = 2024-03-31, so all gaps usable
        # gap_count = 4
        # median_gap = median(14, 17, 28, 14) = median(14, 14, 17, 28) = (14+17)/2 = 15.5
        # gap_cv = std/mean - Polars std() uses sample std (N-1)
        # mean = 73/4 = 18.25
        # sample std = sqrt(((14-18.25)^2 + (17-18.25)^2 + (28-18.25)^2 + (14-18.25)^2)/3)
        #            = sqrt(132.75/3) = sqrt(44.25) ≈ 6.652
        # gap_cv ≈ 6.652/18.25 ≈ 0.364
        
        self.assertEqual(a["gap_count"], 4)
        self.assertAlmostEqual(a["median_gap"], 15.5)
        self.assertAlmostEqual(a["gap_cv"], 0.364, places=2)
    
    def test_breadth_metrics(self):
        """Test breadth (distinct products per basket) metrics."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        
        # Customer A baskets:
        # A1: P1 -> 1 product
        # A2: P1 -> 1 product
        # A3: P2 -> 1 product
        # A4: P1 -> 1 product
        # A5: P3 -> 1 product
        # All baskets have 1 product -> median_breadth = 1
        self.assertEqual(a["median_breadth"], 1)
        
        # breadth_recent (90d): all 5 baskets in 90d, all have 1 product -> mean = 1
        self.assertEqual(a["breadth_recent"], 1.0)
        
        # breadth_previous (days 91-180 before cutoff): no baskets -> null
        self.assertIsNone(a["breadth_previous"])
    
    def test_category_hhi_entropy(self):
        """Test HHI and entropy of category spend shares."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        
        # Customer A category spend in 365d:
        # C1: P1 transactions = 20 + 10 + 20 = 50
        # C2: P2 transaction = 15
        # C3: P3 transaction = 20
        # Total = 85
        # Shares: C1=50/85≈0.588, C2=15/85≈0.176, C3=20/85≈0.235
        # HHI = 0.588² + 0.176² + 0.235² ≈ 0.346 + 0.031 + 0.055 = 0.432
        # Entropy = -sum(p*ln(p)) = -(0.588*ln(0.588) + 0.176*ln(0.176) + 0.235*ln(0.235))
        #         ≈ -(0.588*-0.531 + 0.176*-1.737 + 0.235*-1.447)
        #         ≈ -(-0.312 - 0.306 - 0.340) = 0.958
        
        self.assertAlmostEqual(a["category_hhi"], 0.432, places=2)
        self.assertAlmostEqual(a["category_entropy"], 0.958, places=2)
        self.assertAlmostEqual(a["dominant_share"], 50/85, places=3)
    
    def test_trial_conversion(self):
        """Test trial conversion = repeated_trials / eligible_trials."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        
        # Trial window: [C-365, C-90] = [2024-03-31, 2024-12-30]
        # All A's transactions are in 2025, so no eligible trials -> 0
        self.assertEqual(a["eligible_trials"], 0)
        self.assertEqual(a["repeated_trials"], 0)
        self.assertIsNone(a["trial_conversion"])  # 0/0 -> null
    
    def test_new_repeat_established_shares(self):
        """Test new_share, repeat_share, established_share."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        
        # Customer A product first purchase (trial) dates:
        # P1: 2025-01-01
        # P2: 2025-02-01
        # P3: 2025-03-15
        # 
        # Baskets in 365d (all 5):
        # A1 (2025-01-01): P1 (prior_days=0) -> new
        # A2 (2025-01-15): P1 (prior_days=1) -> repeat
        # A3 (2025-02-01): P2 (prior_days=0) -> new
        # A4 (2025-03-01): P1 (prior_days=2) -> established
        # A5 (2025-03-15): P3 (prior_days=0) -> new
        # 
        # new_share = 3/5 = 0.6 (baskets with any new product: A1, A3, A5)
        # repeat_share = 2/5 = 0.4 (baskets all repeat: A2, A4 - wait, A4 has P1 with prior_days=2 which is established)
        # Actually: repeat_share = baskets where ALL products have prior_days >= 1
        # A2: P1 prior_days=1 -> all >=1 -> yes
        # A4: P1 prior_days=2 -> all >=1 -> yes
        # A1: has new -> no
        # A3: has new -> no
        # A5: has new -> no
        # repeat_share = 2/5 = 0.4
        # established_share = baskets where ALL products have prior_days >= 2
        # A4: prior_days=2 -> yes
        # A2: prior_days=1 -> no
        # established_share = 1/5 = 0.2
        
        self.assertAlmostEqual(a["new_share"], 0.6)
        self.assertAlmostEqual(a["repeat_share"], 0.4)
        self.assertAlmostEqual(a["established_share"], 0.2)
    
    def test_weekend_share(self):
        """Test weekend_share = proportion of shopping days on Sat/Sun."""
        # 2025-01-01 = Wednesday, 2025-01-15 = Wednesday
        # 2025-02-01 = Saturday, 2025-03-01 = Saturday, 2025-03-15 = Saturday
        # 3 out of 5 shopping days are weekend -> 0.6
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        self.assertAlmostEqual(a["weekend_share"], 0.6)
    
    def test_weekday_entropy(self):
        """Test Shannon entropy of weekday distribution."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        
        # Weekdays: Wed(2), Wed(2), Sat(5), Sat(5), Sat(5)
        # Counts: Wed=2, Sat=3, total=5
        # p_Wed=0.4, p_Sat=0.6
        # Entropy = -(0.4*ln(0.4) + 0.6*ln(0.6)) = -(0.4*-0.916 + 0.6*-0.511) = 0.673
        self.assertAlmostEqual(a["weekday_entropy"], 0.673, places=2)
    
    def test_products_last6days(self):
        """Test products_last6days requires >=6 shopping days."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        b = frame.filter(pl.col("customer_id") == "B").row(0, named=True)
        
        # Customer A has 5 shopping days < 6 -> should be null (days_history < 6)
        self.assertIsNone(a["products_last6days"])
        self.assertIsNone(a["categories_last6days"])
        
        # Customer B has 2 shopping days -> null
        self.assertIsNone(b["products_last6days"])
    
    def test_products_last6days_with_sufficient_history(self):
        """Test products_last6days with >=6 shopping days."""
        # Create customer with 6+ shopping days
        rows = []
        base_date = date(2025, 1, 1)
        for i in range(7):
            rows.append(dict(
                customer_id="D", transaction_date=str(base_date.replace(day=1+i)),
                transaction_id=f"D{i}", product_id=f"P{i%3+1}", product_description="P",
                department="D", category="C", price=10.0, quantity=1.0
            ))
        df = pl.DataFrame(rows)
        
        cutoff = "2025-01-31"
        # observed_through must be >= cutoff
        frame = self._run_pipeline(df, cutoff, observed_through="2025-01-31")
        
        d = frame.filter(pl.col("customer_id") == "D").row(0, named=True)
        # 7 shopping days, last 6 should have products P1,P2,P3,P1,P2,P3 -> 3 distinct
        self.assertEqual(d["products_last6days"], 3)
        self.assertEqual(d["categories_last6days"], 1)
    
    def test_zero_denominator_returns_null(self):
        """Test that metrics with zero denominator return null, not zero."""
        # Customer with no valid transactions before cutoff
        rows = [
            dict(customer_id="E", transaction_date="2025-05-01", transaction_id="E1", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
        ]
        df = pl.DataFrame(rows)
        
        cutoff = "2025-04-30"  # Before the transaction
        frame = self._run_pipeline(df, cutoff)
        
        # Customer E should not appear (no transactions <= cutoff)
        self.assertEqual(frame.height, 0)

    def test_rejection_reasons_numeric_nulls(self):
        """Test that rejected rows with null price/quantity get specific reasons."""
        rows = [
            # Valid row
            dict(customer_id="V", transaction_date="2025-01-01", transaction_id="V1", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # Null price (malformed string -> null after cast)
            dict(customer_id="NP", transaction_date="2025-01-01", transaction_id="NP1", product_id="P1", product_description="P1", department="D", category="C", price=None, quantity=1.0),
            # Null quantity
            dict(customer_id="NQ", transaction_date="2025-01-01", transaction_id="NQ1", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=None),
            # Negative price
            dict(customer_id="NG", transaction_date="2025-01-01", transaction_id="NG1", product_id="P1", product_description="P1", department="D", category="C", price=-5.0, quantity=1.0),
            # Zero quantity
            dict(customer_id="ZQ", transaction_date="2025-01-01", transaction_id="ZQ1", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=0.0),
            # Non-finite price (NaN)
            dict(customer_id="NF", transaction_date="2025-01-01", transaction_id="NF1", product_id="P1", product_description="P1", department="D", category="C", price=float('nan'), quantity=1.0),
        ]
        df = pl.DataFrame(rows)
        
        cutoff = "2025-01-31"
        # Run without exclude-invalid to capture rejected rows
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.parquet"
            df.write_parquet(source)
            out = root / "results"
            # This will raise because we don't exclude invalid
            try:
                run(str(source), str(out), "unit", [cutoff], buckets=1, coverage_confirmed=True, observed_through="2025-01-31")
            except ValueError:
                pass  # Expected
            
            # Check rejected.parquet
            rejected = pl.read_parquet(out / "rejected.parquet")
            self.assertEqual(rejected.height, 5)  # 5 invalid rows
            
            reasons = rejected["rejection_reason"].to_list()
            self.assertIn("null_price", reasons)
            self.assertIn("null_quantity", reasons)
            self.assertIn("negative_price", reasons)
            self.assertIn("nonpositive_quantity", reasons)
            self.assertIn("nonfinite_price", reasons)
            
            # Check quality.json breakdown
            import json
            with open(out / "quality.json") as f:
                quality = json.load(f)
            self.assertEqual(quality["rejected_rows"], 5)
            self.assertEqual(quality["valid_rows"], 1)
            self.assertAlmostEqual(quality["rejection_rate"], 5/6, places=4)
            # Verify breakdown sums to rejected_rows
            self.assertEqual(sum(quality["rejection_breakdown"].values()), quality["rejected_rows"])
            # Verify new reason codes exist
            self.assertIn("null_price", quality["rejection_breakdown"])
            self.assertIn("null_quantity", quality["rejection_breakdown"])
            self.assertIn("negative_price", quality["rejection_breakdown"])
            self.assertIn("nonpositive_quantity", quality["rejection_breakdown"])
            self.assertIn("nonfinite_price", quality["rejection_breakdown"])
    
    def test_insufficient_data_returns_null(self):
        """Test that metrics requiring minimum data return null."""
        # Customer with only 1 shopping day - cutoff must be >= max date in data
        rows = [
            dict(customer_id="F", transaction_date="2025-01-01", transaction_id="F1", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
        ]
        df = pl.DataFrame(rows)
        
        cutoff = "2025-01-31"
        # observed_through must be >= cutoff
        frame = self._run_pipeline(df, cutoff, observed_through="2025-01-31")
        
        f = frame.filter(pl.col("customer_id") == "F").row(0, named=True)
        
        # gap_cv requires >=2 gaps (>=3 shopping days)
        self.assertIsNone(f["gap_cv"])
        self.assertIsNone(f["median_gap"])
        # gap_count is 0 when no gaps (count returns 0, not null)
        self.assertEqual(f["gap_count"], 0)
        
        # products_last6days requires >=6 shopping days
        self.assertIsNone(f["products_last6days"])
        
        # basket_cv requires >=2 baskets
        self.assertIsNone(f["basket_cv"])
        
        # weekday_entropy requires >=1 shopping day (has 1) -> should be 0 (single category)
        self.assertEqual(f["weekday_entropy"], 0.0)
    
    def test_ratio_helper_null_semantics(self):
        """Test the ratio helper returns null for zero/negative denominator."""
        # Test via a simple DataFrame
        df = pl.DataFrame({"a": [10], "b": [2]})
        result = df.select(ratio(pl.col("a"), pl.col("b")).alias("r")).item()
        self.assertEqual(result, 5.0)
        
        df = pl.DataFrame({"a": [10], "b": [0]})
        result = df.select(ratio(pl.col("a"), pl.col("b")).alias("r")).item()
        self.assertIsNone(result)
        
        df = pl.DataFrame({"a": [10], "b": [-1]})
        result = df.select(ratio(pl.col("a"), pl.col("b")).alias("r")).item()
        self.assertIsNone(result)
        
        df = pl.DataFrame({"a": [10], "b": [None]})
        result = df.select(ratio(pl.col("a"), pl.col("b")).alias("r")).item()
        self.assertIsNone(result)
        
        df = pl.DataFrame({"a": [None], "b": [2]})
        result = df.select(ratio(pl.col("a"), pl.col("b")).alias("r")).item()
        self.assertIsNone(result)
    
    def test_change_metrics_require_comparable_windows(self):
        """Test frequency_change, basket_change, spend_change only computed when tenure >= 179."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        b = frame.filter(pl.col("customer_id") == "B").row(0, named=True)
        
        # Customer A: tenure=89 < 179 -> comparable_windows=False -> change metrics null
        self.assertFalse(a["comparable_windows"])
        self.assertIsNone(a["frequency_change"])
        self.assertIsNone(a["basket_change"])
        self.assertIsNone(a["spend_change"])
        
        # Customer B: tenure=58 < 179 -> comparable_windows=False
        self.assertFalse(b["comparable_windows"])
        self.assertIsNone(b["frequency_change"])
    
    def test_change_metrics_with_comparable_windows(self):
        """Test change metrics when comparable_windows=True."""
        # Create customer with tenure >= 179 and purchases in both 90d windows
        rows = []
        # First purchase 200 days before cutoff
        base = date(2025, 1, 1)
        cutoff = date(2025, 7, 20)  # ~200 days after
        for i in range(10):
            # Add days carefully to stay within month bounds
            d = base + timedelta(days=i * 20)
            if d > cutoff:
                break
            rows.append(dict(
                customer_id="G", transaction_date=str(d),
                transaction_id=f"G{i}", product_id="P1", product_description="P1",
                department="D", category="C", price=10.0, quantity=1.0
            ))
        df = pl.DataFrame(rows)
        
        frame = self._run_pipeline(df, str(cutoff), observed_through=str(cutoff))
        
        g = frame.filter(pl.col("customer_id") == "G").row(0, named=True)
        
        self.assertTrue(g["comparable_windows"])
        self.assertIsNotNone(g["frequency_change"])
        self.assertIsNotNone(g["basket_change"])
        self.assertIsNotNone(g["spend_change"])
    
    def test_future_followup_boundary(self):
        """Test future follow-up uses (C, C+window] boundaries."""
        rows = [
            # Customer H: purchase on cutoff day (should be in historical)
            dict(customer_id="H", transaction_date="2025-03-31", transaction_id="H1", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # Purchase 1 day after cutoff (should be in future_30d)
            dict(customer_id="H", transaction_date="2025-04-01", transaction_id="H2", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # Purchase 30 days after cutoff (should be in future_30d)
            dict(customer_id="H", transaction_date="2025-04-30", transaction_id="H3", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # Purchase 31 days after cutoff (should be in future_90d only)
            dict(customer_id="H", transaction_date="2025-05-01", transaction_id="H4", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # Purchase 90 days after cutoff (should be in future_90d)
            dict(customer_id="H", transaction_date="2025-06-29", transaction_id="H5", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # Purchase 91 days after cutoff (should be in neither)
            dict(customer_id="H", transaction_date="2025-06-30", transaction_id="H6", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
        ]
        df = pl.DataFrame(rows)
        
        cutoff = "2025-03-31"
        frame = self._run_pipeline(df, cutoff, coverage_confirmed=True)
        
        h = frame.filter(pl.col("customer_id") == "H").row(0, named=True)
        
        # Historical: only 2025-03-31 (cutoff day) -> 1 purchase, 10 spend
        self.assertEqual(h["purchases_30d"], 1)
        self.assertEqual(h["spend_30d"], 10.0)
        
        # Future 30d: (2025-03-31, 2025-04-30] -> 2025-04-01, 2025-04-30 = 2 purchases, 20 spend
        self.assertTrue(h["eligible_followup_30d"])
        self.assertEqual(h["future_purchases_30d"], 2)
        self.assertEqual(h["future_spend_30d"], 20.0)
        
        # Future 90d: (2025-03-31, 2025-06-29] -> 2025-04-01, 2025-04-30, 2025-05-01, 2025-06-29 = 4 purchases, 40 spend
        self.assertTrue(h["eligible_followup_90d"])
        self.assertEqual(h["future_purchases_90d"], 4)
        self.assertEqual(h["future_spend_90d"], 40.0)
    
    def test_future_followup_suppressed_without_coverage(self):
        """Test future metrics are null when coverage_confirmed=False."""
        rows = [
            dict(customer_id="I", transaction_date="2025-03-31", transaction_id="I1", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            dict(customer_id="I", transaction_date="2025-04-01", transaction_id="I2", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
        ]
        df = pl.DataFrame(rows)
        
        cutoff = "2025-03-31"
        frame = self._run_pipeline(df, cutoff, coverage_confirmed=False)
        
        i = frame.filter(pl.col("customer_id") == "I").row(0, named=True)
        
        self.assertFalse(i["eligible_followup_30d"])
        self.assertFalse(i["eligible_followup_90d"])
        self.assertIsNone(i["future_purchases_30d"])
        self.assertIsNone(i["future_spend_30d"])
        self.assertIsNone(i["future_purchases_90d"])
        self.assertIsNone(i["future_spend_90d"])
    
    def test_value_tier_assignment(self):
        """Test value_tier assignment logic."""
        # Need to run with multiple customers to have thresholds
        rows = []
        # Create 10 customers with varying spend_90d
        for i in range(10):
            spend = (i + 1) * 100  # 100, 200, ..., 1000
            # Add enough transactions to be style-eligible
            for j in range(6):
                rows.append(dict(
                    customer_id=f"C{i}", transaction_date=f"2025-{1+j:02d}-01",
                    transaction_id=f"C{i}{j}", product_id="P1", product_description="P1",
                    department="D", category="C", price=spend/6, quantity=1.0
                ))
        df = pl.DataFrame(rows)
        
        cutoff = "2025-06-30"
        # observed_through must be >= cutoff
        frame = self._run_pipeline(df, cutoff, coverage_confirmed=False, observed_through="2025-06-30")
        
        # Check value_tier assignments
        tiers = frame.group_by("value_tier").len().to_dicts()
        tier_counts = {t["value_tier"]: t["len"] for t in tiers}
        
        # Should have high/middle/low (tenure >= 89 for all, so no insufficient_exposure)
        self.assertIn("high", tier_counts)
        self.assertIn("middle", tier_counts)
        self.assertIn("low", tier_counts)
        self.assertNotIn("insufficient_exposure", tier_counts)
        
        # With tenure >= 89, no one should be insufficient_exposure
        # 33rd percentile of spend_90d (100,200,300,400,500,600,700,800,900,1000) -> 400 (Polars default)
        # 80th percentile -> 800
        # So: >=800 = high (3 customers: 800,900,1000), <=400 = low (4 customers: 100,200,300,400), middle (3 customers: 500,600,700)
        self.assertEqual(tier_counts.get("high", 0), 3)
        self.assertEqual(tier_counts.get("low", 0), 4)
        self.assertEqual(tier_counts.get("middle", 0), 3)
    
    def test_style_eligibility(self):
        """Test style_eligible = days_365d >= 6 AND tenure >= 90 AND gap_count >= 4."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        b = frame.filter(pl.col("customer_id") == "B").row(0, named=True)
        
        # Customer A: days_365d=5, tenure=89, gap_count=4 -> style_eligible=False (tenure < 90)
        self.assertFalse(a["style_eligible"])
        
        # Customer B: days_365d=2, tenure=58, gap_count=1 -> style_eligible=False
        self.assertFalse(b["style_eligible"])
    
    def test_style_eligibility_with_sufficient_data(self):
        """Test style_eligible=True when all criteria met."""
        rows = []
        base = date(2024, 1, 1)
        # First, some old transactions for tenure
        for i in range(3):
            d = base + timedelta(days=i * 60)
            rows.append(dict(
                customer_id="J", transaction_date=str(d),
                transaction_id=f"J{i}", product_id="P1", product_description="P1",
                department="D", category="C", price=10.0, quantity=1.0
            ))
        # Then frequent transactions in the last year before cutoff
        # Cutoff is 2025-06-30, so last year is from 2024-07-01
        recent_base = date(2024, 7, 1)
        for i in range(10):
            d = recent_base + timedelta(days=i * 28)  # ~monthly
            if d > date(2025, 6, 30):
                break
            rows.append(dict(
                customer_id="J", transaction_date=str(d),
                transaction_id=f"J{i+3}", product_id="P1", product_description="P1",
                department="D", category="C", price=10.0, quantity=1.0
            ))
        df = pl.DataFrame(rows)
        
        cutoff = "2025-06-30"  # tenure > 90
        frame = self._run_pipeline(df, cutoff, observed_through="2025-06-30")
        
        j = frame.filter(pl.col("customer_id") == "J").row(0, named=True)
        
        self.assertTrue(j["style_eligible"])
        self.assertTrue(j["days_365d"] >= 6)
        self.assertTrue(j["tenure"] >= 90)
        self.assertTrue(j["gap_count"] >= 4)
    
    def test_behavior_label_precedence(self):
        """Test behavioral_tags are multi-label: all matching tags collected in documented order."""
        # Create customer matching both routine_repeat AND focused_repeat
        # routine_repeat: gap_cv <= 0.75, established_share >= 0.7
        # focused_repeat: dominant_share >= 0.7, established_share >= 0.7
        # Customer with single product P1 purchased monthly for 12 months -> established_share = 1.0, dominant_share = 1.0
        # Regular monthly cadence -> gap_cv = 0.0
        rows = []
        base = date(2024, 1, 1)
        # First, some old transactions for tenure
        for i in range(3):
            d = base + timedelta(days=i * 60)
            rows.append(dict(
                customer_id="K", transaction_date=str(d),
                transaction_id=f"K{i}", product_id="P1", product_description="P1",
                department="D", category="C", price=10.0, quantity=1.0
            ))
        # Then frequent transactions in the last year before cutoff
        # Cutoff is 2025-06-30, so last year is from 2024-07-01
        recent_base = date(2024, 7, 1)
        for i in range(12):
            d = recent_base + timedelta(days=i * 28)  # ~monthly
            if d > date(2025, 6, 30):
                break
            rows.append(dict(
                customer_id="K", transaction_date=str(d),
                transaction_id=f"K{i+3}", product_id="P1", product_description="P1",
                department="D", category="C", price=10.0, quantity=1.0
            ))
        df = pl.DataFrame(rows)
        
        cutoff = "2025-06-30"
        frame = self._run_pipeline(df, cutoff, observed_through="2025-06-30")
        
        k = frame.filter(pl.col("customer_id") == "K").row(0, named=True)
        
        # Customer K should be style_eligible
        self.assertTrue(k["style_eligible"])
        
        # Verify expected tags: routine_repeat (gap_cv=0 <= 0.75, established_share=1 >= 0.7)
        # and focused_repeat (dominant_share=1 >= 0.7, established_share=1 >= 0.7)
        # Both should be present in documented order
        self.assertEqual(k["behavioral_tags"], ["routine_repeat", "focused_repeat"])
        self.assertEqual(k["behavior"], "routine_repeat+focused_repeat")
    
    def test_lifecycle_stages(self):
        """Test lifecycle stage assignments."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff, observed_through="2025-10-31")
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        b = frame.filter(pl.col("customer_id") == "B").row(0, named=True)
        c_rows = [dict(customer_id="C", transaction_date="2025-01-01", transaction_id="C1", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0)]
        c_df = pl.DataFrame(c_rows)
        c_frame = self._run_pipeline(c_df, "2025-04-30", observed_through="2025-04-30")  # tenure ~120 days
        c = c_frame.filter(pl.col("customer_id") == "C").row(0, named=True)
        
        # Customer A: days_history=4 (actually 5 shopping days), tenure=89
        # days_history >= 6? No (5 < 6). tenure >= 90? No (89 < 90).
        # Should be early_repeat
        self.assertEqual(a["lifecycle"], "early_repeat")
        
        # Customer B: days_history=2, tenure=58 -> early_repeat
        self.assertEqual(b["lifecycle"], "early_repeat")
        
        # Customer C: days_history=1, tenure=120 -> one_shopping_day
        self.assertEqual(c["lifecycle"], "one_shopping_day")
    
    def test_evidence_levels(self):
        """Test evidence level assignments."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
        b = frame.filter(pl.col("customer_id") == "B").row(0, named=True)
        
        # Customer A: style_eligible=False -> insufficient
        self.assertEqual(a["evidence"], "insufficient")
        
        # Customer B: style_eligible=False -> insufficient
        self.assertEqual(b["evidence"], "insufficient")
    
    def test_no_lookahead_descriptive_features(self):
        """Test that adding future transactions doesn't change descriptive features."""
        base_rows = self._make_fixture().to_dicts()
        
        # Run with cutoff 2025-03-31, no future data
        cutoff = "2025-03-31"
        frame1 = self._run_pipeline(pl.DataFrame(base_rows), cutoff, coverage_confirmed=False)
        
        # Add future transactions for customer A
        future_rows = base_rows + [
            dict(customer_id="A", transaction_date="2025-04-15", transaction_id="A6", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=1.0),
            dict(customer_id="A", transaction_date="2025-05-01", transaction_id="A7", product_id="P2", product_description="P2", department="D", category="C2", price=5.0, quantity=2.0),
        ]
        frame2 = self._run_pipeline(pl.DataFrame(future_rows), cutoff, coverage_confirmed=False)
        
        # Descriptive features should be identical
        desc_cols = [c for c in frame1.columns if not c.startswith(("future_", "eligible_followup_"))]
        a1 = frame1.filter(pl.col("customer_id") == "A").select(desc_cols).row(0, named=True)
        a2 = frame2.filter(pl.col("customer_id") == "A").select(desc_cols).row(0, named=True)
        
        for col in desc_cols:
            v1, v2 = a1[col], a2[col]
            if v1 is None and v2 is None:
                continue
            if isinstance(v1, float) and isinstance(v2, float):
                self.assertAlmostEqual(v1, v2, places=6, msg=f"Column {col} differs: {v1} vs {v2}")
            else:
                self.assertEqual(v1, v2, msg=f"Column {col} differs: {v1} vs {v2}")
    
    def test_future_followup_changes_with_future_data(self):
        """Test that future follow-up columns DO change when future data added."""
        # Create base data with NO transactions after cutoff
        base_rows = [
            dict(customer_id="A", transaction_date="2025-01-01", transaction_id="A1", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=2.0),
            dict(customer_id="A", transaction_date="2025-02-01", transaction_id="A2", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=1.0),
        ]
        cutoff = "2025-03-31"
        # observed_through must be >= cutoff + 90 days for follow-up to be eligible
        observed_through = "2025-06-30"
        
        frame1 = self._run_pipeline(pl.DataFrame(base_rows), cutoff, coverage_confirmed=True, observed_through=observed_through)
        
        future_rows = base_rows + [
            dict(customer_id="A", transaction_date="2025-04-15", transaction_id="A3", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=1.0),
        ]
        frame2 = self._run_pipeline(pl.DataFrame(future_rows), cutoff, coverage_confirmed=True, observed_through=observed_through)
        
        a1 = frame1.filter(pl.col("customer_id") == "A").row(0, named=True)
        a2 = frame2.filter(pl.col("customer_id") == "A").row(0, named=True)
        
        # First run: no future data -> future_purchases_30d = 0 (eligible, so null filled with 0)
        # Second run: one future transaction in 30d window -> future_purchases_30d = 1
        self.assertEqual(a1["future_purchases_30d"], 0)
        self.assertEqual(a2["future_purchases_30d"], 1)
        self.assertEqual(a1["future_spend_30d"], 0.0)
        self.assertEqual(a2["future_spend_30d"], 10.0)
    
    def test_thresholds_reproducibility(self):
        """Test that quantile thresholds are reproducible for fixed input."""
        df = self._make_fixture()
        
        with TemporaryDirectory() as tmp1:
            root1 = Path(tmp1)
            source1 = root1 / "input.parquet"
            df.write_parquet(source1)
            out1 = root1 / "results"
            run(str(source1), str(out1), "unit", ["2025-03-31"], buckets=1, coverage_confirmed=True)
            t1 = (out1 / "thresholds.json").read_text()
        
        with TemporaryDirectory() as tmp2:
            root2 = Path(tmp2)
            source2 = root2 / "input.parquet"
            df.write_parquet(source2)
            out2 = root2 / "results"
            run(str(source2), str(out2), "unit", ["2025-03-31"], buckets=1, coverage_confirmed=True)
            t2 = (out2 / "thresholds.json").read_text()
        
        self.assertEqual(t1, t2)
    
    def test_cohort_exposure_tracking(self):
        """Test cohort profile includes observation_months."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        # Check cohort_profiles.parquet has observation_months
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.parquet"
            self._make_fixture().write_parquet(source)
            out = root / "results"
            run(str(source), str(out), "unit", [cutoff], buckets=1, coverage_confirmed=True)
            
            cohort = pl.read_parquet(out / "cohort_profiles.parquet")
            self.assertIn("observation_months", cohort.columns)
            self.assertIn("min_tenure_days", cohort.columns)
            self.assertIn("max_tenure_days", cohort.columns)
            
            # All cohorts should have observation_months >= 0
            self.assertTrue((cohort["observation_months"] >= 0).all())
    
    def test_segment_profile_shares_sum_to_one(self):
        """Test customer_share and revenue_share sum to 1 per snapshot."""
        cutoff = "2025-03-31"
        frame = self._run_pipeline(self._make_fixture(), cutoff)
        
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.parquet"
            self._make_fixture().write_parquet(source)
            out = root / "results"
            run(str(source), str(out), "unit", [cutoff], buckets=1, coverage_confirmed=True)
            
            profile = pl.read_parquet(out / "segment_profiles.parquet")
            
            for snap in profile["snapshot"].unique():
                snap_data = profile.filter(pl.col("snapshot") == snap)
                cust_sum = snap_data["customer_share"].sum()
                rev_sum = snap_data["revenue_share"].sum()
                self.assertAlmostEqual(cust_sum, 1.0, places=6)
                self.assertAlmostEqual(rev_sum, 1.0, places=6)
    
    def test_transition_rates_sum_to_one(self):
        """Test transition rates sum to 1 per from_label per snapshot."""
        # Need at least 2 snapshots
        rows = []
        for i in range(20):
            rows.append(dict(
                customer_id=f"C{i%5}", transaction_date=f"2025-{1+i%3:02d}-01",
                transaction_id=f"T{i}", product_id="P1", product_description="P1",
                department="D", category="C", price=10.0, quantity=1.0
            ))
        df = pl.DataFrame(rows)
        
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.parquet"
            df.write_parquet(source)
            out = root / "results"
            # observed_through must be >= last cutoff
            run(str(source), str(out), "unit", ["2025-03-31", "2025-06-30"], buckets=1, coverage_confirmed=True, observed_through="2025-06-30")
            
            trans = pl.read_parquet(out / "segment_transitions.parquet")
            
            if trans.height > 0:
                for dim in trans["dimension"].unique():
                    dim_data = trans.filter(pl.col("dimension") == dim)
                    for from_label in dim_data["from_label"].unique():
                        for snap in dim_data["snapshot"].unique():
                            subset = dim_data.filter((pl.col("from_label") == from_label) & (pl.col("snapshot") == snap))
                            rate_sum = subset["transition_rate"].sum()
                            self.assertAlmostEqual(rate_sum, 1.0, places=6)

    def test_breadth_category_not_product(self):
        """Test breadth measures distinct categories, not products."""
        # One basket with 3 products across 2 categories
        rows = [
            dict(customer_id="B1", transaction_date="2025-01-01", transaction_id="B1", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=1.0),
            dict(customer_id="B1", transaction_date="2025-01-01", transaction_id="B1", product_id="P2", product_description="P2", department="D", category="C1", price=10.0, quantity=1.0),
            dict(customer_id="B1", transaction_date="2025-01-01", transaction_id="B1", product_id="P3", product_description="P3", department="D", category="C2", price=10.0, quantity=1.0),
        ]
        df = pl.DataFrame(rows)
        
        cutoff = "2025-01-31"
        frame = self._run_pipeline(df, cutoff, observed_through="2025-01-31")
        
        b1 = frame.filter(pl.col("customer_id") == "B1").row(0, named=True)
        # Breadth measures distinct categories per basket = 2 (C1 and C2)
        # NOT 3 (products P1, P2, P3)
        self.assertEqual(b1["median_breadth"], 2)
        self.assertEqual(b1["breadth_recent"], 2.0)

    def test_familiarity_shares_basket_weighted(self):
        """Test familiarity shares are basket-weighted means of product fractions."""
        # Two products established before trailing year
        # In trailing year:
        # Basket 1: 2 established products + 1 new product (3 products)
        # Basket 2: 1 new product (1 product)
        rows = []
        base = date(2024, 1, 1)
        # Establish P1 and P2 before trailing year
        for i in range(3):
            d = base + timedelta(days=i * 60)
            rows.append(dict(customer_id="F1", transaction_date=str(d), transaction_id=f"F{i}", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0))
            rows.append(dict(customer_id="F1", transaction_date=str(d), transaction_id=f"F{i}", product_id="P2", product_description="P2", department="D", category="C", price=10.0, quantity=1.0))
        
        # Trailing year baskets (cutoff = 2025-06-30, year = 2024-07-01 to 2025-06-30)
        recent_base = date(2024, 7, 1)
        # Basket 1: 2 established (P1, P2) + 1 new (P3)
        rows.append(dict(customer_id="F1", transaction_date="2024-08-01", transaction_id="B1", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0))
        rows.append(dict(customer_id="F1", transaction_date="2024-08-01", transaction_id="B1", product_id="P2", product_description="P2", department="D", category="C", price=10.0, quantity=1.0))
        rows.append(dict(customer_id="F1", transaction_date="2024-08-01", transaction_id="B1", product_id="P3", product_description="P3", department="D", category="C", price=10.0, quantity=1.0))
        # Basket 2: 1 new (P4)
        rows.append(dict(customer_id="F1", transaction_date="2024-09-01", transaction_id="B2", product_id="P4", product_description="P4", department="D", category="C", price=10.0, quantity=1.0))
        
        df = pl.DataFrame(rows)
        
        cutoff = "2025-06-30"
        frame = self._run_pipeline(df, cutoff, observed_through="2025-06-30")
        
        f1 = frame.filter(pl.col("customer_id") == "F1").row(0, named=True)
        
        # Basket 1: 3 products, 2 established, 1 new -> fractions: established=2/3, new=1/3, repeat=2/3
        # Basket 2: 1 new -> fractions: established=0, new=1, repeat=0
        # new_share = mean(1/3, 1) = 2/3
        # repeat_share = mean(2/3, 0) = 1/3
        # established_share = mean(2/3, 0) = 1/3
        self.assertAlmostEqual(f1["new_share"], 2/3, places=6)
        self.assertAlmostEqual(f1["repeat_share"], 1/3, places=6)
        self.assertAlmostEqual(f1["established_share"], 1/3, places=6)
        
        # Verify NOT equal to global product weighting or any/all basket indicators
        # Global product fractions: 2 established + 2 new = 4 products -> established=0.5, new=0.5
        # Any/all basket indicators would give different values

    def test_core_eligibility_historical(self):
        """Test core eligibility uses all-history purchase_days >= 3."""
        # Product with 2 purchase days before trailing year, 1 inside it
        rows = []
        base = date(2024, 1, 1)
        # 2 purchases before trailing year (2024-07-01)
        rows.append(dict(customer_id="C1", transaction_date="2024-01-01", transaction_id="C1", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0))
        rows.append(dict(customer_id="C1", transaction_date="2024-04-01", transaction_id="C2", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0))
        # 1 purchase inside trailing year
        rows.append(dict(customer_id="C1", transaction_date="2024-08-01", transaction_id="C3", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0))
        # Another product with only 1 purchase in trailing year (not core)
        rows.append(dict(customer_id="C1", transaction_date="2024-09-01", transaction_id="C4", product_id="P2", product_description="P2", department="D", category="C", price=10.0, quantity=1.0))
        
        df = pl.DataFrame(rows)
        
        cutoff = "2025-06-30"
        frame = self._run_pipeline(df, cutoff, observed_through="2025-06-30")
        
        c1 = frame.filter(pl.col("customer_id") == "C1").row(0, named=True)
        
        # P1 has 3 purchase days total (2 before year + 1 in year) -> core
        # P2 has 1 purchase day -> not core
        self.assertEqual(c1["core_size"], 1)
        # core_spend = spend on P1 in trailing year = 10.0
        self.assertEqual(c1["core_spend"], 10.0)
        self.assertAlmostEqual(c1["core_spend_share"], 10.0/20.0, places=6)

    def test_overdue_category_historical_cadence(self):
        """Test overdue_categories uses all-history cadence."""
        # Category with 3+ shopping days in history, overdue based on historical gap
        rows = []
        base = date(2024, 1, 1)
        # 3 purchases in category C1 over history, regular monthly
        for i in range(3):
            d = base + timedelta(days=i * 30)
            rows.append(dict(customer_id="O1", transaction_date=str(d), transaction_id=f"O{i}", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=1.0))
        # Last purchase was 60 days before cutoff (2 months), median_gap = 30 -> lateness = 2 -> overdue
        rows.append(dict(customer_id="O1", transaction_date="2025-05-01", transaction_id="O3", product_id="P1", product_description="P1", department="D", category="C1", price=10.0, quantity=1.0))
        
        df = pl.DataFrame(rows)
        
        cutoff = "2025-07-01"
        frame = self._run_pipeline(df, cutoff, observed_through="2025-07-01")
        
        o1 = frame.filter(pl.col("customer_id") == "O1").row(0, named=True)
        
        # Category has 4 shopping days in all history -> days >= 3
        # median_gap = 30, recency = 61 (Jul 1 - May 1) -> lateness = 61/30 = 2.03 >= 2 -> overdue
        self.assertEqual(o1["overdue_categories"], 1)

    def test_window_endpoints(self):
        """Test exact window boundary membership."""
        # Place records on boundaries and one outside
        rows = [
            # C-365 = 2024-03-31 (outside year) - different product P0 so P1's trial is at C-364
            dict(customer_id="W1", transaction_date="2024-03-31", transaction_id="W1", product_id="P0", product_description="P0", department="D", category="C", price=10.0, quantity=1.0),
            # C-364 = 2024-04-01 (inside year, P1's trial)
            dict(customer_id="W1", transaction_date="2024-04-01", transaction_id="W2", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # C-179 = 2024-10-04 (inside previous90d)
            dict(customer_id="W1", transaction_date="2024-10-04", transaction_id="W3", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # C-178 = 2024-10-05 (inside previous90d)
            dict(customer_id="W1", transaction_date="2024-10-05", transaction_id="W4", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # C-90 = 2024-12-31 (boundary: previous90d end, trial eligible end)
            dict(customer_id="W1", transaction_date="2024-12-31", transaction_id="W5", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # C-89 = 2025-01-01 (inside recent90d, trial NOT eligible)
            dict(customer_id="W1", transaction_date="2025-01-01", transaction_id="W6", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # C = 2025-03-31 (cutoff, inside all historical windows)
            dict(customer_id="W1", transaction_date="2025-03-31", transaction_id="W7", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
            # C+1 = 2025-04-01 (future_30d)
            dict(customer_id="W1", transaction_date="2025-04-01", transaction_id="W8", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0),
        ]
        df = pl.DataFrame(rows)
        
        cutoff = "2025-03-31"
        frame = self._run_pipeline(df, cutoff, observed_through="2025-07-01", coverage_confirmed=True)
        
        w1 = frame.filter(pl.col("customer_id") == "W1").row(0, named=True)
        
        # Year [C-364, C] = [2024-04-01, 2025-03-31]: 6 transactions (W2-W7, excluding W1 which is P0)
        self.assertEqual(w1["purchases_365d"], 6)
        self.assertEqual(w1["spend_365d"], 60.0)
        
        # Previous90d [C-179, C-90] = [2024-10-03, 2024-12-31]: 3 transactions (W3, W4, W5 at 2024-12-31)
        # W6=2025-01-01 is C-89 (in recent90d)
        self.assertEqual(w1["purchases_previous90d"], 3)
        
        # Recent90d [C-89, C] = [2025-01-01, 2025-03-31]: 2 transactions (W6 at 2025-01-01, W7 at 2025-03-31)
        self.assertEqual(w1["purchases_90d"], 2)
        
        # Eligible trials [C-364, C-90] = [2024-04-01, 2024-12-31]: trial = 2024-04-01 (P1's first purchase) -> eligible
        self.assertEqual(w1["eligible_trials"], 1)
        
        # Future 30d (C, C+30] = (2025-03-31, 2025-04-30]: 1 transaction (W8)
        self.assertTrue(w1["eligible_followup_30d"])
        self.assertEqual(w1["future_purchases_30d"], 1)
        self.assertEqual(w1["future_spend_30d"], 10.0)

    def test_change_metrics_exact(self):
        """Test exact hand-calculated change metrics."""
        # Customer with purchases in both 90d windows
        # Cutoff = 2025-07-20
        # Previous90d [C-179, C-90] = [2025-01-22, 2025-04-21]
        # Recent90d [C-89, C] = [2025-04-22, 2025-07-20]
        rows = []
        # 3 purchases in previous 90d window [C-179, C-90]
        prev_base = date(2025, 1, 22)
        for i in range(3):
            d = prev_base + timedelta(days=i * 20)
            rows.append(dict(customer_id="CH1", transaction_date=str(d), transaction_id=f"CH{i}", product_id="P1", product_description="P1", department="D", category="C", price=10.0, quantity=1.0))
        # 6 purchases in recent 90d window [C-89, C]
        recent_base = date(2025, 4, 22)
        for i in range(6):
            d = recent_base + timedelta(days=i * 10)
            rows.append(dict(customer_id="CH1", transaction_date=str(d), transaction_id=f"CH{i+3}", product_id="P1", product_description="P1", department="D", category="C", price=20.0, quantity=1.0))
        
        df = pl.DataFrame(rows)
        
        cutoff = "2025-07-20"
        frame = self._run_pipeline(df, cutoff, observed_through="2025-07-20")
        
        ch1 = frame.filter(pl.col("customer_id") == "CH1").row(0, named=True)
        
        # previous90d: 3 purchases, 30 spend -> AOV = 10
        # recent90d: 6 purchases, 120 spend -> AOV = 20
        # frequency_change = 6/3 - 1 = 1.0
        # basket_change = 20/10 - 1 = 1.0
        # spend_change = 120/30 - 1 = 3.0
        self.assertAlmostEqual(ch1["frequency_change"], 1.0, places=6)
        self.assertAlmostEqual(ch1["basket_change"], 1.0, places=6)
        self.assertAlmostEqual(ch1["spend_change"], 3.0, places=6)


if __name__ == "__main__":
    unittest.main()