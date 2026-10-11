"""Pure rule tests run without Polars; integration tests require Polars."""
import importlib.util
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
from patronage import Rules, month_bounds, classify, label_frame, build


def profile(**changes):
    row = dict(days_history=8, tenure=180, complete_months=5, active_complete_months=4, days_365d=8, gap_count=7, median_gap=20.0, gap_cv=.3, recency=10, active_period_share=.8, resume_events=0)
    row.update(changes)
    return row


class RuleTests(unittest.TestCase):
    def test_complete_calendar_months(self):
        self.assertEqual(month_bounds(date(2024, 1, 15), date(2024, 3, 15)), (date(2024, 2, 1), date(2024, 2, 1), 1))
        self.assertEqual(month_bounds(date(2024, 2, 1), date(2024, 2, 29))[2], 1)
        self.assertEqual(month_bounds(date(2024, 2, 15), date(2024, 2, 29))[2], 0)
        with self.assertRaises(ValueError):
            month_bounds(date(2024, 3, 1), date(2024, 2, 29))

    def test_capture_gate(self):
        self.assertEqual(classify(profile()), "capture_unconfirmed")

    def test_one_day_and_recent(self):
        self.assertEqual(classify(profile(days_history=1, tenure=20), capture_confirmed=True), "first_observed")
        self.assertEqual(classify(profile(days_history=1), capture_confirmed=True), "long_observed_one_day")
        self.assertEqual(classify(profile(tenure=40), capture_confirmed=True), "recent_repeat")

    def test_insufficient(self):
        self.assertEqual(classify(profile(complete_months=2), capture_confirmed=True), "insufficient_history")
        self.assertEqual(classify(profile(days_history=2), capture_confirmed=True), "insufficient_history")

    def test_recurring_occasional(self):
        self.assertEqual(classify(profile(), capture_confirmed=True), "established_recurring")
        self.assertEqual(classify(profile(gap_cv=1.2), capture_confirmed=True), "established_occasional")
        self.assertEqual(classify(profile(gap_cv=None), capture_confirmed=True), "established_occasional")

    def test_overdue_and_precedence(self):
        self.assertEqual(classify(profile(recency=40), capture_confirmed=True), "recurring_overdue")
        self.assertEqual(classify(profile(recency=40, resume_events=1), capture_confirmed=True), "recurring_overdue")
        self.assertEqual(classify(profile(resume_events=1), capture_confirmed=True), "reactivated_repeat")

    @unittest.skipUnless(Path(__file__).with_name("loyalty_app.py").exists(), "Patronage UI not installed")
    def test_adapter_reruns_preserve_original_renderer(self):
        import ast
        from types import SimpleNamespace
        calls = []
        app = SimpleNamespace(display_results=lambda r, c: calls.append((r, c)))
        st = SimpleNamespace(radio=lambda *a, **k: "Visual analytics")
        parsed = ast.parse(Path(__file__).with_name("loyalty_app.py").read_text())
        nodes = [n for n in parsed.body if isinstance(n, (ast.Assign, ast.FunctionDef)) or (isinstance(n, ast.If) and not (isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__"))]
        code = compile(ast.Module(body=nodes, type_ignores=[]), "adapter", "exec")
        for _ in range(2):
            namespace = dict(app=app, st=st, render=lambda *a: None)
            exec(code, namespace)
            app.display_results = namespace["display_results"]
            app.display_results("root", False)
        self.assertEqual(calls, [("root", False), ("root", False)])

    def test_threshold_validation(self):
        for changes in [dict(overdue_multiple=1.0), dict(active_share=1.2), dict(max_gap_cv=float("nan")), dict(min_days=0)]:
            with self.assertRaises(ValueError):
                Rules(**changes)


@unittest.skipUnless(importlib.util.find_spec("polars"), "Polars unavailable")
class IntegrationTests(unittest.TestCase):
    def test_vector_labels_match_reference(self):
        import polars as pl
        examples = [profile(), profile(days_history=1), profile(tenure=30), profile(complete_months=1), profile(recency=40), profile(resume_events=1), profile(gap_cv=None)]
        for confirmed in [False, True]:
            actual = label_frame(pl.DataFrame(examples).lazy(), capture_confirmed=confirmed).collect()["observed_patronage"].to_list()
            self.assertEqual(actual, [classify(r, capture_confirmed=confirmed) for r in examples])

    def test_build_reconciliation_and_cutoff_integrity(self):
        import polars as pl
        from retail_polars_workflow import run
        from test_retail_polars_workflow import fixture
        import json
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cutoff = "2025-05-31"
            frames = []
            for name, data in [("full", fixture()), ("past", fixture().filter(pl.col("transaction_date") <= cutoff))]:
                source = root / (name + ".parquet")
                data.write_parquet(source)
                result = root / name
                # observed_through must be >= max date in data
                observed_through = "2025-10-31" if name == "full" else cutoff
                run(str(source), str(result), "unit", [cutoff], buckets=1, coverage_confirmed=True, observed_through=observed_through)
                (result / "run.json").write_text(json.dumps(dict(snapshots=[cutoff], coverage_confirmed=True, source_sha256=name)))
                out = build(result, capture_confirmed=True)
                frame = pl.read_parquet(str(out / "snapshots" / cutoff / "*.parquet")).sort("customer_id")
                frames.append(frame)
                a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
                self.assertEqual(a["complete_months"], 4)
                self.assertEqual(a["active_complete_months"], 4)
                self.assertEqual(a["active_period_share"], 1.0)
                cats = pl.read_parquet(str(out / "categories" / cutoff / "*.parquet"))
                self.assertAlmostEqual(cats["category_spend_365d"].sum(), frame["spend_365d"].sum())
            from polars.testing import assert_frame_equal
            columns = [x for x in frames[0].columns if not x.startswith(("future_", "eligible_followup_"))]
            assert_frame_equal(frames[0].select(columns), frames[1].select(columns))


if __name__ == "__main__":
    unittest.main()
