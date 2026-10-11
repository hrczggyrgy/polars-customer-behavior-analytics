"""Synthetic integration checks; run with python -m unittest test_retail_polars_workflow."""

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from polars.testing import assert_frame_equal

from retail_analytics.core.retail_polars_workflow import prepare, run


def fixture():
    rows = []
    for i, day in enumerate(
        [
            "2025-01-10",
            "2025-02-10",
            "2025-03-10",
            "2025-04-10",
            "2025-05-10",
            "2025-05-20",
            "2025-06-10",
            "2025-07-10",
            "2025-10-31",
        ]
    ):
        rows.append(
            dict(
                customer_id="A",
                transaction_date=day,
                transaction_id=f"A{i}",
                product_id="P1",
                product_description="Example 500 ml",
                department="D",
                category="C",
                price=10.0,
                quantity=2.0,
            )
        )
    rows.append(
        dict(
            customer_id="A",
            transaction_date="2025-05-20",
            transaction_id="AX",
            product_id="P2",
            product_description="Example",
            department="D",
            category="Other",
            price=5.0,
            quantity=1.0,
        )
    )
    rows.append(
        dict(
            customer_id="B",
            transaction_date="2025-04-15",
            transaction_id="B1",
            product_id="P3",
            product_description="Example",
            department="D",
            category="C",
            price=7.0,
            quantity=1.0,
        )
    )
    return pl.DataFrame(rows)


class WorkflowTests(unittest.TestCase):
    def test_features_coverage_and_repeat_semantics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.parquet"
            fixture().write_parquet(source)
            out = root / "results"
            run(
                str(source),
                str(out),
                "unit",
                ["2025-05-31", "2025-10-31"],
                buckets=2,
                coverage_confirmed=True,
            )
            frame = pl.read_parquet(str(out / "snapshots" / "2025-05-31" / "*.parquet"))
            a = frame.filter(pl.col("customer_id") == "A").row(0, named=True)
            b = frame.filter(pl.col("customer_id") == "B").row(0, named=True)
            self.assertEqual(a["purchases_history"], 7)
            self.assertEqual(a["days_history"], 6)
            self.assertAlmostEqual(a["spend_history"], 125.0)
            self.assertEqual(a["gap_count"], 5)
            self.assertAlmostEqual(a["new_share"], 2 / 7)
            self.assertAlmostEqual(a["repeat_share"], 5 / 7)
            self.assertAlmostEqual(a["established_share"], 4 / 7)
            self.assertAlmostEqual(a["new_share"] + a["repeat_share"], 1.0)
            self.assertEqual(a["eligible_trials"], 1)
            self.assertEqual(a["repeated_trials"], 1)
            self.assertEqual(a["future_purchases_30d"], 1)
            self.assertAlmostEqual(a["future_spend_30d"], 20.0)
            self.assertIsNone(b["median_gap"])
            self.assertIsNone(b["trial_conversion"])
            late = pl.read_parquet(str(out / "snapshots" / "2025-10-31" / "*.parquet"))
            self.assertTrue(late["future_spend_90d"].is_null().all())
            trans = pl.read_parquet(out / "segment_transitions.parquet")
            sums = trans.group_by(["from_date", "snapshot", "dimension", "from_label"]).agg(
                pl.col("transition_rate").sum()
            )
            self.assertLess((sums["transition_rate"] - 1).abs().max(), 1e-9)

    def test_unconfirmed_coverage_suppresses_outcomes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.parquet"
            fixture().write_parquet(source)
            out = root / "unconfirmed"
            run(str(source), str(out), "unit", ["2025-05-31"], buckets=1, coverage_confirmed=False)
            frame = pl.read_parquet(str(out / "snapshots" / "2025-05-31" / "*.parquet"))
            for column in [
                "future_purchases_30d",
                "future_spend_30d",
                "future_purchases_90d",
                "future_spend_90d",
            ]:
                self.assertTrue(frame[column].is_null().all())
            for column in ["eligible_followup_30d", "eligible_followup_90d"]:
                self.assertFalse(frame[column].any())
            profile = pl.read_parquet(out / "segment_profiles.parquet")
            self.assertTrue(profile["future_purchase_rate_30d"].is_null().all())

    def test_price_modes_and_no_lookahead(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            full = fixture()
            cutoff = "2025-05-20"
            cases = [
                ("unit", full, "unit"),
                (
                    "total",
                    full.with_columns((pl.col("price") * pl.col("quantity")).alias("price")),
                    "line_total",
                ),
                ("past", full.filter(pl.col("transaction_date") <= cutoff), "unit"),
            ]
            for name, data, mode in cases:
                source = root / f"{name}.parquet"
                data.write_parquet(source)
                out = root / name
                run(str(source), str(out), mode, [cutoff], buckets=1, coverage_confirmed=False)
                frames.append(
                    pl.read_parquet(str(out / "snapshots" / cutoff / "*.parquet")).sort(
                        "customer_id"
                    )
                )
            columns = [
                x for x in frames[0].columns if not x.startswith(("future_", "eligible_followup_"))
            ]
            assert_frame_equal(frames[0].select(columns), frames[1].select(columns))
            assert_frame_equal(frames[0].select(columns), frames[2].select(columns))

    def test_reject_conflicting_observation_end(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.parquet"
            fixture().write_parquet(source)
            for confirmed in [False, True]:
                with self.subTest(coverage_confirmed=confirmed):
                    with self.assertRaisesRegex(
                        ValueError, "observed_through precedes the latest valid transaction date"
                    ):
                        run(
                            str(source),
                            str(root / f"conflict_{confirmed}"),
                            "unit",
                            ["2025-05-31"],
                            buckets=1,
                            observed_through="2025-05-31",
                            coverage_confirmed=confirmed,
                        )

    def test_accept_equal_or_later_observation_end(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.parquet"
            data = fixture()
            data.write_parquet(source)
            for through in ["2025-10-31", "2025-11-30"]:
                for confirmed in [False, True]:
                    with self.subTest(observed_through=through, coverage_confirmed=confirmed):
                        out = root / f"valid_{through}_{confirmed}"
                        out.mkdir()
                        paths, end, _ = prepare(
                            str(source), out, "unit", 1, False, None, through, confirmed
                        )
                        self.assertEqual(end, date.fromisoformat(through))
                        self.assertEqual(
                            pl.read_parquet(out / "validated.parquet").height, data.height
                        )
                        self.assertEqual(len(paths), 1)

    def test_reject_invalid(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "invalid.parquet"
            fixture().with_columns(pl.lit(-1.0).alias("quantity")).write_parquet(source)
            with self.assertRaises(ValueError):
                run(str(source), str(root / "result"), "unit", ["2025-05-31"], buckets=1)


if __name__ == "__main__":
    unittest.main()
