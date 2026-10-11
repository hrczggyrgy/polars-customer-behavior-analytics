"""Observed patronage, not attitudinal loyalty or share of wallet."""
import argparse
import calendar
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path

VERSION = "patronage-0.1"
C = "customer_id"


@dataclass(frozen=True)
class Rules:
    min_tenure: int = 90
    min_days: int = 3
    min_active_months: int = 2
    min_complete_months: int = 3
    cadence_days: int = 6
    cadence_gaps: int = 4
    max_gap_cv: float = 0.75
    active_share: float = 0.60
    overdue_multiple: float = 2.0
    resume_window: int = 30

    def __post_init__(self):
        for name in ["min_tenure", "min_days", "min_active_months", "min_complete_months", "cadence_days", "cadence_gaps", "resume_window"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Evidence thresholds must be positive integers")
        if not all(math.isfinite(v) for v in [self.max_gap_cv, self.active_share, self.overdue_multiple]):
            raise ValueError("Thresholds must be finite")
        if self.max_gap_cv < 0 or not 0 <= self.active_share <= 1 or self.overdue_multiple <= 1:
            raise ValueError("Invalid cadence, activity or overdue threshold")


def month_bounds(first, cutoff):
    if cutoff < first:
        raise ValueError("Cutoff precedes first observation")
    start = first.replace(day=1)
    if first.day != 1:
        start = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    end = cutoff.replace(day=1)
    if cutoff.day != calendar.monthrange(cutoff.year, cutoff.month)[1]:
        end = (end - timedelta(days=1)).replace(day=1)
    count = max(0, (end.year - start.year) * 12 + end.month - start.month + 1)
    return start, end, count


def classify(row, rules=Rules(), capture_confirmed=False):
    if not capture_confirmed:
        return "capture_unconfirmed"
    days, tenure = row["days_history"], row["tenure"]
    if days == 1:
        return "long_observed_one_day" if tenure >= rules.min_tenure else "first_observed"
    if tenure < rules.min_tenure:
        return "recent_repeat"
    enough = days >= rules.min_days and row["complete_months"] >= rules.min_complete_months and row["active_complete_months"] >= rules.min_active_months
    if not enough:
        return "insufficient_history"
    cv, gap = row.get("gap_cv"), row.get("median_gap")
    regular = row.get("days_365d", 0) >= rules.cadence_days and row.get("gap_count", 0) >= rules.cadence_gaps and gap is not None and math.isfinite(gap) and gap > 0 and cv is not None and math.isfinite(cv) and 0 <= cv <= rules.max_gap_cv
    if regular and row["recency"] / gap >= rules.overdue_multiple:
        return "recurring_overdue"
    if row.get("resume_events", 0) > 0:
        return "reactivated_repeat"
    if regular and row.get("active_period_share") is not None and row["active_period_share"] >= rules.active_share:
        return "established_recurring"
    return "established_occasional"


def label_frame(frame, rules=Rules(), capture_confirmed=False):
    import polars as pl
    enough = ((pl.col("days_history") >= rules.min_days) & (pl.col("complete_months") >= rules.min_complete_months) & (pl.col("active_complete_months") >= rules.min_active_months)).fill_null(False)
    regular = ((pl.col("days_365d") >= rules.cadence_days) & (pl.col("gap_count") >= rules.cadence_gaps) & pl.col("median_gap").is_finite() & (pl.col("median_gap") > 0) & pl.col("gap_cv").is_finite() & (pl.col("gap_cv") >= 0) & (pl.col("gap_cv") <= rules.max_gap_cv)).fill_null(False)
    overdue = (regular & (pl.col("recency") / pl.col("median_gap") >= rules.overdue_multiple)).fill_null(False)
    result = (pl.when(~pl.lit(bool(capture_confirmed))).then(pl.lit("capture_unconfirmed"))
        .when(pl.col("days_history") == 1).then(pl.when(pl.col("tenure") >= rules.min_tenure).then(pl.lit("long_observed_one_day")).otherwise(pl.lit("first_observed")))
        .when(pl.col("tenure") < rules.min_tenure).then(pl.lit("recent_repeat"))
        .when(~enough).then(pl.lit("insufficient_history"))
        .when(overdue).then(pl.lit("recurring_overdue"))
        .when(pl.col("resume_events") > 0).then(pl.lit("reactivated_repeat"))
        .when(regular & (pl.col("active_period_share") >= rules.active_share)).then(pl.lit("established_recurring"))
        .otherwise(pl.lit("established_occasional")))
    return frame.with_columns(result.alias("observed_patronage"), regular.alias("patronage_cadence_eligible"), pl.lit(VERSION).alias("patronage_version"))


def destination(root, rules, capture_confirmed):
    manifest = json.loads((Path(root) / "run.json").read_text())
    specification = dict(version=VERSION, rules=asdict(rules), capture_confirmed=bool(capture_confirmed), source_run=manifest)
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:16]
    return Path(root) / "patronage" / key, specification


def build(root, rules=Rules(), capture_confirmed=False):
    import polars as pl
    from retail_analytics.core.retail_polars_workflow import sink
    root = Path(root)
    source_run = json.loads((root / "run.json").read_text())
    capture_confirmed = bool(capture_confirmed and source_run.get("coverage_confirmed"))
    target, specification = destination(root, rules, capture_confirmed)
    if (target / "manifest.json").exists():
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    try:
        for text in source_run["snapshots"]:
            cutoff = date.fromisoformat(text)
            for source in sorted((root / "snapshots" / text).glob("*.parquet")):
                frame = pl.scan_parquet(source)
                event = root / "events" / text / source.stem
                days = pl.scan_parquet(event / "days.parquet")
                first = pl.col("first_purchase")
                start = pl.when(first.dt.day() == 1).then(first.dt.truncate("1mo")).otherwise(first.dt.offset_by("1mo").dt.truncate("1mo"))
                end = month_bounds(cutoff.replace(day=1), cutoff)[1]
                frame = frame.with_columns(start.alias("complete_start"), pl.lit(end).alias("complete_end"))
                periods = (pl.lit(end.year) - pl.col("complete_start").dt.year()) * 12 + pl.lit(end.month) - pl.col("complete_start").dt.month() + 1
                frame = frame.with_columns(periods.clip(lower_bound=0).cast(pl.Int64).alias("complete_months"))
                bounds = frame.select(C, "complete_start", "complete_end")
                monthly = days.join(bounds, on=C, how="inner", validate="m:1").with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
                monthly = monthly.group_by(C).agg(pl.col("month").n_unique().alias("active_months_observed"), pl.col("month").filter(pl.col("month").is_between(pl.col("complete_start"), pl.col("complete_end"))).n_unique().alias("active_complete_months"))
                resume = days.group_by(C).agg(((pl.col("date") >= cutoff - timedelta(days=rules.resume_window - 1)) & (pl.col("baseline_gap") > 0) & (pl.col("gap") / pl.col("baseline_gap") >= rules.overdue_multiple)).fill_null(False).sum().alias("resume_events"))
                line = pl.scan_parquet(root / "partitions" / source.name).filter(pl.col("date").is_between(cutoff - timedelta(days=364), cutoff))
                keys = [C, "department", "category"]
                cat_days = line.select(keys + ["date"]).unique().with_columns(pl.col("date").shift(1).over(keys, order_by="date").alias("previous"))
                cat_days = cat_days.with_columns((pl.col("date") - pl.col("previous")).dt.total_days().alias("category_gap"))
                categories = cat_days.group_by(keys).agg(pl.len().alias("category_days_365d"), pl.col("category_gap").count().alias("category_gap_count"), pl.col("category_gap").median().alias("category_median_gap"), pl.col("date").max().alias("category_last_purchase"))
                spend = line.group_by(keys).agg(pl.col("value").sum().alias("category_spend_365d"))
                categories = categories.join(spend, on=keys, how="left", validate="1:1")
                persistence = categories.group_by(C).agg(pl.len().alias("observed_categories_365d"), (pl.col("category_days_365d") >= 3).sum().alias("recurring_categories_365d"))
                frame = frame.join(monthly, on=C, how="left", validate="1:1").join(resume, on=C, how="left", validate="1:1").join(persistence, on=C, how="left", validate="1:1")
                frame = frame.with_columns(*[pl.col(f).fill_null(0) for f in ["active_complete_months", "active_months_observed", "resume_events", "observed_categories_365d", "recurring_categories_365d"]])
                frame = frame.with_columns(pl.when(pl.col("complete_months") > 0).then(pl.col("active_complete_months") / pl.col("complete_months")).otherwise(None).alias("active_period_share"), pl.when(pl.col("observed_categories_365d") > 0).then(pl.col("recurring_categories_365d") / pl.col("observed_categories_365d")).otherwise(None).alias("category_persistence_share"))
                labeled = label_frame(frame, rules, capture_confirmed)
                out = target / "snapshots" / text / source.name
                sink(labeled, out)
                category_labels = pl.scan_parquet(out).select(C, "observed_patronage")
                sink(categories.join(category_labels, on=C, how="inner", validate="m:1"), target / "categories" / text / source.name)
        marker = target / "manifest.tmp"
        marker.write_text(json.dumps(specification, indent=2))
        marker.replace(target / "manifest.json")
        return target
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--capture-confirmed", action="store_true")
    args = parser.parse_args()
    build(args.root, Rules(**json.loads(Path(args.rules).read_text())), args.capture_confirmed)
