"""Validated, process-isolated adapter for the existing Polars workflow."""
import argparse
import json
import math
import shutil
from datetime import date, timedelta
from pathlib import Path
import polars as pl
from retail_polars_workflow import COLS, prepare, snapshot, label, sink

MAX_ROWS = 2_000_000
MAX_CUTOFFS = 12


def wilson(successes, total):
    if total <= 0:
        return None, None
    if not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p, denom = successes / total, 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0, center - half), min(1, center + half)


def cutoffs(last, first, count):
    if not 1 <= count <= MAX_CUTOFFS or last < first:
        raise ValueError("Invalid cutoff range")
    result, cursor = [last], last
    for _ in range(count - 1):
        cursor = cursor.replace(day=1) - timedelta(days=1)
        if cursor < first:
            break
        result.append(cursor)
    return sorted(result)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=str, indent=2))


def inspect(job):
    root, source = Path(job["workspace"]), Path(job["source"])
    if job["kind"] == "parquet":
        with source.open("rb") as handle:
            start = handle.read(4)
            handle.seek(-4, 2)
            tail = handle.read(4)
        if start != b"PAR1" or tail != b"PAR1":
            raise ValueError("Not a supported, unencrypted Parquet file")
        raw = pl.scan_parquet(source)
    elif job["kind"] == "csv":
        raw = pl.scan_csv(source, separator=job.get("separator", ","), infer_schema=False, encoding="utf8")
    else:
        raise ValueError("Only Parquet and UTF-8 CSV are supported")
    schema = raw.collect_schema()
    names = {x: x.lstrip("\ufeff").strip().lower() for x in schema.names()}
    if len(set(names.values())) != len(names):
        raise ValueError("Column names collide after trimming/case normalization")
    raw = raw.rename(names)
    schema = raw.collect_schema()
    missing = sorted(set(COLS) - set(schema.names()))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if any(schema[x].is_nested() or schema[x] == pl.Binary for x in COLS):
        raise ValueError("Required columns must be scalar, not nested or binary")
    count = raw.select(pl.len()).collect(engine="streaming").item()
    if not 0 < count <= MAX_ROWS:
        raise ValueError(f"Expected 1..{MAX_ROWS:,} rows; got {count:,}")
    dtype = schema["transaction_date"]
    if dtype == pl.String:
        text = pl.col("transaction_date").str.strip_chars()
        fmt = job.get("date_format")
        parsed = (text if fmt else text.str.slice(0, 10)).str.to_date(fmt or "%Y-%m-%d", strict=False)
    elif dtype == pl.Date or isinstance(dtype, pl.Datetime):
        parsed = pl.col("transaction_date").cast(pl.Date)
    else:
        raise ValueError("Numeric epoch dates need conversion before upload")
    numeric = []
    for x in ["price", "quantity"]:
        value = pl.col(x).cast(pl.String).str.strip_chars()
        if job.get("decimal_comma"):
            value = value.str.replace_all(",", ".", literal=True)
        numeric.append(value.cast(pl.Float64, strict=False).alias(x))
    raw = raw.select(COLS).with_columns(parsed.alias("transaction_date"), *numeric,
        *[pl.col(x).cast(pl.String).str.strip_chars().replace("", None) for x in ["customer_id", "transaction_id", "product_id"]],
        *[pl.col(x).cast(pl.String).str.strip_chars().replace("", None).fill_null("__UNKNOWN__") for x in ["department", "category"]],
        pl.col("product_description").cast(pl.String).fill_null(""))
    value = pl.col("price") * pl.col("quantity") if job["price_mode"] == "unit" else pl.col("price")
    valid = (pl.all_horizontal([pl.col(x).is_not_null() for x in ["customer_id", "transaction_id", "product_id", "transaction_date"]]) & pl.col("price").is_finite() & (pl.col("price") >= 0) & pl.col("quantity").is_finite() & (pl.col("quantity") > 0) & value.is_finite()).fill_null(False)
    summary = raw.select(pl.len().alias("rows"), (~valid).sum().alias("invalid_rows"), pl.col("transaction_date").null_count().alias("invalid_dates"), (pl.col("department") == "__UNKNOWN__").sum().alias("unknown_department_rows"), (pl.col("category") == "__UNKNOWN__").sum().alias("unknown_category_rows")).collect(engine="streaming").to_dicts()[0]
    bounds = raw.filter(valid).select(pl.col("transaction_date").min().alias("minimum_date"), pl.col("transaction_date").max().alias("maximum_date"), pl.col("customer_id").n_unique().alias("valid_customers")).collect().to_dicts()[0]
    summary.update(bounds)
    summary["ignored_columns"] = sorted(set(schema.names()) - set(COLS))
    summary["source_schema"] = {x: str(schema[x]) for x in COLS}
    write_json(root / "inspection.json", summary)
    if bounds["minimum_date"] is None:
        raise ValueError("No valid purchases after normalization")
    conflicts = raw.filter(valid).group_by(["customer_id", "transaction_id"]).agg(pl.col("transaction_date").n_unique().alias("dates")).filter(pl.col("dates") > 1).select(pl.len()).collect().item()
    summary["basket_key_conflicts"] = conflicts
    write_json(root / "inspection.json", summary)
    sink(raw, root / "canonical.parquet")
    if conflicts:
        raise ValueError("Customer/transaction keys span dates; resolve the transaction grain")
    return summary


def analyze(job):
    workspace, root = Path(job["workspace"]), Path(job["output"])
    metadata = json.loads((workspace / "inspection.json").read_text())
    dates = sorted({date.fromisoformat(x) for x in job["snapshots"]})
    first, last = date.fromisoformat(metadata["minimum_date"]), date.fromisoformat(metadata["maximum_date"])
    end = date.fromisoformat(job["observed_through"]) if job.get("coverage_confirmed") else last
    if not dates or len(dates) > MAX_CUTOFFS or dates[0] < first or dates[-1] > end or end < last:
        raise ValueError("Invalid cutoff or coverage dates")
    if not 1 <= int(job["buckets"]) <= 32:
        raise ValueError("Bucket count must be 1..32")
    required_free = max(256 * 1024 * 1024, (workspace / "canonical.parquet").stat().st_size * (3 + 3 * len(dates)))
    if shutil.disk_usage(workspace).free < required_free:
        raise ValueError("Insufficient temporary disk space; reduce cutoffs or upload size")
    root.mkdir(parents=True, exist_ok=False)
    coverage = bool(job.get("coverage_confirmed"))
    paths, end = prepare(str(workspace / "canonical.parquet"), root, job["price_mode"], job["buckets"], job["exclude_invalid"], None, str(end), coverage)
    numeric = []
    for d in dates:
        past = pl.scan_parquet(root / "validated.parquet").filter(pl.col("date").is_between(d - timedelta(days=364), d))
        sink(past.group_by("product_id").agg(pl.col("unit_price").median().alias("reference_price"), pl.len().alias("reference_count")), root / "price_references" / f"{d}.parquet")
        for path in paths:
            available = pl.scan_parquet(path).filter(pl.col("date") <= d).select(pl.len()).collect().item()
            if available:
                numeric.append(snapshot(path, d, end, root, coverage))
    if not numeric:
        raise ValueError("No customers at the selected cutoffs")
    ref = pl.scan_parquet([str(p) for p in numeric if p.parent.name == str(dates[0])])
    thresholds = ref.filter((pl.col("tenure") >= 89) & (pl.col("spend_90d") > 0)).select(pl.col("spend_90d").quantile(.33).alias("value_low"), pl.col("spend_90d").quantile(.80).alias("value_high")).collect().to_dicts()[0]
    thresholds.update(ref.filter((pl.col("days_365d") >= 6) & (pl.col("tenure") >= 90) & (pl.col("gap_count") >= 4)).select(pl.col("median_basket").quantile(.33).alias("basket_low"), pl.col("median_basket").quantile(.80).alias("basket_high"), pl.col("median_gap").quantile(.33).alias("gap_low"), pl.col("median_gap").quantile(.80).alias("gap_high")).collect().to_dicts()[0])
    thresholds["reference_date"] = str(dates[0])
    write_json(root / "thresholds.json", thresholds)
    for path in numeric:
        frame = label(pl.scan_parquet(path), thresholds).with_columns(pl.when(pl.col("spend_365d") > 0).then(pl.col("category_hhi")).otherwise(None).alias("category_hhi"), pl.when(pl.col("spend_365d") > 0).then(pl.col("category_entropy")).otherwise(None).alias("category_entropy"))
        sink(frame, root / "snapshots" / path.parent.name / path.name)
    write_json(root / "run.json", {"snapshots": [str(x) for x in dates], "coverage_confirmed": coverage, "observed_through": str(end), "price_mode": job["price_mode"], "polars_version": pl.__version__, "source_sha256": job["sha256"], "invalid_rows_excluded": bool(job["exclude_invalid"]), "reference_date": str(dates[0])})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    result = Path(job["workspace"]) / "worker_status.json"
    try:
        inspect(job) if job["action"] == "inspect" else analyze(job)
        write_json(result, {"ok": True})
    except Exception as exc:
        write_json(result, {"ok": False, "error_type": type(exc).__name__, "message": str(exc)[:1500]})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
