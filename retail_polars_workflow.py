"""Purchase-only customer analytics from a nine-column Parquet source."""
import argparse
import json
from datetime import date, timedelta
from pathlib import Path
import polars as pl

COLS = ["customer_id", "transaction_date", "transaction_id", "product_id", "product_description", "department", "category", "price", "quantity"]
C = "customer_id"
K = [C, "transaction_id", "date"]
P = [C, "product_id"]
G = [C, "department", "category"]
VERSION = "reconstructed-1.1"


def ratio(a, b):
    return pl.when(b.is_not_null() & (b > 0)).then(a / b).otherwise(None)


def sink(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sink_parquet(path, compression="zstd", engine="streaming")


def taglist(pairs):
    return pl.concat_list([pl.when(test.fill_null(False)).then(pl.lit(name)).otherwise(pl.lit(None, dtype=pl.String)) for name, test in pairs]).list.drop_nulls()


def prepare(source, root, mode, buckets, exclude, fmt, through, coverage_confirmed=False):
    src = pl.scan_parquet(source)
    schema = src.collect_schema()
    if set(COLS) - set(schema.names()):
        raise ValueError(f"Missing columns: {set(COLS) - set(schema.names())}")
    dtype = schema["transaction_date"]
    if dtype == pl.String:
        text = pl.col("transaction_date") if fmt else pl.col("transaction_date").str.slice(0, 10)
        dates = text.str.to_date(fmt or "%Y-%m-%d", strict=False)
    elif dtype == pl.Date or isinstance(dtype, pl.Datetime):
        dates = pl.col("transaction_date").cast(pl.Date)
    else:
        raise ValueError("Dates must be Date, Datetime, or strings; epoch units are not guessed")
    raw = src.select(COLS).with_columns(dates.alias("date"),
        *[pl.col(x).cast(pl.String).str.strip_chars().replace("", None) for x in [C, "transaction_id", "product_id"]],
        *[pl.col(x).cast(pl.String).str.strip_chars().replace("", None).fill_null("__UNKNOWN__") for x in ["department", "category"]],
        pl.col("product_description").cast(pl.String).fill_null(""),
        *[pl.col(x).cast(pl.Float64, strict=False) for x in ["price", "quantity"]])
    raw = raw.with_columns((pl.col("price") * pl.col("quantity") if mode == "unit" else pl.col("price")).alias("value"))
    good = (pl.all_horizontal([pl.col(x).is_not_null() for x in [C, "transaction_id", "product_id", "date"]]) & pl.col("price").is_finite() & (pl.col("price") >= 0) & pl.col("quantity").is_finite() & (pl.col("quantity") > 0) & pl.col("value").is_finite()).fill_null(False)
    raw = raw.with_columns(good.alias("valid"))
    audit = raw.select(pl.len().alias("source_rows"), (~pl.col("valid")).sum().alias("rejected_rows"), pl.col("date").min().alias("first_date"), pl.col("date").max().alias("last_date")).collect(engine="streaming").to_dicts()[0]
    (root / "quality.json").write_text(json.dumps(audit, default=str, indent=2))
    if not audit["source_rows"]:
        raise ValueError("Empty input")
    if audit["rejected_rows"]:
        sink(raw.filter(~pl.col("valid")), root / "rejected.parquet")
        if not exclude:
            raise ValueError("Invalid rows: inspect quality.json and rejected.parquet")
    clean = raw.filter(pl.col("valid")).drop("valid", "transaction_date")
    maximum_date = clean.select(pl.col("date").max()).collect().item()
    if maximum_date is None:
        raise ValueError("No valid records")
    end = date.fromisoformat(through) if through else maximum_date
    if end < maximum_date:
        raise ValueError("observed_through precedes the latest valid transaction date; explicitly truncate the source before analysis")
    clean = clean.filter(pl.col("date") <= end).with_columns(
        ratio(pl.col("value"), pl.col("quantity")).alias("unit_price"),
        pl.concat_str([pl.col("department").str.len_bytes().cast(pl.String), pl.lit(":"), pl.col("department"), pl.col("category").str.len_bytes().cast(pl.String), pl.lit(":"), pl.col("category")]).alias("category_key"),
        (pl.col(C).hash(seed=17) % buckets).alias("bucket"))
    sink(clean, root / "validated.parquet")
    clean = pl.scan_parquet(root / "validated.parquet")
    audit["observed_through"] = str(end)
    audit["coverage_confirmed"] = bool(coverage_confirmed)
    audit["coverage_interpretation"] = "Future outcomes are unavailable unless coverage_confirmed is true; a maximum observed date alone does not establish complete capture."
    audit["preserved_duplicate_extras"] = 0
    paths = []
    for b in range(buckets):
        path = root / "partitions" / f"b{b}.parquet"
        sink(clean.filter(pl.col("bucket") == b).drop("bucket"), path)
        part = pl.scan_parquet(path)
        conflicts = part.group_by([C, "transaction_id"]).agg(pl.col("date").n_unique().alias("dates")).filter(pl.col("dates") > 1)
        if conflicts.select(pl.len()).collect().item():
            sink(conflicts, root / "basket_conflicts.parquet")
            raise ValueError("Customer/transaction key spans dates; resolve its grain")
        audit["preserved_duplicate_extras"] += part.group_by(part.collect_schema().names()).len().select((pl.col("len") - 1).sum()).collect().item() or 0
        paths.append(path)
    (root / "quality.json").write_text(json.dumps(audit, default=str, indent=2))
    sink(clean.group_by("product_id").agg(pl.struct(["department", "category"]).n_unique().alias("classifications"), pl.col("product_description").n_unique().alias("descriptions")).filter((pl.col("classifications") > 1) | (pl.col("descriptions") > 1)), root / "product_variation.parquet")
    return paths, end


def snapshot(path, d, end, root, coverage_confirmed=False):
    out = root / "events" / str(d) / path.stem
    line = pl.scan_parquet(path).filter(pl.col("date") <= d)
    s30, s90, s365, old0, old1 = [d - timedelta(days=x) for x in [29, 89, 364, 179, 90]]
    recent, old, year = pl.col("date") >= s90, pl.col("date").is_between(old0, old1), pl.col("date") >= s365
    sink(line.group_by(K).agg(pl.col("value").sum().alias("spend"), pl.col("product_id").n_unique().alias("products"), pl.col("category_key").n_unique().alias("breadth")), out / "baskets.parquet")
    basket = pl.scan_parquet(out / "baskets.parquet")
    totals = line.select(pl.col("value").sum()).collect().item() or 0
    if abs((basket.select(pl.col("spend").sum()).collect().item() or 0) - totals) > 1e-7 * max(1, abs(totals)):
        raise AssertionError("Revenue reconciliation failed")
    base = basket.group_by(C).agg(pl.col("date").min().alias("first_purchase"), pl.col("date").max().alias("last_purchase"), pl.len().alias("purchases_history"), pl.col("spend").sum().alias("spend_history"),
        *[pl.col("date").filter(pl.col("date") >= d - timedelta(days=w - 1)).len().alias(f"purchases_{w}d") for w in [30, 90, 365]],
        *[pl.col("spend").filter(pl.col("date") >= d - timedelta(days=w - 1)).sum().alias(f"spend_{w}d") for w in [30, 90, 365]],
        pl.col("date").filter(old).len().alias("purchases_previous90d"), pl.col("spend").filter(old).sum().alias("spend_previous90d"),
        pl.col("spend").filter(year).mean().alias("aov_365d"), pl.col("spend").filter(year).median().alias("median_basket"), ratio(pl.col("spend").filter(year).std(), pl.col("spend").filter(year).mean()).alias("basket_cv"),
        pl.col("breadth").filter(year).median().alias("median_breadth"), pl.col("breadth").filter(recent).mean().alias("breadth_recent"), pl.col("breadth").filter(old).mean().alias("breadth_previous"))
    day = line.group_by([C, "date"]).agg(pl.col("value").sum().alias("day_spend"), pl.col("product_id").unique().alias("products"), pl.col("category_key").unique().alias("categories"))
    day = day.with_columns(*[pl.col(x).shift(1).over(C, order_by="date").alias("prev_" + x) for x in ["date", "products", "categories"]], pl.col("day_spend").shift(1).rolling_median(5, min_samples=3).over(C, order_by="date").alias("baseline_spend"))
    day = day.with_columns((pl.col("date") - pl.col("prev_date")).dt.total_days().alias("gap"), ratio(pl.col("day_spend"), pl.col("baseline_spend")).alias("relative_spend"),
        *[ratio(pl.col(x).list.set_intersection(pl.col("prev_" + x)).list.len(), pl.col(x).list.set_union(pl.col("prev_" + x)).list.len()).alias(x + "_overlap") for x in ["products", "categories"]])
    day = day.with_columns(pl.col("gap").shift(1).rolling_median(5, min_samples=3).over(C, order_by="date").alias("baseline_gap"))
    sink(day, out / "days.parquet")
    day = pl.scan_parquet(out / "days.parquet")
    usable = year & (pl.col("prev_date") >= s365)
    rhythm = day.group_by(C).agg(pl.len().alias("days_history"), pl.col("date").filter(year).len().alias("days_365d"), pl.col("gap").filter(usable).count().alias("gap_count"), pl.col("gap").filter(usable).median().alias("median_gap"), ratio(pl.col("gap").filter(usable).std(), pl.col("gap").filter(usable).mean()).alias("gap_cv"),
        pl.col("products_overlap").filter(usable).median().alias("product_overlap"), pl.col("categories_overlap").filter(usable).median().alias("category_overlap"),
        (pl.col("date").dt.weekday() >= 6).filter(year).mean().alias("weekend_share"), pl.col("relative_spend").filter(recent).median().alias("personal_spend_ratio"),
        ((pl.col("date") >= s30) & (ratio(pl.col("gap"), pl.col("baseline_gap")) >= 2)).fill_null(False).sum().alias("reactivation_events"))
    weekdays = day.filter(year).group_by([C, pl.col("date").dt.weekday().alias("weekday")]).len().with_columns((pl.col("len") / pl.col("len").sum().over(C)).alias("p"))
    weekday_stats = weekdays.group_by(C).agg((-(pl.col("p") * pl.col("p").log())).sum().alias("weekday_entropy"))
    last6 = day.with_columns(pl.col("date").cum_count().over(C, order_by="date", descending=True).alias("rank")).filter(pl.col("rank") <= 6).select(C, "date")
    breadth = line.join(last6, on=[C, "date"], how="inner", validate="m:1").group_by(C).agg(pl.col("product_id").n_unique().alias("products_last6days"), pl.col("category_key").n_unique().alias("categories_last6days"))
    pd = line.group_by(P + ["date"]).agg(pl.col("quantity").sum().alias("qty"), pl.col("value").sum().alias("spend"))
    pd = pd.with_columns((pl.col("date").cum_count().over(P, order_by="date") - 1).alias("prior_days"), pl.col("date").shift(-1).over(P, order_by="date").alias("next_date"), pl.col("qty").shift(1).rolling_median(5, min_samples=3).over(P, order_by="date").alias("baseline_qty"))
    pd = pd.with_columns((pl.col("next_date") - pl.col("date")).dt.total_days().alias("next_gap"), ratio(pl.col("qty"), pl.col("baseline_qty")).alias("relative_qty"))
    sink(pd, out / "product_days.parquet")
    pd = pl.scan_parquet(out / "product_days.parquet")
    ph = pd.group_by(P).agg(pl.col("date").min().alias("trial"), pl.col("date").filter(pl.col("prior_days") == 1).min().alias("repeat"), pl.len().alias("purchase_days"), pl.col("spend").filter(year).sum().alias("year_spend"))
    sink(ph, out / "product_history.parquet")
    eligible = pl.col("trial").is_between(s365, d - timedelta(days=90))
    success = eligible & pl.col("repeat").is_not_null() & (pl.col("repeat") <= pl.col("trial") + pl.duration(days=90))
    trials = ph.group_by(C).agg(eligible.sum().alias("eligible_trials"), success.fill_null(False).sum().alias("repeated_trials"), (pl.col("purchase_days") >= 3).sum().alias("core_size"), pl.col("year_spend").filter(pl.col("purchase_days") >= 3).sum().alias("core_spend"))
    product_baskets = line.select(K + ["product_id"]).unique().join(pd.select(P + ["date", "prior_days"]), on=P + ["date"], how="left", validate="m:1").group_by(K).agg((pl.col("prior_days") == 0).mean().alias("new_share"), (pl.col("prior_days") >= 1).mean().alias("repeat_share"), (pl.col("prior_days") >= 2).mean().alias("established_share"))
    habits = product_baskets.group_by(C).agg(pl.col("new_share").filter(year).mean().alias("new_share"), pl.col("repeat_share").filter(year).mean().alias("repeat_share"), pl.col("established_share").filter(year).mean().alias("established_share"))
    quantity = pd.group_by(C).agg(pl.col("next_gap").filter(year & (pl.col("relative_qty") >= 1.5)).median().alias("gap_after_large_qty"), pl.col("next_gap").filter(year & pl.col("relative_qty").is_between(0.75, 1.25)).median().alias("gap_after_typical_qty"), pl.col("next_gap").filter(year & (pl.col("relative_qty") >= 1.5)).count().alias("large_qty_gap_count"))
    cd = line.group_by(G + ["date"]).agg(pl.col("value").sum().alias("spend")).with_columns(pl.col("date").shift(1).over(G, order_by="date").alias("prev"))
    cd = cd.with_columns((pl.col("date") - pl.col("prev")).dt.total_days().alias("gap"))
    ch = cd.group_by(G).agg(pl.col("spend").filter(year).sum().alias("year_spend"), pl.len().alias("days"), pl.col("date").filter(year).len().alias("year_days"), pl.col("date").max().alias("last"), pl.col("gap").median().alias("median_gap"))
    ch = ch.with_columns(ratio(pl.col("year_spend"), pl.col("year_spend").sum().over(C)).alias("share"), ratio((pl.lit(d) - pl.col("last")).dt.total_days(), pl.col("median_gap")).alias("lateness"))
    sink(ch, out / "category_history.parquet")
    categories = ch.group_by(C).agg((pl.col("year_days") > 0).sum().alias("category_count"), (pl.col("share") ** 2).sum().alias("category_hhi"), (-(pl.col("share") * pl.col("share").log())).filter(pl.col("share") > 0).sum().alias("category_entropy"), pl.col("share").max().alias("dominant_share"), ((pl.col("days") >= 3) & (pl.col("lateness") >= 2)).fill_null(False).sum().alias("overdue_categories"))
    dominant = ch.filter(pl.col("year_days") > 0).sort([C, "year_spend", "department", "category"], descending=[False, True, False, False]).group_by(C, maintain_order=True).agg(pl.col("department").first().alias("dominant_department"), pl.col("category").first().alias("dominant_category"))
    prices = line.filter(year).join(pl.scan_parquet(root / "price_references" / f"{d}.parquet"), on="product_id", how="left", validate="m:1").with_columns(pl.when(pl.col("reference_count") >= 5).then(ratio(pl.col("unit_price"), pl.col("reference_price"))).otherwise(None).alias("relative_price"))
    prices = prices.group_by(C).agg(pl.col("relative_price").median().alias("relative_paid_price"), pl.col("relative_price").count().alias("comparable_price_lines"))
    frame = base
    for extra in [rhythm, weekday_stats, breadth, trials, habits, quantity, categories, dominant, prices]:
        frame = frame.join(extra, on=C, how="left", validate="1:1")
    frame = frame.with_columns(pl.lit(d).alias("snapshot"), (pl.lit(d) - pl.col("first_purchase")).dt.total_days().alias("tenure"), (pl.lit(d) - pl.col("last_purchase")).dt.total_days().alias("recency"), ratio(pl.col("spend_90d"), pl.col("purchases_90d")).alias("aov_recent"), ratio(pl.col("spend_previous90d"), pl.col("purchases_previous90d")).alias("aov_previous"), ratio(pl.col("repeated_trials"), pl.col("eligible_trials")).alias("trial_conversion"), ratio(pl.col("core_spend"), pl.col("spend_365d")).alias("core_spend_share"))
    frame = frame.with_columns(ratio(pl.col("recency"), pl.col("median_gap")).alias("lateness"), (pl.col("tenure") >= 179).alias("comparable_windows"), (pl.col("breadth_recent") - pl.col("breadth_previous")).alias("breadth_change"), *[pl.when(pl.col("days_history") >= 6).then(pl.col(x)).otherwise(None).alias(x) for x in ["products_last6days", "categories_last6days"]])
    frame = frame.with_columns(*[pl.when(pl.col("comparable_windows")).then(ratio(pl.col(a), pl.col(b)) - 1).otherwise(None).alias(name) for a, b, name in [("purchases_90d", "purchases_previous90d", "frequency_change"), ("aov_recent", "aov_previous", "basket_change"), ("spend_90d", "spend_previous90d", "spend_change")]], pl.lit(VERSION).alias("version"))
    for h in [30, 90]:
        horizon_end = d + timedelta(days=h)
        followup_eligible = bool(coverage_confirmed) and horizon_end <= end
        future = pl.scan_parquet(path).filter(pl.col("date").is_between(d + timedelta(days=1), horizon_end)).group_by(C).agg(pl.col("transaction_id").n_unique().alias(f"future_purchases_{h}d"), pl.col("value").sum().alias(f"future_spend_{h}d"))
        frame = frame.join(future, on=C, how="left", validate="1:1").with_columns(pl.lit(followup_eligible).alias(f"eligible_followup_{h}d"), *[pl.when(pl.lit(followup_eligible)).then(pl.col(x).fill_null(0)).otherwise(None).alias(x) for x in [f"future_purchases_{h}d", f"future_spend_{h}d"]])
    target = root / "numeric" / str(d) / path.name
    sink(frame, target)
    return target


def label(frame, thresholds):
    enough = (pl.col("days_365d") >= 6) & (pl.col("tenure") >= 90) & (pl.col("gap_count") >= 4)
    routine = (pl.col("gap_cv") <= 0.75) & (pl.col("established_share") >= 0.70)
    focused = (pl.col("dominant_share") >= 0.70) & (pl.col("established_share") >= 0.70)
    broad = (pl.col("median_breadth") >= 3) & (pl.col("established_share") >= 0.70)
    explorer = pl.col("new_share") >= 0.40
    style = [("routine_repeat", enough & routine), ("focused_repeat", enough & focused), ("broad_repeat", enough & broad), ("exploration_heavy", enough & explorer)]
    if all(thresholds.get(x) is not None for x in ["basket_low", "basket_high", "gap_low", "gap_high"]) and thresholds["basket_high"] > thresholds["basket_low"] and thresholds["gap_high"] > thresholds["gap_low"]:
        style += [("large_basket_episodic", enough & (pl.col("median_basket") >= thresholds["basket_high"]) & (pl.col("median_gap") >= thresholds["gap_high"])), ("small_basket_frequent", enough & (pl.col("median_basket") <= thresholds["basket_low"]) & (pl.col("median_gap") <= thresholds["gap_low"]))]
    slow, small = pl.col("frequency_change") <= -0.25, pl.col("basket_change") <= -0.25
    narrow = pl.col("comparable_windows") & ((pl.col("breadth_change") <= -0.5) | (pl.col("overdue_categories") >= 2))
    grow = (pl.col("frequency_change") >= 0.25) | (pl.col("basket_change") >= 0.25)
    late, back = enough & (pl.col("lateness") >= 2), pl.col("reactivation_events") > 0
    stable = enough & pl.col("comparable_windows") & pl.col("frequency_change").is_not_null() & ~(slow.fill_null(False) | small.fill_null(False) | narrow.fill_null(False) | grow.fill_null(False) | late.fill_null(False) | back.fill_null(False))
    frame = frame.with_columns(enough.alias("style_eligible"), taglist(style).alias("behavioral_tags"), taglist([("slowing", slow), ("smaller_baskets", small), ("narrowing", narrow), ("broad_decline", slow & small), ("expanding", grow), ("overdue", late), ("reactivated", back), ("stable", stable)]).alias("trajectory_tags"),
        pl.when(pl.col("days_history") == 1).then(pl.when(pl.col("tenure") >= 90).then(pl.lit("one_shopping_day")).otherwise(pl.lit("first_observed"))).when((pl.col("days_history") >= 6) & (pl.col("tenure") >= 90)).then(pl.lit("established_repeat")).otherwise(pl.lit("early_repeat")).alias("lifecycle"),
        pl.when(enough & (pl.col("gap_count") >= 12)).then(pl.lit("stronger")).when(enough).then(pl.lit("limited")).otherwise(pl.lit("insufficient")).alias("evidence"))
    frame = frame.with_columns(pl.when(~pl.col("style_eligible")).then(pl.lit("insufficient_history")).when(pl.col("behavioral_tags").list.len() == 0).then(pl.lit("mixed_other")).otherwise(pl.col("behavioral_tags").list.join("+")).alias("behavior"), pl.when(pl.col("trajectory_tags").list.len() == 0).then(pl.lit("not_classified")).otherwise(pl.col("trajectory_tags").list.join("+")).alias("trajectory"))
    if thresholds.get("value_low") is None or thresholds.get("value_high") is None:
        return frame.with_columns(pl.lit("reference_unavailable").alias("value_tier"))
    return frame.with_columns(pl.when(pl.col("tenure") < 89).then(pl.lit("insufficient_exposure")).when(pl.col("spend_90d") == 0).then(pl.lit("no_recent_spend")).when(pl.col("spend_90d") >= thresholds["value_high"]).then(pl.lit("high")).when(pl.col("spend_90d") <= thresholds["value_low"]).then(pl.lit("low")).otherwise(pl.lit("middle")).alias("value_tier"))


def reports(root, buckets):
    allrows = pl.scan_parquet(str(root / "snapshots" / "*" / "*.parquet"))
    profile = allrows.group_by(["snapshot", "behavior"]).agg(pl.len().alias("customers"), pl.col("spend_90d").sum().alias("revenue"), pl.col("spend_90d").median().alias("median_spend"), pl.col("median_gap").median().alias("median_gap"), pl.col("new_share").mean().alias("exploration"), pl.col("trial_conversion").mean().alias("mean_customer_trial_conversion"), *[(pl.col(f"future_purchases_{h}d") > 0).filter(pl.col(f"eligible_followup_{h}d")).mean().alias(f"future_purchase_rate_{h}d") for h in [30, 90]], *[pl.col(f"future_spend_{h}d").filter(pl.col(f"eligible_followup_{h}d")).mean().alias(f"future_mean_spend_{h}d") for h in [30, 90]], *[pl.col(f"eligible_followup_{h}d").sum().alias(f"eligible_customers_{h}d") for h in [30, 90]])
    profile = profile.with_columns(ratio(pl.col("customers"), pl.col("customers").sum().over("snapshot")).alias("customer_share"), ratio(pl.col("revenue"), pl.col("revenue").sum().over("snapshot")).alias("revenue_share"))
    sink(profile, root / "segment_profiles.parquet")
    sink(allrows.with_columns(pl.col("first_purchase").dt.truncate("1mo").alias("cohort")).group_by(["snapshot", "cohort"]).agg(pl.len().alias("customers"), pl.col("spend_history").mean().alias("mean_cumulative_spend"), (pl.col("purchases_30d") > 0).mean().alias("rolling30d_purchase_rate"), (pl.col("future_purchases_30d") > 0).filter(pl.col("eligible_followup_30d")).mean().alias("future30d_purchase_rate")), root / "cohort_profiles.parquet")
    paths = []
    for b in range(buckets):
        rows = pl.scan_parquet(str(root / "snapshots" / "*" / f"b{b}.parquet"))
        for dim in ["lifecycle", "behavior", "trajectory", "value_tier"]:
            prev = rows.with_columns(pl.col(dim).shift(1).over(C, order_by="snapshot").alias("from_label"), pl.col("snapshot").shift(1).over(C, order_by="snapshot").alias("from_date")).filter(pl.col("from_label").is_not_null())
            trans = prev.group_by(["from_date", "snapshot", "from_label", dim]).len().rename({dim: "to_label", "len": "customers"}).with_columns(pl.lit(dim).alias("dimension"))
            target = root / "transition_parts" / f"b{b}_{dim}.parquet"
            sink(trans, target)
            paths.append(str(target))
    trans = pl.scan_parquet(paths).group_by(["from_date", "snapshot", "dimension", "from_label", "to_label"]).agg(pl.col("customers").sum())
    trans = trans.with_columns(pl.col("customers").sum().over(["from_date", "snapshot", "dimension", "from_label"]).alias("eligible_origin"))
    sink(trans.with_columns(ratio(pl.col("customers"), pl.col("eligible_origin")).alias("transition_rate")), root / "segment_transitions.parquet")


def run(source, output, price_mode, snapshots, buckets=8, exclude_invalid=False, date_format=None, observed_through=None, coverage_confirmed=False):
    if buckets < 1 or price_mode not in ["unit", "line_total"]:
        raise ValueError("Invalid buckets or price_mode")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    dates = sorted({date.fromisoformat(x) for x in snapshots})
    if not dates:
        raise ValueError("At least one cutoff required")
    (root / "configuration.json").write_text(json.dumps(dict(source=source, price_mode=price_mode, snapshots=[str(x) for x in dates], buckets=buckets, exclude_invalid=exclude_invalid, date_format=date_format, observed_through=observed_through, coverage_confirmed=bool(coverage_confirmed), version=VERSION), indent=2))
    paths, end = prepare(source, root, price_mode, buckets, exclude_invalid, date_format, observed_through, coverage_confirmed)
    if dates[-1] > end:
        raise ValueError("Cutoff exceeds observed_through")
    numeric = []
    for d in dates:
        past = pl.scan_parquet(root / "validated.parquet").filter(pl.col("date").is_between(d - timedelta(days=364), d))
        sink(past.group_by("product_id").agg(pl.col("unit_price").median().alias("reference_price"), pl.len().alias("reference_count")), root / "price_references" / f"{d}.parquet")
        for path in paths:
            if pl.scan_parquet(path).select(pl.len()).collect().item() > 0:
                numeric.append(snapshot(path, d, end, root, coverage_confirmed))
    ref = pl.scan_parquet([str(x) for x in numeric if x.parent.name == str(dates[0])])
    thresholds = ref.filter((pl.col("tenure") >= 89) & (pl.col("spend_90d") > 0)).select(pl.col("spend_90d").quantile(0.33).alias("value_low"), pl.col("spend_90d").quantile(0.80).alias("value_high")).collect().to_dicts()[0]
    thresholds.update(ref.filter((pl.col("days_365d") >= 6) & (pl.col("tenure") >= 90) & (pl.col("gap_count") >= 4)).select(pl.col("median_basket").quantile(0.33).alias("basket_low"), pl.col("median_basket").quantile(0.80).alias("basket_high"), pl.col("median_gap").quantile(0.33).alias("gap_low"), pl.col("median_gap").quantile(0.80).alias("gap_high")).collect().to_dicts()[0])
    thresholds["reference_date"] = str(dates[0])
    (root / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    for path in numeric:
        sink(label(pl.scan_parquet(path), thresholds), root / "snapshots" / path.parent.name / path.name)
    reports(root, buckets)
    print(f"Completed {len(dates)} cutoffs: {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, dest="source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--price-mode", required=True, choices=["unit", "line_total"])
    parser.add_argument("--snapshots", nargs="+", required=True)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--exclude-invalid", action="store_true")
    parser.add_argument("--date-format")
    parser.add_argument("--observed-through")
    parser.add_argument("--coverage-confirmed", action="store_true")
    run(**vars(parser.parse_args()))
