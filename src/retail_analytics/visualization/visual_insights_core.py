"""Transaction-only aggregate views. No product-pair matrix or association mining."""
import math
from datetime import date, timedelta

METRICS = {
    "spend_90d": ("Recorded spend, last 90 days", "currency units"),
    "purchases_90d": ("Baskets, last 90 days", "baskets"),
    "recency": ("Days since last recorded purchase", "days"),
    "median_basket": ("Median basket amount, last 365 days", "currency units"),
    "median_gap": ("Median completed shopping-day gap, last 365 days", "days"),
    "median_breadth": ("Median category breadth per basket, last 365 days", "categories"),
    "new_share": ("Mean unfamiliar-product share per basket, last 365 days", "%"),
    "repeat_share": ("Mean repeated-product share per basket, last 365 days", "%"),
    "established_share": ("Mean established-product share per basket, last 365 days", "%"),
    "category_hhi": ("Category spend concentration, last 365 days", "index"),
}


def share(a, b):
    return a / b if b > 0 else None


def top_count(n, fraction):
    if n < 0 or not 0 < fraction <= 1:
        raise ValueError("Invalid population or fraction")
    return min(n, math.ceil(n * fraction))


def month_windows(first, last):
    cursor, result = first.replace(day=1), []
    while first <= last and cursor <= last:
        following = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        end = following - timedelta(days=1)
        lo, hi = max(first, cursor), min(last, end)
        result.append(dict(month=cursor, start=lo, end=hi, partial=lo != cursor or hi != end))
        cursor = following
    return result


def wilson(k, n):
    if n <= 0:
        return None, None
    if not 0 <= k <= n:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p, denominator = k / n, 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0, center-half), min(1, center+half)


def collect(frame):
    return frame.collect(engine="streaming")


def segment_summary(rows, dim, metric):
    import polars as pl
    if metric not in METRICS:
        raise ValueError("Unsupported metric")
    finite = pl.col(metric).filter(pl.col(metric).is_finite())
    data = rows.group_by(dim).agg(pl.len().alias("customers"), finite.count().alias("finite_n"),
        finite.quantile(.25, interpolation="linear").alias("q25"), finite.median().alias("median"),
        finite.quantile(.75, interpolation="linear").alias("q75"), pl.col("spend_90d").sum().alias("spend"),
        (pl.col("purchases_90d")>0).sum().alias("active_n"),
        (pl.col("tenure")<89).sum().alias("short_90d_history_n"),
        (pl.col("tenure")<364).sum().alias("short_365d_history_n"))
    return collect(data.with_columns((pl.col("customers")-pl.col("finite_n")).alias("unavailable_n"),
        (pl.col("customers")/pl.col("customers").sum()).alias("customer_share"),
        pl.when(pl.col("spend").sum()>0).then(pl.col("spend")/pl.col("spend").sum()).otherwise(None).alias("spend_share"),
        (pl.col("active_n")/pl.col("customers")).alias("active_share")).sort("customers", descending=True))


def concentration(rows):
    import polars as pl
    values = rows.select(pl.col("spend_90d").cast(pl.Float64).alias("amount"))
    info = collect(values.select(pl.len().alias("n"), pl.col("amount").is_finite().sum().alias("finite"),
        (pl.col("amount")<0).sum().alias("negative"))).row(0, named=True)
    n = info["n"]
    if info["finite"] != n or info["negative"]:
        raise ValueError("Concentration requires finite nonnegative spend")
    if not n:
        return dict(n=0, total=0., positive_n=0, curve=[], top=[])
    ranked = values.sort("amount", descending=True).with_row_index("rank", offset=1)
    fractions = [.01, .05, .10, .20]
    stats = collect(ranked.select(pl.col("amount").sum().alias("total"), (pl.col("amount")>0).sum().alias("positive_n"),
        *[pl.col("amount").filter(pl.col("rank")<=top_count(n,f)).sum().alias(f"top{i}") for i,f in enumerate(fractions)])).row(0, named=True)
    top = [dict(requested_fraction=f, customers=top_count(n,f), actual_fraction=top_count(n,f)/n,
        spend_share=share(stats[f"top{i}"],stats["total"])) for i,f in enumerate(fractions)]
    curve = []
    if stats["total"]>0:
        endpoint = ranked.with_columns(pl.col("amount").cum_sum().alias("cumulative"),
            (pl.col("rank")*100/n).ceil().cast(pl.Int32).alias("bin"))
        data = collect(endpoint.group_by("bin").agg(pl.col("rank").max(),pl.col("cumulative").max()).sort("rank"))
        curve = [(0.,0.)]+[(r["rank"]/n,r["cumulative"]/stats["total"]) for r in data.to_dicts()]
    return dict(n=n, total=stats["total"], positive_n=stats["positive_n"], curve=curve, top=top)


def monthly_summary(baskets, rows, first, cutoff, months=12):
    import polars as pl
    if not 1 <= months <= 36:
        raise ValueError("Months must be 1..36")
    windows = month_windows(first,cutoff)[-months:]
    calendar = pl.DataFrame(windows,schema={"month":pl.Date,"start":pl.Date,"end":pl.Date,"partial":pl.Boolean})
    if not windows:
        return calendar
    ids = rows.select("customer_id").unique()
    n = collect(ids.select(pl.len())).item()
    selected = baskets.filter(pl.col("date").is_between(windows[0]["start"],cutoff)).join(ids,on="customer_id",how="semi")
    summary = selected.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).group_by("month").agg(
        pl.col("spend").sum().alias("spend"),pl.len().alias("baskets"),pl.col("customer_id").n_unique().alias("buyers"))
    result = calendar.lazy().join(summary,on="month",how="left").with_columns(pl.col("spend").fill_null(0.),
        pl.col("baskets").fill_null(0),pl.col("buyers").fill_null(0),pl.lit(n).alias("population_n"))
    return collect(result.with_columns(pl.when(pl.col("baskets")>0).then(pl.col("spend")/pl.col("baskets")).otherwise(None).alias("amount_per_basket"),
        pl.when(pl.col("buyers")>0).then(pl.col("baskets")/pl.col("buyers")).otherwise(None).alias("baskets_per_buyer"),
        pl.when(pl.lit(n)>0).then(pl.col("buyers")/pl.lit(n)).otherwise(None).alias("buying_share")).sort("month"))


def category_summary(history, rows):
    import polars as pl
    ids = rows.select("customer_id").unique()
    n = collect(ids.select(pl.len())).item()
    data = history.filter(pl.col("year_days")>0).join(ids,on="customer_id",how="semi").group_by("department","category").agg(
        pl.col("customer_id").n_unique().alias("customers"),pl.col("year_spend").sum().alias("spend"),
        pl.col("customer_id").filter(pl.col("year_days")>=3).n_unique().alias("recurring_customers"))
    return collect(data.with_columns(pl.lit(n).alias("population_n"),
        pl.when(pl.lit(n)>0).then(pl.col("customers")/pl.lit(n)).otherwise(None).alias("participation"),
        pl.when(pl.col("spend").sum()>0).then(pl.col("spend")/pl.col("spend").sum()).otherwise(None).alias("spend_share"),
        pl.concat_str([pl.col("department"),pl.lit(" / "),pl.col("category")]).alias("label")).sort("customers",descending=True))


def cohort_summary(baskets, rows, cutoff, coverage, minimum=20):
    import polars as pl
    if minimum < 1:
        raise ValueError("Minimum must be positive")
    ids = rows.select("customer_id","first_purchase").unique().with_columns(pl.col("first_purchase").dt.truncate("1mo").alias("cohort"))
    repeat = baskets.filter(pl.col("date")<=cutoff).join(ids.select("customer_id","first_purchase"),on="customer_id",how="inner",validate="m:1")
    repeat = repeat.filter(pl.col("date")>pl.col("first_purchase")).group_by("customer_id").agg(pl.col("date").min().alias("repeat_date"))
    profiles = ids.join(repeat,on="customer_id",how="left",validate="1:1")
    records = []
    for horizon in [30,60,90]:
        eligible = pl.col("first_purchase")+pl.duration(days=horizon)<=pl.lit(cutoff)
        success = eligible & pl.col("repeat_date").is_not_null() & (pl.col("repeat_date")<=pl.col("first_purchase")+pl.duration(days=horizon))
        data = collect(profiles.group_by("cohort").agg(pl.len().alias("cohort_n"),eligible.sum().alias("eligible_n"),success.fill_null(False).sum().alias("repeat_n")))
        for r in data.to_dicts():
            allowed = coverage and r["eligible_n"]>=minimum
            lo,hi = wilson(r["repeat_n"],r["eligible_n"]) if allowed else (None,None)
            r.update(horizon=horizon,rate=share(r["repeat_n"],r["eligible_n"]) if allowed else None,low=lo,high=hi,
                status="eligible" if allowed else "capture_unconfirmed" if not coverage else "insufficient_followup_or_n")
            if not coverage:
                r["eligible_n"],r["repeat_n"] = None,None
            records.append(r)
    return pl.DataFrame(records).sort("cohort","horizon") if records else pl.DataFrame()


def transition_summary(before, after, dim):
    import polars as pl
    origin = before.select("customer_id",pl.col(dim).alias("from_label"))
    destination = after.select("customer_id",pl.col(dim).alias("to_label"))
    joined = origin.join(destination,on="customer_id",how="inner",validate="1:1")
    n0,n1,matched = [collect(f.select(pl.len())).item() for f in [origin,destination,joined]]
    flows = collect(joined.group_by("from_label","to_label").len().with_columns(
        pl.col("len").sum().over("from_label").alias("origin_matched_n")).with_columns(
        (pl.col("len")/pl.col("origin_matched_n")).alias("rate")).sort("len",descending=True))
    return flows,dict(origin_n=n0,destination_n=n1,matched_n=matched,origin_unmatched=n0-matched,destination_unmatched=n1-matched)
