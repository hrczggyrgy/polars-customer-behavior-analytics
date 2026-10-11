"""Additive patronage views using the existing dashboard helpers."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import app as base
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from patronage import Rules, destination, label_frame

FEATURES = [
    "days_365d",
    "median_gap",
    "gap_cv",
    "active_period_share",
    "median_basket",
    "median_breadth",
    "new_share",
    "repeat_share",
    "established_share",
    "category_persistence_share",
    "frequency_change",
]


def render(root, coverage):
    manifest = base.read_json(root / "run.json")
    st.subheader("Observed patronage and segment characteristics")
    st.info(
        "Observed purchasing, not emotional loyalty, retailer exclusivity, share of wallet or permanent churn. Recorded-spend tiers and shopping styles remain separate."
    )
    with st.expander("Rules and evidence", expanded=False):
        tenure = st.number_input("Minimum observed history (days)", 30, 730, 90)
        months = st.number_input("Minimum complete calendar months", 1, 24, 3)
        factor = st.number_input("Overdue / resumed-gap multiple", 1.1, 5.0, 2.0, 0.1)
        share = st.slider("Recurring active-period share", 0.0, 1.0, 0.6, 0.05)
        cv = st.number_input("Maximum gap CV for regular cadence", 0.0, 3.0, 0.75, 0.05)
        confirmed = st.checkbox("I confirm complete capture within the observed history used here")
        st.caption(
            "Calendar completeness means entire calendar bins after first observation; it does not establish acquisition or data completeness. Confirm capture only if justified by the source. Changed rules create a separate result set."
        )
    rules = Rules(
        min_tenure=int(tenure),
        min_complete_months=int(months),
        overdue_multiple=float(factor),
        active_share=float(share),
        max_gap_cv=float(cv),
    )
    confirmed = bool(confirmed and coverage and manifest.get("coverage_confirmed"))
    if not confirmed:
        st.warning(
            "Patronage labels remain capture_unconfirmed until both the original coverage decision and the history confirmation are enabled. Descriptive feature profiles are still available."
        )
    target, specification = destination(root, rules, confirmed)
    if not (target / "manifest.json").exists():
        if st.button("Build patronage layer", type="primary"):
            control = root / "patronage_rules.json"
            control.write_text(json.dumps(asdict(rules)))
            command = [
                sys.executable,
                str(Path(__file__).with_name("patronage.py")),
                "--root",
                str(root),
                "--rules",
                str(control),
            ]
            if confirmed:
                command.append("--capture-confirmed")
            try:
                with st.spinner(
                    "Building point-in-time patronage features in a separate worker..."
                ):
                    process = subprocess.run(command, capture_output=True, timeout=base.TIMEOUT)
                if process.returncode or not (target / "manifest.json").exists():
                    st.error(
                        "Patronage worker failed. Check schemas, dependencies and resources; no completed result was accepted."
                    )
                    return
            except subprocess.TimeoutExpired:
                st.error(
                    "Patronage worker timed out. Retry with fewer cutoffs or a smaller dataset."
                )
                return
        else:
            st.json(asdict(rules))
            return
    dates = manifest["snapshots"]
    chosen = st.selectbox("Patronage cutoff", dates, index=len(dates) - 1)
    rows = pl.scan_parquet(str(target / "snapshots" / chosen / "*.parquet"))
    labels = base.collect(rows.select("observed_patronage").unique().sort("observed_patronage"))[
        "observed_patronage"
    ].to_list()
    section = st.radio(
        "Patronage view",
        [
            "Profiles",
            "Distributions",
            "Style overlap",
            "Categories",
            "Transitions",
            "Sensitivity",
            "Customer",
            "Definitions",
        ],
        horizontal=True,
    )
    if section == "Profiles":
        expressions = [pl.len().alias("customers")]
        for f in FEATURES:
            value = pl.col(f).filter(pl.col(f).is_finite())
            expressions.extend(
                [
                    value.count().alias(f + "_n"),
                    value.quantile(0.25).alias(f + "_q25"),
                    value.median().alias(f + "_median"),
                    value.quantile(0.75).alias(f + "_q75"),
                ]
            )
        profile = base.collect(
            rows.group_by("observed_patronage").agg(expressions).sort("customers", descending=True)
        )
        base.bars(
            profile,
            "observed_patronage",
            "customers",
            "Observed patronage populations",
            "patronage_counts",
        )
        base.table(profile)
        st.caption(
            "Counts, medians and quartiles describe within-segment variation. History, cadence and activity define these rules; their differences are not independent validation. Basket breadth, product familiarity and category mix are additional characteristics, not causal explanations."
        )
        st.download_button(
            "Export segment characteristics",
            base.export_csv(profile),
            "patronage_profiles.csv",
            "text/csv",
        )
    elif section == "Distributions":
        f = st.selectbox("Characteristic", FEATURES)
        sample = base.collect(
            rows.filter(pl.col(f).is_finite())
            .sort(pl.col("customer_id").hash(seed=41))
            .limit(base.PAGE_ROWS)
            .select("observed_patronage", f)
        )
        fig = go.Figure()
        for name in labels:
            values = sample.filter(pl.col("observed_patronage") == name)[f].to_list()
            if values:
                fig.add_trace(
                    go.Box(y=values, name=name, boxpoints=False, marker_color=base.color(name))
                )
        base.chart(
            fig,
            "Within-segment characteristic distributions",
            "patronage_box",
            "Observed patronage",
            f,
        )
        scatter = base.collect(
            rows.filter(pl.col("gap_cv").is_finite() & pl.col("lateness").is_finite())
            .sort(pl.col("customer_id").hash(seed=41))
            .limit(base.PAGE_ROWS)
            .select("gap_cv", "lateness", "observed_patronage")
        )
        fig = go.Figure()
        for name in labels:
            part = scatter.filter(pl.col("observed_patronage") == name)
            fig.add_trace(
                go.Scattergl(
                    x=part["gap_cv"].to_list(),
                    y=part["lateness"].to_list(),
                    name=name,
                    mode="markers",
                    marker=dict(size=5, opacity=0.5, color=base.color(name)),
                )
            )
        base.chart(
            fig,
            "Cadence variability and current hiatus",
            "patronage_cadence",
            "Gap CV",
            "Recency / median gap",
        )
        st.caption(
            "Each plot uses at most 2,000 hash-selected finite observations. No customer IDs are transmitted. These are defining-feature diagnostics, not calibrated churn probabilities."
        )
    elif section == "Style overlap":
        cross = base.collect(rows.group_by("observed_patronage", "behavior").len())
        xs, ys, z = base.bounded_grid(
            cross.to_dicts(), "observed_patronage", "behavior", "len", limit=20
        )
        base.chart(
            go.Figure(go.Heatmap(x=xs, y=ys, z=z, colorscale="Blues", hoverongaps=False)),
            "Patronage and existing shopping styles",
            "patronage_style",
            "Shopping style",
            "Observed patronage",
        )
        st.caption(
            "Counts, with up to 20 labels per axis; missing combinations are blank. Familiar-product repetition and exploratory shopping are separate from observed relationship continuity."
        )
        base.table(cross.sort("len", descending=True))
    elif section == "Categories":
        segment = st.selectbox("Patronage type", labels)
        denominator = base.collect(
            rows.filter(pl.col("observed_patronage") == segment).select(pl.len())
        ).item()
        categories = pl.scan_parquet(str(target / "categories" / chosen / "*.parquet")).filter(
            pl.col("observed_patronage") == segment
        )
        summary = base.collect(
            categories.group_by("department", "category")
            .agg(
                pl.len().alias("customers"),
                pl.col("category_spend_365d").sum().alias("spend"),
                (pl.col("category_days_365d") >= 3).sum().alias("recurring_customers"),
                pl.col("category_median_gap").median().alias("median_customer_category_gap"),
                pl.col("category_gap_count").sum().alias("observed_gap_count"),
            )
            .with_columns(
                (pl.col("customers") / denominator).alias("participation"),
                pl.when(pl.col("spend").sum() > 0)
                .then(pl.col("spend") / pl.col("spend").sum())
                .otherwise(None)
                .alias("spend_share"),
            )
            .sort("customers", descending=True)
        )
        if summary.height:
            plotted = summary.with_columns(
                pl.concat_str([pl.col("department"), pl.lit(" / "), pl.col("category")]).alias(
                    "category_label"
                )
            )
            base.bars(
                plotted,
                "category_label",
                "participation",
                "Category participation, past 365 days",
                "patronage_category",
            )
        base.table(summary)
        st.caption(
            f"Denominator: all {denominator:,} customers in the segment, including recently inactive customers. Spend share is separate from participation. Recurring means purchases on at least three distinct days; gap medians use completed in-window gaps and are not pooled event medians."
        )
        st.download_button(
            "Export category characteristics",
            base.export_csv(summary),
            "patronage_categories.csv",
            "text/csv",
        )
    elif section == "Transitions":
        prior = [d for d in dates if d < chosen]
        if not prior:
            st.info("Choose a cutoff with an earlier snapshot.")
            return
        origin = st.selectbox("Earlier patronage cutoff", prior, index=len(prior) - 1)
        parts = []
        for p in (target / "snapshots" / chosen).glob("*.parquet"):
            before = target / "snapshots" / origin / p.name
            if before.exists():
                a = pl.scan_parquet(before).select(
                    "customer_id", pl.col("observed_patronage").alias("from_label")
                )
                b = pl.scan_parquet(p).select(
                    "customer_id", pl.col("observed_patronage").alias("to_label")
                )
                parts.append(
                    a.join(b, on="customer_id", how="inner", validate="1:1")
                    .group_by("from_label", "to_label")
                    .len()
                )
        if not parts:
            st.info("No shared customer partitions.")
            return
        flows = base.collect(
            pl.concat(parts)
            .group_by("from_label", "to_label")
            .agg(pl.col("len").sum())
            .with_columns((pl.col("len") / pl.col("len").sum().over("from_label")).alias("rate"))
        )
        xs, ys, z = base.bounded_grid(flows.to_dicts(), "from_label", "to_label", "rate", limit=20)
        base.chart(
            go.Figure(
                go.Heatmap(
                    x=xs,
                    y=ys,
                    z=z,
                    zmin=0,
                    zmax=1,
                    colorscale="Blues",
                    colorbar=dict(tickformat=".0%"),
                    hoverongaps=False,
                )
            ),
            "Origin-normalized patronage transitions",
            "patronage_transitions",
            "Destination",
            "Origin",
        )
        base.table(flows)
        st.caption(
            "Same frozen rules at both cutoffs; only matched customers are included. Calendar exposure changes can move labels. Transitions do not establish churn or causality."
        )
    elif section == "Sensitivity":
        outputs = []
        for multiple in [1.5, 2.0, 3.0]:
            alternative = replace(rules, overdue_multiple=multiple)
            alternative_rows = label_frame(
                rows.with_columns(pl.col("observed_patronage").alias("original")),
                alternative,
                confirmed,
            )
            summary = base.collect(
                alternative_rows.group_by("observed_patronage").agg(
                    pl.len().alias("customers"),
                    (pl.col("original") != pl.col("observed_patronage"))
                    .sum()
                    .alias("changed_customers"),
                )
            ).with_columns(pl.lit(multiple).alias("overdue_multiple"))
            outputs.append(summary)
        base.table(pl.concat(outputs))
        st.caption(
            "Local sensitivity of the current hiatus threshold only; resume_events remains fixed at the build rule. To test full rule sensitivity, change the build parameters and rebuild. Stability is not evidence of business usefulness."
        )
    elif section == "Customer":
        consent = st.checkbox("Show identifiable patronage details")
        identifier = st.text_input("Exact customer_id")
        if consent and identifier:
            base.table(rows.filter(pl.col("customer_id") == identifier))
            base.table(
                pl.scan_parquet(str(target / "categories" / chosen / "*.parquet")).filter(
                    pl.col("customer_id") == identifier
                )
            )
            st.caption(
                "These details support interpretation, not an inferred demographic persona. Use the original visual dashboard for the purchase-day journey."
            )
    else:
        st.json(specification)
        st.markdown(
            "Precedence: unconfirmed capture; one-day history; recent repeat; insufficient established evidence; regular-and-overdue; resumed repeat; regular-and-active recurring; established occasional. See PATRONAGE.md for definitions and limitations."
        )
        st.download_button(
            "Export patronage definitions and source identity",
            json.dumps(specification, indent=2),
            "patronage_manifest.json",
            "application/json",
        )
