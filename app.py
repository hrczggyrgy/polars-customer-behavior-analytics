"""Scientific Streamlit dashboard for retail_polars_workflow.py."""
import hashlib
import json
import subprocess
import shutil
import sys
import tempfile
import uuid
from datetime import date
from pathlib import Path
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from dashboard_backend import cutoffs, wilson

MAX_BYTES = 100 * 1024 * 1024
PAGE_ROWS = 2000
TIMEOUT = 600
FEATURES = ["spend_90d", "purchases_90d", "median_basket", "median_gap", "gap_cv", "lateness", "new_share", "repeat_share", "trial_conversion", "category_hhi", "products_last6days", "categories_last6days", "frequency_change", "basket_change"]


def read_json(path):
    return json.loads(Path(path).read_text())


def reset():
    workspace = st.session_state.pop("workspace", None)
    if workspace:
        workspace.cleanup()
    for key in ["fingerprint", "inspection_key", "result", "result_key"]:
        st.session_state.pop(key, None)


def drop_result():
    old = st.session_state.pop("result", None)
    st.session_state.pop("result_key", None)
    if old and "workspace" in st.session_state:
        parent = Path(st.session_state["workspace"].name).resolve()
        target = Path(old).resolve()
        if target.parent == parent and target.name.startswith("run_"):
            shutil.rmtree(target, ignore_errors=True)


def worker(job):
    root = Path(job["workspace"])
    control = root / "job.json"
    control.write_text(json.dumps(job))
    status = root / "worker_status.json"
    status.unlink(missing_ok=True)
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).with_name("dashboard_backend.py")), "--job", str(control)], capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        raise ValueError("Analysis timed out. Use a smaller upload or fewer cutoffs.")
    if not status.exists():
        raise ValueError("Worker stopped unexpectedly. Check server resources and dependencies; no result was accepted.")
    report = read_json(status)
    if process.returncode or not report["ok"]:
        message = report.get("message", "Worker failed")
        if report.get("error_type") not in ["ValueError", "AssertionError"]:
            message = "Data processing failed. Inspect schema, formats and server resources; raw server diagnostics are not displayed."
        raise ValueError(message)


def collect(frame):
    return frame.collect(engine="streaming")


def table(frame, caption=None):
    data = collect(frame.limit(PAGE_ROWS)) if isinstance(frame, pl.LazyFrame) else frame.head(PAGE_ROWS)
    st.dataframe(data, hide_index=True, use_container_width=True)
    if caption:
        st.caption(caption)


def bar(frame, x, y, title):
    data = collect(frame) if isinstance(frame, pl.LazyFrame) else frame
    if data.is_empty():
        st.info("No eligible observations for this view.")
        return
    figure = go.Figure(go.Bar(x=data[x].to_list(), y=data[y].to_list()))
    figure.update_layout(title=title, xaxis_title=x, yaxis_title=y)
    st.plotly_chart(figure, use_container_width=True)


def export_csv(data):
    textcols = [x for x, dtype in data.schema.items() if dtype == pl.String]
    safe = data.with_columns(*[pl.when(pl.col(x).str.contains(r"^[=+\-@\t\r]")).then(pl.concat_str([pl.lit("'"), pl.col(x)])).otherwise(pl.col(x)).alias(x) for x in textcols])
    return safe.write_csv().encode("utf8")


def display_results(root, coverage):
    manifest = read_json(root / "run.json")
    chosen = st.selectbox("Result cutoff", manifest["snapshots"], index=len(manifest["snapshots"]) - 1)
    snapshots = pl.scan_parquet(str(root / "snapshots" / chosen / "*.parquet"))
    metric = collect(snapshots.select(pl.len().alias("customers"), pl.col("spend_90d").sum().alias("sales"), pl.col("purchases_90d").sum().alias("baskets"), (pl.col("purchases_90d") > 0).sum().alias("active"))).row(0, named=True)
    for box, label, value in zip(st.columns(4), ["Observed customers", "Recorded spend: 90d", "Baskets: 90d", "Active customers: 90d"], [metric["customers"], f'{metric["sales"]:,.2f}', metric["baskets"], metric["active"]]):
        box.metric(label, value)
    st.caption("Amounts are recorded currency units, not profit. Customers include inactive observed customers. Windows end at the selected cutoff.")
    views = st.tabs(["Overview", "Data quality", "Segments", "Feature diagnostics", "Cohorts", "Transitions", "Customer explorer", "Methods & export"])
    with views[0]:
        rows = pl.scan_parquet(str(root / "snapshots" / "*" / "*.parquet"))
        summary = collect(rows.group_by("snapshot").agg(pl.col("spend_90d").sum().alias("spend_90d"), (pl.col("purchases_90d") > 0).sum().alias("active_90d")).sort("snapshot"))
        fig = go.Figure(go.Scatter(x=summary["snapshot"].to_list(), y=summary["spend_90d"].to_list(), mode="lines+markers"))
        fig.update_layout(title="Rolling 90-day recorded spend", xaxis_title="Cutoff", yaxis_title="Recorded spend")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Overlapping rolling windows are not independent; observed customer populations may change.")
        bar(snapshots.group_by("lifecycle").len(), "lifecycle", "len", "Lifecycle distribution")
    with views[1]:
        st.json(read_json(root.parent / "inspection.json"))
        st.json(read_json(root / "quality.json"))
        table(snapshots.group_by("evidence").len(), "Evidence labels are rules, not confidence probabilities.")
        table(pl.scan_parquet(root / "product_variation.parquet"), "Product variation audit; at most 2,000 displayed rows. Descriptions may contain commercially sensitive information.")
    with views[2]:
        dimension = st.selectbox("Segmentation layer", ["behavior", "trajectory", "lifecycle", "value_tier", "evidence"])
        minimum = st.number_input("Minimum group size for outcome comparison", 1, 10000, 20)
        aggregate = collect(snapshots.group_by(dimension).agg(pl.len().alias("customers"), pl.col("spend_90d").sum().alias("spend_90d"), pl.col("median_gap").median().alias("median_gap"), pl.col("new_share").mean().alias("exploration"), pl.col("future_purchases_30d").count().alias("followup_n"), (pl.col("future_purchases_30d") > 0).sum().alias("future_buyers")).sort("customers", descending=True))
        bar(aggregate.head(30), dimension, "customers", "Customer counts by segment")
        records = []
        for row in aggregate.to_dicts():
            n = row["followup_n"]
            allowed = coverage and n >= minimum
            lo, hi = wilson(row["future_buyers"], n) if allowed else (None, None)
            row.update(future_rate=row["future_buyers"] / n if allowed else None, wilson95_low=lo, wilson95_high=hi)
            if not coverage:
                row["followup_n"], row["future_buyers"] = None, None
            records.append(row)
        data = pl.DataFrame(records) if records else aggregate
        table(data)
        st.caption("95% Wilson intervals assume independent customers at one cutoff. They are not causal effects or certainty about future performance. Small groups and unconfirmed coverage suppress outcome estimates.")
        st.download_button("Download segment summary CSV", export_csv(data), "segment_summary.csv", "text/csv")
    with views[3]:
        available = [x for x in FEATURES if x in snapshots.collect_schema().names()]
        diagnostics = []
        for feature in available:
            item = collect(snapshots.select(pl.lit(feature).alias("feature"), pl.col(feature).count().alias("non_null"), pl.col(feature).null_count().alias("missing"), pl.col(feature).median().alias("median"), pl.col(feature).quantile(.1).alias("p10"), pl.col(feature).quantile(.9).alias("p90"))).row(0, named=True)
            diagnostics.append(item)
        table(pl.DataFrame(diagnostics))
        x = st.selectbox("Scatter X", available, index=0)
        y = st.selectbox("Scatter Y", available, index=min(1, len(available) - 1))
        sample = collect(snapshots.filter(pl.col(x).is_finite() & pl.col(y).is_finite()).with_columns(pl.col("customer_id").hash(seed=29).alias("rank")).sort("rank").limit(PAGE_ROWS).select(x, y, "evidence")) if x != y else None
        if sample is not None and not sample.is_empty():
            figure = go.Figure(go.Scattergl(x=sample[x].to_list(), y=sample[y].to_list(), mode="markers", marker=dict(size=5, opacity=.5)))
            figure.update_layout(xaxis_title=x, yaxis_title=y)
            st.plotly_chart(figure, use_container_width=True)
        st.caption("Scatter uses up to 2,000 hash-selected customers; it is exploratory, not a statistical test. Missing means unavailable, not zero.")
    with views[4]:
        cohort = snapshots.with_columns(pl.col("first_purchase").dt.truncate("1mo").alias("first_observed_month")).group_by("first_observed_month").agg(pl.len().alias("customers"), pl.col("spend_history").mean().alias("mean_cumulative_spend"), (pl.col("purchases_30d") > 0).mean().alias("rolling30d_activity")).sort("first_observed_month")
        table(cohort, "Observed cohorts, not confirmed acquisition cohorts. Calendar exposure differs; cumulative spend is not a matched-age comparison.")
    with views[5]:
        dimension = st.selectbox("Transition dimension", ["behavior", "trajectory", "lifecycle", "value_tier"], key="transition_dim")
        fragments = []
        for path in sorted((root / "snapshots").glob("*/*.parquet")):
            if path.parent.name <= chosen:
                fragments.append(path)
        groups = {}
        for path in fragments:
            groups.setdefault(path.name, []).append(str(path))
        parts = []
        for paths in groups.values():
            rows = pl.scan_parquet(paths)
            parts.append(rows.with_columns(pl.col(dimension).shift(1).over("customer_id", order_by="snapshot").alias("from_label"), pl.col("snapshot").shift(1).over("customer_id", order_by="snapshot").alias("from_date")).filter((pl.col("snapshot") == date.fromisoformat(chosen)) & pl.col("from_label").is_not_null()).group_by(["from_date", "from_label", dimension]).len().rename({dimension: "to_label"}))
        transitions = pl.concat(parts).group_by(["from_date", "from_label", "to_label"]).agg(pl.col("len").sum()).with_columns((pl.col("len") / pl.col("len").sum().over(["from_date", "from_label"])).alias("rate")).sort("len", descending=True)
        table(transitions, "Rates use observed origin populations in both snapshots; definitions and reference thresholds are frozen. One cutoff yields no transitions.")
    with views[6]:
        consent = st.checkbox("Show identifiable customer details in this session")
        customer = st.text_input("Exact customer_id; no full ID list is loaded")
        if consent and customer:
            profile = snapshots.filter(pl.col("customer_id") == customer)
            if not coverage:
                profile = profile.select(pl.exclude("^future_.*$", "^eligible_followup_.*$"))
            table(profile)
            paths = list((root / "events" / chosen).glob("*/baskets.parquet"))
            timeline = collect(pl.scan_parquet(paths).filter(pl.col("customer_id") == customer).group_by("date").agg(pl.col("spend").sum()).sort("date", descending=True).limit(PAGE_ROWS).sort("date"))
            if not timeline.is_empty():
                fig = go.Figure(go.Scatter(x=timeline["date"].to_list(), y=timeline["spend"].to_list(), mode="lines+markers"))
                st.plotly_chart(fig, use_container_width=True)
    with views[7]:
        st.json(manifest)
        st.json(read_json(root / "thresholds.json"))
        st.info("No permanent-churn, profit-CLV, price-elasticity or causal marketing conclusions. Rule thresholds require temporal validation. Package attributes are not inferred. No automatic seasonal adjustment.")
        st.caption("Data remain in this app server's temporary session workspace, not GitHub. Clear the session to remove its workspace. Browser disconnect cleanup is best-effort, not guaranteed secure erasure.")
        st.download_button("Download reproducibility manifest", json.dumps(manifest, indent=2), "analysis_manifest.json", "application/json")


def main():
    st.set_page_config(page_title="Retail behavioral science", page_icon="📊", layout="wide")
    st.title("Retail behavioral science dashboard")
    st.caption("Nine-column transaction evidence · Polars · transparent segmentation · eligibility-aware outcomes")
    st.sidebar.warning("Uploads are processed on the app server. Use a trusted, access-controlled deployment for customer data.")
    if st.sidebar.button("Clear data and results"):
        reset()
        st.session_state["upload_nonce"] = st.session_state.get("upload_nonce", 0) + 1
        st.rerun()
    uploaded = st.sidebar.file_uploader("Upload Parquet or UTF-8 CSV", type=["parquet", "csv"], key=f'upload_{st.session_state.get("upload_nonce", 0)}')
    if uploaded is None:
        reset()
        st.info("Upload a file with customer_id, transaction_date, transaction_id, product_id, product_description, department, category, price, quantity.")
        return
    if uploaded.size > MAX_BYTES:
        st.error("Upload limit is 100 MiB; reduce the file or run the CLI on larger data.")
        return
    digest = hashlib.sha256(uploaded.getbuffer()).hexdigest()
    if st.session_state.get("fingerprint") != digest:
        reset()
        st.session_state["workspace"] = tempfile.TemporaryDirectory(prefix="retail_dashboard_")
        st.session_state["fingerprint"] = digest
        root = Path(st.session_state["workspace"].name)
        suffix = Path(uploaded.name).suffix.lower()
        source = root / ("input" + suffix)
        source.write_bytes(uploaded.getbuffer())
    root = Path(st.session_state["workspace"].name)
    source = next(root.glob("input.*"))
    kind = "parquet" if source.suffix == ".parquet" else "csv"
    mode = st.sidebar.selectbox("Meaning of price", ["Select explicitly", "Unit selling price", "Line total"])
    if mode == "Select explicitly":
        st.info("Confirm what price represents before analysis.")
        return
    fmt = st.sidebar.text_input("Date format (blank = ISO YYYY-MM-DD)")
    decimal = st.sidebar.checkbox("Decimal comma in numeric text; no thousands separators")
    separator = st.sidebar.selectbox("CSV delimiter", [",", ";", "\t", "|"]) if kind == "csv" else ","
    base = dict(workspace=str(root), source=str(source), kind=kind, separator=separator, date_format=fmt or None, decimal_comma=decimal, price_mode="unit" if mode == "Unit selling price" else "line_total", sha256=digest)

    # Column mapping step - preview schema first
    if st.session_state.get("inspection_key") != inspection_key:
        drop_result()

    # Always show column mapping if we have an upload but haven't validated yet
    if "column_mapping" not in st.session_state:
        st.session_state["column_mapping"] = {}

    # Preview schema button
    if not st.session_state.get("inspection_key"):
        if st.sidebar.button("Preview columns"):
            try:
                with st.spinner("Reading file schema..."):
                    worker(dict(base, action="preview"))
                st.session_state["preview_schema"] = read_json(root / "preview.json")
            except Exception as exc:
                st.error(str(exc))
                return

    # Column mapping UI if preview exists
    if "preview_schema" in st.session_state and not st.session_state.get("inspection_key"):
        preview = st.session_state["preview_schema"]
        available_cols = preview.get("columns", [])
        st.sidebar.subheader("Column Mapping")
        st.sidebar.caption("Map your columns to the required schema:")
        
        required_cols = ["customer_id", "transaction_date", "transaction_id", "product_id", "product_description", "department", "category", "price", "quantity"]
        mapping = {}
        for req in required_cols:
            default = st.session_state["column_mapping"].get(req, "")
            options = [""] + available_cols
            try:
                index = options.index(default) if default in options else 0
            except ValueError:
                index = 0
            selected = st.sidebar.selectbox(f"{req}", options, index=index, key=f"map_{req}")
            if selected:
                mapping[req] = selected
        
        st.session_state["column_mapping"] = mapping
        
        # Check if all required columns are mapped
        missing_mapped = [c for c in required_cols if c not in mapping or not mapping[c]]
        if missing_mapped:
            st.sidebar.warning(f"Missing mappings: {', '.join(missing_mapped)}")
        else:
            # Add mapping to base and validate
            base["column_mapping"] = mapping
            if st.sidebar.button("Validate upload"):
                try:
                    with st.spinner("Validating schema, values and basket keys in an isolated worker..."):
                        worker(dict(base, action="inspect"))
                    st.session_state["inspection_key"] = inspection_key
                    st.session_state.pop("preview_schema", None)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
                    if (root / "inspection.json").exists():
                        st.json(read_json(root / "inspection.json"))
                    return

    metadata = read_json(root / "inspection.json") if st.session_state.get("inspection_key") else None
    first, last = date.fromisoformat(metadata["minimum_date"]), date.fromisoformat(metadata["maximum_date"])
    st.sidebar.caption(f'{metadata["rows"]:,} rows · {metadata["valid_customers"]:,} valid customer IDs · {first} to {last}')
    exclude = st.sidebar.checkbox("Exclude invalid records after review", value=False)
    if metadata["invalid_rows"] and not exclude:
        st.warning(f'{metadata["invalid_rows"]:,} rows are invalid. Review the inspection and explicitly choose exclusion to continue.')
        st.json(metadata)
        return
    coverage = st.sidebar.checkbox("I confirm complete purchase capture through the date below")
    end = st.sidebar.date_input("Confirmed coverage end", value=last, min_value=last, max_value=max(last, date.today())) if coverage else last
    chosen = st.sidebar.date_input("Latest analysis cutoff", value=last, min_value=first, max_value=end)
    count = st.sidebar.slider("Historical cutoffs (latest plus prior month ends)", 1, 12, 3)
    partitions = st.sidebar.slider("Customer partitions", 1, 32, 8)
    dates = cutoffs(chosen, first, count)
    st.sidebar.caption("Cutoffs: " + ", ".join(map(str, dates)))
    job = dict(base, action="analyze", snapshots=[str(x) for x in dates], buckets=partitions, exclude_invalid=exclude, coverage_confirmed=coverage, observed_through=str(end))
    result_key = json.dumps(job, sort_keys=True)
    if st.session_state.get("result_key") != result_key:
        drop_result()
    if st.sidebar.button("Run analysis", type="primary"):
        drop_result()
        target = root / ("run_" + uuid.uuid4().hex)
        try:
            with st.spinner("Building historical events, features and segments; 10-minute time limit..."):
                worker(dict(job, output=str(target)))
            st.session_state["result"], st.session_state["result_key"] = str(target), result_key
            st.success("Analysis completed. Labels are descriptive; inspect eligibility and uncertainty.")
        except Exception as exc:
            st.error(str(exc))
            if (target / "quality.json").exists():
                st.json(read_json(target / "quality.json"))
            shutil.rmtree(target, ignore_errors=True)
            return
    if not coverage:
        st.warning("Coverage is unconfirmed: future-purchase comparisons are suppressed. Missing transactions cannot be interpreted as non-purchases.")
    if st.session_state.get("result"):
        try:
            display_results(Path(st.session_state["result"]), coverage)
        except Exception as exc:
            st.error("A dashboard view could not render. Check output schemas and dependency compatibility.")
            st.caption(type(exc).__name__)
    else:
        st.json(metadata)
        st.info("Run analysis to create the scientific dashboard.")


if __name__ == "__main__":
    main()
