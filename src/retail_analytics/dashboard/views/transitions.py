"""Transitions view module."""

from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from retail_analytics.visualization.chart_components import (
    ChartEvidence,
    ChartCard,
    create_chart_card,
    build_color_map,
    get_color,
    safe_list,
    bounded_grid,
)
from retail_analytics.visualization.visual_insights_core import transition_summary
from retail_analytics.dashboard.app import read_json, apply_filters, collect, table


def render_transitions(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Transitions view - Sankey and transition matrix."""
    from retail_analytics.dashboard.app import read_json, apply_filters

    manifest = read_json(root / "run.json")
    history = [x for x in manifest["snapshots"] if x <= st.session_state.get("chosen", "")]

    if len(history) < 2:
        st.info("Choose a cutoff with an earlier snapshot to compare transitions.")
        return

    origin = st.selectbox("Origin cutoff", history[:-1], index=len(history) - 2)
    dim = st.selectbox("Transition layer", ["behavior", "trajectory", "lifecycle", "value_tier", "evidence"])

    if dim != "value_tier":
        st.info(
            f"Showing {dim} transitions as Sankey diagram. For value-tier waterfall chart, use the 'Value-Tier Transitions' view."
        )

    try:
        before = apply_filters(
            pl.scan_parquet([str(p) for p in (root / "snapshots" / origin).glob("*.parquet")]),
            {},
        )
        after = apply_filters(
            pl.scan_parquet([str(p) for p in (root / "snapshots" / st.session_state.get("chosen", "")).glob("*.parquet")]),
            {},
        )
        flows, counts = transition_summary(before, after, dim)
        if flows.is_empty():
            st.info("No matched customers at both cutoffs.")
            return

        # Show reconciliation counts
        st.caption(
            f"Reconciliation: origin={counts['origin_n']}, destination={counts['destination_n']}, "
            f"matched={counts['matched_n']}, origin_unmatched={counts['origin_unmatched']}, "
            f"destination_unmatched={counts['destination_unmatched']}. "
            f"Unmatched origin customers are not churn; unmatched destination customers are new arrivals."
        )

        # Sankey diagram
        sources = sorted(flows["from_label"].unique().to_list())
        targets = sorted(flows["to_label"].unique().to_list())
        fig = go.Figure(
            go.Sankey(
                node=dict(
                    label=["From: " + s for s in sources] + ["To: " + t for t in targets],
                    color=[get_color(s, color_map) for s in sources + targets],
                    pad=15,
                ),
                link=dict(
                    source=[sources.index(r["from_label"]) for r in flows.to_dicts()],
                    target=[
                        len(sources) + targets.index(r["to_label"]) for r in flows.to_dicts()
                    ],
                    value=safe_list(flows["len"]),
                ),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        xs, ys, z = bounded_grid(flows.to_dicts(), "from_label", "to_label", "rate", limit=20)
        fig = go.Figure(
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
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Matched customers: {flows['len'].sum():,}. Sankey shows the 30 largest of {flows.height} flows; heatmap axes are capped at 20 labels. "
            f"Rates use all matched, destination-filtered customers per origin, so displayed cells need not sum to 100%. "
            f"Missing pairs are blank. Unmatched origin customers are not churn; unmatched destination customers are new arrivals."
        )
        table(flows)
        st.download_button(
            "Export transition summary",
            data=flows.write_csv().encode("utf8"),
            file_name="transitions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Transitions analysis failed: {type(e).__name__}: {e}")


__all__ = ["render_transitions"]