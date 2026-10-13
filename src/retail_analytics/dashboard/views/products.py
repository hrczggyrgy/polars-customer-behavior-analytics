"""Products & Categories views: Category Penetration×Growth, Category Opportunity Map, Category Participation, Product Association Network, Trial Conversion Intervals, Product Affinity."""

from datetime import date, timedelta
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from retail_analytics.visualization.chart_components import (
    ChartEvidence,
    ChartSpec,
    ChartCard,
    create_chart_card,
    SupportThresholds,
    EvidenceState,
    evaluate_evidence,
    build_color_map,
    get_color,
    safe_list,
)
from retail_analytics.visualization.visual_insights_core import (
    category_opportunity_map,
    category_participation_comparison,
    category_penetration_growth_portfolio,
    product_association_network,
    trial_conversion_interval_plot,
)
from retail_analytics.visualization.product_affinity import (
    ProductAffinityConfig,
    basket_composition_by_segment,
    cross_sell_recommendations,
    new_to_repeat_pathways,
    product_affinity_analysis,
)
from retail_analytics.dashboard.app import read_json


def render_category_penetration_growth(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Category Penetration × Growth Portfolio."""
    st.subheader("Category Penetration vs Growth Portfolio")
    st.caption(
        "X: Current category penetration (% of customers buying category). "
        "Y: Change in participation vs previous window (percentage points). "
        "Bubble size: Category spend share. Color: Repeat buyer share. "
        "Identifies expanding/declining categories and their commercial weight."
    )
    
    try:
        manifest = read_json(root / "run.json")
        history = [x for x in manifest["snapshots"] if x <= st.session_state.get("chosen", "")]
        
        if len(history) < 2:
            st.info("Need at least two cutoffs for period comparison.")
        else:
            origin = st.selectbox(
                "Previous cutoff", history[:-1], index=len(history) - 2, key="cpg_origin"
            )
            top_n = st.slider("Top categories", 10, 50, 30, key="cpg_top")
            
            validated_path = root / "validated.parquet"
            if not validated_path.exists():
                st.info("No validated data available.")
            else:
                validated = pl.scan_parquet(validated_path)
                current_cutoff = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
                previous_cutoff = date.fromisoformat(origin)
                
                from retail_analytics.visualization.visual_insights_core import category_penetration_growth_portfolio
                
                portfolio = category_penetration_growth_portfolio(
                    validated, rows, current_cutoff, previous_cutoff, top_n
                )
                
                if portfolio.is_empty():
                    st.info("No category data for this comparison.")
                else:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=portfolio["curr_penetration"].to_list(),
                            y=portfolio["penetration_change_pp"].to_list(),
                            mode="markers+text",
                            text=portfolio["label"].to_list(),
                            textposition="top center",
                            textfont=dict(size=9),
                            marker=dict(
                                size=[
                                    max(10, min(80, s * 300))
                                    for s in portfolio["spend_share"].to_list()
                                ],
                                color=portfolio["repeat_buyer_share"].to_list(),
                                colorscale="Viridis",
                                showscale=True,
                                colorbar=dict(title="Repeat buyer share", thickness=15),
                                opacity=0.7,
                                line=dict(width=1, color="white"),
                            ),
                            customdata=list(
                                zip(
                                    portfolio["curr_buyers"].to_list(),
                                    portfolio["prev_buyers"].to_list(),
                                    portfolio["curr_spend"].to_list(),
                                    portfolio["prev_spend"].to_list(),
                                    portfolio["spend_share"].to_list(),
                                    portfolio["repeat_buyer_share"].to_list(),
                                )
                            ),
                            hovertemplate=(
                                "%{text}<br>"
                                "Current penetration: %{x:.1%}<br>"
                                "Penetration change (pp): %{y:.2f}<br>"
                                "Current buyers: %{customdata[0]:,}<br>"
                                "Previous buyers: %{customdata[1]:,}<br>"
                                "Current spend: %{customdata[2]:,.2f}<br>"
                                "Previous spend: %{customdata[3]:,.2f}<br>"
                                "Spend share: %{customdata[4]:.1%}<br>"
                                "Repeat buyer share: %{customdata[5]:.1%}<extra></extra>"
                            ),
                        )
                    )
                    # Add quadrant lines
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                    fig.add_vline(x=portfolio["curr_penetration"].median(), line_dash="dash", line_color="gray", line_width=1)
                    fig.update_xaxes(tickformat=".0%", title="Current Penetration (% of customers)")
                    fig.update_yaxes(tickformat=".1f", title="Penetration Change (percentage points)")
                    
                    card = create_chart_card(
                        key="cat_penetration_growth",
                        title="Category Penetration × Growth Portfolio",
                        figure=fig,
                        question="Which categories are growing/shrinking in customer reach?",
                        takeaway=f"Top {top_n} categories; bubble size = spend share, color = repeat buyer share",
                        evidence=ChartEvidence(
                            numerator="Category buyers in current period",
                            denominator="Population customers in each period",
                            population_size=None,
                            comparison_period=f"{origin} → {st.session_state.get('chosen')}",
                            confidence_interval="Not shown",
                            interval_type="Descriptive change",
                            eligibility_criteria="Customers with ≥1 shopping day in window",
                        ),
                        how_to_read=(
                            "X-axis: Current penetration (% of customers buying category). "
                            "Y-axis: Change in penetration (pp, current - previous). "
                            "Bubble size: Current spend share. Color: Repeat buyer share. "
                            "Top-right: growing reach + high penetration. Bottom-left: declining reach."
                        ),
                        caveat=(
                            "Participation = category buyers / period population (same denominator per period). "
                            "Categories with zero buyers in a period show 0% participation. "
                            "Change in percentage points, not relative change."
                        ),
                        next_step="Check 'Category Opportunity Map' for penetration×intensity analysis.",
                        alt_description=f"Scatter plot showing category penetration vs penetration change with bubble size by spend share and color by repeat buyer share.",
                    )
                    card.render()
                    
                    display_cols = [
                        "label",
                        "curr_penetration",
                        "prev_penetration",
                        "penetration_change_pp",
                        "curr_buyers",
                        "prev_buyers",
                        "curr_spend",
                        "prev_spend",
                        "spend_share",
                        "repeat_buyer_share",
                    ]
                    table(portfolio.select(display_cols))
                    
    except Exception as e:
        st.error(f"Category Penetration × Growth failed: {type(e).__name__}: {e}")


def render_category_opportunity_map(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Category Opportunity Map."""
    st.subheader("Category Opportunity Map: Penetration × Intensity")
    st.caption(
        "X: Penetration (% of customers buying category). Y: Purchase occasions per category buyer. "
        "Bubble: Category spend share. Color: Repeat buyer share. "
        "Research basis: Dirichlet model for repeat buying (Ehrenberg, 1988)."
    )
    
    try:
        validated_path = root / "validated.parquet"
        if not validated_path.exists():
            st.info("No validated data available.")
        else:
            validated = pl.scan_parquet(validated_path)
            cutoff_date = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
            top_categories = st.slider("Top categories to show", 10, 50, 30)
            
            opp_map = category_opportunity_map(validated, rows, cutoff_date, top_categories=top_categories)
            
            if opp_map.is_empty():
                st.info("No category data for this population.")
            else:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=opp_map["penetration"].to_list(),
                        y=opp_map["occasions_per_buyer"].to_list(),
                        mode="markers+text",
                        text=opp_map["label"].to_list(),
                        textposition="top center",
                        marker=dict(
                            size=[
                                max(10, min(80, s * 200))
                                for s in opp_map["spend_share"].to_list()
                            ],
                            color=opp_map["repeat_buyer_share"].to_list(),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Repeat buyer share", thickness=15),
                            opacity=0.7,
                        ),
                        customdata=list(
                            zip(
                                opp_map["buyers"].to_list(),
                                opp_map["spend_share"].to_list(),
                                opp_map["repeat_buyer_share"].to_list(),
                                opp_map["total_spend"].to_list(),
                            )
                        ),
                        hovertemplate=(
                            "%{text}<br>"
                            "Penetration: %{x:.1%}<br>"
                            "Occasions/buyer: %{y:.2f}<br>"
                            "Buyers: %{customdata[0]:,}<br>"
                            "Spend share: %{customdata[1]:.1%}<br>"
                            "Repeat buyer share: %{customdata[2]:.1%}<br>"
                            "Total spend: %{customdata[3]:,.2f}<extra></extra>"
                        ),
                    )
                )
                
                card = create_chart_card(
                    key="cat_opp_map",
                    title="Category Opportunity Map",
                    figure=fig,
                    question="Which categories have broad reach but low intensity, or narrow reach but high intensity?",
                    takeaway=f"Top {top_categories} categories by spend share; bubble size=spend share, color=repeat buyer share",
                    evidence=ChartEvidence(
                        numerator="Category buyers",
                        denominator="Population customers",
                        population_size=None,
                        comparison_period="365-day window",
                        confidence_interval="Not applicable",
                        interval_type="Descriptive rates only",
                        eligibility_criteria="Customers with ≥1 shopping day in window",
                    ),
                    how_to_read=(
                        "X-axis: Penetration (% of customers buying category). "
                        "Y-axis: Purchase occasions per category buyer. "
                        "Bubble size ∝ spend share. Color = repeat buyer share. "
                        "Top-right: high penetration + high intensity (core categories). "
                        "Bottom-left: low penetration + low intensity (niche). "
                        "Top-left: broad reach, low intensity (opportunity). "
                        "Bottom-right: narrow reach, high intensity (loyal niche)."
                    ),
                    caveat=(
                        "Categories may overlap within customers, so category shares need not sum to 100%. "
                        "Spend is recorded spend, not margin. "
                        "Research: Ehrenberg (1988), Repeat Buying; Dirichlet model benchmarks."
                    ),
                    next_step="Check 'Category Penetration × Growth' for period-over-period changes.",
                    alt_description="Scatter plot showing category penetration vs purchase occasions per buyer, bubble size by spend share, color by repeat buyer share.",
                )
                card.render()
                
                table(
                    opp_map.select(
                        "label",
                        "buyers",
                        "penetration",
                        "occasions_per_buyer",
                        "spend_share",
                        "repeat_buyer_share",
                        "total_spend",
                    )
                )
                
    except Exception as e:
        st.error(f"Category opportunity map failed: {type(e).__name__}: {e}")


def render_category_participation(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Category Participation."""
    st.subheader("Category Participation: Recent 90d vs Previous 90d")
    st.caption(
        "Participation = customers purchasing from category during period ÷ fixed comparison population. "
        "Recent 90d: [C-89, C]. Previous 90d: [C-179, C-90]. "
        "Same population denominator for both periods. Change in percentage points shown."
    )
    
    try:
        manifest = read_json(root / "run.json")
        history = [x for x in manifest["snapshots"] if x <= st.session_state.get("chosen", "")]
        
        if len(history) < 2:
            st.info("Need at least two cutoffs for category participation comparison.")
        else:
            origin = st.selectbox(
                "Previous cutoff", history[:-1], index=len(history) - 2, key="cat_part_origin"
            )
            top_n = st.slider("Top N categories to show", 5, 50, 20)
            
            validated_path = root / "validated.parquet"
            if not validated_path.exists():
                st.info("No validated data available for this analysis.")
            else:
                validated = pl.scan_parquet(validated_path)
                from retail_analytics.visualization.visual_insights_core import category_participation_comparison
                
                part = category_participation_comparison(validated, rows)
                
                if part.is_empty():
                    st.info(
                        "No category participation data for this population. This may be due to filters removing all customers, or no transactions in the comparison windows."
                    )
                else:
                    display = part.head(top_n)
                    
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            y=safe_list(display["label"]),
                            x=safe_list(display["recent_participation"]),
                            name="Recent 90d",
                            orientation="h",
                            marker_color="#0072B2",
                            customdata=list(
                                zip(
                                    safe_list(display["recent_buyers"]),
                                    safe_list(display["recent_spend"]),
                                    safe_list(display["previous_buyers"]),
                                    safe_list(display["previous_spend"]),
                                    safe_list(display["participation_change_pp"]),
                                )
                            ),
                            hovertemplate=(
                                "%{y}<br>"
                                "Recent participation: %{x:.1%}<br>"
                                "Recent buyers: %{customdata[0]:,}<br>"
                                "Recent spend: %{customdata[1]:,.2f}<br>"
                                "Previous buyers: %{customdata[2]:,}<br>"
                                "Previous spend: %{customdata[3]:,.2f}<br>"
                                "Change (pp): %{customdata[4]:.2f}<extra></extra>"
                            ),
                        )
                    )
                    fig.add_trace(
                        go.Bar(
                            y=safe_list(display["label"]),
                            x=safe_list(display["previous_participation"]),
                            name="Previous 90d",
                            orientation="h",
                            marker_color="#E69F00",
                            opacity=0.6,
                            customdata=list(
                                zip(
                                    safe_list(display["recent_buyers"]),
                                    safe_list(display["recent_spend"]),
                                    safe_list(display["previous_buyers"]),
                                    safe_list(display["previous_spend"]),
                                    safe_list(display["participation_change_pp"]),
                                )
                            ),
                            hovertemplate=(
                                "%{y}<br>"
                                "Previous participation: %{x:.1%}<br>"
                                "Recent buyers: %{customdata[0]:,}<br>"
                                "Recent spend: %{customdata[1]:,.2f}<br>"
                                "Previous buyers: %{customdata[2]:,}<br>"
                                "Previous spend: %{customdata[3]:,.2f}<br>"
                                "Change (pp): %{customdata[4]:.2f}<extra></extra>"
                            ),
                        )
                    )
                    fig.update_layout(barmode="group")
                    
                    card = create_chart_card(
                        key="cat_participation_bars",
                        title=f"Category Participation (top {top_n})",
                        figure=fig,
                        question="How has category reach changed between two periods?",
                        takeaway=f"Population: {part['population_n'][0]:,} customers; change in pp = recent - previous",
                        evidence=ChartEvidence(
                            numerator="Category buyers in period",
                            denominator="Fixed population per period",
                            population_size=part["population_n"][0] if part.height > 0 else 0,
                            comparison_period=f"Recent 90d ending {st.session_state.get('chosen')} vs Previous 90d ending {origin}",
                            confidence_interval="Not applicable",
                            interval_type="Descriptive paired comparison",
                            eligibility_criteria="Customers present in filtered population at both cutoffs",
                        ),
                        how_to_read=(
                            "Paired horizontal bars: blue=recent 90d, orange=previous 90d. "
                            "Bars grouped by category. Hover shows buyers, spend, and change (pp). "
                            "Participation = category buyers / population (same denominator per period)."
                        ),
                        caveat=(
                            "Categories with zero buyers in a period show 0% participation, not missing. "
                            "Current period uses current filtered population; previous uses previous filtered population. "
                            "For fixed-population comparison, use this view; for changing populations, use 'Category Sankey'."
                        ),
                        next_step="Check 'Category Opportunity Map' for penetration×intensity analysis.",
                        alt_description=f"Paired horizontal bar chart showing category participation in recent vs previous 90-day windows.",
                    )
                    card.render()
                    
                    table(
                        display.select(
                            "label",
                            "population_n",
                            "recent_participation",
                            "previous_participation",
                            "participation_change_pp",
                            "buyer_change",
                            "recent_buyers",
                            "previous_buyers",
                            "recent_spend",
                            "previous_spend",
                        )
                    )
                    
                    st.caption(
                        f"Population (denominator): {part['population_n'][0] if part.height > 0 else 0:,} customers (fixed for both periods). "
                        f"Participation is share of this population. Change in percentage points = recent - previous. "
                        f"Categories with zero buyers in a period show 0% participation, not missing."
                    )
                    
    except Exception as e:
        st.error(f"Category participation analysis failed: {type(e).__name__}: {e}")


def render_product_association_network(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Product Association Network."""
    st.subheader("Product Association Network")
    st.caption(
        "Network visualization of product co-occurrence associations. "
        "Nodes = products, edges = associations exceeding minimum lift/confidence. "
        "Node size = degree (connections), color = max lift. "
        "Association ≠ causation; does not imply incremental cross-sell lift."
    )
    
    if not coverage:
        st.warning(
            "Coverage unconfirmed: segment-specific affinity requires confirmed follow-up."
        )
    else:
        try:
            col1, col2, col3 = st.columns(3)
            with col1:
                min_support = st.number_input(
                    "Min support", 0.0001, 0.1, 0.001, format="%.4f", key="net_support"
                )
            with col2:
                min_conf = st.number_input(
                    "Min confidence", 0.01, 1.0, 0.15, format="%.2f", key="net_conf"
                )
            with col3:
                min_lift = st.number_input(
                    "Min lift", 0.5, 5.0, 1.2, format="%.2f", key="net_lift"
                )
            
            top_k = st.number_input("Top-K pairs", 50, 2000, 200, step=50, key="net_topk")
            
            validated_path = root / "validated.parquet"
            if not validated_path.exists():
                st.info("No validated data available.")
            else:
                validated = pl.scan_parquet(validated_path)
                chosen = st.session_state.get("chosen", str(date.today()))
                
                with st.spinner("Computing product affinities..."):
                    config = ProductAffinityConfig(
                        min_support=min_support,
                        min_confidence=min_conf,
                        min_lift=min_lift,
                        top_k_items=500,
                        top_k_pairs=top_k,
                        max_items_per_basket=50,
                        category_level=False,
                    )
                    result = product_affinity_analysis(validated, rows, chosen, config, None)
                
                pairs = result["product_pairs"]
                if pairs.is_empty():
                    st.info("No product pairs meet the quality thresholds.")
                else:
                    # Build network
                    network = product_association_network(
                        pairs, min_lift=min_lift, min_confidence=min_conf, top_n=top_k
                    )
                    
                    nodes = network["nodes"]
                    edges = network["edges"]
                    
                    if nodes.is_empty():
                        st.info("No network to display after filtering.")
                    else:
                        # Create network layout using spring layout
                        try:
                            import networkx as nx
                            
                            G = nx.Graph()
                            for _, node in nodes.iter_rows(named=True):
                                G.add_node(
                                    node["product_id"],
                                    size=node["degree"],
                                    lift=node["max_lift"],
                                )
                            for _, edge in edges.iter_rows(named=True):
                                G.add_edge(
                                    edge["item_a"], edge["item_b"], weight=edge["lift"] or 1.0
                                )
                            
                            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
                            
                            # Build edge traces
                            edge_x, edge_y = [], []
                            for edge in G.edges():
                                x0, y0 = pos[edge[0]]
                                x1, y1 = pos[edge[1]]
                                edge_x.extend([x0, x1, None])
                                edge_y.extend([y0, y1, None])
                            
                            edge_trace = go.Scatter(
                                x=edge_x,
                                y=edge_y,
                                line=dict(width=0.5, color="#888"),
                                hoverinfo="none",
                                mode="lines",
                                showlegend=False,
                            )
                            
                            # Build node traces
                            node_x = [pos[node][0] for node in G.nodes()]
                            node_y = [pos[node][1] for node in G.nodes()]
                            node_size = [
                                max(10, min(50, G.nodes()[node].get("size", 1) * 5))
                                for node in G.nodes()
                            ]
                            node_color = [
                                G.nodes()[node].get("lift", 1.0) for node in G.nodes()
                            ]
                            node_text = list(G.nodes())
                            
                            node_trace = go.Scatter(
                                x=node_x,
                                y=node_y,
                                mode="markers+text",
                                text=node_text,
                                textposition="top center",
                                textfont=dict(size=8),
                                marker=dict(
                                    size=node_size,
                                    color=node_color,
                                    colorscale="Viridis",
                                    showscale=True,
                                    colorbar=dict(title="Max lift", thickness=15),
                                    line=dict(width=1, color="white"),
                                ),
                                hovertemplate="Product: %{text}<br>Connections: %{marker.size}<br>Max lift: %{marker.color:.2f}<extra></extra>",
                                showlegend=False,
                            )
                            
                            fig = go.Figure(data=[edge_trace, node_trace])
                            fig.update_layout(
                                showlegend=False,
                                hovermode="closest",
                                xaxis=dict(
                                    showgrid=False, zeroline=False, showticklabels=False
                                ),
                                yaxis=dict(
                                    showgrid=False, zeroline=False, showticklabels=False
                                ),
                                height=600,
                            )
                            
                            card = create_chart_card(
                                key="prod_network",
                                title="Product Association Network",
                                figure=fig,
                                question="Which products co-occur in baskets with strong associations?",
                                takeaway=f"Network of {nodes.height} products with {edges.height} associations (min lift={min_lift:.2f})",
                                evidence=ChartEvidence(
                                    numerator="Association edges meeting thresholds",
                                    denominator="All possible product pairs",
                                    population_size=None,
                                    comparison_period="Analysis cutoff window",
                                    confidence_interval="Not applicable",
                                    interval_type="Descriptive",
                                    eligibility_criteria=f"Support ≥ {min_support:.3%}, Confidence ≥ {min_conf:.0%}, Lift ≥ {min_lift:.2f}",
                                ),
                                how_to_read=(
                                    "Nodes: products. Edges: associations exceeding thresholds. "
                                    "Node size = degree (connections). Node color = max lift. "
                                    "Layout: force-directed (spring) for clustering. "
                                    "Hover shows product, degree, max lift."
                                ),
                                caveat=(
                                    "Association ≠ causation; does not imply incremental cross-sell lift. "
                                    "Network layout is algorithmic, not semantically meaningful. "
                                    "Rare pairs with high lift may dominate visual attention."
                                ),
                                next_step="Check 'Product Affinity' for tabular association rules.",
                                alt_description=f"Network graph showing product co-occurrence associations with node size by degree and color by max lift.",
                            )
                            card.render()
                            
                            st.caption(
                                f"Network from {edges.height:,} associations (min support={min_support:.3%}, "
                                f"min confidence={min_conf:.2f}, min lift={min_lift:.2f}). "
                                f"Node size = degree, color = max lift. Association ≠ causation."
                            )
                            
                        except ImportError:
                            st.warning("networkx not installed; showing edge table instead.")
                            table(edges.head(100))
                        
                        # Also show tabular view
                        st.write("**Top Product Pairs (Table)**")
                        table(
                            pairs.head(100).select(
                                "item_a",
                                "item_b",
                                "cooccurrence_count",
                                "support",
                                "confidence_a_to_b",
                                "confidence_b_to_a",
                                "lift",
                                "conviction_a_to_b",
                                "conviction_b_to_a",
                            )
                        )
                        
        except Exception as e:
            st.error(f"Product association network failed: {type(e).__name__}: {e}")


def render_trial_conversion_intervals(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Trial Conversion Intervals."""
    st.subheader("Trial-to-Repeat Conversion with Confidence Intervals")
    st.caption(
        "Dot-and-interval plot: each product's repeat conversion rate with 95% Wilson CI. "
        "Dot size/label = eligible trial count. "
        "Filters by minimum eligible trials. Identifies products with consistently high repeat behavior."
    )
    
    try:
        min_trials = st.number_input(
            "Minimum eligible trials per product", 1, 1000, 10, key="tci_min"
        )
        show_category = st.checkbox(
            "Show category-level aggregation", value=False, key="tci_cat"
        )
        
        prod_history_paths = list((root / "events" / st.session_state.get("chosen", "")).glob("*/product_history.parquet"))
        if not prod_history_paths:
            st.info("No product history available for this cutoff.")
        else:
            prod_history = pl.scan_parquet(prod_history_paths)
            cutoff_date = date.fromisoformat(st.session_state.get("chosen", str(date.today())))
            
            # Category lookup if needed
            category_lookup = None
            if show_category:
                validated_path = root / "validated.parquet"
                if validated_path.exists():
                    validated = pl.scan_parquet(validated_path)
                    category_lookup = collect(
                        validated.select("product_id", "department", "category")
                        .unique()
                        .with_columns(
                            pl.concat_str(
                                [pl.col("department"), pl.lit(" / "), pl.col("category")]
                            ).alias("category_label")
                        )
                    )
            
            from retail_analytics.visualization.visual_insights_core import trial_conversion_interval_plot
            
            trial_data = trial_conversion_interval_plot(
                prod_history, rows, cutoff_date, min_trials, category_lookup
            )
            
            if trial_data.is_empty():
                st.info(f"No products with ≥ {min_trials} eligible trials.")
            else:
                # Dot-and-interval plot
                fig = go.Figure()
                
                products = trial_data.sort("trial_conversion", descending=True).head(50)
                
                for _, row in products.iter_rows(named=True):
                    prod_id = row["product_id"]
                    rate = row["trial_conversion"]
                    lo = row["lower_ci"]
                    hi = row["upper_ci"]
                    n = row["eligible_trials"]
                    buyers = row["repeated_trials"]
                    customers = row["trialling_customers"]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[rate],
                            y=[prod_id],
                            mode="markers",
                            marker=dict(
                                size=max(8, min(30, n ** 0.5 * 1.5)),
                                color="#0072B2",
                                opacity=0.8,
                            ),
                            name=prod_id,
                            error_x=dict(
                                type="data",
                                symmetric=False,
                                array=[hi - rate] if hi else None,
                                arrayminus=[rate - lo] if lo else None,
                                visible=True,
                                color="gray",
                                thickness=2,
                            )
                            if lo is not None
                            else None,
                            customdata=[[n, buyers, customers]],
                            hovertemplate=f"Product: {prod_id}<br>Conversion: {rate:.1%}<br>Trials: {n}<br>Repeats: {buyers}<br>Customers: {customers}<br>CI: [{lo:.1% if lo else 'N/A'}, {hi:.1% if hi else 'N/A'}]<extra></extra>",
                            showlegend=False,
                        )
                    )
                
                fig.update_xaxes(
                    range=[0, 1],
                    tickformat=".0%",
                    title="Trial conversion rate (95% Wilson CI)",
                )
                fig.update_yaxes(autorange="reversed", title="Product ID")
                fig.update_layout(height=max(400, products.height * 20))
                
                card = create_chart_card(
                    key="trial_conv_intervals",
                    title=f"Trial-to-Repeat Conversion (Min {min_trials} Trials)",
                    figure=fig,
                    question="Among products first observed within an eligible trial window, what share has a qualifying repeat?",
                    takeaway=f"{products.height} products with ≥{min_trials} trials; Wilson 95% CI shown for ≥20 trials",
                    evidence=ChartEvidence(
                        numerator="Repeated trials",
                        denominator="Eligible trials",
                        population_size=None,
                        comparison_period="90-day repeat window after trial",
                        confidence_interval="Wilson 95% (products with ≥20 eligible trials)",
                        interval_type="Binomial proportion CI",
                        eligibility_criteria=f"First purchase in [C-364, C-90] with 90-day follow-up fully observed; min {min_trials} trials",
                    ),
                    how_to_read=(
                        "Dots: pooled trial conversion rate per product (total repeats / total eligible trials). "
                        "Horizontal lines: Wilson 95% CI (products with ≥20 trials). "
                        "Dot size: number of eligible trials. "
                        "Products without CI have high uncertainty."
                    ),
                    caveat=(
                        "Suppresses ineligible or immature trial windows. "
                        "Product availability and source capture affect repeat rates. "
                        "Avoid equating repeat with satisfaction. "
                        "Pooled conversion = Σ repeats / Σ eligible trials across customers."
                    ),
                    next_step="Check 'Product Association Network' for sequential purchase patterns.",
                    alt_description=f"Dot-and-interval plot showing trial-to-repeat conversion rate by product with Wilson 95% confidence intervals.",
                )
                card.render()
                
                st.caption(
                    "Wilson 95% CI shown for products with ≥20 eligible trials. "
                    "Products with fewer trials have no CI (high uncertainty). "
                    "Pooled conversion = total repeats / total eligible trials across customers."
                )
                
                display_cols = [
                    "product_id",
                    "eligible_trials",
                    "repeated_trials",
                    "trialling_customers",
                    "trial_conversion",
                    "lower_ci",
                    "upper_ci",
                ]
                if "category_label" in trial_data.columns:
                    display_cols.insert(1, "category_label")
                table(trial_data.select(display_cols))
                
    except Exception as e:
        st.error(f"Trial conversion intervals failed: {type(e).__name__}: {e}")


def render_product_affinity(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Product Affinity & Market-Basket Analysis."""
    st.subheader("Product Affinity & Market-Basket Analysis")
    st.caption(
        "Lightweight association mining using sparse basket-level co-occurrence. "
        "No dense customer×product matrices. "
        "Support = fraction of baskets containing itemset. "
        "Confidence = P(B|A). Lift = P(A∪B)/(P(A)P(B)). Conviction = (1-P(B))/(1-P(B|A)). "
        "Minimum support and top-K limits keep computation tractable."
    )
    
    if not coverage:
        st.warning(
            "Coverage unconfirmed: segment-specific affinity requires confirmed follow-up. "
            "Enable coverage confirmation in sidebar to analyze by segment."
        )
    else:
        # Configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            min_support = st.number_input(
                "Minimum support",
                0.0001,
                0.1,
                0.001,
                format="%.4f",
                help="Fraction of baskets containing itemset. Lower = more pairs, more compute.",
            )
        with col2:
            min_confidence = st.number_input(
                "Minimum confidence",
                0.01,
                1.0,
                0.10,
                format="%.2f",
                help="P(B|A) threshold for association rules.",
            )
        with col3:
            min_lift = st.number_input(
                "Minimum lift",
                0.5,
                5.0,
                1.0,
                format="%.2f",
                help="Lift = P(A∪B)/(P(A)P(B)). >1 means positive association.",
            )
        
        col4, col5, col6 = st.columns(3)
        with col4:
            top_k_items = st.number_input(
                "Top-K items", 50, 2000, 500, step=50, help="Max frequent items to consider."
            )
        with col5:
            top_k_pairs = st.number_input(
                "Top-K pairs", 100, 10000, 2000, step=100, help="Max pairs to emit by support."
            )
        with col6:
            max_items_per_basket = st.number_input(
                "Max items/basket", 10, 100, 50, help="Cap basket size to limit pair explosion."
            )
        
        category_level = st.checkbox("Include category-level analysis", value=True)
        
        # Segment filter
        segment_options = ["All customers"] + sorted(
            set(
                str(v)
                for dim in ["behavior", "trajectory", "value_tier", "lifecycle", "evidence"]
                for v in collect(rows.select(dim).unique().sort(dim))[dim].to_list()
            )
        )
        segment_filter = st.selectbox("Segment filter (optional)", segment_options)
        segment = None if segment_filter == "All customers" else segment_filter
        
        # Load validated data for basket-level analysis
        validated_path = root / "validated.parquet"
        if not validated_path.exists():
            st.info("No validated data available for this analysis.")
        else:
            try:
                validated = pl.scan_parquet(validated_path)
                chosen = st.session_state.get("chosen", str(date.today()))
                
                with st.spinner("Computing product affinities..."):
                    config = ProductAffinityConfig(
                        min_support=min_support,
                        min_confidence=min_confidence,
                        min_lift=min_lift,
                        top_k_items=top_k_items,
                        top_k_pairs=top_k_pairs,
                        max_items_per_basket=max_items_per_basket,
                        category_level=category_level,
                    )
                    result = product_affinity_analysis(validated, rows, chosen, config, segment)
                
                # Frequent products
                st.write("**Frequent Products**")
                pf = result["product_frequent"]
                if pf.is_empty():
                    st.info("No products meet the minimum support threshold.")
                else:
                    display_pf = pf.head(50)
                    bars(
                        display_pf,
                        "product_id",
                        "basket_count",
                        "Top 50 products by basket count",
                        "freq_products",
                        color_map,
                    )
                    st.caption(
                        f"Total frequent products: {pf.height:,} (min support = {min_support:.3%} of {pf['total_baskets'][0]:,} baskets)"
                    )
                    table(display_pf.select("product_id", "basket_count", "support"))
                
                # Product association rules
                st.write("**Product Association Rules**")
                pp = result["product_pairs"]
                if pp.is_empty():
                    st.info(
                        "No product pairs meet the minimum support/confidence/lift thresholds."
                    )
                else:
                    display_pp = pp.head(50)
                    bars(
                        display_pp,
                        "item_a",
                        "support",
                        "Top 50 product pairs by support",
                        "prod_pairs_support",
                        color_map,
                    )
                    table(
                        display_pp.select(
                            "item_a",
                            "item_b",
                            "cooccurrence_count",
                            "support",
                            "confidence_a_to_b",
                            "confidence_b_to_a",
                            "lift",
                            "conviction_a_to_b",
                            "conviction_b_to_a",
                        )
                    )
                
                # Cross-sell recommendations
                st.write("**Cross-Sell Recommendations**")
                recs = cross_sell_recommendations(
                    result["product_pairs"],
                    result.get("category_pairs"),
                    top_n=10,
                    min_lift=min_lift,
                    min_confidence=min_confidence,
                )
                prod_recs = recs["product_recommendations"]
                if prod_recs.is_empty():
                    st.info("No recommendations meet the quality thresholds.")
                else:
                    table(
                        prod_recs.head(100).select(
                            "source_item",
                            "target_item",
                            "support",
                            "confidence",
                            "lift",
                            "conviction",
                            "direction",
                        )
                    )
                
                # Category-level analysis
                if category_level:
                    st.write("**Frequent Categories**")
                    cf = result.get("category_frequent", pl.DataFrame())
                    if cf.is_empty():
                        st.info("No categories meet the minimum support threshold.")
                    else:
                        display_cf = cf.head(50)
                        bars(
                            display_cf,
                            "category_key",
                            "basket_count",
                            "Top 50 categories by basket count",
                            "freq_categories",
                            color_map,
                        )
                        st.caption(f"Total frequent categories: {cf.height:,}")
                        table(display_cf.select("category_key", "basket_count", "support"))
                    
                    st.write("**Category Association Rules**")
                    cp = result.get("category_pairs", pl.DataFrame())
                    if cp.is_empty():
                        st.info(
                            "No category pairs meet the minimum support/confidence/lift thresholds."
                        )
                    else:
                        display_cp = cp.head(50)
                        bars(
                            display_cp,
                            "item_a",
                            "support",
                            "Top 50 category pairs by support",
                            "cat_pairs_support",
                            color_map,
                        )
                        table(
                            display_cp.select(
                                "item_a",
                                "item_b",
                                "cooccurrence_count",
                                "support",
                                "confidence_a_to_b",
                                "confidence_b_to_a",
                                "lift",
                                "conviction_a_to_b",
                                "conviction_b_to_a",
                            )
                        )
                    
                    cat_recs = recs["category_recommendations"]
                    if not cat_recs.is_empty():
                        st.write("**Category Cross-Sell Recommendations**")
                        table(
                            cat_recs.head(100).select(
                                "source_item",
                                "target_item",
                                "support",
                                "confidence",
                                "lift",
                                "conviction",
                                "direction",
                            )
                        )
                
                # New-to-repeat pathways
                st.write("**New-to-Repeat Pathways**")
                st.caption(
                    "Products/categories customers tend to buy AFTER their first purchase "
                    "of a given product/category. Shows sequential purchase patterns."
                )
                pathways = new_to_repeat_pathways(validated, rows, st.session_state.get("chosen", str(date.today())), min_trials=5)
                prod_paths = pathways["product_transitions"]
                cat_paths = pathways["category_transitions"]
                
                if not prod_paths.is_empty():
                    st.write("Product → next product transitions")
                    table(
                        prod_paths.head(50).select(
                            "prev_product", "product_id", "transition_count"
                        )
                    )
                else:
                    st.info("No product transitions meet minimum threshold.")
                
                if not cat_paths.is_empty():
                    st.write("Category → next category transitions")
                    table(
                        cat_paths.head(50).select(
                            "prev_category", "category_key", "transition_count"
                        )
                    )
                else:
                    st.info("No category transitions meet minimum threshold.")
                
                # Basket composition by segment
                if segment_filter == "All customers":
                    st.write("**Basket Composition by Segment**")
                    st.caption(
                        "Spend share of products/categories within each segment. "
                        "Shows what different customer segments actually buy."
                    )
                    comp = basket_composition_by_segment(
                        rows,
                        validated,
                        st.session_state.get("chosen", str(date.today())),
                        dimensions=["behavior", "value_tier", "lifecycle"],
                    )
                    for dim, data in comp.items():
                        st.write(f"By {dim}")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("Product composition (top 20)")
                            table(
                                data["product_composition"]
                                .head(20)
                                .select(dim, "product_id", "spend_share", "buyers")
                            )
                        with col_b:
                            st.write("Category composition (top 20)")
                            table(
                                data["category_composition"]
                                .head(20)
                                .select(dim, "category_key", "spend_share", "buyers")
                            )
                
            except Exception as e:
                st.error(f"Product affinity analysis failed: {type(e).__name__}: {e}")
                st.caption(
                    "Check that validated.parquet exists and has required columns. Large datasets may need adjusted thresholds."
                )


__all__ = [
    "render_category_penetration_growth",
    "render_category_opportunity_map",
    "render_category_participation",
    "render_product_association_network",
    "render_trial_conversion_intervals",
    "render_product_affinity",
]