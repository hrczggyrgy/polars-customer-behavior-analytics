"""Model UI integration views for probabilistic customer behavior models."""

from datetime import date, timedelta
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from retail_analytics.modeling.models import (
    BGNBDParams,
    GammaGammaParams,
    bg_nbd_expected_transactions,
    fit_bg_nbd,
    fit_gamma_gamma,
    gamma_gamma_expected_avg_value,
    predict_clv,
)
from retail_analytics.visualization.chart_components import collect


def render_model_evaluation(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """Model Evaluation & Holdout Validation."""
    st.subheader("📊 Model Evaluation: Temporal Holdout Validation")
    st.caption(
        "**Question:** How well do probabilistic models predict future behavior on held-out data? "
        "Uses temporal train/validation/test splits with no data leakage."
    )

    if not coverage:
        st.warning(
            "⚠ **Coverage not confirmed** — Model evaluation requires confirmed follow-up data. "
            "Enable coverage confirmation to evaluate predictive models."
        )
        return

    # Configuration
    st.write("**Evaluation Configuration**")
    col1, col2, col3 = st.columns(3)
    with col1:
        train_months = st.slider("Training window (months)", 3, 24, 12)
    with col2:
        val_months = st.slider("Validation window (months)", 1, 6, 3)
    with col3:
        test_months = st.slider("Test window (months)", 1, 6, 3)

    # Get date range
    chosen = st.session_state.get("chosen", str(date.today()))
    chosen_date = date.fromisoformat(chosen)

    # Calculate split dates
    test_end = chosen_date
    test_start = test_end - timedelta(days=test_months * 30)
    val_end = test_start - timedelta(days=1)
    val_start = val_end - timedelta(days=val_months * 30)
    train_end = val_start - timedelta(days=1)
    train_start = train_end - timedelta(days=train_months * 30)

    st.write(f"**Train:** {train_start} to {train_end} ({train_months} months)")
    st.write(f"**Validation:** {val_start} to {val_end} ({val_months} months)")
    st.write(f"**Test:** {test_start} to {test_end} ({test_months} months)")

    if st.button("Run Model Evaluation", type="primary"):
        with st.spinner("Fitting models and evaluating on holdout..."):
            run_model_evaluation(root, rows, coverage, color_map, train_start, train_end, val_start, val_end, test_start, test_end)


def run_model_evaluation(
    root: Path,
    rows: pl.LazyFrame,
    coverage: bool,
    color_map: dict,
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    test_start: date,
    test_end: date,
):
    """Run model evaluation with temporal splits."""
    from retail_analytics.visualization.visual_insights_core import collect

    try:
        # Get training data
        train_rows = rows.filter(
            (pl.col("first_purchase") <= pl.lit(train_end)) &
            (pl.col("recency") <= (train_end - train_start).days)
        )
        train_data = collect(train_rows)

        if train_data.is_empty():
            st.error("No training data available for the selected period.")
            return

        # Prepare RFM data for training
        freq = train_data["purchases_90d"].to_numpy()
        rec = train_data["recency"].to_numpy()
        T = train_data["tenure"].to_numpy()
        mon = train_data["spend_90d"].to_numpy()

        # Fit BG/NBD
        with st.spinner("Fitting BG/NBD..."):
            bg_params, bg_converged, bg_ll = fit_bg_nbd(freq, rec, T)

        if not bg_converged:
            st.warning("BG/NBD did not converge. Results may be unreliable.")

        st.success(f"BG/NBD converged: {bg_converged} (LL: {bg_ll:.2f})")
        st.write(f"Params: r={bg_params.r:.4f}, α={bg_params.alpha:.4f}, a={bg_params.a:.4f}, b={bg_params.b:.4f}")

        # Fit Gamma-Gamma
        with st.spinner("Fitting Gamma-Gamma..."):
            gg_params, gg_converged, gg_ll = fit_gamma_gamma(freq, mon)

        if not gg_converged:
            st.warning("Gamma-Gamma did not converge. Results may be unreliable.")

        st.success(f"Gamma-Gamma converged: {gg_converged} (LL: {gg_ll:.2f})")
        st.write(f"Params: p={gg_params.p:.4f}, q={gg_params.q:.4f}, γ={gg_params.gamma:.4f}")

        # Prepare test data
        test_rows = rows.filter(
            (pl.col("first_purchase") <= pl.lit(test_end)) &
            (pl.col("recency") <= (test_end - test_start).days)
        )
        test_data = collect(test_rows)

        if test_data.is_empty():
            st.warning("No test data available for evaluation.")
            return

        # Generate predictions for test period
        test_freq = test_data["purchases_90d"].to_numpy()
        test_rec = test_data["recency"].to_numpy()
        test_T = test_data["tenure"].to_numpy()
        test_mon = test_data["spend_90d"].to_numpy()
        test_ids = test_data["customer_id"].to_numpy()

        horizon_days = (test_end - test_start).days

        # Predict transactions
        pred_trans = bg_nbd_expected_transactions(
            BGNBDParams(**bg_params.__dict__),
            test_freq, test_rec, test_T, horizon_days
        )

        # Predict monetary
        pred_avg_val = gamma_gamma_expected_avg_value(
            GammaGammaParams(**gg_params.__dict__),
            test_freq, test_mon
        )
        pred_monetary = pred_trans * pred_avg_val

        # Actual outcomes (need to get from future columns if available)
        actual_trans = test_data["future_purchases_90d"].to_numpy() if "future_purchases_90d" in test_data.columns else None
        actual_mon = test_data["future_spend_90d"].to_numpy() if "future_spend_90d" in test_data.columns else None

        if actual_trans is not None:
            # Evaluation metrics
            from retail_analytics.modeling.evaluation import compute_evaluation_metrics

            trans_metrics = compute_evaluation_metrics(pred_trans, actual_trans)
            mon_metrics = compute_evaluation_metrics(pred_monetary, actual_mon)

            st.write("### Transaction Prediction Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{trans_metrics.mae:.3f}")
            col2.metric("RMSE", f"{trans_metrics.rmse:.3f}")
            col3.metric("MAPE", f"{trans_metrics.mape:.1f}%")
            col4.metric("Calibration R²", f"{trans_metrics.calibration_r2:.3f}")

            st.write("### Monetary Prediction Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mon_metrics.mae:.2f}")
            col2.metric("RMSE", f"{mon_metrics.rmse:.2f}")
            col3.metric("MAPE", f"{mon_metrics.mape:.1f}%")
            col4.metric("Calibration R²", f"{mon_metrics.calibration_r2:.3f}")

            # Calibration plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=actual_trans,
                y=pred_trans,
                mode="markers",
                marker=dict(size=4, opacity=0.5, color="#0072B2"),
                name="Predictions",
                hovertemplate="Actual: %{x}<br>Predicted: %{y}<extra></extra>"
            ))
            # Perfect calibration line
            max_val = max(actual_trans.max(), pred_trans.max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                name="Perfect calibration"
            ))
            fig.update_layout(
                title="Calibration: Actual vs Predicted Transactions",
                xaxis_title="Actual Transactions",
                yaxis_title="Predicted Transactions",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Segment-level metrics
            if "behavior" in test_data.columns:
                st.write("### Metrics by Behavioral Segment")
                test_df = test_data.with_columns([
                    pl.Series("pred_trans", pred_trans),
                    pl.Series("pred_monetary", pred_monetary),
                ])

                segment_metrics = test_df.group_by("behavior").agg([
                    pl.col("pred_trans").mean().alias("avg_pred_trans"),
                    pl.col("future_purchases_90d").mean().alias("avg_actual_trans"),
                    pl.col("pred_monetary").mean().alias("avg_pred_mon"),
                    pl.col("future_spend_90d").mean().alias("avg_actual_mon"),
                    pl.len().alias("n"),
                ])
                st.dataframe(segment_metrics, hide_index=True)

        else:
            st.info("Future outcome columns not available in test data. Cannot compute evaluation metrics.")
            st.write("**Generated Predictions (sample):**")
            pred_df = pl.DataFrame({
                "customer_id": test_ids[:20],
                "predicted_transactions": pred_trans[:20],
                "predicted_monetary": pred_monetary[:20],
            })
            st.dataframe(pred_df, hide_index=True)

    except Exception as e:
        st.error(f"Model evaluation failed: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_clv_predictions(root: Path, rows: pl.LazyFrame, coverage: bool, color_map: dict):
    """CLV Predictions View."""
    st.subheader("💰 Customer Lifetime Value Predictions")
    st.caption(
        "**Question:** What is the expected future value of each customer? "
        "Combines BG/NBD (transaction count) + Gamma-Gamma (monetary value). "
        "Only available with coverage confirmation."
    )

    if not coverage:
        st.warning(
            "⚠ **Coverage not confirmed** — CLV predictions require confirmed follow-up data. "
            "Enable coverage confirmation to generate predictions."
        )
        return

    # Fit models on full data
    from retail_analytics.modeling.models import (
        BGNBDParams,
        GammaGammaParams,
        fit_bg_nbd,
        fit_gamma_gamma,
    )

    data = collect(rows)
    if data.is_empty():
        st.info("No customers in filtered population.")
        return

    freq = data["purchases_90d"].to_numpy()
    rec = data["recency"].to_numpy()
    T = data["tenure"].to_numpy()
    mon = data["spend_90d"].to_numpy()

    col1, col2 = st.columns(2)
    with col1:
        horizon = st.selectbox("Prediction horizon (days)", [30, 60, 90, 180, 365], index=2)
    with col2:
        top_n = st.number_input("Show top N customers", 10, 1000, 50, step=10)

    if st.button("Generate CLV Predictions", type="primary"):
        with st.spinner("Fitting models and generating predictions..."):
            bg_params, bg_conv, bg_ll = fit_bg_nbd(
                data["purchases_90d"].to_numpy(),
                data["recency"].to_numpy(),
                data["tenure"].to_numpy()
            )
            gg_params, gg_conv, gg_ll = fit_gamma_gamma(
                data["purchases_90d"].to_numpy(),
                data["spend_90d"].to_numpy()
            )

            if not bg_conv or not gg_conv:
                st.warning("One or both models did not converge. Predictions may be unreliable.")

            st.success(f"BG/NBD: {bg_conv} (LL={bg_ll:.1f}) | Gamma-Gamma: {gg_conv} (LL={gg_ll:.1f})")

            predictions = predict_clv(
                BGNBDParams(**bg_params.__dict__),
                GammaGammaParams(**gg_params.__dict__),
                data["purchases_90d"].to_numpy(),
                data["recency"].to_numpy(),
                data["tenure"].to_numpy(),
                data["spend_90d"].to_numpy(),
                horizon_days=horizon,
                customer_ids=data["customer_id"].to_numpy(),
                prediction_date=date.fromisoformat(st.session_state.get("chosen", str(date.today()))),
            )

            # Sort by expected monetary
            pred_df = pl.DataFrame([p.__dict__ for p in predictions])
            pred_df = pred_df.sort("expected_monetary", descending=True).head(top_n)

            # Display
            st.write(f"### Top {top_n} Customers by Expected {horizon}-Day Value")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pred_df["customer_id"].to_list(),
                y=pred_df["expected_monetary"].to_list(),
                marker_color="#0072B2",
                hovertemplate="Customer: %{x}<br>Expected CLV: %{y:,.2f}<br>Expected Trans: %{customdata[0]:.2f}<br>P(alive): %{customdata[1]:.2%}<extra></extra>",
                customdata=list(zip(pred_df["expected_transactions"], pred_df["p_alive"]))
            ))
            fig.update_layout(
                title=f"Top {top_n} Customers by Expected {horizon}-Day CLV",
                xaxis_title="Customer ID",
                yaxis_title="Expected Monetary Value",
                template="plotly_white",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(pred_df, hide_index=True, use_container_width=True)

            st.caption(
                f"Models: BG/NBD (converged={bg_conv}, LL={bg_ll:.1f}) + Gamma-Gamma (converged={gg_conv}, LL={gg_ll:.1f}). "
                f"Horizon: {horizon} days. Expected transactions × expected avg value = expected monetary. "
                f"CLV = expected future revenue, not profit. Requires margin/cost data for net CLV."
            )


# Export
__all__ = [
    "render_model_evaluation",
    "render_clv_predictions",
]
