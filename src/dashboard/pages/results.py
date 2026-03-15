"""Saved Results page."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.storage.db import list_runs, get_run_data, delete_run

st.title("Saved Results")

runs = list_runs()

if not runs:
    st.info("No saved backtest results yet. Run a backtest and save it first.")
    st.stop()

# --- Results table ---
runs_df = pd.DataFrame(runs)
display_cols = [
    "id", "strategy_name", "timeframe", "total_return", "sharpe_ratio",
    "win_rate", "profit_factor", "max_drawdown_pct", "total_trades", "created_at",
]
existing_cols = [c for c in display_cols if c in runs_df.columns]
st.dataframe(
    runs_df[existing_cols].rename(columns={
        "id": "ID", "strategy_name": "Strategy", "timeframe": "TF",
        "total_return": "Return %", "sharpe_ratio": "Sharpe",
        "win_rate": "Win Rate %", "profit_factor": "PF",
        "max_drawdown_pct": "Max DD %", "total_trades": "Trades",
        "created_at": "Date",
    }),
    use_container_width=True,
    hide_index=True,
)

# --- Detail view ---
st.markdown("---")
run_ids = [r["id"] for r in runs]
selected_id = st.selectbox("View Run Details", run_ids, format_func=lambda x: f"Run #{x}")

if selected_id:
    run = next(r for r in runs if r["id"] == selected_id)
    data = get_run_data(selected_id)

    if data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy", run["strategy_name"])
        col2.metric("Timeframe", run["timeframe"])
        col3.metric("Return", f"{run['total_return']:.2f}%")
        col4.metric("Sharpe", f"{run['sharpe_ratio']:.2f}")

        params = json.loads(run.get("params_json", "{}"))
        if params:
            st.caption(f"Parameters: {params}")

        # Equity curve
        eq_data = json.loads(data["equity_curve_json"])
        if eq_data:
            eq_df = pd.DataFrame(eq_data)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_df["datetime"], y=eq_df["value"],
                mode="lines", name="Equity",
            ))
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Value ($)")
            st.plotly_chart(fig, use_container_width=True)

        # Drawdown
        dd_data = json.loads(data["drawdown_json"])
        if dd_data:
            dd_df = pd.DataFrame(dd_data)
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd_df["datetime"], y=dd_df["drawdown"],
                mode="lines", fill="tozeroy", name="Drawdown",
                line=dict(color="red"),
            ))
            fig_dd.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="DD (%)")
            st.plotly_chart(fig_dd, use_container_width=True)

        # Trades
        trades_data = json.loads(data["trades_json"])
        if trades_data:
            st.subheader("Trades")
            st.dataframe(pd.DataFrame(trades_data), use_container_width=True, height=300)

        # Delete
        if st.button(f"Delete Run #{selected_id}", type="secondary"):
            delete_run(selected_id)
            st.rerun()
