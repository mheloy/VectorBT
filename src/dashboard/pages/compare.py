"""Compare multiple saved backtest runs."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.storage.db import list_runs, get_run_data

st.title("Compare Runs")

runs = list_runs()

if len(runs) < 2:
    st.info("Save at least 2 backtest runs to compare them.")
    st.stop()

# Select runs to compare
run_options = {r["id"]: f"#{r['id']} {r['strategy_name']} ({r['timeframe']}) — {r['total_return']:.1f}%" for r in runs}
selected_ids = st.multiselect(
    "Select runs to compare",
    options=list(run_options.keys()),
    format_func=lambda x: run_options[x],
    default=list(run_options.keys())[:2],
)

if len(selected_ids) < 2:
    st.warning("Select at least 2 runs.")
    st.stop()

selected_runs = [r for r in runs if r["id"] in selected_ids]

# Comparison table
st.subheader("Metrics Comparison")
compare_data = []
for r in selected_runs:
    compare_data.append({
        "Run": f"#{r['id']}",
        "Strategy": r["strategy_name"],
        "TF": r["timeframe"],
        "Return %": r["total_return"],
        "Sharpe": r["sharpe_ratio"],
        "Sortino": r["sortino_ratio"],
        "Win Rate %": r["win_rate"],
        "PF": r["profit_factor"],
        "Max DD %": r["max_drawdown_pct"],
        "Trades": r["total_trades"],
    })
st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

# Overlay equity curves
st.subheader("Equity Curves")
fig = go.Figure()
for r in selected_runs:
    data = get_run_data(r["id"])
    if data:
        eq_data = json.loads(data["equity_curve_json"])
        if eq_data:
            eq_df = pd.DataFrame(eq_data)
            label = f"#{r['id']} {r['strategy_name']} ({r['timeframe']})"
            fig.add_trace(go.Scatter(
                x=eq_df["datetime"], y=eq_df["value"],
                mode="lines", name=label,
            ))

fig.update_layout(height=450, yaxis_title="Equity ($)", margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# Overlay drawdowns
st.subheader("Drawdowns")
fig_dd = go.Figure()
for r in selected_runs:
    data = get_run_data(r["id"])
    if data:
        dd_data = json.loads(data["drawdown_json"])
        if dd_data:
            dd_df = pd.DataFrame(dd_data)
            label = f"#{r['id']} {r['strategy_name']}"
            fig_dd.add_trace(go.Scatter(
                x=dd_df["datetime"], y=dd_df["drawdown"],
                mode="lines", name=label,
            ))

fig_dd.update_layout(height=350, yaxis_title="Drawdown (%)", margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_dd, use_container_width=True)
