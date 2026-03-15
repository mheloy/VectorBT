"""Run Backtest page."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_m5, resample, TIMEFRAMES
from src.strategies.base import BaseStrategy
from src.strategies.registry import get_all_strategies
from src.engine.runner import run_backtest
from src.engine.metrics import extract_metrics, get_equity_curve, get_drawdown_series, get_trades_df
from src.engine.kelly import kelly_from_metrics
from src.storage.db import save_run
from src.storage.models import BacktestRun, BacktestData

st.title("Run Backtest")

# --- Load data (cached) ---
@st.cache_data(show_spinner="Loading XAUUSD data...")
def cached_load():
    return load_m5()

raw_data = cached_load()

# --- Sidebar ---
strategies = get_all_strategies()
strategy_name = st.sidebar.selectbox("Strategy", list(strategies.keys()))
strategy = strategies[strategy_name]

timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1H"))

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

# Build parameter inputs dynamically
param_values = {}
for p in strategy.parameters():
    if p.choices:
        param_values[p.name] = st.sidebar.selectbox(
            p.description or p.name, p.choices,
            index=p.choices.index(p.default),
            key=f"param_{p.name}",
        )
    elif p.min_val is not None and p.max_val is not None:
        param_values[p.name] = st.sidebar.slider(
            p.description or p.name,
            min_value=p.min_val,
            max_value=p.max_val,
            value=p.default,
            step=p.step or 1,
            key=f"param_{p.name}",
        )
    else:
        param_values[p.name] = st.sidebar.number_input(
            p.description or p.name,
            value=p.default,
            key=f"param_{p.name}",
        )

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Management")
init_cash = st.sidebar.number_input("Initial Cash ($)", value=10000.0, step=1000.0)
fees = st.sidebar.number_input("Fees (fraction)", value=0.0001, step=0.0001, format="%.4f")

# Check if strategy provides dynamic ATR-based stops
_has_dynamic_stops = type(strategy).compute_stops is not BaseStrategy.compute_stops

if _has_dynamic_stops:
    st.sidebar.info("SL/TP: ATR-based (see strategy params above)")
    use_override = st.sidebar.checkbox("Override with fixed %")
    if use_override:
        sl_stop = st.sidebar.number_input("SL (%)", value=2.0, step=0.5) / 100
        tp_stop = st.sidebar.number_input("TP (%)", value=4.0, step=0.5) / 100
    else:
        sl_stop = None
        tp_stop = None
else:
    use_sl = st.sidebar.checkbox("Stop Loss")
    sl_stop = st.sidebar.number_input("SL (%)", value=2.0, step=0.5, disabled=not use_sl) / 100 if use_sl else None
    use_tp = st.sidebar.checkbox("Take Profit")
    tp_stop = st.sidebar.number_input("TP (%)", value=4.0, step=0.5, disabled=not use_tp) / 100 if use_tp else None

# --- Run ---
if st.sidebar.button("Run Backtest", type="primary", use_container_width=True):
    with st.spinner("Running backtest..."):
        df = resample(raw_data, timeframe)

        freq_map = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h", "D": "1D"}
        portfolio = run_backtest(
            strategy=strategy,
            df=df,
            params=param_values,
            init_cash=init_cash,
            fees=fees,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            freq=freq_map.get(timeframe),
        )

    st.session_state["portfolio"] = portfolio
    st.session_state["backtest_df"] = df
    st.session_state["backtest_params"] = param_values
    st.session_state["backtest_strategy"] = strategy_name
    st.session_state["backtest_tf"] = timeframe
    st.session_state["backtest_config"] = {
        "init_cash": init_cash, "fees": fees, "sl_stop": sl_stop, "tp_stop": tp_stop,
    }

# --- Display results ---
if "portfolio" in st.session_state:
    portfolio = st.session_state["portfolio"]
    df = st.session_state["backtest_df"]
    metrics = extract_metrics(portfolio)
    kelly = kelly_from_metrics(metrics)

    # --- Metrics cards ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Return", f"{metrics['total_return']:.2f}%")
    col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col3.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    col4.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    col5.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    col6.metric("Total Trades", f"{metrics['total_trades']}")

    col7, col8, col9, col10 = st.columns(4)
    col7.metric("Sortino", f"{metrics['sortino_ratio']:.2f}")
    col8.metric("Calmar", f"{metrics['calmar_ratio']:.2f}")
    col9.metric("Kelly (Half)", f"{kelly.half_kelly_pct:.1f}%")
    col10.metric("Recommended Risk", f"{kelly.recommended_risk_pct:.1f}%")

    # --- Equity curve ---
    st.subheader("Equity Curve")
    equity = get_equity_curve(portfolio)
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Portfolio Value"))
    fig_eq.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Value ($)")
    st.plotly_chart(fig_eq, use_container_width=True)

    # --- Drawdown ---
    st.subheader("Drawdown")
    dd = get_drawdown_series(portfolio)
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", fill="tozeroy",
        name="Drawdown", line=dict(color="red"),
    ))
    fig_dd.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    # --- Trade table ---
    st.subheader("Trades")
    trades_df = get_trades_df(portfolio)
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True, height=400)
    else:
        st.info("No trades were generated.")

    # --- Save button ---
    st.markdown("---")
    if st.button("Save Results", use_container_width=True):
        config = st.session_state["backtest_config"]
        run = BacktestRun(
            strategy_name=st.session_state["backtest_strategy"],
            timeframe=st.session_state["backtest_tf"],
            params_json=json.dumps(st.session_state["backtest_params"]),
            date_range_start=str(df.index[0]),
            date_range_end=str(df.index[-1]),
            total_return=metrics["total_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            calmar_ratio=metrics["calmar_ratio"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            max_drawdown_pct=metrics["max_drawdown_pct"],
            total_trades=metrics["total_trades"],
            init_cash=config["init_cash"],
            fees=config["fees"],
            sl_stop=config["sl_stop"],
            tp_stop=config["tp_stop"],
        )

        # Serialize data
        eq_data = equity.reset_index()
        eq_data.columns = ["datetime", "value"]
        eq_data["datetime"] = eq_data["datetime"].astype(str)

        dd_data = dd.reset_index()
        dd_data.columns = ["datetime", "drawdown"]
        dd_data["datetime"] = dd_data["datetime"].astype(str)

        data = BacktestData(
            equity_curve_json=eq_data.to_json(orient="records"),
            trades_json=trades_df.to_json(orient="records", default_handler=str),
            drawdown_json=dd_data.to_json(orient="records"),
            metrics_json=json.dumps(metrics, default=str),
        )

        run_id = save_run(run, data)
        st.success(f"Saved as Run #{run_id}")
