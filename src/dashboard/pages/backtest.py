"""Run Backtest page."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_m5, resample, TIMEFRAMES
from src.strategies.base import BaseStrategy
from src.strategies.registry import get_all_strategies
from src.engine.runner import run_backtest
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

# Build parameter inputs dynamically — separate core params from PM params
param_values = {}
pm_param_names = {"adv_pm", "tp1_r", "tp1_pct", "tp2_r", "tp2_pct", "be_trigger_r", "final_tp_r", "trail_mode", "risk_pct"}
h1_param_names = {"h1_period", "h1_factor", "h1_source"}

for p in strategy.parameters():
    if p.name in pm_param_names or p.name in h1_param_names:
        continue  # Show PM and H1 params separately below
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

# --- H1 Filter params (shown when H1 filter is On) ---
if param_values.get("h1_filter") == "On":
    h1_params = {p.name: p for p in strategy.parameters() if p.name in h1_param_names}
    if h1_params:
        st.sidebar.caption("H1 Filter Settings")
        param_values["h1_period"] = st.sidebar.slider("H1 Period", 10, 100, 50, 1, key="param_h1_period")
        param_values["h1_factor"] = st.sidebar.slider("H1 Factor", 0.5, 5.0, 3.0, 0.1, key="param_h1_factor")
        param_values["h1_source"] = st.sidebar.selectbox("H1 Source", ["hl2", "close", "hlc3", "ohlc4"], index=1, key="param_h1_source")
else:
    # Set defaults so they're in param_values
    param_values.setdefault("h1_period", 50)
    param_values.setdefault("h1_factor", 3.0)
    param_values.setdefault("h1_source", "close")

# --- Position Management section (only for strategies that support it) ---
has_pm_support = type(strategy).position_management is not BaseStrategy.position_management
if has_pm_support:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Position Management")

    pm_params = {p.name: p for p in strategy.parameters() if p.name in pm_param_names}

    adv_pm = st.sidebar.selectbox(
        "Advanced PM (Partial TP, BE, Trail)",
        ["On", "Off"],
        index=1,  # Default Off
        key="param_adv_pm",
    )
    param_values["adv_pm"] = adv_pm

    if adv_pm == "On":
        st.sidebar.caption("Partial Take Profits")
        col_a, col_b = st.sidebar.columns(2)
        param_values["tp1_r"] = col_a.slider("TP1 R-mult", 0.5, 5.0, 1.5, 0.1, key="param_tp1_r")
        param_values["tp1_pct"] = col_b.slider("TP1 %", 0.1, 0.9, 0.50, 0.05, key="param_tp1_pct")
        col_c, col_d = st.sidebar.columns(2)
        param_values["tp2_r"] = col_c.slider("TP2 R-mult", 1.0, 8.0, 2.9, 0.1, key="param_tp2_r")
        param_values["tp2_pct"] = col_d.slider("TP2 %", 0.1, 0.5, 0.30, 0.05, key="param_tp2_pct")

        st.sidebar.caption("Break-Even & Final TP")
        param_values["be_trigger_r"] = st.sidebar.slider("BE Trigger R", 0.3, 3.0, 1.0, 0.1, key="param_be_trigger_r")
        param_values["final_tp_r"] = st.sidebar.slider("Final TP R (runner)", 1.5, 10.0, 3.0, 0.5, key="param_final_tp_r")

        st.sidebar.caption("Trailing & Sizing")
        param_values["trail_mode"] = st.sidebar.selectbox("Trail Mode", ["st_line", "atr_stages"], index=0, key="param_trail_mode")
        param_values["risk_pct"] = st.sidebar.slider("Risk % per trade", 0.5, 10.0, 3.0, 0.5, key="param_risk_pct") / 100

        if param_values["trail_mode"] == "atr_stages":
            st.sidebar.caption("ATR Stages: 0.67R/1.0x, 1.0R/0.8x, 1.33R/0.6x")
        else:
            st.sidebar.caption("Runner trails SuperTrend line (natural S/R)")
    else:
        # Set defaults so they're in param_values but PM is off
        for pn in pm_param_names:
            if pn not in param_values:
                param_values[pn] = pm_params[pn].default if pn in pm_params else "Off"

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Management")
init_cash = st.sidebar.number_input("Initial Cash ($)", value=10000.0, step=1000.0)
fees = st.sidebar.number_input("Fees (fraction)", value=0.0001, step=0.0001, format="%.4f")

# Check if strategy provides dynamic ATR-based stops
_has_dynamic_stops = type(strategy).compute_stops is not BaseStrategy.compute_stops
# If advanced PM is on, stops are handled by the simulator
_pm_active = param_values.get("adv_pm") == "On"

if _pm_active:
    st.sidebar.info("SL/TP: Managed by advanced PM (partial TP + trailing)")
    sl_stop = None
    tp_stop = None
elif _has_dynamic_stops:
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
        result = run_backtest(
            strategy=strategy,
            df=df,
            params=param_values,
            init_cash=init_cash,
            fees=fees,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            freq=freq_map.get(timeframe),
        )

    st.session_state["bt_result"] = result
    st.session_state["backtest_df"] = df
    st.session_state["backtest_params"] = param_values
    st.session_state["backtest_strategy"] = strategy_name
    st.session_state["backtest_tf"] = timeframe
    st.session_state["backtest_config"] = {
        "init_cash": init_cash, "fees": fees, "sl_stop": sl_stop, "tp_stop": tp_stop,
    }

# --- Display results ---
if "bt_result" in st.session_state:
    result = st.session_state["bt_result"]
    df = st.session_state["backtest_df"]
    metrics = result.metrics
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

    # --- Exit type breakdown (simulator path only) ---
    if result.is_simulator and "exit_type_breakdown" in metrics:
        breakdown = metrics["exit_type_breakdown"]
        if breakdown:
            st.subheader("Exit Type Breakdown")
            cols = st.columns(len(breakdown))
            for idx, (exit_type, count) in enumerate(breakdown.items()):
                cols[idx].metric(exit_type, count)

    # --- Equity curve ---
    st.subheader("Equity Curve")
    equity = result.equity_curve
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Portfolio Value"))
    fig_eq.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Value ($)")
    st.plotly_chart(fig_eq, use_container_width=True)

    # --- Drawdown ---
    st.subheader("Drawdown")
    dd = result.drawdown_series
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", fill="tozeroy",
        name="Drawdown", line=dict(color="red"),
    ))
    fig_dd.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    # --- Trade table ---
    st.subheader("Trades")
    trades_df = result.trades_df
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
