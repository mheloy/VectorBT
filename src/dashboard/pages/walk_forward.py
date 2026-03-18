"""Walk-Forward Optimization page."""

import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_m5, resample, TIMEFRAMES
from src.strategies.registry import get_all_strategies
from src.engine.walk_forward import run_walk_forward

st.title("Walk-Forward Optimization")

@st.cache_data(show_spinner="Loading XAUUSD data...")
def cached_load():
    return load_m5()

raw_data = cached_load()

# --- Sidebar ---
strategies = get_all_strategies()
strategy_name = st.sidebar.selectbox("Strategy", list(strategies.keys()), key="wf_strategy")
strategy = strategies[strategy_name]
timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1H"), key="wf_tf")

st.sidebar.markdown("---")
st.sidebar.subheader("Sweep Parameters")

numeric_params = [p for p in strategy.parameters() if p.min_val is not None]
sweep_params = {}
for p in numeric_params:
    vals = st.sidebar.slider(
        f"{p.name} range", p.min_val, p.max_val, (p.min_val, p.max_val),
        step=p.step or 1, key=f"wf_{p.name}",
    )
    default_step = float(p.step or 5)
    step = st.sidebar.number_input(f"{p.name} step", value=default_step, min_value=default_step, step=0.1, format="%.4f", key=f"wf_{p.name}_step")
    if isinstance(step, float) and step != int(step):
        sweep_vals = []
        v = float(vals[0])
        while v <= float(vals[1]) + 1e-9:
            sweep_vals.append(round(v, 4))
            v += float(step)
        sweep_params[p.name] = sweep_vals
    else:
        sweep_params[p.name] = list(range(int(vals[0]), int(vals[1]) + 1, int(step)))

st.sidebar.markdown("---")
st.sidebar.subheader("Walk-Forward Settings")
n_windows = st.sidebar.slider("Number of OOS windows", 4, 20, 8, key="wf_nwin")
anchored = st.sidebar.checkbox("Anchored IS (expanding window)", value=False, key="wf_anchored")
min_trades = st.sidebar.slider("Min trades per IS window", 1, 20, 3, key="wf_min_trades")
metric = st.sidebar.selectbox("Optimization metric", ["sharpe_ratio", "total_return", "sortino_ratio", "profit_factor", "calmar_ratio"], key="wf_metric")
init_cash = st.sidebar.number_input("Initial Cash ($)", value=10000.0, step=1000.0, key="wf_cash")
fees = st.sidebar.number_input("Fees", value=0.0, step=0.0001, format="%.6f", key="wf_fees")

freq_map = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h", "D": "1D"}

if st.sidebar.button("Run Walk-Forward", type="primary", use_container_width=True):
    progress_bar = st.progress(0, text="Preparing walk-forward...")

    def on_wf_progress(current, total, phase):
        if phase == "window":
            progress_bar.progress(current / (total + 1), text=f"Window {current}/{total}: optimizing IS → testing OOS")
        elif phase == "full_sample":
            progress_bar.progress(total / (total + 1), text="Running full-sample optimization...")

    df = resample(raw_data, timeframe)
    result = run_walk_forward(
        strategy=strategy, df=df, sweep_params=sweep_params,
        n_windows=n_windows, anchored=anchored, min_trades=min_trades,
        metric=metric, init_cash=init_cash, fees=fees,
        freq=freq_map.get(timeframe),
        progress_cb=on_wf_progress,
    )
    progress_bar.progress(1.0, text="Done!")
    import time; time.sleep(0.5)
    progress_bar.empty()
    st.session_state["wf_result"] = result

if "wf_result" in st.session_state:
    result = st.session_state["wf_result"]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("WF OOS Return", f"{result.oos_total_return:.2f}%")
    col2.metric("Full-Sample Return", f"{result.full_sample_return:.2f}%")
    col3.metric("Avg OOS Sharpe", f"{result.oos_sharpe:.2f}")

    overfitting_ratio = 0.0
    if result.full_sample_return != 0:
        overfitting_ratio = result.oos_total_return / result.full_sample_return * 100
    st.info(f"OOS/Full-Sample ratio: **{overfitting_ratio:.1f}%** (closer to 100% = less overfitting)")

    # Equity comparison
    st.subheader("OOS Equity vs Full-Sample Equity")
    fig = go.Figure()
    if not result.oos_equity_curve.empty:
        fig.add_trace(go.Scatter(
            x=result.oos_equity_curve.index, y=result.oos_equity_curve.values,
            mode="lines", name="Walk-Forward OOS", line=dict(color="blue", width=2),
        ))
    fig.add_trace(go.Scatter(
        x=result.full_sample_equity.index, y=result.full_sample_equity.values,
        mode="lines", name="Full-Sample Optimized", line=dict(color="gray", width=1, dash="dash"),
    ))
    fig.update_layout(height=400, yaxis_title="Equity ($)", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # IS vs OOS comparison per window
    st.subheader("In-Sample vs Out-of-Sample per Window")
    fig2 = go.Figure()
    windows = list(range(len(result.summary_df)))
    fig2.add_trace(go.Bar(x=windows, y=result.summary_df["IS Sharpe"], name="IS Sharpe", marker_color="steelblue"))
    fig2.add_trace(go.Bar(x=windows, y=result.summary_df["OOS Sharpe"], name="OOS Sharpe", marker_color="coral"))
    fig2.update_layout(barmode="group", height=350, xaxis_title="Window", yaxis_title="Sharpe Ratio", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    # Window details table
    st.subheader("Window Details")
    st.dataframe(result.summary_df, use_container_width=True, hide_index=True)
