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
timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("5M"), key="wf_tf")

st.sidebar.markdown("---")
st.sidebar.subheader("Sweep Parameters")

# Params to sweep in WFA (signal + SL only, matching backtest-engine approach).
# Other params (TP levels, BE, risk_pct) are kept fixed at defaults.
# Centered on WFA-optimized values (2026-03-19): period=20, factor=1.2, sl_atr_mult=1.0
# 2 steps each direction from center = 5 variations per param.
_WF_SWEEP_PARAMS = {
    "period":      {"range": (16, 24), "step": 2},      # 16,18,20,22,24
    "factor":      {"range": (0.8, 1.6), "step": 0.2},  # 0.8,1.0,1.2,1.4,1.6
    "sl_atr_mult": {"range": (0.6, 1.4), "step": 0.2},  # 0.6,0.8,1.0,1.2,1.4
}

numeric_params = [p for p in strategy.parameters() if p.min_val is not None]
sweep_params = {}

# Show swept params with range/step controls
st.sidebar.caption("Swept in WFA (signal + SL)")
for p in numeric_params:
    if p.name not in _WF_SWEEP_PARAMS:
        continue
    wf_def = _WF_SWEEP_PARAMS[p.name]
    lo = max(p.min_val, wf_def["range"][0])
    hi = min(p.max_val, wf_def["range"][1])

    vals = st.sidebar.slider(
        f"{p.name} range", p.min_val, p.max_val, (lo, hi),
        step=p.step or 1, key=f"wf_{p.name}",
    )
    def_step = float(wf_def["step"])
    min_step = float(p.step or 0.1)
    step = st.sidebar.number_input(
        f"{p.name} step", value=def_step,
        min_value=min_step, step=0.1, format="%.4f",
        key=f"wf_{p.name}_step",
    )
    if isinstance(step, float) and step != int(step):
        sweep_vals = []
        v = float(vals[0])
        while v <= float(vals[1]) + 1e-9:
            sweep_vals.append(round(v, 4))
            v += float(step)
        sweep_params[p.name] = sweep_vals
    else:
        sweep_params[p.name] = list(range(int(vals[0]), int(vals[1]) + 1, int(step)))

# Show fixed params (held at defaults, not swept)
fixed_names = [p.name for p in numeric_params if p.name not in _WF_SWEEP_PARAMS]
if fixed_names:
    with st.sidebar.expander("Fixed params (not swept)", expanded=False):
        defaults = strategy.default_params()
        for p in numeric_params:
            if p.name in _WF_SWEEP_PARAMS:
                continue
            st.text(f"{p.name} = {defaults.get(p.name, p.default)}")

# Force adv_pm=On for SuperTrend PM path
if "adv_pm" in {p.name for p in strategy.parameters()}:
    sweep_params["adv_pm"] = ["On"]

st.sidebar.markdown("---")
st.sidebar.subheader("Walk-Forward Settings")
n_windows = st.sidebar.slider("Number of OOS windows", 4, 50, 8, key="wf_nwin")
anchored = st.sidebar.checkbox("Anchored IS (expanding window)", value=False, key="wf_anchored")
min_trades = st.sidebar.slider("Min trades per IS window", 5, 50, 20, key="wf_min_trades")
metric = st.sidebar.selectbox("Optimization metric", ["calmar_ratio", "sharpe_ratio", "total_return", "sortino_ratio", "profit_factor"], key="wf_metric")
init_cash = st.sidebar.number_input("Initial Cash ($)", value=10000.0, step=1000.0, key="wf_cash")
fees = st.sidebar.number_input("Commission (fraction)", value=0.000006, step=0.000001, format="%.6f",
                                help="ECN commission as fraction of notional", key="wf_fees")
slippage = st.sidebar.number_input("Slippage (fraction)", value=0.000004, step=0.000001, format="%.6f",
                                    help="Half-spread as fraction of price", key="wf_slippage")

st.sidebar.markdown("---")
st.sidebar.subheader("Execution Model")
execution_mode = st.sidebar.selectbox(
    "Execution Timing",
    ["next_bar_open", "same_bar_close"],
    index=0,
    format_func=lambda x: "Next Bar Open (realistic)" if x == "next_bar_open" else "Same Bar Close (legacy)",
    key="wf_exec_mode",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hold-Out Validation")
holdout_enabled = st.sidebar.checkbox("Reserve hold-out set", value=True, key="wf_holdout")
holdout_pct = st.sidebar.slider("Hold-out %", 0.05, 0.30, 0.10, 0.01, key="wf_holdout_pct",
                                 disabled=not holdout_enabled,
                                 help="Fraction of data reserved as unseen validation set")

freq_map = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h", "D": "1D"}

if st.sidebar.button("Run Walk-Forward", type="primary", use_container_width=True):
    progress_bar = st.progress(0, text="Preparing walk-forward...")

    def on_wf_progress(current, total, phase):
        if phase == "window":
            progress_bar.progress(current / (total + 1), text=f"Window {current}/{total}: optimizing IS -> testing OOS")
        elif phase == "full_sample":
            progress_bar.progress(total / (total + 1), text="Running full-sample optimization...")

    df = resample(raw_data, timeframe)
    result = run_walk_forward(
        strategy=strategy, df=df, sweep_params=sweep_params,
        n_windows=n_windows, anchored=anchored, min_trades=min_trades,
        metric=metric, init_cash=init_cash, fees=fees,
        slippage=slippage,
        freq=freq_map.get(timeframe),
        progress_cb=on_wf_progress,
        execution_mode=execution_mode,
        holdout_enabled=holdout_enabled,
        holdout_pct=holdout_pct if holdout_enabled else 0.0,
    )
    progress_bar.progress(1.0, text="Done!")
    import time; time.sleep(0.5)
    progress_bar.empty()
    st.session_state["wf_result"] = result

if "wf_result" in st.session_state:
    result = st.session_state["wf_result"]

    # --- Verdict banner ---
    if result.verdict == "PASS":
        st.success(f"**PASS** — {result.verdict_reason}")
    else:
        st.error(f"**FAIL** — {result.verdict_reason}")

    # --- Metrics cards ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg OOS Return/Window", f"{result.oos_total_return:.2f}%")
    col2.metric("Avg Efficiency Ratio", f"{result.avg_efficiency_ratio:.2f}")
    col3.metric("Profitable Windows", f"{result.profitable_windows_pct:.0f}%")
    col4.metric("Full-Sample Return", f"{result.full_sample_return:.2f}%")

    col5, col6, col7 = st.columns(3)
    col5.metric("Avg OOS Sharpe", f"{result.oos_sharpe:.2f}")
    overfitting_ratio = 0.0
    if result.full_sample_return != 0:
        overfitting_ratio = result.oos_total_return / result.full_sample_return * 100
    col6.metric("OOS/Full-Sample Ratio", f"{overfitting_ratio:.1f}%")
    col7.metric("Total Windows", f"{len(result.windows)}")

    # --- Parameter Stability ---
    if result.param_stability:
        st.subheader("Parameter Stability Across Windows")
        stability_data = []
        for pname, stats in result.param_stability.items():
            stability_data.append({
                "Parameter": pname,
                "Mean": f"{stats['mean']:.2f}",
                "Std Dev": f"{stats['stddev']:.2f}",
                "Stable": "Yes" if stats['stddev'] < stats['mean'] * 0.1 else "No",
                "Values": str([round(v, 2) for v in stats['values']]),
            })
        st.dataframe(stability_data, use_container_width=True, hide_index=True)

    # --- Equity comparison ---
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

    # --- IS vs OOS Sharpe + Efficiency Ratio per window ---
    st.subheader("In-Sample vs Out-of-Sample per Window")
    win_indices = list(range(len(result.summary_df)))

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=win_indices, y=result.summary_df["IS Sharpe"], name="IS Sharpe", marker_color="steelblue"))
    fig2.add_trace(go.Bar(x=win_indices, y=result.summary_df["OOS Sharpe"], name="OOS Sharpe", marker_color="coral"))
    fig2.update_layout(barmode="group", height=350, xaxis_title="Window", yaxis_title="Sharpe Ratio", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    # Efficiency Ratio per window
    if "Efficiency Ratio" in result.summary_df.columns:
        st.subheader("Efficiency Ratio per Window")
        fig3 = go.Figure()
        er_vals = result.summary_df["Efficiency Ratio"]
        colors = ["green" if v >= 0.5 else "red" for v in er_vals]
        fig3.add_trace(go.Bar(x=win_indices, y=er_vals, marker_color=colors, name="ER"))
        fig3.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="ER = 0.5 threshold")
        fig3.update_layout(height=300, xaxis_title="Window", yaxis_title="Efficiency Ratio (OOS Sharpe / IS Sharpe)", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    # --- Hold-Out Validation ---
    if result.holdout_metrics is not None:
        st.markdown("---")
        st.subheader("Hold-Out Validation (Unseen Data)")
        st.caption("These results are from data never seen during optimization or WFA")
        hm = result.holdout_metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Return", f"{hm.get('total_return', 0):.2f}%")
        col2.metric("Sharpe", f"{hm.get('sharpe_ratio', 0):.2f}")
        col3.metric("Win Rate", f"{hm.get('win_rate', 0):.1f}%")
        col4.metric("Trades", f"{hm.get('total_trades', 0)}")

        if result.holdout_equity is not None and not result.holdout_equity.empty:
            st.subheader("Hold-Out Equity Curve")
            fig_ho = go.Figure()
            fig_ho.add_trace(go.Scatter(
                x=result.holdout_equity.index, y=result.holdout_equity.values,
                mode="lines", name="Hold-Out", line=dict(color="green", width=2),
            ))
            fig_ho.update_layout(height=350, yaxis_title="Equity ($)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_ho, use_container_width=True)

        if result.holdout_params:
            st.caption(f"Params used: {result.holdout_params}")

    # --- Window details table ---
    st.subheader("Window Details")
    st.dataframe(result.summary_df, use_container_width=True, hide_index=True)
