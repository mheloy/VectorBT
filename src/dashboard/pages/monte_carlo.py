"""Monte Carlo simulation page."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_m5, resample, TIMEFRAMES
from src.strategies.registry import get_all_strategies
from src.engine.runner import run_backtest
from src.engine.monte_carlo import run_monte_carlo
from src.engine.robustness import test_signal_delay, test_noise_injection, test_param_sensitivity

st.title("Monte Carlo & Robustness Analysis")

@st.cache_data(show_spinner="Loading XAUUSD data...")
def cached_load():
    return load_m5()

raw_data = cached_load()

# --- Sidebar ---
strategies = get_all_strategies()
strategy_name = st.sidebar.selectbox("Strategy", list(strategies.keys()), key="mc_strategy")
strategy = strategies[strategy_name]
timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1H"), key="mc_tf")

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")
param_values = {}
for p in strategy.parameters():
    if p.choices:
        param_values[p.name] = st.sidebar.selectbox(p.description or p.name, p.choices, index=p.choices.index(p.default), key=f"mc_{p.name}")
    elif p.min_val is not None:
        param_values[p.name] = st.sidebar.slider(p.description or p.name, p.min_val, p.max_val, p.default, p.step or 1, key=f"mc_{p.name}")
    else:
        param_values[p.name] = st.sidebar.number_input(p.description or p.name, value=p.default, key=f"mc_{p.name}")

st.sidebar.markdown("---")
st.sidebar.subheader("Monte Carlo Settings")
n_sims = st.sidebar.slider("Simulations", 100, 5000, 1000, 100, key="mc_nsims")
ruin_pct = st.sidebar.slider("Ruin threshold (%)", 10, 80, 50, 5, key="mc_ruin")
init_cash = st.sidebar.number_input("Initial Cash ($)", value=10000.0, step=1000.0, key="mc_cash")
fees = st.sidebar.number_input("Fees", value=0.0, step=0.0001, format="%.6f", key="mc_fees")

freq_map = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h", "D": "1D"}

# --- Tabs ---
tab_mc, tab_delay, tab_noise, tab_sens = st.tabs([
    "Monte Carlo", "Signal Delay", "Noise Injection", "Param Sensitivity",
])

# ======== MONTE CARLO TAB ========
with tab_mc:
    if st.button("Run Monte Carlo", type="primary", key="run_mc"):
        with st.spinner(f"Running {n_sims} Monte Carlo simulations..."):
            df = resample(raw_data, timeframe)
            portfolio = run_backtest(strategy, df, param_values, init_cash, fees, freq=freq_map.get(timeframe))
            mc = run_monte_carlo(portfolio, n_simulations=n_sims, ruin_threshold_pct=ruin_pct)
        st.session_state["mc_result"] = mc

    if "mc_result" in st.session_state:
        mc = st.session_state["mc_result"]

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ruin Probability", f"{mc.ruin_probability:.1f}%")
        col2.metric("Original Max DD", f"{mc.original_max_dd:.1f}%")
        col3.metric("Median Max DD", f"{mc.median_max_dd:.1f}%")
        col4.metric("95th Pctl DD", f"{mc.worst_case_dd_95:.1f}%")

        # Fan chart
        st.subheader("Equity Fan Chart")
        trade_nums = np.arange(len(mc.p50))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trade_nums, y=mc.p95, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=trade_nums, y=mc.p5, fill="tonexty", mode="lines", line=dict(width=0), fillcolor="rgba(68,114,196,0.15)", name="5th-95th pctl"))
        fig.add_trace(go.Scatter(x=trade_nums, y=mc.p75, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=trade_nums, y=mc.p25, fill="tonexty", mode="lines", line=dict(width=0), fillcolor="rgba(68,114,196,0.3)", name="25th-75th pctl"))
        fig.add_trace(go.Scatter(x=trade_nums, y=mc.p50, mode="lines", line=dict(color="blue", width=1.5), name="Median"))
        fig.add_trace(go.Scatter(x=trade_nums, y=mc.original_equity, mode="lines", line=dict(color="red", width=2), name="Original"))
        fig.update_layout(height=450, xaxis_title="Trade #", yaxis_title="Equity ($)", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Max DD distribution
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Max Drawdown Distribution")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Histogram(x=mc.max_drawdowns, nbinsx=50, name="MC Max DD"))
            fig_dd.add_vline(x=mc.original_max_dd, line_dash="dash", line_color="red", annotation_text="Original")
            fig_dd.update_layout(height=350, xaxis_title="Max Drawdown (%)", yaxis_title="Count", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_dd, use_container_width=True)

        with col_b:
            st.subheader("Final Equity Distribution")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Histogram(x=mc.final_equities, nbinsx=50, name="MC Final Equity"))
            fig_eq.add_vline(x=mc.original_final_equity, line_dash="dash", line_color="red", annotation_text="Original")
            fig_eq.update_layout(height=350, xaxis_title="Final Equity ($)", yaxis_title="Count", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_eq, use_container_width=True)

# ======== SIGNAL DELAY TAB ========
with tab_delay:
    if st.button("Run Signal Delay Test", type="primary", key="run_delay"):
        with st.spinner("Testing signal delays..."):
            df = resample(raw_data, timeframe)
            delay_df = test_signal_delay(strategy, df, param_values, freq=freq_map.get(timeframe), init_cash=init_cash, fees=fees)
        st.session_state["delay_result"] = delay_df

    if "delay_result" in st.session_state:
        delay_df = st.session_state["delay_result"]
        st.dataframe(delay_df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=delay_df["delay_bars"].astype(str), y=delay_df["sharpe_ratio"], name="Sharpe"))
        fig.update_layout(height=350, xaxis_title="Signal Delay (bars)", yaxis_title="Sharpe Ratio", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ======== NOISE INJECTION TAB ========
with tab_noise:
    if st.button("Run Noise Test", type="primary", key="run_noise"):
        with st.spinner("Testing noise injection..."):
            df = resample(raw_data, timeframe)
            noise_df = test_noise_injection(strategy, df, param_values, freq=freq_map.get(timeframe), init_cash=init_cash, fees=fees)
        st.session_state["noise_result"] = noise_df

    if "noise_result" in st.session_state:
        noise_df = st.session_state["noise_result"]
        st.dataframe(noise_df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=noise_df["noise_pct"], y=noise_df["mean_sharpe"], mode="lines+markers", name="Mean Sharpe", error_y=dict(type="data", array=noise_df["std_sharpe"])))
        fig.update_layout(height=350, xaxis_title="Noise (%)", yaxis_title="Sharpe Ratio", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ======== PARAM SENSITIVITY TAB ========
with tab_sens:
    numeric_params = [p for p in strategy.parameters() if p.min_val is not None]
    if numeric_params:
        sens_param = st.selectbox("Parameter to test", [p.name for p in numeric_params], key="sens_param")
        if st.button("Run Sensitivity Test", type="primary", key="run_sens"):
            with st.spinner(f"Testing sensitivity for {sens_param}..."):
                df = resample(raw_data, timeframe)
                sens_df = test_param_sensitivity(strategy, df, param_values, sens_param, freq=freq_map.get(timeframe), init_cash=init_cash, fees=fees)
            st.session_state["sens_result"] = sens_df

        if "sens_result" in st.session_state:
            sens_df = st.session_state["sens_result"]
            st.dataframe(sens_df, use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=sens_df["perturbation"], y=sens_df["sharpe_ratio"], name="Sharpe"))
            fig.update_layout(height=350, xaxis_title="Perturbation", yaxis_title="Sharpe Ratio", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
