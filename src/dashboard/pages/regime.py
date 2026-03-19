"""Regime Analysis page."""

import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_m5, resample, TIMEFRAMES
from src.strategies.registry import get_all_strategies
from src.engine.regime import detect_regimes, backtest_by_regime, RegimeMethod

st.title("Market Regime Analysis")

@st.cache_data(show_spinner="Loading XAUUSD data...")
def cached_load():
    return load_m5()

raw_data = cached_load()

# --- Sidebar ---
strategies = get_all_strategies()
strategy_name = st.sidebar.selectbox("Strategy", list(strategies.keys()), key="rg_strategy")
strategy = strategies[strategy_name]
timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1H"), key="rg_tf")

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")
param_values = {}
for p in strategy.parameters():
    if p.choices:
        param_values[p.name] = st.sidebar.selectbox(p.description or p.name, p.choices, index=p.choices.index(p.default), key=f"rg_{p.name}")
    elif p.min_val is not None:
        param_values[p.name] = st.sidebar.slider(p.description or p.name, p.min_val, p.max_val, p.default, p.step or 1, key=f"rg_{p.name}")
    else:
        param_values[p.name] = st.sidebar.number_input(p.description or p.name, value=p.default, key=f"rg_{p.name}")

st.sidebar.markdown("---")
st.sidebar.subheader("Regime Settings")
method = st.sidebar.selectbox("Detection Method", [RegimeMethod.HMM.value, RegimeMethod.ADX.value], key="rg_method")
n_regimes = st.sidebar.slider("Number of Regimes (HMM)", 2, 4, 3, key="rg_nreg", disabled=method != "HMM")

init_cash = st.sidebar.number_input("Initial Cash ($)", value=10000.0, step=1000.0, key="rg_cash")
fees = st.sidebar.number_input("Fees", value=0.0, step=0.0001, format="%.6f", key="rg_fees")

freq_map = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h", "D": "1D"}

if st.sidebar.button("Analyze Regimes", type="primary", use_container_width=True):
    with st.spinner("Detecting regimes and running backtests..."):
        df = resample(raw_data, timeframe)
        regime_result = detect_regimes(df, RegimeMethod(method), n_regimes)
        bt_result = backtest_by_regime(
            strategy, df, param_values, regime_result,
            init_cash=init_cash, fees=fees, freq=freq_map.get(timeframe),
        )
    st.session_state["regime_result"] = regime_result
    st.session_state["regime_bt"] = bt_result
    st.session_state["regime_df"] = df

if "regime_result" not in st.session_state:
    st.info("Configure and click 'Analyze Regimes' to begin.")
    st.stop()

regime_result = st.session_state["regime_result"]
bt_result = st.session_state["regime_bt"]
df = st.session_state["regime_df"]

# --- Full backtest metrics ---
fm = bt_result.full_metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Full Return", f"{fm['total_return']:.2f}%")
col2.metric("Full Sharpe", f"{fm['sharpe_ratio']:.2f}")
col3.metric("Full Win Rate", f"{fm['win_rate']:.1f}%")
col4.metric("Full Trades", f"{fm['total_trades']}")

# --- Tabs ---
tab_chart, tab_stats, tab_backtest, tab_matrix = st.tabs([
    "Regime Chart", "Regime Stats", "Per-Regime Backtest", "Transition Matrix",
])

# ======== REGIME CHART ========
with tab_chart:
    st.subheader("Price with Regime Overlay")

    # Subsample for performance (max 5000 points)
    step = max(1, len(df) // 5000)
    plot_df = df.iloc[::step]
    plot_labels = regime_result.regime_labels.iloc[::step]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["close"], mode="lines",
        line=dict(color="black", width=1), name="Price",
    ))

    # Add regime background colors
    for regime_id, name in regime_result.regime_names.items():
        color = regime_result.regime_colors.get(regime_id, "gray")
        mask = plot_labels == regime_id
        if mask.any():
            fig.add_trace(go.Scatter(
                x=plot_df.index[mask], y=plot_df["close"][mask],
                mode="markers", marker=dict(color=color, size=2, opacity=0.5),
                name=name,
            ))

    fig.update_layout(height=500, yaxis_title="Price", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ======== REGIME STATS ========
with tab_stats:
    st.subheader("Regime Statistics")
    st.dataframe(regime_result.per_regime_stats, use_container_width=True, hide_index=True)

# ======== PER-REGIME BACKTEST ========
with tab_backtest:
    st.subheader("Strategy Performance by Regime")
    st.dataframe(bt_result.per_regime_metrics, use_container_width=True, hide_index=True)

    if not bt_result.per_regime_metrics.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bt_result.per_regime_metrics["Regime"],
            y=bt_result.per_regime_metrics["Return %"],
            marker_color=["green" if r > 0 else "red" for r in bt_result.per_regime_metrics["Return %"]],
            text=bt_result.per_regime_metrics["Return %"].round(2),
            textposition="outside",
        ))
        fig.update_layout(height=400, yaxis_title="Return (%)", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=bt_result.per_regime_metrics["Regime"],
                y=bt_result.per_regime_metrics["Sharpe"],
                marker_color="steelblue",
            ))
            fig2.update_layout(height=300, yaxis_title="Sharpe Ratio", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=bt_result.per_regime_metrics["Regime"],
                y=bt_result.per_regime_metrics["Win Rate %"],
                marker_color="coral",
            ))
            fig3.update_layout(height=300, yaxis_title="Win Rate (%)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig3, use_container_width=True)

# ======== TRANSITION MATRIX ========
with tab_matrix:
    st.subheader("Regime Transition Probabilities")
    st.caption("Probability of moving from row-regime to column-regime")
    if not regime_result.transition_matrix.empty:
        fig = go.Figure(data=go.Heatmap(
            z=regime_result.transition_matrix.values * 100,
            x=list(regime_result.transition_matrix.columns),
            y=list(regime_result.transition_matrix.index),
            text=[[f"{v:.1f}%" for v in row] for row in regime_result.transition_matrix.values * 100],
            texttemplate="%{text}",
            colorscale="Blues",
        ))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
