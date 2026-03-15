"""Deep Trade Analysis page: R-multiples, MAE/MFE, streaks, sessions."""

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data.loader import load_m5, resample, TIMEFRAMES
from src.strategies.registry import get_all_strategies
from src.engine.runner import run_backtest
from src.engine.trade_analysis import analyze_trades

st.title("Deep Trade Analysis")

@st.cache_data(show_spinner="Loading XAUUSD data...")
def cached_load():
    return load_m5()

raw_data = cached_load()

# --- Sidebar ---
strategies = get_all_strategies()
strategy_name = st.sidebar.selectbox("Strategy", list(strategies.keys()), key="td_strategy")
strategy = strategies[strategy_name]
timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1H"), key="td_tf")

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")
param_values = {}
for p in strategy.parameters():
    if p.choices:
        param_values[p.name] = st.sidebar.selectbox(p.description or p.name, p.choices, index=p.choices.index(p.default), key=f"td_{p.name}")
    elif p.min_val is not None:
        param_values[p.name] = st.sidebar.slider(p.description or p.name, p.min_val, p.max_val, p.default, p.step or 1, key=f"td_{p.name}")
    else:
        param_values[p.name] = st.sidebar.number_input(p.description or p.name, value=p.default, key=f"td_{p.name}")

init_cash = st.sidebar.number_input("Initial Cash ($)", value=10000.0, step=1000.0, key="td_cash")
fees = st.sidebar.number_input("Fees", value=0.0001, step=0.0001, format="%.4f", key="td_fees")

freq_map = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h", "D": "1D"}

if st.sidebar.button("Analyze Trades", type="primary", use_container_width=True):
    with st.spinner("Running backtest and analyzing trades..."):
        df = resample(raw_data, timeframe)
        portfolio = run_backtest(strategy, df, param_values, init_cash, fees, freq=freq_map.get(timeframe))
        result = analyze_trades(portfolio, df)
    st.session_state["trade_analysis"] = result
    st.session_state["trade_portfolio"] = portfolio

if "trade_analysis" not in st.session_state:
    st.info("Configure strategy and click 'Analyze Trades' to begin.")
    st.stop()

result = st.session_state["trade_analysis"]

# --- Exposure metrics ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Time in Market", f"{result.exposure_pct:.1f}%")
col2.metric("Long Exposure", f"{result.long_pct:.1f}%")
col3.metric("Short Exposure", f"{result.short_pct:.1f}%")
col4.metric("Max Win Streak", f"{result.max_win_streak}")
col5.metric("Max Loss Streak", f"{result.max_loss_streak}")

# --- Tabs ---
tab_r, tab_mae, tab_streak, tab_session, tab_dur = st.tabs([
    "R-Multiples", "MAE/MFE", "Streaks", "Sessions", "Duration vs PnL",
])

# ======== R-MULTIPLES ========
with tab_r:
    if result.r_multiples is not None and not result.r_multiples.empty:
        st.subheader("R-Multiple Distribution")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Mean R", f"{result.r_stats.get('mean_r', 0):.2f}")
        col_b.metric("% Positive R", f"{result.r_stats.get('positive_r_pct', 0):.1f}%")
        col_c.metric("% > 2R", f"{result.r_stats.get('gt_2r_pct', 0):.1f}%")

        fig = go.Figure()
        colors = ["green" if r > 0 else "red" for r in result.r_multiples]
        fig.add_trace(go.Histogram(x=result.r_multiples, nbinsx=40, marker_color="steelblue"))
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.add_vline(x=result.r_stats["mean_r"], line_dash="dash", line_color="orange", annotation_text="Mean R")
        fig.update_layout(height=400, xaxis_title="R-Multiple", yaxis_title="Count", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No R-multiple data available.")

# ======== MAE/MFE ========
with tab_mae:
    if result.mae_mfe_df is not None and not result.mae_mfe_df.empty:
        st.subheader("Maximum Adverse / Favorable Excursion")

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("MAE vs Final PnL — helps identify optimal stop-loss")
            fig_mae = go.Figure()
            colors = ["green" if p > 0 else "red" for p in result.mae_mfe_df["pnl"]]
            fig_mae.add_trace(go.Scatter(
                x=result.mae_mfe_df["mae_pct"], y=result.mae_mfe_df["return_pct"],
                mode="markers", marker=dict(color=colors, size=6, opacity=0.7),
            ))
            fig_mae.update_layout(height=400, xaxis_title="MAE (%)", yaxis_title="Trade Return (%)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_mae, use_container_width=True)

        with col_b:
            st.caption("MFE vs Final PnL — helps identify optimal take-profit")
            fig_mfe = go.Figure()
            fig_mfe.add_trace(go.Scatter(
                x=result.mae_mfe_df["mfe_pct"], y=result.mae_mfe_df["return_pct"],
                mode="markers", marker=dict(color=colors, size=6, opacity=0.7),
            ))
            fig_mfe.update_layout(height=400, xaxis_title="MFE (%)", yaxis_title="Trade Return (%)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_mfe, use_container_width=True)
    else:
        st.info("No MAE/MFE data available.")

# ======== STREAKS ========
with tab_streak:
    st.subheader("Win/Loss Streaks")
    if not result.streaks_df.empty:
        st.dataframe(result.streaks_df, use_container_width=True, hide_index=True)

        # Cumulative PnL with color coding
        portfolio = st.session_state.get("trade_portfolio")
        if portfolio:
            trades = portfolio.trades.records_readable
            if "PnL" in trades.columns:
                cumulative = trades["PnL"].cumsum()
                colors = ["green" if p > 0 else "red" for p in trades["PnL"]]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(len(trades))), y=trades["PnL"].values, marker_color=colors, name="Trade PnL"))
                fig.add_trace(go.Scatter(x=list(range(len(trades))), y=cumulative.values, mode="lines", name="Cumulative PnL", yaxis="y2"))
                fig.update_layout(
                    height=400,
                    xaxis_title="Trade #",
                    yaxis_title="Trade PnL ($)",
                    yaxis2=dict(title="Cumulative PnL ($)", overlaying="y", side="right"),
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

# ======== SESSIONS ========
with tab_session:
    if result.session_df is not None and not result.session_df.empty:
        st.subheader("Performance by Gold Trading Session")
        st.dataframe(result.session_df, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=result.session_df["session"], y=result.session_df["total_pnl"],
                marker_color=["green" if p > 0 else "red" for p in result.session_df["total_pnl"]],
            ))
            fig.update_layout(height=350, yaxis_title="Total PnL ($)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=result.session_df["session"], y=result.session_df["win_rate"],
                marker_color="steelblue",
            ))
            fig.update_layout(height=350, yaxis_title="Win Rate (%)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No session data available.")

# ======== DURATION vs PnL ========
with tab_dur:
    if result.duration_pnl_df is not None and not result.duration_pnl_df.empty:
        st.subheader("Trade Duration vs PnL")
        colors = ["green" if p > 0 else "red" for p in result.duration_pnl_df["pnl"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.duration_pnl_df["duration_hours"],
            y=result.duration_pnl_df["pnl"],
            mode="markers", marker=dict(color=colors, size=6, opacity=0.7),
        ))
        fig.update_layout(height=400, xaxis_title="Duration (hours)", yaxis_title="PnL ($)", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No duration data available.")
