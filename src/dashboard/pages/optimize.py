"""Optimization page with parameter grid search and heatmaps."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_m5, resample, TIMEFRAMES
from src.strategies.registry import get_all_strategies
from src.engine.optimizer import optimize


def _build_sweep_values(start, end, step):
    """Build sweep values supporting both int and float steps."""
    if isinstance(step, float) or isinstance(start, float) or isinstance(end, float):
        values = []
        v = float(start)
        while v <= float(end) + 1e-9:
            values.append(round(v, 4))
            v += float(step)
        return values
    return list(range(int(start), int(end) + 1, int(step)))


st.title("Parameter Optimization")

@st.cache_data(show_spinner="Loading XAUUSD data...")
def cached_load():
    return load_m5()

raw_data = cached_load()

# --- Sidebar ---
strategies = get_all_strategies()
strategy_name = st.sidebar.selectbox("Strategy", list(strategies.keys()), key="opt_strategy")
strategy = strategies[strategy_name]

timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1H"), key="opt_tf")

# Separate numeric and categorical params
all_params = strategy.parameters()
numeric_params = [p for p in all_params if p.min_val is not None and p.max_val is not None]
categorical_params = [p for p in all_params if p.choices is not None]

if len(numeric_params) < 1 and len(categorical_params) < 1:
    st.warning("This strategy has no parameters to optimize.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Sweep Parameters")

# --- Sweep X axis ---
sweepable_names = [p.name for p in numeric_params]
sweep_x_param = st.sidebar.selectbox("Sweep X axis", sweepable_names, key="sweep_x")
sweep_x_def = next(p for p in numeric_params if p.name == sweep_x_param)
sweep_x_range = st.sidebar.slider(
    f"{sweep_x_param} range",
    min_value=sweep_x_def.min_val, max_value=sweep_x_def.max_val,
    value=(sweep_x_def.min_val, sweep_x_def.max_val),
    step=sweep_x_def.step or 1,
    key="sweep_x_range",
)
sweep_x_step = st.sidebar.number_input(
    f"{sweep_x_param} step", value=float(sweep_x_def.step or 5),
    min_value=0.1, step=0.1, format="%.1f", key="sweep_x_step",
)
sweep_x_values = _build_sweep_values(sweep_x_range[0], sweep_x_range[1], sweep_x_step)

# --- Sweep Y axis (optional — defaults to second numeric param if available) ---
remaining_numeric = [p.name for p in numeric_params if p.name != sweep_x_param]
sweep_y_param = None
sweep_y_values = None

if remaining_numeric:
    default_y_idx = 1  # Default to first remaining param (not "None")
    sweep_y_param = st.sidebar.selectbox(
        "Sweep Y axis", ["None"] + remaining_numeric,
        index=default_y_idx, key="sweep_y",
    )
    if sweep_y_param != "None":
        sweep_y_def = next(p for p in numeric_params if p.name == sweep_y_param)
        sweep_y_range = st.sidebar.slider(
            f"{sweep_y_param} range",
            min_value=sweep_y_def.min_val, max_value=sweep_y_def.max_val,
            value=(sweep_y_def.min_val, sweep_y_def.max_val),
            step=sweep_y_def.step or 1,
            key="sweep_y_range",
        )
        sweep_y_step = st.sidebar.number_input(
            f"{sweep_y_param} step", value=float(sweep_y_def.step or 1),
            min_value=0.1, step=0.1, format="%.1f", key="sweep_y_step",
        )
        sweep_y_values = _build_sweep_values(sweep_y_range[0], sweep_y_range[1], sweep_y_step)
    else:
        sweep_y_param = None

# --- Categorical params: option to sweep or fix ---
categorical_sweep = {}
if categorical_params:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Categorical Parameters")
    for p in categorical_params:
        sweep_cat = st.sidebar.checkbox(f"Sweep {p.name}", value=True, key=f"opt_cat_sweep_{p.name}")
        if sweep_cat:
            selected = st.sidebar.multiselect(
                f"{p.name} values", p.choices, default=p.choices, key=f"opt_cat_{p.name}",
            )
            if selected:
                categorical_sweep[p.name] = selected
        else:
            fixed_val = st.sidebar.selectbox(
                f"{p.name} (fixed)", p.choices,
                index=p.choices.index(p.default), key=f"opt_cat_fixed_{p.name}",
            )
            categorical_sweep[p.name] = [fixed_val]

# --- Fixed values for non-swept numeric params ---
non_swept_numeric = [p for p in numeric_params if p.name != sweep_x_param and p.name != sweep_y_param]
fixed_params = {}
if non_swept_numeric:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Fixed Parameters")
    for p in non_swept_numeric:
        fixed_params[p.name] = st.sidebar.slider(
            f"{p.description or p.name} (fixed)",
            min_value=p.min_val, max_value=p.max_val,
            value=p.default, step=p.step or 1,
            key=f"opt_fixed_{p.name}",
        )

# Target metric
st.sidebar.markdown("---")
metric_options = [
    "sharpe_ratio", "sortino_ratio", "total_return", "calmar_ratio",
    "win_rate", "profit_factor", "max_drawdown_pct",
]
target_metric = st.sidebar.selectbox("Optimize for", metric_options, key="opt_metric")

# Settings
st.sidebar.markdown("---")
init_cash = st.sidebar.number_input("Initial Cash ($)", value=10000.0, step=1000.0, key="opt_cash")
fees = st.sidebar.number_input("Fees", value=0.0001, step=0.0001, format="%.4f", key="opt_fees")

# Combo count
n_combos = len(sweep_x_values)
if sweep_y_values:
    n_combos *= len(sweep_y_values)
for cat_vals in categorical_sweep.values():
    n_combos *= len(cat_vals)
st.sidebar.info(f"Testing {n_combos} combinations")

# --- Run ---
if st.sidebar.button("Run Optimization", type="primary", use_container_width=True):
    sweep_params = {sweep_x_param: sweep_x_values}
    if sweep_y_param and sweep_y_values:
        sweep_params[sweep_y_param] = sweep_y_values
    # Add categorical sweeps
    for cat_name, cat_vals in categorical_sweep.items():
        sweep_params[cat_name] = cat_vals

    freq_map = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h", "D": "1D"}

    progress_bar = st.progress(0, text="Preparing optimization...")
    status_text = st.empty()

    def on_progress(current, total, phase):
        pct = current / total
        if phase == "signals":
            progress_bar.progress(pct * 0.7, text=f"Generating signals: {current}/{total} combos")
        else:
            progress_bar.progress(0.7 + pct * 0.3, text=f"Extracting metrics: {current}/{total}")

    df = resample(raw_data, timeframe)
    result = optimize(
        strategy=strategy,
        df=df,
        sweep_params=sweep_params,
        metric=target_metric,
        init_cash=init_cash,
        fees=fees,
        freq=freq_map.get(timeframe),
        progress_cb=on_progress,
    )
    progress_bar.progress(1.0, text="Done!")
    import time; time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    st.session_state["opt_result"] = result

# --- Display results ---
if "opt_result" in st.session_state:
    result = st.session_state["opt_result"]

    st.success(f"Best {result.metric_name}: **{result.best_metric_value:.4f}** with params: {result.best_params}")

    # Heatmap for 2D sweeps
    if result.heatmap_data is not None:
        st.subheader(f"Parameter Heatmap: {result.metric_name}")
        fig = go.Figure(data=go.Heatmap(
            z=result.heatmap_data.values,
            x=[str(c) for c in result.heatmap_data.columns],
            y=[str(i) for i in result.heatmap_data.index],
            colorscale="RdYlGn" if result.metric_name != "max_drawdown_pct" else "RdYlGn_r",
            text=np.round(result.heatmap_data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=f"{result.sweep_x}: %{{x}}<br>{result.sweep_y}: %{{y}}<br>{result.metric_name}: %{{z:.4f}}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title=result.sweep_x,
            yaxis_title=result.sweep_y,
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 1D sweep: bar chart
        st.subheader(f"{result.metric_name} by {result.sweep_x}")
        fig = go.Figure(data=go.Bar(
            x=result.results_df[result.sweep_x].astype(str),
            y=result.results_df[result.metric_name],
        ))
        fig.update_layout(
            xaxis_title=result.sweep_x,
            yaxis_title=result.metric_name,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top N table
    st.subheader("Top 10 Combinations")
    if result.metric_name == "max_drawdown_pct":
        top = result.results_df.nsmallest(10, result.metric_name)
    else:
        top = result.results_df.nlargest(10, result.metric_name)
    st.dataframe(top, use_container_width=True, hide_index=True)

    # Full results
    with st.expander("All Results"):
        st.dataframe(result.results_df, use_container_width=True, hide_index=True)
