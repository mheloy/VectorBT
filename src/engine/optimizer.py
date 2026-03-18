"""Grid search optimizer using VectorBT's native vectorized parameter sweeps.

Uses VectorBT's ability to accept multi-column DataFrames in Portfolio.from_signals(),
running all parameter combinations in a single vectorized call.
For strategies with advanced position management, falls back to per-combo simulation.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.strategies.base import BaseStrategy, StrategyParam
from src.engine.simulator import simulate
from src.engine.sim_result import build_simulation_result


@dataclass
class OptimizationResult:
    """Result of a grid search optimization."""
    results_df: pd.DataFrame  # All combinations with metrics
    best_params: dict
    best_metric_value: float
    metric_name: str
    sweep_x: str
    sweep_y: str | None
    heatmap_data: pd.DataFrame | None  # Pivot table for 2D heatmap


def optimize(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    sweep_params: dict[str, list],
    metric: str = "sharpe_ratio",
    init_cash: float = 10_000.0,
    fees: float = 0.0001,
    freq: str | None = None,
    sl_stop: float | None = None,
    tp_stop: float | None = None,
    progress_cb=None,
) -> OptimizationResult:
    """Run grid search over parameter combinations using VectorBT's vectorized engine.

    Generates entry/exit signals for all parameter combos, stacks them into
    multi-column DataFrames, and passes them to a single Portfolio.from_signals()
    call for maximum performance.

    Args:
        strategy: Strategy instance.
        df: OHLCV DataFrame.
        sweep_params: Dict of param_name -> list of values to test.
        metric: Metric to optimize.
        init_cash: Starting cash.
        fees: Fee per trade as fraction.
        freq: Data frequency string.
        sl_stop: Stop-loss fraction.
        tp_stop: Take-profit fraction.
        progress_cb: Optional callback(current, total, phase) for progress updates.
    """
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    combinations = list(itertools.product(*param_values))

    defaults = strategy.default_params()
    fixed_params = {k: v for k, v in defaults.items() if k not in sweep_params}

    # Check if strategy uses advanced position management
    test_params = dict(zip(param_names, combinations[0]))
    test_params.update(fixed_params)
    has_pm = strategy.position_management(**test_params) is not None

    if has_pm:
        return _optimize_with_pm(
            strategy, df, param_names, combinations, fixed_params,
            metric, init_cash, fees, freq, progress_cb,
        )

    # Generate signals for all combinations and stack into multi-column DataFrames
    all_entries = {}
    all_exits = {}
    combo_labels = []

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        params.update(fixed_params)
        entries, exits = strategy.generate_signals(df, **params)
        label = tuple(combo)
        all_entries[label] = entries.values
        all_exits[label] = exits.values
        combo_labels.append(label)
        if progress_cb is not None:
            progress_cb(i + 1, len(combinations), "signals")

    # Build multi-column DataFrames — each column is one param combo
    if len(param_names) == 1:
        col_index = pd.Index(
            [c[0] for c in combo_labels], name=param_names[0]
        )
    else:
        col_index = pd.MultiIndex.from_tuples(combo_labels, names=param_names)

    entries_df = pd.DataFrame(
        np.column_stack(list(all_entries.values())),
        index=df.index,
        columns=col_index,
    )
    exits_df = pd.DataFrame(
        np.column_stack(list(all_exits.values())),
        index=df.index,
        columns=col_index,
    )

    # Single vectorized Portfolio.from_signals() call across all combos
    pf_kwargs = dict(
        close=df["close"],
        entries=entries_df,
        exits=exits_df,
        init_cash=init_cash,
        fees=fees,
    )
    if freq:
        pf_kwargs["freq"] = freq
    if sl_stop is not None:
        pf_kwargs["sl_stop"] = sl_stop
    if tp_stop is not None:
        pf_kwargs["tp_stop"] = tp_stop

    pf = vbt.Portfolio.from_signals(**pf_kwargs)

    # Extract per-column stats
    metric_map = {
        "total_return": "Total Return [%]",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "calmar_ratio": "Calmar Ratio",
        "win_rate": "Win Rate [%]",
        "profit_factor": "Profit Factor",
        "max_drawdown_pct": "Max Drawdown [%]",
        "total_trades": "Total Trades",
    }

    # Extract per-column stats using column label selection
    results = []
    for i, combo in enumerate(combinations):
        label = combo_labels[i]
        try:
            col_pf = pf[label]
            stats = col_pf.stats()
        except Exception:
            # Fallback: run individual backtest for this combo
            params = dict(zip(param_names, combo))
            params.update(fixed_params)
            e, x = strategy.generate_signals(df, **params)
            fallback_kwargs = dict(close=df["close"], entries=e, exits=x, init_cash=init_cash, fees=fees)
            if freq:
                fallback_kwargs["freq"] = freq
            if sl_stop is not None:
                fallback_kwargs["sl_stop"] = sl_stop
            if tp_stop is not None:
                fallback_kwargs["tp_stop"] = tp_stop
            col_pf = vbt.Portfolio.from_signals(**fallback_kwargs)
            stats = col_pf.stats()

        row = dict(zip(param_names, combo))
        for our_name, vbt_name in metric_map.items():
            row[our_name] = _safe_get(stats, vbt_name)
        row["total_trades"] = int(row["total_trades"])
        results.append(row)

    results_df = pd.DataFrame(results)

    # Find best
    if metric == "max_drawdown_pct":
        best_idx = results_df[metric].idxmin()
    else:
        best_idx = results_df[metric].idxmax()

    best_row = results_df.iloc[best_idx]
    best_params = {name: best_row[name] for name in param_names}
    best_value = best_row[metric]

    # Build heatmap for 2D sweeps
    sweep_x = param_names[0]
    sweep_y = param_names[1] if len(param_names) >= 2 else None
    heatmap_data = None

    if sweep_y:
        heatmap_data = results_df.pivot_table(
            index=sweep_y, columns=sweep_x, values=metric, aggfunc="first"
        )

    return OptimizationResult(
        results_df=results_df,
        best_params=best_params,
        best_metric_value=best_value,
        metric_name=metric,
        sweep_x=sweep_x,
        sweep_y=sweep_y,
        heatmap_data=heatmap_data,
    )


def _optimize_with_pm(
    strategy, df, param_names, combinations, fixed_params,
    metric, init_cash, fees, freq, progress_cb,
):
    """Optimize strategies with advanced position management (per-combo simulation)."""
    metric_map = {
        "total_return": "total_return",
        "sharpe_ratio": "sharpe_ratio",
        "sortino_ratio": "sortino_ratio",
        "calmar_ratio": "calmar_ratio",
        "win_rate": "win_rate",
        "profit_factor": "profit_factor",
        "max_drawdown_pct": "max_drawdown_pct",
        "total_trades": "total_trades",
    }

    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        full_params = {**fixed_params, **params}

        entries, exits = strategy.generate_signals(df, **full_params)
        pm_config = strategy.position_management(**full_params)
        sl_distances = strategy.compute_sl_distances(df, **full_params)

        st_values = None
        if pm_config.trail_mode == "st_line" and hasattr(strategy, 'compute_supertrend_values'):
            st_values = strategy.compute_supertrend_values(df, **full_params)

        equity_arr, trade_records, n_trades = simulate(
            df=df, entries=entries, exits=exits, sl_distances=sl_distances,
            config=pm_config, init_cash=init_cash, fees=fees,
            risk_pct=pm_config.risk_pct, max_lot_value=pm_config.max_lot_value,
            st_values=st_values,
        )

        sim_result = build_simulation_result(
            equity_arr, trade_records, n_trades, df.index, init_cash, fees,
        )

        row = dict(zip(param_names, combo))
        for our_name, metrics_key in metric_map.items():
            row[our_name] = sim_result.metrics.get(metrics_key, 0.0)
        row["total_trades"] = int(row["total_trades"])
        results.append(row)

        if progress_cb is not None:
            progress_cb(i + 1, len(combinations), "simulate")

    results_df = pd.DataFrame(results)

    # Find best
    if metric == "max_drawdown_pct":
        best_idx = results_df[metric].idxmin()
    else:
        best_idx = results_df[metric].idxmax()

    best_row = results_df.iloc[best_idx]
    best_params = {name: best_row[name] for name in param_names}
    best_value = best_row[metric]

    sweep_x = param_names[0]
    sweep_y = param_names[1] if len(param_names) >= 2 else None
    heatmap_data = None

    if sweep_y:
        heatmap_data = results_df.pivot_table(
            index=sweep_y, columns=sweep_x, values=metric, aggfunc="first"
        )

    return OptimizationResult(
        results_df=results_df,
        best_params=best_params,
        best_metric_value=best_value,
        metric_name=metric,
        sweep_x=sweep_x,
        sweep_y=sweep_y,
        heatmap_data=heatmap_data,
    )


def _safe_get(stats, key, default=0.0):
    try:
        val = stats[key]
        if pd.isna(val):
            return default
        return float(val)
    except (KeyError, IndexError):
        return default
