"""Walk-forward optimization with tiled OOS coverage.

Splits data into anchored or rolling IS/OOS windows where OOS segments
tile the entire dataset with no gaps. Optimizes parameters on IS data,
validates on OOS data, and concatenates OOS results.

Methodology aligned with backtest-engine:
- Default metric: Calmar ratio (return / max drawdown)
- Default IS/OOS ratio: 5:1
- Min trades: 20
- Efficiency Ratio: OOS Sharpe / IS Sharpe per window
- Pass/Fail verdict: ER >= 0.5 AND >= 60% windows profitable
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.strategies.base import BaseStrategy
from src.engine.optimizer import optimize
from src.engine.runner import run_backtest


@dataclass
class WalkForwardWindow:
    """A single IS/OOS window result."""
    window_idx: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    best_params: dict
    is_sharpe: float
    is_return: float
    oos_sharpe: float
    oos_return: float
    oos_trades: int
    oos_profit_factor: float
    oos_win_rate: float
    efficiency_ratio: float  # OOS Sharpe / IS Sharpe (0 if IS Sharpe <= 0)


@dataclass
class WalkForwardResult:
    """Result of walk-forward optimization."""
    windows: list[WalkForwardWindow]
    oos_equity_curve: pd.Series  # Concatenated OOS equity
    full_sample_equity: pd.Series  # Single-optimization equity for comparison
    summary_df: pd.DataFrame  # Per-window summary
    oos_total_return: float
    full_sample_return: float
    oos_sharpe: float
    avg_oos_profit_factor: float
    avg_oos_win_rate: float
    # Robustness metrics (aligned with backtest-engine)
    avg_efficiency_ratio: float
    profitable_windows_pct: float
    verdict: str  # "PASS" or "FAIL"
    verdict_reason: str
    param_stability: dict  # Per-param {mean, stddev} across windows
    holdout_metrics: dict | None = None
    holdout_equity: pd.Series | None = None
    holdout_params: dict | None = None


def run_walk_forward(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    sweep_params: dict[str, list],
    n_windows: int = 8,
    is_bars: int | None = None,
    oos_bars: int | None = None,
    anchored: bool = False,
    min_trades: int = 20,
    metric: str = "calmar_ratio",
    init_cash: float = 10_000.0,
    fees: float = 0.000006,
    slippage: float = 0.000004,
    freq: str | None = None,
    progress_cb=None,
    execution_mode: str = "next_bar_open",
    holdout_enabled: bool = True,
    holdout_pct: float = 0.10,
) -> WalkForwardResult:
    """Run walk-forward optimization with full OOS data coverage.

    OOS windows tile the dataset from start to end with no gaps.
    IS window is either anchored (expanding from start) or rolling (fixed lookback).

    Args:
        strategy: Strategy instance.
        df: OHLCV DataFrame.
        sweep_params: Parameter grid for optimization.
        n_windows: Number of OOS windows to tile across the data.
        is_bars: Fixed IS window size in bars. If None, auto-calculated.
        oos_bars: Fixed OOS window size in bars. If None, auto-calculated from n_windows.
        anchored: If True, IS always starts from beginning (expanding window).
            If False, IS is a rolling fixed-size window.
        min_trades: Minimum trades required in IS optimization.
            Combos with fewer trades are penalized.
        metric: Metric to optimize on IS data (default: calmar_ratio).
        init_cash: Starting cash.
        fees: Fee fraction.
        freq: Data frequency string.
    """
    # Hold-out split
    if holdout_enabled and holdout_pct > 0:
        split_idx = int(len(df) * (1 - holdout_pct))
        train_df = df.iloc[:split_idx]
        holdout_df = df.iloc[split_idx:]
    else:
        train_df = df
        holdout_df = None

    # Use train_df for all WFA windows
    total_bars = len(train_df)

    # Calculate OOS size: tile the entire dataset into n_windows
    if oos_bars is None:
        oos_bars = total_bars // n_windows

    # Calculate IS size: default to 5x OOS (matches backtest-engine)
    if is_bars is None:
        is_bars = oos_bars * 5

    # Generate tiled IS/OOS splits
    splits = []
    oos_start = 0  # OOS starts from the very beginning

    while oos_start < total_bars:
        oos_end = min(oos_start + oos_bars, total_bars)

        if anchored:
            # IS = everything from start up to OOS start
            is_start_idx = 0
        else:
            # IS = rolling window ending at OOS start
            is_start_idx = max(0, oos_start - is_bars)

        is_end_idx = oos_start

        # Need minimum IS data
        if is_end_idx - is_start_idx < 100:
            oos_start = oos_end
            continue

        splits.append((
            (is_start_idx, is_end_idx),
            (oos_start, oos_end),
        ))
        oos_start = oos_end

    windows = []
    oos_equities = []

    for i, ((is_s, is_e), (oos_s, oos_e)) in enumerate(splits):
        is_df = train_df.iloc[is_s:is_e]
        oos_df = train_df.iloc[oos_s:oos_e]

        if progress_cb is not None:
            progress_cb(i + 1, len(splits), "window")

        if len(oos_df) < 5:
            continue

        # Optimize on IS with min_trades filter
        opt_result = optimize(
            strategy=strategy,
            df=is_df,
            sweep_params=sweep_params,
            metric=metric,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            freq=freq,
            execution_mode=execution_mode,
        )

        # Filter: pick best combo that has >= min_trades
        results_filtered = opt_result.results_df[
            opt_result.results_df["total_trades"] >= min_trades
        ]
        if results_filtered.empty:
            # Fall back to combo with most trades
            results_filtered = opt_result.results_df

        if metric == "max_drawdown_pct":
            best_idx = results_filtered[metric].idxmin()
        else:
            best_idx = results_filtered[metric].idxmax()

        best_row = results_filtered.loc[best_idx]
        param_names = list(sweep_params.keys())
        best = {name: best_row[name] for name in param_names}

        # Run best params on OOS via run_backtest (supports both VBT and PM paths)
        defaults = strategy.default_params()
        full_params = {**defaults, **best}

        # Clean up numpy types for strategy params
        for k, v in full_params.items():
            if hasattr(v, 'item'):
                full_params[k] = v.item()

        oos_result = run_backtest(strategy, oos_df, full_params, init_cash, fees, slippage=slippage, freq=freq, execution_mode=execution_mode)
        oos_metrics = oos_result.metrics

        # IS stats for the best params
        is_result = run_backtest(strategy, is_df, full_params, init_cash, fees, slippage=slippage, freq=freq, execution_mode=execution_mode)
        is_metrics = is_result.metrics

        is_sharpe_val = is_metrics.get("sharpe_ratio", 0.0)
        if np.isinf(is_sharpe_val):
            is_sharpe_val = 0.0
        oos_sharpe_val = oos_metrics.get("sharpe_ratio", 0.0)
        if np.isinf(oos_sharpe_val):
            oos_sharpe_val = 0.0

        # Efficiency Ratio: OOS Sharpe / IS Sharpe (matches backtest-engine)
        if is_sharpe_val > 0:
            er = oos_sharpe_val / is_sharpe_val
        else:
            er = 0.0

        window = WalkForwardWindow(
            window_idx=i,
            is_start=str(is_df.index[0]),
            is_end=str(is_df.index[-1]),
            oos_start=str(oos_df.index[0]),
            oos_end=str(oos_df.index[-1]),
            best_params=best,
            is_sharpe=min(is_sharpe_val, 99.0),
            is_return=is_metrics.get("total_return", 0.0),
            oos_sharpe=oos_sharpe_val,
            oos_return=oos_metrics.get("total_return", 0.0),
            oos_trades=int(oos_metrics.get("total_trades", 0)),
            oos_profit_factor=oos_metrics.get("profit_factor", 0.0),
            oos_win_rate=oos_metrics.get("win_rate", 0.0),
            efficiency_ratio=er,
        )
        windows.append(window)

        # Collect OOS equity, chain from previous window
        oos_equity = oos_result.equity_curve
        if oos_equities:
            prev_end = oos_equities[-1].iloc[-1]
            scale = prev_end / oos_equity.iloc[0]
            oos_equity = oos_equity * scale
        oos_equities.append(oos_equity)

    # Concatenate OOS equity curves
    oos_equity_curve = pd.concat(oos_equities) if oos_equities else pd.Series(dtype=float)

    # Full-sample single optimization for comparison
    if progress_cb is not None:
        progress_cb(len(splits), len(splits), "full_sample")
    full_opt = optimize(strategy, train_df, sweep_params, metric, init_cash, fees, slippage=slippage, freq=freq, execution_mode=execution_mode)
    # Apply min_trades filter to full optimization too
    full_filtered = full_opt.results_df[full_opt.results_df["total_trades"] >= min_trades]
    if full_filtered.empty:
        full_filtered = full_opt.results_df
    if metric == "max_drawdown_pct":
        full_best_idx = full_filtered[metric].idxmin()
    else:
        full_best_idx = full_filtered[metric].idxmax()
    full_best_row = full_filtered.loc[full_best_idx]
    full_best = {name: full_best_row[name] for name in list(sweep_params.keys())}
    full_params = {**strategy.default_params(), **full_best}
    for k, v in full_params.items():
        if hasattr(v, 'item'):
            full_params[k] = v.item()

    full_result = run_backtest(strategy, train_df, full_params, init_cash, fees, slippage=slippage, freq=freq, execution_mode=execution_mode)
    full_equity = full_result.equity_curve
    full_metrics = full_result.metrics

    # --- Robustness metrics ---

    # Efficiency Ratio aggregate
    valid_ers = [w.efficiency_ratio for w in windows if w.efficiency_ratio > 0]
    avg_er = float(np.mean(valid_ers)) if valid_ers else 0.0

    # Profitable windows
    profitable_count = sum(1 for w in windows if w.oos_return > 0)
    profitable_pct = (profitable_count / len(windows) * 100) if windows else 0.0

    # Verdict (matches backtest-engine criteria)
    er_pass = avg_er >= 0.5
    win_pass = profitable_pct >= 60
    if er_pass and win_pass:
        verdict = "PASS"
        verdict_reason = f"ER {avg_er:.2f} >= 0.5, {profitable_pct:.0f}% >= 60% windows profitable"
    else:
        reasons = []
        if not er_pass:
            reasons.append(f"ER {avg_er:.2f} < 0.5")
        if not win_pass:
            reasons.append(f"{profitable_pct:.0f}% < 60% windows profitable")
        verdict = "FAIL"
        verdict_reason = ", ".join(reasons)

    # Parameter stability: mean ± stddev per swept param
    param_stability = {}
    for pname in sweep_params.keys():
        values = []
        for w in windows:
            val = w.best_params.get(pname)
            if val is not None:
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    pass  # Skip non-numeric (categorical) params
        if values:
            param_stability[pname] = {
                "mean": float(np.mean(values)),
                "stddev": float(np.std(values)),
                "values": values,
            }

    # Summary DataFrame
    summary_data = []
    for w in windows:
        summary_data.append({
            "Window": w.window_idx,
            "IS Period": f"{w.is_start[:10]} to {w.is_end[:10]}",
            "OOS Period": f"{w.oos_start[:10]} to {w.oos_end[:10]}",
            "Best Params": str(w.best_params),
            "IS Sharpe": w.is_sharpe,
            "IS Return %": w.is_return,
            "OOS Sharpe": w.oos_sharpe,
            "OOS Return %": w.oos_return,
            "OOS Trades": w.oos_trades,
            "OOS PF": w.oos_profit_factor,
            "OOS WR %": w.oos_win_rate,
            "Efficiency Ratio": w.efficiency_ratio,
        })
    summary_df = pd.DataFrame(summary_data)

    # Compute OOS aggregate metrics
    # Use average per-window return (not compounded total, which becomes unrealistic
    # with leveraged risk-based sizing across many windows)
    oos_window_returns = [w.oos_return for w in windows]
    oos_total_return = float(np.mean(oos_window_returns)) if oos_window_returns else 0.0

    # Average OOS Sharpe (excluding inf)
    valid_sharpes = [w.oos_sharpe for w in windows if not np.isinf(w.oos_sharpe)]
    avg_oos_sharpe = float(np.mean(valid_sharpes)) if valid_sharpes else 0.0

    # Average OOS profit factor and win rate (position-size independent metrics)
    oos_pfs = [w.oos_profit_factor for w in windows if w.oos_profit_factor > 0]
    avg_oos_pf = float(np.mean(oos_pfs)) if oos_pfs else 0.0
    oos_wrs = [w.oos_win_rate for w in windows]
    avg_oos_wr = float(np.mean(oos_wrs)) if oos_wrs else 0.0

    # Hold-out backtest
    holdout_metrics = None
    holdout_equity = None
    holdout_params = None
    if holdout_df is not None and len(holdout_df) > 0 and windows:
        last_params = windows[-1].best_params
        full_holdout_params = {**strategy.default_params(), **last_params}
        for k, v in full_holdout_params.items():
            if hasattr(v, 'item'):
                full_holdout_params[k] = v.item()

        holdout_bt = run_backtest(
            strategy, holdout_df, full_holdout_params, init_cash, fees,
            slippage=slippage, freq=freq, execution_mode=execution_mode,
        )
        holdout_metrics = holdout_bt.metrics
        holdout_equity = holdout_bt.equity_curve
        holdout_params = last_params

    return WalkForwardResult(
        windows=windows,
        oos_equity_curve=oos_equity_curve,
        full_sample_equity=full_equity,
        summary_df=summary_df,
        oos_total_return=oos_total_return,
        full_sample_return=full_metrics.get("total_return", 0.0),
        oos_sharpe=avg_oos_sharpe,
        avg_oos_profit_factor=avg_oos_pf,
        avg_oos_win_rate=avg_oos_wr,
        avg_efficiency_ratio=avg_er,
        profitable_windows_pct=profitable_pct,
        verdict=verdict,
        verdict_reason=verdict_reason,
        param_stability=param_stability,
        holdout_metrics=holdout_metrics,
        holdout_equity=holdout_equity,
        holdout_params=holdout_params,
    )


def _safe(stats, key, default=0.0):
    try:
        val = stats[key]
        if pd.isna(val):
            return default
        return float(val)
    except (KeyError, IndexError):
        return default
