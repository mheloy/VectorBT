"""Robustness testing: noise injection, signal delay, parameter sensitivity."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.strategies.base import BaseStrategy
from src.engine.runner import run_backtest


@dataclass
class RobustnessResult:
    """Result of robustness tests."""
    # Signal delay results
    delay_results: pd.DataFrame | None  # delay_bars, sharpe, return, win_rate, pf, trades
    # Noise injection results
    noise_results: pd.DataFrame | None  # noise_pct, mean_sharpe, std_sharpe, mean_return, etc.
    # Parameter sensitivity
    sensitivity_results: pd.DataFrame | None  # param perturbation results


def test_signal_delay(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    params: dict,
    delays: list[int] | None = None,
    init_cash: float = 10_000.0,
    fees: float = 0.0001,
    freq: str | None = None,
) -> pd.DataFrame:
    """Test strategy with delayed entry/exit signals.

    Args:
        delays: List of bar delays to test. Default [0, 1, 2, 3, 5].
    """
    if delays is None:
        delays = [0, 1, 2, 3, 5]

    results = []

    for delay in delays:
        delayed_params = params.copy()
        # For delay testing, we run full backtest with modified signals
        result = run_backtest(strategy, df, delayed_params, init_cash, fees, freq=freq)

        if delay > 0:
            # Re-run with delayed signals (only for VBT path, PM path handles internally)
            entries, exits = strategy.generate_signals(df, **params)
            delayed_entries = entries.shift(delay).fillna(False).astype(bool)
            delayed_exits = exits.shift(delay).fillna(False).astype(bool)

            pm_config = strategy.position_management(**params)
            if pm_config is not None:
                from src.engine.simulator import simulate
                from src.engine.sim_result import build_simulation_result
                sl_distances = strategy.compute_sl_distances(df, **params)
                eq, tr, nt = simulate(df, delayed_entries, delayed_exits, sl_distances, pm_config, init_cash, fees)
                sim_result = build_simulation_result(eq, tr, nt, df.index, init_cash, fees)
                from src.engine.sim_result import BacktestResult
                result = BacktestResult(sim_result=sim_result)
            else:
                pf_kwargs = dict(close=df["close"], entries=delayed_entries, exits=delayed_exits, init_cash=init_cash, fees=fees)
                if freq:
                    pf_kwargs["freq"] = freq
                pf = vbt.Portfolio.from_signals(**pf_kwargs)
                from src.engine.sim_result import BacktestResult
                result = BacktestResult(portfolio=pf)

        metrics = result.metrics
        results.append({
            "delay_bars": delay,
            "total_return": metrics.get("total_return", 0.0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "max_drawdown": metrics.get("max_drawdown_pct", 0.0),
            "total_trades": int(metrics.get("total_trades", 0)),
        })

    return pd.DataFrame(results)


def test_noise_injection(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    params: dict,
    noise_levels: list[float] | None = None,
    n_trials: int = 20,
    init_cash: float = 10_000.0,
    fees: float = 0.0001,
    freq: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Test strategy with noise injected into price data.

    Args:
        noise_levels: List of noise % to test. Default [0.01, 0.05, 0.1, 0.2, 0.5].
        n_trials: Number of noisy runs per noise level.
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]

    rng = np.random.default_rng(seed)
    results = []

    for noise_pct in noise_levels:
        trial_sharpes = []
        trial_returns = []
        trial_dds = []

        for trial in range(n_trials):
            noisy_df = df.copy()
            noise = 1 + rng.normal(0, noise_pct / 100, size=len(df))
            for col in ["open", "high", "low", "close"]:
                noisy_df[col] = noisy_df[col] * noise

            result = run_backtest(strategy, noisy_df, params, init_cash, fees, freq=freq)
            metrics = result.metrics
            trial_sharpes.append(metrics.get("sharpe_ratio", 0.0))
            trial_returns.append(metrics.get("total_return", 0.0))
            trial_dds.append(metrics.get("max_drawdown_pct", 0.0))

        results.append({
            "noise_pct": noise_pct,
            "mean_sharpe": np.mean(trial_sharpes),
            "std_sharpe": np.std(trial_sharpes),
            "mean_return": np.mean(trial_returns),
            "std_return": np.std(trial_returns),
            "mean_max_dd": np.mean(trial_dds),
            "std_max_dd": np.std(trial_dds),
        })

    return pd.DataFrame(results)


def test_param_sensitivity(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    base_params: dict,
    param_name: str,
    perturbations: list[float] | None = None,
    init_cash: float = 10_000.0,
    fees: float = 0.0001,
    freq: str | None = None,
) -> pd.DataFrame:
    """Test how sensitive the strategy is to small changes in a parameter.

    Args:
        param_name: The parameter to perturb.
        perturbations: Fractional perturbations. Default [-0.2, -0.1, 0, 0.1, 0.2].
    """
    if perturbations is None:
        perturbations = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]

    base_value = base_params[param_name]
    results = []

    for pct in perturbations:
        perturbed_value = base_value * (1 + pct)
        if isinstance(base_value, int):
            perturbed_value = max(1, round(perturbed_value))

        test_params = base_params.copy()
        test_params[param_name] = perturbed_value

        result = run_backtest(strategy, df, test_params, init_cash, fees, freq=freq)
        metrics = result.metrics

        results.append({
            "perturbation": f"{pct:+.0%}",
            "param_value": perturbed_value,
            "total_return": metrics.get("total_return", 0.0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "max_drawdown": metrics.get("max_drawdown_pct", 0.0),
            "total_trades": int(metrics.get("total_trades", 0)),
        })

    return pd.DataFrame(results)


def _safe(stats, key, default=0.0):
    try:
        val = stats[key]
        if pd.isna(val):
            return default
        return float(val)
    except (KeyError, IndexError):
        return default
