"""Monte Carlo simulation via trade shuffle."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo trade shuffle simulation."""
    n_simulations: int
    equity_curves: np.ndarray  # (n_simulations, n_trades+1) - each row is a sim
    original_equity: np.ndarray  # Original equity curve
    trade_pnls: np.ndarray  # Original trade PnLs
    # Percentile equity curves
    p5: np.ndarray
    p25: np.ndarray
    p50: np.ndarray
    p75: np.ndarray
    p95: np.ndarray
    # Max drawdown distribution
    max_drawdowns: np.ndarray  # (n_simulations,)
    # Final equity distribution
    final_equities: np.ndarray  # (n_simulations,)
    # Ruin probability
    ruin_threshold: float
    ruin_probability: float
    # Stats
    original_max_dd: float
    median_max_dd: float
    worst_case_dd_95: float  # 95th percentile DD
    original_final_equity: float


def run_monte_carlo(
    portfolio,
    n_simulations: int = 1000,
    ruin_threshold_pct: float = 50.0,
    seed: int | None = 42,
) -> MonteCarloResult:
    """Run Monte Carlo trade shuffle simulation.

    Takes the actual trade PnLs from a backtest, shuffles them randomly
    N times, and builds alternative equity curves to assess robustness.

    Args:
        portfolio: VectorBT Portfolio object.
        n_simulations: Number of random shuffles.
        ruin_threshold_pct: Account loss % considered "ruin" (e.g., 50 = 50% loss).
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Extract trade PnLs
    trades = portfolio.trades.records_readable
    if trades.empty or "PnL" not in trades.columns:
        raise ValueError("No trades found in portfolio or PnL column missing")

    trade_pnls = trades["PnL"].values
    n_trades = len(trade_pnls)
    init_cash = portfolio.init_cash

    if isinstance(init_cash, pd.Series):
        init_cash = init_cash.iloc[0]
    init_cash = float(init_cash)

    # Original equity curve from trades
    original_equity = np.concatenate([[init_cash], init_cash + np.cumsum(trade_pnls)])

    # Simulate shuffled equity curves
    equity_curves = np.zeros((n_simulations, n_trades + 1))
    equity_curves[:, 0] = init_cash

    for i in range(n_simulations):
        shuffled = rng.permutation(trade_pnls)
        equity_curves[i, 1:] = init_cash + np.cumsum(shuffled)

    # Percentiles at each trade step
    p5 = np.percentile(equity_curves, 5, axis=0)
    p25 = np.percentile(equity_curves, 25, axis=0)
    p50 = np.percentile(equity_curves, 50, axis=0)
    p75 = np.percentile(equity_curves, 75, axis=0)
    p95 = np.percentile(equity_curves, 95, axis=0)

    # Max drawdown for each simulation
    max_drawdowns = np.zeros(n_simulations)
    for i in range(n_simulations):
        curve = equity_curves[i]
        running_max = np.maximum.accumulate(curve)
        drawdowns = (running_max - curve) / running_max * 100
        max_drawdowns[i] = np.max(drawdowns)

    # Original max drawdown
    orig_running_max = np.maximum.accumulate(original_equity)
    orig_drawdowns = (orig_running_max - original_equity) / orig_running_max * 100
    original_max_dd = np.max(orig_drawdowns)

    # Final equities
    final_equities = equity_curves[:, -1]

    # Ruin probability
    ruin_level = init_cash * (1 - ruin_threshold_pct / 100)
    min_equities = np.min(equity_curves, axis=1)
    ruin_probability = np.mean(min_equities <= ruin_level) * 100

    return MonteCarloResult(
        n_simulations=n_simulations,
        equity_curves=equity_curves,
        original_equity=original_equity,
        trade_pnls=trade_pnls,
        p5=p5, p25=p25, p50=p50, p75=p75, p95=p95,
        max_drawdowns=max_drawdowns,
        final_equities=final_equities,
        ruin_threshold=ruin_threshold_pct,
        ruin_probability=ruin_probability,
        original_max_dd=original_max_dd,
        median_max_dd=float(np.median(max_drawdowns)),
        worst_case_dd_95=float(np.percentile(max_drawdowns, 95)),
        original_final_equity=float(original_equity[-1]),
    )
