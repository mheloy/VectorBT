"""Backtest runner: bridges strategies to VectorBT Portfolio."""

import vectorbt as vbt
import pandas as pd

from src.strategies.base import BaseStrategy


def run_backtest(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    params: dict | None = None,
    init_cash: float = 10_000.0,
    fees: float = 0.0001,
    sl_stop: float | None = None,
    tp_stop: float | None = None,
    freq: str | None = None,
) -> vbt.Portfolio:
    """Run a single backtest and return a VectorBT Portfolio.

    Args:
        strategy: Strategy instance to generate signals.
        df: OHLCV DataFrame with datetime index.
        params: Strategy parameter overrides. Uses defaults if None.
        init_cash: Starting cash.
        fees: Fee per trade as fraction (0.0001 = 1bp).
        sl_stop: Stop-loss as fraction of entry price (e.g., 0.02 = 2%).
        tp_stop: Take-profit as fraction of entry price.
        freq: Data frequency string (e.g., '5min', '1h'). Auto-detected if None.
    """
    effective_params = strategy.default_params()
    if params:
        effective_params.update(params)

    entries, exits = strategy.generate_signals(df, **effective_params)

    pf_kwargs = dict(
        close=df["close"],
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
    )

    if freq:
        pf_kwargs["freq"] = freq
    if sl_stop is not None:
        pf_kwargs["sl_stop"] = sl_stop
    if tp_stop is not None:
        pf_kwargs["tp_stop"] = tp_stop

    portfolio = vbt.Portfolio.from_signals(**pf_kwargs)
    return portfolio
