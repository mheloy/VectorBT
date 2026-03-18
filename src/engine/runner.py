"""Backtest runner: bridges strategies to VectorBT Portfolio or custom simulator."""

import vectorbt as vbt
import pandas as pd

from src.strategies.base import BaseStrategy
from src.engine.sim_result import BacktestResult, build_simulation_result
from src.engine.simulator import simulate


def run_backtest(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    params: dict | None = None,
    init_cash: float = 10_000.0,
    fees: float = 0.0,
    sl_stop: float | None = None,
    tp_stop: float | None = None,
    freq: str | None = None,
) -> BacktestResult:
    """Run a single backtest and return a BacktestResult.

    Automatically routes to the custom simulator if the strategy provides
    a position management config, otherwise uses VBT's from_signals().

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

    # Check if strategy wants advanced position management
    pm_config = strategy.position_management(**effective_params)

    if pm_config is not None:
        # Custom simulator path
        sl_distances = strategy.compute_sl_distances(df, **effective_params)

        # Get SuperTrend line values for ST-line trailing mode
        st_values = None
        if pm_config.trail_mode == "st_line" and hasattr(strategy, 'compute_supertrend_values'):
            st_values = strategy.compute_supertrend_values(df, **effective_params)

        fixed_lot = pm_config.fixed_lot_units if pm_config.sizing_mode == "fixed_lot" else 0.0

        equity_arr, trade_records, n_trades = simulate(
            df=df,
            entries=entries,
            exits=exits,
            sl_distances=sl_distances,
            config=pm_config,
            init_cash=init_cash,
            fees=fees,
            risk_pct=pm_config.risk_pct,
            max_lot_value=pm_config.max_lot_value,
            st_values=st_values,
            fixed_lot_units=fixed_lot,
        )

        sim_result = build_simulation_result(
            equity_arr=equity_arr,
            trade_records=trade_records,
            n_trades=n_trades,
            index=df.index,
            init_cash=init_cash,
            fees=fees,
        )
        return BacktestResult(sim_result=sim_result)

    # VBT path (existing behavior)
    if sl_stop is None and tp_stop is None:
        stops = strategy.compute_stops(df, **effective_params)
        if stops is not None:
            sl_stop, tp_stop = stops

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
    return BacktestResult(portfolio=portfolio)
