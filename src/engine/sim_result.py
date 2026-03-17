"""Simulation result wrapper and metric computation.

Provides a SimulationResult that computes the same metrics as VBT's Portfolio,
plus a BacktestResult that unifies both VBT and custom simulator paths.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .simulator import EXIT_TYPE_LABELS, TR_ENTRY_BAR, TR_EXIT_BAR, TR_ENTRY_PRICE, \
    TR_EXIT_PRICE, TR_FRACTION, TR_PNL, TR_EXIT_TYPE, TR_DIRECTION


@dataclass
class SimulationResult:
    """Result from the custom bar-by-bar simulator."""

    equity_curve: pd.Series
    drawdown_series: pd.Series
    trades_df: pd.DataFrame
    metrics: dict
    init_cash: float
    fees: float


def build_simulation_result(
    equity_arr: np.ndarray,
    trade_records: np.ndarray,
    n_trades: int,
    index: pd.DatetimeIndex,
    init_cash: float,
    fees: float,
) -> SimulationResult:
    """Convert raw simulator output into a SimulationResult.

    Args:
        equity_arr: Equity values per bar.
        trade_records: 2D array of trade records from simulator.
        n_trades: Number of actual trades.
        index: DatetimeIndex from the original DataFrame.
        init_cash: Starting capital.
        fees: Fee fraction.
    """
    equity = pd.Series(equity_arr, index=index, name="equity")

    # Drawdown series
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100  # As percentage
    drawdown = drawdown.fillna(0.0)

    # Build trades DataFrame
    trades_df = _build_trades_df(trade_records, n_trades, index)

    # Compute metrics
    metrics = compute_metrics(equity, trades_df, init_cash)

    return SimulationResult(
        equity_curve=equity,
        drawdown_series=drawdown,
        trades_df=trades_df,
        metrics=metrics,
        init_cash=init_cash,
        fees=fees,
    )


def _build_trades_df(
    trade_records: np.ndarray,
    n_trades: int,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Convert raw trade records array to a DataFrame."""
    if n_trades == 0:
        return pd.DataFrame(columns=[
            "Entry Timestamp", "Exit Timestamp", "Entry Price", "Exit Price",
            "Position %", "PnL", "Exit Type", "Direction", "Return %",
        ])

    records = trade_records[:n_trades]
    entry_bars = records[:, TR_ENTRY_BAR].astype(int)
    exit_bars = records[:, TR_EXIT_BAR].astype(int)

    df = pd.DataFrame({
        "Entry Timestamp": index[entry_bars],
        "Exit Timestamp": index[exit_bars],
        "Entry Price": records[:, TR_ENTRY_PRICE],
        "Exit Price": records[:, TR_EXIT_PRICE],
        "Position %": records[:, TR_FRACTION] * 100,
        "PnL": records[:, TR_PNL],
        "Exit Type": [EXIT_TYPE_LABELS.get(int(t), "Unknown") for t in records[:, TR_EXIT_TYPE]],
        "Direction": ["Long" if d == 1 else "Short" for d in records[:, TR_DIRECTION]],
    })

    # Return % relative to allocated capital per fraction
    df["Return %"] = np.where(
        records[:, TR_ENTRY_PRICE] > 0,
        (records[:, TR_EXIT_PRICE] - records[:, TR_ENTRY_PRICE])
        / records[:, TR_ENTRY_PRICE]
        * np.sign(records[:, TR_DIRECTION])
        * 100,
        0.0,
    )

    return df


def compute_metrics(
    equity: pd.Series,
    trades_df: pd.DataFrame,
    init_cash: float,
) -> dict:
    """Compute comprehensive metrics from equity curve and trades.

    Returns dict with same keys as engine/metrics.py extract_metrics().
    """
    metrics = {}

    # Performance
    end_value = equity.iloc[-1] if len(equity) > 0 else init_cash
    total_return = (end_value / init_cash - 1) * 100
    metrics["total_return"] = total_return
    metrics["start_value"] = init_cash
    metrics["end_value"] = end_value

    # Returns series for ratio calculations
    returns = equity.pct_change().dropna()

    # Sharpe ratio (annualized, assuming 5min bars → 252*288 bars/year for 5M)
    # Use generic approach: annualize based on number of bars
    n_bars = len(returns)
    if n_bars > 1 and returns.std() > 0:
        # Estimate bars per year from data duration
        if hasattr(equity.index, 'freq') and equity.index.freq is not None:
            bars_per_year = _estimate_bars_per_year(equity.index)
        else:
            # Estimate from data span
            duration_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
            if duration_days > 0:
                bars_per_year = n_bars / duration_days * 252
            else:
                bars_per_year = 252 * 288  # Default 5M

        ann_factor = np.sqrt(bars_per_year)
        metrics["sharpe_ratio"] = float(returns.mean() / returns.std() * ann_factor)

        # Sortino (downside deviation)
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            metrics["sortino_ratio"] = float(returns.mean() / downside.std() * ann_factor)
        else:
            metrics["sortino_ratio"] = 0.0
    else:
        metrics["sharpe_ratio"] = 0.0
        metrics["sortino_ratio"] = 0.0

    # Calmar ratio
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd_pct = abs(drawdown.min()) * 100
    metrics["max_drawdown_pct"] = max_dd_pct

    if max_dd_pct > 0:
        metrics["calmar_ratio"] = total_return / max_dd_pct
    else:
        metrics["calmar_ratio"] = 0.0

    # Omega ratio (threshold = 0)
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    metrics["omega_ratio"] = float(gains / losses) if losses > 0 else 0.0

    # Max drawdown duration
    in_dd = drawdown < 0
    if in_dd.any():
        dd_groups = (~in_dd).cumsum()
        dd_durations = in_dd.groupby(dd_groups).sum()
        max_dd_dur = int(dd_durations.max()) if len(dd_durations) > 0 else 0
        metrics["max_dd_duration"] = str(max_dd_dur) + " bars"
    else:
        metrics["max_dd_duration"] = "0 bars"

    # Trade statistics
    if len(trades_df) > 0:
        # Group partial closes by entry bar to get "logical" trades
        pnl_by_trade = trades_df.groupby("Entry Timestamp")["PnL"].sum()
        n_logical_trades = len(pnl_by_trade)
        winners = pnl_by_trade[pnl_by_trade > 0]
        losers = pnl_by_trade[pnl_by_trade <= 0]

        metrics["total_trades"] = n_logical_trades
        metrics["win_rate"] = len(winners) / n_logical_trades * 100 if n_logical_trades > 0 else 0.0

        total_wins = winners.sum() if len(winners) > 0 else 0.0
        total_losses = abs(losers.sum()) if len(losers) > 0 else 0.0
        metrics["profit_factor"] = float(total_wins / total_losses) if total_losses > 0 else 0.0

        metrics["expectancy"] = float(pnl_by_trade.mean()) if n_logical_trades > 0 else 0.0

        # Avg winning/losing trade as %
        if len(winners) > 0:
            metrics["avg_winning_trade"] = float(winners.mean() / init_cash * 100)
        else:
            metrics["avg_winning_trade"] = 0.0
        if len(losers) > 0:
            metrics["avg_losing_trade"] = float(losers.mean() / init_cash * 100)
        else:
            metrics["avg_losing_trade"] = 0.0

        metrics["best_trade"] = float(pnl_by_trade.max() / init_cash * 100)
        metrics["worst_trade"] = float(pnl_by_trade.min() / init_cash * 100)

        # Avg trade duration
        durations = trades_df.groupby("Entry Timestamp").agg(
            entry=("Entry Timestamp", "first"),
            exit=("Exit Timestamp", "last"),
        )
        avg_dur = (durations["exit"] - durations["entry"]).mean()
        metrics["avg_trade_duration"] = str(avg_dur)

        # Exit type breakdown (unique to simulator)
        exit_counts = trades_df["Exit Type"].value_counts().to_dict()
        metrics["exit_type_breakdown"] = exit_counts
    else:
        metrics["total_trades"] = 0
        metrics["win_rate"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["expectancy"] = 0.0
        metrics["avg_winning_trade"] = 0.0
        metrics["avg_losing_trade"] = 0.0
        metrics["best_trade"] = 0.0
        metrics["worst_trade"] = 0.0
        metrics["avg_trade_duration"] = "N/A"
        metrics["exit_type_breakdown"] = {}

    return metrics


def _estimate_bars_per_year(index: pd.DatetimeIndex) -> float:
    """Estimate number of bars per year from a DatetimeIndex."""
    if len(index) < 2:
        return 252 * 288  # Default 5M
    median_gap = pd.Series(index).diff().dropna().median()
    seconds = median_gap.total_seconds()
    if seconds <= 0:
        return 252 * 288
    # Trading year ≈ 252 days × 24h (forex)
    bars_per_day = 86400 / seconds
    return bars_per_day * 252


class BacktestResult:
    """Unified result wrapper for both VBT Portfolio and custom simulator paths.

    Provides a consistent interface regardless of which execution path was used.
    """

    def __init__(self, portfolio=None, sim_result: SimulationResult | None = None):
        self._portfolio = portfolio
        self._sim_result = sim_result
        self._cached_metrics = None

    @property
    def is_simulator(self) -> bool:
        """True if result came from custom simulator (not VBT)."""
        return self._sim_result is not None

    @property
    def portfolio(self):
        """Access raw VBT Portfolio (None if simulator path)."""
        return self._portfolio

    @property
    def equity_curve(self) -> pd.Series:
        if self._sim_result:
            return self._sim_result.equity_curve
        return self._portfolio.value()

    @property
    def drawdown_series(self) -> pd.Series:
        if self._sim_result:
            return self._sim_result.drawdown_series
        return self._portfolio.drawdown() * 100

    @property
    def trades_df(self) -> pd.DataFrame:
        if self._sim_result:
            return self._sim_result.trades_df
        return self._portfolio.trades.records_readable

    @property
    def metrics(self) -> dict:
        if self._cached_metrics is not None:
            return self._cached_metrics

        if self._sim_result:
            self._cached_metrics = self._sim_result.metrics
        else:
            from .metrics import extract_metrics
            self._cached_metrics = extract_metrics(self._portfolio)

        return self._cached_metrics

    @property
    def init_cash(self) -> float:
        if self._sim_result:
            return self._sim_result.init_cash
        return float(self._portfolio.init_cash)

    @property
    def trade_pnls(self) -> pd.Series:
        """Get PnL per logical trade (grouped by entry for simulator)."""
        if self._sim_result:
            df = self._sim_result.trades_df
            if len(df) == 0:
                return pd.Series(dtype=float)
            return df.groupby("Entry Timestamp")["PnL"].sum()
        trades = self._portfolio.trades.records_readable
        if len(trades) == 0:
            return pd.Series(dtype=float)
        return trades["PnL"].reset_index(drop=True)
