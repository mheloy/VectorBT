"""Extract comprehensive metrics from a VectorBT Portfolio."""

import numpy as np
import pandas as pd
import vectorbt as vbt


def extract_metrics(portfolio: vbt.Portfolio) -> dict:
    """Extract all key metrics from a Portfolio into a flat dict."""
    stats = portfolio.stats()

    # Build metrics dict from stats Series
    metrics = {}

    # Performance
    metrics["total_return"] = _get(stats, "Total Return [%]", 0.0)
    metrics["sharpe_ratio"] = _get(stats, "Sharpe Ratio", 0.0)
    metrics["sortino_ratio"] = _get(stats, "Sortino Ratio", 0.0)
    metrics["calmar_ratio"] = _get(stats, "Calmar Ratio", 0.0)
    metrics["omega_ratio"] = _get(stats, "Omega Ratio", 0.0)

    # Risk
    metrics["max_drawdown_pct"] = _get(stats, "Max Drawdown [%]", 0.0)
    metrics["max_dd_duration"] = str(_get(stats, "Max Drawdown Duration", "N/A"))

    # Trade stats
    metrics["total_trades"] = int(_get(stats, "Total Trades", 0))
    metrics["win_rate"] = _get(stats, "Win Rate [%]", 0.0)
    metrics["profit_factor"] = _get(stats, "Profit Factor", 0.0)
    metrics["expectancy"] = _get(stats, "Expectancy", 0.0)
    metrics["avg_winning_trade"] = _get(stats, "Avg Winning Trade [%]", 0.0)
    metrics["avg_losing_trade"] = _get(stats, "Avg Losing Trade [%]", 0.0)
    metrics["best_trade"] = _get(stats, "Best Trade [%]", 0.0)
    metrics["worst_trade"] = _get(stats, "Worst Trade [%]", 0.0)
    metrics["avg_trade_duration"] = str(_get(stats, "Avg Winning Trade Duration", "N/A"))

    # Cash / value
    metrics["start_value"] = _get(stats, "Start Value", 0.0)
    metrics["end_value"] = _get(stats, "End Value", 0.0)

    return metrics


def get_equity_curve(portfolio: vbt.Portfolio) -> pd.Series:
    """Get the equity curve (portfolio value over time)."""
    return portfolio.value()


def get_drawdown_series(portfolio: vbt.Portfolio) -> pd.Series:
    """Get the drawdown series (underwater curve)."""
    return portfolio.drawdown() * 100  # as percentage


def get_trades_df(portfolio: vbt.Portfolio) -> pd.DataFrame:
    """Get trades as a DataFrame."""
    trades = portfolio.trades.records_readable
    return trades


def _get(stats: pd.Series, key: str, default=None):
    """Safely get a value from stats Series."""
    try:
        val = stats[key]
        if pd.isna(val):
            return default
        return val
    except (KeyError, IndexError):
        return default
