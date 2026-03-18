"""Dataclass models for backtest results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class BacktestRun:
    """A saved backtest run."""
    id: int | None = None
    strategy_name: str = ""
    timeframe: str = ""
    params_json: str = "{}"
    date_range_start: str = ""
    date_range_end: str = ""
    created_at: str = ""
    # Key metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    init_cash: float = 10000.0
    fees: float = 0.0
    sl_stop: float | None = None
    tp_stop: float | None = None


@dataclass
class BacktestData:
    """Detailed data for a saved backtest run."""
    run_id: int = 0
    equity_curve_json: str = "[]"
    trades_json: str = "[]"
    drawdown_json: str = "[]"
    metrics_json: str = "{}"
