"""SQLite storage for backtest results."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .models import BacktestRun, BacktestData

DB_PATH = Path(__file__).resolve().parents[2] / "results" / "backtest.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            params_json TEXT DEFAULT '{}',
            date_range_start TEXT,
            date_range_end TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            total_return REAL DEFAULT 0,
            sharpe_ratio REAL DEFAULT 0,
            sortino_ratio REAL DEFAULT 0,
            calmar_ratio REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            profit_factor REAL DEFAULT 0,
            max_drawdown_pct REAL DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            init_cash REAL DEFAULT 10000,
            fees REAL DEFAULT 0.0001,
            sl_stop REAL,
            tp_stop REAL
        );

        CREATE TABLE IF NOT EXISTS run_data (
            run_id INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
            equity_curve_json TEXT DEFAULT '[]',
            trades_json TEXT DEFAULT '[]',
            drawdown_json TEXT DEFAULT '[]',
            metrics_json TEXT DEFAULT '{}'
        );
    """)
    conn.commit()
    conn.close()


def save_run(run: BacktestRun, data: BacktestData) -> int:
    """Save a backtest run and its data. Returns the run ID."""
    init_db()
    conn = _get_conn()
    cursor = conn.execute(
        """INSERT INTO runs (
            strategy_name, timeframe, params_json, date_range_start, date_range_end,
            total_return, sharpe_ratio, sortino_ratio, calmar_ratio,
            win_rate, profit_factor, max_drawdown_pct, total_trades,
            init_cash, fees, sl_stop, tp_stop
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run.strategy_name, run.timeframe, run.params_json,
            run.date_range_start, run.date_range_end,
            run.total_return, run.sharpe_ratio, run.sortino_ratio, run.calmar_ratio,
            run.win_rate, run.profit_factor, run.max_drawdown_pct, run.total_trades,
            run.init_cash, run.fees, run.sl_stop, run.tp_stop,
        ),
    )
    run_id = cursor.lastrowid

    conn.execute(
        """INSERT INTO run_data (run_id, equity_curve_json, trades_json, drawdown_json, metrics_json)
        VALUES (?, ?, ?, ?, ?)""",
        (run_id, data.equity_curve_json, data.trades_json, data.drawdown_json, data.metrics_json),
    )
    conn.commit()
    conn.close()
    return run_id


def list_runs() -> list[dict]:
    """List all saved runs as dicts."""
    init_db()
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run(run_id: int) -> dict | None:
    """Get a single run by ID."""
    init_db()
    conn = _get_conn()
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_run_data(run_id: int) -> dict | None:
    """Get detailed data for a run."""
    init_db()
    conn = _get_conn()
    row = conn.execute("SELECT * FROM run_data WHERE run_id = ?", (run_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_run(run_id: int):
    """Delete a run and its data."""
    init_db()
    conn = _get_conn()
    conn.execute("DELETE FROM run_data WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()
