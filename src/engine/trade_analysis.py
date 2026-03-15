"""Deep trade analysis: R-multiples, MAE/MFE, streaks, sessions, exposure."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt


@dataclass
class TradeAnalysisResult:
    """Result of deep trade analysis."""
    # R-multiples
    r_multiples: pd.Series | None  # PnL / risk_unit per trade
    r_stats: dict  # mean_r, median_r, std_r, expectancy_r
    # MAE/MFE
    mae_mfe_df: pd.DataFrame | None  # per-trade MAE, MFE, final PnL
    # Streaks
    streaks_df: pd.DataFrame  # streak analysis
    max_win_streak: int
    max_loss_streak: int
    # Session performance
    session_df: pd.DataFrame | None  # per-session stats
    # Duration vs PnL
    duration_pnl_df: pd.DataFrame | None
    # Exposure
    exposure_pct: float  # % time in market
    long_pct: float
    short_pct: float


def analyze_trades(
    portfolio: vbt.Portfolio,
    df: pd.DataFrame,
    risk_unit: float | None = None,
) -> TradeAnalysisResult:
    """Run comprehensive trade analysis.

    Args:
        portfolio: VectorBT Portfolio from backtest.
        df: OHLCV DataFrame used in backtest.
        risk_unit: Dollar risk per trade for R-multiple calc.
            If None, uses ATR(14) * close as fallback.
    """
    trades = portfolio.trades.records_readable
    if trades.empty:
        return _empty_result()

    # --- R-Multiples ---
    r_multiples = None
    r_stats = {"mean_r": 0, "median_r": 0, "std_r": 0, "expectancy_r": 0}

    if risk_unit is None:
        # Use average ATR as risk unit
        atr = _calc_atr(df, period=14)
        risk_unit = atr.mean() if not atr.empty else 1.0

    if risk_unit > 0 and "PnL" in trades.columns:
        r_multiples = trades["PnL"] / risk_unit
        r_stats = {
            "mean_r": float(r_multiples.mean()),
            "median_r": float(r_multiples.median()),
            "std_r": float(r_multiples.std()),
            "expectancy_r": float(r_multiples.mean()),
            "positive_r_pct": float((r_multiples > 0).mean() * 100),
            "gt_2r_pct": float((r_multiples > 2).mean() * 100),
        }

    # --- MAE/MFE ---
    mae_mfe_df = _calc_mae_mfe(portfolio, df)

    # --- Streaks ---
    streaks_df, max_win, max_loss = _calc_streaks(trades)

    # --- Session Performance ---
    session_df = _calc_session_performance(trades)

    # --- Duration vs PnL ---
    duration_pnl_df = None
    if "Entry Timestamp" in trades.columns and "Exit Timestamp" in trades.columns:
        durations = pd.to_datetime(trades["Exit Timestamp"]) - pd.to_datetime(trades["Entry Timestamp"])
        duration_pnl_df = pd.DataFrame({
            "duration_hours": durations.dt.total_seconds() / 3600,
            "pnl": trades["PnL"].values,
            "return_pct": trades["Return"].values * 100 if "Return" in trades.columns else 0,
        })

    # --- Exposure ---
    positions = portfolio.asset_value()
    in_market = (positions.abs() > 0).mean() * 100
    long_time = (positions > 0).mean() * 100
    short_time = (positions < 0).mean() * 100

    return TradeAnalysisResult(
        r_multiples=r_multiples,
        r_stats=r_stats,
        mae_mfe_df=mae_mfe_df,
        streaks_df=streaks_df,
        max_win_streak=max_win,
        max_loss_streak=max_loss,
        session_df=session_df,
        duration_pnl_df=duration_pnl_df,
        exposure_pct=float(in_market),
        long_pct=float(long_time),
        short_pct=float(short_time),
    )


def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean().dropna()


def _calc_mae_mfe(portfolio: vbt.Portfolio, df: pd.DataFrame) -> pd.DataFrame | None:
    """Calculate Maximum Adverse/Favorable Excursion per trade."""
    trades = portfolio.trades.records_readable
    if trades.empty:
        return None

    results = []
    for _, trade in trades.iterrows():
        entry_ts = trade.get("Entry Timestamp")
        exit_ts = trade.get("Exit Timestamp")
        # VectorBT uses "Avg Entry Price"
        entry_price = trade.get("Avg Entry Price", trade.get("Entry Price", 0))

        if pd.isna(entry_ts) or pd.isna(exit_ts) or entry_price == 0:
            continue

        # Get price data during trade
        trade_data = df.loc[entry_ts:exit_ts]

        if trade_data.empty or len(trade_data) < 1:
            continue

        direction = trade.get("Direction", "Long")

        if direction == "Long":
            mae = (entry_price - trade_data["low"].min()) / entry_price * 100
            mfe = (trade_data["high"].max() - entry_price) / entry_price * 100
        else:  # Short
            mae = (trade_data["high"].max() - entry_price) / entry_price * 100
            mfe = (entry_price - trade_data["low"].min()) / entry_price * 100

        results.append({
            "mae_pct": mae,
            "mfe_pct": mfe,
            "pnl": trade.get("PnL", 0),
            "return_pct": trade.get("Return", 0) * 100,
        })

    return pd.DataFrame(results) if results else None


def _calc_streaks(trades: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Calculate win/loss streaks."""
    if "PnL" not in trades.columns or trades.empty:
        return pd.DataFrame(), 0, 0

    wins = (trades["PnL"] > 0).astype(int)

    # Calculate streaks
    streak_groups = (wins != wins.shift()).cumsum()
    streaks = wins.groupby(streak_groups).agg(["first", "count"])
    streaks.columns = ["is_win", "length"]

    win_streaks = streaks[streaks["is_win"] == 1]["length"]
    loss_streaks = streaks[streaks["is_win"] == 0]["length"]

    max_win = int(win_streaks.max()) if not win_streaks.empty else 0
    max_loss = int(loss_streaks.max()) if not loss_streaks.empty else 0

    summary = pd.DataFrame({
        "Metric": [
            "Max Win Streak", "Avg Win Streak",
            "Max Loss Streak", "Avg Loss Streak",
            "Total Win Streaks", "Total Loss Streaks",
        ],
        "Value": [
            max_win,
            f"{win_streaks.mean():.1f}" if not win_streaks.empty else "0",
            max_loss,
            f"{loss_streaks.mean():.1f}" if not loss_streaks.empty else "0",
            len(win_streaks),
            len(loss_streaks),
        ],
    })

    return summary, max_win, max_loss


def _calc_session_performance(trades: pd.DataFrame) -> pd.DataFrame | None:
    """Classify trades by Gold trading session and compute per-session stats.

    Sessions (GMT):
        Asian:   23:00 - 07:00
        London:  08:00 - 16:00
        NY:      13:00 - 21:00
        Overlap: 13:00 - 16:00 (London-NY)
    """
    if "Entry Timestamp" not in trades.columns or "PnL" not in trades.columns:
        return None

    entry_hours = pd.to_datetime(trades["Entry Timestamp"]).dt.hour

    def classify_session(hour):
        if 23 <= hour or hour < 7:
            return "Asian"
        elif 8 <= hour < 13:
            return "London"
        elif 13 <= hour < 16:
            return "London-NY Overlap"
        elif 16 <= hour < 21:
            return "New York"
        else:
            return "Off-Hours"

    sessions = entry_hours.apply(classify_session)
    trades_with_session = trades.copy()
    trades_with_session["session"] = sessions

    session_stats = trades_with_session.groupby("session").agg(
        trades=("PnL", "count"),
        total_pnl=("PnL", "sum"),
        avg_pnl=("PnL", "mean"),
        win_rate=("PnL", lambda x: (x > 0).mean() * 100),
        avg_winner=("PnL", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        avg_loser=("PnL", lambda x: x[x < 0].mean() if (x < 0).any() else 0),
    ).round(2)

    return session_stats.reset_index()


def _empty_result() -> TradeAnalysisResult:
    return TradeAnalysisResult(
        r_multiples=None, r_stats={}, mae_mfe_df=None,
        streaks_df=pd.DataFrame(), max_win_streak=0, max_loss_streak=0,
        session_df=None, duration_pnl_df=None,
        exposure_pct=0, long_pct=0, short_pct=0,
    )
