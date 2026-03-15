"""SuperTrend strategy with optional H1 timeframe confirmation filter.

Ported from live MT5 bot at /home/mheloy/SuperTrendMT5Linux/.
Indicator logic matches PineScript's ta.supertrend() convention:
  direction = -1  -> price ABOVE SuperTrend line -> UPTREND  -> BUY signal
  direction = +1  -> price BELOW SuperTrend line -> DOWNTREND -> SELL signal
"""

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyParam


# ---------------------------------------------------------------------------
# SuperTrend indicator helpers (ported from bot/core/indicators.py)
# ---------------------------------------------------------------------------


def _price_source(df: pd.DataFrame, source: str = "hl2") -> pd.Series:
    """Compute price source series from OHLC data."""
    if source == "close":
        return df["close"].copy()
    elif source == "hl2":
        return (df["high"] + df["low"]) / 2.0
    elif source == "hlc3":
        return (df["high"] + df["low"] + df["close"]) / 3.0
    elif source == "ohlc4":
        return (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    else:
        raise ValueError(f"Unknown source: {source}")


def _true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate True Range."""
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return tr


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA) matching PineScript's ta.rma().

    Seeded with SMA of first `period` values, then:
        rma = alpha * value + (1 - alpha) * rma_prev
    where alpha = 1 / period.
    """
    values = series.values
    n = len(values)
    result = np.full(n, np.nan)
    alpha = 1.0 / period

    # Find first window of `period` non-NaN values for SMA seed
    count = 0
    seed_end = -1
    for i in range(n):
        if not np.isnan(values[i]):
            count += 1
            if count == period:
                seed_end = i
                break
        else:
            count = 0

    if seed_end < 0:
        return pd.Series(result, index=series.index)

    # Seed with SMA
    result[seed_end] = np.mean(values[seed_end - period + 1 : seed_end + 1])

    # RMA forward
    for i in range(seed_end + 1, n):
        if np.isnan(values[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR using Wilder's RMA smoothing."""
    return _rma(_true_range(df), period)


def calc_supertrend(
    df: pd.DataFrame,
    period: int = 16,
    factor: float = 1.4,
    source: str = "hl2",
) -> pd.DataFrame:
    """Calculate SuperTrend indicator.

    Returns DataFrame with columns: supertrend, direction
    direction: -1 = uptrend (BUY), +1 = downtrend (SELL)
    """
    src = _price_source(df, source)
    spread = _atr(df, period)

    upper_band = src + factor * spread
    lower_band = src - factor * spread

    n = len(df)
    close = df["close"].values
    direction = np.zeros(n, dtype=np.int8)
    supertrend = np.full(n, np.nan, dtype=np.float64)

    final_upper = upper_band.values.copy()
    final_lower = lower_band.values.copy()

    # Find first valid index (where spread is not NaN)
    first_valid = 0
    for i in range(n):
        if not np.isnan(final_upper[i]) and not np.isnan(final_lower[i]):
            first_valid = i
            break

    # Band ratcheting
    for i in range(first_valid + 1, n):
        prev_upper = final_upper[i - 1]
        prev_lower = final_lower[i - 1]
        prev_close = close[i - 1]

        # Upper band only ratchets down
        if not np.isnan(prev_upper):
            if not (final_upper[i] < prev_upper or prev_close > prev_upper):
                final_upper[i] = prev_upper

        # Lower band only ratchets up
        if not np.isnan(prev_lower):
            if not (final_lower[i] > prev_lower or prev_close < prev_lower):
                final_lower[i] = prev_lower

    # Determine direction and SuperTrend value
    for i in range(first_valid, n):
        if i == first_valid:
            direction[i] = 1  # default downtrend (matches PineScript)
            supertrend[i] = final_upper[i]
            continue

        prev_dir = direction[i - 1]

        if prev_dir == -1:  # was uptrend
            if close[i] < final_lower[i]:
                direction[i] = 1  # flip to downtrend
                supertrend[i] = final_upper[i]
            else:
                direction[i] = -1  # stay uptrend
                supertrend[i] = final_lower[i]
        else:  # was downtrend
            if close[i] > final_upper[i]:
                direction[i] = -1  # flip to uptrend
                supertrend[i] = final_lower[i]
            else:
                direction[i] = 1  # stay downtrend
                supertrend[i] = final_upper[i]

    return pd.DataFrame(
        {"supertrend": supertrend, "direction": direction},
        index=df.index,
    )


# ---------------------------------------------------------------------------
# H1 filter helper
# ---------------------------------------------------------------------------


def _h1_direction(df: pd.DataFrame, period: int, factor: float) -> pd.Series:
    """Resample to H1, compute SuperTrend, forward-fill direction to original index.

    Shifts H1 direction by 1 bar before forward-filling to avoid look-ahead bias
    (only act on completed H1 candles).
    """
    h1 = df.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna(subset=["open"])

    if len(h1) < period + 2:
        # Not enough H1 data — return neutral (no filtering)
        return pd.Series(0, index=df.index, dtype=np.int8)

    st = calc_supertrend(h1, period, factor)
    # Shift by 1 so current hour uses previous hour's completed direction
    h1_dir = st["direction"].shift(1)
    return h1_dir.reindex(df.index, method="ffill").fillna(0).astype(np.int8)


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


class SuperTrendStrategy(BaseStrategy):
    """SuperTrend trend-following strategy.

    Enters long on uptrend flip (direction changes from +1 to -1).
    Exits on downtrend flip (direction changes from -1 to +1).
    Optional H1 SuperTrend filter confirms trend direction.
    """

    @property
    def name(self) -> str:
        return "SuperTrend"

    def parameters(self) -> list[StrategyParam]:
        return [
            StrategyParam(
                "period", default=16, min_val=5, max_val=50, step=1,
                description="ATR period",
            ),
            StrategyParam(
                "factor", default=1.4, min_val=0.5, max_val=5.0, step=0.1,
                description="ATR multiplier",
            ),
            StrategyParam(
                "source", default="hl2",
                choices=["hl2", "close", "hlc3", "ohlc4"],
                description="Price source",
            ),
            StrategyParam(
                "h1_filter", default="On",
                choices=["On", "Off"],
                description="H1 SuperTrend filter",
            ),
            StrategyParam(
                "h1_period", default=16, min_val=5, max_val=50, step=1,
                description="H1 filter ATR period",
            ),
            StrategyParam(
                "h1_factor", default=1.4, min_val=0.5, max_val=5.0, step=0.1,
                description="H1 filter multiplier",
            ),
            StrategyParam(
                "sl_atr_mult", default=1.5, min_val=0.5, max_val=5.0, step=0.1,
                description="SL ATR multiplier",
            ),
            StrategyParam(
                "rr_ratio", default=3.0, min_val=1.0, max_val=5.0, step=0.5,
                description="Risk:Reward ratio",
            ),
        ]

    def generate_signals(
        self,
        df: pd.DataFrame,
        period=16,
        factor=1.4,
        source="hl2",
        h1_filter="On",
        h1_period=16,
        h1_factor=1.4,
        sl_atr_mult=1.5,
        rr_ratio=3.0,
    ) -> tuple[pd.Series, pd.Series]:
        st = calc_supertrend(df, int(period), float(factor), source)
        direction = st["direction"]

        # Direction change signals
        entries = (direction == -1) & (direction.shift(1) == 1)
        exits = (direction == 1) & (direction.shift(1) == -1)

        # H1 filter: only enter long when H1 is in uptrend
        if str(h1_filter) == "On":
            h1_dir = _h1_direction(df, int(h1_period), float(h1_factor))
            entries = entries & (h1_dir == -1)

        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        return entries, exits

    def compute_stops(
        self, df: pd.DataFrame, sl_atr_mult=1.5, rr_ratio=3.0, **params
    ) -> tuple[pd.Series, pd.Series] | None:
        """Compute per-bar SL/TP from ATR(14) on M15.

        SL = ATR(14, M15) * sl_atr_mult, converted to fraction of close.
        TP = SL * rr_ratio.
        """
        sl_atr_mult = float(sl_atr_mult)
        rr_ratio = float(rr_ratio)

        # Resample to M15 and compute ATR(14)
        m15 = df.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna(subset=["open"])

        if len(m15) < 15:
            return None

        atr_m15 = _atr(m15, period=14)
        # Shift by 1 to avoid look-ahead (use completed M15 candle's ATR)
        atr_m15 = atr_m15.shift(1)
        # Forward-fill back to original timeframe
        atr_aligned = atr_m15.reindex(df.index, method="ffill")

        # Convert dollar-based ATR stop to fraction of close price
        sl_pct = (atr_aligned * sl_atr_mult) / df["close"]
        tp_pct = sl_pct * rr_ratio

        # Fill NaN with conservative fallback (first few bars before ATR warms up)
        sl_pct = sl_pct.fillna(sl_pct.dropna().iloc[0] if sl_pct.dropna().any() else 0.01)
        tp_pct = tp_pct.fillna(tp_pct.dropna().iloc[0] if tp_pct.dropna().any() else 0.03)

        return sl_pct, tp_pct
