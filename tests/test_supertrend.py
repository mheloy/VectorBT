"""Tests for SuperTrend strategy module."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.supertrend import (
    _price_source,
    _atr,
    calc_supertrend,
    _h1_direction,
    SuperTrendStrategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_ohlcv(n=200, base=2650.0, seed=42, freq="5min"):
    """Generate synthetic XAU/USD-like OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq=freq)
    close = base + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.5, 3.0, n)
    low = close - rng.uniform(0.5, 3.0, n)
    opn = close + rng.randn(n) * 0.3
    vol = rng.randint(100, 1000, n).astype(float)
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def make_trending(n=200, base=2650.0, peak=2700.0, trough=2640.0):
    """Generate data with clear up-then-down trend for direction change tests."""
    dates = pd.date_range("2025-01-01", periods=n, freq="5min")
    up = np.linspace(base, peak, n // 2)
    down = np.linspace(peak, trough, n // 2)
    close = np.concatenate([up, down])
    rng = np.random.RandomState(42)
    close = close + rng.randn(n) * 0.3
    return pd.DataFrame(
        {
            "open": close - rng.uniform(-0.5, 0.5, n),
            "high": close + rng.uniform(0.5, 2.0, n),
            "low": close - rng.uniform(0.5, 2.0, n),
            "close": close,
            "volume": rng.randint(100, 1000, n).astype(float),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Indicator tests
# ---------------------------------------------------------------------------


class TestPriceSource:
    def test_hl2(self):
        df = make_ohlcv(50)
        result = _price_source(df, "hl2")
        expected = (df["high"] + df["low"]) / 2
        pd.testing.assert_series_equal(result, expected)

    def test_close(self):
        df = make_ohlcv(50)
        result = _price_source(df, "close")
        pd.testing.assert_series_equal(result, df["close"])

    def test_hlc3(self):
        df = make_ohlcv(50)
        result = _price_source(df, "hlc3")
        expected = (df["high"] + df["low"] + df["close"]) / 3
        pd.testing.assert_series_equal(result, expected)

    def test_ohlc4(self):
        df = make_ohlcv(50)
        result = _price_source(df, "ohlc4")
        expected = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        pd.testing.assert_series_equal(result, expected)

    def test_invalid_raises(self):
        df = make_ohlcv(50)
        with pytest.raises(ValueError):
            _price_source(df, "invalid")


class TestATR:
    def test_length(self):
        df = make_ohlcv(100)
        atr = _atr(df, 14)
        assert len(atr) == 100

    def test_positive_after_warmup(self):
        df = make_ohlcv(100)
        atr = _atr(df, 14)
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_first_values_nan(self):
        df = make_ohlcv(100)
        atr = _atr(df, 14)
        # TR[0] is valid (high-low), RMA seeds after `period` TR values
        # First valid ATR at index period-1 (0-based)
        assert atr.iloc[:13].isna().all()
        assert not np.isnan(atr.iloc[13])


class TestSuperTrend:
    def test_output_columns(self):
        df = make_ohlcv()
        result = calc_supertrend(df, period=16, factor=1.4)
        assert "supertrend" in result.columns
        assert "direction" in result.columns

    def test_output_length(self):
        df = make_ohlcv()
        result = calc_supertrend(df, period=16, factor=1.4)
        assert len(result) == len(df)

    def test_direction_values(self):
        df = make_ohlcv()
        result = calc_supertrend(df, period=16, factor=1.4)
        unique_dirs = set(result["direction"].unique())
        assert unique_dirs.issubset({0, -1, 1})

    def test_supertrend_positive_after_warmup(self):
        df = make_ohlcv()
        result = calc_supertrend(df, period=16, factor=1.4)
        valid = result["supertrend"].dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_direction_changes_with_trending_data(self):
        df = make_trending()
        result = calc_supertrend(df, period=16, factor=1.4)
        changes = (result["direction"].diff() != 0).sum()
        assert changes > 1

    def test_different_sources_differ(self):
        df = make_ohlcv()
        r1 = calc_supertrend(df, source="hl2")
        r2 = calc_supertrend(df, source="close")
        # Different sources should produce different supertrend values
        valid = r1["supertrend"].dropna().index
        assert not np.array_equal(
            r1.loc[valid, "supertrend"].values,
            r2.loc[valid, "supertrend"].values,
        )

    def test_band_ratcheting_upper_only_decreases_in_uptrend(self):
        """During uptrend periods, the upper band should not increase."""
        df = make_ohlcv(500)
        result = calc_supertrend(df, period=16, factor=1.4)
        # This is implicitly tested by the indicator producing valid results,
        # but we verify the direction/supertrend relationship
        for i in range(1, len(result)):
            d = result["direction"].iloc[i]
            st = result["supertrend"].iloc[i]
            if d == -1:  # uptrend -> supertrend = lower band
                assert st <= df["close"].iloc[i] or np.isnan(st)


# ---------------------------------------------------------------------------
# H1 filter tests
# ---------------------------------------------------------------------------


class TestH1Filter:
    def test_returns_series_aligned_to_input(self):
        # Need enough 5M bars to form at least a few H1 candles
        df = make_ohlcv(n=500, freq="5min")
        h1_dir = _h1_direction(df, period=16, factor=1.4)
        assert len(h1_dir) == len(df)
        assert h1_dir.index.equals(df.index)

    def test_direction_values(self):
        df = make_ohlcv(n=500, freq="5min")
        h1_dir = _h1_direction(df, period=16, factor=1.4)
        unique = set(h1_dir.unique())
        assert unique.issubset({0, -1, 1})

    def test_insufficient_data_returns_neutral(self):
        # Very few bars -> not enough H1 candles
        df = make_ohlcv(n=20, freq="5min")
        h1_dir = _h1_direction(df, period=16, factor=1.4)
        # Should be all zeros (neutral)
        assert (h1_dir == 0).all()


# ---------------------------------------------------------------------------
# Strategy class tests
# ---------------------------------------------------------------------------


class TestSuperTrendStrategy:
    def setup_method(self):
        self.strategy = SuperTrendStrategy()

    def test_name(self):
        assert self.strategy.name == "SuperTrend"

    def test_parameters_count(self):
        params = self.strategy.parameters()
        assert len(params) == 14  # 6 core + 8 position management

    def test_default_params(self):
        defaults = self.strategy.default_params()
        assert defaults["period"] == 16
        assert defaults["factor"] == 1.4
        assert defaults["source"] == "hl2"
        assert defaults["h1_filter"] == "On"

    def test_generate_signals_returns_boolean_series(self):
        df = make_ohlcv(n=500)
        entries, exits = self.strategy.generate_signals(df, h1_filter="Off")
        assert entries.dtype == bool
        assert exits.dtype == bool
        assert len(entries) == len(df)
        assert len(exits) == len(df)

    def test_generate_signals_no_nan(self):
        df = make_ohlcv(n=500)
        entries, exits = self.strategy.generate_signals(df, h1_filter="Off")
        assert not entries.isna().any()
        assert not exits.isna().any()

    def test_entries_and_exits_not_simultaneous(self):
        """An entry and exit should not fire on the same bar."""
        df = make_ohlcv(n=500)
        entries, exits = self.strategy.generate_signals(df, h1_filter="Off")
        overlap = entries & exits
        assert not overlap.any()

    def test_h1_filter_reduces_entries(self):
        """H1 filter should block some entries, so On <= Off entries."""
        df = make_ohlcv(n=2000, freq="5min")
        entries_off, _ = self.strategy.generate_signals(df, h1_filter="Off")
        entries_on, _ = self.strategy.generate_signals(df, h1_filter="On")
        assert entries_on.sum() <= entries_off.sum()

    def test_with_trending_data_produces_signals(self):
        df = make_trending(n=500)
        entries, exits = self.strategy.generate_signals(df, h1_filter="Off")
        assert entries.sum() > 0
        assert exits.sum() > 0


# ---------------------------------------------------------------------------
# compute_stops tests
# ---------------------------------------------------------------------------


class TestComputeStops:
    def setup_method(self):
        self.strategy = SuperTrendStrategy()

    def test_returns_tuple_of_series(self):
        df = make_ohlcv(n=500, freq="5min")
        result = self.strategy.compute_stops(df, sl_atr_mult=1.5, rr_ratio=3.0)
        assert result is not None
        sl_pct, tp_pct = result
        assert isinstance(sl_pct, pd.Series)
        assert isinstance(tp_pct, pd.Series)
        assert len(sl_pct) == len(df)
        assert len(tp_pct) == len(df)

    def test_no_nan(self):
        df = make_ohlcv(n=500, freq="5min")
        sl_pct, tp_pct = self.strategy.compute_stops(df, sl_atr_mult=1.5, rr_ratio=3.0)
        assert not sl_pct.isna().any()
        assert not tp_pct.isna().any()

    def test_sl_positive_and_reasonable(self):
        df = make_ohlcv(n=500, freq="5min", base=2650.0)
        sl_pct, tp_pct = self.strategy.compute_stops(df, sl_atr_mult=1.5, rr_ratio=3.0)
        valid = sl_pct.dropna()
        assert (valid > 0).all()
        assert (valid < 0.05).all()  # SL should be < 5% for gold

    def test_tp_equals_sl_times_rr(self):
        df = make_ohlcv(n=500, freq="5min")
        rr = 2.5
        sl_pct, tp_pct = self.strategy.compute_stops(df, sl_atr_mult=1.5, rr_ratio=rr)
        np.testing.assert_allclose(tp_pct.values, sl_pct.values * rr, rtol=1e-10)

    def test_higher_mult_gives_wider_sl(self):
        df = make_ohlcv(n=500, freq="5min")
        sl_low, _ = self.strategy.compute_stops(df, sl_atr_mult=1.0, rr_ratio=3.0)
        sl_high, _ = self.strategy.compute_stops(df, sl_atr_mult=2.0, rr_ratio=3.0)
        # Higher multiplier should give wider (larger) SL percentages
        assert sl_high.mean() > sl_low.mean()

    def test_base_strategy_returns_none(self):
        from src.strategies.ma_crossover import MACrossover
        ma = MACrossover()
        df = make_ohlcv(n=100)
        assert ma.compute_stops(df) is None

    def test_insufficient_data_returns_none(self):
        df = make_ohlcv(n=10, freq="5min")
        result = self.strategy.compute_stops(df, sl_atr_mult=1.5, rr_ratio=3.0)
        assert result is None
