"""Tests for the advanced position management simulator."""

import numpy as np
import pandas as pd
import pytest

from src.engine.position_management import (
    PartialTPConfig,
    PositionManagementConfig,
    TrailingStageConfig,
)
from src.engine.simulator import (
    EXIT_BE_SL,
    EXIT_FINAL_TP,
    EXIT_INITIAL_SL,
    EXIT_SIGNAL,
    EXIT_TP1,
    EXIT_TP2,
    EXIT_TRAIL_SL,
    TR_DIRECTION,
    TR_EXIT_TYPE,
    TR_FRACTION,
    TR_PNL,
    simulate,
)
from src.engine.sim_result import build_simulation_result, BacktestResult


def _make_df(prices, n_bars=100):
    """Create a simple OHLC DataFrame from close prices.

    For simplicity, open=close of prev bar, high=max(open,close)+1, low=min(open,close)-1.
    """
    close = np.array(prices, dtype=np.float64)
    n = len(close)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    index = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=index,
    )


def _make_signals(n, entry_bars, exit_bars=None):
    """Create boolean entry/exit Series."""
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    for b in entry_bars:
        entries[b] = True
    if exit_bars:
        for b in exit_bars:
            exits[b] = True
    index = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.Series(entries, index=index), pd.Series(exits, index=index)


class TestSimulatorBasic:
    """Test basic entry/exit without advanced PM."""

    def test_no_trades_when_no_entries(self):
        """No entries → equity stays at init_cash."""
        prices = [2000.0] * 20
        df = _make_df(prices)
        entries, exits = _make_signals(20, [])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False,
            be_enabled=False,
            trailing_sl_enabled=False,
        )
        equity, trades, n = simulate(df, entries, exits, sl_dist, config, init_cash=10000)
        assert n == 0
        assert equity[-1] == pytest.approx(10000.0)

    def test_signal_exit(self):
        """Entry at bar 2, signal exit at bar 10."""
        prices = [2000.0] * 20
        df = _make_df(prices)
        entries, exits = _make_signals(20, [2], [10])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False,
            be_enabled=False,
            trailing_sl_enabled=False,
            final_tp_r=0,  # Disable final TP
        )
        equity, trades, n = simulate(df, entries, exits, sl_dist, config, init_cash=10000)
        assert n == 1
        assert int(trades[0, TR_EXIT_TYPE]) == EXIT_SIGNAL

    def test_sl_hit(self):
        """Price drops below SL → closes at SL price."""
        # Entry at 2000, SL distance = 10 → SL at 1990
        prices = [2000.0] * 5 + [1985.0] * 5  # Drops below SL
        df = _make_df(prices)
        entries, exits = _make_signals(10, [1])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False,
            be_enabled=False,
            trailing_sl_enabled=False,
            final_tp_r=0,
        )
        equity, trades, n = simulate(df, entries, exits, sl_dist, config, init_cash=10000)
        assert n >= 1
        assert int(trades[0, TR_EXIT_TYPE]) == EXIT_INITIAL_SL
        # Should lose money
        assert trades[0, TR_PNL] < 0


class TestPartialTP:
    """Test partial take profit logic."""

    def test_tp1_closes_50pct(self):
        """Price rises to 1.5R → TP1 closes 50% of position."""
        # Entry at 2000, SL dist = 10, TP1 trigger = 1.5R = 15 above entry = 2015
        prices = [2000.0] * 3 + [2020.0] * 7  # Rises above TP1
        df = _make_df(prices)
        entries, exits = _make_signals(10, [1])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tps=[PartialTPConfig(trigger_r=1.5, close_pct=0.50)],
            partial_tp_enabled=True,
            be_enabled=False,
            trailing_sl_enabled=False,
            final_tp_r=0,
        )
        equity, trades, n = simulate(df, entries, exits, sl_dist, config, init_cash=10000)
        # Should have at least 1 partial close
        tp1_trades = [t for t in range(n) if int(trades[t, TR_EXIT_TYPE]) == EXIT_TP1]
        assert len(tp1_trades) >= 1
        assert trades[tp1_trades[0], TR_FRACTION] == pytest.approx(0.50)
        assert trades[tp1_trades[0], TR_PNL] > 0

    def test_tp1_and_tp2(self):
        """Price rises to 2.9R → both TP1 and TP2 fire."""
        # Entry at 2000, SL dist = 10
        # TP1 at 1.5R = 2015, TP2 at 2.9R = 2029
        prices = [2000.0] * 3 + [2035.0] * 7  # Goes above both TPs
        df = _make_df(prices)
        entries, exits = _make_signals(10, [1])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tps=[
                PartialTPConfig(trigger_r=1.5, close_pct=0.50),
                PartialTPConfig(trigger_r=2.9, close_pct=0.30),
            ],
            partial_tp_enabled=True,
            be_enabled=False,
            trailing_sl_enabled=False,
            final_tp_r=0,
        )
        equity, trades, n = simulate(df, entries, exits, sl_dist, config, init_cash=10000)

        tp1_trades = [t for t in range(n) if int(trades[t, TR_EXIT_TYPE]) == EXIT_TP1]
        tp2_trades = [t for t in range(n) if int(trades[t, TR_EXIT_TYPE]) == EXIT_TP2]
        assert len(tp1_trades) >= 1
        assert len(tp2_trades) >= 1
        assert trades[tp1_trades[0], TR_FRACTION] == pytest.approx(0.50)
        # TP2 closes 30% of remaining (0.50), not 30% of initial
        # 0.50 * 0.30 = 0.15 of initial (matches backtest-engine)
        assert trades[tp2_trades[0], TR_FRACTION] == pytest.approx(0.15)


class TestBreakEven:
    """Test break-even SL logic."""

    def test_be_triggered_on_tp1(self):
        """When TP1 fires, SL should move to breakeven."""
        # Entry at 2000, SL dist = 10, TP1 at 1.5R = 2015
        # Price goes up to hit TP1, then reverses
        prices = [2000.0] * 3 + [2020.0] * 3 + [1995.0] * 4
        df = _make_df(prices)
        entries, exits = _make_signals(10, [1])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tps=[PartialTPConfig(trigger_r=1.5, close_pct=0.50)],
            partial_tp_enabled=True,
            be_enabled=True,
            be_trigger_r=1.0,
            be_offset_dollars=1.0,
            trailing_sl_enabled=False,
            final_tp_r=0,
        )
        equity, trades, n = simulate(df, entries, exits, sl_dist, config, init_cash=10000)

        # Should have TP1 trade + BE SL trade (remaining closed at ~entry+1)
        exit_types = [int(trades[t, TR_EXIT_TYPE]) for t in range(n)]
        assert EXIT_TP1 in exit_types
        # Remaining 50% should close at BE (entry+1=2001) or SL
        be_trades = [t for t in range(n) if int(trades[t, TR_EXIT_TYPE]) == EXIT_BE_SL]
        if len(be_trades) > 0:
            # BE trade PnL should be slightly positive (closed at entry + offset)
            assert trades[be_trades[0], TR_PNL] >= 0


class TestTrailingSL:
    """Test trailing stop loss logic."""

    def test_trailing_ratchets_up(self):
        """As price rises, trailing SL should move up (never back down)."""
        # Gradual rise then reversal
        prices = [2000.0] + list(np.linspace(2000, 2025, 15)) + list(np.linspace(2025, 2005, 10))
        df = _make_df(prices)
        entries, exits = _make_signals(len(prices), [1])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False,
            be_enabled=False,
            trailing_stages=[
                TrailingStageConfig(trigger_r=0.5, sl_multiplier=1.0),
                TrailingStageConfig(trigger_r=1.0, sl_multiplier=0.8),
            ],
            trailing_sl_enabled=True,
            trail_mode="atr_stages",  # Explicitly test ATR-stages mode
            final_tp_r=0,
        )
        equity, trades, n = simulate(df, entries, exits, sl_dist, config, init_cash=10000)
        # Should eventually hit trailing SL on reversal
        trail_trades = [t for t in range(n) if int(trades[t, TR_EXIT_TYPE]) == EXIT_TRAIL_SL]
        assert len(trail_trades) >= 1


class TestFinalTP:
    """Test final take profit for runner."""

    def test_final_tp_at_3r(self):
        """Price hits 3R → remaining position closes at final TP."""
        # Entry at 2000, SL dist = 10, Final TP at 3.0R = 2030
        prices = [2000.0] * 3 + [2035.0] * 7
        df = _make_df(prices)
        entries, exits = _make_signals(10, [1])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False,
            be_enabled=False,
            trailing_sl_enabled=False,
            final_tp_r=3.0,
        )
        equity, trades, n = simulate(df, entries, exits, sl_dist, config, init_cash=10000)
        tp_trades = [t for t in range(n) if int(trades[t, TR_EXIT_TYPE]) == EXIT_FINAL_TP]
        assert len(tp_trades) >= 1
        assert trades[tp_trades[0], TR_PNL] > 0


class TestFullLifecycle:
    """Test complete trade lifecycle matching live bot behavior."""

    def test_full_pm_lifecycle(self):
        """Entry → TP1 (50%) → TP2 (30%) → Final TP (20%) runner."""
        # Entry at 2000, SL dist = 10
        # TP1 at 1.5R = 2015, TP2 at 2.9R = 2029, Final TP at 3.0R = 2030
        prices = [2000.0] * 3 + list(np.linspace(2000, 2035, 20)) + [2035.0] * 7
        df = _make_df(prices)
        entries, exits = _make_signals(len(prices), [1])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig()  # All defaults
        equity, trades, n = simulate(df, entries, exits, sl_dist, config, init_cash=10000)

        # Should have 3 trade records: TP1, TP2, and Final TP (or trail SL)
        assert n >= 2  # At minimum TP1 + something
        assert equity[-1] > 10000  # Should be profitable


class TestSimResultIntegration:
    """Test SimulationResult and BacktestResult wrappers."""

    def test_build_simulation_result(self):
        prices = [2000.0] * 3 + [2035.0] * 7
        df = _make_df(prices)
        entries, exits = _make_signals(10, [1])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False,
            be_enabled=False,
            trailing_sl_enabled=False,
            final_tp_r=3.0,
        )
        equity, trades, n = simulate(df, entries, exits, sl_dist, config, init_cash=10000)
        result = build_simulation_result(equity, trades, n, df.index, 10000, 0.0001)

        assert isinstance(result.equity_curve, pd.Series)
        assert isinstance(result.trades_df, pd.DataFrame)
        assert "total_return" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "total_trades" in result.metrics
        assert result.metrics["total_trades"] >= 1

    def test_backtest_result_simulator_path(self):
        prices = [2000.0] * 3 + [2035.0] * 7
        df = _make_df(prices)
        entries, exits = _make_signals(10, [1])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False, be_enabled=False,
            trailing_sl_enabled=False, final_tp_r=3.0,
        )
        equity, trades, n = simulate(df, entries, exits, sl_dist, config, init_cash=10000)
        sim_result = build_simulation_result(equity, trades, n, df.index, 10000, 0.0001)

        br = BacktestResult(sim_result=sim_result)
        assert br.is_simulator
        assert br.portfolio is None
        assert len(br.equity_curve) == 10
        assert isinstance(br.metrics, dict)
        assert len(br.trades_df) >= 1


class TestShortDirection:
    """Test short selling support."""

    def test_short_sl_hit(self):
        """Short entry, price rises above SL -> loss."""
        prices = [2000.0] * 5 + [2015.0] * 5
        df = _make_df(prices)
        entries, _ = _make_signals(10, [])
        short_entries, short_exits = _make_signals(10, [1])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False, be_enabled=False,
            trailing_sl_enabled=False, final_tp_r=0,
        )
        equity, trades, n = simulate(
            df, entries, pd.Series(False, index=df.index),
            sl_dist, config, init_cash=10000,
            short_entries=short_entries, short_exits=short_exits,
            execution_mode="same_bar_close",
        )
        assert n >= 1
        assert int(trades[0, TR_EXIT_TYPE]) == EXIT_INITIAL_SL
        assert trades[0, TR_PNL] < 0
        assert trades[0, TR_DIRECTION] == -1

    def test_short_profitable(self):
        """Short entry, price drops -> profit."""
        prices = [2000.0] * 3 + [1980.0] * 7
        df = _make_df(prices)
        entries, _ = _make_signals(10, [])
        short_entries, short_exits = _make_signals(10, [1], [8])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False, be_enabled=False,
            trailing_sl_enabled=False, final_tp_r=0,
        )
        equity, trades, n = simulate(
            df, entries, pd.Series(False, index=df.index),
            sl_dist, config, init_cash=10000,
            short_entries=short_entries, short_exits=short_exits,
            execution_mode="same_bar_close",
        )
        assert n >= 1
        assert trades[0, TR_PNL] > 0
        assert trades[0, TR_DIRECTION] == -1

    def test_same_bar_reversal(self):
        """Long exit + short entry on same bar."""
        prices = [2000.0] * 3 + [1990.0] * 7
        df = _make_df(prices)
        long_entries, long_exits = _make_signals(10, [1], [4])
        short_entries, short_exits = _make_signals(10, [4], [8])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False, be_enabled=False,
            trailing_sl_enabled=False, final_tp_r=0,
        )
        equity, trades, n = simulate(
            df, long_entries, long_exits, sl_dist, config, init_cash=10000,
            short_entries=short_entries, short_exits=short_exits,
            execution_mode="same_bar_close",
        )
        assert n >= 2


class TestSlippage:
    """Test slippage on execution price."""

    def test_slippage_reduces_profit(self):
        prices = [2000.0] * 3 + [2020.0] * 7
        df = _make_df(prices)
        entries, exits = _make_signals(10, [1], [8])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False, be_enabled=False,
            trailing_sl_enabled=False, final_tp_r=0,
        )
        _, trades_no, n1 = simulate(
            df, entries, exits, sl_dist, config, init_cash=10000,
            slippage=0.0, execution_mode="same_bar_close",
        )
        _, trades_yes, n2 = simulate(
            df, entries, exits, sl_dist, config, init_cash=10000,
            slippage=0.001, execution_mode="same_bar_close",
        )
        assert n1 >= 1 and n2 >= 1
        assert trades_yes[0, TR_PNL] < trades_no[0, TR_PNL]


class TestNextBarOpen:
    """Test next-bar-open execution mode."""

    def test_entry_at_next_bar_open(self):
        """Entry signal on bar 1 should execute at bar 2's open price."""
        prices = [2000.0] * 3 + [2020.0] * 7
        df = _make_df(prices)
        entries, exits = _make_signals(10, [1], [8])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False, be_enabled=False,
            trailing_sl_enabled=False, final_tp_r=0,
        )
        _, trades_nbo, n_nbo = simulate(
            df, entries, exits, sl_dist, config, init_cash=10000,
            execution_mode="next_bar_open",
        )
        _, trades_sbc, n_sbc = simulate(
            df, entries, exits, sl_dist, config, init_cash=10000,
            execution_mode="same_bar_close",
        )
        assert n_nbo >= 1 and n_sbc >= 1
        # next_bar_open uses open of bar 2; same_bar_close uses close of bar 1
        # They should differ (open[2] vs close[1])
        # Both should produce trades but potentially different entry prices
