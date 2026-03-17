"""Numba JIT bar-by-bar simulator for advanced position management.

Handles partial take profits, break-even, and multi-stage trailing stop loss.
Operates on numpy arrays for maximum performance (~5-15ms per 236K-bar run).
"""

import numpy as np
import pandas as pd
from numba import njit

from .position_management import PositionManagementConfig

# Exit type constants (used in trade records)
EXIT_INITIAL_SL = 0
EXIT_TP1 = 1
EXIT_TP2 = 2
EXIT_TP3 = 3  # For future use
EXIT_BE_SL = 4
EXIT_TRAIL_SL = 5
EXIT_FINAL_TP = 6
EXIT_SIGNAL = 7

EXIT_TYPE_LABELS = {
    EXIT_INITIAL_SL: "Initial SL",
    EXIT_TP1: "TP1",
    EXIT_TP2: "TP2",
    EXIT_TP3: "TP3",
    EXIT_BE_SL: "BE SL",
    EXIT_TRAIL_SL: "Trail SL",
    EXIT_FINAL_TP: "Final TP",
    EXIT_SIGNAL: "Signal Exit",
}

# Trade record column indices
TR_ENTRY_BAR = 0
TR_EXIT_BAR = 1
TR_ENTRY_PRICE = 2
TR_EXIT_PRICE = 3
TR_FRACTION = 4  # Fraction of initial position closed
TR_PNL = 5
TR_EXIT_TYPE = 6
TR_DIRECTION = 7  # 1 = long, -1 = short
TR_COLS = 8


@njit(cache=True)
def _simulate_core(
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    entries_arr,
    exits_arr,
    sl_distance_arr,
    # Partial TP config (flat arrays for Numba)
    n_partial_tps,
    pt_triggers,  # R-multiple triggers
    pt_pcts,  # fraction of initial to close
    partial_tp_enabled,
    # Break-even config
    be_enabled,
    be_trigger_r,
    be_offset,  # in dollars
    # Trailing SL config (flat arrays)
    n_trail_stages,
    trail_triggers,  # R-multiple triggers
    trail_mults,  # SL distance multipliers
    trailing_sl_enabled,
    # Final TP
    final_tp_r,
    # Trade params
    fees,
    init_cash,
):
    """Core Numba-JIT simulation loop.

    Returns:
        equity: float64 array of portfolio value at each bar
        trades: float64 2D array of trade records (max_trades x TR_COLS)
        n_trades: int, number of actual trades recorded
    """
    n_bars = len(close_arr)
    max_trades = n_bars  # Upper bound; will be much less in practice

    equity = np.empty(n_bars, dtype=np.float64)
    trades = np.empty((max_trades, TR_COLS), dtype=np.float64)
    n_trades = 0

    # Portfolio state
    cash = init_cash
    in_position = False
    direction = 0  # 1 = long, -1 = short (only long for now)

    # Position state
    entry_price = 0.0
    entry_bar = 0
    position_fraction = 0.0  # Fraction of initial position still open
    sl_price = 0.0
    initial_sl_distance = 0.0
    tp_price = 0.0  # Final TP price
    allocated_capital = 0.0  # Capital allocated to this trade at entry

    # PM state
    partial_done = np.zeros(10, dtype=np.bool_)  # Up to 10 partial TPs
    be_done = False
    trail_stage = 0  # 0 = none, 1+ = active stage

    for i in range(n_bars):
        if in_position:
            # --- Check exits (conservative order: SL first) ---

            # Current unrealized distance from entry (for long)
            # Use high/low to check intra-bar triggers

            # 1. SL check (worst case first)
            sl_hit = False
            if direction == 1:
                sl_hit = low_arr[i] <= sl_price
            else:
                sl_hit = high_arr[i] >= sl_price

            if sl_hit:
                # Close remaining at SL price
                exit_price = sl_price
                pnl_per_unit = (exit_price - entry_price) * direction
                trade_pnl = pnl_per_unit * position_fraction * allocated_capital / entry_price
                fee_cost = abs(position_fraction * allocated_capital) * fees
                trade_pnl -= fee_cost
                cash += allocated_capital * position_fraction + trade_pnl

                # Determine exit type
                if be_done and not trailing_sl_enabled:
                    exit_type = EXIT_BE_SL
                elif trail_stage > 0:
                    exit_type = EXIT_TRAIL_SL
                elif be_done:
                    exit_type = EXIT_BE_SL
                else:
                    exit_type = EXIT_INITIAL_SL

                # Record trade
                if n_trades < max_trades:
                    trades[n_trades, TR_ENTRY_BAR] = entry_bar
                    trades[n_trades, TR_EXIT_BAR] = i
                    trades[n_trades, TR_ENTRY_PRICE] = entry_price
                    trades[n_trades, TR_EXIT_PRICE] = exit_price
                    trades[n_trades, TR_FRACTION] = position_fraction
                    trades[n_trades, TR_PNL] = trade_pnl
                    trades[n_trades, TR_EXIT_TYPE] = exit_type
                    trades[n_trades, TR_DIRECTION] = direction
                    n_trades += 1

                in_position = False
                position_fraction = 0.0
                allocated_capital = 0.0

                equity[i] = cash
                continue

            # 2. Partial TP checks (on favorable side)
            if partial_tp_enabled and n_partial_tps > 0:
                for p in range(n_partial_tps):
                    if partial_done[p]:
                        continue

                    trigger_price = entry_price + direction * initial_sl_distance * pt_triggers[p]
                    triggered = False
                    if direction == 1:
                        triggered = high_arr[i] >= trigger_price
                    else:
                        triggered = low_arr[i] <= trigger_price

                    if triggered:
                        # Close this partial
                        exit_price = trigger_price
                        close_frac = pt_pcts[p]
                        # Don't close more than remaining
                        if close_frac > position_fraction:
                            close_frac = position_fraction

                        pnl_per_unit = (exit_price - entry_price) * direction
                        trade_pnl = pnl_per_unit * close_frac * allocated_capital / entry_price
                        fee_cost = abs(close_frac * allocated_capital) * fees
                        trade_pnl -= fee_cost
                        cash += allocated_capital * close_frac + trade_pnl

                        # Record partial close
                        exit_type = EXIT_TP1 + p  # TP1=1, TP2=2, etc.
                        if n_trades < max_trades:
                            trades[n_trades, TR_ENTRY_BAR] = entry_bar
                            trades[n_trades, TR_EXIT_BAR] = i
                            trades[n_trades, TR_ENTRY_PRICE] = entry_price
                            trades[n_trades, TR_EXIT_PRICE] = exit_price
                            trades[n_trades, TR_FRACTION] = close_frac
                            trades[n_trades, TR_PNL] = trade_pnl
                            trades[n_trades, TR_EXIT_TYPE] = exit_type
                            trades[n_trades, TR_DIRECTION] = direction
                            n_trades += 1

                        position_fraction -= close_frac
                        partial_done[p] = True

                        # Auto-trigger BE on first partial TP
                        if p == 0 and be_enabled and not be_done:
                            if direction == 1:
                                new_sl = entry_price + be_offset
                                if new_sl > sl_price:
                                    sl_price = new_sl
                            else:
                                new_sl = entry_price - be_offset
                                if new_sl < sl_price:
                                    sl_price = new_sl
                            be_done = True

                        # If fully closed by partials
                        if position_fraction <= 1e-9:
                            in_position = False
                            allocated_capital = 0.0
                            break

            if not in_position:
                equity[i] = cash
                continue

            # 3. Break-even check (if not already triggered by partial)
            if be_enabled and not be_done:
                be_trigger_price = entry_price + direction * initial_sl_distance * be_trigger_r
                be_triggered = False
                if direction == 1:
                    be_triggered = high_arr[i] >= be_trigger_price
                else:
                    be_triggered = low_arr[i] <= be_trigger_price

                if be_triggered:
                    if direction == 1:
                        new_sl = entry_price + be_offset
                        if new_sl > sl_price:
                            sl_price = new_sl
                    else:
                        new_sl = entry_price - be_offset
                        if new_sl < sl_price:
                            sl_price = new_sl
                    be_done = True

            # 4. Trailing SL update (using previous bar body as reference)
            if trailing_sl_enabled and n_trail_stages > 0 and i > 0:
                # Current distance from entry
                if direction == 1:
                    current_dist = high_arr[i] - entry_price
                else:
                    current_dist = entry_price - low_arr[i]

                # Check which stage we're in
                new_stage = trail_stage
                for s in range(n_trail_stages):
                    stage_trigger_dist = initial_sl_distance * trail_triggers[s]
                    if current_dist >= stage_trigger_dist:
                        if s + 1 > new_stage:
                            new_stage = s + 1

                if new_stage > 0:
                    trail_stage = new_stage
                    mult = trail_mults[trail_stage - 1]
                    trail_distance = initial_sl_distance * mult

                    # Reference: previous bar body
                    prev_body_top = max(open_arr[i - 1], close_arr[i - 1])
                    prev_body_bot = min(open_arr[i - 1], close_arr[i - 1])

                    if direction == 1:
                        new_sl = prev_body_bot - trail_distance
                        # Only ratchet up
                        if new_sl > sl_price:
                            sl_price = new_sl
                    else:
                        new_sl = prev_body_top + trail_distance
                        # Only ratchet down
                        if new_sl < sl_price:
                            sl_price = new_sl

            # 5. Final TP check (for runner)
            if final_tp_r > 0:
                final_tp_price = entry_price + direction * initial_sl_distance * final_tp_r
                final_tp_hit = False
                if direction == 1:
                    final_tp_hit = high_arr[i] >= final_tp_price
                else:
                    final_tp_hit = low_arr[i] <= final_tp_price

                if final_tp_hit:
                    exit_price = final_tp_price
                    pnl_per_unit = (exit_price - entry_price) * direction
                    trade_pnl = pnl_per_unit * position_fraction * allocated_capital / entry_price
                    fee_cost = abs(position_fraction * allocated_capital) * fees
                    trade_pnl -= fee_cost
                    cash += allocated_capital * position_fraction + trade_pnl

                    if n_trades < max_trades:
                        trades[n_trades, TR_ENTRY_BAR] = entry_bar
                        trades[n_trades, TR_EXIT_BAR] = i
                        trades[n_trades, TR_ENTRY_PRICE] = entry_price
                        trades[n_trades, TR_EXIT_PRICE] = exit_price
                        trades[n_trades, TR_FRACTION] = position_fraction
                        trades[n_trades, TR_PNL] = trade_pnl
                        trades[n_trades, TR_EXIT_TYPE] = EXIT_FINAL_TP
                        trades[n_trades, TR_DIRECTION] = direction
                        n_trades += 1

                    in_position = False
                    position_fraction = 0.0
                    allocated_capital = 0.0

                    equity[i] = cash
                    continue

            # 6. Signal exit check
            if exits_arr[i]:
                exit_price = close_arr[i]
                pnl_per_unit = (exit_price - entry_price) * direction
                trade_pnl = pnl_per_unit * position_fraction * allocated_capital / entry_price
                fee_cost = abs(position_fraction * allocated_capital) * fees
                trade_pnl -= fee_cost
                cash += allocated_capital * position_fraction + trade_pnl

                if n_trades < max_trades:
                    trades[n_trades, TR_ENTRY_BAR] = entry_bar
                    trades[n_trades, TR_EXIT_BAR] = i
                    trades[n_trades, TR_ENTRY_PRICE] = entry_price
                    trades[n_trades, TR_EXIT_PRICE] = exit_price
                    trades[n_trades, TR_FRACTION] = position_fraction
                    trades[n_trades, TR_PNL] = trade_pnl
                    trades[n_trades, TR_EXIT_TYPE] = EXIT_SIGNAL
                    trades[n_trades, TR_DIRECTION] = direction
                    n_trades += 1

                in_position = False
                position_fraction = 0.0
                allocated_capital = 0.0

                equity[i] = cash
                continue

            # Mark-to-market for open position
            unrealized = (close_arr[i] - entry_price) * direction
            unrealized_pnl = unrealized * position_fraction * allocated_capital / entry_price
            equity[i] = cash + allocated_capital * position_fraction + unrealized_pnl

        else:
            # Not in position — check for entry
            if entries_arr[i]:
                sl_dist = sl_distance_arr[i]
                if sl_dist > 0 and not np.isnan(sl_dist):
                    entry_price = close_arr[i]
                    entry_bar = i
                    direction = 1  # Long only for now
                    position_fraction = 1.0
                    initial_sl_distance = sl_dist
                    sl_price = entry_price - sl_dist  # Long: SL below
                    tp_price = entry_price + sl_dist * final_tp_r

                    # Allocate capital (full available cash)
                    allocated_capital = cash
                    fee_cost = allocated_capital * fees
                    cash = 0.0
                    allocated_capital -= fee_cost  # Entry fee

                    in_position = True
                    be_done = False
                    trail_stage = 0
                    for p in range(10):
                        partial_done[p] = False

            equity[i] = cash
            if in_position:
                # First bar mark-to-market
                unrealized = (close_arr[i] - entry_price) * direction
                unrealized_pnl = unrealized * position_fraction * allocated_capital / entry_price
                equity[i] = cash + allocated_capital * position_fraction + unrealized_pnl

    # Close any remaining position at last bar
    if in_position:
        exit_price = close_arr[n_bars - 1]
        pnl_per_unit = (exit_price - entry_price) * direction
        trade_pnl = pnl_per_unit * position_fraction * allocated_capital / entry_price
        fee_cost = abs(position_fraction * allocated_capital) * fees
        trade_pnl -= fee_cost
        cash += allocated_capital * position_fraction + trade_pnl

        if n_trades < max_trades:
            trades[n_trades, TR_ENTRY_BAR] = entry_bar
            trades[n_trades, TR_EXIT_BAR] = n_bars - 1
            trades[n_trades, TR_ENTRY_PRICE] = entry_price
            trades[n_trades, TR_EXIT_PRICE] = exit_price
            trades[n_trades, TR_FRACTION] = position_fraction
            trades[n_trades, TR_PNL] = trade_pnl
            trades[n_trades, TR_EXIT_TYPE] = EXIT_SIGNAL
            trades[n_trades, TR_DIRECTION] = direction
            n_trades += 1

        equity[n_bars - 1] = cash

    return equity, trades, n_trades


def simulate(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    sl_distances: pd.Series,
    config: PositionManagementConfig,
    init_cash: float = 10_000.0,
    fees: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run the advanced position management simulation.

    Args:
        df: OHLCV DataFrame with datetime index.
        entries: Boolean Series of entry signals.
        exits: Boolean Series of exit signals.
        sl_distances: Series of SL distances in dollars (1R) per bar.
        config: Position management configuration.
        init_cash: Starting capital.
        fees: Fee fraction per trade side.

    Returns:
        Tuple of (equity_array, trade_records_array, n_trades).
    """
    # Convert config to flat Numba-compatible arrays
    n_partial_tps = len(config.partial_tps) if config.partial_tp_enabled else 0
    pt_triggers = np.array(
        [pt.trigger_r for pt in config.partial_tps], dtype=np.float64
    ) if n_partial_tps > 0 else np.empty(0, dtype=np.float64)
    pt_pcts = np.array(
        [pt.close_pct for pt in config.partial_tps], dtype=np.float64
    ) if n_partial_tps > 0 else np.empty(0, dtype=np.float64)

    n_trail_stages = len(config.trailing_stages) if config.trailing_sl_enabled else 0
    trail_triggers = np.array(
        [ts.trigger_r for ts in config.trailing_stages], dtype=np.float64
    ) if n_trail_stages > 0 else np.empty(0, dtype=np.float64)
    trail_mults = np.array(
        [ts.sl_multiplier for ts in config.trailing_stages], dtype=np.float64
    ) if n_trail_stages > 0 else np.empty(0, dtype=np.float64)

    equity, trades, n_trades = _simulate_core(
        open_arr=df["open"].values.astype(np.float64),
        high_arr=df["high"].values.astype(np.float64),
        low_arr=df["low"].values.astype(np.float64),
        close_arr=df["close"].values.astype(np.float64),
        entries_arr=entries.values.astype(np.bool_),
        exits_arr=exits.values.astype(np.bool_),
        sl_distance_arr=sl_distances.values.astype(np.float64),
        n_partial_tps=n_partial_tps,
        pt_triggers=pt_triggers,
        pt_pcts=pt_pcts,
        partial_tp_enabled=config.partial_tp_enabled,
        be_enabled=config.be_enabled,
        be_trigger_r=config.be_trigger_r,
        be_offset=config.be_offset_dollars,
        n_trail_stages=n_trail_stages,
        trail_triggers=trail_triggers,
        trail_mults=trail_mults,
        trailing_sl_enabled=config.trailing_sl_enabled,
        final_tp_r=config.final_tp_r,
        fees=fees,
        init_cash=init_cash,
    )

    return equity, trades[:n_trades], n_trades
