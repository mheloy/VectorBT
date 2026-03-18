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
    short_entries_arr,    # bool array: short entry signals
    short_exits_arr,      # bool array: short exit signals
    slippage,             # float64: fraction of price for slippage
    use_next_bar_open,    # bool: True = enter at next bar open
    sl_distance_arr,
    st_values_arr,  # SuperTrend line values for ST-line trailing
    trail_mode_flag,  # 0 = atr_stages, 1 = st_line
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
    ignore_signal_exits,
    risk_pct,          # % of equity to risk per trade (e.g., 0.03 = 3%)
    max_lot_value,     # Max notional per trade in $ (0 = no limit)
    fixed_lot_units,   # Fixed lot size in units (>0 overrides risk-based sizing)
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

    # Portfolio state — leveraged model
    # cash = account balance; num_units = position size (e.g., oz of gold)
    # PnL is added/subtracted from cash directly (no capital "allocated")
    cash = init_cash
    in_position = False
    direction = 0  # 1 = long, -1 = short (only long for now)

    # Position state
    entry_price = 0.0
    entry_bar = 0
    position_fraction = 0.0  # Fraction of initial position still open (1.0 at entry)
    initial_units = 0.0      # Total units at entry (e.g., oz of gold)
    sl_price = 0.0
    initial_sl_distance = 0.0
    tp_price = 0.0  # Final TP price

    # PM state
    partial_done = np.zeros(10, dtype=np.bool_)  # Up to 10 partial TPs
    be_done = False
    trail_stage = 0  # 0 = none, 1+ = active stage

    for i in range(n_bars):
        # ============================================================
        # PHASE 1: EXIT PROCESSING (if in position)
        # ============================================================
        if in_position:
            # --- Check exits (conservative order: SL first) ---
            # PnL = (exit - entry) * direction * units_closed
            # units_closed = initial_units * fraction_closed

            # 1. SL check (worst case first)
            sl_hit = False
            if direction == 1:
                sl_hit = low_arr[i] <= sl_price
            else:
                sl_hit = high_arr[i] >= sl_price

            if sl_hit:
                # Apply slippage to SL exit price
                if direction == 1:
                    exit_price = sl_price * (1.0 - slippage)
                else:
                    exit_price = sl_price * (1.0 + slippage)
                units_closed = initial_units * position_fraction
                trade_pnl = (exit_price - entry_price) * direction * units_closed
                fee_cost = units_closed * exit_price * fees
                trade_pnl -= fee_cost
                cash += trade_pnl

                # Determine exit type
                if be_done and not trailing_sl_enabled:
                    exit_type = EXIT_BE_SL
                elif trail_stage > 0:
                    exit_type = EXIT_TRAIL_SL
                elif be_done:
                    exit_type = EXIT_BE_SL
                else:
                    exit_type = EXIT_INITIAL_SL

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
                initial_units = 0.0
                # NO continue — fall through to Phase 2 for same-bar reversal

            # 2. Partial TP checks (on favorable side)
            if in_position and partial_tp_enabled and n_partial_tps > 0:
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
                        # Apply slippage to partial TP exit price
                        if direction == 1:
                            exit_price = trigger_price * (1.0 - slippage)
                        else:
                            exit_price = trigger_price * (1.0 + slippage)
                        if p == 0:
                            # TP1: fraction of initial position
                            close_frac = pt_pcts[p]
                        else:
                            # TP2+: fraction of remaining position
                            # (matches backtest-engine behaviour)
                            close_frac = position_fraction * pt_pcts[p]
                        if close_frac > position_fraction:
                            close_frac = position_fraction

                        units_closed = initial_units * close_frac
                        trade_pnl = (exit_price - entry_price) * direction * units_closed
                        fee_cost = units_closed * exit_price * fees
                        trade_pnl -= fee_cost
                        cash += trade_pnl

                        exit_type = EXIT_TP1 + p
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

                        if position_fraction <= 1e-9:
                            in_position = False
                            initial_units = 0.0
                            break

            # 3. Break-even check (if not already triggered by partial)
            if in_position and be_enabled and not be_done:
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

            # 4. Trailing SL update
            if in_position and trailing_sl_enabled and i > 0:
                if trail_mode_flag == 1 and be_done:
                    # ST-line trailing: runner stop follows the SuperTrend line
                    st_val = st_values_arr[i]
                    if not np.isnan(st_val) and st_val > 0:
                        if direction == 1:
                            # For long: ST line is support, only ratchet up
                            if st_val > sl_price:
                                sl_price = st_val
                        else:
                            # For short: ST line is resistance, only ratchet down
                            if st_val < sl_price:
                                sl_price = st_val

                elif trail_mode_flag == 0 and n_trail_stages > 0:
                    # ATR-stages trailing (previous bar body reference)
                    if direction == 1:
                        current_dist = high_arr[i] - entry_price
                    else:
                        current_dist = entry_price - low_arr[i]

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

                        prev_body_top = max(open_arr[i - 1], close_arr[i - 1])
                        prev_body_bot = min(open_arr[i - 1], close_arr[i - 1])

                        if direction == 1:
                            new_sl = prev_body_bot - trail_distance
                            if new_sl > sl_price:
                                sl_price = new_sl
                        else:
                            new_sl = prev_body_top + trail_distance
                            if new_sl < sl_price:
                                sl_price = new_sl

            # 5. Final TP check (for runner) — skip if final_tp_r <= 0 (no cap)
            if in_position and final_tp_r > 1e-9:
                final_tp_price = entry_price + direction * initial_sl_distance * final_tp_r
                final_tp_hit = False
                if direction == 1:
                    final_tp_hit = high_arr[i] >= final_tp_price
                else:
                    final_tp_hit = low_arr[i] <= final_tp_price

                if final_tp_hit:
                    # Apply slippage to final TP exit price
                    if direction == 1:
                        exit_price = final_tp_price * (1.0 - slippage)
                    else:
                        exit_price = final_tp_price * (1.0 + slippage)
                    units_closed = initial_units * position_fraction
                    trade_pnl = (exit_price - entry_price) * direction * units_closed
                    fee_cost = units_closed * exit_price * fees
                    trade_pnl -= fee_cost
                    cash += trade_pnl

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
                    initial_units = 0.0
                    # NO continue — fall through to Phase 2

            # 6. Signal exit check (skipped when PM manages all exits)
            if in_position and not ignore_signal_exits:
                exit_signal = False
                if direction == 1 and exits_arr[i]:
                    exit_signal = True
                elif direction == -1 and short_exits_arr[i]:
                    exit_signal = True

                if exit_signal:
                    # Apply slippage to signal exit price
                    if direction == 1:
                        exit_price = close_arr[i] * (1.0 - slippage)
                    else:
                        exit_price = close_arr[i] * (1.0 + slippage)
                    units_closed = initial_units * position_fraction
                    trade_pnl = (exit_price - entry_price) * direction * units_closed
                    fee_cost = units_closed * exit_price * fees
                    trade_pnl -= fee_cost
                    cash += trade_pnl

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
                    initial_units = 0.0
                    # NO continue — fall through to Phase 2

        # ============================================================
        # PHASE 2: ENTRY PROCESSING (if not in position)
        # ============================================================
        if not in_position:
            enter_long = entries_arr[i]
            enter_short = short_entries_arr[i]

            if enter_long or enter_short:
                sl_dist = sl_distance_arr[i]
                if sl_dist > 0 and not np.isnan(sl_dist):
                    if use_next_bar_open:
                        if i + 1 < n_bars:
                            base_price = open_arr[i + 1]
                            entry_bar = i + 1
                        else:
                            # Last bar, skip entry
                            equity[i] = cash
                            continue
                    else:
                        base_price = close_arr[i]
                        entry_bar = i

                    if enter_long:
                        direction = 1
                        entry_price = base_price * (1.0 + slippage)
                    else:
                        direction = -1
                        entry_price = base_price * (1.0 - slippage)

                    position_fraction = 1.0
                    initial_sl_distance = sl_dist
                    sl_price = entry_price - direction * sl_dist  # Direction-aware!
                    tp_price = entry_price + direction * sl_dist * final_tp_r

                    # Position sizing
                    if fixed_lot_units > 0:
                        # Fixed lot mode (matches backtst-engine --fixed-lots)
                        initial_units = fixed_lot_units
                    else:
                        # Risk-based position sizing (leveraged)
                        # At SL, loss = sl_dist * num_units = risk_pct * equity
                        current_equity = cash
                        initial_units = (current_equity * risk_pct) / sl_dist

                        # Cap at max lot
                        if max_lot_value > 0:
                            max_units = max_lot_value / entry_price
                            if initial_units > max_units:
                                initial_units = max_units

                    # Entry fee based on notional
                    fee_cost = initial_units * entry_price * fees
                    cash -= fee_cost

                    in_position = True
                    be_done = False
                    trail_stage = 0
                    for p in range(10):
                        partial_done[p] = False

        # ============================================================
        # PHASE 3: MARK-TO-MARKET (always)
        # ============================================================
        if in_position:
            unrealized_pnl = (close_arr[i] - entry_price) * direction * initial_units * position_fraction
            equity[i] = cash + unrealized_pnl
        else:
            equity[i] = cash

    # Close any remaining position at last bar
    if in_position:
        base_exit = close_arr[n_bars - 1]
        if direction == 1:
            exit_price = base_exit * (1.0 - slippage)
        else:
            exit_price = base_exit * (1.0 + slippage)
        units_closed = initial_units * position_fraction
        trade_pnl = (exit_price - entry_price) * direction * units_closed
        fee_cost = units_closed * exit_price * fees
        trade_pnl -= fee_cost
        cash += trade_pnl

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
    fees: float = 0.0,
    slippage: float = 0.0,
    risk_pct: float = 0.03,
    max_lot_value: float = 0.0,
    st_values: pd.Series | None = None,
    fixed_lot_units: float = 0.0,
    short_entries: pd.Series | None = None,
    short_exits: pd.Series | None = None,
    execution_mode: str = "next_bar_open",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run the advanced position management simulation.

    Args:
        df: OHLCV DataFrame with datetime index.
        entries: Boolean Series of long entry signals.
        exits: Boolean Series of long exit signals.
        sl_distances: Series of SL distances in dollars (1R) per bar.
        config: Position management configuration.
        init_cash: Starting capital.
        fees: Fee fraction per trade side.
        slippage: Fraction of price for slippage (e.g. 0.001 = 0.1%).
        risk_pct: Fraction of equity to risk per trade.
        max_lot_value: Max notional per trade in $ (0 = no limit).
        st_values: SuperTrend line values for ST-line trailing.
        fixed_lot_units: Fixed lot size in units (>0 overrides risk-based sizing).
        short_entries: Boolean Series of short entry signals.
        short_exits: Boolean Series of short exit signals.
        execution_mode: "next_bar_open" or "same_bar_close".

    Returns:
        Tuple of (equity_array, trade_records_array, n_trades).
    """
    # Convert short signals to arrays
    if short_entries is not None:
        short_entries_arr = short_entries.values.astype(np.bool_)
        short_exits_arr = short_exits.values.astype(np.bool_)
    else:
        short_entries_arr = np.zeros(len(df), dtype=np.bool_)
        short_exits_arr = np.zeros(len(df), dtype=np.bool_)

    use_next_bar_open = execution_mode == "next_bar_open"

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

    # Prepare ST values for trailing
    if st_values is not None:
        st_vals_arr = st_values.values.astype(np.float64)
    else:
        st_vals_arr = np.zeros(len(df), dtype=np.float64)

    trail_mode_flag = 1 if config.trail_mode == "st_line" else 0

    equity, trades, n_trades = _simulate_core(
        open_arr=df["open"].values.astype(np.float64),
        high_arr=df["high"].values.astype(np.float64),
        low_arr=df["low"].values.astype(np.float64),
        close_arr=df["close"].values.astype(np.float64),
        entries_arr=entries.values.astype(np.bool_),
        exits_arr=exits.values.astype(np.bool_),
        short_entries_arr=short_entries_arr,
        short_exits_arr=short_exits_arr,
        slippage=float(slippage),
        use_next_bar_open=bool(use_next_bar_open),
        sl_distance_arr=sl_distances.values.astype(np.float64),
        st_values_arr=st_vals_arr,
        trail_mode_flag=trail_mode_flag,
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
        ignore_signal_exits=True,  # PM handles all exits; don't close on signal flip
        risk_pct=risk_pct,
        max_lot_value=max_lot_value,
        fixed_lot_units=fixed_lot_units,
    )

    return equity, trades[:n_trades], n_trades
