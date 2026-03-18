# Backtest Engine Best Practices Review — Design Spec

**Date**: 2026-03-19
**Branch**: `feature/backtest-best-practices` (branched from `WFASuperTrend`)
**Goal**: Align VectorBT backtest engine with gold/forex best practices and live MT5 bot behavior

## Context

The backtest engine is long-only, missing 76% of actual trades (user trades both BUY and SELL on XAUUSD). Fee model was miscalibrated (original `fees=0.0001` was 10-17x actual cost). No slippage model. Same-bar execution is optimistic. No hold-out set for final validation.

User trades on an ECN account (XAUUSD.ecn) with:
- Commission: $3.00 per standard lot per side
- Spread: ~4 points typical
- Both BUY and SELL directions
- SuperTrend strategy with H1 filter and 200MA filter variants

---

## Change 1: Bidirectional Trading (Short Support)

### 1.1 SignalResult Dataclass

Add to `src/strategies/base.py`:

```python
@dataclass
class SignalResult:
    entries: pd.Series              # long entries (always required)
    exits: pd.Series                # long exits (always required)
    short_entries: pd.Series | None = None
    short_exits: pd.Series | None = None
```

- `generate_signals()` return type changes from `tuple[pd.Series, pd.Series]` to `SignalResult`
- Backward compatible: existing simple strategies return `SignalResult(entries=..., exits=...)` with no short signals
- Runner checks `signal_result.short_entries is not None` to decide whether to pass short signals to VBT

### 1.2 SuperTrend Signal Logic

Current (long-only):
```python
entries = (direction == -1) & (direction.shift(1) == 1)
exits   = (direction == 1) & (direction.shift(1) == -1)
```

Proposed (bidirectional):
```python
long_entries  = (direction == -1) & (direction.shift(1) == 1)
long_exits    = (direction == 1) & (direction.shift(1) == -1)
short_entries = (direction == 1) & (direction.shift(1) == -1)
short_exits   = (direction == -1) & (direction.shift(1) == 1)
```

Filter application:
```python
if filter_type == "h1_supertrend":
    h1_dir = _h1_direction(df, period, factor, source, atr_method)
    long_entries  = long_entries  & (h1_dir == -1)   # H1 uptrend
    short_entries = short_entries & (h1_dir == 1)     # H1 downtrend
elif filter_type == "200ma":
    ma = _ma_filter(df, ma_filter_period, ma_filter_type)
    long_entries  = long_entries  & (df["close"] > ma)
    short_entries = short_entries & (df["close"] < ma)

if direction_mode == "long_only":
    short_entries[:] = False
    short_exits[:] = False
elif direction_mode == "short_only":
    long_entries[:] = False
    long_exits[:] = False
```

### 1.3 New Strategy Parameters

| Parameter | Default | Choices | Description |
|-----------|---------|---------|-------------|
| `filter_type` | `"h1_supertrend"` | `h1_supertrend`, `200ma`, `none` | Trend filter type (replaces old `h1_filter` param) |
| `direction_mode` | `"both"` | `both`, `long_only`, `short_only` | Allowed trade directions |
| `ma_filter_period` | `200` | 50-500, step 10 | MA filter period |
| `ma_filter_type` | `"SMA"` | `SMA`, `EMA` | MA filter smoothing |

**Migration**: Remove the old `h1_filter` parameter (On/Off). Replace with `filter_type`:
- `h1_filter="On"` → `filter_type="h1_supertrend"`
- `h1_filter="Off"` → `filter_type="none"`

### 1.4 Simulator Short Support

In `src/engine/simulator.py`:

**Signal input change**: The Numba kernel `_simulate_core` currently accepts `entries_arr` and `exits_arr`. Add `short_entries_arr` and `short_exits_arr` parameters:

```python
def _simulate_core(
    ...,
    entries_arr,          # long entries (existing)
    exits_arr,            # long exits (existing)
    short_entries_arr,    # NEW: short entries
    short_exits_arr,      # NEW: short exits
    ...
):
```

**Entry logic change** (replaces hardcoded `direction = 1` at line 371):

```python
# Check for entries (long or short)
if not in_position:
    if entries_arr[i]:
        direction = 1   # Long
        # ... open position
    elif short_entries_arr[i]:
        direction = -1  # Short
        # ... open position
```

**SL placement direction fix** (line 374, currently hardcoded for longs):

```python
# Current (broken for shorts):
sl_price = entry_price - sl_dist

# Fixed (direction-aware):
sl_price = entry_price - direction * sl_dist
# Long:  sl_price = entry - dist (SL below)
# Short: sl_price = entry + dist (SL above)
```

**Same-bar exit+entry sequencing**: When SuperTrend flips, `long_exits[i]` and `short_entries[i]` fire on the same bar (and vice versa). The simulator should:
1. First check exits for the current position
2. Close the position (realize PnL)
3. Then check entries for the opposite direction
4. Open the new position

This matches live MT5 behavior where the bot closes one trade and opens the reverse.

**Exit signal routing**: Long positions respond to `exits_arr`, short positions respond to `short_exits_arr`:

```python
if in_position and direction == 1 and exits_arr[i]:
    # close long
elif in_position and direction == -1 and short_exits_arr[i]:
    # close short
```

**Units note**: The simulator works in ounces (not lots). 1 standard lot = 100 oz. The `fixed_lot_units` parameter is in ounces. Fee calculation `units * price * fees` is correct because `fees` is calibrated as fraction-of-notional where notional = units(oz) * price($/oz).

### 1.5 VBT Path Short Support

In `src/engine/runner.py`, pass short signals to `Portfolio.from_signals()`:
```python
if signal_result.short_entries is not None:
    pf_kwargs["short_entries"] = signal_result.short_entries
    pf_kwargs["short_exits"] = signal_result.short_exits
```

### 1.6 Other Strategies

MA Crossover, RSI Reversal, Bollinger Breakout: wrap existing `(entries, exits)` in `SignalResult` with no short signals. No logic changes.

---

## Change 2: Fee & Slippage Calibration

### 2.1 Calibration Source

From user's ECN trade history (68 trades, week of 2026-03-16):
- Commission: $3.00/lot/side → at $5,000 gold: `fees ≈ 0.000006`
- Spread: 4 points → half-spread: `slippage ≈ 0.000004`

### 2.2 Implementation

**Runner** (`src/engine/runner.py`):
- Add `slippage` parameter (default `0.000004`)
- Change `fees` default from `0.0` to `0.000006`
- VBT open-source `Portfolio.from_signals()` accepts `slippage` as a parameter (confirmed in API docs). Pass both: `Portfolio.from_signals(fees=fees, slippage=slippage, ...)`
- Fallback: if VBT version does not support `slippage` kwarg, combine into fees: `effective_fees = fees + slippage`

**Simulator** (`src/engine/simulator.py`):
- Add `slippage` parameter to `_simulate_core` and `simulate()` wrapper
- Apply to execution price at entry:
  - Long entry: `entry_price = base_price * (1 + slippage)` (buy at worse/higher price)
  - Short entry: `entry_price = base_price * (1 - slippage)` (sell at worse/lower price)
- Apply to execution price at exit (reverse):
  - Long exit: `exit_price = base_price * (1 - slippage)` (sell at worse/lower price)
  - Short exit: `exit_price = base_price * (1 + slippage)` (buy at worse/higher price)
- Fee application unchanged: `fee = units * price * fees`
- Note: `base_price` is either `open[i+1]` (next-bar-open mode) or `close[i]` (same-bar-close mode)

**Dashboard**: Both `fees` and `slippage` configurable in sidebar with these defaults.

**Optimizer/WFA**: Pass through fees and slippage from dashboard config.

---

## Change 3: Execution Model

### 3.1 Next-Bar-Open Execution

**VBT path**: Pass `open`, `high`, `low` to `Portfolio.from_signals()`:
```python
pf_kwargs["open"] = df["open"]
pf_kwargs["high"] = df["high"]
pf_kwargs["low"] = df["low"]
```

When `open` is provided, VBT executes entries at next bar's open price. Passing `high`/`low` enables intra-bar stop evaluation (SL/TP checked against bar range, not just close).

**Simulator path**: Change entry price from `close[signal_bar]` to `open[signal_bar + 1]`. Edge case: if a signal fires on the last bar, skip the entry (no next bar to execute on).

### 3.2 Configuration

| Parameter | Default | Choices | Description |
|-----------|---------|---------|-------------|
| `execution_mode` | `"next_bar_open"` | `next_bar_open`, `same_bar_close` | Trade execution timing |

Runner-level parameter (not per-strategy). Configurable on dashboard sidebar.

When `same_bar_close`: don't pass `open` to VBT, simulator uses `close[signal_bar]` (legacy behavior for backtest-engine parity).

---

## Change 4: Overfitting Protection

### 4.1 Hold-Out Set

**Location**: `src/engine/walk_forward.py` and `src/dashboard/pages/walk_forward.py`

Split data before any analysis:
```python
if holdout_enabled:
    split_idx = int(len(df) * (1 - holdout_pct))
    train_df = df.iloc[:split_idx]
    holdout_df = df.iloc[split_idx:]
else:
    train_df = df
    holdout_df = None
```

- All WFA windows, optimization, Monte Carlo operate on `train_df` only
- After WFA completes, use the **last OOS window's best params** for the hold-out backtest (these are the params that would be used for live trading)
- Single backtest on `holdout_df` gives the unbiased performance estimate
- Dashboard shows hold-out result in a separate clearly-labeled "HOLD-OUT (unseen data)" section
- The warmup guard in `generate_signals()` automatically handles the start of the hold-out slice (suppresses signals during first `warmup_bars`), so no additional warmup logic needed

| Parameter | Default | Description |
|-----------|---------|-------------|
| `holdout_enabled` | `True` | Enable hold-out set |
| `holdout_pct` | `0.10` | Fraction of data reserved (10%) |

### 4.2 ATR Warmup Guard

In `generate_signals()`, suppress signals during warmup:
```python
warmup_bars = max(period, 14) + 1
entries.iloc[:warmup_bars] = False
exits.iloc[:warmup_bars] = False
short_entries.iloc[:warmup_bars] = False
short_exits.iloc[:warmup_bars] = False
```

Prevents trades with unreliable ATR/SL values at the start of data or each WFA window.

### 4.3 Minimum Trade Count Warning

After any backtest, if `total_trades < 30`:
- Dashboard shows warning banner: "Low trade count — results may not be statistically significant"
- Not a hard block, just informational
- Threshold configurable: `min_trades_warning = 30`

---

## Files Modified

| File | Changes |
|------|---------|
| `src/strategies/base.py` | Add `SignalResult` dataclass, update ABC signature |
| `src/strategies/supertrend.py` | Bidirectional signals, 200MA filter, direction_mode, filter_type, warmup guard |
| `src/strategies/ma_crossover.py` | Wrap return in `SignalResult` |
| `src/strategies/rsi_reversal.py` | Wrap return in `SignalResult` |
| `src/strategies/bollinger_breakout.py` | Wrap return in `SignalResult` |
| `src/engine/runner.py` | Add `slippage`, pass OHLC + short signals to VBT, next-bar execution, `execution_mode` |
| `src/engine/simulator.py` | Short direction, slippage on execution price, next-bar-open entry |
| `src/engine/optimizer.py` | Unpack `SignalResult` for both VBT + PM paths, pass slippage, OHLC |
| `src/engine/walk_forward.py` | Hold-out split, pass new params |
| `src/dashboard/pages/backtest.py` | Sidebar: direction_mode, filter_type, fees, slippage, execution_mode |
| `src/dashboard/pages/optimize.py` | New params in sweep grid |
| `src/dashboard/pages/walk_forward.py` | Hold-out toggle + result display |
| `tests/test_supertrend.py` | Short signal tests, filter tests, warmup tests |
| `tests/test_simulator.py` | Short direction, slippage tests |
| `docs/strategies/supertrend.md` | Document all new params and parameter history |
| `CLAUDE.md` | Update strategy documentation |

## Optimizer Detail

The optimizer currently unpacks `entries, exits = strategy.generate_signals(...)` at line 88. With `SignalResult`:

**VBT vectorized path** (`_optimize_vbt`):
- Unpack `signal_result = strategy.generate_signals(...)`
- Stack `entries`, `exits` into multi-column DataFrames (existing logic)
- Additionally stack `short_entries`, `short_exits` if present
- Pass all 4 to `Portfolio.from_signals()` — VBT handles multi-column alignment natively

**PM simulation path** (`_optimize_with_pm`):
- Unpack `signal_result` per combo (line 226)
- Pass `short_entries_arr`, `short_exits_arr` to `simulate()`
- No multi-column stacking needed (runs per-combo anyway)

---

## Not Changing

- SuperTrend indicator calculation (unchanged)
- Numba simulator structure (adding direction + slippage, not restructuring)
- Storage/DB schema
- Monte Carlo, regime analysis, robustness, trade analysis (consume results, don't generate)
- Other strategies stay long-only (opt-in to shorts later)

## Implementation Order

1. `SignalResult` dataclass + base ABC update
2. SuperTrend bidirectional signals + filters
3. Runner: fees, slippage, OHLC, short signals, execution_mode
4. Simulator: short direction + slippage + next-bar-open
5. Optimizer passthrough
6. WFA: hold-out set + passthrough
7. Dashboard UI updates
8. Tests
9. Documentation (supertrend.md + CLAUDE.md)

## Risks & Mitigations

- **Breaking change on SignalResult**: Mitigated by making short signals optional (None). Simple strategies only need to wrap existing tuple.
- **Simulator short bugs**: Existing direction-aware code paths already handle direction=-1, but untested. Mitigated by dedicated short direction tests.
- **Fee sensitivity**: New defaults are calibrated from actual trades. User can always set to 0 for comparison.
- **Hold-out too small**: 10% of 3+ years ≈ 4 months. Should be sufficient for M5 data with high trade frequency.
