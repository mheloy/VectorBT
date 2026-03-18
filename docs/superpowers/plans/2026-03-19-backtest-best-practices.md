# Backtest Best Practices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bidirectional trading, calibrated fees/slippage, next-bar execution, and overfitting protection to the XAUUSD backtest engine.

**Architecture:** `SignalResult` dataclass replaces tuple returns from strategies. Runner and simulator gain `slippage`, `short_entries`/`short_exits`, and `execution_mode` parameters. WFA splits data into train/hold-out before analysis.

**Tech Stack:** Python 3.12, VectorBT, Numba, pandas, Streamlit

**Spec:** `docs/superpowers/specs/2026-03-19-backtest-best-practices-design.md`

---

### Task 1: SignalResult Dataclass + Base ABC

**Files:**
- Modify: `src/strategies/base.py`
- Test: `tests/test_supertrend.py`

- [ ] **Step 1: Add SignalResult dataclass to base.py**

Add after the `StrategyParam` dataclass (around line 23):

```python
@dataclass
class SignalResult:
    """Result of signal generation, supporting both long-only and bidirectional strategies."""
    entries: pd.Series              # long entries (always required)
    exits: pd.Series                # long exits (always required)
    short_entries: pd.Series | None = None
    short_exits: pd.Series | None = None
```

- [ ] **Step 2: Update generate_signals return type annotation**

Change the ABC method signature at line 40-52 from:

```python
@abstractmethod
def generate_signals(
    self, df: pd.DataFrame, **params
) -> tuple[pd.Series, pd.Series]:
```

to:

```python
@abstractmethod
def generate_signals(
    self, df: pd.DataFrame, **params
) -> "SignalResult":
```

Update the docstring to say "Returns: SignalResult with entries and exits (and optionally short_entries/short_exits)."

- [ ] **Step 3: Update import in base.py**

The `dataclass` import already exists. No new imports needed. Ensure `SignalResult` is exported — add it to any `__init__.py` if needed, or just import from `base` directly.

- [ ] **Step 4: Run existing tests to verify nothing breaks yet**

Run: `cd /home/mheloy/VectorBackTest && uv run pytest tests/ -v`

Expected: Tests will FAIL because strategies still return tuples, not SignalResult. This is expected — we fix them in Task 2 and 3.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/base.py
git commit -m "feat: add SignalResult dataclass to BaseStrategy ABC"
```

---

### Task 2: Migrate Simple Strategies to SignalResult

**Files:**
- Modify: `src/strategies/ma_crossover.py`
- Modify: `src/strategies/rsi_reversal.py`
- Modify: `src/strategies/bollinger_breakout.py`

- [ ] **Step 1: Update ma_crossover.py**

Add import at top:
```python
from .base import BaseStrategy, StrategyParam, SignalResult
```

Change `generate_signals` return (line 34 return type, line 44 return statement):

```python
def generate_signals(
    self, df: pd.DataFrame, fast_period=10, slow_period=50, ma_type="SMA"
) -> SignalResult:
    close = df["close"]
    ewm = ma_type.upper() == "EMA"

    fast_ma = vbt.MA.run(close, window=int(fast_period), ewm=ewm).ma
    slow_ma = vbt.MA.run(close, window=int(slow_period), ewm=ewm).ma

    entries = fast_ma.vbt.crossed_above(slow_ma)
    exits = fast_ma.vbt.crossed_below(slow_ma)

    return SignalResult(entries=entries, exits=exits)
```

- [ ] **Step 2: Update rsi_reversal.py**

Add import: `from .base import BaseStrategy, StrategyParam, SignalResult`

Change return type and statement:
```python
def generate_signals(
    self, df: pd.DataFrame, rsi_period=14, oversold=30, overbought=70
) -> SignalResult:
    ...
    return SignalResult(entries=entries, exits=exits)
```

- [ ] **Step 3: Update bollinger_breakout.py**

Add import: `from .base import BaseStrategy, StrategyParam, SignalResult`

Change return type and statement:
```python
def generate_signals(
    self, df: pd.DataFrame, bb_period=20, bb_std=2
) -> SignalResult:
    ...
    return SignalResult(entries=entries, exits=exits)
```

- [ ] **Step 4: Commit**

```bash
git add src/strategies/ma_crossover.py src/strategies/rsi_reversal.py src/strategies/bollinger_breakout.py
git commit -m "refactor: migrate simple strategies to SignalResult"
```

---

### Task 3: SuperTrend Bidirectional Signals + Filters

**Files:**
- Modify: `src/strategies/supertrend.py`
- Test: `tests/test_supertrend.py`

- [ ] **Step 1: Write failing tests for bidirectional signals**

Add to `tests/test_supertrend.py`:

```python
from src.strategies.base import SignalResult


class TestBidirectionalSignals:
    def setup_method(self):
        self.strategy = SuperTrendStrategy()

    def test_generate_signals_returns_signal_result(self):
        df = make_ohlcv(n=500)
        result = self.strategy.generate_signals(df, filter_type="none", direction_mode="both")
        assert isinstance(result, SignalResult)
        assert result.entries.dtype == bool
        assert result.exits.dtype == bool
        assert result.short_entries is not None
        assert result.short_exits is not None
        assert result.short_entries.dtype == bool

    def test_short_entries_exist_with_trending_data(self):
        df = make_trending(n=500)
        result = self.strategy.generate_signals(df, filter_type="none", direction_mode="both")
        assert result.short_entries.sum() > 0
        assert result.short_exits.sum() > 0

    def test_long_only_mode_suppresses_shorts(self):
        df = make_trending(n=500)
        result = self.strategy.generate_signals(df, filter_type="none", direction_mode="long_only")
        assert result.short_entries.sum() == 0
        assert result.short_exits.sum() == 0
        assert result.entries.sum() > 0

    def test_short_only_mode_suppresses_longs(self):
        df = make_trending(n=500)
        result = self.strategy.generate_signals(df, filter_type="none", direction_mode="short_only")
        assert result.entries.sum() == 0
        assert result.exits.sum() == 0
        assert result.short_entries.sum() > 0

    def test_short_entry_equals_long_exit_unfiltered(self):
        """Without filter, short entries = long exits (ST flip)."""
        df = make_trending(n=500)
        result = self.strategy.generate_signals(df, filter_type="none", direction_mode="both")
        pd.testing.assert_series_equal(result.short_entries, result.exits, check_names=False)
        pd.testing.assert_series_equal(result.short_exits, result.entries, check_names=False)

    def test_h1_filter_reduces_short_entries(self):
        df = make_ohlcv(n=2000, freq="5min")
        result_none = self.strategy.generate_signals(df, filter_type="none", direction_mode="both")
        result_h1 = self.strategy.generate_signals(df, filter_type="h1_supertrend", direction_mode="both")
        assert result_h1.short_entries.sum() <= result_none.short_entries.sum()

    def test_200ma_filter(self):
        df = make_ohlcv(n=500, freq="5min")
        result = self.strategy.generate_signals(
            df, filter_type="200ma", ma_filter_period=200, ma_filter_type="SMA", direction_mode="both"
        )
        assert isinstance(result, SignalResult)
        # With 200MA filter, some entries should be filtered
        result_none = self.strategy.generate_signals(df, filter_type="none", direction_mode="both")
        total_filtered = result.entries.sum() + result.short_entries.sum()
        total_unfiltered = result_none.entries.sum() + result_none.short_entries.sum()
        assert total_filtered <= total_unfiltered


class TestWarmupGuard:
    def setup_method(self):
        self.strategy = SuperTrendStrategy()

    def test_no_signals_during_warmup(self):
        df = make_ohlcv(n=500, freq="5min")
        result = self.strategy.generate_signals(df, period=17, filter_type="none", direction_mode="both")
        warmup = max(17, 14) + 1  # 18
        assert result.entries.iloc[:warmup].sum() == 0
        assert result.exits.iloc[:warmup].sum() == 0
        assert result.short_entries.iloc[:warmup].sum() == 0
        assert result.short_exits.iloc[:warmup].sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/mheloy/VectorBackTest && uv run pytest tests/test_supertrend.py::TestBidirectionalSignals -v`

Expected: FAIL — `generate_signals` doesn't accept `filter_type`/`direction_mode` yet.

- [ ] **Step 3: Add 200MA filter helper to supertrend.py**

Add after the `_h1_direction` function (around line 212):

```python
def _ma_filter(
    df: pd.DataFrame,
    period: int = 200,
    ma_type: str = "SMA",
) -> pd.Series:
    """Compute moving average for trend filtering."""
    close = df["close"]
    if ma_type.upper() == "EMA":
        return close.ewm(span=period, adjust=False).mean()
    return close.rolling(window=period, min_periods=period).mean()
```

- [ ] **Step 4: Update SuperTrend parameters() — replace h1_filter with new params**

Replace the `h1_filter` StrategyParam (line 251-254) with:

```python
StrategyParam(
    "filter_type", default="h1_supertrend",
    choices=["h1_supertrend", "200ma", "none"],
    description="Trend filter type",
),
StrategyParam(
    "direction_mode", default="both",
    choices=["both", "long_only", "short_only"],
    description="Allowed trade directions",
),
StrategyParam(
    "ma_filter_period", default=200, min_val=50, max_val=500, step=10,
    description="MA filter period (used when filter_type=200ma)",
),
StrategyParam(
    "ma_filter_type", default="SMA",
    choices=["SMA", "EMA"],
    description="MA filter smoothing (used when filter_type=200ma)",
),
```

- [ ] **Step 5: Rewrite generate_signals for bidirectional + filters + warmup**

Replace the entire `generate_signals` method (lines 305-334):

```python
def generate_signals(
    self,
    df: pd.DataFrame,
    period=17,
    factor=1.8,
    source="hl2",
    atr_method="sma",
    filter_type="h1_supertrend",
    direction_mode="both",
    ma_filter_period=200,
    ma_filter_type="SMA",
    sl_atr_mult=1.9,
    rr_ratio=3.0,
    **kwargs,
) -> SignalResult:
    from .base import SignalResult

    st = calc_supertrend(df, int(period), float(factor), source,
                         atr_method=str(atr_method))
    direction = st["direction"]

    # Direction change signals
    long_entries = (direction == -1) & (direction.shift(1) == 1)
    long_exits = (direction == 1) & (direction.shift(1) == -1)
    short_entries = (direction == 1) & (direction.shift(1) == -1)
    short_exits = (direction == -1) & (direction.shift(1) == 1)

    # Apply trend filter
    filter_type = str(filter_type)
    if filter_type == "h1_supertrend":
        h1_dir = _h1_direction(df, int(period), float(factor), str(source),
                               atr_method=str(atr_method))
        long_entries = long_entries & (h1_dir == -1)
        short_entries = short_entries & (h1_dir == 1)
    elif filter_type == "200ma":
        ma = _ma_filter(df, int(ma_filter_period), str(ma_filter_type))
        long_entries = long_entries & (df["close"] > ma)
        short_entries = short_entries & (df["close"] < ma)
    # filter_type == "none": no filtering

    # Apply direction mode
    direction_mode = str(direction_mode)
    if direction_mode == "long_only":
        short_entries = pd.Series(False, index=df.index)
        short_exits = pd.Series(False, index=df.index)
    elif direction_mode == "short_only":
        long_entries = pd.Series(False, index=df.index)
        long_exits = pd.Series(False, index=df.index)

    # Warmup guard — suppress signals before indicators stabilize
    warmup_bars = max(int(period), 14) + 1
    long_entries.iloc[:warmup_bars] = False
    long_exits.iloc[:warmup_bars] = False
    short_entries.iloc[:warmup_bars] = False
    short_exits.iloc[:warmup_bars] = False

    # Clean NaN
    long_entries = long_entries.fillna(False).astype(bool)
    long_exits = long_exits.fillna(False).astype(bool)
    short_entries = short_entries.fillna(False).astype(bool)
    short_exits = short_exits.fillna(False).astype(bool)

    return SignalResult(
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
    )
```

- [ ] **Step 6: Update existing tests for new return type**

In `tests/test_supertrend.py`, update `TestSuperTrendStrategy`:

- `test_parameters_count`: Change expected count from 16 to 19 (removed `h1_filter`, added `filter_type`, `direction_mode`, `ma_filter_period`, `ma_filter_type` = net +3)
- `test_default_params`: Change `assert defaults["h1_filter"] == "On"` to `assert defaults["filter_type"] == "h1_supertrend"` and `assert defaults["direction_mode"] == "both"`
- All tests that call `self.strategy.generate_signals(df, h1_filter="Off")`: change to `self.strategy.generate_signals(df, filter_type="none")` and unpack with `result = ...`, then `entries = result.entries`, `exits = result.exits`
- `test_h1_filter_reduces_entries`: change to use `filter_type="none"` vs `filter_type="h1_supertrend"`, compare `result.entries.sum()`

- [ ] **Step 7: Run all tests**

Run: `cd /home/mheloy/VectorBackTest && uv run pytest tests/test_supertrend.py -v`

Expected: All tests PASS including new bidirectional tests.

- [ ] **Step 8: Commit**

```bash
git add src/strategies/supertrend.py tests/test_supertrend.py
git commit -m "feat: add bidirectional signals, 200MA filter, warmup guard to SuperTrend"
```

---

### Task 4: Runner — Fees, Slippage, OHLC, Short Signals, Execution Mode

**Files:**
- Modify: `src/engine/runner.py`
- Modify: `src/strategies/base.py` (import)

- [ ] **Step 1: Update runner.py imports**

Add at top of `runner.py`:
```python
from src.strategies.base import BaseStrategy, SignalResult
```

- [ ] **Step 2: Update run_backtest signature**

Replace the function signature (lines 11-20):

```python
def run_backtest(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    params: dict | None = None,
    init_cash: float = 10_000.0,
    fees: float = 0.000006,
    slippage: float = 0.000004,
    sl_stop: float | None = None,
    tp_stop: float | None = None,
    freq: str | None = None,
    execution_mode: str = "next_bar_open",
) -> BacktestResult:
```

Update docstring to document `slippage` and `execution_mode`.

- [ ] **Step 3: Update signal unpacking (line 40)**

Change:
```python
entries, exits = strategy.generate_signals(df, **effective_params)
```

To:
```python
signal_result = strategy.generate_signals(df, **effective_params)
if isinstance(signal_result, SignalResult):
    entries = signal_result.entries
    exits = signal_result.exits
    short_entries = signal_result.short_entries
    short_exits = signal_result.short_exits
else:
    # Backward compat: tuple return
    entries, exits = signal_result
    short_entries = None
    short_exits = None
```

- [ ] **Step 4: Update simulator path (lines 45-78)**

Pass `short_entries`, `short_exits`, `slippage`, and `execution_mode` to `simulate()`:

```python
if pm_config is not None:
    sl_distances = strategy.compute_sl_distances(df, **effective_params)

    st_values = None
    if pm_config.trail_mode == "st_line" and hasattr(strategy, 'compute_supertrend_values'):
        st_values = strategy.compute_supertrend_values(df, **effective_params)

    fixed_lot = pm_config.fixed_lot_units if pm_config.sizing_mode == "fixed_lot" else 0.0

    equity_arr, trade_records, n_trades = simulate(
        df=df,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        sl_distances=sl_distances,
        config=pm_config,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        risk_pct=pm_config.risk_pct,
        max_lot_value=pm_config.max_lot_value,
        st_values=st_values,
        fixed_lot_units=fixed_lot,
        execution_mode=execution_mode,
    )

    sim_result = build_simulation_result(
        equity_arr=equity_arr,
        trade_records=trade_records,
        n_trades=n_trades,
        index=df.index,
        init_cash=init_cash,
        fees=fees,
    )
    return BacktestResult(sim_result=sim_result)
```

- [ ] **Step 5: Update VBT path (lines 80-102)**

Replace the VBT path section:

```python
# VBT path
if sl_stop is None and tp_stop is None:
    stops = strategy.compute_stops(df, **effective_params)
    if stops is not None:
        sl_stop, tp_stop = stops

pf_kwargs = dict(
    close=df["close"],
    entries=entries,
    exits=exits,
    init_cash=init_cash,
    fees=fees,
    slippage=slippage,
)

# Pass OHLC for intra-bar stop evaluation and next-bar-open execution
if execution_mode == "next_bar_open":
    pf_kwargs["open"] = df["open"]
pf_kwargs["high"] = df["high"]
pf_kwargs["low"] = df["low"]

# Short signals
if short_entries is not None:
    pf_kwargs["short_entries"] = short_entries
    pf_kwargs["short_exits"] = short_exits

if freq:
    pf_kwargs["freq"] = freq
if sl_stop is not None:
    pf_kwargs["sl_stop"] = sl_stop
if tp_stop is not None:
    pf_kwargs["tp_stop"] = tp_stop

try:
    portfolio = vbt.Portfolio.from_signals(**pf_kwargs)
except TypeError:
    # Fallback: VBT version may not support slippage kwarg
    pf_kwargs.pop("slippage", None)
    pf_kwargs["fees"] = fees + slippage
    portfolio = vbt.Portfolio.from_signals(**pf_kwargs)

return BacktestResult(portfolio=portfolio)
```

- [ ] **Step 6: Commit**

```bash
git add src/engine/runner.py
git commit -m "feat: add slippage, short signals, OHLC, execution_mode to runner"
```

---

### Task 5: Simulator — Short Direction + Slippage + Next-Bar-Open

**Files:**
- Modify: `src/engine/simulator.py`
- Test: `tests/test_simulator.py`

- [ ] **Step 1: Write failing tests for short direction and slippage**

Add to `tests/test_simulator.py`:

```python
class TestShortDirection:
    """Test short selling support in simulator."""

    def test_short_sl_hit(self):
        """Short entry, price rises above SL → closes at SL."""
        # Short at 2000, SL dist = 10 → SL at 2010
        prices = [2000.0] * 5 + [2015.0] * 5  # Rises above SL
        df = _make_df(prices)
        entries, _ = _make_signals(10, [])  # No long entries
        short_entries, short_exits = _make_signals(10, [1])  # Short entry at bar 1
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False,
            be_enabled=False,
            trailing_sl_enabled=False,
            final_tp_r=0,
        )
        equity, trades, n = simulate(
            df, entries, pd.Series(False, index=df.index),
            sl_dist, config, init_cash=10000,
            short_entries=short_entries, short_exits=short_exits,
        )
        assert n >= 1
        assert int(trades[0, TR_EXIT_TYPE]) == EXIT_INITIAL_SL
        assert trades[0, TR_PNL] < 0  # Lost money on short going up

    def test_short_profitable(self):
        """Short entry, price drops → profitable."""
        prices = [2000.0] * 3 + [1980.0] * 7  # Drops
        df = _make_df(prices)
        entries, _ = _make_signals(10, [])
        short_entries, short_exits = _make_signals(10, [1], [8])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False,
            be_enabled=False,
            trailing_sl_enabled=False,
            final_tp_r=0,
        )
        equity, trades, n = simulate(
            df, entries, pd.Series(False, index=df.index),
            sl_dist, config, init_cash=10000,
            short_entries=short_entries, short_exits=short_exits,
        )
        assert n >= 1
        assert trades[0, TR_PNL] > 0

    def test_long_exit_then_short_entry_same_bar(self):
        """When long exit and short entry on same bar, both execute."""
        prices = [2000.0] * 3 + [1990.0] * 7
        df = _make_df(prices)
        # Long entry bar 1, long exit bar 4, short entry bar 4
        long_entries, long_exits = _make_signals(10, [1], [4])
        short_entries, short_exits = _make_signals(10, [4], [8])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False,
            be_enabled=False,
            trailing_sl_enabled=False,
            final_tp_r=0,
        )
        equity, trades, n = simulate(
            df, long_entries, long_exits, sl_dist, config, init_cash=10000,
            short_entries=short_entries, short_exits=short_exits,
        )
        # Should have at least 2 trades (long closed + short closed)
        assert n >= 2


class TestSlippage:
    """Test slippage on execution price."""

    def test_slippage_reduces_profit(self):
        """Same trade with slippage should be less profitable."""
        prices = [2000.0] * 3 + [2020.0] * 7
        df = _make_df(prices)
        entries, exits = _make_signals(10, [1], [8])
        sl_dist = pd.Series(10.0, index=df.index)

        config = PositionManagementConfig(
            partial_tp_enabled=False,
            be_enabled=False,
            trailing_sl_enabled=False,
            final_tp_r=0,
        )
        _, trades_no_slip, n1 = simulate(
            df, entries, exits, sl_dist, config, init_cash=10000, slippage=0.0,
        )
        _, trades_slip, n2 = simulate(
            df, entries, exits, sl_dist, config, init_cash=10000, slippage=0.001,
        )
        assert n1 >= 1 and n2 >= 1
        assert trades_slip[0, TR_PNL] < trades_no_slip[0, TR_PNL]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/mheloy/VectorBackTest && uv run pytest tests/test_simulator.py::TestShortDirection -v`

Expected: FAIL — `simulate()` doesn't accept `short_entries` yet.

- [ ] **Step 3: Update simulate() wrapper to accept new params**

Update the `simulate` function signature (line 434-446):

```python
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
```

Add array conversion for short signals and pass to `_simulate_core`:

```python
# Short signal arrays
if short_entries is not None:
    short_entries_arr = short_entries.values.astype(np.bool_)
    short_exits_arr = short_exits.values.astype(np.bool_)
else:
    short_entries_arr = np.zeros(len(df), dtype=np.bool_)
    short_exits_arr = np.zeros(len(df), dtype=np.bool_)

use_next_bar_open = execution_mode == "next_bar_open"
```

Pass to `_simulate_core`:
```python
equity, trades, n_trades = _simulate_core(
    ...,
    entries_arr=entries.values.astype(np.bool_),
    exits_arr=exits.values.astype(np.bool_),
    short_entries_arr=short_entries_arr,
    short_exits_arr=short_exits_arr,
    ...,
    slippage=slippage,
    use_next_bar_open=use_next_bar_open,
    ...
)
```

- [ ] **Step 4: Update _simulate_core Numba kernel — COMPLETE RESTRUCTURED LOOP**

**IMPORTANT**: This function is `@njit(cache=True)`. After changing the signature, delete the `__pycache__` directories to clear the Numba cache: `find . -name "__pycache__" -exec rm -rf {} +`. All new parameters must be Numba-compatible scalar types (float64, bool_, int64) or numpy arrays.

Add parameters to `_simulate_core` signature (after `exits_arr`):
```python
short_entries_arr,
short_exits_arr,
slippage,         # float64
use_next_bar_open,  # bool_ (Numba-compatible)
```

**COMPLETE restructured main loop** — replaces the entire `for i in range(n_bars):` block (lines 116-407). The key change: separate exit processing from entry processing (no if/else), so same-bar reversal works:

```python
    for i in range(n_bars):
        # ============================================================
        # PHASE 1: EXIT PROCESSING (when in position)
        # ============================================================
        if in_position:
            # --- 1. SL check (worst case first) ---
            sl_hit = False
            if direction == 1:
                sl_hit = low_arr[i] <= sl_price
            else:
                sl_hit = high_arr[i] >= sl_price

            if sl_hit:
                exit_price = sl_price
                if direction == 1:
                    exit_price = exit_price * (1.0 - slippage)
                else:
                    exit_price = exit_price * (1.0 + slippage)
                units_closed = initial_units * position_fraction
                trade_pnl = (exit_price - entry_price) * direction * units_closed
                fee_cost = units_closed * abs(exit_price) * fees
                trade_pnl -= fee_cost
                cash += trade_pnl

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
                # DO NOT continue — fall through to entry check for same-bar reversal

            # --- 2-5: Partial TP, BE, Trailing, Final TP (only if still in position) ---
            if in_position:
                # [Keep existing partial TP, BE, trailing SL, final TP logic UNCHANGED]
                # These blocks already use direction-aware checks.
                # Only change: apply slippage to partial TP and final TP exit prices:
                #   exit_price = trigger_price
                #   if direction == 1: exit_price *= (1.0 - slippage)
                #   else: exit_price *= (1.0 + slippage)
                pass  # (Copy existing blocks 2-5 here, with slippage on exit prices)

            # --- 6. Signal exit check ---
            if in_position:
                exit_signal = False
                if direction == 1 and exits_arr[i]:
                    exit_signal = True
                elif direction == -1 and short_exits_arr[i]:
                    exit_signal = True

                if exit_signal and not ignore_signal_exits:
                    exit_price = close_arr[i]
                    if direction == 1:
                        exit_price = exit_price * (1.0 - slippage)
                    else:
                        exit_price = exit_price * (1.0 + slippage)
                    units_closed = initial_units * position_fraction
                    trade_pnl = (exit_price - entry_price) * direction * units_closed
                    fee_cost = units_closed * abs(exit_price) * fees
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
                    # DO NOT continue — fall through to entry check

        # ============================================================
        # PHASE 2: ENTRY PROCESSING (when NOT in position)
        # ============================================================
        if not in_position:
            enter_long = entries_arr[i]
            enter_short = short_entries_arr[i]

            if enter_long or enter_short:
                sl_dist = sl_distance_arr[i]
                if sl_dist > 0 and not np.isnan(sl_dist):
                    # Entry price based on execution mode
                    if use_next_bar_open:
                        if i + 1 < n_bars:
                            base_price = open_arr[i + 1]
                            entry_bar = i + 1
                        else:
                            # Last bar — skip entry
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
                    sl_price = entry_price - direction * sl_dist  # Direction-aware

                    # Position sizing (unchanged)
                    if fixed_lot_units > 0:
                        initial_units = fixed_lot_units
                    else:
                        current_equity = cash
                        initial_units = (current_equity * risk_pct) / sl_dist
                        if max_lot_value > 0:
                            max_units = max_lot_value / entry_price
                            if initial_units > max_units:
                                initial_units = max_units

                    fee_cost = initial_units * entry_price * fees
                    cash -= fee_cost

                    in_position = True
                    be_done = False
                    trail_stage = 0
                    for p in range(10):
                        partial_done[p] = False

        # ============================================================
        # PHASE 3: MARK-TO-MARKET
        # ============================================================
        equity[i] = cash
        if in_position:
            unrealized_pnl = (close_arr[i] - entry_price) * direction * initial_units * position_fraction
            equity[i] = cash + unrealized_pnl
```

**Key structural change**: The old `if in_position: ... else: ...` with `continue` statements is replaced by three sequential phases. After an SL hit or signal exit closes the position, execution falls through to Phase 2 (entry), allowing same-bar reversal. The `continue` statements from the old code are removed (except for the last-bar skip case).

- [ ] **Step 5: Apply slippage to SL/TP exit prices too**

In the SL hit block (around line 130), apply slippage:
```python
if sl_hit:
    exit_price = sl_price
    # Apply slippage to SL exit
    if direction == 1:
        exit_price = exit_price * (1.0 - slippage)
    else:
        exit_price = exit_price * (1.0 + slippage)
```

Apply same pattern to partial TP exits and final TP exits.

- [ ] **Step 6: Run tests**

Run: `cd /home/mheloy/VectorBackTest && uv run pytest tests/test_simulator.py -v`

Expected: All tests PASS including new short direction and slippage tests.

- [ ] **Step 7: Commit**

```bash
git add src/engine/simulator.py tests/test_simulator.py
git commit -m "feat: add short direction, slippage, next-bar-open to simulator"
```

---

### Task 6: Optimizer Passthrough

**Files:**
- Modify: `src/engine/optimizer.py`

- [ ] **Step 1: Update optimize() signature**

Add `slippage` parameter:
```python
def optimize(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    sweep_params: dict[str, list],
    metric: str = "sharpe_ratio",
    init_cash: float = 10_000.0,
    fees: float = 0.000006,
    slippage: float = 0.000004,
    freq: str | None = None,
    sl_stop: float | None = None,
    tp_stop: float | None = None,
    progress_cb=None,
    execution_mode: str = "next_bar_open",
) -> OptimizationResult:
```

- [ ] **Step 2: Update VBT vectorized path signal unpacking (line 88)**

Change:
```python
entries, exits = strategy.generate_signals(df, **params)
label = tuple(combo)
all_entries[label] = entries.values
all_exits[label] = exits.values
```

To:
```python
from src.strategies.base import SignalResult
signal_result = strategy.generate_signals(df, **params)
if isinstance(signal_result, SignalResult):
    entries = signal_result.entries
    exits = signal_result.exits
    short_entries = signal_result.short_entries
    short_exits = signal_result.short_exits
else:
    entries, exits = signal_result
    short_entries = None
    short_exits = None

label = tuple(combo)
all_entries[label] = entries.values
all_exits[label] = exits.values
if short_entries is not None:
    all_short_entries[label] = short_entries.values
    all_short_exits[label] = short_exits.values
```

Initialize `all_short_entries = {}` and `all_short_exits = {}` alongside the existing dicts. Build corresponding DataFrames and pass to `Portfolio.from_signals()`.

- [ ] **Step 3: Add slippage, OHLC, execution_mode to VBT Portfolio call**

```python
pf_kwargs = dict(
    close=df["close"],
    entries=entries_df,
    exits=exits_df,
    init_cash=init_cash,
    fees=fees,
    slippage=slippage,
)

if execution_mode == "next_bar_open":
    pf_kwargs["open"] = df["open"]
pf_kwargs["high"] = df["high"]
pf_kwargs["low"] = df["low"]

if all_short_entries:
    short_entries_df = pd.DataFrame(
        np.column_stack(list(all_short_entries.values())),
        index=df.index, columns=col_index,
    )
    short_exits_df = pd.DataFrame(
        np.column_stack(list(all_short_exits.values())),
        index=df.index, columns=col_index,
    )
    pf_kwargs["short_entries"] = short_entries_df
    pf_kwargs["short_exits"] = short_exits_df
```

- [ ] **Step 4: Update PM path signal unpacking (line 226)**

Change:
```python
entries, exits = strategy.generate_signals(df, **full_params)
```

To:
```python
signal_result = strategy.generate_signals(df, **full_params)
if isinstance(signal_result, SignalResult):
    entries = signal_result.entries
    exits = signal_result.exits
    short_entries = signal_result.short_entries
    short_exits = signal_result.short_exits
else:
    entries, exits = signal_result
    short_entries = None
    short_exits = None
```

And pass `short_entries`, `short_exits`, `slippage`, `execution_mode` to `simulate()`.

- [ ] **Step 5: Pass slippage and execution_mode through _optimize_with_pm**

Update the `_optimize_with_pm` function signature to include `slippage` and `execution_mode`, and pass them to `simulate()`.

- [ ] **Step 6: Run tests**

Run: `cd /home/mheloy/VectorBackTest && uv run pytest tests/ -v`

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/engine/optimizer.py
git commit -m "feat: pass short signals, slippage, OHLC through optimizer"
```

---

### Task 7: Walk-Forward — Hold-Out Set + Passthrough

**Files:**
- Modify: `src/engine/walk_forward.py`

- [ ] **Step 1: Update run_walk_forward signature**

Add new parameters:
```python
def run_walk_forward(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    sweep_params: dict[str, list],
    n_windows: int = 8,
    is_bars: int | None = None,
    oos_bars: int | None = None,
    anchored: bool = False,
    min_trades: int = 20,
    metric: str = "calmar_ratio",
    init_cash: float = 10_000.0,
    fees: float = 0.000006,
    slippage: float = 0.000004,
    freq: str | None = None,
    progress_cb=None,
    execution_mode: str = "next_bar_open",
    holdout_enabled: bool = True,
    holdout_pct: float = 0.10,
) -> WalkForwardResult:
```

- [ ] **Step 2: Add hold-out split at start of function**

After the docstring, before the window calculation:

```python
# Hold-out split
if holdout_enabled and holdout_pct > 0:
    split_idx = int(len(df) * (1 - holdout_pct))
    train_df = df.iloc[:split_idx]
    holdout_df = df.iloc[split_idx:]
else:
    train_df = df
    holdout_df = None

# Use train_df for all WFA windows
total_bars = len(train_df)
```

Replace all references to `df` in the WFA loop with `train_df`.

- [ ] **Step 3: Pass slippage and execution_mode to optimize() and run_backtest()**

In the loop where `optimize()` is called (line 148-156), add `slippage=slippage` and `execution_mode=execution_mode`.

In `run_backtest()` calls (lines 184 and 188), add `slippage=slippage` and `execution_mode=execution_mode`.

Same for the full-sample optimization (line 234).

- [ ] **Step 4: Add hold-out backtest after WFA completes**

Before the final `return`, add:

```python
# Hold-out backtest (if enabled)
holdout_result = None
holdout_metrics = None
if holdout_df is not None and len(holdout_df) > 0 and windows:
    # Use last OOS window's best params (most recent)
    last_params = windows[-1].best_params
    full_holdout_params = {**strategy.default_params(), **last_params}
    for k, v in full_holdout_params.items():
        if hasattr(v, 'item'):
            full_holdout_params[k] = v.item()

    holdout_bt = run_backtest(
        strategy, holdout_df, full_holdout_params, init_cash, fees,
        slippage=slippage, freq=freq, execution_mode=execution_mode,
    )
    holdout_metrics = holdout_bt.metrics
    holdout_result = holdout_bt
```

- [ ] **Step 5: Add holdout fields to WalkForwardResult**

Update the `WalkForwardResult` dataclass:

```python
@dataclass
class WalkForwardResult:
    ...  # existing fields
    holdout_metrics: dict | None = None
    holdout_equity: pd.Series | None = None
    holdout_params: dict | None = None
```

Pass in the return statement:
```python
holdout_metrics=holdout_metrics,
holdout_equity=holdout_result.equity_curve if holdout_result else None,
holdout_params=last_params if holdout_df is not None and windows else None,
```

- [ ] **Step 6: Commit**

```bash
git add src/engine/walk_forward.py
git commit -m "feat: add hold-out set and slippage passthrough to WFA"
```

---

### Task 8: Dashboard UI Updates

**Files:**
- Modify: `src/dashboard/pages/backtest.py`
- Modify: `src/dashboard/pages/optimize.py`
- Modify: `src/dashboard/pages/walk_forward.py`

- [ ] **Step 1: Update backtest.py sidebar — fees, slippage, execution_mode**

Replace the fees input (line 120):

```python
st.sidebar.subheader("Execution Model")
execution_mode = st.sidebar.selectbox(
    "Execution Timing",
    ["next_bar_open", "same_bar_close"],
    index=0,
    format_func=lambda x: "Next Bar Open (realistic)" if x == "next_bar_open" else "Same Bar Close (legacy)",
)

st.sidebar.subheader("Cost Model")
fees = st.sidebar.number_input("Commission (fraction)", value=0.000006, step=0.000001, format="%.6f",
                                help="ECN commission as fraction of notional. Default: $3/lot at $5000 gold")
slippage = st.sidebar.number_input("Slippage (fraction)", value=0.000004, step=0.000001, format="%.6f",
                                    help="Half-spread as fraction of price. Default: 4pt spread on XAUUSD")
```

- [ ] **Step 2: Pass new params to run_backtest call (line 152-161)**

```python
result = run_backtest(
    strategy=strategy,
    df=df,
    params=param_values,
    init_cash=init_cash,
    fees=fees,
    slippage=slippage,
    sl_stop=sl_stop,
    tp_stop=tp_stop,
    freq=freq_map.get(timeframe),
    execution_mode=execution_mode,
)
```

Also store in session state:
```python
st.session_state["backtest_config"] = {
    "init_cash": init_cash, "fees": fees, "slippage": slippage,
    "sl_stop": sl_stop, "tp_stop": tp_stop, "execution_mode": execution_mode,
}
```

- [ ] **Step 3: Add min trade count warning**

After the metrics cards, add:
```python
if metrics["total_trades"] < 30:
    st.warning("Low trade count — results may not be statistically significant (< 30 trades)")
```

- [ ] **Step 4: Update optimize.py — pass slippage**

Read `src/dashboard/pages/optimize.py` and add `slippage` and `execution_mode` parameters to the `optimize()` call, matching the same sidebar pattern as backtest.py.

- [ ] **Step 5: Update walk_forward.py — hold-out toggle + display**

Read `src/dashboard/pages/walk_forward.py` and add:
- `holdout_enabled` checkbox in sidebar (default True)
- `holdout_pct` slider (default 0.10)
- `slippage` and `execution_mode` inputs
- Pass all to `run_walk_forward()`
- After results display, show hold-out section:

```python
if wf_result.holdout_metrics is not None:
    st.markdown("---")
    st.subheader("Hold-Out Validation (Unseen Data)")
    st.caption("These results are from data never seen during optimization or WFA")
    hm = wf_result.holdout_metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Return", f"{hm.get('total_return', 0):.2f}%")
    col2.metric("Sharpe", f"{hm.get('sharpe_ratio', 0):.2f}")
    col3.metric("Win Rate", f"{hm.get('win_rate', 0):.1f}%")
    col4.metric("Trades", f"{hm.get('total_trades', 0)}")

    if wf_result.holdout_equity is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wf_result.holdout_equity.index,
            y=wf_result.holdout_equity.values,
            mode="lines", name="Hold-Out Equity"
        ))
        fig.update_layout(height=300, title="Hold-Out Equity Curve")
        st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 6: Commit**

```bash
git add src/dashboard/pages/backtest.py src/dashboard/pages/optimize.py src/dashboard/pages/walk_forward.py
git commit -m "feat: add slippage, execution_mode, hold-out UI to dashboard"
```

---

### Task 9: Run Full Test Suite + Fix Issues

**Files:**
- All test files

- [ ] **Step 1: Run full test suite**

Run: `cd /home/mheloy/VectorBackTest && uv run pytest tests/ -v`

- [ ] **Step 2: Fix any failing tests**

Common issues to watch for:
- Tests that unpack `entries, exits = strategy.generate_signals(...)` need to handle `SignalResult`
- Simulator tests that don't pass `short_entries`/`short_exits` — should use defaults (None → empty arrays)
- Import errors for `SignalResult`

- [ ] **Step 3: Commit fixes**

```bash
git add tests/
git commit -m "fix: update tests for SignalResult and new params"
```

---

### Task 10: Documentation

**Files:**
- Modify: `docs/strategies/supertrend.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update supertrend.md**

Add to the parameter history table:
```
| 2026-03-19 | Added bidirectional (long+short) signals, 200MA filter, direction_mode, warmup guard. Replaced h1_filter with filter_type. Added slippage (0.000004) and calibrated fees (0.000006) from ECN trade data. Next-bar-open execution. |
```

Update the Parameters table to include new params: `filter_type`, `direction_mode`, `ma_filter_period`, `ma_filter_type`.

Remove `h1_filter` from the params table.

- [ ] **Step 2: Update CLAUDE.md**

Update the SuperTrend section in Strategies to reflect:
- Bidirectional (long+short) signals
- `filter_type` replaces `h1_filter`
- Default fees/slippage calibrated from ECN data
- Next-bar-open execution default
- Hold-out set in WFA

Update the Conventions section:
- `Fees default: 0.000006 (ECN commission fraction)`
- `Slippage default: 0.000004 (half-spread fraction)`
- `Execution: next-bar-open (default)`

- [ ] **Step 3: Commit**

```bash
git add docs/strategies/supertrend.md CLAUDE.md
git commit -m "docs: update SuperTrend and CLAUDE.md for bidirectional + cost model"
```
