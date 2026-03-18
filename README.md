# XAUUSD Backtest Engine

VectorBT-based backtest engine for Gold (XAUUSD) trading strategies. Optimizes signal-based strategies against 3+ years of 5-minute candle data with a Streamlit dashboard for visualization.

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure data path (point to directory containing XAUUSD_M5_*.csv files)
echo "DATA_DIR=/path/to/your/forex-data/data" > .env
```

### Data Requirements

The engine expects 5-minute OHLCV CSVs in the `DATA_DIR` directory matching the pattern `XAUUSD_M5_*.csv` with columns: `datetime,open,high,low,close,volume`. The loader concatenates all CSVs, deduplicates week boundaries, and caches as parquet for fast reloads. Resampling supports 5M/15M/30M/1H/4H/D timeframes.

## Starting the Dashboard

```bash
uv run streamlit run src/dashboard/app.py
```

Opens at **http://localhost:8501**. The sidebar navigation has 3 sections with 8 pages total.

To stop, press `Ctrl+C` in the terminal.

## How to Use

### 1. Run a Backtest

1. Open the **Run Backtest** page (default landing page)
2. In the sidebar, select a **Strategy** (MA Crossover, RSI Reversal, Bollinger Breakout, or SuperTrend)
3. Select a **Timeframe** (5M, 15M, 30M, 1H, 4H, D) — 5M is the default for SuperTrend
4. Adjust **strategy parameters** using the sliders (each strategy has different params)
5. Set **Initial Cash** and **Fees** under Risk Management
6. For SuperTrend: toggle **Advanced PM** to "On" to enable partial take profits, break-even, and trailing stop loss (replicates live MT5 bot behavior)
7. Click **Run Backtest**
8. View results: metrics cards, equity curve, drawdown chart, and trade table
9. Click **Save Results** to persist the run to SQLite for later comparison

### 2. Optimize Parameters

1. Open the **Optimize** page
2. Select strategy and timeframe
3. Choose 1 or 2 parameters to sweep (X axis and optional Y axis)
4. Set the range and step for each sweep parameter
5. Pick the **target metric** to optimize (Sharpe, return, win rate, etc.)
6. Click **Run Optimization**
7. View the results table sorted by target metric, and a 2D heatmap if sweeping 2 params
8. Use heatmap to identify stable parameter regions (not just the single best point)

### 3. Walk-Forward Analysis

1. Open the **Walk-Forward** page
2. Configure the same strategy/timeframe and sweep parameters as optimization
3. Set number of **OOS windows** (default 8 — tiles the entire dataset)
4. Click **Run Walk-Forward**
5. Compare the **WF OOS equity curve** (blue) vs **full-sample optimized** (orange)
   - If OOS tracks full-sample closely, the strategy is robust
   - If OOS is much worse, the strategy is likely overfit
6. Check the per-window table for parameter stability across windows

### 4. Monte Carlo Simulation

1. Open the **Monte Carlo** page
2. Set number of simulations (1000 default), ruin threshold, and strategy params
3. Click **Run Monte Carlo**
4. View the equity fan chart with confidence bands (5th-95th percentile)
5. Check **ruin probability** — the % chance of account dropping below the ruin threshold
6. Review max drawdown distribution to understand worst-case scenarios

### 5. Trade Analysis

1. Open the **Trade Analysis** page
2. Select strategy/timeframe and click **Analyze Trades**
3. Review:
   - **R-Multiples**: Distribution of trade PnL as multiples of risk (1R = initial SL distance)
   - **MAE/MFE**: Max adverse/favorable excursion per trade — helps tune SL/TP placement
   - **Streaks**: Consecutive win/loss streaks and averages
   - **Session Performance**: Win rate and PnL by Gold trading session (Asian, London, NY, Overlap)
   - **Duration vs PnL**: Whether quick or long trades perform better

### 6. Regime Analysis

1. Open the **Regime Analysis** page
2. Choose detection method: **HMM** (Hidden Markov Model) or **ADX** (simpler fallback)
3. Select number of regimes (2-3)
4. Click **Detect Regimes** to see the price chart with regime overlay
5. Click **Backtest by Regime** to run the strategy only in selected regimes
6. Compare per-regime metrics to identify which market conditions suit the strategy

### 7. Compare Runs

1. Run and save multiple backtests with different strategies or parameters
2. Open **Saved Results** to see all persisted runs with key metrics
3. Open **Compare Runs** to select 2+ runs and overlay their equity curves

## Strategies

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| MA Crossover | SMA/EMA crossover entries/exits | fast_period, slow_period, ma_type |
| RSI Reversal | Overbought/oversold RSI signals | rsi_period, oversold, overbought |
| Bollinger Breakout | Price breaking Bollinger Bands | bb_period, bb_std |
| SuperTrend | Trend-following with H1 filter, ATR stops, optional advanced PM | period, factor, source, h1_filter, sl_atr_mult, adv_pm |

### Adding a New Strategy

1. Create a new file in `src/strategies/` (e.g., `my_strategy.py`)
2. Subclass `BaseStrategy` and implement:
   - `name` property — human-readable name
   - `parameters()` — list of `StrategyParam` defining tunable params with ranges
   - `generate_signals(df, **params)` — returns `(entries, exits)` as boolean Series
   - Optionally: `compute_stops(df, **params)` for dynamic SL/TP
   - Optionally: `position_management(**params)` for advanced PM (partial TP, BE, trailing)
3. Import the module in `src/strategies/registry.py`
4. The strategy auto-appears in all dashboard pages

### SuperTrend Advanced Position Management

When `adv_pm=On`, the SuperTrend strategy uses a custom Numba simulator instead of VBT's `from_signals()`, replicating the live MT5 bot's position management:

- **Partial TP1**: Close 33% of position at 1.2R profit (triggers break-even)
- **Partial TP2**: Close 50% at 2.0R profit
- **Runner**: Remaining portion trails SuperTrend line (st_line mode) or uses 3-stage ATR trailing
- **Break-Even**: SL moves to entry price when TP1 is hit (auto-triggered)
- **3-Stage Trailing SL**: Progressively tightens as profit grows (1.0x → 0.8x → 0.6x of initial SL distance)

All PM parameters are individually tunable and optimizable via grid search.

### SuperTrend Defaults (Aligned with backtest-engine)

Current defaults match the profitable configuration from the Node.js backtest-engine:

```
source=hl2, period=17, factor=1.8, atr_method=sma
sl_atr_mult=1.9, h1_filter=On (same params as entry)
adv_pm=On recommended: tp1_r=1.2, tp1_pct=0.33, tp2_r=2.0, tp2_pct=0.50
trail_mode=st_line, risk_pct=0.03 (3% per trade)
```

Equivalent backtest-engine command:
```bash
node src/backtest.js \
  --source hl2 --band ATR --period 17 --factor 1.8 --atr-mult 1.9 \
  --htf-confirm \
  --partials --tp1-r 1.2 --tp1-pct 33 --tp2-r 2.0 --tp2-pct 50
```

## Cross-Engine Alignment Notes

This engine was aligned with the Node.js [backtest-engine](../backtest-engine) to produce comparable results. Key findings from the alignment process (2026-03-18):

### Changes Made

| Area | Before | After (aligned) | Impact |
|------|--------|-----------------|--------|
| ATR smoothing | Wilder's RMA | **SMA** (rolling mean of TR) | HIGH — different band widths, signals, SL distances |
| SL ATR timeframe | Resampled to M15 | **Entry timeframe** (e.g. M5) | HIGH — M15 ATR ~3x larger, causing oversized stops |
| Default params | hlc3/P15/F1.5/atr_mult=2.5 | **hl2/P17/F1.8/atr_mult=1.9** | Config alignment |
| Default TP levels | tp1=0.5R, tp2=1.5R | **tp1=1.2R, tp2=2.0R** | Config alignment |
| Dashboard timeframe | 1H | **5M** | Matches entry signal timeframe |

### Known Remaining Differences

| Difference | backtest-engine | VectorBackTest | Impact |
|------------|-----------------|----------------|--------|
| TP2 fraction basis | % of **remaining** lots | % of **initial** position | Runner is 33.5% vs 17% of position |
| Runner TP cap | None (trails ST line only) | Capped at final_tp_r=3.0 | Limits upside on runners |
| Trail update order | Update trail BEFORE SL check | SL check first, trail after | 1-bar lag on trail exits |
| BE offset | $0 (exact entry) | $1 above entry | Minor |
| Fee model | Zero (no spread/commission) | 0.0001 (1bp per side) | **Destroys profitability with M5 ATR sizing** |

### Fee Sensitivity

With M5-ATR-based position sizing, the position notional is large relative to account equity. The default 1bp fee (`fees=0.0001`) compounds aggressively across thousands of partial exits and completely overwhelms the edge:

| Fee Setting | Total Return | Win Rate | Profit Factor | Max DD |
|-------------|-------------|----------|---------------|--------|
| 0 (matches backtest-engine) | +3,347% | 50.8% | 1.16 | 74.4% |
| 0.0001 (1bp default) | -100% | 50.8% | 0.90 | 100% |

**Recommendation**: When comparing with the backtest-engine, use `fees=0`. When modelling realistic execution, calibrate fees to actual broker spread/commission for XAUUSD M5 trades.

## Running Tests

```bash
uv run pytest           # Run all tests
uv run pytest -v        # Verbose output
uv run pytest tests/test_simulator.py  # Run simulator tests only
```

## Project Structure

```
src/
  data/loader.py          # CSV loading, resampling, parquet cache
  strategies/             # Strategy definitions (BaseStrategy ABC)
  engine/
    runner.py             # Backtest runner (dual-path: VBT or custom simulator)
    optimizer.py          # Vectorized grid search
    metrics.py            # Metric extraction from Portfolio
    kelly.py              # Kelly Criterion position sizing
    monte_carlo.py        # Trade shuffle Monte Carlo
    robustness.py         # Noise injection, signal delay, param sensitivity
    walk_forward.py       # Walk-forward optimization (tiled OOS)
    regime.py             # HMM/ADX regime detection
    trade_analysis.py     # R-multiples, MAE/MFE, streaks, sessions
    position_management.py # PM config dataclasses
    simulator.py          # Numba JIT bar-by-bar simulator
    sim_result.py         # SimulationResult + BacktestResult wrappers
  storage/
    db.py                 # SQLite persistence
    models.py             # Result dataclasses
  dashboard/
    app.py                # Streamlit entry point
    pages/                # Dashboard pages (8 pages)
results/                  # SQLite DB + parquet cache (gitignored)
docs/strategies/          # Strategy documentation + parameter history
```

## Key Design Decisions

- **Fee model**: Default 1bp/side (`fees=0.0001`), approximating XAUUSD spread on MT5. Set to 0 for backtest-engine parity.
- **ATR method**: SMA (default, matches backtest-engine) or RMA (Wilder's, matches PineScript). Configurable via `atr_method` parameter.
- **SL sizing**: ATR(14) on entry timeframe (e.g. M5), not resampled. Matches backtest-engine behaviour.
- **No volume indicators**: Volume is always 0 from Twelve Data
- **Vectorized optimization**: Multi-column DataFrame passed to single `Portfolio.from_signals()` call
- **Dual-path runner**: Simple strategies use VBT; PM strategies use custom Numba simulator (~5-15ms per 236K bars)
- **Walk-forward**: Tiled OOS windows cover 100% of data with no gaps
- **Kelly sizing**: Half Kelly shown as primary recommendation
- **Gold sessions**: Asian (23:00-07:00 GMT), London (08:00-16:00), NY (13:00-21:00)
