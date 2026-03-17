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
3. Select a **Timeframe** (5M, 15M, 30M, 1H, 4H, D) — 1H is a good starting point
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

- **Partial TP1**: Close 50% of position at 1.5R profit
- **Partial TP2**: Close 30% at 2.9R profit
- **Runner**: Remaining 20% trails to final TP (3.0R) or trailing SL
- **Break-Even**: SL moves to entry + $1 when price reaches 1.0R (auto-triggers on TP1)
- **3-Stage Trailing SL**: Progressively tightens as profit grows (1.0x → 0.8x → 0.6x of initial SL distance)

All PM parameters are individually tunable and optimizable via grid search.

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

- **Fee model**: Default 1bp/side (`fees=0.0001`), approximating XAUUSD spread on MT5
- **No volume indicators**: Volume is always 0 from Twelve Data
- **Vectorized optimization**: Multi-column DataFrame passed to single `Portfolio.from_signals()` call
- **Dual-path runner**: Simple strategies use VBT; PM strategies use custom Numba simulator (~5-15ms per 236K bars)
- **Walk-forward**: Tiled OOS windows cover 100% of data with no gaps
- **Kelly sizing**: Half Kelly shown as primary recommendation
- **Gold sessions**: Asian (23:00-07:00 GMT), London (08:00-16:00), NY (13:00-21:00)
