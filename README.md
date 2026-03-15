# XAUUSD Backtest Engine

VectorBT-based backtest engine for Gold (XAUUSD) trading strategies. Optimizes signal-based strategies against 3+ years of 5-minute candle data with a Streamlit dashboard for visualization.

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure data path
echo "DATA_DIR=/path/to/your/forex-data/data" > .env
```

## Running the Dashboard

```bash
uv run streamlit run src/dashboard/app.py
```

Opens at http://localhost:8501 with pages for:
- **Run Backtest** — single strategy run with equity curve, drawdown, metrics, Kelly sizing, trade table
- **Optimize** — grid search over parameter ranges with heatmap visualization
- **Monte Carlo** — trade shuffle simulation with confidence bands and ruin probability
- **Walk-Forward** — rolling IS/OOS optimization to detect overfitting
- **Trade Analysis** — R-multiples, MAE/MFE, streaks, session performance, exposure
- **Regime Analysis** — HMM/ADX regime detection with conditional backtesting
- **Saved Results** — browse and manage persisted backtest runs
- **Compare Runs** — overlay equity curves and metrics from multiple runs

## Data

Expects 5-minute OHLCV CSVs in `DATA_DIR` matching the pattern `XAUUSD_M5_*.csv` with columns: `datetime,open,high,low,close,volume`. The loader concatenates all CSVs, deduplicates, and caches as parquet. Resampling supports 5M/15M/30M/1H/4H/D timeframes.

## Strategies

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| MA Crossover | SMA/EMA crossover entries/exits | fast_period, slow_period, ma_type |
| RSI Reversal | Overbought/oversold RSI signals | rsi_period, oversold, overbought |
| Bollinger Breakout | Price breaking Bollinger Bands | bb_period, bb_std |

Add new strategies by subclassing `BaseStrategy` in `src/strategies/` and importing in `registry.py`.

## Project Structure

```
src/
  data/loader.py          # CSV loading, resampling, parquet cache
  strategies/             # Strategy definitions (BaseStrategy ABC)
  engine/
    runner.py             # Single backtest via Portfolio.from_signals()
    optimizer.py          # Vectorized grid search
    metrics.py            # Metric extraction from Portfolio
    kelly.py              # Kelly Criterion position sizing
    monte_carlo.py        # Trade shuffle Monte Carlo
    robustness.py         # Noise injection, signal delay, param sensitivity
    walk_forward.py       # Walk-forward optimization (tiled OOS)
    regime.py             # HMM/ADX regime detection
    trade_analysis.py     # R-multiples, MAE/MFE, streaks, sessions
  storage/
    db.py                 # SQLite persistence
    models.py             # Result dataclasses
  dashboard/
    app.py                # Streamlit entry point
    pages/                # Dashboard pages
    components/           # Reusable chart components
results/                  # SQLite DB + parquet cache (gitignored)
```

## Key Design Decisions

- **Fee model**: Default 1bp/side (`fees=0.0001`), approximating XAUUSD spread on MT5
- **No volume indicators**: Volume is always 0 from Twelve Data
- **Vectorized optimization**: Multi-column DataFrame passed to single `Portfolio.from_signals()` call
- **Walk-forward**: Tiled OOS windows cover 100% of data with no gaps
- **Kelly sizing**: Half Kelly shown as primary recommendation
- **Gold sessions**: Asian (23:00-07:00 GMT), London (08:00-16:00), NY (13:00-21:00)
