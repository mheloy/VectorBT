# CLAUDE.md

## Project Overview
VectorBT-based backtest engine for XAUUSD (Gold) trading strategies. Single-user tool for parameter optimization and strategy validation.

## Commands
- `uv sync` — install dependencies
- `uv run streamlit run src/dashboard/app.py` — start dashboard
- `uv run pytest` — run tests

## Architecture
- **Engine layer** (`src/engine/`): Pure computation, no UI. Each module returns a dataclass result.
- **Strategy layer** (`src/strategies/`): `BaseStrategy` ABC with `generate_signals(df, **params) -> SignalResult`. `SignalResult` contains `entries`, `exits`, and optional `short_entries`/`short_exits`. Registry auto-discovers via `__subclasses__()`.
- **Storage** (`src/storage/`): SQLite for metadata + JSON blobs for equity curves and trade lists.
- **Dashboard** (`src/dashboard/`): Streamlit multipage app. Pages import from `src.*` (sys.path set in `app.py`).

## Strategies
- **MA Crossover** (`ma_crossover.py`): Fast/slow MA crossover with SMA/EMA choice
- **RSI Reversal** (`rsi_reversal.py`): RSI oversold/overbought reversals
- **Bollinger Breakout** (`bollinger_breakout.py`): Bollinger Bands upper/lower breakout
- **SuperTrend** (`supertrend.py`): SuperTrend trend-following with bidirectional (long+short) signals and ATR-based dynamic SL/TP. Ported from live MT5 bot. `filter_type` controls trend confirmation (h1_supertrend, 200ma, or none — replaces old `h1_filter`). `direction_mode` controls signal direction (both/long_only/short_only). 200MA filter available as alternative to H1 SuperTrend. Supports advanced position management (partial TP, break-even, trailing SL) via custom Numba simulator. ATR uses SMA by default (matching backtest-engine; RMA/Wilder's available via `atr_method` param). SL uses ATR(14) on entry timeframe (not resampled). Defaults: period=20, factor=1.2, source=hl2, atr_method=sma, sl_atr_mult=1.0, adv_pm=On, tp1_r=1.2, tp2_r=2.0 (WFA-optimized 2026-03-19). See `docs/strategies/supertrend.md` for parameter history.

## Key Patterns
- Strategies return boolean pd.Series for entries/exits — VectorBT handles position management
- Strategies can optionally override `compute_stops(df, **params)` to provide per-bar dynamic SL/TP (e.g., ATR-based) as fraction of entry price
- Strategies can optionally override `position_management(**params)` to enable advanced PM (partial TP, break-even, trailing SL) — routes to custom Numba simulator instead of VBT
- **Dual-path runner**: `run_backtest()` returns `BacktestResult` which wraps either VBT Portfolio (simple strategies) or `SimulationResult` (PM strategies) with a unified interface
- Optimizer stacks all param combos into multi-column DataFrames for a single `vbt.Portfolio.from_signals()` call (or per-combo simulation for PM strategies)
- Walk-forward uses custom tiled OOS approach (not VBT's RollingSplitter) for full data coverage; supports hold-out set (last 10% of data by default). WFA metrics use position-size-independent measures (profit factor, win rate, Sharpe) instead of return % which is distorted by compound risk-based sizing. Default sweep centered on WFA-optimized values: period=[16-24], factor=[0.8-1.6], sl_atr_mult=[0.6-1.4] (5 values each, 125 combos)
- Warmup guard suppresses signals during first `max(period, 14) + 1` bars to avoid indicator instability
- Monte Carlo, regime analysis, and robustness testing are custom implementations (not in VBT open-source)
- `StrategyParam` dataclass drives both dashboard sliders and optimizer grid ranges

## Data
- Source: `DATA_DIR` env var pointing to directory of `XAUUSD_M5_*.csv` files
- Loader caches concatenated data as parquet in `results/` directory
- Resampling: 5M/15M/30M/1H/4H/D via pandas resample

## Conventions
- Fees default: `0.000006` (ECN commission fraction, ~$3/lot at $5000 gold)
- Slippage default: `0.000004` (half-spread fraction, ~4pt spread on XAUUSD)
- Execution: next-bar-open (default, realistic); same-bar-close available for comparison
- Init cash default: `$10,000`
- Frequency strings for VBT: `{"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h", "D": "1D"}`
- Gold sessions (GMT): Asian 23:00-07:00, London 08:00-16:00, NY 13:00-21:00

## System Instructions
When I say update the markdown files, you need to update the appropriate strategy markdown files you have just made changes and make of the history of the changes made so we can study or find the best parameters for that specific stratgey. Then update the CLAUDE.md file to update the dcoumentaiton of the current strategy implemented. Also Apply best practices with git. Maybe every new session would be a different branch to ensure it can be rolled back etc.
