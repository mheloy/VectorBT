# CLAUDE.md

## Project Overview
VectorBT-based backtest engine for XAUUSD (Gold) trading strategies. Single-user tool for parameter optimization and strategy validation.

## Commands
- `uv sync` — install dependencies
- `uv run streamlit run src/dashboard/app.py` — start dashboard
- `uv run pytest` — run tests

## Architecture
- **Engine layer** (`src/engine/`): Pure computation, no UI. Each module returns a dataclass result.
- **Strategy layer** (`src/strategies/`): `BaseStrategy` ABC with `generate_signals(df, **params) -> (entries, exits)`. Registry auto-discovers via `__subclasses__()`.
- **Storage** (`src/storage/`): SQLite for metadata + JSON blobs for equity curves and trade lists.
- **Dashboard** (`src/dashboard/`): Streamlit multipage app. Pages import from `src.*` (sys.path set in `app.py`).

## Key Patterns
- Strategies return boolean pd.Series for entries/exits — VectorBT handles position management
- Optimizer stacks all param combos into multi-column DataFrames for a single `vbt.Portfolio.from_signals()` call
- Walk-forward uses custom tiled OOS approach (not VBT's RollingSplitter) for full data coverage
- Monte Carlo, regime analysis, and robustness testing are custom implementations (not in VBT open-source)
- `StrategyParam` dataclass drives both dashboard sliders and optimizer grid ranges

## Data
- Source: `DATA_DIR` env var pointing to directory of `XAUUSD_M5_*.csv` files
- Loader caches concatenated data as parquet in `results/` directory
- Resampling: 5M/15M/30M/1H/4H/D via pandas resample

## Conventions
- Fees default: `0.0001` (1bp per side)
- Init cash default: `$10,000`
- Frequency strings for VBT: `{"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h", "D": "1D"}`
- Gold sessions (GMT): Asian 23:00-07:00, London 08:00-16:00, NY 13:00-21:00
