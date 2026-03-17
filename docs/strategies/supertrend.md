# SuperTrend Strategy

## Overview

Trend-following strategy based on the SuperTrend indicator. Ported from live MT5 bot (`/home/mheloy/SuperTrendMT5Linux/`). Enters long when SuperTrend flips to uptrend, exits when it flips to downtrend. Optional H1 timeframe filter confirms trend direction before entry.

## Indicator Logic

1. **Price source** (configurable): hl2, close, hlc3, or ohlc4
2. **ATR** calculated using Wilder's RMA smoothing (matches PineScript `ta.atr()`)
3. **Bands**: `upper = source + factor * ATR`, `lower = source - factor * ATR`
4. **Band ratcheting**: upper band only moves down, lower band only moves up (prevents whipsaws)
5. **Direction**: -1 = uptrend (price above ST), +1 = downtrend (price below ST)

## Signal Rules

- **Entry (long)**: Direction flips from +1 to -1 (downtrend -> uptrend)
- **Exit**: Direction flips from -1 to +1 (uptrend -> downtrend)
- **H1 Filter**: When enabled, only enters long if H1 SuperTrend is also in uptrend (-1)

## Parameters

| Parameter | Default | Range | Step | Description |
|-----------|---------|-------|------|-------------|
| period | 16 | 5-50 | 1 | ATR period for SuperTrend bands |
| factor | 1.4 | 0.5-5.0 | 0.1 | ATR multiplier for band width |
| source | hl2 | hl2/close/hlc3/ohlc4 | - | Price source for band center |
| h1_filter | On | On/Off | - | H1 timeframe confirmation filter (uses same period/factor/source) |
| sl_atr_mult | 1.5 | 0.5-5.0 | 0.1 | SL = ATR(14, M15) * multiplier |
| rr_ratio | 3.0 | 1.0-5.0 | 0.5 | TP = SL * R:R ratio |
| adv_pm | Off | On/Off | - | Advanced position management (partial TP, BE, trailing) |
| tp1_r | 1.5 | 0.5-5.0 | 0.1 | Partial TP1 R-multiple trigger |
| tp1_pct | 0.50 | 0.1-0.9 | 0.05 | TP1 close fraction of initial position |
| tp2_r | 2.9 | 1.0-8.0 | 0.1 | Partial TP2 R-multiple trigger |
| tp2_pct | 0.30 | 0.1-0.5 | 0.05 | TP2 close fraction of initial position |
| be_trigger_r | 1.0 | 0.3-3.0 | 0.1 | Break-even trigger R-multiple |
| final_tp_r | 3.0 | 1.5-10.0 | 0.5 | Final TP R-multiple for runner portion |

## Risk Management

### Simple Mode (adv_pm=Off)
Dynamic ATR-based SL/TP via VBT's `from_signals()`:
- **Stop Loss**: ATR(14) on M15 timeframe * `sl_atr_mult` (default 1.5x), converted to % of entry price per bar
- **Take Profit**: SL distance * `rr_ratio` (default 3.0 R:R)
- M15 ATR is shifted by 1 bar to avoid look-ahead bias (uses completed M15 candle)

### Advanced PM Mode (adv_pm=On)
Custom Numba JIT bar-by-bar simulator replicating the live MT5 bot's position management:
- **1R** = ATR(14, M15) * sl_atr_mult (initial SL distance)
- **Partial TP1**: Close 50% at 1.5R, auto-triggers break-even
- **Partial TP2**: Close 30% at 2.9R
- **Runner**: Remaining 20% trails to final TP (3.0R) or trailing SL
- **Break-Even**: At 1.0R, SL moves to entry + $1 offset
- **Trailing SL (3-stage)**:
  - Stage 1: 0.67R -> trail at 1.0x initial SL distance
  - Stage 2: 1.0R -> trail at 0.8x (tighter)
  - Stage 3: 1.33R -> trail at 0.6x (tightest)
- Reference: previous bar body, SL only ratchets in profitable direction
- All PM parameters are optimizable via the grid search optimizer

## Parameter History

| Date | Parameter | Old | New | Notes |
|------|-----------|-----|-----|-------|
| 2025-03-15 | Initial | - | period=16, factor=1.4 | WFA-optimized defaults from live MT5 bot |
| 2026-03-15 | SL/TP | fixed % | ATR-based dynamic | Added sl_atr_mult=1.5, rr_ratio=3.0 matching live bot |
| 2026-03-18 | Position Mgmt | approximated | full replication | Added Numba simulator with partial TP, BE, trailing SL matching live MT5 bot |

## Notes

- H1 filter uses shift(1) before forward-fill to avoid look-ahead bias (only acts on completed H1 candles).
- Long-only for now. The live bot generates both BUY and SELL signals but only long is implemented here for consistency with other strategies.
- When `adv_pm=On`, the VBT `from_signals()` path is bypassed entirely — all position management is handled by the custom simulator in `src/engine/simulator.py`.
- The simulator within-bar execution order is conservative: SL checked first, then partials, then BE, trailing, final TP, signal exit. When SL and TP could both trigger on the same bar, SL takes priority.
- Live bot uses M1 candle body for trailing reference; backtester uses the previous bar at whatever timeframe is selected (simplification due to bar-level simulation).
