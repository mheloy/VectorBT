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
| h1_filter | On | On/Off | - | H1 timeframe confirmation filter |
| h1_period | 16 | 5-50 | 1 | H1 SuperTrend ATR period |
| h1_factor | 1.4 | 0.5-5.0 | 0.1 | H1 SuperTrend multiplier |

## Parameter History

| Date | Parameter | Old | New | Notes |
|------|-----------|-----|-----|-------|
| 2025-03-15 | Initial | - | period=16, factor=1.4 | WFA-optimized defaults from live MT5 bot |

## Notes

- The live bot also uses complex position management (partial TPs, trailing SL, breakeven) which is not part of this strategy module — VectorBT handles SL/TP via the runner.
- H1 filter uses shift(1) before forward-fill to avoid look-ahead bias (only acts on completed H1 candles).
- Long-only for now. The live bot generates both BUY and SELL signals but only long is implemented here for consistency with other strategies.
