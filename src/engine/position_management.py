"""Position management configuration for advanced trade management.

Defines dataclasses for partial take profits, break-even, and trailing stop loss
that mirror the live MT5 bot's position management logic.
"""

from dataclasses import dataclass, field


@dataclass
class PartialTPConfig:
    """Configuration for a single partial take-profit level."""

    trigger_r: float  # R-multiple threshold to trigger (e.g., 1.2)
    close_pct: float  # Fraction to close: of initial (TP1) or of remaining (TP2+)


@dataclass
class TrailingStageConfig:
    """Configuration for a single trailing stop stage."""

    trigger_r: float  # R-multiple threshold to activate this stage
    sl_multiplier: float  # Trail at this fraction of initial SL distance


@dataclass
class PositionManagementConfig:
    """Full position management configuration.

    Matches the live MT5 bot defaults from ~/SuperTrendMT5Linux/config.py.
    All distance thresholds are expressed in R-multiples where 1R = initial SL distance.
    """

    # Partial take profits (ordered by trigger level)
    partial_tps: list[PartialTPConfig] = field(default_factory=lambda: [
        PartialTPConfig(trigger_r=1.2, close_pct=0.33),  # TP1: 33% of initial at 1.2R
        PartialTPConfig(trigger_r=2.0, close_pct=0.50),  # TP2: 50% of remaining at 2.0R
    ])
    partial_tp_enabled: bool = True

    # Break-even
    be_enabled: bool = True
    be_trigger_r: float = 1.0  # R-multiple to trigger BE move
    be_offset_dollars: float = 0.0  # SL moves to entry + this offset ($)

    # Trailing stop loss stages (ordered by trigger level)
    trailing_stages: list[TrailingStageConfig] = field(default_factory=lambda: [
        TrailingStageConfig(trigger_r=0.67, sl_multiplier=1.0),  # Stage 1: full distance
        TrailingStageConfig(trigger_r=1.0, sl_multiplier=0.8),  # Stage 2: tighter
        TrailingStageConfig(trigger_r=1.33, sl_multiplier=0.6),  # Stage 3: tightest
    ])
    trailing_sl_enabled: bool = True

    # Final take profit for the runner portion (0 = no cap, trails ST line only)
    final_tp_r: float = 0.0

    # Runner trailing mode: "st_line" (trail SuperTrend line) or "atr_stages" (3-stage ATR)
    trail_mode: str = "st_line"

    # Position sizing
    sizing_mode: str = "risk_pct"  # "risk_pct" or "fixed_lot"
    risk_pct: float = 0.03  # Risk 3% of equity per trade (when sizing_mode="risk_pct")
    fixed_lot_units: float = 1.0  # Fixed units per trade (when sizing_mode="fixed_lot")
    max_lot_value: float = 0.0  # Max notional per trade in $ (0 = no limit)
