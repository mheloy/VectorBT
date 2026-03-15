"""Bollinger Bands Breakout strategy."""

import vectorbt as vbt
import pandas as pd

from .base import BaseStrategy, StrategyParam


class BollingerBreakout(BaseStrategy):
    """Buy when price breaks above upper band, sell when it breaks below lower band."""

    @property
    def name(self) -> str:
        return "Bollinger Breakout"

    def parameters(self) -> list[StrategyParam]:
        return [
            StrategyParam(
                "bb_period", default=20, min_val=10, max_val=50, step=5,
                description="Bollinger Bands period",
            ),
            StrategyParam(
                "bb_std", default=2.0, min_val=1.0, max_val=3.0, step=0.5,
                description="Standard deviations",
            ),
        ]

    def generate_signals(
        self, df: pd.DataFrame, bb_period=20, bb_std=2
    ) -> tuple[pd.Series, pd.Series]:
        close = df["close"]
        bb = vbt.BBANDS.run(close, window=int(bb_period), alpha=float(bb_std))

        # Buy when close crosses above upper band
        entries = close.vbt.crossed_above(bb.upper)
        # Sell when close crosses below lower band
        exits = close.vbt.crossed_below(bb.lower)

        return entries, exits
