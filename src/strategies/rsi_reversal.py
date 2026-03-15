"""RSI Reversal strategy: buy oversold, sell overbought."""

import vectorbt as vbt
import pandas as pd

from .base import BaseStrategy, StrategyParam


class RSIReversal(BaseStrategy):
    """Buy when RSI crosses below oversold level, sell when it crosses above overbought."""

    @property
    def name(self) -> str:
        return "RSI Reversal"

    def parameters(self) -> list[StrategyParam]:
        return [
            StrategyParam(
                "rsi_period", default=14, min_val=5, max_val=50, step=1,
                description="RSI period",
            ),
            StrategyParam(
                "oversold", default=30, min_val=10, max_val=45, step=5,
                description="Oversold level",
            ),
            StrategyParam(
                "overbought", default=70, min_val=55, max_val=90, step=5,
                description="Overbought level",
            ),
        ]

    def generate_signals(
        self, df: pd.DataFrame, rsi_period=14, oversold=30, overbought=70
    ) -> tuple[pd.Series, pd.Series]:
        close = df["close"]
        rsi = vbt.RSI.run(close, window=int(rsi_period)).rsi

        # Buy when RSI crosses above oversold from below
        entries = rsi.vbt.crossed_above(oversold)
        # Sell when RSI crosses below overbought from above
        exits = rsi.vbt.crossed_below(overbought)

        return entries, exits
