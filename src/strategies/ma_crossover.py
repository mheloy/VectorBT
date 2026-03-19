"""Moving Average Crossover strategy."""

import vectorbt as vbt
import pandas as pd

from .base import BaseStrategy, StrategyParam, SignalResult


class MACrossover(BaseStrategy):
    """Buy when fast MA crosses above slow MA, sell when it crosses below."""

    @property
    def name(self) -> str:
        return "MA Crossover"

    def parameters(self) -> list[StrategyParam]:
        return [
            StrategyParam(
                "fast_period", default=10, min_val=5, max_val=50, step=5,
                description="Fast moving average period",
            ),
            StrategyParam(
                "slow_period", default=50, min_val=20, max_val=200, step=10,
                description="Slow moving average period",
            ),
            StrategyParam(
                "ma_type", default="SMA", choices=["SMA", "EMA"],
                description="Moving average type",
            ),
        ]

    def generate_signals(
        self, df: pd.DataFrame, fast_period=10, slow_period=50, ma_type="SMA"
    ) -> SignalResult:
        close = df["close"]
        ewm = ma_type.upper() == "EMA"

        fast_ma = vbt.MA.run(close, window=int(fast_period), ewm=ewm).ma
        slow_ma = vbt.MA.run(close, window=int(slow_period), ewm=ewm).ma

        entries = fast_ma.vbt.crossed_above(slow_ma)
        exits = fast_ma.vbt.crossed_below(slow_ma)

        return SignalResult(entries=entries, exits=exits)
