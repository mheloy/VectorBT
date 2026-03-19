"""Base strategy class and parameter definition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategyParam:
    """Describes a tunable strategy parameter.

    Drives dashboard UI (sliders/dropdowns) and optimizer (grid ranges).
    """
    name: str
    default: Any
    min_val: Any = None
    max_val: Any = None
    step: Any = None
    description: str = ""
    choices: list[Any] | None = None  # For categorical params (e.g., MA type)


@dataclass
class SignalResult:
    """Result of signal generation, supporting both long-only and bidirectional strategies."""
    entries: pd.Series              # long entries (always required)
    exits: pd.Series                # long exits (always required)
    short_entries: pd.Series | None = None
    short_exits: pd.Series | None = None


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @abstractmethod
    def parameters(self) -> list[StrategyParam]:
        """Declare tunable parameters with ranges for optimization."""
        ...

    @abstractmethod
    def generate_signals(
        self, df: pd.DataFrame, **params
    ) -> SignalResult:
        """Generate entry and exit signals.

        Args:
            df: OHLCV DataFrame with datetime index.
            **params: Strategy parameter values.

        Returns:
            SignalResult with entries/exits (and optional short_entries/short_exits).
        """
        ...

    def compute_stops(
        self, df: pd.DataFrame, **params
    ) -> tuple[pd.Series, pd.Series] | None:
        """Compute per-bar SL/TP as fraction of entry price.

        Override in strategies that provide dynamic stop-loss and take-profit
        (e.g., ATR-based stops). Returns None if the strategy uses fixed stops.

        Returns:
            Tuple of (sl_stop, tp_stop) as pd.Series of fractions, or None.
        """
        return None

    def position_management(self, **params):
        """Return advanced position management config, or None for simple SL/TP.

        Override in strategies that need partial TPs, break-even, or trailing SL.
        Returns a PositionManagementConfig or None.
        """
        return None

    def default_params(self) -> dict[str, Any]:
        """Get default parameter values."""
        return {p.name: p.default for p in self.parameters()}
