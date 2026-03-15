"""Auto-discover all registered strategy classes."""

from .base import BaseStrategy
# Import strategy modules so their classes register as subclasses
from . import ma_crossover  # noqa: F401
from . import rsi_reversal  # noqa: F401
from . import bollinger_breakout  # noqa: F401


def get_all_strategies() -> dict[str, BaseStrategy]:
    """Return a dict of strategy_name -> strategy_instance for all registered strategies."""
    strategies = {}
    for cls in BaseStrategy.__subclasses__():
        instance = cls()
        strategies[instance.name] = instance
    return strategies
