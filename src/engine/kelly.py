"""Kelly Criterion and position sizing models."""

from dataclasses import dataclass


@dataclass
class KellyResult:
    """Result of Kelly Criterion calculation."""
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    full_kelly_pct: float
    half_kelly_pct: float
    recommended_risk_pct: float  # Half Kelly, clamped to [0, 25%]


def calculate_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,
) -> KellyResult:
    """Calculate Kelly Criterion position sizing.

    Args:
        win_rate: Probability of winning trade (0 to 1).
        avg_win: Average winning trade return (absolute value, e.g., 0.02 for 2%).
        avg_loss: Average losing trade return (absolute value, e.g., 0.01 for 1%).
        fraction: Kelly fraction to use (0.5 = Half Kelly recommended).

    Returns:
        KellyResult with full and fractional Kelly percentages.
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return KellyResult(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=0.0,
            full_kelly_pct=0.0,
            half_kelly_pct=0.0,
            recommended_risk_pct=0.0,
        )

    wl_ratio = avg_win / avg_loss
    # Kelly formula: f* = W - (1-W)/R
    full_kelly = win_rate - (1 - win_rate) / wl_ratio
    fractional_kelly = full_kelly * fraction

    # Clamp recommended to [0%, 25%]
    recommended = max(0.0, min(fractional_kelly * 100, 25.0))

    return KellyResult(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=wl_ratio,
        full_kelly_pct=full_kelly * 100,
        half_kelly_pct=fractional_kelly * 100,
        recommended_risk_pct=recommended,
    )


def kelly_from_metrics(metrics: dict, fraction: float = 0.5) -> KellyResult:
    """Calculate Kelly from extracted backtest metrics dict."""
    win_rate = metrics.get("win_rate", 0.0) / 100  # Convert from percentage
    avg_win = abs(metrics.get("avg_winning_trade", 0.0)) / 100
    avg_loss = abs(metrics.get("avg_losing_trade", 0.0)) / 100

    return calculate_kelly(win_rate, avg_win, avg_loss, fraction)
