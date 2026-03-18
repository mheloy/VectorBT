"""Market regime detection and regime-conditional backtesting.

Supports HMM-based regime detection and simpler ADX/ATR fallback.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import vectorbt as vbt


class RegimeMethod(str, Enum):
    HMM = "HMM"
    ADX = "ADX"


@dataclass
class RegimeResult:
    """Result of regime analysis."""
    regime_labels: pd.Series  # Per-bar regime label
    regime_names: dict[int, str]  # regime_id -> name
    regime_colors: dict[int, str]
    per_regime_stats: pd.DataFrame  # Metrics per regime
    transition_matrix: pd.DataFrame  # Regime transition probabilities


@dataclass
class RegimeBacktestResult:
    """Result of regime-conditional backtesting."""
    regime_result: RegimeResult
    # Filtered backtest results per regime
    per_regime_metrics: pd.DataFrame
    # Full backtest for comparison
    full_metrics: dict
    full_portfolio: object


def detect_regimes(
    df: pd.DataFrame,
    method: RegimeMethod = RegimeMethod.HMM,
    n_regimes: int = 3,
    lookback: int = 14,
) -> RegimeResult:
    """Detect market regimes.

    Args:
        df: OHLCV DataFrame.
        method: Detection method (HMM or ADX).
        n_regimes: Number of regimes for HMM.
        lookback: Period for indicators (ADX, ATR).
    """
    if method == RegimeMethod.HMM:
        return _detect_hmm(df, n_regimes)
    else:
        return _detect_adx(df, lookback)


def _detect_hmm(df: pd.DataFrame, n_regimes: int = 3) -> RegimeResult:
    """HMM-based regime detection using returns and volatility."""
    from hmmlearn.hmm import GaussianHMM

    returns = df["close"].pct_change().dropna()
    volatility = returns.rolling(20).std().dropna()

    # Align
    common_idx = returns.index.intersection(volatility.index)
    features = np.column_stack([
        returns.loc[common_idx].values,
        volatility.loc[common_idx].values,
    ])

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(features)
    hidden_states = model.predict(features)

    # Create full-length label series (NaN for missing bars)
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    labels.loc[common_idx] = hidden_states
    labels = labels.ffill().bfill().astype(int)

    # Sort regimes by mean volatility (0=low vol, 1=medium, 2=high vol)
    regime_vols = {}
    for r in range(n_regimes):
        mask = hidden_states == r
        if mask.any():
            regime_vols[r] = volatility.loc[common_idx].values[mask].mean()
        else:
            regime_vols[r] = 0

    sorted_regimes = sorted(regime_vols.keys(), key=lambda r: regime_vols[r])
    remap = {old: new for new, old in enumerate(sorted_regimes)}
    labels = labels.map(remap)

    names = {0: "Low Volatility", 1: "Normal", 2: "High Volatility"}
    colors = {0: "green", 1: "blue", 2: "red"}
    if n_regimes == 2:
        names = {0: "Low Volatility", 1: "High Volatility"}
        colors = {0: "green", 1: "red"}

    # Per-regime stats
    stats_data = []
    for r in range(n_regimes):
        mask = labels == r
        regime_returns = df["close"].pct_change().loc[mask].dropna()
        stats_data.append({
            "Regime": names.get(r, f"Regime {r}"),
            "Bars": int(mask.sum()),
            "% of Time": f"{mask.mean() * 100:.1f}%",
            "Mean Return": f"{regime_returns.mean() * 100:.4f}%",
            "Volatility": f"{regime_returns.std() * 100:.4f}%",
            "Sharpe (approx)": f"{regime_returns.mean() / regime_returns.std():.2f}" if regime_returns.std() > 0 else "N/A",
        })

    # Transition matrix
    transitions = pd.crosstab(labels.shift(1).dropna().astype(int), labels.iloc[1:].values, normalize="index")
    transitions.index = [names.get(i, f"R{i}") for i in transitions.index]
    transitions.columns = [names.get(i, f"R{i}") for i in transitions.columns]

    return RegimeResult(
        regime_labels=labels,
        regime_names=names,
        regime_colors=colors,
        per_regime_stats=pd.DataFrame(stats_data),
        transition_matrix=transitions,
    )


def _detect_adx(df: pd.DataFrame, lookback: int = 14) -> RegimeResult:
    """ADX-based regime detection: trending vs ranging vs volatile."""
    # Calculate ADX
    high, low, close = df["high"], df["low"], df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(lookback).mean()
    plus_di = 100 * (plus_dm.rolling(lookback).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(lookback).mean() / atr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    adx = dx.rolling(lookback).mean()

    # ATR percentile for volatility
    atr_pct = atr.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    # Classify: ADX > 25 = trending, ADX < 20 = ranging, high ATR = volatile
    labels = pd.Series(1, index=df.index)  # Default: Normal
    labels[adx > 25] = 2  # Trending
    labels[adx < 20] = 0  # Ranging
    labels = labels.fillna(1).astype(int)

    names = {0: "Ranging", 1: "Normal", 2: "Trending"}
    colors = {0: "orange", 1: "blue", 2: "green"}

    stats_data = []
    for r in range(3):
        mask = labels == r
        regime_returns = df["close"].pct_change().loc[mask].dropna()
        stats_data.append({
            "Regime": names[r],
            "Bars": int(mask.sum()),
            "% of Time": f"{mask.mean() * 100:.1f}%",
            "Mean Return": f"{regime_returns.mean() * 100:.4f}%",
            "Volatility": f"{regime_returns.std() * 100:.4f}%",
        })

    transitions = pd.crosstab(labels.shift(1).dropna().astype(int), labels.iloc[1:].values, normalize="index")
    transitions.index = [names.get(i, f"R{i}") for i in transitions.index]
    transitions.columns = [names.get(i, f"R{i}") for i in transitions.columns]

    return RegimeResult(
        regime_labels=labels,
        regime_names=names,
        regime_colors=colors,
        per_regime_stats=pd.DataFrame(stats_data),
        transition_matrix=transitions,
    )


def backtest_by_regime(
    strategy,
    df: pd.DataFrame,
    params: dict,
    regime_result: RegimeResult,
    selected_regimes: list[int] | None = None,
    init_cash: float = 10_000.0,
    fees: float = 0.0,
    freq: str | None = None,
) -> RegimeBacktestResult:
    """Run backtest filtered by regime.

    Args:
        selected_regimes: Only take trades when in these regimes.
            None = run in all regimes (for per-regime breakdown).
    """
    entries, exits = strategy.generate_signals(df, **params)

    # Full (unfiltered) backtest
    pf_kwargs = dict(close=df["close"], entries=entries, exits=exits, init_cash=init_cash, fees=fees)
    if freq:
        pf_kwargs["freq"] = freq
    full_pf = vbt.Portfolio.from_signals(**pf_kwargs)
    full_stats = full_pf.stats()
    full_metrics = {
        "total_return": _safe(full_stats, "Total Return [%]"),
        "sharpe_ratio": _safe(full_stats, "Sharpe Ratio"),
        "win_rate": _safe(full_stats, "Win Rate [%]"),
        "profit_factor": _safe(full_stats, "Profit Factor"),
        "max_drawdown": _safe(full_stats, "Max Drawdown [%]"),
        "total_trades": int(_safe(full_stats, "Total Trades", 0)),
    }

    # Per-regime backtest
    regime_metrics = []
    regimes_to_test = selected_regimes if selected_regimes else list(regime_result.regime_names.keys())

    for r in regimes_to_test:
        mask = regime_result.regime_labels == r
        filtered_entries = entries & mask
        filtered_exits = exits  # Always allow exits

        pf_kwargs_r = dict(close=df["close"], entries=filtered_entries, exits=filtered_exits, init_cash=init_cash, fees=fees)
        if freq:
            pf_kwargs_r["freq"] = freq

        rpf = vbt.Portfolio.from_signals(**pf_kwargs_r)
        rstats = rpf.stats()

        regime_metrics.append({
            "Regime": regime_result.regime_names.get(r, f"R{r}"),
            "Return %": _safe(rstats, "Total Return [%]"),
            "Sharpe": _safe(rstats, "Sharpe Ratio"),
            "Win Rate %": _safe(rstats, "Win Rate [%]"),
            "Profit Factor": _safe(rstats, "Profit Factor"),
            "Max DD %": _safe(rstats, "Max Drawdown [%]"),
            "Trades": int(_safe(rstats, "Total Trades", 0)),
        })

    return RegimeBacktestResult(
        regime_result=regime_result,
        per_regime_metrics=pd.DataFrame(regime_metrics),
        full_metrics=full_metrics,
        full_portfolio=full_pf,
    )


def _safe(stats, key, default=0.0):
    try:
        val = stats[key]
        if pd.isna(val):
            return default
        return float(val)
    except (KeyError, IndexError):
        return default
