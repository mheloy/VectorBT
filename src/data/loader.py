"""Data loading pipeline: CSV concatenation, resampling, and parquet caching."""

import glob
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "/home/mheloy/forex-data/data")
CACHE_DIR = Path(__file__).resolve().parents[2] / "results"


def load_m5(data_dir: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Load all XAUUSD 5-minute CSVs into a single DataFrame.

    Globs all CSV files, concatenates, deduplicates, and caches as parquet.
    """
    data_dir = data_dir or DATA_DIR
    cache_path = CACHE_DIR / "xauusd_m5_combined.parquet"

    if use_cache and cache_path.exists():
        # Auto-invalidate cache if any CSV is newer or CSV count changed
        csv_files = sorted(glob.glob(os.path.join(data_dir, "XAUUSD_M5_*.csv")))
        cache_mtime = cache_path.stat().st_mtime
        csvs_newer = any(os.path.getmtime(f) > cache_mtime for f in csv_files)
        if not csvs_newer:
            return pd.read_parquet(cache_path)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "XAUUSD_M5_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No XAUUSD_M5_*.csv files found in {data_dir}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, parse_dates=["datetime"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    combined = combined.set_index("datetime")
    combined = combined[~combined.index.duplicated(keep="first")]

    # Verify monotonic
    assert combined.index.is_monotonic_increasing, "Index is not monotonic after dedup"

    # Cache as parquet
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)

    return combined


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe.

    Args:
        df: DataFrame with OHLCV columns and datetime index.
        timeframe: One of '5M', '15M', '30M', '1H', '4H', 'D'.
    """
    tf_map = {
        "5M": "5min",
        "15M": "15min",
        "30M": "30min",
        "1H": "1h",
        "4H": "4h",
        "D": "1D",
    }
    rule = tf_map.get(timeframe.upper())
    if rule is None:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Use one of: {list(tf_map.keys())}")

    if timeframe.upper() == "5M":
        return df  # Already 5M, no resampling needed

    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    return resampled


TIMEFRAMES = ["5M", "15M", "30M", "1H", "4H", "D"]
