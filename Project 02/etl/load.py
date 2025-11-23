# etl/load.py
from pathlib import Path
import pandas as pd

CLEAN_CACHE_DIR = Path("cache/clean")
CLEAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def clean_cache_path(symbol: str, interval: str) -> Path:
    return CLEAN_CACHE_DIR / f"{symbol.upper()}_{interval}_clean.csv"

def save_clean_df(df: pd.DataFrame, symbol: str, interval: str):
    p = clean_cache_path(symbol, interval)
    df.to_csv(p, index=True)

def load_clean_df_if_exists(symbol: str, interval: str):
    p = clean_cache_path(symbol, interval)
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)
    return None
