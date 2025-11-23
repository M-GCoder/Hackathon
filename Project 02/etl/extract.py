# etl/extract.py
import os
import json
from pathlib import Path
from typing import Dict, Any
from api_client.alpha_vantage_client import AlphaVantageClient

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def cache_path(symbol: str, interval: str) -> Path:
    sanitized = f"{symbol.upper()}_{interval}"
    return CACHE_DIR / f"{sanitized}_raw.json"

def fetch_and_cache_intraday(symbol: str, interval: str = "5min", use_cache=True, cache_ttl_seconds=300) -> Dict[str, Any]:
    """
    Fetch intraday JSON from Alpha Vantage and cache to disk.
    cache_ttl_seconds: how long cache is considered fresh (default 5 minutes)
    """
    p = cache_path(symbol, interval)
    if use_cache and p.exists():
        age = (p.stat().st_mtime)
        import time
        if (time.time() - age) < cache_ttl_seconds:
            with p.open("r") as f:
                return json.load(f)

    client = AlphaVantageClient()
    data = client.get_intraday(symbol=symbol, interval=interval)
    with p.open("w") as f:
        json.dump(data, f)
    return data