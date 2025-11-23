# api_client/alpha_vantage_client.py
import os
import time
import requests
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

BASE_URL = "https://www.alphavantage.co/query"

class AlphaVantageClient:
    def __init__(self, api_key: str = None, max_retries: int = 2, retry_delay: float = 1.0):
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not set. Set ALPHA_VANTAGE_API_KEY env var.")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _call_api(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["apikey"] = self.api_key
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                # Handle rate-limit message or errors
                if "Note" in data:
                    # Rate limit hit — wait then retry
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        raise RuntimeError(f"Alpha Vantage rate limit or API note: {data.get('Note')}")
                if "Error Message" in data:
                    raise ValueError(f"Alpha Vantage error: {data.get('Error Message')}")
                return data
            except requests.RequestException as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"Network/API request failed: {e}")
        raise RuntimeError("Failed to call Alpha Vantage API after retries.")

    def get_intraday(self, symbol: str, interval: str = "5min", outputsize: str = "compact") -> Dict[str, Any]:
        """
        Fetch intraday time series.
        interval: '1min','5min','15min','30min','60min'
        outputsize: 'compact' (last 100) or 'full'
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "datatype": "json",
        }
        return self._call_api(params)