# etl/transform.py
import pandas as pd
from typing import Dict, Any
import re

def parse_intraday_json(json_data: Dict[str, Any], interval: str):
    """
    Parse Alpha Vantage intraday JSON into a pandas DataFrame
    """
    time_series_key = None
    # find the time series key (it includes the interval)
    for k in json_data.keys():
        if re.match(r"Time Series.*", k):
            time_series_key = k
            break
    if time_series_key is None:
        raise ValueError("Unexpected Alpha Vantage response format (no Time Series found).")

    ts = json_data[time_series_key]
    df = pd.DataFrame.from_dict(ts, orient="index")
    # columns come as '1. open', '2. high', etc. rename them
    df.columns = [c.split(". ", 1)[-1] if ". " in c else c for c in df.columns]
    df.index = pd.to_datetime(df.index)
    # convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_index()
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.index.name = "datetime"
    return df