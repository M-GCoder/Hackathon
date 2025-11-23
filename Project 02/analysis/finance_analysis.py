import pandas as pd

def add_moving_averages(df: pd.DataFrame, windows=(10, 20)):
    """
    Adds moving average columns for 'close' price. Returns a copy.
    """
    df = df.copy()
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    for w in windows:
        df[f"ma_{w}"] = df["close"].rolling(window=w, min_periods=1).mean()
    return df

def compute_basic_stats(df: pd.DataFrame):
    stats = {}
    stats["latest_close"] = float(df["close"].iloc[-1])
    stats["min_close"] = float(df["close"].min())
    stats["max_close"] = float(df["close"].max())
    stats["mean_close"] = float(df["close"].mean())
    stats["volume_sum"] = int(df["volume"].sum()) if "volume" in df.columns else None
    return stats

def add_returns(df: pd.DataFrame):
    df = df.copy()
    df["return"] = df["close"].pct_change().fillna(0)
    return df

def add_rsi(df: pd.DataFrame, window=14):
    """
    Calculates the Relative Strength Index (RSI) for the close price.
    """
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def analyze_candlesticks(df: pd.DataFrame):
    """
    Analyzes the most recent candle for simple patterns.
    """
    if df.empty or len(df) < 1:
        return "Not enough data for analysis."

    summary = []
    latest_candle = df.iloc[-1]
    
    # Check for Hammer pattern
    # Long lower wick (shadow) and a small body
    body_size = abs(latest_candle['close'] - latest_candle['open'])
    low_wick_size = latest_candle['open'] - latest_candle['low'] if latest_candle['close'] > latest_candle['open'] else latest_candle['close'] - latest_candle['low']
    
    # A hammer is where the lower shadow is at least twice the size of the body.
    if low_wick_size > 2 * body_size:
        summary.append("Latest candle shows a **Hammer pattern**, which can indicate a potential bullish reversal.")
    
    # Check for Doji pattern (open and close are very close)
    if abs(latest_candle['open'] - latest_candle['close']) < 0.001 * latest_candle['close']:
        summary.append("Latest candle is a **Doji**, indicating market indecision.")
    
    return " ".join(summary) if summary else "No significant candlestick patterns detected."

def generate_signals(df: pd.DataFrame):
    """
    Generates trading signals based on MA crossover and RSI.
    """
    signals = {}
    if df.empty or len(df) < max(df.columns.str.extract(r'ma_(\d+)').dropna()[0].astype(int).max(), 14) + 1:
        signals['warning'] = "Not enough data for signal generation."
        return signals
    
    # Moving Average Crossover signal (using ma_10 and ma_20)
    if df['ma_10'].iloc[-1] > df['ma_20'].iloc[-1] and df['ma_10'].iloc[-2] <= df['ma_20'].iloc[-2]:
        signals['ma_crossover'] = "A **Golden Cross** has occurred (10-period MA crossed above 20-period MA), indicating potential upward momentum."
    elif df['ma_10'].iloc[-1] < df['ma_20'].iloc[-1] and df['ma_10'].iloc[-2] >= df['ma_20'].iloc[-2]:
        signals['ma_crossover'] = "A **Death Cross** has occurred (10-period MA crossed below 20-period MA), indicating potential downward momentum."
    
    # RSI signal
    if 'rsi' in df.columns:
        last_rsi = df['rsi'].iloc[-1]
        if last_rsi > 70:
            signals['rsi'] = "The stock may be **overbought** (RSI > 70)."
        elif last_rsi < 30:
            signals['rsi'] = "The stock may be **oversold** (RSI < 30)."

    return signals