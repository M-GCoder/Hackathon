# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.graph_objs as go 
from datetime import datetime, timedelta
from etl.extract import fetch_and_cache_intraday, cache_path
from etl.transform import parse_intraday_json
from etl.load import save_clean_df, load_clean_df_if_exists
from analysis.finance_analysis import add_moving_averages, compute_basic_stats, add_returns
import os
from analysis.finance_analysis import add_moving_averages, compute_basic_stats, add_returns, add_rsi, analyze_candlesticks, generate_signals
from groq import Groq
import json

st.set_page_config(page_title="Intraday ETL Dashboard", layout="wide")

st.title("Intraday ETL & Live Dashboard — Alpha Vantage")

def get_ai_summary_and_prediction(df, symbol, stats, signals, summary_text):
    """
    Uses Groq API to generate a summary and prediction based on data.
    """
    if df.empty:
        return "Not enough data for AI analysis."

    # The Groq client automatically uses the GROQ_API_KEY env var if set
    try:
        client = Groq()
    except Exception as e:
        return f"Groq API client failed to initialize. Error: {e}"

    # Prepare a prompt with relevant data for the LLM
    prompt = f"""
    You are an expert financial analyst. Provide a summary of the stock {symbol} based on the provided technical analysis data.
    Include the date range covered, key statistics, detected candlestick patterns, and identified trading signals.

    **Data Context:**
    - Stock Symbol: {symbol}
    - Date Range: {df.index.min().date()} to {df.index.max().date()}
    - Latest Close: ${stats['latest_close']:.2f}
    - Mean Close: ${stats['mean_close']:.2f}
    - Total Volume: {stats['volume_sum']:,}
    - Candlestick Patterns: {summary_text}
    - Signals: {json.dumps(signals)}

    Analyze this data and provide a concise, professional summary.
    Then, offer a simple, non-guaranteed prediction or suggestion: "Buy", "Sell", or "Hold". 
    Conclude with a strong disclaimer that this is not financial advice.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="GPT OSS Safeguard 20B", # A fast and capable model
            temperature=0.5,
            max_tokens=512,
        )
        # === Missing return statement ===
        return chat_completion.choices.message.content
    except Exception as e:
        # === Missing return statement in exception handler ===
        return f"Groq API call failed: {e}"


with st.sidebar:
    st.markdown("## Stock Comparison")
    compare_symbols_input = st.text_input("Compare Symbols (comma separated, e.g. MSFT, GOOG)", value="").upper().strip()
    symbol = st.text_input("Stock Symbol (e.g. AAPL)", value="AAPL").upper().strip()
    interval = st.selectbox("Interval", options=["1min", "5min", "15min", "30min", "60min"], index=1)
    outputsize = st.selectbox("Output size", options=["compact", "full"], index=0)
    cache_ttl = st.number_input("Cache TTL seconds", value=300, step=60)
    fetch_button = st.button("Fetch / Refresh")

def load_data(symbol, interval, outputsize, cache_ttl):
    # Try to load clean cache first
    clean = load_clean_df_if_exists(symbol, interval)
    if clean is not None and not fetch_button:
        st.info(f"Loaded cleaned data for {symbol} from cache.")
        return clean

    st.info(f"Fetching raw data for {symbol} from Alpha Vantage...")
    try:
        raw = fetch_and_cache_intraday(symbol=symbol, interval=interval, use_cache=True, cache_ttl_seconds=cache_ttl)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None

    # Parse and transform
    try:
        df = parse_intraday_json(raw, interval)
    except Exception as e:
        st.error(f"Failed to parse API response: {e}")
        return None

    # standardize column names if needed: ensure open/high/low/close/volume present
    # AV often returns columns ['open', 'high', 'low', 'close', 'volume']
    # Save clean df
    save_clean_df(df, symbol, interval)
    return df

df = load_data(symbol, interval, outputsize, cache_ttl)
if df is None:
    st.stop()

# --- Comparison Logic (Updated to handle multiple stocks) ---
compare_symbols = [s.strip() for s in compare_symbols_input.split(',') if s.strip()]
comparison_dfs = {}
common_start = df.index.min()
common_end = df.index.max()

for comp_sym in compare_symbols:
    comp_df = load_data(comp_sym, interval, outputsize, cache_ttl)
    if comp_df is not None and not comp_df.empty:
        comparison_dfs[comp_sym] = comp_df
        # Adjust common date range to the intersection of all available data
        common_start = max(common_start, comp_df.index.min())
        common_end = min(common_end, comp_df.index.max())
    else:
        st.warning(f"Could not load data for comparison symbol {comp_sym}.")

# Update primary df date range based on the new common range
df = df.loc[(df.index >= common_start) & (df.index <= common_end)]

# allow optional date range filtering
min_date = df.index.min()
max_date = df.index.max()
with st.expander("Data range / filter"):
    # ... (st.date_input logic remains the same, uses min/max_date) ...
    start = st.date_input("Start date", min_value=min_date, value=max(min_date, max_date - timedelta(days=1)))
    end = st.date_input("End date", min_value=min_date, value=max_date)
    if start > end:
        st.warning("Start must be <= End. Resetting to full range.")
        start, end = min_date, max_date

# Filter primary and comparison dataframes based on user selected range
filtered = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
filtered_compare_dfs = {}
for sym, c_df in comparison_dfs.items():
    filtered_compare_dfs[sym] = c_df.loc[(c_df.index >= pd.to_datetime(start)) & (c_df.index <= pd.to_datetime(end))]

# Add check for empty primary filtered dataframe (remains the same)
if filtered.empty:
    # ... (existing empty check logic) ...
    st.warning("The selected date range contains no data for primary stock. Please adjust the range.")
    st.dataframe(df.tail(20))
    st.stop()

filtered = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

# Add a check to see if the filtered dataframe is empty
if filtered.empty:
    st.warning("The selected date range contains no data. Please adjust the range.")
    st.dataframe(df.tail(20))
    st.stop()

# Ensure standard column names
col_map = {c.lower(): c for c in filtered.columns}
# Prefer standard names if different
for key in ["open","high","low","close","volume"]:
    if key not in filtered.columns:
        # detect keys like '1. open' replaced in transform, so this is probably fine
        pass

# Analysis
filtered = add_returns(filtered)
filtered = add_moving_averages(filtered, windows=(10,20))
filtered = add_rsi(filtered) # Add RSI calculation here
stats = compute_basic_stats(filtered)
candlestick_summary = analyze_candlesticks(filtered)
signals = generate_signals(filtered)

candlestick_summary_text = analyze_candlesticks(filtered) # Rename variable to avoid confusion

signals = generate_signals(filtered)
# === Call the LLM function ===
# Add an expander for the AI Summary in the sidebar for cleaner layout
with st.sidebar:
    st.subheader("AI Summary & Prediction (Groq)")
    # Use a button to trigger the LLM call to avoid calling on every interaction
    if st.button("Generate AI Insight"):
        with st.spinner("Asking AI Analyst (Groq)..."):
            ai_insight = get_ai_summary_and_prediction(filtered, symbol, stats, signals, candlestick_summary_text)
            st.markdown(ai_insight)
    else:
        st.info("Click 'Generate AI Insight' to get an LLM summary.")

# Layout charts
left_col, right_col = st.columns([3,1])

with left_col:
    if filtered_compare_dfs:
        st.subheader(f"Normalized Performance: {symbol} vs {', '.join(filtered_compare_dfs.keys())}")
        comp_fig = go.Figure()
        # Normalize primary stock
        comp_fig.add_trace(go.Scatter(x=filtered.index, y=filtered["close"] / filtered["close"].iloc[0] * 100, name=symbol))
        # Normalize comparison stocks
        for sym, c_df in filtered_compare_dfs.items():
            comp_fig.add_trace(go.Scatter(x=c_df.index, y=c_df["close"] / c_df["close"].iloc[0] * 100, name=sym))
        
        comp_fig.update_layout(yaxis_title="Normalized Price (Start=100)", height=500)
        st.plotly_chart(comp_fig, use_container_width=True)
        st.caption("This chart displays the relative performance of selected stocks, normalized to a starting value of 100 on the first date shown. It helps visualize which stock has performed best regardless of absolute price.")
    
    st.subheader(f"{symbol} Price Chart ({interval})")
    fig = go.Figure()
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=filtered.index,
        open=filtered["open"],
        high=filtered["high"],
        low=filtered["low"],
        close=filtered["close"],
        name="OHLC"
    ))
    st.plotly_chart(fig, use_container_width=True) 
    st.caption("The Candlestick chart shows the Open, High, Low, and Close (OHLC) prices for each time interval, providing insight into market sentiment and price action. Moving averages (MA 10 and MA 20) indicate price trends.")
    # moving averages if present (MUST be added before st.plotly_chart)
    if "ma_10" in filtered.columns:
        fig.add_trace(go.Scatter(x=filtered.index, y=filtered["ma_10"], name="MA 10"))
    if "ma_20" in filtered.columns:
        fig.add_trace(go.Scatter(x=filtered.index, y=filtered["ma_20"], name="MA 20"))
        
    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True) # Chart is rendered here
    st.caption("The Candlestick chart shows the Open, High, Low, and Close (OHLC) prices for each time interval, providing insight into market sentiment and price action. Moving averages (MA 10 and MA 20) indicate price trends.")

    # Add RSI chart below the main price chart
    st.subheader("RSI (Relative Strength Index)")
    if 'rsi' in filtered.columns:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=filtered.index, y=filtered["rsi"], name="RSI"))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.update_layout(height=250)
        st.plotly_chart(rsi_fig, use_container_width=True)
    else:
        st.info("Not enough data to compute RSI.")
        
    # Add Volume Indicator
    st.subheader("Volume")
    vol_fig = go.Figure()
    if "volume" in filtered.columns:
        vol_fig.add_trace(go.Bar(x=filtered.index, y=filtered["volume"], name="Volume"))
        vol_fig.update_layout(height=250)
        st.plotly_chart(vol_fig, use_container_width=True)
        st.caption("Volume indicates the number of shares traded in each period. High volume often confirms price trends or signals potential reversals.")

    else:
        st.info("No volume column in data.")

# Right Column
with right_col:
    st.subheader("Quick Stats")
    st.metric("Latest close", f"{stats['latest_close']:.2f}")
    st.metric("Min close", f"{stats['min_close']:.2f}")
    st.metric("Max close", f"{stats['max_close']:.2f}")
    st.metric("Mean close", f"{stats['mean_close']:.2f}")
    if stats.get("volume_sum") is not None:
        st.metric("Total volume", f"{stats['volume_sum']:,}")
        
    # Add Analysis and Signals here
    st.subheader("Analysis & Signals")
    if signals or candlestick_summary_text: 
        if 'ma_crossover' in signals:
            st.markdown(signals['ma_crossover'])
        if 'rsi' in signals:
            st.markdown(signals['rsi'])
        if candlestick_summary_text:
            st.markdown(candlestick_summary_text)
    else:
        st.info("No significant local signals detected.")

def get_ai_summary_and_prediction(df, symbol, stats, signals, summary_text):
    """
    Uses Groq API to generate a summary and prediction based on data.
    """
    if df.empty:
        return "Not enough data for AI analysis."

    # The Groq client automatically uses the GROQ_API_KEY env var if set
    try:
        client = Groq()
    except Exception as e:
        return f"Groq API client failed to initialize. Error: {e}"

    # Prepare a prompt with relevant data for the LLM
    prompt = f"""
    You are an expert financial analyst. Provide a summary of the stock {symbol} based on the provided technical analysis data.
    Include the date range covered, key statistics, detected candlestick patterns, and identified trading signals.

    **Data Context:**
    - Stock Symbol: {symbol}
    - Date Range: {df.index.min().date()} to {df.index.max().date()}
    - Latest Close: ${stats['latest_close']:.2f}
    - Mean Close: ${stats['mean_close']:.2f}
    - Total Volume: {stats['volume_sum']:,}
    - Candlestick Patterns: {summary_text}
    - Signals: {json.dumps(signals)}

    Analyze this data and provide a concise, professional summary.
    Then, offer a simple, non-guaranteed prediction or suggestion: "Buy", "Sell", or "Hold". 
    Conclude with a strong disclaimer that this is not financial advice.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", # A fast and capable model
            temperature=0.5,
            max_tokens=512,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Groq API call failed: {e}"

st.caption("Data source: Alpha Vantage (Intraday). Cache TTL helps avoid hitting API rate limits.")

st.subheader(f"Recent Data for {symbol} (top 20 rows)")
st.dataframe(filtered.tail(20))
st.caption(f"A preview of the raw data used for the {symbol} analysis.")

if filtered_compare_dfs:
    for sym, c_df in filtered_compare_dfs.items():
        st.subheader(f"Recent Data for {sym} (top 20 rows)")
        st.dataframe(c_df.tail(20))
        st.caption(f"A preview of the raw data used for the {sym} comparison analysis.")

st.caption("Data source: Alpha Vantage (Intraday). Cache TTL helps avoid hitting API rate limits.")

# Add the final disclaimer
st.markdown("---")
st.warning("**Disclaimer:** Stock market predictions are inherently uncertain. The information and indicators presented are for educational and informational purposes only and do not constitute financial advice. Always conduct your own research before making any investment decisions.")