"""
fetcher.py — Yahoo Finance historical data downloader.

Responsibilities:
  1. Download OHLCV (Open, High, Low, Close, Volume) data for any ticker.
  2. Clean the raw DataFrame (forward-fill gaps, drop NaN rows).
  3. Return a ready-to-use pandas DataFrame indexed by date.

Usage:
    from data.fetcher import fetch_stock_data
    df = fetch_stock_data("AAPL", period="2y")

Note:
  This module uses the Yahoo Finance Chart API directly (v8 endpoint)
  to avoid SSL compatibility issues with older Python/LibreSSL versions
  that can affect the yfinance library.
"""

import json
import subprocess
import datetime
import pandas as pd
import numpy as np


# Period → seconds mapping (approximate)
_PERIOD_SECONDS = {
    "1mo": 30 * 86400,
    "3mo": 90 * 86400,
    "6mo": 180 * 86400,
    "1y": 365 * 86400,
    "2y": 730 * 86400,
    "5y": 1826 * 86400,
    "10y": 3652 * 86400,
    "max": int(50 * 365.25 * 86400),
}


def fetch_stock_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL", "GOOGL", "RELIANCE.NS").
    period : str
        How far back to fetch — "1y", "2y", "5y", "max", etc.
    interval : str
        Bar interval — "1d" (daily), "1h" (hourly), etc.

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume.
        Index:   DatetimeIndex (timezone-naive).

    Raises
    ------
    ValueError
        If no data is returned (invalid ticker or no trading history).
    """
    # First try yfinance directly
    try:
        df = _fetch_via_yfinance(ticker, period, interval)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass  # Fall through to direct API method

    # Fallback: fetch directly via Yahoo Chart API using curl
    # (bypasses Python SSL issues)
    try:
        df = _fetch_via_chart_api(ticker, period, interval)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        raise ValueError(
            f"Failed to fetch data for ticker '{ticker}': {str(e)}. "
            "Check if the symbol is valid on Yahoo Finance."
        )

    raise ValueError(
        f"No data returned for ticker '{ticker}'. "
        "Check if the symbol is valid on Yahoo Finance."
    )


def _fetch_via_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Try using yfinance library directly."""
    import yfinance as yf

    raw = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel("Ticker")

    if raw.empty:
        return None

    return _clean_dataframe(raw)


def _fetch_via_chart_api(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch data directly from Yahoo Finance Chart API using curl.
    This bypasses Python SSL/TLS issues by using the system curl binary.
    """
    now = int(datetime.datetime.now().timestamp())
    period_secs = _PERIOD_SECONDS.get(period, 730 * 86400)
    period1 = now - period_secs

    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={now}&interval={interval}"
    )

    # Use system curl to avoid Python SSL issues
    try:
        result = subprocess.run(
            ["curl", "-s", url, "-H", "User-Agent: Mozilla/5.0"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to fetch data from Yahoo Finance: {e}")

    # Parse the chart API response
    chart = data.get("chart", {})
    result_data = chart.get("result", [])

    if not result_data:
        error = chart.get("error", {})
        raise ValueError(
            f"Yahoo Finance API error: {error.get('description', 'Unknown error')}"
        )

    result_entry = result_data[0]
    timestamps = result_entry.get("timestamp", [])
    indicators = result_entry.get("indicators", {})
    quotes = indicators.get("quote", [{}])[0]

    if not timestamps:
        raise ValueError(f"No trading data available for '{ticker}'")

    # Build DataFrame
    df = pd.DataFrame({
        "Open": quotes.get("open", []),
        "High": quotes.get("high", []),
        "Low": quotes.get("low", []),
        "Close": quotes.get("close", []),
        "Volume": quotes.get("volume", []),
    })

    # Set DatetimeIndex
    df.index = pd.to_datetime(timestamps, unit="s")
    df.index.name = "Date"

    # Handle adjusted close if available
    adj_close_data = indicators.get("adjclose", [])
    if adj_close_data and adj_close_data[0].get("adjclose"):
        adj_close = adj_close_data[0]["adjclose"]
        if len(adj_close) == len(df):
            # Calculate adjustment factor and apply to OHLC
            raw_close = df["Close"].values
            adj_factor = np.array(adj_close) / np.where(raw_close == 0, 1, raw_close)

            df["Open"] = df["Open"] * adj_factor
            df["High"] = df["High"] * adj_factor
            df["Low"] = df["Low"] * adj_factor
            df["Close"] = np.array(adj_close)

    return _clean_dataframe(df)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the OHLCV DataFrame.

    Steps:
      1. Keep only needed columns.
      2. Forward-fill → backward-fill → drop remaining NaN.
      3. Remove timezone info for consistency.
    """
    columns_needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df[columns_needed].copy()

    # Handle missing data
    df = df.ffill().bfill().dropna()

    # Make timezone-naive
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df
