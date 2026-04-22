"""
indicators.py — Technical indicator calculations (ratio-based).

IMPORTANT DESIGN DECISION:
  All indicators are computed as RATIOS or NORMALIZED values, not
  absolute prices.  This makes them scale-invariant — a stock at
  $100 and a stock at $1000 will have similar indicator values.

  Why? Raw indicators like SMA_20 = $250 are meaningless without
  knowing the stock's price level.  SMA_20_Ratio = 0.98 means
  "price is 2% below its 20-day average" — universal meaning.

Indicators:
  • Returns       — daily percentage change (the prediction target)
  • Volume_Norm   — volume / 20-day rolling mean volume
  • RSI           — already 0-100, scale-invariant
  • SMA Ratios    — Close / SMA (captures trend relative to averages)
  • EMA Ratios    — Close / EMA
  • MACD_Norm     — MACD / Close (percentage-based)
  • BB_Width      — Bollinger Band width as % of price
  • BB_Position   — where price sits within the bands (0=lower, 1=upper)
"""

import pandas as pd
import numpy as np


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every technical indicator and append to the DataFrame.
    Returns DataFrame with all ratio-based features.
    """
    df = df.copy()

    # ── Returns (prediction target) ─────────────────────
    df["Returns"] = df["Close"].pct_change() * 100   # percentage change

    # ── Volume normalized ───────────────────────────────
    vol_mean = df["Volume"].rolling(window=20).mean()
    df["Volume_Norm"] = df["Volume"] / vol_mean

    # ── RSI (already 0-100, scale invariant) ────────────
    df = _add_rsi(df)

    # ── Moving Averages as RATIOS ───────────────────────
    sma_20 = df["Close"].rolling(window=20).mean()
    sma_50 = df["Close"].rolling(window=50).mean()
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["SMA_20_Ratio"] = df["Close"] / sma_20
    df["SMA_50_Ratio"] = df["Close"] / sma_50
    df["EMA_12_Ratio"] = df["Close"] / ema_12
    df["EMA_26_Ratio"] = df["Close"] / ema_26

    # ── MACD normalized ─────────────────────────────────
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Norm"] = (macd / df["Close"]) * 100         # as percentage
    df["MACD_Signal_Norm"] = (macd_signal / df["Close"]) * 100

    # ── Bollinger Bands (width and position) ────────────
    bb_std = df["Close"].rolling(window=20).std()
    bb_upper = sma_20 + (bb_std * 2)
    bb_lower = sma_20 - (bb_std * 2)

    bb_range = bb_upper - bb_lower
    df["BB_Width"] = (bb_range / df["Close"]) * 100       # width as % of price
    df["BB_Position"] = (df["Close"] - bb_lower) / bb_range  # 0-1 position

    return df


def _add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    RSI: Relative Strength Index (0-100 scale).
    Already scale-invariant — no normalization needed.
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df
