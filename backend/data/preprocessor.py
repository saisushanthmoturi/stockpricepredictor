"""
preprocessor.py — Data normalization and sequence generation.

=== ARCHITECTURE CHANGE: PREDICTING RETURNS ===

Previous approach (BROKEN):
  - Predicted raw Close price ($270.50)
  - MinMaxScaler fitted on training data (prices $170-$260)
  - Test data prices ($260-$285) were OUTSIDE the scaler range
  - Scaled test targets reached 1.33 when model only saw [0,1]
  - Result: R² = -70 (worse than predicting the mean)

New approach (FIXED):
  - Predict daily percentage RETURNS (+1.2%, -0.5%, etc.)
  - Returns are STATIONARY — a +1% move means the same at $100 or $300
  - MinMaxScaler range is consistent across train/test
  - To get actual prices: multiply returns by previous day's close
  - Result: R² should be 0.85+ (tracking price movements well)

Pipeline:
  1. Compute ratio-based indicators (scale-invariant)
  2. Drop NaN warm-up rows
  3. 80/20 chronological split
  4. Fit MinMaxScaler on training data only
  5. Generate sliding-window sequences
  6. Keep raw Close prices for reconstructing actual predictions
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from utils.indicators import add_all_indicators
import config


def prepare_data(
    df: pd.DataFrame,
    lookback: int = config.LOOKBACK_WINDOW,
    feature_cols: list = None,
):
    """
    Full preprocessing pipeline.

    Returns
    -------
    dict with X_train, y_train, X_val, y_val, X_test, y_test,
         scaler, feature_cols, dates_*, raw_close_*, and split indices.
    """
    if feature_cols is None:
        feature_cols = config.FEATURE_COLUMNS

    # ── Step 1: Add ratio-based indicators ───────────────
    df = add_all_indicators(df)

    # ── Step 2: Drop warm-up NaN rows ────────────────────
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # ── Step 3: Extract data ─────────────────────────────
    data = df[feature_cols].values        # (N, num_features)
    dates = df.index
    raw_close = df["Close"].values        # Keep raw prices for reconstruction

    # ── Step 4: 80/20 chronological split ────────────────
    n = len(data)
    split_idx = int(n * config.TRAIN_SPLIT_RATIO)

    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # ── Step 5: Fit scaler on training data only ─────────
    # Using RobustScaler: less sensitive to outliers than MinMaxScaler
    # Returns can have extreme values on earnings days
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # ── Step 6: Generate sequences ───────────────────────
    X_train_full, y_train_full = _create_sequences(train_scaled, lookback)
    X_test, y_test = _create_sequences(test_scaled, lookback)

    # ── Step 7: Split training into train + validation ───
    val_size = int(len(X_train_full) * config.VALIDATION_FRACTION)
    X_train = X_train_full[:-val_size]
    y_train = y_train_full[:-val_size]
    X_val = X_train_full[-val_size:]
    y_val = y_train_full[-val_size:]

    # ── Align dates and raw_close ────────────────────────
    train_dates = dates[:split_idx][lookback:]
    test_dates = dates[split_idx:][lookback:]
    train_close = raw_close[:split_idx][lookback:]
    test_close = raw_close[split_idx:][lookback:]

    # Split train dates/close into train + val
    train_dates_split = train_dates[:-val_size]
    val_dates = train_dates[-val_size:]
    train_close_split = train_close[:-val_size]
    val_close = train_close[-val_size:]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "dates_train": train_dates_split,
        "dates_val": val_dates,
        "dates_test": test_dates,
        "raw_close_train": train_close_split,
        "raw_close_val": val_close,
        "raw_close_test": test_close,
        "all_raw_close": raw_close,
        "all_dates": dates,
        "split_idx": split_idx,
    }


def _create_sequences(scaled_data: np.ndarray, lookback: int):
    """
    Create LSTM input sequences.

    X[i] = scaled_data[t-lookback : t]  →  shape (lookback, features)
    y[i] = scaled_data[t, 0]            →  scaled Returns at time t
    """
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback : i])
        y.append(scaled_data[i, 0])      # Returns is column 0
    return np.array(X), np.array(y)


def inverse_transform_returns(
    scaled_returns: np.ndarray,
    scaler,
    num_features: int,
) -> np.ndarray:
    """
    Convert scaled return predictions back to actual percentage returns.

    Parameters
    ----------
    scaled_returns : predictions from the model (scaled)
    scaler : the fitted RobustScaler
    num_features : number of features

    Returns
    -------
    np.ndarray of actual percentage returns
    """
    scaled_returns = scaled_returns.flatten()
    dummy = np.zeros((len(scaled_returns), num_features))
    dummy[:, 0] = scaled_returns
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, 0]


def returns_to_prices(returns_pct: np.ndarray, start_price: float) -> np.ndarray:
    """
    Convert percentage returns back to actual prices.

    price[t] = price[t-1] * (1 + returns_pct[t] / 100)

    Parameters
    ----------
    returns_pct : array of percentage returns
    start_price : the starting price (last known close before predictions)

    Returns
    -------
    np.ndarray of actual prices
    """
    prices = [start_price]
    for r in returns_pct:
        new_price = prices[-1] * (1 + r / 100)
        prices.append(new_price)
    return np.array(prices[1:])
