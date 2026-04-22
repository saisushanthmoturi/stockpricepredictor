"""
Configuration constants for the Stock Price Prediction Model.

This file centralizes all tunable parameters so you can adjust
the model's behavior from a single location.

=== KEY DESIGN DECISIONS ===

1. PREDICT RETURNS, NOT RAW PRICE
   Stock prices are non-stationary (they trend up or down over time).
   A MinMaxScaler fitted on 2020 prices becomes useless by 2025 because
   the price has moved completely out of range.
   
   Instead, we predict the *percentage change* from the previous day.
   Returns are stationary: a +1% move today has the same meaning whether
   the stock is at $100 or $300.  This eliminates the scaler mismatch
   problem that was causing R² = -70.

2. USE 80/20 SPLIT, VALIDATE ON SUBSET OF TRAIN
   With only 2y of data (~500 days), a 3-way split is wasteful.
   We use 80/20 train/test and carve out 20% of the training set
   for validation (early stopping).  The test set is fully held out.

3. SCALE EACH FEATURE INDEPENDENTLY
   Each feature (Returns, Volume, RSI, etc.) has a different scale.
   MinMaxScaler handles this, but we must fit it on training data only.
"""

import os

# ─────────────────────────────────────────────
#  Data Pipeline Settings
# ─────────────────────────────────────────────

# Number of past trading days used as input for each prediction
LOOKBACK_WINDOW = 60

# Chronological train/test split (80% train, 20% test)
TRAIN_SPLIT_RATIO = 0.80

# Fraction of training data used for validation (early stopping)
VALIDATION_FRACTION = 0.15

# Default historical data period
DEFAULT_PERIOD = "5y"      # 5 years gives ~1250 data points — much more training data

# Features fed into the LSTM
FEATURE_COLUMNS = [
    "Returns",           # Target: percentage change in Close
    "Volume_Norm",       # Volume normalized by its rolling mean
    "RSI",
    "SMA_20_Ratio",      # Price / SMA_20 (ratio, not absolute)
    "SMA_50_Ratio",      # Price / SMA_50
    "EMA_12_Ratio",      # Price / EMA_12
    "EMA_26_Ratio",      # Price / EMA_26
    "MACD_Norm",         # MACD / Close (normalized)
    "MACD_Signal_Norm",  # MACD_Signal / Close
    "BB_Width",          # (BB_Upper - BB_Lower) / Close (%)
    "BB_Position",       # (Close - BB_Lower) / (BB_Upper - BB_Lower)
]

# ─────────────────────────────────────────────
#  LSTM Model Hyper-parameters
# ─────────────────────────────────────────────

LSTM_UNITS_LAYER1 = 128
LSTM_UNITS_LAYER2 = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.2

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 150

# EarlyStopping patience
EARLY_STOP_PATIENCE = 15

# ReduceLROnPlateau
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 7

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# ─────────────────────────────────────────────
#  API Settings
# ─────────────────────────────────────────────

API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 5001))
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
