"""
predictor.py — End-to-end prediction pipeline.

=== RETURNS-BASED PREDICTION ===

The model predicts daily percentage RETURNS, not raw prices.
This is the standard approach in quantitative finance because:

  1. Returns are stationary  → consistent distribution across time
  2. No scaler mismatch     → train and test have similar ranges
  3. Better generalization   → model learns price MOVEMENTS, not levels

To show prices on the dashboard, we reconstruct them:
  predicted_price[t] = actual_price[t-1] × (1 + predicted_return[t] / 100)

For the TEST set evaluation:
  - We predict returns for each test day
  - Convert predicted returns to prices using the day-before-actual close
  - Compare predicted prices vs actual prices → RMSE, MAE, MAPE, R²
  
  This means each test prediction starts from the TRUE previous close,
  not from a previous prediction. This gives honest per-day accuracy
  (called "one-step-ahead" evaluation).
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

import config
from data.fetcher import fetch_stock_data
from data.preprocessor import (
    prepare_data,
    inverse_transform_returns,
    returns_to_prices,
)
from models.lstm_model import build_lstm_model, AttentionLayer


class StockPredictor:
    """End-to-end predictor: fetch → preprocess → train → predict → evaluate."""

    def __init__(self, ticker: str, period: str = "5y", future_days: int = 30):
        self.ticker = ticker.upper()
        self.period = period
        self.future_days = future_days

        self.model = None
        self.history = None
        self.data = None
        self.raw_df = None
        self.epochs_trained = 0

    def run(self) -> dict:
        """Execute full pipeline and return JSON-serializable results."""
        # Step 1: Fetch raw data
        self.raw_df = fetch_stock_data(self.ticker, period=self.period)

        # Step 2: Preprocess
        self.data = prepare_data(self.raw_df)

        # Step 3: Train
        self._train_model()

        # Step 4: Predict returns
        train_returns = self._predict_returns(self.data["X_train"])
        val_returns = self._predict_returns(self.data["X_val"])
        test_returns = self._predict_returns(self.data["X_test"])

        # Step 5: Convert returns → prices (one-step-ahead)
        train_prices = self._returns_to_one_step_prices(
            train_returns, self.data["raw_close_train"]
        )
        val_prices = self._returns_to_one_step_prices(
            val_returns, self.data["raw_close_val"]
        )
        test_prices = self._returns_to_one_step_prices(
            test_returns, self.data["raw_close_test"]
        )

        # Step 6: Future predictions (auto-regressive)
        future_preds, future_dates = self._predict_future()

        # Step 7: Metrics on test set
        metrics = self._evaluate(test_prices, self.data["raw_close_test"])

        # Step 8: Build response
        return self._build_response(
            train_prices, val_prices, test_prices,
            future_preds, future_dates, metrics,
        )

    # ────────────────────────────────────────────────────────
    #  Training
    # ────────────────────────────────────────────────────────

    def _train_model(self):
        """Train with validation-based early stopping."""
        X_train = self.data["X_train"]
        y_train = self.data["y_train"]
        X_val = self.data["X_val"]
        y_val = self.data["y_val"]

        num_features = X_train.shape[2]
        lookback = X_train.shape[1]

        self.model = build_lstm_model(lookback, num_features)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        )

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=[early_stop, reduce_lr],
            shuffle=True,
            verbose=1,
        )

        self.epochs_trained = len(self.history.history["loss"])

    # ────────────────────────────────────────────────────────
    #  Prediction
    # ────────────────────────────────────────────────────────

    def _predict_returns(self, X: np.ndarray) -> np.ndarray:
        """Predict scaled returns → inverse transform to actual % returns."""
        scaled_preds = self.model.predict(X, verbose=0)
        num_features = len(self.data["feature_cols"])
        return inverse_transform_returns(scaled_preds, self.data["scaler"], num_features)

    def _returns_to_one_step_prices(
        self, predicted_returns: np.ndarray, actual_close: np.ndarray
    ) -> np.ndarray:
        """
        One-step-ahead price reconstruction.

        For each day t:
          predicted_price[t] = actual_close[t-1] × (1 + predicted_return[t] / 100)

        We use the ACTUAL previous close (not the predicted one) so each
        prediction is independent. This gives honest daily accuracy metrics.
        """
        # The predicted return at position i corresponds to the return from
        # close[i-1] to close[i]. We use close[i-1] shifted by 1.
        # Since close and predictions are aligned (same length),
        # we shift: previous_close = actual_close shifted right by 1.
        # For position 0, use the close from just before the sequence.

        predicted_prices = np.zeros(len(predicted_returns))
        for i in range(len(predicted_returns)):
            if i == 0:
                # Use first actual close as the "previous"
                prev_close = actual_close[0]
            else:
                prev_close = actual_close[i - 1]
            predicted_prices[i] = prev_close * (1 + predicted_returns[i] / 100)

        return predicted_prices

    def _predict_future(self):
        """
        Auto-regressive future price prediction.

        Uses the model's own predictions to generate the next day's
        input features, rolling forward day by day.
        """
        scaler = self.data["scaler"]
        feature_cols = self.data["feature_cols"]
        num_features = len(feature_cols)
        lookback = config.LOOKBACK_WINDOW

        # Get full scaled dataset for the last window
        from utils.indicators import add_all_indicators

        df = add_all_indicators(self.raw_df.copy())
        df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        full_data = df[feature_cols].values
        full_scaled = scaler.transform(full_data)

        last_known_price = df["Close"].values[-1]
        current_window = full_scaled[-lookback:].copy()

        predicted_prices = []
        current_price = last_known_price

        for _ in range(self.future_days):
            input_seq = current_window.reshape(1, lookback, num_features)
            pred_scaled = self.model.predict(input_seq, verbose=0)[0, 0]

            # Inverse transform to get return percentage
            dummy = np.zeros((1, num_features))
            dummy[0, 0] = pred_scaled
            pred_return = scaler.inverse_transform(dummy)[0, 0]

            # Convert return to price
            new_price = current_price * (1 + pred_return / 100)
            predicted_prices.append(new_price)
            current_price = new_price

            # Update window with predicted values
            new_row = current_window[-1].copy()
            new_row[0] = pred_scaled  # Update Returns column
            current_window = np.vstack([current_window[1:], new_row])

        # Generate business dates
        last_date = df.index[-1]
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.future_days,
        )

        return np.array(predicted_prices), future_dates

    # ────────────────────────────────────────────────────────
    #  Evaluation
    # ────────────────────────────────────────────────────────

    def _evaluate(self, predicted_prices: np.ndarray, actual_close: np.ndarray) -> dict:
        """
        Compute metrics comparing predicted vs actual prices.

        These metrics are on the TEST set which the model never saw.
        """
        # Align lengths
        min_len = min(len(predicted_prices), len(actual_close))
        pred = predicted_prices[:min_len]
        actual = actual_close[:min_len]

        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mae = float(mean_absolute_error(actual, pred))
        r2 = float(r2_score(actual, pred))

        # MAPE
        non_zero = actual != 0
        mape = float(
            np.mean(np.abs((actual[non_zero] - pred[non_zero]) / actual[non_zero])) * 100
        )

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 4),
            "r2_score": round(r2, 4),
        }

    # ────────────────────────────────────────────────────────
    #  Response Builder
    # ────────────────────────────────────────────────────────

    def _build_response(
        self, train_prices, val_prices, test_prices,
        future_preds, future_dates, metrics,
    ) -> dict:
        """Assemble JSON-serializable response."""

        def _series(dates, prices):
            return [
                {
                    "date": str(d.date()) if hasattr(d, "date") else str(d)[:10],
                    "price": round(float(p), 2),
                }
                for d, p in zip(dates, prices)
            ]

        # Full historical prices
        from utils.indicators import add_all_indicators

        df = add_all_indicators(self.raw_df.copy())
        df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()

        historical = [
            {"date": str(d.date()), "price": round(float(p), 2)}
            for d, p in zip(df.index, df["Close"].values)
        ]

        # Training history
        training_history = {
            "loss": [round(float(v), 6) for v in self.history.history["loss"]],
            "val_loss": [round(float(v), 6) for v in self.history.history["val_loss"]],
        }

        # Combine val + test predictions for display
        combined_dates = list(self.data["dates_val"]) + list(self.data["dates_test"])
        combined_prices = np.concatenate([val_prices, test_prices])

        return {
            "ticker": self.ticker,
            "historical_prices": historical,
            "train_predictions": _series(self.data["dates_train"], train_prices),
            "test_predictions": _series(combined_dates, combined_prices),
            "future_predictions": _series(future_dates, future_preds),
            "metrics": metrics,
            "model_info": {
                "architecture": "Stacked LSTM + Attention + BatchNorm",
                "lstm_layer_1_units": config.LSTM_UNITS_LAYER1,
                "lstm_layer_2_units": config.LSTM_UNITS_LAYER2,
                "attention": True,
                "batch_normalization": True,
                "dropout_rate": config.DROPOUT_RATE,
                "features_used": self.data["feature_cols"],
                "lookback_window": config.LOOKBACK_WINDOW,
                "epochs_trained": self.epochs_trained,
                "batch_size": config.BATCH_SIZE,
                "optimizer": "Adam",
                "loss_function": "Huber",
                "learning_rate": config.LEARNING_RATE,
                "prediction_target": "Daily Returns (%)",
                "train_samples": len(self.data["X_train"]),
                "val_samples": len(self.data["X_val"]),
                "test_samples": len(self.data["X_test"]),
            },
            "training_history": training_history,
        }
