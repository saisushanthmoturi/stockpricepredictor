"""
app.py — Flask REST API for the Stock Price Prediction Model.

Endpoints:
  GET  /api/health       → Health check (returns {"status": "ok"})
  POST /api/predict       → Train model + return predictions for a ticker

This is the entry point for both local development and production deployment.

Local:   python app.py
Render:  gunicorn app:app --bind 0.0.0.0:$PORT
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

import config
from models.predictor import StockPredictor

# ────────────────────────────────────────────────────────
#  App Setup
# ────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend


# ────────────────────────────────────────────────────────
#  Routes
# ────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check to confirm the API is running."""
    return jsonify({
        "status": "ok",
        "model": "LSTM + Attention Stock Predictor",
        "version": "1.0.0",
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Train the LSTM model on the requested stock and return predictions.

    Expected JSON body:
    {
        "ticker": "AAPL",           (required)
        "period": "2y",             (optional, default "2y")
        "future_days": 30           (optional, default 30)
    }

    Returns:
    {
        "ticker": "AAPL",
        "historical_prices": [...],
        "train_predictions": [...],
        "test_predictions": [...],
        "future_predictions": [...],
        "metrics": { "rmse": ..., "mae": ..., "mape": ..., "r2_score": ... },
        "model_info": { ... },
        "training_history": { "loss": [...], "val_loss": [...] }
    }
    """
    try:
        # ── Parse request ─────────────────────────────
        data = request.get_json(force=True)

        ticker = data.get("ticker", "").strip()
        if not ticker:
            return jsonify({"error": "Missing required field: 'ticker'"}), 400

        period = data.get("period", config.DEFAULT_PERIOD)
        future_days = int(data.get("future_days", 30))

        # Validate future_days range
        if future_days < 1 or future_days > 90:
            return jsonify({"error": "future_days must be between 1 and 90"}), 400

        # ── Run the prediction pipeline ───────────────
        predictor = StockPredictor(
            ticker=ticker,
            period=period,
            future_days=future_days,
        )
        result = predictor.run()

        return jsonify(result)

    except ValueError as e:
        # Known errors (e.g., invalid ticker)
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        # Unexpected errors — log the full traceback for debugging
        traceback.print_exc()
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "hint": "Check the server logs for a full traceback.",
        }), 500


# ────────────────────────────────────────────────────────
#  Entry Point
# ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🚀 Stock Predictor API starting on http://localhost:{config.API_PORT}")
    print(f"   POST /api/predict  — run predictions")
    print(f"   GET  /api/health   — health check\n")
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG_MODE,
    )
