# 📈 Stock Price Predictor — LSTM + Attention Deep Learning

A production-ready stock price prediction system powered by **Stacked LSTM with Attention mechanism**, built with TensorFlow / Keras. Features a premium glassmorphism frontend dashboard and a Flask REST API backend.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.1-green?logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **Stacked LSTM + Attention** — Two LSTM layers with a custom Bahdanau-style Attention mechanism for capturing the most relevant historical patterns
- **Technical Indicators** — RSI, SMA, EMA, MACD, Bollinger Bands as additional input features
- **Proper Data Pipeline** — Chronological train/test split, train-only scaler fitting (no data leakage)
- **Real-time Training** — Train on any stock ticker via the API (AAPL, GOOGL, TSLA, etc.)
- **Future Forecasting** — Auto-regressive prediction for up to 90 days ahead
- **Premium Dashboard** — Dark glassmorphism UI with Chart.js visualizations, animated metrics, and interactive architecture diagram
- **Production Ready** — Flask API with CORS, error handling, and Render deployment support

---

## 🏗️ Architecture

```
Input (60 days × 11 features)
        │
        ▼
  LSTM Layer 1 (128 units)  →  Dropout (20%)
        │
        ▼
  LSTM Layer 2 (64 units)   →  Dropout (20%)
        │
        ▼
  Attention Layer (Bahdanau) — learns which timesteps matter
        │
        ▼
  Dense (32 units, ReLU)    →  Dropout (20%)
        │
        ▼
  Dense (1 unit) — Predicted Closing Price
```

### Features Used (11 total)
| Feature | Type | Description |
|---------|------|-------------|
| Close | Price | Closing price |
| Volume | Volume | Trading volume |
| RSI | Momentum | Relative Strength Index (14-day) |
| SMA_20 | Trend | 20-day Simple Moving Average |
| SMA_50 | Trend | 50-day Simple Moving Average |
| EMA_12 | Trend | 12-day Exponential Moving Average |
| EMA_26 | Trend | 26-day Exponential Moving Average |
| MACD | Momentum | Moving Average Convergence Divergence |
| MACD_Signal | Momentum | 9-day EMA of MACD |
| BB_Upper | Volatility | Upper Bollinger Band |
| BB_Lower | Volatility | Lower Bollinger Band |

---

## 📂 Project Structure

```
recommendationmodel/
├── backend/
│   ├── app.py                 # Flask API entry point
│   ├── config.py              # All tunable parameters
│   ├── requirements.txt       # Python dependencies
│   ├── Procfile               # Render deployment config
│   ├── data/
│   │   ├── fetcher.py         # Yahoo Finance data downloader
│   │   └── preprocessor.py    # Normalization & sequence generation
│   ├── models/
│   │   ├── lstm_model.py      # LSTM + Attention architecture
│   │   └── predictor.py       # End-to-end prediction pipeline
│   ├── utils/
│   │   └── indicators.py      # Technical indicator calculations
│   └── saved_models/          # Cached trained models
├── frontend/
│   ├── index.html             # Dashboard UI
│   ├── css/styles.css         # Dark glassmorphism theme
│   └── js/
│       ├── api.js             # API communication layer
│       ├── charts.js          # Chart.js visualizations
│       └── app.js             # Application logic
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/recommendationmodel.git
cd recommendationmodel
```

### 2. Set Up the Backend
```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py
```

The API will start at `http://localhost:5000`.

### 3. Open the Frontend
Open `frontend/index.html` in your browser (or use a simple HTTP server):
```bash
cd frontend
python -m http.server 8080
```

Then visit `http://localhost:8080` in your browser.

### 4. Make a Prediction
1. Enter a stock ticker (e.g., `AAPL`)
2. Select the historical data period
3. Set the number of forecast days
4. Click **Predict** and wait ~30-60 seconds for the model to train

---

## 🌐 API Reference

### Health Check
```http
GET /api/health
```
Returns: `{ "status": "ok", "model": "LSTM + Attention Stock Predictor", "version": "1.0.0" }`

### Predict
```http
POST /api/predict
Content-Type: application/json

{
  "ticker": "AAPL",
  "period": "2y",
  "future_days": 30
}
```

Returns predictions, metrics (RMSE, MAE, MAPE, R²), training history, and model configuration.

---

## 🚢 Deployment

### Backend → Render
1. Push your code to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your GitHub repo
4. Set **Build Command**: `pip install -r requirements.txt`
5. Set **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
6. Set the root directory to `backend/`

### Frontend → GitHub Pages
1. Go to your repo **Settings → Pages**
2. Set the source to the `main` branch and `/frontend` directory
3. Update `API_BASE_URL` in `frontend/js/api.js` with your Render URL

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock market predictions are inherently uncertain. This model should not be used as financial advice or for actual trading decisions. Past performance does not guarantee future results.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
