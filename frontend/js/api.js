/**
 * api.js — API Communication Layer
 *
 * Handles all HTTP communication with the Flask backend.
 * Provides a clean interface with error handling and configurable base URL.
 *
 * Configuration:
 *   - For local development: API_BASE_URL = "http://localhost:5000"
 *   - For production:        API_BASE_URL = "https://your-app.onrender.com"
 */

const StockAPI = (() => {
    // ── Configuration ─────────────────────────────────
    // Change this to your deployed backend URL when hosting
    const API_BASE_URL = "http://localhost:5001";

    // ── Public Methods ────────────────────────────────

    /**
     * Send a prediction request to the backend.
     *
     * @param {string} ticker     - Stock ticker symbol (e.g. "AAPL")
     * @param {string} period     - Historical data period ("1y", "2y", "5y")
     * @param {number} futureDays - Number of days to forecast (1-90)
     * @returns {Promise<Object>} - Prediction results from the API
     * @throws {Error}            - On network failure or API error
     */
    async function predict(ticker, period, futureDays) {
        const url = `${API_BASE_URL}/api/predict`;

        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                ticker: ticker,
                period: period,
                future_days: futureDays,
            }),
        });

        const data = await response.json();

        if (!response.ok) {
            // The API returned an error (4xx or 5xx)
            throw new Error(data.error || `Server error (${response.status})`);
        }

        return data;
    }

    /**
     * Check if the backend API is reachable.
     *
     * @returns {Promise<boolean>} - true if the API is up
     */
    async function healthCheck() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/health`);
            return response.ok;
        } catch {
            return false;
        }
    }

    // ── Expose Public Interface ───────────────────────
    return {
        predict,
        healthCheck,
        API_BASE_URL,
    };
})();
