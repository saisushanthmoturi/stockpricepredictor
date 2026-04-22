/**
 * app.js — Main Application Logic
 *
 * Handles:
 *   1. Form submission and validation
 *   2. Loading / error / results state management
 *   3. Populating the dashboard with API response data
 *   4. Loading text animation during model training
 *
 * Dependencies: api.js, charts.js (must be loaded before this file)
 */

document.addEventListener("DOMContentLoaded", () => {
    // ── DOM References ────────────────────────────────
    const form           = document.getElementById("prediction-form");
    const tickerInput    = document.getElementById("ticker-input");
    const periodSelect   = document.getElementById("period-select");
    const futureInput    = document.getElementById("future-input");
    const predictBtn     = document.getElementById("predict-btn");

    const loadingSection = document.getElementById("loading-section");
    const loadingText    = document.getElementById("loading-text");
    const errorSection   = document.getElementById("error-section");
    const errorMessage   = document.getElementById("error-message");
    const errorHint      = document.getElementById("error-hint");
    const resultsSection = document.getElementById("results-section");

    // ── State ─────────────────────────────────────────
    let isLoading = false;
    let loadingInterval = null;

    // ── Loading Messages (cycle through these) ────────
    const loadingMessages = [
        "Fetching market data...",
        "Computing technical indicators...",
        "Generating training sequences...",
        "Building LSTM + Attention model...",
        "Training neural network...",
        "Optimizing weights (this takes a bit)...",
        "Training in progress — patience pays off...",
        "Evaluating model accuracy...",
        "Predicting future prices...",
        "Almost there — assembling results...",
    ];

    // ────────────────────────────────────────────────────
    //  State Management
    // ────────────────────────────────────────────────────

    function showLoading() {
        isLoading = true;
        loadingSection.classList.add("active");
        errorSection.classList.remove("active");
        resultsSection.classList.remove("active");
        predictBtn.disabled = true;

        // Cycle through loading messages
        let msgIndex = 0;
        loadingText.textContent = loadingMessages[0];
        loadingInterval = setInterval(() => {
            msgIndex = Math.min(msgIndex + 1, loadingMessages.length - 1);
            loadingText.textContent = loadingMessages[msgIndex];
        }, 5000);
    }

    function showError(message, hint = "") {
        isLoading = false;
        clearInterval(loadingInterval);
        loadingSection.classList.remove("active");
        resultsSection.classList.remove("active");
        errorSection.classList.add("active");
        predictBtn.disabled = false;

        errorMessage.textContent = message;
        errorHint.textContent = hint || "Please check the ticker symbol and try again.";
    }

    function showResults(data) {
        isLoading = false;
        clearInterval(loadingInterval);
        loadingSection.classList.remove("active");
        errorSection.classList.remove("active");
        resultsSection.classList.add("active");
        predictBtn.disabled = false;

        renderResults(data);
    }

    // ────────────────────────────────────────────────────
    //  Form Handling
    // ────────────────────────────────────────────────────

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (isLoading) return;

        const ticker = tickerInput.value.trim().toUpperCase();
        const period = periodSelect.value;
        const futureDays = parseInt(futureInput.value) || 30;

        if (!ticker) {
            tickerInput.focus();
            return;
        }

        if (futureDays < 1 || futureDays > 90) {
            showError("Forecast days must be between 1 and 90.");
            return;
        }

        showLoading();

        try {
            const data = await StockAPI.predict(ticker, period, futureDays);
            showResults(data);
        } catch (err) {
            showError(
                err.message || "Failed to get predictions",
                "Make sure the backend is running and the ticker is valid."
            );
        }
    });

    // ── Ticker chip shortcuts ─────────────────────────
    document.querySelectorAll(".ticker-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            tickerInput.value = chip.dataset.ticker;
            // Trigger form submit
            form.dispatchEvent(new Event("submit", { cancelable: true }));
        });
    });

    // ────────────────────────────────────────────────────
    //  Render Results
    // ────────────────────────────────────────────────────

    function renderResults(data) {
        // ── Ticker Info Bar ──────────────────────────
        document.getElementById("result-ticker").textContent = data.ticker;

        const periodMap = { "1y": "1 Year", "2y": "2 Years", "5y": "5 Years" };
        document.getElementById("result-period").textContent =
            `${periodMap[periodSelect.value] || periodSelect.value} of Historical Data`;

        // Stats: current price, prediction range, data points
        const lastPrice = data.historical_prices[data.historical_prices.length - 1].price;
        const futureHigh = Math.max(...data.future_predictions.map(p => p.price));
        const futureLow = Math.min(...data.future_predictions.map(p => p.price));

        document.getElementById("ticker-stats").innerHTML = `
            <div class="stat-mini">
                <div class="stat-mini__value">$${lastPrice.toFixed(2)}</div>
                <div class="stat-mini__label">Last Close</div>
            </div>
            <div class="stat-mini">
                <div class="stat-mini__value">$${futureLow.toFixed(2)}</div>
                <div class="stat-mini__label">Forecast Low</div>
            </div>
            <div class="stat-mini">
                <div class="stat-mini__value">$${futureHigh.toFixed(2)}</div>
                <div class="stat-mini__label">Forecast High</div>
            </div>
            <div class="stat-mini">
                <div class="stat-mini__value">${data.historical_prices.length}</div>
                <div class="stat-mini__label">Data Points</div>
            </div>
        `;

        // ── Metrics Cards ────────────────────────────
        animateMetric("metric-rmse", data.metrics.rmse);
        animateMetric("metric-mae", data.metrics.mae);
        animateMetric("metric-mape", data.metrics.mape, "%");
        animateMetric("metric-r2", data.metrics.r2_score);

        // ── Charts ───────────────────────────────────
        StockCharts.createPriceChart(data);
        StockCharts.createLossChart(data.training_history);
        StockCharts.createFutureChart(data.future_predictions, lastPrice);

        // ── Predictions Table ────────────────────────
        renderPredictionsTable(data.future_predictions, lastPrice);

        // ── Architecture Info (update from model_info) ─
        if (data.model_info) {
            const mi = data.model_info;
            document.getElementById("arch-input-info").textContent =
                `${mi.lookback_window} × ${mi.features_used.length}`;
            document.getElementById("arch-lstm1-info").textContent =
                `${mi.lstm_layer_1_units} units`;
            document.getElementById("arch-lstm2-info").textContent =
                `${mi.lstm_layer_2_units} units`;

            renderModelInfo(mi);
        }

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ────────────────────────────────────────────────────
    //  Animated Metric Counter
    // ────────────────────────────────────────────────────

    function animateMetric(elementId, targetValue, suffix = "") {
        const el = document.getElementById(elementId);
        if (!el) return;

        const duration = 1200;
        const steps = 40;
        const stepDuration = duration / steps;
        let current = 0;
        const increment = targetValue / steps;

        const timer = setInterval(() => {
            current += increment;
            if (current >= targetValue) {
                current = targetValue;
                clearInterval(timer);
            }
            el.textContent = current.toFixed(4) + suffix;
        }, stepDuration);
    }

    // ────────────────────────────────────────────────────
    //  Predictions Table
    // ────────────────────────────────────────────────────

    function renderPredictionsTable(predictions, lastPrice) {
        const tbody = document.getElementById("predictions-tbody");
        if (!tbody) return;

        let prevPrice = lastPrice;

        tbody.innerHTML = predictions.map((pred, i) => {
            const change = ((pred.price - prevPrice) / prevPrice) * 100;
            const changeFromStart = ((pred.price - lastPrice) / lastPrice) * 100;
            const changeClass = change >= 0 ? "change-positive" : "change-negative";
            const changeIcon = change >= 0 ? "▲" : "▼";

            prevPrice = pred.price;

            return `
                <tr>
                    <td>Day ${i + 1}</td>
                    <td>${pred.date}</td>
                    <td>$${pred.price.toFixed(2)}</td>
                    <td class="${changeClass}">${changeIcon} ${Math.abs(changeFromStart).toFixed(2)}%</td>
                </tr>
            `;
        }).join("");
    }

    // ────────────────────────────────────────────────────
    //  Model Info Grid
    // ────────────────────────────────────────────────────

    function renderModelInfo(modelInfo) {
        const grid = document.getElementById("model-info-grid");
        if (!grid) return;

        const items = [
            { label: "Architecture", value: modelInfo.architecture },
            { label: "Optimizer", value: modelInfo.optimizer },
            { label: "Loss Function", value: modelInfo.loss_function },
            { label: "Lookback Window", value: `${modelInfo.lookback_window} days` },
            { label: "Epochs Trained", value: modelInfo.epochs_trained },
            { label: "Batch Size", value: modelInfo.batch_size },
            { label: "Dropout Rate", value: `${(modelInfo.dropout_rate * 100).toFixed(0)}%` },
            { label: "Features", value: modelInfo.features_used.length },
        ];

        grid.innerHTML = items.map(item => `
            <div class="glass-card model-info-item">
                <div class="model-info-item__label">${item.label}</div>
                <div class="model-info-item__value">${item.value}</div>
            </div>
        `).join("");
    }
});
