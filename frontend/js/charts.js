/**
 * charts.js — Chart.js Visualization Configurations
 *
 * Creates and manages three charts:
 *   1. Price Chart    — Historical prices + train/test predictions + future forecast
 *   2. Loss Chart     — Training vs validation loss over epochs
 *   3. Future Chart   — Zoomed view of future predicted prices
 *
 * Design:
 *   - Dark theme matching the CSS glassmorphism style
 *   - Gradient fills for visual depth
 *   - Smooth animations on data load
 *   - Responsive sizing
 */

const StockCharts = (() => {
    // Store chart instances so we can destroy them before re-creating
    let priceChart = null;
    let lossChart = null;
    let futureChart = null;

    // ── Chart.js Global Defaults (dark theme) ─────────
    Chart.defaults.color = "#a0a0b8";
    Chart.defaults.borderColor = "rgba(255, 255, 255, 0.06)";
    Chart.defaults.font.family = "'Inter', sans-serif";

    // ── Color Palette ─────────────────────────────────
    const COLORS = {
        cyan: "#00d4ff",
        cyanFaded: "rgba(0, 212, 255, 0.15)",
        purple: "#a855f7",
        purpleFaded: "rgba(168, 85, 247, 0.15)",
        green: "#22c55e",
        greenFaded: "rgba(34, 197, 94, 0.15)",
        red: "#ef4444",
        redFaded: "rgba(239, 68, 68, 0.1)",
        orange: "#f59e0b",
        orangeFaded: "rgba(245, 158, 11, 0.15)",
        white: "rgba(255, 255, 255, 0.7)",
        whiteFaded: "rgba(255, 255, 255, 0.05)",
    };

    // ────────────────────────────────────────────────────
    //  1. PRICE PREDICTION CHART (main chart)
    // ────────────────────────────────────────────────────

    function createPriceChart(data) {
        const ctx = document.getElementById("price-chart");
        if (!ctx) return;

        // Destroy existing chart
        if (priceChart) {
            priceChart.destroy();
            priceChart = null;
        }

        // ── Prepare datasets ─────────────────────────

        // Historical actual prices
        const historicalDates = data.historical_prices.map(p => p.date);
        const historicalPrices = data.historical_prices.map(p => p.price);

        // Train predictions (aligned with historical dates)
        const trainDates = data.train_predictions.map(p => p.date);
        const trainPrices = data.train_predictions.map(p => p.price);

        // Test predictions
        const testDates = data.test_predictions.map(p => p.date);
        const testPrices = data.test_predictions.map(p => p.price);

        // Future predictions
        const futureDates = data.future_predictions.map(p => p.date);
        const futurePrices = data.future_predictions.map(p => p.price);

        // Combined date labels for x-axis
        const allDates = [...new Set([...historicalDates, ...trainDates, ...testDates, ...futureDates])].sort();

        // Map each dataset to align with allDates (null for missing dates)
        const mapToAllDates = (dates, prices) => {
            const lookup = {};
            dates.forEach((d, i) => lookup[d] = prices[i]);
            return allDates.map(d => lookup[d] !== undefined ? lookup[d] : null);
        };

        priceChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: allDates,
                datasets: [
                    {
                        label: "Actual Price",
                        data: mapToAllDates(historicalDates, historicalPrices),
                        borderColor: COLORS.white,
                        backgroundColor: COLORS.whiteFaded,
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: true,
                        tension: 0.1,
                        order: 3,
                    },
                    {
                        label: "Train Prediction",
                        data: mapToAllDates(trainDates, trainPrices),
                        borderColor: COLORS.cyan,
                        backgroundColor: COLORS.cyanFaded,
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1,
                        order: 2,
                    },
                    {
                        label: "Test Prediction",
                        data: mapToAllDates(testDates, testPrices),
                        borderColor: COLORS.green,
                        backgroundColor: COLORS.greenFaded,
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1,
                        order: 1,
                    },
                    {
                        label: "Future Forecast",
                        data: mapToAllDates(futureDates, futurePrices),
                        borderColor: COLORS.orange,
                        backgroundColor: COLORS.orangeFaded,
                        borderWidth: 2.5,
                        borderDash: [6, 3],
                        pointRadius: 0,
                        fill: true,
                        tension: 0.2,
                        order: 0,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1500,
                    easing: "easeOutQuart",
                },
                interaction: {
                    mode: "index",
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: "top",
                        labels: {
                            usePointStyle: true,
                            pointStyle: "circle",
                            padding: 20,
                            font: { size: 12 },
                        },
                    },
                    tooltip: {
                        backgroundColor: "rgba(10, 10, 26, 0.9)",
                        borderColor: "rgba(255, 255, 255, 0.1)",
                        borderWidth: 1,
                        titleFont: { size: 13, weight: 600 },
                        bodyFont: { size: 12 },
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                if (context.parsed.y !== null) {
                                    return `${context.dataset.label}: $${context.parsed.y.toFixed(2)}`;
                                }
                                return null;
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        type: "category",
                        ticks: {
                            maxTicksLimit: 12,
                            maxRotation: 45,
                            font: { size: 10 },
                        },
                        grid: {
                            display: false,
                        },
                    },
                    y: {
                        ticks: {
                            callback: val => "$" + val.toFixed(0),
                            font: { size: 11 },
                        },
                        grid: {
                            color: "rgba(255, 255, 255, 0.04)",
                        },
                    },
                },
            },
        });
    }

    // ────────────────────────────────────────────────────
    //  2. TRAINING LOSS CHART
    // ────────────────────────────────────────────────────

    function createLossChart(trainingHistory) {
        const ctx = document.getElementById("loss-chart");
        if (!ctx) return;

        if (lossChart) {
            lossChart.destroy();
            lossChart = null;
        }

        const epochs = trainingHistory.loss.map((_, i) => `Epoch ${i + 1}`);

        lossChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: "Training Loss",
                        data: trainingHistory.loss,
                        borderColor: COLORS.cyan,
                        backgroundColor: COLORS.cyanFaded,
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: true,
                        tension: 0.3,
                    },
                    {
                        label: "Validation Loss",
                        data: trainingHistory.val_loss,
                        borderColor: COLORS.purple,
                        backgroundColor: COLORS.purpleFaded,
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: true,
                        tension: 0.3,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 1200 },
                interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: {
                        position: "top",
                        labels: {
                            usePointStyle: true,
                            pointStyle: "circle",
                            padding: 15,
                            font: { size: 11 },
                        },
                    },
                    tooltip: {
                        backgroundColor: "rgba(10, 10, 26, 0.9)",
                        borderColor: "rgba(255, 255, 255, 0.1)",
                        borderWidth: 1,
                        padding: 10,
                        cornerRadius: 8,
                    },
                },
                scales: {
                    x: {
                        ticks: {
                            maxTicksLimit: 8,
                            font: { size: 10 },
                        },
                        grid: { display: false },
                    },
                    y: {
                        title: {
                            display: true,
                            text: "MSE Loss",
                            font: { size: 11 },
                        },
                        ticks: { font: { size: 10 } },
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                    },
                },
            },
        });
    }

    // ────────────────────────────────────────────────────
    //  3. FUTURE FORECAST CHART (zoomed)
    // ────────────────────────────────────────────────────

    function createFutureChart(futurePredictions, lastKnownPrice) {
        const ctx = document.getElementById("future-chart");
        if (!ctx) return;

        if (futureChart) {
            futureChart.destroy();
            futureChart = null;
        }

        const dates = futurePredictions.map(p => p.date);
        const prices = futurePredictions.map(p => p.price);

        // Add last known price as starting reference point
        const allDates = ["Last Close", ...dates];
        const allPrices = [lastKnownPrice, ...prices];

        // Color gradient based on price direction
        const isUptrend = prices[prices.length - 1] >= lastKnownPrice;
        const lineColor = isUptrend ? COLORS.green : COLORS.red;
        const fillColor = isUptrend ? COLORS.greenFaded : COLORS.redFaded;

        futureChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: allDates,
                datasets: [
                    {
                        label: "Forecasted Price",
                        data: allPrices,
                        borderColor: lineColor,
                        backgroundColor: fillColor,
                        borderWidth: 2.5,
                        pointRadius: 3,
                        pointBackgroundColor: lineColor,
                        pointBorderColor: "transparent",
                        pointHoverRadius: 6,
                        fill: true,
                        tension: 0.3,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 1200 },
                interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: {
                        position: "top",
                        labels: {
                            usePointStyle: true,
                            pointStyle: "circle",
                            padding: 15,
                            font: { size: 11 },
                        },
                    },
                    tooltip: {
                        backgroundColor: "rgba(10, 10, 26, 0.9)",
                        borderColor: "rgba(255, 255, 255, 0.1)",
                        borderWidth: 1,
                        padding: 10,
                        cornerRadius: 8,
                        callbacks: {
                            label: ctx => `$${ctx.parsed.y.toFixed(2)}`,
                        },
                    },
                },
                scales: {
                    x: {
                        ticks: {
                            maxTicksLimit: 8,
                            maxRotation: 45,
                            font: { size: 10 },
                        },
                        grid: { display: false },
                    },
                    y: {
                        ticks: {
                            callback: val => "$" + val.toFixed(0),
                            font: { size: 10 },
                        },
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                    },
                },
            },
        });
    }

    // ── Public API ────────────────────────────────────
    return {
        createPriceChart,
        createLossChart,
        createFutureChart,
    };
})();
