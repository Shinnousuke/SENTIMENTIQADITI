
const API_BASE = "/api/indic-evaluation";

let predictionOffset = 0;
const predictionLimit = 50;
let languageChart = null;

async function fetchAPI(endpoint) {
    const response = await fetch(`${API_BASE}/${endpoint}`);

    if (!response.ok) {
        throw new Error(`API error ${response.status}: ${endpoint}`);
    }

    return response.json();
}

function formatValue(value) {
    if (typeof value === "number") {
        return Number.isInteger(value)
            ? value.toLocaleString()
            : value.toFixed(4);
    }

    if (value && typeof value === "object") {
        return JSON.stringify(value);
    }

    return String(value ?? "--");
}

function flattenMetrics(data, prefix = "") {
    const result = {};

    if (!data || typeof data !== "object") {
        return result;
    }

    for (const [key, value] of Object.entries(data)) {
        const name = prefix ? `${prefix} / ${key}` : key;

        if (value && typeof value === "object" && !Array.isArray(value)) {
            Object.assign(result, flattenMetrics(value, name));
        } else if (
            value === null ||
            ["string", "number", "boolean"].includes(typeof value)
        ) {
            result[name] = value;
        }
    }

    return result;
}

function renderMetricCards(data) {
    const container = document.getElementById("metric-cards");
    const metrics = flattenMetrics(data);

    container.innerHTML = "";

    const entries = Object.entries(metrics);

    if (!entries.length) {
        container.textContent = "No overall metrics available.";
        return;
    }

    entries.forEach(([name, value]) => {
        const card = document.createElement("div");
        card.className = "indic-metric-card";

        const label = document.createElement("span");
        label.textContent = name;

        const heading = document.createElement("h3");
        heading.textContent = formatValue(value);

        card.append(label, heading);
        container.appendChild(card);
    });
}

function renderTable(tableId, rows) {
    const table = document.getElementById(tableId);
    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");

    thead.innerHTML = "";
    tbody.innerHTML = "";

    if (!Array.isArray(rows) || rows.length === 0) {
        tbody.innerHTML = "<tr><td>No records available.</td></tr>";
        return;
    }

    const columns = Object.keys(rows[0]);

    const headerRow = document.createElement("tr");

    columns.forEach(column => {
        const th = document.createElement("th");
        th.textContent = column;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);

    rows.forEach(row => {
        const tr = document.createElement("tr");

        columns.forEach(column => {
            const td = document.createElement("td");
            td.textContent = formatValue(row[column]);
            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });
}

function renderLanguageChart(rows) {
    if (!rows.length) return;

    const keys = Object.keys(rows[0]);

    const languageKey = keys.find(key =>
        key.toLowerCase().includes("language")
    ) || keys.find(key =>
        key.toLowerCase().includes("lang")
    );

    const metricNames = ["accuracy", "f1", "precision", "recall"];

    const metricKeys = metricNames
        .map(metric => keys.find(key =>
            key.toLowerCase().includes(metric)
        ))
        .filter(Boolean);

    if (!languageKey || !metricKeys.length) {
        console.warn("Could not identify language or metric columns.");
        return;
    }

    const labels = rows.map(row => row[languageKey]);

    const datasets = metricKeys.map(key => ({
        label: key,
        data: rows.map(row => Number(row[key])),
        borderWidth: 1
    }));

    const ctx = document.getElementById("language-chart");

    if (languageChart) {
        languageChart.destroy();
    }

    languageChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

async function loadSummary() {
    const data = await fetchAPI("summary");
    renderMetricCards(data);
}

async function loadLanguages() {
    const rows = await fetchAPI("languages");

    document.getElementById("language-count").textContent = rows.length;

    renderTable("language-table", rows);
    renderLanguageChart(rows);
}

async function loadErrors() {
    const data = await fetchAPI("errors");

    document.getElementById("error-count").textContent = data.total;

    renderTable("error-table", data.results);
}

async function loadPredictions() {
    const data = await fetchAPI(
        `predictions?limit=${predictionLimit}&offset=${predictionOffset}`
    );

    document.getElementById("prediction-count").textContent =
        data.total.toLocaleString();

    const table = document.getElementById("prediction-table");

    if (predictionOffset === 0) {
        renderTable("prediction-table", data.results);
    } else {
        const tbody = table.querySelector("tbody");

        data.results.forEach(row => {
            const tr = document.createElement("tr");

            Object.values(row).forEach(value => {
                const td = document.createElement("td");
                td.textContent = formatValue(value);
                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });
    }

    predictionOffset += data.results.length;

    document.getElementById("load-more").style.display =
        predictionOffset >= data.total ? "none" : "inline-block";
}

async function loadCharts() {
    const charts = await fetchAPI("charts");
    const gallery = document.getElementById("chart-gallery");

    gallery.innerHTML = "";

    if (!charts.length) {
        gallery.textContent = "No PNG charts found.";
        return;
    }

    charts.forEach(chart => {
        const figure = document.createElement("figure");
        const image = document.createElement("img");
        const caption = document.createElement("figcaption");

        image.src = chart.url;
        image.alt = chart.name;

        caption.textContent = chart.name;

        figure.append(image, caption);
        gallery.appendChild(figure);
    });
}

async function initializeIndicDashboard() {
    try {
        await Promise.all([
            loadSummary(),
            loadLanguages(),
            loadErrors(),
            loadPredictions(),
            loadCharts()
        ]);
    } catch (error) {
        console.error("Multilingual dashboard error:", error);
        alert(
            "Unable to load multilingual evaluation data. " +
            "Check the backend server and evaluation files."
        );
    }
}

document.getElementById("load-more").addEventListener("click", () => {
    loadPredictions().catch(console.error);
});

initializeIndicDashboard();
