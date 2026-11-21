"""Vercel entrypoint for the Task 16 forecast API."""
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

FORECAST_PATH = Path(__file__).resolve().parent / "outputs" / "headcount_forecast.csv"

app = FastAPI(title="CortexHR Task 16 Forecast API", version="1.0.0")

_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#101a2b">
  <title>CortexHR | Workforce outlook</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0b1220;
      color: #e8eef8;
      font-synthesis: none;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; background: radial-gradient(ellipse at 15% 0%, #173050 0, #0b1220 48rem); }
    a { color: #8fc9ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .shell { max-width: 1120px; margin: 0 auto; padding: 36px 24px 56px; }
    .topbar, .brand, .actions, .section-heading, .filters { display: flex; align-items: center; }
    .topbar, .section-heading { justify-content: space-between; gap: 20px; }
    .brand { gap: 12px; color: #c9d8ec; font-size: 14px; font-weight: 650; letter-spacing: .04em; }
    .brand-mark { display: grid; width: 36px; height: 36px; place-items: center; border: 1px solid #31628c; border-radius: 11px; background: #142a42; color: #8fc9ff; font-weight: 800; }
    .actions { gap: 18px; font-size: 13px; }
    .eyebrow { margin: 0 0 10px; color: #7ebcff; font-size: 12px; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
    h1 { margin: 0; font-size: clamp(32px, 5vw, 48px); letter-spacing: -.045em; line-height: 1.08; }
    .intro { max-width: 670px; margin: 14px 0 0; color: #a9b8cb; font-size: 16px; line-height: 1.6; }
    .hero { padding: 72px 0 38px; }
    .status { display: inline-flex; align-items: center; gap: 8px; padding: 8px 11px; border: 1px solid #234a3a; border-radius: 999px; background: #10251e; color: #a8e8c1; font-size: 12px; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #52d18a; box-shadow: 0 0 12px #52d18a; }
    .cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 4px 0 34px; }
    .card, .panel { border: 1px solid #26354a; border-radius: 16px; background: linear-gradient(145deg, rgba(24, 37, 56, .96), rgba(15, 25, 40, .96)); }
    .card { min-height: 130px; padding: 20px; }
    .card-label { color: #9eafc4; font-size: 13px; }
    .card-value { margin: 14px 0 6px; font-size: 29px; font-weight: 700; letter-spacing: -.035em; }
    .card-note { color: #8192a9; font-size: 12px; }
    .panel { overflow: hidden; }
    .section-heading { padding: 22px 24px 18px; }
    h2 { margin: 0; font-size: 18px; letter-spacing: -.02em; }
    .section-note { margin: 6px 0 0; color: #91a2b8; font-size: 13px; }
    .filters { gap: 10px; }
    select { min-width: 155px; padding: 9px 30px 9px 11px; border: 1px solid #364860; border-radius: 9px; background: #111d2d; color: #dce7f5; font: inherit; font-size: 13px; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; text-align: left; }
    th { padding: 12px 18px; border-top: 1px solid #29384d; border-bottom: 1px solid #29384d; background: rgba(11, 18, 32, .35); color: #8fa2ba; font-size: 11px; font-weight: 650; letter-spacing: .07em; text-transform: uppercase; white-space: nowrap; }
    td { padding: 14px 18px; border-bottom: 1px solid rgba(41, 56, 77, .72); color: #d7e1ef; font-size: 13px; white-space: nowrap; }
    tbody tr:hover { background: rgba(87, 157, 224, .06); }
    .number { font-variant-numeric: tabular-nums; font-weight: 650; }
    .empty { padding: 34px 18px; color: #9eafc4; text-align: center; }
    .footer { display: flex; justify-content: space-between; gap: 16px; padding: 16px 24px; color: #8192a9; font-size: 12px; }
    .error { display: none; margin: 18px 0; padding: 14px 16px; border: 1px solid #783e48; border-radius: 10px; background: #2a1820; color: #ffc0c8; }
    @media (max-width: 720px) {
      .shell { padding: 22px 16px 36px; }
      .topbar, .section-heading { align-items: flex-start; flex-direction: column; }
      .hero { padding: 48px 0 28px; }
      .cards { grid-template-columns: 1fr; gap: 10px; }
      .card { min-height: 112px; }
      .filters { width: 100%; }
      select { flex: 1; min-width: 0; }
      .footer { flex-direction: column; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <nav class="topbar" aria-label="Main navigation">
      <div class="brand"><span class="brand-mark">C</span><span>CORTEXHR <span style="color:#70839a">/</span> PEOPLE ANALYTICS</span></div>
      <div class="actions"><a href="/health">Service health</a><a href="/forecast">Forecast data</a></div>
    </nav>
    <section class="hero">
      <p class="eyebrow">Task 16 · Planning horizon</p>
      <div class="section-heading" style="padding:0">
        <div>
          <h1>Workforce outlook</h1>
          <p class="intro">A six-month view of projected headcount and attrition, with 80% prediction intervals to help plan with uncertainty in mind.</p>
        </div>
        <span class="status"><span class="dot"></span>Forecast ready</span>
      </div>
    </section>
    <section class="cards" aria-label="Forecast summary">
      <article class="card"><div class="card-label">Company headcount · next month</div><div class="card-value" id="headcount-value">Loading…</div><div class="card-note" id="headcount-note">Preparing forecast summary</div></article>
      <article class="card"><div class="card-label">Forecast horizon</div><div class="card-value" id="horizon-value">—</div><div class="card-note">Six monthly periods</div></article>
      <article class="card"><div class="card-label">Forecast records</div><div class="card-value" id="record-count">—</div><div class="card-note" id="trained-through">Including company and department outlooks</div></article>
    </section>
    <section class="panel" aria-labelledby="table-title">
      <div class="section-heading">
        <div><h2 id="table-title">Forecast details</h2><p class="section-note">Explore the forecast by measure and department.</p></div>
        <div class="filters">
          <label><span class="sr-only" style="position:absolute;left:-10000px">Measure</span><select id="metric-filter" aria-label="Filter by measure"><option value="">All measures</option></select></label>
          <label><span class="sr-only" style="position:absolute;left:-10000px">Department</span><select id="department-filter" aria-label="Filter by department"><option value="">All departments</option></select></label>
        </div>
      </div>
      <div class="error" id="load-error" role="alert"></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Month</th><th>Department</th><th>Measure</th><th>Forecast</th><th>80% interval</th><th>Method</th></tr></thead>
          <tbody id="forecast-table"><tr><td class="empty" colspan="6">Loading forecast…</td></tr></tbody>
        </table>
      </div>
      <div class="footer"><span id="table-count">Loading forecast data…</span><span>Intervals reflect model uncertainty, not guaranteed limits.</span></div>
    </section>
  </main>
  <script>
    const metricFilter = document.getElementById("metric-filter");
    const departmentFilter = document.getElementById("department-filter");
    const tableBody = document.getElementById("forecast-table");
    const numberFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
    let forecasts = [];

    function formatMonth(value) {
      const date = new Date(value + "T00:00:00");
      return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { month: "short", year: "numeric" }).format(date);
    }

    function measureName(value) {
      return value === "headcount_end" ? "Headcount" : value === "attrition_rate_monthly_pct" ? "Monthly attrition" : value;
    }

    function addOption(select, value, label) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.append(option);
    }

    function renderTable() {
      const metric = metricFilter.value;
      const department = departmentFilter.value;
      const rows = forecasts.filter((row) =>
        (!metric || row.metric === metric) && (!department || row.department_id === department)
      );
      tableBody.replaceChildren();
      if (!rows.length) {
        const row = tableBody.insertRow();
        const cell = row.insertCell();
        cell.colSpan = 6;
        cell.className = "empty";
        cell.textContent = "No forecasts match these filters.";
      } else {
        for (const forecast of rows) {
          const row = tableBody.insertRow();
          const values = [
            formatMonth(forecast.forecast_month),
            forecast.department_id,
            measureName(forecast.metric),
            numberFormat.format(Number(forecast.forecast_value)),
            `${numberFormat.format(Number(forecast.lower_80))} – ${numberFormat.format(Number(forecast.upper_80))}`,
            forecast.method
          ];
          values.forEach((value, index) => {
            const cell = row.insertCell();
            cell.textContent = value ?? "—";
            if (index === 3) cell.className = "number";
          });
        }
      }
      document.getElementById("table-count").textContent = `${rows.length} forecast records`;
    }

    async function loadForecast() {
      try {
        const response = await fetch("/forecast");
        if (!response.ok) throw new Error(`Forecast request failed (${response.status}).`);
        const payload = await response.json();
        forecasts = payload.forecasts;
        if (!Array.isArray(forecasts) || forecasts.length === 0) throw new Error("No forecast records are available.");

        const totalHeadcount = forecasts.filter((row) => row.metric === "headcount_end" && row.department_id === "ALL");
        if (totalHeadcount.length) {
          document.getElementById("headcount-value").textContent = numberFormat.format(Number(totalHeadcount[0].forecast_value));
          document.getElementById("headcount-note").textContent = `Company total · ${formatMonth(totalHeadcount[0].forecast_month)}`;
        }
        const months = [...new Set(forecasts.map((row) => row.forecast_month))].sort();
        document.getElementById("horizon-value").textContent = months.length ? `${formatMonth(months[0])} – ${formatMonth(months[months.length - 1])}` : "—";
        document.getElementById("record-count").textContent = numberFormat.format(forecasts.length);
        const trainedThrough = forecasts[0].trained_through;
        if (trainedThrough) document.getElementById("trained-through").textContent = `Model trained through ${trainedThrough}`;

        [...new Set(forecasts.map((row) => row.metric))].forEach((value) => addOption(metricFilter, value, measureName(value)));
        [...new Set(forecasts.map((row) => row.department_id))].sort().forEach((value) => addOption(departmentFilter, value, value === "ALL" ? "Company total" : value));
        metricFilter.addEventListener("change", renderTable);
        departmentFilter.addEventListener("change", renderTable);
        renderTable();
      } catch (error) {
        document.getElementById("load-error").textContent = error.message || "Unable to load forecast data.";
        document.getElementById("load-error").style.display = "block";
        tableBody.replaceChildren();
        const row = tableBody.insertRow();
        const cell = row.insertCell();
        cell.colSpan = 6;
        cell.className = "empty";
        cell.textContent = "Forecast data could not be loaded.";
        document.getElementById("table-count").textContent = "Forecast unavailable";
      }
    }

    loadForecast();
  </script>
</body>
</html>"""


@app.get("/")
def root() -> HTMLResponse:
    return HTMLResponse(_DASHBOARD_HTML)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/forecast")
def forecast() -> dict[str, list[dict[str, object]]]:
    if not FORECAST_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Forecast is unavailable. Run scripts/run_task16.py to generate it.",
        )

    return {"forecasts": pd.read_csv(FORECAST_PATH).to_dict(orient="records")}
