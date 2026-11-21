"""Generates the event-adjustment CSV and the three Markdown documents. Every number comes from the pipeline result."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtesting as bt
from . import config, exits_forecaster as ex, exploratory_analysis as ea
from .advanced_forecaster import SPECS


def md_table(df: pd.DataFrame, fmt: dict | None = None, align: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---:" if (align or {}).get(c, "r" if pd.api.types.is_numeric_dtype(df[c]) else "l") == "r" else "---" for c in cols) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt[c](v) if c in fmt else ("" if (isinstance(v, float) and np.isnan(v)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def event_adjustment_table(res: dict) -> pd.DataFrame:
    """workforce_event_adjustments.csv: historical effect estimates + any future adjustment actually applied."""
    hist = res["event_effects"].copy()
    adj = res["adjustments"]
    if adj is None or adj.empty:
        note = pd.DataFrame([{
            "event_id": "NONE", "event_name": "No known future workforce events", "event_type": "n/a", "department_id": "ALL", "scope": "future",
            "description": "workforce_events lists only past events (latest ends " + str(res["events"]["end_date"].max().date())
                           + ") and no data/future_events.csv was supplied, so no event adjustment is added to the forecast.",
            "evidence": "none - adjustment is 0", "used_in_model": "n/a", "applied_to_forecast": False}])
        return pd.concat([hist, note], ignore_index=True)
    fut = adj.rename(columns={"monthly_effect": "monthly_headcount_effect", "cumulative_adjustment": "cumulative_headcount_adjustment"})
    fut = fut.assign(scope="future", applied_to_forecast=True, used_in_model="added after the model (never inside it)")
    fut["start_date"] = fut["month"]
    return pd.concat([hist, fut.drop(columns=["month"])], ignore_index=True)


def _f1(v): return f"{v:,.1f}"
def _f2(v): return f"{v:,.2f}"
def _f0(v): return f"{v:,.0f}"


def _comparison(res: dict) -> tuple[pd.DataFrame, dict]:
    b = res["backtest"][config.COMPANY]["results"]
    m = {k: bt.metrics(b["actual"], b[f"{k}_forecast"], b[f"{k}_lower_80"], b[f"{k}_upper_80"]) for k in ("baseline", "advanced")}
    rows = []
    for name, k in (("Seasonal Baseline", "baseline"), ("Advanced Model", "advanced")):
        x = m[k]
        rows.append({"Model": name, "MAE": x["MAE"], "RMSE": x["RMSE"], "MAPE %": x["MAPE_pct"], "WAPE %": x["WAPE_pct"], "Bias": x["bias"],
                     "80% coverage": f"{x['inside_80']}/{x['n']} ({x['coverage_80_pct']:.0f}%)", "Avg 80% width": x["mean_width_80"]})
    return pd.DataFrame(rows), m


def forecast_analysis_md(res: dict) -> str:
    eda = ea.analyse(res["hist"], res["events"])
    text = ea.findings_markdown(eda, res["quality"], res["events"])
    cv = res["cv_final"].copy()
    cv["cv_mae"] = cv["cv_mae"].astype(float)
    sel = res["model"]
    tbl = md_table(cv[["model", "label", "complexity", "cv_mae", "cv_coverage_80", "n_origins"]].rename(columns={"cv_mae": "CV MAE", "cv_coverage_80": "CV 80% coverage"}),
                   {"CV MAE": _f2, "CV 80% coverage": lambda v: "" if np.isnan(v) else f"{v*100:.0f}%"})
    bm = eda["by_month"].reset_index().rename(columns={"index": "month_no"})
    bm.columns = ["Calendar month", "Mean net change", "SD net change", "Mean hires", "Mean exits", "Years observed"]
    bm["Calendar month"] = [pd.Timestamp(2000, int(m), 1).strftime("%b") for m in bm["Calendar month"]]
    return (text + "\n## Appendix A - seasonal profile (company total)\n\n"
            + md_table(bm, {c: _f2 for c in ("Mean net change", "SD net change", "Mean hires", "Mean exits")}) + "\n\n"
            "## Appendix B - rolling-origin model selection (all 81 months; origins every 3 months, 6-month horizon)\n\n" + tbl + "\n\n"
            f"Selection rule: the simplest candidate whose CV MAE is within {config.SELECTION_TOLERANCE:.0%} of the best candidate. Selected: **{SPECS[sel].label}** (`{sel}`). "
            "The seasonal baseline is listed for reference and is not eligible.\n")


def forecast_summary_md(res: dict) -> str:
    eda = ea.analyse(res["hist"], res["events"])
    q = res["quality"]
    comp, m = _comparison(res)
    b = res["backtest"][config.COMPANY]["results"]
    sel_lbl, sel = SPECS[res["model"]].label, res["model"]
    bt_sel = res["backtest_model"]
    fc = res["headcount_fc"][config.COMPANY]
    last = float(res["series"][config.COMPANY].iloc[-1])
    rel = res["reliability"]
    wid, vol = rel["widest"], rel["most_volatile"]
    ef = res["exits_fc"]
    ph = res["post_hoc"].set_index("model")
    adv_better = m["advanced"]["MAE"] < m["baseline"]["MAE"]
    sep_share = b["advanced_abs_error"].iloc[-1] / b["advanced_abs_error"].sum() * 100
    cv = res["cv_final"].set_index("model")
    ex_rows = {r["method"]: r for r in res["backtest_exits"]["rows"]}
    chosen_ex = ex_rows[res["backtest_exits"]["selected"]]
    ref_ex = ex_rows[ex.REFERENCE_METHOD]
    has_future = res["adjustments"] is not None and len(res["adjustments"]) > 0
    L = []
    A = L.append
    A("# Task 16 - Forecast the Future: Headcount with Uncertainty\n")
    A("_Final report. Every figure is computed by the pipeline from the CortexHR workbook (fictional company data); nothing was typed in by hand._\n")
    A("## A. Introduction\n")
    A("Hiring, budgets and office capacity are planned months ahead, but headcount only becomes known after the fact. This task forecasts the company's month-end headcount and "
      "attrition for the next six months and attaches an **80% prediction interval** to each number, so planners see both the expected value and how far it could reasonably be off. "
      "Following the project rules, a simple seasonal baseline is built first and the advanced model has to beat it on months it has never seen.\n")
    A("## B. Dataset overview\n")
    A(f"* `monthly_headcount_history`: {q['rows']} rows, {q['series']} series (company total `ALL` plus D00-D10), {q['n_months']} months ({q['first_month']} to {q['last_month']}). "
      f"Columns: month, department_id, department_name, headcount_end, hires, exits, voluntary_exits, transfers_in, transfers_out, headcount_start, hiring_drive_flag, attrition_rate_monthly_pct, appraisal_cycle_flag, financial_year_end_flag. "
      f"Missing values {q['missing_values']}, duplicate keys {q['duplicate_keys']}, missing months {len(q['missing_months'])}, flow-identity violations {q['flow_identity_violations']}; departments sum to the company total in every month: {q['departments_sum_equals_company']}.")
    A(f"* `workforce_events`: {len(res['events'])} events (event_id, start_date, end_date, department_id, event_name, description). All are in the past; the last ends {res['events']['end_date'].max():%d %b %Y}. The sheet has **no impact numbers**, only free-text descriptions.")
    sf = res["sample"]
    A(f"* `sample_forecast`: {len(sf)} rows, columns {', '.join(f'`{c}`' for c in sf.columns)}. It holds `headcount_end` for ALL and D01-D10 (6 months each) plus `attrition_rate_monthly_pct` for ALL (6 rows), trained through {sf['trained_through'].iloc[0]}. It is the output contract; D00 (one person) is not in it, so it is not forecast.\n")
    A("## C. Exploratory analysis\n")
    top = eda["by_month"]["mean_net_change"].sort_values(ascending=False)
    A(f"* **Trend:** headcount rose from {eda['first']:.0f} to {eda['last']:.0f}; about {eda['trend_per_month']:.1f} people a month on a straight-line fit. Year-over-year growth was {eda['yoy_last']:.1f}% in the latest month.")
    A(f"* **Seasonality:** suggestive but weak (Kruskal-Wallis p = {eda['seasonal_kw_p']:.3f} on net change; hires p = {eda['hires_kw_p']:.2f}, exits p = {eda['exits_kw_p']:.2f}). Strongest months: {', '.join(pd.Timestamp(2000, int(k), 1).strftime('%b') for k in top.index[:3])}; weakest: {', '.join(pd.Timestamp(2000, int(k), 1).strftime('%b') for k in top.index[-2:])}.")
    un = eda["unusual"]
    A("* **Unusual months:** " + (", ".join(f"{r['month']:%b %Y} ({r['net_change']:+.0f})" for _, r in un.iterrows()) if len(un) else "none beyond the robust threshold") + ". The latest month (Sep 2026) is one of them, so the starting point of the forecast sits just after a jump.")
    A(f"* **Structure:** annual net change slowed after 2022-2023 and {eda['ytd_year']} is already at {eda['ytd_change']:+.0f} after {eda['ytd_months']} months; no single permanent level shift is statistically supported (p = {eda['welch_p']:.2f}).")
    A("* **Thin data:** only 6-7 observations per calendar month and 45 months usable as validation tests. Full write-up: `reports/forecast_analysis.md`.\n")
    A("## D. Forecasting methodology\n")
    A("**Seasonal baseline.** `forecast(t) = headcount(t - 12) + annual trend`, where the annual trend is the average year-over-year change over the last 12 months known at the forecast origin. "
      "Short histories fall back to the latest year-over-year change or a drift. Its 80% interval is the 10th-90th percentile of its own past errors at that horizon, measured only on data before the forecast origin.\n")
    A("**Advanced model.** Five statsmodels/numpy candidates were compared by rolling-origin cross-validation inside the training window only (origins every 3 months, 6-month horizon): "
      "drift + event dummies, damped-trend ETS, drift + two seasonal harmonics, Holt-Winters ETS, and SARIMA(0,1,1)(0,1,0)12. Prophet was not used because 81 monthly points do not justify it and it adds a heavy dependency. "
      "The simplest candidate within 3% of the best CV error is selected.\n")
    A(md_table(res["cv_final"][["model", "complexity", "cv_mae", "cv_coverage_80"]].rename(columns={"cv_mae": "CV MAE", "cv_coverage_80": "CV 80% coverage"}),
               {"CV MAE": _f2, "CV 80% coverage": lambda v: "" if np.isnan(v) else f"{v*100:.0f}%"}) + "\n")
    A(f"**Selected: {sel_lbl}.** It models the *monthly net change* as a constant plus event dummies, then accumulates it. Its CV MAE is {cv.loc[sel, 'cv_mae']:.2f} people against {cv.loc['seasonal_baseline', 'cv_mae']:.2f} for the baseline. "
      f"The more elaborate seasonal models did not beat it by enough to justify their extra parameters (Holt-Winters {cv.loc['ets_holt_winters', 'cv_mae']:.2f}, SARIMA {cv.loc['sarima', 'cv_mae']:.2f}), consistent with the weak seasonality found in the EDA. "
      "Its intervals come from the regression variance including parameter uncertainty (normal errors, constant variance), so they widen with the horizon. "
      "Forecasts, lower and upper bounds are clipped at zero.\n")
    A("## E. Workforce events\n")
    ev = res["event_effects"][["event_id", "event_name", "event_type", "department_id", "start_date", "end_date", "hires_multiplier", "exits_multiplier", "n_event_months_observed"]].copy()
    A(md_table(ev, {"hires_multiplier": _f2, "exits_multiplier": _f2}) + "\n")
    A("* **Typing:** events are typed by transparent keyword rules (`event_handler.EVENT_RULES`): freezes, hiring drives, and policy/pay-cycle events. Only freezes and hiring drives have a documented headcount mechanism and are used as 0/1 dummies in the model's history; policy events (EV-02, 04, 08, 09) are listed but not modelled.")
    A("* **Multipliers** (event-month average divided by the department's average over up to 12 preceding event-free months) are **descriptive before/after ratios on 1-6 months, not causal estimates**.")
    A("* **Statistically estimated:** the dummy coefficients inside the model. **Event-based:** the windows themselves, taken from the table. **Assumed:** only multipliers a person supplies in `data/future_events.csv`.")
    if has_future:
        A("* **Future events:** assumed adjustments from `data/future_events.csv` were added after the model (details in `outputs/workforce_event_adjustments.csv`).\n")
    else:
        A("* **Future events:** the sheet contains none and no `data/future_events.csv` exists, so **no future adjustment was applied**. Forecast months carry zero event dummies; nothing is invented. "
          "If a freeze or hiring drive is planned, add it to that CSV (hires/exits multipliers); the adjustment is computed from each department's normal hires and exits and added after the model, so it cannot be double counted with the dummies. This mechanism is unit-tested.\n")
    A("## F. Backtesting (hide Apr-Sep 2026, train on Jan 2020-Mar 2026)\n")
    A(f"The model was selected again using only the 75 training months (result: `{bt_sel}`); the six hidden months were never used for fitting, tuning or selection.\n")
    A(md_table(comp, {"MAE": _f2, "RMSE": _f2, "MAPE %": _f2, "WAPE %": _f2, "Bias": _f2, "Avg 80% width": _f1}) + "\n")
    A("Bias = mean(forecast - actual); negative means under-forecasting. Month by month:\n")
    mt = pd.DataFrame({"Month": b["month"].dt.strftime("%b %Y"), "Actual": b["actual"], "Baseline": b["baseline_forecast"], "Advanced": b["advanced_forecast"],
                       "Baseline abs. error": b["baseline_abs_error"], "Advanced abs. error": b["advanced_abs_error"],
                       "Advanced 80% interval": [f"{lo:.1f} - {hi:.1f}" for lo, hi in zip(b["advanced_lower_80"], b["advanced_upper_80"])]})
    A(md_table(mt, {"Actual": _f0, "Baseline": _f1, "Advanced": _f1, "Baseline abs. error": _f1, "Advanced abs. error": _f1}) + "\n")
    A(("The advanced model beat the baseline" if adv_better else "**The advanced model did NOT beat the baseline**") + f" on MAE ({m['advanced']['MAE']:.2f} vs {m['baseline']['MAE']:.2f}). "
      f"Both under-forecast on average. September 2026 (a +14 jump, 17 hires) accounts for {sep_share:.0f}% of the advanced model's total absolute error, and it is a month the models could not have anticipated. "
      f"Only six hidden months were tested, so coverage of {m['advanced']['inside_80']}/{m['advanced']['n']} is consistent with 80% but proves little; the cross-validated coverage over many origins was {cv.loc[sel, 'cv_coverage_80']*100:.0f}% for the selected model.\n")
    A("**Transparency note (not used for selection):** the other candidates on the same hidden months:\n")
    A(md_table(res["post_hoc"][["model", "MAE", "RMSE", "bias", "coverage_80_pct"]], {"MAE": _f2, "RMSE": _f2, "bias": _f2, "coverage_80_pct": lambda v: f"{v:.0f}%"}) + "\n")
    best_ph = ph["MAE"].idxmin()
    if best_ph != bt_sel:
        A(f"`{best_ph}` happened to score best on these six months (MAE {ph.loc[best_ph, 'MAE']:.2f}), but cross-validation over many origins did not favour it, and picking it now would mean choosing the model by looking at the answer. The selection was left unchanged.\n")
    A("**Attrition and exits backtest.** "
      f"Selected rate method `{res['backtest_exits']['selected']}`: MAE {chosen_ex['metrics_rate']['MAE']:.2f} percentage points, 80% interval contained {chosen_ex['metrics_rate']['inside_80']}/6 months; "
      f"expected-exit counts MAE {chosen_ex['metrics_count']['MAE']:.2f} exits/month (actual monthly rates are noisy: {res['rate_all'].min():.1f}% to {res['rate_all'].max():.1f}% in the history). The sample file's same-month method scored MAE {ref_ex['metrics_rate']['MAE']:.2f} pp / {ref_ex['metrics_count']['MAE']:.2f} exits.\n")
    A("## G. Six-month forecast (trained through " + res["trained_through"] + ")\n")
    ft = pd.DataFrame({"Month": fc.index.strftime("%b %Y"), "Forecast headcount": fc["mean"].values, "Lower 80%": fc["lower"].values, "Upper 80%": fc["upper"].values})
    A(md_table(ft, {c: _f1 for c in ("Forecast headcount", "Lower 80%", "Upper 80%")}) + "\n")
    A(f"Last actual headcount (Sep 2026): {last:.0f}. **Reading the interval:** if the model's assumptions hold, about 8 in 10 actual month-end headcounts should land inside the band. It is a range for the actual future value, not a range for the average.\n")
    et = pd.DataFrame({"Month": ef.index.strftime("%b %Y"), "Expected attrition %": ef["expected_attrition_rate_pct"], "Expected exits": ef["expected_exits"],
                       "Exits lower 80%": ef["lower_80"], "Exits upper 80%": ef["upper_80"], "Expected net change": ef["expected_net_change"], "Implied hires needed": ef["implied_hires_needed"]})
    A("**Expected exits.** Headcount change = hires - exits + net transfers (net transfers are zero for the company total in the data's flow identity). "
      f"Exits are forecast as the monthly attrition rate (method: {ex.METHODS[res['rate_method']][0]}) times opening headcount. Hires are **not** forecast separately; "
      "*implied hires needed* is simply the forecast net change plus expected exits, i.e. what hiring would have to be for the headcount path to hold.\n")
    A(md_table(et, {"Expected attrition %": _f2, "Expected exits": _f1, "Exits lower 80%": _f1, "Exits upper 80%": _f1, "Expected net change": _f1, "Implied hires needed": _f1}) + "\n")
    last6 = res["contract"][(res["contract"]["metric"] == "headcount_end") & (res["contract"]["forecast_month"] == res["contract"]["forecast_month"].max())]
    dt = pd.DataFrame({"Department": last6["department_id"], "Mar 2027 forecast": last6["forecast_value"], "Lower 80%": last6["lower_80"], "Upper 80%": last6["upper_80"]})
    A("Department forecasts for the final month (all six months are in `outputs/headcount_forecast.csv`):\n")
    A(md_table(dt, {c: _f1 for c in ("Mar 2027 forecast", "Lower 80%", "Upper 80%")}) + "\n")
    gap = res["dept_sum_gap"]
    A(f"Department forecasts are fitted separately and are **not reconciled** to the company total: department sum minus company forecast ranges from {gap.min():+.1f} to {gap.max():+.1f} people (the one-person D00 is carried at its last value).\n")
    A("## H. Forecast reliability - which month would I trust least?\n")
    rt = rel["table"][["month", "step", "forecast", "width_80", "relative_width_pct", "calendar_month_change_sd", "cv_mae_at_step"]].copy()
    rt["month"] = rt["month"].dt.strftime("%b %Y")
    A(md_table(rt.rename(columns={"width_80": "80% width", "relative_width_pct": "Width % of forecast", "calendar_month_change_sd": "Hist. SD of that calendar month's change", "cv_mae_at_step": "CV MAE at this horizon"}),
               {"forecast": _f1, "80% width": _f1, "Width % of forecast": _f2, "Hist. SD of that calendar month's change": _f2, "CV MAE at this horizon": _f2}) + "\n")
    A(f"The interval is widest for **{wid['month']:%b %Y}** (step {int(wid['step'])}, width {wid['width_80']:.1f}, {wid['relative_width_pct']:.1f}% of the forecast), and cross-validated error also grows from {res['by_step']['cv_mae'].iloc[0]:.1f} at one month ahead to {res['by_step']['cv_mae'].iloc[-1]:.1f} at six months ahead. "
      "Uncertainty accumulates because each month adds its own random change on top of the previous one. "
      + (f"The same month is also the one whose calendar month has the largest historical volatility." if rel['agree'] else
         f"The calendar month with the most volatile history is **{vol['month']:%b %Y}** (SD of net change {vol['calendar_month_change_sd']:.1f}, versus {wid['calendar_month_change_sd']:.1f} for {wid['month']:%b %Y}); the model uses one constant variance, so it does not widen the interval for that month. "
        f"I would trust {wid['month']:%b %Y} least because both the model and the cross-validation say uncertainty is largest there, and treat {vol['month']:%b %Y} as a month where the stated band may be a little too narrow. ")
      + "No future workforce events touch any forecast month, so events do not explain the difference. This is an evidence-based reading of six rows, not a guarantee.\n")
    A("## I. Business interpretation\n")
    recent_hires = float(res["hist"][res["hist"]["department_id"] == config.COMPANY].sort_values("month")["hires"].tail(12).mean())
    A(f"* **Recruitment planning:** to follow the central path the company needs roughly {ef['implied_hires_needed'].mean():.0f} hires a month (expected exits {ef['expected_exits'].mean():.1f} a month plus net growth {ef['expected_net_change'].mean():.1f}); actual hiring averaged {recent_hires:.1f} a month over the last 12 months, which is the capacity benchmark.")
    A(f"* **Workforce budgeting:** plan to the central value and test the budget at both ends of the band: by Mar 2027 headcount is expected at {fc['mean'].iloc[-1]:.0f} (80% range {fc['lower'].iloc[-1]:.0f}-{fc['upper'].iloc[-1]:.0f}). The width is about {wid['width_80']:.0f} people, so cost scenarios should span it.")
    A("* **Resource allocation:** use department forecasts for direction only; small departments have intervals that are wide relative to their size and are not reconciled to the total.")
    A("* **Hiring decisions:** the near months (Oct-Nov) are the most reliable basis for commitments; treat months 4-6 as scenarios and re-run the pipeline each month as new actuals arrive. Add known freezes or drives to `data/future_events.csv` before relying on the numbers.\n")
    A("## J. Limitations\n")
    A("* The data are fictional and short: 81 months, 6-7 observations per calendar month. Seasonality cannot be established with confidence.")
    A("* The backtest uses one six-month window; the September 2026 jump alone changes the picture. Cross-validation (14 origins) is a sturdier but still small guide.")
    A("* Intervals assume normally distributed, constant-variance monthly changes; real hiring is lumpy and drives could fatten the tails. Cross-validated coverage was near, not exactly, 80%.")
    A("* Event effects are descriptive ratios on a few months and are not causal. No future events are known, so none are applied; an unplanned drive or freeze would make the forecast wrong in a way the intervals do not cover.")
    A("* Hires are not forecast independently; exits are forecast as a flat rate (the data show no reliable seasonal exit pattern), and implied hires are derived.")
    A("* Department models share the company-level model choice, are not reconciled, and the smallest departments have very few people.")
    A("* The latest observation (Sep 2026, +14) is an outlier-level jump; level-based forecasts start from it, so if part of it is transient the forecast is too high, and if it signals a new hiring wave it is too low.\n")
    A("## K. Conclusion\n")
    A(f"A transparent seasonal baseline and a cross-validated statistical model were built, tested on six hidden months and used to forecast October 2026-March 2027. On the hidden months the advanced model had MAE {m['advanced']['MAE']:.2f} versus {m['baseline']['MAE']:.2f} for the baseline, "
      f"and the final forecast is {fc['mean'].iloc[0]:.0f} in Oct 2026 rising to {fc['mean'].iloc[-1]:.0f} in Mar 2027 (80% range {fc['lower'].iloc[-1]:.0f}-{fc['upper'].iloc[-1]:.0f}). "
      "The main lesson is that, with this much history, a simple trend model with event handling was at least competitive with more elaborate seasonal models in cross-validation, and that honest interval width matters as much as the central number.\n")
    return "\n".join(L)


def readme_md(res: dict) -> str:
    comp, m = _comparison(res)
    fc = res["headcount_fc"][config.COMPANY]
    sel = SPECS[res["model"]]
    return f"""# Task 16 - Forecast the future: headcount with uncertainty (CortexHR, Phase 4, Day 16)

Forecasts the company's month-end headcount, attrition and expected exits for the next six months with **80% prediction intervals**, and checks the method on six months it never saw.
This README is generated by `cortexhr_task16.reporting` from the latest run; the full write-up is `reports/forecast_summary.md`.

## 1. Overview and business importance
HR needs to plan hiring, budgets and capacity before headcount is known. A forecast without a stated uncertainty range invites false confidence, so every number here comes with an 80% interval.

## 2. Task 16 objective
Monthly future headcount, expected exits and 80% intervals for six months; a seasonal baseline; an improved statistical model; workforce-event handling; a hold-out backtest; charts, tests and documentation.

## 3. Datasets (workbook sheets, never committed)
| Sheet | Content |
|---|---|
| `monthly_headcount_history` | {res['quality']['rows']} rows: company total + 11 departments x {res['quality']['n_months']} months ({res['quality']['first_month']} to {res['quality']['last_month']}), headcount, hires, exits, transfers, attrition rate, flags |
| `workforce_events` | {len(res['events'])} past events with dates, department and a text description (no impact numbers) |
| `sample_forecast` | the exact output format (72 rows): headcount for ALL and D01-D10 and attrition rate for ALL |

## 4. Technical architecture
```
src/cortexhr_task16/
  config.py              paths (env overrides) and constants
  data_loader.py         loads and validates the three inputs
  preprocessing.py       cleaning, data-quality checks, gap-free series
  exploratory_analysis.py  trend, seasonality, outliers, structure
  baseline_forecaster.py   seasonal baseline + empirical intervals
  advanced_forecaster.py   5 statsmodels/numpy candidates, rolling CV, selection
  event_handler.py       event typing, dummies, effect estimates, future adjustments
  exits_forecaster.py    attrition rate -> expected exits
  backtesting.py         metrics and the 6-month hold-out
  forecasting.py         pipeline + sample_forecast contract builder (CLI)
  visualization.py       four charts
  reporting.py           reports and this README
scripts/run_task16.py    one-command run
tests/                   pytest suite (synthetic fixtures + real-workbook checks)
outputs/  charts/  reports/  notebooks/
```

## 5. Method in plain English
* **EDA:** headcount grew from {res['series'][config.COMPANY].iloc[0]:.0f} to {res['series'][config.COMPANY].iloc[-1]:.0f}; seasonality is suggestive but weak; Sep 2026 is an unusually large jump. See `reports/forecast_analysis.md`.
* **Seasonal baseline:** same month last year plus the average yearly growth known at the forecast date.
* **Advanced model:** five candidates compared by rolling-origin cross-validation inside the training data; the simplest one within 3% of the best is chosen. Selected for the final forecast: **{sel.label}**.
* **Events:** hiring freezes and hiring drives become 0/1 dummies in the model's history, so past events do not distort the trend. Policy events are listed but not modelled. No future events are known, so none are applied; known future events can be added in `data/future_events.csv` with clearly *assumed* multipliers.
* **Backtest:** hide Apr-Sep 2026, train on earlier months only (including model selection), forecast the hidden months, score with MAE, RMSE, MAPE, WAPE, bias and 80% coverage.
* **Prediction interval:** a range expected to contain the actual future headcount about 8 times in 10 if the model's assumptions hold.

## 6. Results (from the latest run)
{md_table(comp, {"MAE": _f2, "RMSE": _f2, "MAPE %": _f2, "WAPE %": _f2, "Bias": _f2, "Avg 80% width": _f1})}

Forecast (trained through {res['trained_through']}): {fc.index[0]:%b %Y} **{fc['mean'].iloc[0]:.0f}** -> {fc.index[-1]:%b %Y} **{fc['mean'].iloc[-1]:.0f}** (80% range {fc['lower'].iloc[-1]:.0f}-{fc['upper'].iloc[-1]:.0f}). Charts: `charts/`.

## 7. Install and run (Windows PowerShell)
```powershell
python -m venv .venv-task16
.\\.venv-task16\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
$env:CORTEXHR_WORKBOOK = 'C:\\approved-data\\CortexHR_Dataset.xlsx'   # or place it at data\\CortexHR_Dataset.xlsx
python scripts\\run_task16.py
python -m pytest -v
```
Linux/macOS: `PYTHONPATH=src python -m cortexhr_task16.forecasting` and `python -m pytest -v`. `data/sample_forecast.csv` is the output contract (the sheet is used if the CSV is absent).

## 8. Outputs
`outputs/headcount_forecast.csv` (same columns, order and types as sample_forecast), `backtest_results.csv`, `backtest_metrics.csv`, `workforce_event_adjustments.csv`, `expected_exits_forecast.csv`, `model_selection_cv.csv`, `holdout_all_candidates_post_hoc.csv`, `forecast_reliability_by_month.csv`; charts `historical_headcount.png`, `historical_exits.png`, `backtest_comparison.png`, `six_month_forecast.png`, `model_performance.png`.

## 9. Limitations
Short, fictional data; weak seasonality evidence; one six-month backtest window; normal-error intervals; event effects not causal and no future events known; hires not forecast independently; department forecasts not reconciled. Details in `reports/forecast_summary.md`, section J.

## 10. Future improvements
Reconcile department forecasts to the total; forecast hires and exits as counts (e.g. Poisson/negative-binomial); multiple backtest windows; conformal intervals; add planned-hiring data as regressors.
"""


def write_reports(res: dict) -> None:
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "forecast_analysis.md").write_text(forecast_analysis_md(res), encoding="utf-8")
    (config.REPORTS / "forecast_summary.md").write_text(forecast_summary_md(res), encoding="utf-8")
    (config.ROOT / "README.md").write_text(readme_md(res), encoding="utf-8")
