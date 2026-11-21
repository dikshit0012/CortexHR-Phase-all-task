"""End-to-end Task 16 pipeline: load -> validate -> EDA -> backtest -> final forecast -> contract CSV, charts and reports.

Run:  python scripts/run_task16.py        (or)   PYTHONPATH=src python -m cortexhr_task16.forecasting
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import advanced_forecaster as af
from . import backtesting as bt
from . import baseline_forecaster as bf
from . import config, data_loader, event_handler as eh, exits_forecaster as ex, preprocessing as pp

log = logging.getLogger("cortexhr_task16")


# ------------------------------------------------------------------ output contract
def validate_contract(out: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    """Problems found when comparing a forecast file with the sample_forecast contract (empty list = compatible)."""
    problems = []
    if list(out.columns) != list(sample.columns):
        problems.append(f"columns differ: {list(out.columns)} vs {list(sample.columns)}")
        return problems
    if len(out) != len(sample):
        problems.append(f"row count {len(out)} != {len(sample)}")
    for c in sample.columns:
        if str(out[c].dtype) != str(sample[c].dtype) and not (pd.api.types.is_string_dtype(out[c]) and pd.api.types.is_string_dtype(sample[c])):
            problems.append(f"dtype of {c}: {out[c].dtype} vs {sample[c].dtype}")
    keys = ["metric", "department_id"]
    if [tuple(x) for x in out[keys].drop_duplicates().values] != [tuple(x) for x in sample[keys].drop_duplicates().values]:
        problems.append("metric/department blocks differ in content or order")
    if not out["forecast_month"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        problems.append("forecast_month is not YYYY-MM-DD")
    if not (out["lower_80"] <= out["forecast_value"]).all() or not (out["forecast_value"] <= out["upper_80"]).all():
        problems.append("lower_80 <= forecast_value <= upper_80 violated")
    if (out[["forecast_value", "lower_80", "upper_80"]] < 0).any().any():
        problems.append("negative values")
    return problems


def build_contract(sample: pd.DataFrame, headcount: dict[str, pd.DataFrame], rate: pd.DataFrame, trained_through: str,
                   headcount_method: dict[str, str], rate_method: str) -> pd.DataFrame:
    rows = []
    for metric, dept in sample[["metric", "department_id"]].drop_duplicates().itertuples(index=False):
        fc = rate if metric == "attrition_rate_monthly_pct" else headcount[dept]
        method = rate_method if metric == "attrition_rate_monthly_pct" else headcount_method[dept]
        dec = 2 if metric == "attrition_rate_monthly_pct" else 1
        for month, r in fc.iterrows():
            v, lo, up = (round(float(r[c]), dec) for c in ("mean", "lower", "upper"))
            rows.append({"metric": metric, "department_id": dept, "forecast_month": month.strftime("%Y-%m-%d"),
                         "forecast_value": v, "lower_80": min(lo, v), "upper_80": max(up, v), "method": method,
                         "trained_through": trained_through, "model_version": config.MODEL_VERSION})
    out = pd.DataFrame(rows)
    out.insert(0, "forecast_id", [f"FC-{i:04d}" for i in range(1, len(out) + 1)])
    return out[list(sample.columns)]


# ------------------------------------------------------------------ analysis helpers
def cv_error_by_step(y: pd.Series, cal: pd.DataFrame, model: str, h: int = config.HORIZON) -> pd.DataFrame:
    """Mean absolute CV error and coverage at each forecast step (1..h), for the selected model."""
    errs = {k: [] for k in range(1, h + 1)}
    cov = {k: [] for k in range(1, h + 1)}
    for o in range(config.CV_MIN_TRAIN, len(y) - h + 1, config.CV_STEP):
        fc = af.run_candidate(af.SPECS[model], y.iloc[:o], cal.iloc[:o], h)
        a = y.iloc[o:o + h].values
        for k in range(h):
            errs[k + 1].append(abs(a[k] - fc["mean"].values[k]))
            cov[k + 1].append(fc["lower"].values[k] <= a[k] <= fc["upper"].values[k])
    return pd.DataFrame({"step": list(errs), "cv_mae": [np.mean(v) for v in errs.values()], "cv_coverage_80": [np.mean(v) for v in cov.values()]})


def reliability_assessment(y: pd.Series, fc: pd.DataFrame, by_step: pd.DataFrame, events_future: pd.DataFrame) -> dict:
    """Which forecast month deserves least trust? Evidence: interval width, historical volatility of that calendar month,
    whether a future event touches it, and cross-validated error at that horizon. Nothing is assumed in advance."""
    d = y.diff().dropna()
    vol = d.groupby(d.index.month).std()
    width = (fc["upper"] - fc["lower"]).values
    table = pd.DataFrame({
        "month": fc.index, "step": range(1, len(fc) + 1), "forecast": fc["mean"].values, "width_80": width,
        "relative_width_pct": width / fc["mean"].values * 100,
        "calendar_month_change_sd": [float(vol[m.month]) for m in fc.index],
        "cv_mae_at_step": by_step["cv_mae"].values,
        "future_event": [bool(len(events_future) and ((pd.to_datetime(events_future["start_date"]).dt.to_period("M").dt.to_timestamp() <= m)
                                                        & (pd.to_datetime(events_future["end_date"]).dt.to_period("M").dt.to_timestamp() >= m)).any()) for m in fc.index],
    })
    widest = table.loc[table["width_80"].idxmax()]
    most_volatile = table.loc[table["calendar_month_change_sd"].idxmax()]
    return {"table": table, "widest": widest, "most_volatile": most_volatile,
            "agree": bool(widest["month"] == most_volatile["month"])}


# ------------------------------------------------------------------ pipeline
def run(workbook: Path | None = None, write: bool = True, horizon: int = config.HORIZON) -> dict:
    """Execute the full pipeline and return every intermediate result (used by reports, charts and tests)."""
    raw = data_loader.load_headcount_history(workbook)
    events_raw = data_loader.load_workforce_events(workbook)
    sample = data_loader.load_sample_forecast(workbook=workbook)
    hist = pp.clean_history(raw)
    quality = pp.validate_history(hist)
    events = eh.prepare_events(events_raw)
    event_effects = eh.estimate_event_effects(hist, events)
    future_events = eh.load_future_events()

    contract_depts = list(dict.fromkeys(sample.loc[sample["metric"] == "headcount_end", "department_id"]))
    series, cal = {}, {}
    for dept in contract_depts:
        series[dept], _ = pp.get_series(hist, dept)
        cal[dept] = eh.build_calendar(events, series[dept].index, dept)
    y_all = series[config.COMPANY]
    last = y_all.index[-1]
    trained_through = last.strftime("%Y-%m")

    # ---------------- backtest (hide the last `horizon` months)
    log.info("Backtest: company total")
    bt_all = bt.backtest_series(y_all, cal[config.COMPANY], None, horizon)
    bt_model = bt_all["model"]
    post_hoc = bt.holdout_candidates(y_all, cal[config.COMPANY], horizon)
    bt_dept = {config.COMPANY: bt_all}
    for dept in contract_depts:
        if dept != config.COMPANY:
            bt_dept[dept] = bt.backtest_series(series[dept], cal[dept], bt_model, horizon)
    rate_all, _ = pp.get_series(hist, config.COMPANY, "attrition_rate_monthly_pct")
    exits_all, _ = pp.get_series(hist, config.COMPANY, "exits")
    bt_exits = bt.backtest_exits(rate_all, exits_all, y_all, bt_all["results"]["advanced_forecast"].values, horizon)

    # ---------------- final fit on all history
    log.info("Final model selection on all %d months", len(y_all))
    cv_final = af.cross_validate(y_all, cal[config.COMPANY], horizon)
    model = af.select_model(cv_final)
    months = pp.future_index(last, horizon)
    adj = eh.future_adjustments(hist, events, future_events, months, contract_depts)
    hc_fc, hc_method = {}, {}
    for dept in contract_depts:
        try:
            fc = af.fit_forecast(series[dept], cal[dept], model, horizon)
            used = model
        except Exception as exc:  # pragma: no cover - defensive; reported, never silent
            log.warning("%s: %s failed (%s); falling back to drift_events", dept, model, exc)
            used = "drift_events"
            fc = af.fit_forecast(series[dept], cal[dept], used, horizon)
        fc = eh.apply_adjustments(fc, adj, dept)
        hc_fc[dept] = fc
        hc_method[dept] = f"{af.SPECS[used].label} (CV-selected)" + (" + assumed event adjustment" if adj is not None and len(adj) else "")
    by_step = cv_error_by_step(y_all, cal[config.COMPANY], model, horizon)
    reliability = reliability_assessment(y_all, hc_fc[config.COMPANY], by_step, future_events)

    cv_rate = ex.cv_methods(rate_all, horizon)
    rate_method = ex.select_method(cv_rate)
    rate_fc = ex.forecast_rate(rate_all, rate_method, horizon)
    exits_fc = ex.expected_exits(rate_fc, hc_fc[config.COMPANY], float(y_all.iloc[-1]))
    contract = build_contract(sample, hc_fc, rate_fc, trained_through, hc_method, ex.METHODS[rate_method][0])
    problems = validate_contract(contract, sample)
    if problems:
        raise RuntimeError(f"Forecast does not match the sample_forecast contract: {problems}")

    dept_sum = sum(hc_fc[d]["mean"] for d in contract_depts if d != config.COMPANY)
    for d in sorted(set(hist["department_id"]) - set(contract_depts) - {config.COMPANY}):   # departments outside the contract (D00): carry last value
        dept_sum = dept_sum + float(pp.get_series(hist, d)[0].iloc[-1])
    res = dict(hist=hist, quality=quality, events=events, event_effects=event_effects, future_events=future_events, adjustments=adj,
               sample=sample, contract=contract, series=series, calendars=cal, backtest=bt_dept, backtest_model=bt_model,
               post_hoc=post_hoc, backtest_exits=bt_exits, cv_final=cv_final, model=model, headcount_fc=hc_fc, by_step=by_step,
               reliability=reliability, rate_all=rate_all, exits_all=exits_all, rate_fc=rate_fc, rate_method=rate_method, cv_rate=cv_rate,
               exits_fc=exits_fc, dept_sum_gap=(dept_sum - hc_fc[config.COMPANY]["mean"]), trained_through=trained_through,
               continuity=af.continuity_check(y_all, hc_fc[config.COMPANY]), horizon=horizon, contract_depts=contract_depts)
    if write:
        write_outputs(res)
    return res


def backtest_tables(res: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """backtest_results.csv (month-wise, every series) and backtest_metrics.csv (one row per model and metric)."""
    results, mets = [], []
    for dept, b in res["backtest"].items():
        r = b["results"].copy(); r.insert(0, "department_id", dept); results.append(r)
        for name, p in (("Seasonal Baseline", "baseline"), ("Advanced Model", "advanced")):
            m = bt.metrics(r["actual"], r[f"{p}_forecast"], r[f"{p}_lower_80"], r[f"{p}_upper_80"])
            mets.append({"metric": "headcount_end", "department_id": dept, "model": name,
                         "method": "seasonal naive + annual trend" if p == "baseline" else af.SPECS[b["model"]].label, **m})
    for row in res["backtest_exits"]["rows"]:
        tag = "selected" if row["selected"] else "reference (sample method)"
        mets.append({"metric": "attrition_rate_monthly_pct", "department_id": config.COMPANY, "model": f"Rate: {tag}",
                     "method": ex.METHODS[row["method"]][0], **row["metrics_rate"]})
        mets.append({"metric": "exits_count", "department_id": config.COMPANY, "model": f"Exits: {tag}",
                     "method": ex.METHODS[row["method"]][0], **row["metrics_count"]})
    return pd.concat(results, ignore_index=True), pd.DataFrame(mets)


def write_outputs(res: dict) -> None:
    from . import reporting, visualization
    for d in (config.OUTPUTS, config.CHARTS, config.REPORTS):
        Path(d).mkdir(parents=True, exist_ok=True)
    res["contract"].to_csv(config.OUTPUTS / "headcount_forecast.csv", index=False)
    results, mets = backtest_tables(res)
    results.to_csv(config.OUTPUTS / "backtest_results.csv", index=False)
    mets.round(4).to_csv(config.OUTPUTS / "backtest_metrics.csv", index=False)
    res["cv_final"].round(4).to_csv(config.OUTPUTS / "model_selection_cv.csv", index=False)
    res["post_hoc"].round(4).to_csv(config.OUTPUTS / "holdout_all_candidates_post_hoc.csv", index=False)
    res["exits_fc"].round(3).rename_axis("month").to_csv(config.OUTPUTS / "expected_exits_forecast.csv")
    reporting.event_adjustment_table(res).to_csv(config.OUTPUTS / "workforce_event_adjustments.csv", index=False)
    res["reliability"]["table"].round({"forecast": 3, "width_80": 3, "relative_width_pct": 3, "calendar_month_change_sd": 3, "cv_mae_at_step": 3}).to_csv(config.OUTPUTS / "forecast_reliability_by_month.csv", index=False)
    visualization.make_all_charts(res)
    reporting.write_reports(res)
    log.info("Outputs written to %s", config.OUTPUTS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CortexHR Task 16: headcount forecast with 80% prediction intervals")
    ap.add_argument("--workbook", type=Path, default=None, help="path to the CortexHR master workbook (default: CORTEXHR_WORKBOOK or data/)")
    ap.add_argument("--no-write", action="store_true", help="compute only; do not write files")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        res = run(args.workbook, write=not args.no_write)
    except data_loader.DataError as exc:
        log.error("%s", exc)
        return 2
    print(f"Model: {res['model']} | trained through {res['trained_through']} | rows: {len(res['contract'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
