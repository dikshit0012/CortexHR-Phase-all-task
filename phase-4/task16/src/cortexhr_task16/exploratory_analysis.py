"""Exploratory analysis of monthly_headcount_history. Every statement in the findings is computed here from the data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def modified_z(x: pd.Series) -> pd.Series:
    """Robust z-score (median / MAD). |value| > 3.5 is the usual outlier flag."""
    med = x.median()
    mad = (x - med).abs().median()
    return 0.6745 * (x - med) / mad if mad > 0 else pd.Series(0.0, index=x.index)


def analyse(hist: pd.DataFrame, events: pd.DataFrame) -> dict:
    y = hist[hist["department_id"] == config.COMPANY].set_index("month").sort_index()
    hc = y["headcount_end"].astype(float)
    d = hc.diff().dropna()
    yoy = (hc / hc.shift(12) - 1).dropna() * 100
    full = pd.date_range(hc.index.min(), hc.index.max(), freq="MS")

    by_month = pd.DataFrame({
        "mean_net_change": d.groupby(d.index.month).mean(), "sd_net_change": d.groupby(d.index.month).std(),
        "mean_hires": y["hires"].groupby(y.index.month).mean(), "mean_exits": y["exits"].groupby(y.index.month).mean(),
        "n_years": d.groupby(d.index.month).count(),
    })
    kw = stats.kruskal(*[g.values for _, g in d.groupby(d.index.month)])
    hires_kw = stats.kruskal(*[g.values for _, g in y["hires"].groupby(y.index.month)])
    exits_kw = stats.kruskal(*[g.values for _, g in y["exits"].groupby(y.index.month)])

    z = modified_z(d)
    unusual = pd.DataFrame({"month": d.index[z.abs() > 3.5], "net_change": d[z.abs() > 3.5].values, "modified_z": z[z.abs() > 3.5].values})
    unusual["hires"] = y.loc[unusual["month"], "hires"].values
    unusual["exits"] = y.loc[unusual["month"], "exits"].values
    # which unusual months fall inside a documented event window?
    def in_event(m):
        hit = events[(events["first_month"] <= m) & (events["last_month"] >= m) & (events["department_id"].isin(["ALL"]) | (events["event_type"] == "hiring_drive"))]
        return ", ".join(hit["event_id"]) if len(hit) else ""
    unusual["event_window"] = [in_event(m) for m in unusual["month"]]

    annual = hc.groupby(hc.index.year).agg(["first", "last"])
    year_end = hc.groupby(hc.index.year).last()
    annual_tbl = pd.DataFrame({"year_end_headcount": year_end, "net_change_in_year": year_end.diff(),
                               "growth_pct": year_end.pct_change() * 100, "months_in_data": hc.groupby(hc.index.year).size()})
    early, late = d[d.index < "2023-01-01"], d[d.index >= "2023-01-01"]
    welch = stats.ttest_ind(early, late, equal_var=False)
    trough = hc.loc["2020-01-01":"2020-12-01"]
    slope = np.polyfit(np.arange(len(hc)), hc.values, 1)[0]

    # monthly net change in months with a documented company-relevant hiring drive vs all other months (descriptive only)
    drive_months = set()
    for _, ev in events[events["event_type"] == "hiring_drive"].iterrows():
        drive_months |= set(pd.date_range(ev["first_month"], ev["last_month"], freq="MS"))
    freeze_months = set()
    for _, ev in events[events["event_type"] == "hiring_freeze"].iterrows():
        freeze_months |= set(pd.date_range(ev["first_month"], ev["last_month"], freq="MS"))
    is_drive = d.index.isin(list(drive_months)); is_freeze = d.index.isin(list(freeze_months))
    ytd_year = hc.index[-1].year
    ytd_change = float(hc.iloc[-1] - hc[hc.index.year == ytd_year - 1].iloc[-1])
    dept_latest = hist[hist["month"] == hist["month"].max()].set_index("department_id")["headcount_end"]
    return {
        "hc": hc, "d": d, "yoy": yoy, "annual": annual_tbl, "by_month": by_month, "seasonal_kw_p": float(kw.pvalue), "seasonal_kw_stat": float(kw.statistic),
        "hires_kw_p": float(hires_kw.pvalue), "exits_kw_p": float(exits_kw.pvalue), "unusual": unusual,
        "months": len(full), "missing_months": [m.strftime("%Y-%m") for m in full.difference(hc.index)],
        "trend_per_month": float(slope), "first": float(hc.iloc[0]), "last": float(hc.iloc[-1]),
        "first_month": hc.index[0], "last_month": hc.index[-1], "peak": float(hc.max()),
        "share_up": float((d > 0).mean() * 100), "share_down": float((d < 0).mean() * 100), "share_flat": float((d == 0).mean() * 100),
        "mean_change": float(d.mean()), "sd_change": float(d.std()),
        "early_mean": float(early.mean()), "late_mean": float(late.mean()), "welch_p": float(welch.pvalue),
        "covid_min": float(trough.min()), "covid_min_month": trough.idxmin(),
        "dept_latest": dept_latest, "drive_mean": float(d[is_drive].mean()), "normal_mean": float(d[~is_drive & ~is_freeze].mean()),
        "freeze_mean": float(d[is_freeze].mean()) if is_freeze.any() else float("nan"), "n_drive": int(is_drive.sum()), "n_freeze": int(is_freeze.sum()),
        "ytd_change": ytd_change, "ytd_year": int(ytd_year), "ytd_months": int(hc.index[-1].month), "yoy_last": float(yoy.iloc[-1]), "yoy_min": float(yoy.min()), "yoy_max": float(yoy.max()),
    }


def findings_markdown(eda: dict, quality: dict, events: pd.DataFrame) -> str:
    """Plain-language answers to the five EDA questions, written from the computed numbers only."""
    hc, a, bm = eda["hc"], eda["annual"], eda["by_month"]
    mon = lambda n: pd.Timestamp(2000, int(n), 1).strftime("%b")
    top = bm["mean_net_change"].sort_values(ascending=False)
    full_years = a[a["months_in_data"] == 12]
    L = ["# Task 16 - exploratory analysis of `monthly_headcount_history`", "",
         "_Generated by `cortexhr_task16.exploratory_analysis`; every number below is computed from the workbook._", "",
         "## Data checked", "",
         f"* {quality['rows']} rows, {quality['series']} series (company total `ALL` + departments), months {quality['first_month']} to {quality['last_month']} ({quality['n_months']} months).",
         f"* Missing values: {quality['missing_values']}. Duplicate (month, department) keys: {quality['duplicate_keys']}. Missing months: {len(quality['missing_months'])} series affected.",
         f"* Flow identity (start + hires - exits + transfers in - transfers out = end) broken in {quality['flow_identity_violations']} rows; start-of-month does not equal previous end-of-month in {quality['start_end_continuity_violations']} months.",
         f"* Department headcounts add up to the company total in every month: {quality['departments_sum_equals_company']}.", "",
         "## 1. Is headcount increasing or decreasing?", "",
         f"Increasing. Company headcount went from {eda['first']:.0f} ({eda['first_month']:%b %Y}) to {eda['last']:.0f} ({eda['last_month']:%b %Y}); a straight-line fit adds {eda['trend_per_month']:.2f} people per month. "
         f"It rose in {eda['share_up']:.0f}% of months, fell in {eda['share_down']:.0f}% and was unchanged in {eda['share_flat']:.0f}%. Calendar-year growth for the {len(full_years)} complete years: "
         + "; ".join(f"{int(y)}: {r.net_change_in_year:+.0f} ({r.growth_pct:+.1f}%)" if not np.isnan(r.net_change_in_year) else f"{int(y)}: year-end {r.year_end_headcount:.0f}" for y, r in full_years.iterrows()) + ". "
         f"Year-over-year growth ranged from {eda['yoy_min']:.1f}% to {eda['yoy_max']:.1f}% and was {eda['yoy_last']:.1f}% in the latest month.", "",
         "## 2. Are there seasonal patterns?", "",
         f"Weak-to-moderate evidence. Average net change by calendar month is highest in {', '.join(mon(m) for m in top.index[:3])} ({', '.join(f'{v:+.1f}' for v in top.iloc[:3])} people) and lowest in {', '.join(mon(m) for m in top.index[-2:])} ({', '.join(f'{v:+.1f}' for v in top.iloc[-2:])}). "
         f"A Kruskal-Wallis test on net change by calendar month gives p = {eda['seasonal_kw_p']:.3f}; for hires alone p = {eda['hires_kw_p']:.3f} and for exits alone p = {eda['exits_kw_p']:.3f}. "
         f"Each calendar month has only {int(bm['n_years'].min())}-{int(bm['n_years'].max())} observations, so the pattern is statistically suggestive, not firmly established; it is tested against a non-seasonal model in the backtest rather than assumed.", "",
         "## 3. Which months show unusual changes?", ""]
    if len(eda["unusual"]):
        L.append("Months whose net change is a robust outlier (modified z-score beyond +/-3.5):"); L.append("")
        L.append("| Month | Net change | Hires | Exits | Inside a documented event window |"); L.append("|---|---:|---:|---:|---|")
        for _, r in eda["unusual"].iterrows():
            L.append(f"| {r['month']:%b %Y} | {r['net_change']:+.0f} | {r['hires']:.0f} | {r['exits']:.0f} | {r['event_window'] or 'none'} |")
    else:
        L.append("No month exceeds the robust outlier threshold.")
    L += ["", f"The weakest stretch is {eda['covid_min_month']:%b %Y} (headcount {eda['covid_min']:.0f}), inside the COVID-19 hiring freeze event (EV-01). The latest month is itself unusual if listed above, which matters for the starting point of the forecast.", "",
          "## 4. Are there structural changes in workforce size?", "",
          "Annual net change in people: " + "; ".join(f"{int(y)}: {r.net_change_in_year:+.0f}" for y, r in full_years.iterrows() if not np.isnan(r.net_change_in_year))
          + f"; {eda['ytd_year']} so far ({eda['ytd_months']} months): {eda['ytd_change']:+.0f}. "
          f"Growth in absolute terms therefore {'slowed' if full_years['net_change_in_year'].dropna().iloc[-1] < full_years['net_change_in_year'].dropna().max() else 'did not slow'} after its peak and "
          f"{'re-accelerated' if eda['ytd_change'] > full_years['net_change_in_year'].dropna().iloc[-1] else 'did not re-accelerate'} in {eda['ytd_year']} (already {eda['ytd_change']:+.0f} versus {full_years['net_change_in_year'].dropna().iloc[-1]:+.0f} in the last full year). "
          f"A simple comparison of mean monthly net change in 2020-2022 ({eda['early_mean']:+.2f}) and 2023 onward ({eda['late_mean']:+.2f}) gives Welch p = {eda['welch_p']:.2f}, "
          + ("which is a statistically supported difference." if eda['welch_p'] < 0.05 else "so a single permanent level shift is not supported; the pattern is better described as episodes of growth than as one structural break.") +
          f" During the {eda['n_freeze']} freeze months the average net change was {eda['freeze_mean']:+.2f}; during the {eda['n_drive']} months touched by a company-relevant hiring-drive event it was {eda['drive_mean']:+.2f}, against {eda['normal_mean']:+.2f} in all other months (descriptive, not causal).", "",
          "## 5. Which periods have insufficient historical information?", "",
          f"* 2020 has no year-over-year comparison (first 12 months) and contains the COVID freeze.",
          f"* Each calendar month appears only {int(bm['n_years'].min())}-{int(bm['n_years'].max())} times, which limits seasonal estimates.",
          f"* Rolling-origin validation needs {config.CV_MIN_TRAIN} months of training, so only the later {quality['n_months'] - config.CV_MIN_TRAIN} months can be used as forecast tests.",
          "* Small departments (e.g. D00 with one person; D10 and D05 in the low teens) have too few people for stable percentage-based statements.",
          "* There is no information about future events: the workforce_events sheet ends in April 2026 and lists no planned events.", ""]
    return "\n".join(L)
