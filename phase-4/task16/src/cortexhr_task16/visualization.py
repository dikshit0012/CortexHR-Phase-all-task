"""The four Task 16 charts (PNG, 200 dpi). All data comes from the pipeline result dictionary."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import backtesting as bt
from . import config, exploratory_analysis as ea
from .advanced_forecaster import SPECS

BLUE, ORANGE, GREEN, RED, GREY = "#1f4e79", "#d9822b", "#2e8b57", "#c0392b", "#7f8c8d"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def _save(fig, name: str) -> Path:
    config.CHARTS.mkdir(parents=True, exist_ok=True)
    path = config.CHARTS / name
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _dates(ax, years=True):
    ax.xaxis.set_major_locator(mdates.YearLocator() if years else mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y" if years else "%b %Y"))


def historical_headcount(res: dict) -> Path:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    y = res["series"][config.COMPANY]
    ev = res["events"]
    eda = ea.analyse(res["hist"], ev)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    ax.plot(y.index, y.values, color=BLUE, lw=2, label="Monthly headcount (end of month)")
    sm = lowess(y.values, np.arange(len(y)), frac=0.3, return_sorted=False)
    ax.plot(y.index, sm, color=ORANGE, lw=2, ls="--", label="Trend (LOWESS smoother)")
    seen = set()
    for _, e in ev.iterrows():
        if e["event_type"] in ("hiring_freeze", "hiring_drive"):
            col = RED if e["event_type"] == "hiring_freeze" else GREEN
            lab = "Hiring freeze" if e["event_type"] == "hiring_freeze" else "Hiring drive (any department)"
            ax.axvspan(e["first_month"], e["last_month"] + pd.offsets.MonthEnd(0), color=col, alpha=0.15, label=None if lab in seen else lab)
            seen.add(lab)
            ax.text(e["last_month"] + pd.Timedelta(days=34), y.min() - 12, e["event_id"], fontsize=7, color=col, rotation=90, va="bottom")
    for _, r in eda["unusual"].iterrows():
        ax.scatter([r["month"]], [y.loc[r["month"]]], s=70, facecolor="none", edgecolor=RED, lw=2, zorder=5,
                   label="Unusual monthly change" if "u" not in seen else None)
        seen.add("u")
        ax.annotate(f"{r['month']:%b %Y}\n{r['net_change']:+.0f}", (r["month"], y.loc[r["month"]]), xytext=((-70, -38) if r["month"] == y.index[-1] else (-48, 18)), textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=GREY))
    ax.set_ylabel("Employees"); ax.set_title("Company headcount, January 2020 - September 2026", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    d = eda["d"]
    colors = [RED if m in set(eda["unusual"]["month"]) else (BLUE if v >= 0 else GREY) for m, v in d.items()]
    ax2.bar(d.index, d.values, width=24, color=colors)
    ax2.axhline(0, color="black", lw=0.8); ax2.set_ylabel("Net change\nper month")
    _dates(ax2)
    return _save(fig, "historical_headcount.png")


def historical_exits(res: dict) -> Path:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    exits = res["exits_all"]
    rate = res["rate_all"]
    ev = res["events"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.5]})

    ax1.plot(exits.index, exits.values, color=RED, lw=1.8, marker="o", ms=3, label="Monthly Employee Exits (Count)")
    sm = lowess(exits.values, np.arange(len(exits)), frac=0.25, return_sorted=False)
    ax1.plot(exits.index, sm, color=ORANGE, lw=2, ls="--", label="Exit Trend (LOWESS)")

    seen = set()
    for _, e in ev.iterrows():
        if e["event_type"] in ("hiring_freeze", "hiring_drive"):
            col = GREY if e["event_type"] == "hiring_freeze" else GREEN
            lab = "Hiring freeze" if e["event_type"] == "hiring_freeze" else "Hiring drive"
            ax1.axvspan(e["first_month"], e["last_month"] + pd.offsets.MonthEnd(0), color=col, alpha=0.12, label=None if lab in seen else lab)
            seen.add(lab)

    ax1.set_ylabel("Exit Count (Employees)")
    ax1.set_title("Historical Monthly Employee Exits & Attrition Rate (2020 - 2026)", fontweight="bold")
    ax1.legend(loc="upper left", frameon=False)

    ax2.bar(rate.index, rate.values, width=24, color=BLUE, alpha=0.7, label="Monthly Attrition Rate (%)")
    rate_sm = lowess(rate.values, np.arange(len(rate)), frac=0.25, return_sorted=False)
    ax2.plot(rate.index, rate_sm, color=ORANGE, lw=2, ls="--", label="Attrition Rate Trend (LOWESS)")
    ax2.axhline(rate.mean(), color=RED, ls=":", lw=1.2, label=f"Historical Mean ({rate.mean():.2f}%)")
    ax2.set_ylabel("Attrition Rate (%)")
    ax2.legend(loc="upper left", frameon=False)
    _dates(ax2)

    return _save(fig, "historical_exits.png")


def backtest_comparison(res: dict) -> Path:
    b = res["backtest"][config.COMPANY]["results"]
    y = res["series"][config.COMPANY]
    train_tail = y.loc[: b["month"].iloc[0] - pd.offsets.MonthBegin(1)].iloc[-18:]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(train_tail.index, train_tail.values, color=GREY, lw=2, label="Training data (visible to models)")
    ax.plot(b["month"], b["actual"], color="black", lw=2.5, marker="o", label="Actual (hidden from models)")
    ax.fill_between(b["month"], b["baseline_lower_80"], b["baseline_upper_80"], color=ORANGE, alpha=0.15, label="Baseline 80% interval")
    ax.plot(b["month"], b["baseline_forecast"], color=ORANGE, lw=2, ls="--", marker="s", label="Seasonal baseline")
    ax.fill_between(b["month"], b["advanced_lower_80"], b["advanced_upper_80"], color=BLUE, alpha=0.2, label="Advanced 80% interval")
    ax.plot(b["month"], b["advanced_forecast"], color=BLUE, lw=2, marker="^", label=f"Advanced: {SPECS[res['backtest_model']].label}")
    ax.axvline(b["month"].iloc[0] - pd.Timedelta(days=15), color=GREY, ls=":")
    ax.text(b["month"].iloc[0] - pd.Timedelta(days=12), ax.get_ylim()[0] + 2, "hold-out starts", fontsize=8, color=GREY)
    ax.set_title("Six-month hold-out backtest (April - September 2026)", fontweight="bold"); ax.set_ylabel("Employees")
    ax.legend(loc="upper left", frameon=False, fontsize=8); _dates(ax, years=False)
    return _save(fig, "backtest_comparison.png")


def six_month_forecast(res: dict) -> Path:
    y = res["series"][config.COMPANY]
    fc = res["headcount_fc"][config.COMPANY]
    tail = y.iloc[-30:]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(tail.index, tail.values, color=BLUE, lw=2, marker="o", ms=3, label="Actual headcount")
    ax.fill_between(fc.index, fc["lower"], fc["upper"], color=ORANGE, alpha=0.3, label="80% prediction interval")
    ax.plot(fc.index, fc["lower"], color=ORANGE, lw=1); ax.plot(fc.index, fc["upper"], color=ORANGE, lw=1)
    ax.plot(np.r_[tail.index[-1], fc.index], np.r_[tail.values[-1], fc["mean"].values], color=ORANGE, lw=2.5, marker="o", label="Forecast headcount")
    ax.axvline(tail.index[-1] + pd.Timedelta(days=15), color=GREY, ls=":")
    ax.text(tail.index[-1] + pd.Timedelta(days=18), tail.min(), "forecast starts", fontsize=8, color=GREY)
    for m, r in fc.iterrows():
        ax.annotate(f"{r['mean']:.0f}", (m, r["mean"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_title(f"Six-month headcount forecast with 80% prediction interval ({SPECS[res['model']].label})", fontweight="bold", fontsize=11)
    ax.set_ylabel("Employees"); ax.legend(loc="upper left", frameon=False); _dates(ax, years=False)
    return _save(fig, "six_month_forecast.png")


def model_performance(res: dict) -> Path:
    b = res["backtest"][config.COMPANY]["results"]
    m = {k: bt.metrics(b["actual"], b[f"{k}_forecast"], b[f"{k}_lower_80"], b[f"{k}_upper_80"]) for k in ("baseline", "advanced")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    names, x, w = ["MAE", "RMSE"], np.arange(2), 0.35
    a1.bar(x - w / 2, [m["baseline"][n] for n in names], w, color=ORANGE, label="Seasonal baseline")
    a1.bar(x + w / 2, [m["advanced"][n] for n in names], w, color=BLUE, label="Advanced model")
    for i, n in enumerate(names):
        a1.text(i - w / 2, m["baseline"][n], f"{m['baseline'][n]:.1f}", ha="center", va="bottom", fontsize=9)
        a1.text(i + w / 2, m["advanced"][n], f"{m['advanced'][n]:.1f}", ha="center", va="bottom", fontsize=9)
    a1.set_xticks(x, [f"{n} (employees)" for n in names]); a1.set_title("Error size on the 6 hidden months", fontweight="bold"); a1.legend(frameon=False)
    mx = np.arange(len(b))
    a2.bar(mx - w / 2, b["baseline_abs_error"], w, color=ORANGE, label="Seasonal baseline")
    a2.bar(mx + w / 2, b["advanced_abs_error"], w, color=BLUE, label="Advanced model")
    a2.set_xticks(mx, [f"{d:%b %Y}" for d in b["month"]], rotation=20); a2.set_ylabel("Absolute error (employees)")
    a2.set_title("Absolute error by month", fontweight="bold"); a2.legend(frameon=False)
    fig.suptitle(f"Forecast performance - 80% interval coverage: baseline {m['baseline']['coverage_80_pct']:.0f}% (avg width {m['baseline']['mean_width_80']:.0f}), "
                 f"advanced {m['advanced']['coverage_80_pct']:.0f}% (avg width {m['advanced']['mean_width_80']:.0f})", fontsize=10)
    return _save(fig, "model_performance.png")


def make_all_charts(res: dict) -> list[Path]:
    return [historical_headcount(res), historical_exits(res), backtest_comparison(res), six_month_forecast(res), model_performance(res)]
