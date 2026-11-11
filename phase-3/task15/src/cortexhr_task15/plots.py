"""Charts drawn from computed results only."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def salary_distribution(y: pd.Series, bands: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].hist(y / 1e5, bins=50, color="#1f77b4"); ax[0].set_xlabel("Offered annual fixed pay (lakh INR)"); ax[0].set_ylabel("hires"); ax[0].set_title(f"Training table: salary distribution (n={len(y):,})")
    ax[1].hist(np.log10(y), bins=50, color="#2ca02c"); ax[1].set_xlabel("log10(offered annual fixed pay)"); ax[1].set_title("Log scale: much less skewed")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def model_comparison(table: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(table["Model"], table["MAE"] / 1e3, color=["#999", "#1f77b4", "#d62728"]); ax[0].set_ylabel("MAE (thousand INR) - lower is better"); ax[0].set_title("Held-out test MAE")
    ax[1].bar(table["Model"], table["R2"], color=["#999", "#1f77b4", "#d62728"]); ax[1].set_ylim(min(0.9, table["R2"].min() - 0.02), 1.0); ax[1].set_ylabel("R2 - higher is better"); ax[1].set_title("Held-out test R2")
    for a in ax:
        a.tick_params(axis="x", labelrotation=15)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def fairness_gaps(tables: dict, path) -> None:
    """tables: {title: DataFrame from group_table for the selected model}. Bars = mean signed error (predicted - actual) as % of actual, with n."""
    fig, axes = plt.subplots(1, len(tables), figsize=(max(8.5, 6 * len(tables)), 4.8), squeeze=False)
    for ax, (title, t) in zip(axes[0], tables.items()):
        cols = ["#2ca02c" if r else "#bbbbbb" for r in t["reliable_n"]]
        ax.bar([f"{g}\n(n={n})" for g, n in zip(t["group"], t["n"])], t["mean_signed_error_pct_of_actual"], color=cols)
        ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("mean (predicted - actual) as % of actual"); ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Prediction error by group (grey = fewer than 30 people: unreliable).\nPositive = the model predicts more than the historical pay.", fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig(path, dpi=140); plt.close()
