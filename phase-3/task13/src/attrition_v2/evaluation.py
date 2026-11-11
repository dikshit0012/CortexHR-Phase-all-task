"""Phase 7 + 9: same-fold comparison, paired comparison against the baseline, official-benchmark comparison, selection rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from cortexhr.baselines import bootstrap_auc_interval, load_benchmarks, per_fold_scores
from cortexhr.evaluation import binary_report

METRICS = ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]


def compare_models(y, oof: dict[str, np.ndarray], folds, kinds: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per model. Per-fold mean and sd of every metric (threshold 0.5 for yes/no metrics, as in Task 3), the pooled
    out-of-fold ROC-AUC with a bootstrap interval, and the share-of-leavers-caught in the top 10% list (Task 4's view)."""
    rows = []
    for name, p in oof.items():
        pf = per_fold_scores(y, p, folds)
        lo, hi = bootstrap_auc_interval(y, p) if len(np.unique(p)) > 1 else (0.5, 0.5)
        top = binary_report(y, p, top_frac=0.10, tie_seed=0)
        row = {"model": name, "kind": (kinds or {}).get(name, "")}
        for m in METRICS:
            row[f"{m}_mean"] = pf[m].mean()
            row[f"{m}_std"] = pf[m].std(ddof=1)
        row.update({"roc_auc_pooled": roc_auc_score(y, p), "roc_auc_pooled_ci_low": lo, "roc_auc_pooled_ci_high": hi,
                    "precision_at_top10pct": top["precision_at_top"], "recall_at_top10pct": top["recall_at_top"],
                    "lift_at_top10pct": top["lift_at_top"]})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(y, p_new, p_ref, folds, n_boot: int = 2000, seed: int = 0) -> dict:
    """Is ``p_new`` really better than ``p_ref``? Both scored on the SAME folds and the SAME rows.

    * per-fold ROC-AUC difference (mean, sd, how many folds the new model wins)
    * paired bootstrap of the pooled ROC-AUC difference (resample rows once, score both models on that resample)
    * paired t-test across the 5 folds (only 4 degrees of freedom, so a rough guide, not proof)
    """
    y = np.asarray(y)
    pn, pr = np.asarray(p_new), np.asarray(p_ref)
    d_fold = np.array([roc_auc_score(y[te], pn[te]) - roc_auc_score(y[te], pr[te]) for _, te in folds])
    d_pr = np.array([_ap(y[te], pn[te]) - _ap(y[te], pr[te]) for _, te in folds])
    rng = np.random.default_rng(seed)
    diffs, pr_diffs = [], []
    while len(diffs) < n_boot:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], pn[i]) - roc_auc_score(y[i], pr[i]))
        pr_diffs.append(_ap(y[i], pn[i]) - _ap(y[i], pr[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pr_lo, pr_hi = np.percentile(pr_diffs, [2.5, 97.5])
    t = stats.ttest_rel([roc_auc_score(y[te], pn[te]) for _, te in folds], [roc_auc_score(y[te], pr[te]) for _, te in folds])
    return {"roc_auc_diff_per_fold": d_fold.round(4).tolist(), "roc_auc_diff_mean": float(d_fold.mean()),
            "roc_auc_diff_sd": float(d_fold.std(ddof=1)), "folds_won": int((d_fold > 0).sum()), "n_folds": len(folds),
            "pr_auc_diff_mean": float(d_pr.mean()), "pooled_pr_auc_diff": float(_ap(y, pn) - _ap(y, pr)),
            "pooled_pr_auc_diff_ci95": [float(pr_lo), float(pr_hi)],
            "pooled_roc_auc_diff": float(roc_auc_score(y, pn) - roc_auc_score(y, pr)),
            "pooled_diff_ci95": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "paired_t_p_value_5_folds": float(t.pvalue)}


def _ap(y, p):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, p)


def per_source_auc(y, source, oof: dict[str, np.ndarray]) -> pd.DataFrame:
    """ROC-AUC inside each data source. HIST rows have no monthly data, so a model can partly 'detect the source'; the
    within-source AUC shows whether the ranking still works when that shortcut is removed."""
    rows = []
    y = np.asarray(y)
    src = np.asarray(source)
    for name, p in oof.items():
        p = np.asarray(p)
        row = {"model": name}
        for s in np.unique(src):
            m = src == s
            row[f"auc_{s}"] = roc_auc_score(y[m], p[m]) if len(np.unique(y[m])) > 1 and len(np.unique(p[m])) > 1 else 0.5
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_comparison(ours: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Official ``baseline_benchmarks`` rows next to our reproduced / new rows. ``mapping``: our model name -> benchmark name.

    Official numbers are copied from the workbook sheet, never typed in. Benchmark rows with no counterpart are listed too.
    """
    bench = load_benchmarks()
    bench = bench.set_index("model")
    rows = []
    for _, b in bench.iterrows():
        rows.append({"row_type": "official_benchmark", "model": b.name, "roc_auc_mean": b["roc_auc_mean"],
                     "roc_auc_std": b["roc_auc_sd"], "pr_auc_mean": b["pr_auc_mean"], "precision_mean": b["precision_mean"],
                     "recall_mean": b["recall_mean"], "f1_mean": b["f1_mean"], "accuracy_mean": b["accuracy_mean"],
                     "comment": b["comment"]})
    o = ours.set_index("model")
    for our_name, bench_name in mapping.items():
        r = o.loc[our_name]
        rows.append({"row_type": "reproduced_baseline", "model": our_name, "roc_auc_mean": r["roc_auc_mean"],
                     "roc_auc_std": r["roc_auc_std"], "pr_auc_mean": r["pr_auc_mean"], "precision_mean": r["precision_mean"],
                     "recall_mean": r["recall_mean"], "f1_mean": r["f1_mean"], "accuracy_mean": r["accuracy_mean"],
                     "comment": f"compare with official row: {bench_name}",
                     "diff_roc_auc_vs_official": r["roc_auc_mean"] - bench.loc[bench_name, "roc_auc_mean"],
                     "diff_pr_auc_vs_official": r["pr_auc_mean"] - bench.loc[bench_name, "pr_auc_mean"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Phase 9: the rule that decides whether the complex model ships (written down BEFORE the final numbers are read)
# ----------------------------------------------------------------------------------------------
MIN_MEAN_FOLD_GAIN = 0.01     # a 1-point ROC-AUC gain is the smallest we call meaningful for a ranking model
MIN_FOLDS_WON = 4             # and it must hold in at least 4 of the 5 folds
CALIBRATION_TOLERANCE = 0.01  # the shipped model's ECE may not be worse than the baseline's by more than this


def selection_decision(paired: dict, ece_candidate: float, ece_baseline: float) -> dict:
    """Ship the gradient-boosting model only if ALL of these hold against the simplest strong baseline:
    mean per-fold ROC-AUC gain >= 0.01, paired-bootstrap 95% interval excludes 0, wins >= 4 of 5 folds,
    and its calibration is not clearly worse. Otherwise keep the simpler model and say so."""
    checks = {
        "mean_fold_gain_at_least_0.01": paired["roc_auc_diff_mean"] >= MIN_MEAN_FOLD_GAIN,
        "bootstrap_ci_excludes_zero_in_favour": paired["pooled_diff_ci95"][0] > 0,
        "wins_at_least_4_of_5_folds": paired["folds_won"] >= MIN_FOLDS_WON,
        "calibration_not_worse": ece_candidate <= ece_baseline + CALIBRATION_TOLERANCE,
    }
    return {"ship_gradient_boosting": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()},
            "rule": ("Ship the more complex model only if the mean per-fold ROC-AUC gain >= 0.01, the paired bootstrap "
                     "95% interval excludes 0, it wins >= 4 of 5 folds, and its calibration is not clearly worse.")}
