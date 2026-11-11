"""Phase 6: limited, controlled tuning on the shared Task 3 folds + an honest (nested) estimate of the tuned model.

Two different questions, kept apart on purpose:

1. *Which configuration?*  Every configuration is scored on the 5 shared folds (``run_tuning``) and one is picked with the
   one-standard-error rule: the simplest model whose mean ROC-AUC is within one standard error of the best. Picking the
   top mean from a handful of noisy fold scores would reward luck.
2. *How good is "tune then pick"?*  The score in (1) is optimistic because the same folds chose the winner. So the tuned
   model is ALSO evaluated with nested cross-validation (``nested_cv``): inside each outer training part, a separate inner
   split does the choosing, and the outer test part is only used to score. The outer folds are the Task 3 folds.
There is no separate "final test set" in this project (Task 3 and 4 use cross-validation only), so none is touched.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config
from .model_training import BRIEF_DEFAULTS, v2_pipeline
from cortexhr.baselines import per_fold_scores

# id, description, parameter overrides on top of BRIEF_DEFAULTS (kept small on purpose: 13 experiments)
CONFIGS: list[dict] = [
    dict(id="E00", note="brief defaults (reference)", params={}),
    dict(id="E01", note="shallower: depth 3, 7 leaves, min child 30, lr 0.03, 300 trees",
         params=dict(max_depth=3, num_leaves=7, min_child_samples=30, learning_rate=0.03, n_estimators=300)),
    dict(id="E02", note="depth 2, 4 leaves, min child 40, lr 0.05, 200 trees",
         params=dict(max_depth=2, num_leaves=4, min_child_samples=40, learning_rate=0.05, n_estimators=200)),
    dict(id="E03", note="E01 + L2 reg 5 + min child 50",
         params=dict(max_depth=3, num_leaves=7, min_child_samples=50, learning_rate=0.03, n_estimators=300, reg_lambda=5.0)),
    dict(id="E04", note="depth 4, 8 leaves, slow lr 0.02, 400 trees, L2 10",
         params=dict(max_depth=4, num_leaves=8, min_child_samples=40, learning_rate=0.02, n_estimators=400, reg_lambda=10.0)),
    dict(id="E05", note="E01 with 100 trees, lr 0.05 (fewer, faster)",
         params=dict(max_depth=3, num_leaves=7, min_child_samples=30, learning_rate=0.05, n_estimators=100)),
    dict(id="E06", note="very shallow: depth 2, 3 leaves, min child 60, lr 0.03, 400 trees, L2 5",
         params=dict(max_depth=2, num_leaves=3, min_child_samples=60, learning_rate=0.03, n_estimators=400, reg_lambda=5.0)),
    dict(id="E07", note="E01 with colsample 0.5, subsample 0.7",
         params=dict(max_depth=3, num_leaves=7, min_child_samples=30, learning_rate=0.03, n_estimators=300,
                     colsample_bytree=0.5, subsample=0.7)),
    dict(id="E08", note="brief shape (depth 5, 15 leaves) but slow lr 0.02, 300 trees, L1 1, L2 10",
         params=dict(max_depth=5, num_leaves=15, min_child_samples=20, learning_rate=0.02, n_estimators=300,
                     reg_alpha=1.0, reg_lambda=10.0)),
    dict(id="E09", note="decision stumps (depth 1, 2 leaves): additive model, lr 0.05, 300 trees",
         params=dict(max_depth=1, num_leaves=2, min_child_samples=30, learning_rate=0.05, n_estimators=300)),
    dict(id="E10", note="depth 2, 4 leaves, min child 30, lr 0.03, 150 trees",
         params=dict(max_depth=2, num_leaves=4, min_child_samples=30, learning_rate=0.03, n_estimators=150)),
    dict(id="E11", note="depth 3, 7 leaves, min child 30, lr 0.01, 500 trees, L2 5, colsample 0.6",
         params=dict(max_depth=3, num_leaves=7, min_child_samples=30, learning_rate=0.01, n_estimators=500,
                     reg_lambda=5.0, colsample_bytree=0.6)),
    dict(id="E12", note="stumps, slower: depth 1, lr 0.02, 500 trees, L2 5",
         params=dict(max_depth=1, num_leaves=2, min_child_samples=30, learning_rate=0.02, n_estimators=500, reg_lambda=5.0)),
]
CONFIG_BY_ID = {c["id"]: c for c in CONFIGS}


def full_params(config_id: str) -> dict:
    p = dict(BRIEF_DEFAULTS)
    p.update(CONFIG_BY_ID[config_id]["params"])
    return p


def complexity(params: dict) -> int:
    """Rough size of the boosted model: number of leaves per tree x number of trees (smaller = simpler)."""
    leaves = min(params["num_leaves"], 2 ** params["max_depth"])
    return int(leaves * params["n_estimators"])


def one_se_select(table: pd.DataFrame, score_col: str = "auc_mean", se_col: str = "auc_se",
                  complexity_col: str = "complexity") -> str:
    """Simplest configuration whose mean score is within one standard error of the best one."""
    best = table.loc[table[score_col].idxmax()]
    floor = best[score_col] - best[se_col]
    ok = table[table[score_col] >= floor]
    return str(ok.sort_values([complexity_col, score_col], ascending=[True, False]).iloc[0]["id"])


# ----------------------------------------------------------------------------------------------
# (1) scoring every configuration on the shared folds
# ----------------------------------------------------------------------------------------------
def score_config_on_folds(params: dict, X, y, folds, class_weight="balanced") -> dict:
    """Fit/score one configuration fold by fold. Returns per-fold ROC-AUC/PR-AUC, train ROC-AUC and the OOF scores."""
    y_arr = np.asarray(y)
    oof = np.zeros(len(y_arr))
    auc, pr, train_auc = [], [], []
    for tr, te in folds:
        est = v2_pipeline(params, class_weight).fit(X.iloc[tr], y_arr[tr])
        p_te = est.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p_te
        auc.append(roc_auc_score(y_arr[te], p_te))
        pr.append(average_precision_score(y_arr[te], p_te))
        train_auc.append(roc_auc_score(y_arr[tr], est.predict_proba(X.iloc[tr])[:, 1]))
    return dict(auc=np.array(auc), pr=np.array(pr), train_auc=np.array(train_auc), oof=oof)


def run_tuning(X, y, folds, configs=None, class_weight="balanced"):
    """Score every configuration on the shared folds. Returns ``(log_dataframe, {id: oof_scores})``."""
    configs = configs or CONFIGS
    rows, oofs = [], {}
    k = len(folds)
    for c in configs:
        params = dict(BRIEF_DEFAULTS); params.update(c["params"])
        t0 = time.time()
        r = score_config_on_folds(params, X, y, folds, class_weight)
        oofs[c["id"]] = r["oof"]
        pf = per_fold_scores(y, r["oof"], folds)          # Task 3's per-fold scoring at the 0.5 cut-off
        rows.append({
            "id": c["id"], "model": "LightGBM", "hyperparameters": str({k_: params[k_] for k_ in sorted(params)}),
            "max_depth": params["max_depth"], "num_leaves": params["num_leaves"], "n_estimators": params["n_estimators"],
            "learning_rate": params["learning_rate"], "min_child_samples": params["min_child_samples"],
            "subsample": params["subsample"], "colsample_bytree": params["colsample_bytree"],
            "reg_alpha": params["reg_alpha"], "reg_lambda": params["reg_lambda"], "class_weight": class_weight,
            "auc_mean": r["auc"].mean(), "auc_sd": r["auc"].std(ddof=1), "auc_se": r["auc"].std(ddof=1) / np.sqrt(k),
            "auc_min_fold": r["auc"].min(), "auc_max_fold": r["auc"].max(),
            "pr_auc_mean": r["pr"].mean(), "f1_mean": pf["f1"].mean(), "precision_mean": pf["precision"].mean(),
            "recall_mean": pf["recall"].mean(), "train_auc_mean": r["train_auc"].mean(),
            "train_minus_cv_auc": r["train_auc"].mean() - r["auc"].mean(), "complexity": complexity(params),
            "seconds": round(time.time() - t0, 2), "description": c["note"],
        })
    log = pd.DataFrame(rows)
    chosen = one_se_select(log)
    best_id = str(log.loc[log["auc_mean"].idxmax(), "id"])
    log["selected"] = log["id"] == chosen
    notes = []
    for _, r in log.iterrows():
        n = []
        if r["id"] == best_id: n.append("highest mean ROC-AUC")
        if r["selected"]: n.append("SELECTED by one-standard-error rule (simplest within 1 SE of best)")
        if r["train_minus_cv_auc"] > 0.25: n.append("strong overfitting (train AUC far above CV)")
        notes.append("; ".join(n))
    log["notes"] = notes
    return log, oofs


def tuning_log_for_csv(log: pd.DataFrame) -> pd.DataFrame:
    """Column layout requested in the brief."""
    out = pd.DataFrame({
        "experiment_id": log["id"], "model": log["model"], "hyperparameters": log["hyperparameters"],
        "cv_roc_auc_mean": log["auc_mean"].round(4), "cv_roc_auc_sd": log["auc_sd"].round(4),
        "cv_pr_auc_mean": log["pr_auc_mean"].round(4), "cv_f1_mean": log["f1_mean"].round(4),
        "cv_precision_mean": log["precision_mean"].round(4), "cv_recall_mean": log["recall_mean"].round(4),
        "train_roc_auc_mean": log["train_auc_mean"].round(4), "train_minus_cv_auc": log["train_minus_cv_auc"].round(4),
        "complexity_leaves_x_trees": log["complexity"], "selected": log["selected"], "notes": log["notes"],
        "description": log["description"],
    })
    return out


# ----------------------------------------------------------------------------------------------
# (2) nested cross-validation on the Task 3 outer folds
# ----------------------------------------------------------------------------------------------
def _inner_scores(params, Xtr, ytr, inner_folds, class_weight):
    aucs = []
    ytr = np.asarray(ytr)
    for a, b in inner_folds:
        est = v2_pipeline(params, class_weight).fit(Xtr.iloc[a], ytr[a])
        aucs.append(roc_auc_score(ytr[b], est.predict_proba(Xtr.iloc[b])[:, 1]))
    return np.array(aucs)


def nested_cv(X, y, folds, configs=None, methods=("sigmoid", "isotonic"), inner_splits: int = 5,
              class_weight="balanced") -> dict:
    """Outer = the Task 3 folds. Inner = a different stratified split of each outer TRAINING part.

    Returns the out-of-fold scores of the tuned model (raw and calibrated with each method) and which configuration won in
    each outer fold. The outer test part never influences configuration choice or calibration.
    """
    configs = configs or CONFIGS
    y_arr = np.asarray(y)
    raw = np.zeros(len(y_arr))
    cal = {m: np.zeros(len(y_arr)) for m in methods}
    chosen, inner_tables = [], []
    for fold_no, (tr, te) in enumerate(folds):
        Xtr, ytr = X.iloc[tr], y_arr[tr]
        inner = list(StratifiedKFold(inner_splits, shuffle=True, random_state=config.RANDOM_STATE + 1).split(Xtr, ytr))
        rows = []
        for c in configs:
            params = dict(BRIEF_DEFAULTS); params.update(c["params"])
            s = _inner_scores(params, Xtr, ytr, inner, class_weight)
            rows.append({"id": c["id"], "auc_mean": s.mean(), "auc_se": s.std(ddof=1) / np.sqrt(len(s)),
                         "complexity": complexity(params)})
        t = pd.DataFrame(rows)
        best = one_se_select(t)
        chosen.append(best)
        inner_tables.append(t.assign(outer_fold=fold_no, chosen=t["id"] == best))
        params = full_params(best)
        est = v2_pipeline(params, class_weight).fit(Xtr, ytr)
        raw[te] = est.predict_proba(X.iloc[te])[:, 1]
        for m in methods:
            cal_est = CalibratedClassifierCV(
                estimator=v2_pipeline(params, class_weight), method=m,
                cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE + 2)).fit(Xtr, ytr)
            cal[m][te] = cal_est.predict_proba(X.iloc[te])[:, 1]
    return {"raw": raw, "calibrated": cal, "chosen_per_fold": chosen, "inner_tables": pd.concat(inner_tables)}
