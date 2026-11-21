"""Final Challenge experiment: what does silent imputation of the 3 monthly-derived features cost, and what does the guard trade?

Run from the repo root after configuring the Task 13 environment and authorised workbook:
    python phase-4/task20/run_experiment.py <scoring-csv>
Writes docs/final_challenge/results/*.json|csv. Nothing here changes a model or any production file."""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
TASK_ROOT = Path(__file__).resolve().parent
ROOT = TASK_ROOT.parents[1]
sys.path[:0] = [str(TASK_ROOT), str(ROOT / "phase-3/task13/src"), str(ROOT / "phase-1/task04/src")]
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from attrition_v2 import data_validation as dv, model_versioning as mv
from cortexhr.baselines import model_pipeline
from completeness_guard import MONTHLY_DERIVED, check_batch

OUT = ROOT / "docs/final_challenge/results"; OUT.mkdir(parents=True, exist_ok=True)
X, y, _ = dv.load_data(); y = pd.Series(np.asarray(y)).reset_index(drop=True); X = X.reset_index(drop=True)
folds = dv.get_task3_folds(y)
rng = np.random.default_rng(42)

def top10(y_, p_):
    k = max(1, int(round(0.10 * len(p_)))); idx = np.argsort(-p_)[:k]
    return float(np.mean(y_[idx])), float(np.sum(y_[idx]) / max(np.sum(y_), 1))

def metrics(y_, p_):
    pr, rc = top10(y_, p_)
    return {"n": int(len(y_)), "roc_auc": float(roc_auc_score(y_, p_)), "pr_auc": float(average_precision_score(y_, p_)), "precision_top10": pr, "recall_top10": rc}

pA = np.zeros(len(y)); pB = np.zeros(len(y)); inc = np.zeros(len(y), bool)
for tr, te in folds:
    m = model_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)).fit(X.iloc[tr], y.iloc[tr])
    pA[te] = m.predict_proba(X.iloc[te])[:, 1]
    Xm = X.iloc[te].copy(); Xm[list(MONTHLY_DERIVED)] = np.nan
    pB[te] = m.predict_proba(Xm)[:, 1]
    chosen = rng.choice(te, size=int(round(0.30 * len(te))), replace=False); inc[chosen] = True
yv = y.to_numpy()
# scenario C: 30% of rows lack the monthly features (random, seed 42)
pC = pA.copy()
for tr, te in folds:
    pass
Xc = X.copy(); Xc.loc[inc, list(MONTHLY_DERIVED)] = np.nan
for tr, te in folds:
    m = model_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)).fit(X.iloc[tr], y.iloc[tr])
    pC[te] = m.predict_proba(Xc.iloc[te])[:, 1]
res = {"A_all_complete": metrics(yv, pA),
       "B_all_missing_silently_imputed": metrics(yv, pB),
       "C_30pct_missing_baseline_score_everyone": metrics(yv, pC),
       "C_30pct_missing_guard_score_complete_only": {**metrics(yv[~inc], pC[~inc]), "coverage": float((~inc).mean()),
                                                      "leavers_in_unscored_rows": int(yv[inc].sum()), "unscored_rows": int(inc.sum())},
       "C_30pct_missing_guard_review_queue_only_incomplete_rows_scored_by_baseline": metrics(yv[inc], pC[inc])}
# per-fold mean AUC for A and B (to see spread)
res["per_fold_auc_A"] = [float(roc_auc_score(yv[te], pA[te])) for _, te in folds]
res["per_fold_auc_B"] = [float(roc_auc_score(yv[te], pB[te])) for _, te in folds]
# real scoring file: delivered CSV vs correctly built monthly features
loaded = mv.load_model("attrition-v2")
sc = mv.build_scoring_table()
p_full = mv.predict_attrition(sc, loaded).iloc[:, 0].to_numpy()
sc_missing = sc.copy(); present = [c for c in MONTHLY_DERIVED if c in sc_missing.columns]; sc_missing[present] = np.nan
p_miss = mv.predict_attrition(sc_missing, loaded).iloc[:, 0].to_numpy()
k = int(round(0.10 * len(p_full)))
res["scoring_2026_430_rows"] = {"n": int(len(p_full)), "spearman_complete_vs_imputed": float(spearmanr(p_full, p_miss).statistic),
    "mean_abs_probability_diff": float(np.mean(np.abs(p_full - p_miss))), "max_abs_probability_diff": float(np.max(np.abs(p_full - p_miss))),
    "top10pct_overlap": float(len(set(np.argsort(-p_full)[:k]) & set(np.argsort(-p_miss)[:k])) / k)}
if len(sys.argv) != 2:
    raise SystemExit("usage: python phase-4/task20/run_experiment.py <authorized-scoring-csv>")
raw_path = Path(sys.argv[1]).expanduser()
if not raw_path.is_file():
    raise FileNotFoundError(f"scoring CSV not found: {raw_path}")
raw = pd.read_csv(raw_path)
feats = json.load(open(ROOT / "phase-3/task13/model_artifacts/attrition-v2/metadata.json"))["feature_list"]
g = check_batch(raw.to_dict("records"), feats)
res["guard_on_delivered_scoring_csv"] = {k_: g[k_] for k_ in ("n_records", "n_complete", "n_review", "coverage")}
json.dump(res, open(OUT / "experiment_results.json", "w"), indent=2)
print(json.dumps(res, indent=2))
