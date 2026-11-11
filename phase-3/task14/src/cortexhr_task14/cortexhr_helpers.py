"""Small, tested helpers so you can spend your time on modelling. Read them, then use them."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
DATA = Path("../03_Data/csv")
def load(sheet): return pd.read_csv(DATA / f"{sheet}.csv")
def risk_bands(scores):
    """High = top 10% of scores, Medium = next 20%, Low = the rest (the company's rule)."""
    s = pd.Series(scores); hi, md = s.quantile(0.90), s.quantile(0.70)
    return np.where(s >= hi, "High", np.where(s >= md, "Medium", "Low"))
def binary_report(y_true, y_score, top_fraction=0.10):
    """Honest report for an imbalanced target: AUC, PR-AUC and precision/recall for the top X% flagged."""
    y_true = np.asarray(y_true); y_score = np.asarray(y_score); k = max(1, int(len(y_true) * top_fraction))
    flag = np.zeros(len(y_true), int); flag[np.argsort(-y_score)[:k]] = 1
    return dict(roc_auc=roc_auc_score(y_true, y_score), pr_auc=average_precision_score(y_true, y_score),
                precision_at_top=precision_score(y_true, flag), recall_at_top=recall_score(y_true, flag), f1_at_top=f1_score(y_true, flag), base_rate=y_true.mean())
def catch_rate(flagged_ids, truth_ids):
    """For anomaly detectors: precision = how many flags were real; recall = how many real ones you found."""
    f, t = set(flagged_ids), set(truth_ids); hit = len(f & t)
    return dict(flagged=len(f), real=len(t), caught=hit, precision=hit / len(f) if f else 0, recall=hit / len(t) if t else 0)
def group_gap(scores, groups):
    """Average score per group and the largest gap between groups. A big gap is a fairness warning, not proof."""
    g = pd.Series(scores).groupby(pd.Series(groups).values).mean(); return g.to_dict(), float(g.max() - g.min())
def psi(expected, actual, bins=10):
    """Population Stability Index: <0.10 stable, 0.10-0.25 watch, >0.25 drift."""
    e, a = np.asarray(expected, float), np.asarray(actual, float); cuts = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    ce = np.histogram(e, cuts)[0] / len(e); ca = np.histogram(np.clip(a, cuts[0], cuts[-1]), cuts)[0] / len(a)
    ce, ca = np.clip(ce, 1e-6, None), np.clip(ca, 1e-6, None); return float(np.sum((ca - ce) * np.log(ca / ce)))
