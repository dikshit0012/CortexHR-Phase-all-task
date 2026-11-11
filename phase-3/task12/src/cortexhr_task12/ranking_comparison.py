"""Ranking quality of each screener on the Task 11 human labels (evaluation only; labels never reach a screener)."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from . import paths

paths.ensure_task11_importable()
from cortexhr_task11.evaluation import evaluate_pool, labeled_pool  # noqa: E402

K = 5


def quality_row(ranked: pd.DataFrame, labels: pd.DataFrame, k: int = K) -> Dict[str, float]:
    t = evaluate_pool(labeled_pool(ranked, labels), k).iloc[-1]
    return {"top5_hits": int(t[f"top{k}_hits"]), "precision_at_5": float(t[f"precision_at_{k}"]), "recall_at_5": float(t[f"recall_at_{k}"]),
            "ndcg_at_5": float(t[f"ndcg_at_{k}"])}


def compare_quality(rankings: Dict[str, pd.DataFrame], labels: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    rows = []
    q = {n: quality_row(r, labels) for n, r in rankings.items()}
    for key, label in [("precision_at_5", "Precision@5"), ("recall_at_5", "Recall@5"), ("ndcg_at_5", "NDCG@5"), ("top5_hits", "Top-5 Hits")]:
        row = {"metric": label}
        for n in rankings:
            row[n] = q[n][key]
        row["difference_vs_" + baseline] = q[candidate][key] - q[baseline][key]
        rows.append(row)
    return pd.DataFrame(rows)


def per_job_quality(ranked: pd.DataFrame, labels: pd.DataFrame, k: int = K) -> pd.DataFrame:
    return evaluate_pool(labeled_pool(ranked, labels), k)
