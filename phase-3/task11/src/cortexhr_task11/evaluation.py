"""Ranking evaluation against the human judgement labels (evaluation only - labels are never a scoring input).

Label interpretation (verified on the data: shortlist == 'Y'  <=>  human_suitability_score >= 4)
* binary relevance : shortlist == 'Y' (score 4 or 5)                         -> Top-5 hits, Precision@5, Recall@5
* graded relevance : rel = human_suitability_score - 1  (1->0 ... 5->4)       -> NDCG@5, gain = 2^rel - 1 (also linear gain reported)

Evaluation pool: for each job, only resumes that have a human label for that job. Resumes without a label for a job are NOT
assumed non-relevant; they are simply outside that job's evaluation pool. Ranks used here are re-computed inside the pool.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

K_DEFAULT = 5


def labeled_pool(ranked: pd.DataFrame, labels: pd.DataFrame, score_col: str = "final_score", tie_break_ascending: bool = True) -> pd.DataFrame:
    """Join scores with labels (inner) and rank within each job's labelled pool (exact ties -> resume_id, ascending by default)."""
    pool = ranked.merge(labels, on=["job_id", "resume_id"], how="inner", validate="one_to_one")
    pool = pool.rename(columns={"rank": "rank_all_resumes"})
    pool = pool.sort_values(["job_id", score_col, "resume_id"], ascending=[True, False, tie_break_ascending], kind="mergesort")
    pool["pool_rank"] = pool.groupby("job_id").cumcount() + 1
    pool["n_pool"] = pool.groupby("job_id")["resume_id"].transform("count")
    pool["relevant"] = (pool["shortlist"] == "Y").astype(int)
    pool["rel_grade"] = pool["human_suitability_score"] - 1
    return pool


def top_k_hits(relevant_in_rank_order: List[int], k: int = K_DEFAULT) -> int:
    return int(sum(relevant_in_rank_order[:k]))


def precision_at_k(relevant_in_rank_order: List[int], k: int = K_DEFAULT) -> float:
    n = min(k, len(relevant_in_rank_order))
    return float("nan") if n == 0 else top_k_hits(relevant_in_rank_order, k) / n


def recall_at_k(relevant_in_rank_order: List[int], k: int = K_DEFAULT) -> float:
    total = sum(relevant_in_rank_order)
    return float("nan") if total == 0 else top_k_hits(relevant_in_rank_order, k) / total


def dcg_at_k(grades_in_rank_order: List[float], k: int = K_DEFAULT, exponential: bool = True) -> float:
    s = 0.0
    for i, g in enumerate(grades_in_rank_order[:k], start=1):
        gain = (2.0 ** g - 1.0) if exponential else float(g)
        s += gain / math.log2(i + 1)
    return s


def ndcg_at_k(grades_in_rank_order: List[float], k: int = K_DEFAULT, exponential: bool = True) -> float:
    ideal = dcg_at_k(sorted(grades_in_rank_order, reverse=True), k, exponential)
    return float("nan") if ideal == 0 else dcg_at_k(grades_in_rank_order, k, exponential) / ideal


def evaluate_pool(pool: pd.DataFrame, k: int = K_DEFAULT) -> pd.DataFrame:
    """Per-job ranking quality table plus micro- and macro-average rows."""
    rows: List[Dict] = []
    for job_id, g in pool.sort_values(["job_id", "pool_rank"]).groupby("job_id", sort=True):
        rel, grades = g["relevant"].tolist(), g["rel_grade"].tolist()
        n, R = len(g), int(sum(rel))
        rows.append({
            "job_id": job_id, "job_title": g["job_title"].iloc[0], "n_labeled_candidates": n, "n_relevant": R,
            f"top{k}_hits": top_k_hits(rel, k), f"precision_at_{k}": precision_at_k(rel, k), f"recall_at_{k}": recall_at_k(rel, k),
            f"max_possible_recall_at_{k}": (min(k, R) / R) if R else float("nan"),
            f"random_expected_precision_at_{k}": (R / n) if n else float("nan"),
            f"ndcg_at_{k}": ndcg_at_k(grades, k, True), f"ndcg_at_{k}_linear_gain": ndcg_at_k(grades, k, False),
        })
    t = pd.DataFrame(rows)
    num = t.drop(columns=["job_id", "job_title"])
    avg = {c: (num[c].sum() if c in ("n_labeled_candidates", "n_relevant", f"top{k}_hits") else num[c].mean()) for c in num.columns}
    micro = evaluate_micro(pool, k)
    micro_row = {
        "job_id": "ALL (micro avg; counts summed)", "job_title": "",
        "n_labeled_candidates": int(len(pool)), "n_relevant": int(pool["relevant"].sum()),
        f"top{k}_hits": micro["top_k_hits"], f"precision_at_{k}": micro["precision_at_k"],
        f"recall_at_{k}": micro["recall_at_k"], f"max_possible_recall_at_{k}": micro["max_possible_recall_at_k"],
        f"random_expected_precision_at_{k}": micro["random_expected_precision_at_k"],
        f"ndcg_at_{k}": micro["ndcg_at_k"], f"ndcg_at_{k}_linear_gain": micro["ndcg_at_k_linear_gain"],
    }
    macro_row = {"job_id": "ALL (macro avg; counts summed)", "job_title": "", **avg}
    t = pd.concat([t, pd.DataFrame([micro_row, macro_row])], ignore_index=True)
    return t


def evaluate_micro(pool: pd.DataFrame, k: int = K_DEFAULT) -> Dict[str, float | int]:
    """Aggregate candidate-level counts and DCG/IDCG across jobs (unlike the per-job macro means)."""
    hits = slots = relevant = 0
    expected_hits = possible_hits = dcg = idcg = dcg_linear = idcg_linear = 0.0
    for _, g in pool.sort_values(["job_id", "pool_rank"]).groupby("job_id", sort=True):
        rel = g["relevant"].tolist()
        grades = g["rel_grade"].tolist()
        job_hits = top_k_hits(rel, k)
        job_slots = min(k, len(g))
        job_relevant = int(sum(rel))
        hits += job_hits
        slots += job_slots
        relevant += job_relevant
        expected_hits += job_slots * job_relevant / len(g) if len(g) else 0.0
        possible_hits += min(k, job_relevant)
        dcg += dcg_at_k(grades, k)
        idcg += dcg_at_k(sorted(grades, reverse=True), k)
        dcg_linear += dcg_at_k(grades, k, exponential=False)
        idcg_linear += dcg_at_k(sorted(grades, reverse=True), k, exponential=False)
    return {
        "top_k_hits": hits,
        "precision_at_k": hits / slots if slots else float("nan"),
        "recall_at_k": hits / relevant if relevant else float("nan"),
        "max_possible_recall_at_k": possible_hits / relevant if relevant else float("nan"),
        "random_expected_precision_at_k": expected_hits / slots if slots else float("nan"),
        "ndcg_at_k": dcg / idcg if idcg else float("nan"),
        "ndcg_at_k_linear_gain": dcg_linear / idcg_linear if idcg_linear else float("nan"),
    }


def ablation(ranked: pd.DataFrame, labels: pd.DataFrame, k: int = K_DEFAULT) -> pd.DataFrame:
    """Report-only comparison of TF-IDF alone, skill overlap alone and the hybrid. Weights are NOT tuned on labels."""
    out = []
    for name, col in (("tfidf_only", "tfidf_similarity"), ("skill_overlap_only", "skill_overlap"), ("hybrid_60_40", "final_score")):
        t = evaluate_pool(labeled_pool(ranked, labels, col), k)
        last = t.iloc[-1]
        out.append({"method": name, f"top{k}_hits_total": int(last[f"top{k}_hits"]), f"mean_precision_at_{k}": last[f"precision_at_{k}"],
                    f"mean_recall_at_{k}": last[f"recall_at_{k}"], f"mean_ndcg_at_{k}": last[f"ndcg_at_{k}"],
                    "mean_random_expected_precision": last[f"random_expected_precision_at_{k}"]})
    return pd.DataFrame(out)


def tie_sensitivity(ranked: pd.DataFrame, labels: pd.DataFrame, k: int = K_DEFAULT) -> pd.DataFrame:
    """Macro metrics when exact score ties are broken by resume_id ascending vs descending. A large gap would mean the result
    depends on an arbitrary tie-break; identical numbers mean ties inside the evaluated region do not change the metrics."""
    rows = []
    for name, asc in (("ties_by_resume_id_ascending", True), ("ties_by_resume_id_descending", False)):
        last = evaluate_pool(labeled_pool(ranked, labels, tie_break_ascending=asc), k).iloc[-1]
        rows.append({"tie_break": name, f"top{k}_hits_total": int(last[f"top{k}_hits"]), f"mean_precision_at_{k}": last[f"precision_at_{k}"],
                     f"mean_recall_at_{k}": last[f"recall_at_{k}"], f"mean_ndcg_at_{k}": last[f"ndcg_at_{k}"]})
    return pd.DataFrame(rows)
