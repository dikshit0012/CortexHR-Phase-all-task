"""Score both twins of every pair with a screener's ranked table. No model logic here - only lookup and comparison."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .fairness_metrics import TOLERANCE, ZERO_EPS, relative_gap


def competition_rank(scores: pd.Series) -> pd.Series:
    """Rank 1 = best; exactly tied scores share a rank (so a tie-break by resume_id is not mistaken for different treatment)."""
    v = scores.to_numpy(float)
    return pd.Series([1 + int((v > x + 1e-12).sum()) for x in v], index=scores.index)


def primary_jobs(pairs: pd.DataFrame, resumes: pd.DataFrame, column: str = "target_job_family") -> pd.Series:
    """The job each pair is written for (audit set-up only). Both twins must agree."""
    r = resumes.set_index("resume_id")[column]
    a, b = r.loc[pairs["resume_id_original"]].to_numpy(), r.loc[pairs["resume_id_twin"]].to_numpy()
    if (a != b).any():
        raise ValueError("twins of a pair target different jobs")
    return pd.Series(a, index=pairs["pair_id"].to_numpy())


def score_pairs(ranked: pd.DataFrame, pairs: pd.DataFrame, job_by_pair: Optional[pd.Series] = None, tolerance: float = TOLERANCE) -> pd.DataFrame:
    """One row per (pair, job). With `job_by_pair`, only that job per pair (the primary audit); otherwise all jobs."""
    required_ranked = {"job_id", "resume_id", "final_score", "tfidf_similarity", "skill_overlap"}
    required_pairs = {"pair_id", "resume_id_original", "resume_id_twin", "attribute_changed"}
    if required_ranked - set(ranked.columns):
        raise ValueError(f"ranked table lacks columns {sorted(required_ranked - set(ranked.columns))}")
    if required_pairs - set(pairs.columns):
        raise ValueError(f"pairs table lacks columns {sorted(required_pairs - set(pairs.columns))}")
    if pairs.empty or pairs["pair_id"].isna().any() or pairs["pair_id"].duplicated().any():
        raise ValueError("pairs must contain unique, non-missing pair_id values")
    if pairs[["resume_id_original", "resume_id_twin", "attribute_changed"]].isna().any().any():
        raise ValueError("pairs contain missing resume IDs or attribute descriptions")
    if ranked.duplicated(["job_id", "resume_id"]).any():
        raise ValueError("ranked table contains duplicate (job_id, resume_id) rows")
    if ranked.empty:
        raise ValueError("ranked table contains no scores")
    numeric_scores = ranked[["final_score", "tfidf_similarity", "skill_overlap"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_scores).all():
        raise ValueError("ranked table contains non-finite scores")
    if ((numeric_scores < -1e-12) | (numeric_scores > 1 + 1e-12)).any():
        raise ValueError("ranked table contains scores outside [0, 1]")
    if job_by_pair is not None:
        missing_jobs = set(pairs["pair_id"]) - set(job_by_pair.index)
        if missing_jobs:
            raise ValueError(f"job_by_pair lacks pair IDs: {sorted(missing_jobs)[:5]}")
        if job_by_pair.loc[pairs["pair_id"]].isna().any():
            raise ValueError("job_by_pair contains missing job IDs")

    r = ranked.copy()
    r["rank_competition"] = r.groupby("job_id")["final_score"].transform(competition_rank)
    idx = r.set_index(["job_id", "resume_id"])
    rows = []
    for p in pairs.itertuples():
        jobs = [job_by_pair.loc[p.pair_id]] if job_by_pair is not None else sorted(r["job_id"].unique())
        for jid in jobs:
            missing = {p.resume_id_original, p.resume_id_twin} - set(idx.loc[jid].index) if jid in idx.index.get_level_values(0) else {
                p.resume_id_original, p.resume_id_twin
            }
            if missing:
                raise ValueError(f"ranked table lacks both twin scores for pair {p.pair_id} and job {jid}: {sorted(missing)}")
            a, b = idx.loc[(jid, p.resume_id_original)], idx.loc[(jid, p.resume_id_twin)]
            signed_gap = float(b["final_score"] - a["final_score"])
            gap = abs(signed_gap)
            rows.append({"pair_id": p.pair_id, "resume_a_id": p.resume_id_original, "resume_b_id": p.resume_id_twin, "job_id": jid,
                         "attribute_changed": p.attribute_changed, "score_a": a["final_score"], "score_b": b["final_score"],
                         "signed_gap_b_minus_a": signed_gap,
                         "twin_gap": gap, "relative_gap_pct": relative_gap(a["final_score"], b["final_score"]) * 100,
                         "tfidf_gap": abs(a["tfidf_similarity"] - b["tfidf_similarity"]), "skill_overlap_gap": abs(a["skill_overlap"] - b["skill_overlap"]),
                         "rank_a": int(a["rank_competition"]), "rank_b": int(b["rank_competition"]),
                         "rank_difference": int(abs(a["rank_competition"] - b["rank_competition"])),
                         "higher_scored": "tie" if gap <= ZERO_EPS else ("A" if a["final_score"] > b["final_score"] else "B"),
                         "non_zero_gap": bool(gap > ZERO_EPS), "exceeds_tolerance": bool(gap > tolerance)})
    return pd.DataFrame(rows)
