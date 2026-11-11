"""Hybrid, explainable resume ranking.

    final_score = 0.60 * tfidf_similarity + 0.40 * required_skill_overlap          (both components in [0, 1])

* tfidf_similarity   : cosine similarity between the TF-IDF vector of the (scrubbed) resume text and of the job text. TF-IDF is
                       fitted on resumes + job descriptions only (unsupervised; no labels), unigrams + bigrams, sublinear tf.
* skill_overlap      : matched required skills / required skills. Preferred skills are excluded from this component but appear in
                       the job TF-IDF text, so they can affect tfidf_similarity.

This module NEVER receives human labels, candidate names, gender, age, college tier, city or any other protected / proxy
attribute. Experience is extracted for reporting only (`experience_years`, `meets_min_experience`) and does not enter the score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .job_processor import JobProfile
from .preprocessing import tfidf_text

SCORE_COLUMNS = ["tfidf_similarity", "skill_overlap", "final_score"]
RANKED_COLUMNS = ["job_id", "job_title", "resume_id", "rank", "final_score", "tfidf_similarity", "skill_overlap",
                  "matched_skills", "missing_skills", "preferred_skills_matched", "n_required_skills",
                  "experience_years", "experience_source", "min_experience_years", "meets_min_experience"]


@dataclass(frozen=True)
class RankingConfig:
    tfidf_weight: float = 0.60
    skill_weight: float = 0.40
    ngram_range: tuple = (1, 2)
    sublinear_tf: bool = True
    stop_words: str | None = "english"

    def __post_init__(self):
        if min(self.tfidf_weight, self.skill_weight) < 0 or abs(self.tfidf_weight + self.skill_weight - 1.0) > 1e-9:
            raise ValueError("weights must be non-negative and sum to 1")


def skill_overlap_for(required_keys: Sequence[str], resume_keys: set) -> float:
    """Required-skill coverage in [0, 1]; a job with no required skills gets 0.0 (flagged by n_required_skills == 0)."""
    if not required_keys:
        return 0.0
    return sum(1 for k in required_keys if k in resume_keys) / len(required_keys)


def fit_tfidf(processed_resumes: pd.DataFrame, jobs: List[JobProfile], config: RankingConfig = RankingConfig()):
    """Fit the TF-IDF vectoriser on resumes + job texts (unsupervised) and return (vectoriser, resume_matrix, job_matrix)."""
    vec = TfidfVectorizer(ngram_range=config.ngram_range, sublinear_tf=config.sublinear_tf, stop_words=config.stop_words,
                          lowercase=False, norm="l2")
    vec.fit(list(processed_resumes["scoring_text"]) + [j.scoring_text for j in jobs])   # labels never involved
    return vec, vec.transform(processed_resumes["scoring_text"]), vec.transform([j.scoring_text for j in jobs])


def explain_tfidf(vec, job: JobProfile, resume_scoring_text: str, top_n: int = 5) -> Dict[str, Any]:
    """Which terms create the TF-IDF similarity, and how much of it comes from terms that are NOT part of a job skill phrase
    (i.e. generic words). Used for explanations only."""
    names = vec.get_feature_names_out()
    prod = vec.transform([job.scoring_text]).multiply(vec.transform([resume_scoring_text])).tocoo()
    contrib = sorted(zip(prod.col, prod.data), key=lambda x: -x[1])
    total = float(sum(v for _, v in contrib)) or 1.0
    skill_tokens = {t for d, _ in job.required + job.preferred for t in tfidf_text(d).split()}
    skill_part = sum(v for c, v in contrib if any(t in skill_tokens for t in names[c].split()))
    return {"top_terms": [names[c] for c, _ in contrib[:top_n]], "generic_share": max(0.0, 1.0 - skill_part / total), "total": total}


def rank_resumes(processed_resumes: pd.DataFrame, jobs: List[JobProfile], config: RankingConfig = RankingConfig()) -> pd.DataFrame:
    """Score every (job, resume) pair and rank resumes within each job (rank 1 = best).

    Ties are broken by resume_id ascending (deterministic and unrelated to any attribute); the number of exactly tied scores is
    reported in the run summary. Returns one row per pair with all explanation fields.
    """
    if processed_resumes.empty or not jobs:
        return pd.DataFrame(columns=RANKED_COLUMNS)
    if processed_resumes["resume_id"].duplicated().any():
        raise ValueError("duplicate resume_id in processed resumes")
    _, R, J = fit_tfidf(processed_resumes, jobs, config)
    sim = np.clip(cosine_similarity(J, R), 0.0, 1.0)         # shape (n_jobs, n_resumes)

    rids = processed_resumes["resume_id"].tolist()
    keysets = [set(k) for k in processed_resumes["skill_keys"]]
    exp = processed_resumes["experience_years"].tolist()
    exp_src = processed_resumes["experience_source"].tolist()
    rows: List[Dict[str, Any]] = []
    for ji, job in enumerate(jobs):
        req_keys = [k for _, k in job.required]
        for ri, rid in enumerate(rids):
            overlap = skill_overlap_for(req_keys, keysets[ri])
            tf = float(sim[ji, ri])
            final = config.tfidf_weight * tf + config.skill_weight * overlap
            e = exp[ri]
            meets = (pd.NA if (job.min_experience_years is None or e != e) else bool(e >= job.min_experience_years))
            rows.append({
                "job_id": job.job_id, "job_title": job.job_title, "resume_id": rid,
                "final_score": final, "tfidf_similarity": tf, "skill_overlap": overlap,
                "matched_skills": "; ".join(d for d, k in job.required if k in keysets[ri]),
                "missing_skills": "; ".join(d for d, k in job.required if k not in keysets[ri]),
                "preferred_skills_matched": "; ".join(d for d, k in job.preferred if k in keysets[ri]),
                "n_required_skills": len(req_keys),
                "experience_years": e, "experience_source": exp_src[ri],
                "min_experience_years": job.min_experience_years, "meets_min_experience": meets,
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["job_id", "final_score", "resume_id"], ascending=[True, False, True], kind="mergesort")
    df["rank"] = df.groupby("job_id").cumcount() + 1
    validate_scores(df)
    return df[RANKED_COLUMNS].reset_index(drop=True)


def validate_scores(df: pd.DataFrame) -> None:
    """Raises if any score is missing / outside [0, 1] or a (job, resume) pair is duplicated."""
    for c in SCORE_COLUMNS:
        v = df[c]
        if v.isna().any() or ((v < -1e-12) | (v > 1 + 1e-12)).any():
            raise ValueError(f"invalid values in {c}")
    if df.duplicated(["job_id", "resume_id"]).any():
        raise ValueError("duplicate (job_id, resume_id) pair")


def count_ties(df: pd.DataFrame) -> int:
    """Number of resumes whose final_score is exactly equal to another resume's within the same job."""
    return int(df.duplicated(["job_id", "final_score"], keep=False).sum())


def top_k_for_job(ranked: pd.DataFrame, job_id: str, top_k: int = 5) -> Dict[str, Any]:
    """Response body for the proposed POST /v1/resume/rank endpoint (see docs/API_PROPOSAL.md)."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    sub = ranked[ranked["job_id"] == job_id].sort_values("rank").head(top_k)
    if sub.empty:
        raise KeyError(job_id)
    split = lambda s: [x for x in str(s).split("; ") if x]
    cands = []
    for r in sub.to_dict("records"):
        e = r["experience_years"]
        cands.append({"resume_id": r["resume_id"], "rank": int(r["rank"]), "final_score": round(float(r["final_score"]), 4),
                      "tfidf_similarity": round(float(r["tfidf_similarity"]), 4), "skill_overlap": round(float(r["skill_overlap"]), 4),
                      "matched_skills": split(r["matched_skills"]), "missing_skills": split(r["missing_skills"]),
                      "experience_years": None if e != e else float(e)})
    return {"job_id": job_id, "job_title": str(sub.iloc[0]["job_title"]), "ranked_candidates": cands}
