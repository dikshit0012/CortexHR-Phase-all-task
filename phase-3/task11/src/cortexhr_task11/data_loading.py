"""Loading and validation of the three Task 11 inputs. Nothing is invented: every column comes from the supplied CSVs.

job_descriptions   : job_id, job_title, department_id, job_level, min_experience_years, required_skills,
                     nice_to_have_skills, education_preference, work_location, openings, description_text
resumes            : resume_id, candidate_name, email, city, total_experience_years, current_title, current_company,
                     highest_education, college_tier, skills, certifications, source_format, layout_difficulty,
                     target_job_family, resume_text
resume_job_labels  : resume_id, job_id, human_suitability_score (1-5), shortlist (Y/N), required_skill_coverage,
                     nice_skill_coverage, reviewer_note      <- EVALUATION ONLY, never a scoring input
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from . import paths

JOB_COLUMNS = ["job_id", "job_title", "min_experience_years", "required_skills", "nice_to_have_skills", "description_text"]
RESUME_COLUMNS = ["resume_id", "resume_text"]
LABEL_COLUMNS = ["resume_id", "job_id", "human_suitability_score", "shortlist"]


class DataError(ValueError):
    """Raised when an input file does not have the structure this task relies on."""


def _read(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise DataError(f"Input file not found: {path}")
    return pd.read_csv(path)


def _require(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f"{name}: missing required column(s) {missing}; found {list(df.columns)}")


def _strip_ids(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("string").str.strip()
    return df


def load_jobs(path: Path | None = None) -> pd.DataFrame:
    df = _read(path or paths.JOBS_CSV)
    _require(df, JOB_COLUMNS, "job_descriptions")
    df = _strip_ids(df, ["job_id"])
    if df["job_id"].isna().any() or df["job_id"].duplicated().any():
        raise DataError("job_descriptions: job_id must be present and unique")
    return df.reset_index(drop=True)


def load_resumes(path: Path | None = None) -> pd.DataFrame:
    df = _read(path or paths.RESUMES_CSV)
    _require(df, RESUME_COLUMNS, "resumes")
    df = _strip_ids(df, ["resume_id"])
    if df["resume_id"].isna().any() or df["resume_id"].duplicated().any():
        raise DataError("resumes: resume_id must be present and unique")
    return df.reset_index(drop=True)


def load_labels(path: Path | None = None) -> pd.DataFrame:
    df = _read(path or paths.LABELS_CSV)
    _require(df, LABEL_COLUMNS, "resume_job_labels")
    df = _strip_ids(df, ["resume_id", "job_id"])
    df["shortlist"] = df["shortlist"].astype("string").str.strip().str.upper()
    if df.duplicated(["resume_id", "job_id"]).any():
        raise DataError("resume_job_labels: duplicate (resume_id, job_id) pairs")
    scores = pd.to_numeric(df["human_suitability_score"], errors="coerce")
    if scores.isna().any() or not scores.between(1, 5).all():
        raise DataError("resume_job_labels: human_suitability_score must be numeric in 1..5")
    df["human_suitability_score"] = scores.astype(int)
    if not set(df["shortlist"]) <= {"Y", "N"}:
        raise DataError("resume_job_labels: shortlist must be Y or N")
    return df.reset_index(drop=True)


def check_referential_integrity(jobs: pd.DataFrame, resumes: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
    bad_res = sorted(set(labels["resume_id"]) - set(resumes["resume_id"]))
    bad_job = sorted(set(labels["job_id"]) - set(jobs["job_id"]))
    if bad_res or bad_job:
        raise DataError(f"labels reference unknown ids: resumes={bad_res[:5]} jobs={bad_job[:5]}")
    return {"labels_with_unknown_resume": 0, "labels_with_unknown_job": 0}


def inspect_datasets(jobs: pd.DataFrame, resumes: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
    """Machine-readable inspection summary (saved to reports/dataset_inspection.json)."""
    pool = labels.groupby("job_id").size()
    rel = labels[labels["shortlist"] == "Y"].groupby("job_id").size()
    return {
        "job_descriptions": {"rows": len(jobs), "columns": list(jobs.columns),
                             "missing_values": {k: int(v) for k, v in jobs.isna().sum().items()}},
        "resumes": {"rows": len(resumes), "columns": list(resumes.columns),
                    "missing_values": {k: int(v) for k, v in resumes.isna().sum().items()},
                    "layout_difficulty_counts": resumes["layout_difficulty"].value_counts().to_dict()
                    if "layout_difficulty" in resumes else {}},
        "resume_job_labels": {"rows": len(labels), "columns": list(labels.columns),
                              "missing_values": {k: int(v) for k, v in labels.isna().sum().items()},
                              "score_counts": {int(k): int(v) for k, v in labels["human_suitability_score"].value_counts().sort_index().items()},
                              "shortlist_counts": labels["shortlist"].value_counts().to_dict(),
                              "labelled_resumes": int(labels["resume_id"].nunique()),
                              "labelled_pairs_per_job": {k: int(v) for k, v in pool.items()},
                              "shortlisted_per_job": {k: int(rel.get(k, 0)) for k in pool.index},
                              "shortlist_equals_score_ge_4": bool(((labels["human_suitability_score"] >= 4) == (labels["shortlist"] == "Y")).all())},
        "relationships": {
            "possible_resume_job_pairs": len(jobs) * len(resumes),
            "labelled_pairs": len(labels),
            "labelled_fraction_of_possible_pairs": round(len(labels) / (len(jobs) * len(resumes)), 4),
            "note": "Only a subset of resume-job pairs carries a human label; evaluation uses that labelled pool per job. "
                    "Unlabelled pairs are NOT treated as non-relevant.",
        },
    }
