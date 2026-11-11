"""The three screeners compared in this task. All reuse Task 11's skill extractor, job processing, TF-IDF + overlap ranking engine
(`rank_resumes`) UNCHANGED; they differ only in how a resume is turned into scoring text / skills.

  task11_as_shipped : Task 11 `process_resumes` with the full resumes table (it scrubs name, e-mail, city, education, tier by looking
                      up the structured column values). Original behaviour, untouched.
  task11_text_only  : the same Task 11 code when only `resume_id` + `resume_text` are available (structured columns withheld).
  task12_fair       : text-only personal-information masking (fair_preprocessing) in front of the same engine; it is given only
                      `resume_id` + `resume_text` and cannot see any other column.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

from . import paths

paths.ensure_task11_importable()
from cortexhr_task11.job_processor import process_jobs                       # noqa: E402
from cortexhr_task11.preprocessing import extract_experience_years, tfidf_text  # noqa: E402
from cortexhr_task11.ranking_engine import RankingConfig, rank_resumes        # noqa: E402
from cortexhr_task11.resume_processor import process_resumes                  # noqa: E402
from cortexhr_task11.skill_extractor import SkillExtractor, build_default_extractor  # noqa: E402

from .fair_preprocessing import ALL_CATEGORIES, SCORING_INPUT_COLUMNS, mask_personal_information  # noqa: E402

AS_SHIPPED, TEXT_ONLY, FAIR = "task11_as_shipped", "task11_text_only", "task12_fair"
SCREENERS = (AS_SHIPPED, TEXT_ONLY, FAIR)


def text_only_view(resumes: pd.DataFrame) -> pd.DataFrame:
    """Feature isolation: the ONLY columns a text-only screener may see."""
    missing = [c for c in SCORING_INPUT_COLUMNS if c not in resumes.columns]
    if missing:
        raise ValueError(f"resumes table lacks {missing}")
    return resumes[list(SCORING_INPUT_COLUMNS)].copy()


def process_resumes_fair(resumes: pd.DataFrame, extractor: SkillExtractor, categories: Iterable[str] = ALL_CATEGORIES) -> pd.DataFrame:
    """Same output schema as Task 11 `process_resumes`, built from masked text only. Extra column `masked_items` records what was masked."""
    rows = []
    for rid, text in text_only_view(resumes).itertuples(index=False):
        masked, counts = mask_personal_information(text, categories)
        keys = extractor.extract_keys(masked)
        years, src = extract_experience_years(masked)
        scoring = tfidf_text(masked)
        rows.append({"resume_id": str(rid), "scoring_text": scoring, "skills": [extractor.display_name(k) for k in keys], "skill_keys": keys,
                     "n_skills": len(keys), "experience_years": years if years is not None else float("nan"), "experience_source": src,
                     "empty_text": not scoring, "masked_items": sum(counts.values())})
    return pd.DataFrame(rows)


def run_screener(name: str, resumes: pd.DataFrame, jobs: pd.DataFrame, config: RankingConfig = RankingConfig(),
                 categories: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Rank every resume for every job with the named screener. Returns the Task 11 ranked table (one row per job x resume)."""
    if name == AS_SHIPPED:
        ex = build_default_extractor(jobs, resumes)                      # exactly what Task 11's pipeline does
        processed = process_resumes(resumes, ex)
    elif name == TEXT_ONLY:
        view = text_only_view(resumes)
        ex = build_default_extractor(jobs)                               # vocabulary from the job descriptions only
        processed = process_resumes(view, ex)
    elif name == FAIR:
        ex = build_default_extractor(jobs)
        processed = process_resumes_fair(resumes, ex, ALL_CATEGORIES if categories is None else categories)
    else:
        raise ValueError(f"unknown screener {name!r}; expected one of {SCREENERS}")
    return rank_resumes(processed, process_jobs(jobs, ex), config)
