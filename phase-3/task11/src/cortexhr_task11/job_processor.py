"""Job description processing. Extracts title, required / preferred skills, experience requirement and the text used for TF-IDF.

Education preference and work location are deliberately NOT part of the scored job text: education preference leans toward degree /
institution proxies and location is a region proxy; neither is used to rank.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from .preprocessing import tfidf_text
from .skill_extractor import SkillExtractor, split_skill_list

_DROP_SENTENCE = re.compile(r"(?:^|(?<=\.)\s)(?:Education|Location)\s*:[^.]*\.", re.I)


@dataclass
class JobProfile:
    job_id: str
    job_title: str
    required: List[Tuple[str, str]] = field(default_factory=list)    # (display name as written in the JD, canonical key)
    preferred: List[Tuple[str, str]] = field(default_factory=list)
    min_experience_years: Optional[float] = None
    description_text: str = ""
    scoring_text: str = ""

    @property
    def required_skills(self) -> List[str]:
        return [d for d, _ in self.required]


def _to_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f and f >= 0 else None
    except (TypeError, ValueError):
        return None


def _skills(value, extractor: SkillExtractor) -> List[Tuple[str, str]]:
    seen, out = set(), []
    for s in split_skill_list(value):
        k = extractor.canonical_key(s)
        if k not in seen:
            seen.add(k)
            out.append((s, k))
    return out


def process_job(row, extractor: SkillExtractor) -> JobProfile:
    get = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d)
    desc = get("description_text")
    desc = "" if desc is None or (isinstance(desc, float) and desc != desc) or desc is pd.NA else str(desc)
    desc_scored = _DROP_SENTENCE.sub(" ", desc)
    title = "" if get("job_title") is None or get("job_title") is pd.NA else str(get("job_title"))
    req, pref = _skills(get("required_skills"), extractor), _skills(get("nice_to_have_skills"), extractor)
    text = " ".join([title, " ".join(d for d, _ in req), " ".join(d for d, _ in pref), desc_scored])
    return JobProfile(job_id=str(get("job_id")), job_title=title, required=req, preferred=pref,
                      min_experience_years=_to_float(get("min_experience_years")),
                      description_text=desc, scoring_text=tfidf_text(text))


def process_jobs(jobs: pd.DataFrame, extractor: SkillExtractor) -> List[JobProfile]:
    return [process_job(rec, extractor) for rec in jobs.to_dict("records")]
