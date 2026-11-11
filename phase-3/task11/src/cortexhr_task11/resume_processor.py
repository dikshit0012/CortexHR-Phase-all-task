"""Resume processing: free text -> scoring text, skills and experience. Uses ONLY the resume text (plus identity strings that are
removed from it). Structured demographic / proxy columns are never turned into features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .preprocessing import extract_experience_years, normalize_text, scrub_identity, tfidf_text
from .skill_extractor import SkillExtractor

# Protected/proxy values are used only to redact matching free-text mentions, never as scoring features.
SCRUB_COLUMNS = [
    "candidate_name", "email", "city", "highest_education", "college_tier", "college_name", "university",
    "institution", "alma_mater", "age", "date_of_birth", "dob", "gender", "sex", "pronouns",
]


@dataclass
class ProcessedResume:
    resume_id: str
    scoring_text: str            # scrubbed + normalised + numeric tokens dropped -> TF-IDF input
    skills: List[str]            # display names extracted from the text
    skill_keys: List[str]
    experience_years: Optional[float]
    experience_source: str


def process_resume(resume_id: str, text, extractor: SkillExtractor, identity_values=()) -> ProcessedResume:
    scrubbed = scrub_identity(text, identity_values)
    keys = extractor.extract_keys(scrubbed)
    years, src = extract_experience_years(scrubbed)
    return ProcessedResume(
        resume_id=str(resume_id),
        scoring_text=tfidf_text(scrubbed),
        skills=[extractor.display_name(k) for k in keys],
        skill_keys=keys,
        experience_years=years,
        experience_source=src,
    )


def process_resumes(resumes: pd.DataFrame, extractor: SkillExtractor) -> pd.DataFrame:
    """One row per resume. Missing text / missing identity columns never raise."""
    rows = []
    for rec in resumes.to_dict("records"):
        ident = [rec.get(c) for c in SCRUB_COLUMNS if c in rec]
        p = process_resume(rec["resume_id"], rec.get("resume_text"), extractor, ident)
        rows.append({"resume_id": p.resume_id, "scoring_text": p.scoring_text, "skills": p.skills, "skill_keys": p.skill_keys,
                     "n_skills": len(p.skills), "experience_years": p.experience_years if p.experience_years is not None else float("nan"),
                     "experience_source": p.experience_source, "empty_text": not p.scoring_text})
    return pd.DataFrame(rows)
