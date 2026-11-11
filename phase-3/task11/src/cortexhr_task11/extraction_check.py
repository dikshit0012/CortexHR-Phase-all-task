"""Validate skill / experience extraction against the structured fields of the resumes (diagnostics only; not label-based).

* skills  : skills extracted from `resume_text` vs the structured `skills` column, restricted to skills in the extractor vocabulary.
* experience : years parsed from `resume_text` vs `total_experience_years`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .skill_extractor import SkillExtractor, split_skill_list


def validate_extraction(resumes: pd.DataFrame, processed: pd.DataFrame, extractor: SkillExtractor) -> pd.DataFrame:
    p = processed.set_index("resume_id")
    rows = []
    for rec in resumes.to_dict("records"):
        rid = rec["resume_id"]
        truth = {extractor.canonical_key(s) for s in split_skill_list(rec.get("skills"))}
        found = set(p.loc[rid, "skill_keys"])
        tp = len(truth & found)
        exp_t, exp_c = p.loc[rid, "experience_years"], rec.get("total_experience_years")
        rows.append({"resume_id": rid, "layout_difficulty": rec.get("layout_difficulty"), "structured_skills": len(truth),
                     "found_in_text": len(found), "true_positive": tp,
                     "recall": tp / len(truth) if truth else np.nan, "precision": tp / len(found) if found else np.nan,
                     "experience_text": exp_t, "experience_structured": exp_c,
                     "experience_matches": (bool(abs(exp_t - exp_c) < 1e-9) if (exp_t == exp_t and pd.notna(exp_c)) else np.nan)})
    return pd.DataFrame(rows)


def summarize_extraction(v: pd.DataFrame) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        tp, truth, found = g["true_positive"].sum(), g["structured_skills"].sum(), g["found_in_text"].sum()
        return pd.Series({"resumes": len(g), "skill_recall_micro": tp / truth if truth else np.nan,
                          "skill_precision_micro": tp / found if found else np.nan,
                          "experience_match_rate": g["experience_matches"].dropna().astype(float).mean()})
    by = v.groupby("layout_difficulty").apply(agg, include_groups=False).reset_index()
    allrow = agg(v).to_frame().T
    allrow.insert(0, "layout_difficulty", "ALL")
    return pd.concat([by, allrow], ignore_index=True)
