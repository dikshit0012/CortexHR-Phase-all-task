"""Evidence for WHERE score differences come from. Everything is measured on the supplied pairs; nothing is assumed."""
from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import chi2_contingency

from . import paths
from .counterfactual_testing import score_pairs
from .fair_preprocessing import ALL_CATEGORIES, mask_personal_information
from .screeners import FAIR, process_resumes_fair

paths.ensure_task11_importable()
from cortexhr_task11.job_processor import process_jobs                          # noqa: E402
from cortexhr_task11.ranking_engine import rank_resumes                          # noqa: E402
from cortexhr_task11.skill_extractor import build_default_extractor              # noqa: E402

VARIANTS: Dict[str, frozenset] = {
    "nothing masked (raw text)": frozenset(),
    "e-mail/URL only": frozenset({"email"}),
    "institution tier only": frozenset({"tier"}),
    "name only": frozenset({"name"}),
    "name + e-mail": frozenset({"name", "email"}),
    "name + e-mail + location": frozenset({"name", "email", "location"}),
    "all categories (task12_fair)": ALL_CATEGORIES,
}


def ablation(resumes: pd.DataFrame, jobs: pd.DataFrame, pairs: pd.DataFrame, job_by_pair: pd.Series) -> pd.DataFrame:
    """Re-score the pairs with different masking categories switched on. Shows which masked attribute removes how much of the gap."""
    ex = build_default_extractor(jobs)
    profiles = process_jobs(jobs, ex)
    rows = []
    for label, cats in VARIANTS.items():
        ranked = rank_resumes(process_resumes_fair(resumes, ex, cats), profiles)
        prim, allj = score_pairs(ranked, pairs, job_by_pair), score_pairs(ranked, pairs)
        by_attr = prim.groupby("attribute_changed")["twin_gap"].mean()
        rows.append({"variant": label, "masked_categories": ", ".join(sorted(cats)) or "none",
                     "mean_gap_primary_job": prim["twin_gap"].mean(), "max_gap_primary_job": prim["twin_gap"].max(),
                     "nonzero_pairs_primary_job": int(prim["non_zero_gap"].sum()), "mean_gap_all_jobs": allj["twin_gap"].mean(),
                     "mean_gap_name_only_pairs": float(by_attr.get("candidate_name (gender-coded) only", float("nan"))),
                     "mean_gap_name_and_tier_pairs": float(by_attr.get("candidate_name (gender-coded) and college tier", float("nan")))})
    return pd.DataFrame(rows)


def _line_category(line: str, index: int) -> str:
    low = line.lower()
    if "@" in low or "mail-demo" in low:
        return "contact line (e-mail / name)"
    if "tier" in low and "institution" in low:
        return "education line (institution tier)"
    if index == 0:
        return "header (candidate name)"
    return "other (unintended difference)"


def text_differences(resumes: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Classify every line that differs between the twins' raw texts. 'other' would indicate an unintended content difference."""
    r = resumes.set_index("resume_id")["resume_text"]
    rows = []
    for p in pairs.itertuples():
        a, b = (r.loc[x].replace("\\n", "\n").splitlines() for x in (p.resume_id_original, p.resume_id_twin))
        cats: Dict[str, int] = {}
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        for tag, i1, i2, _, _ in sm.get_opcodes():
            if tag == "equal":
                continue
            for i in range(i1, max(i2, i1 + 1)):
                c = _line_category(a[i] if i < len(a) else "", i)
                cats[c] = cats.get(c, 0) + 1
        rows.append({"pair_id": p.pair_id, "attribute_changed": p.attribute_changed, "n_differing_lines": sum(cats.values()),
                     **{k: cats.get(k, 0) for k in ["header (candidate name)", "contact line (e-mail / name)", "education line (institution tier)",
                                                    "other (unintended difference)"]}})
    return pd.DataFrame(rows)


def skill_extraction_consistency(ranked: pd.DataFrame, pairs: pd.DataFrame) -> Dict[str, int]:
    """Twins have identical skills and experience, so both must receive identical extracted skills and experience for every job."""
    idx = ranked.set_index(["job_id", "resume_id"])
    bad_skill = bad_exp = n = 0
    for p in pairs.itertuples():
        for jid in ranked["job_id"].unique():
            a, b = idx.loc[(jid, p.resume_id_original)], idx.loc[(jid, p.resume_id_twin)]
            n += 1
            bad_skill += int(a["matched_skills"] != b["matched_skills"] or a["missing_skills"] != b["missing_skills"])
            ea, eb = a["experience_years"], b["experience_years"]
            bad_exp += int(not ((ea != ea and eb != eb) or ea == eb))
    return {"pair_job_comparisons": n, "inconsistent_skill_extraction": bad_skill, "inconsistent_experience_extraction": bad_exp}


def text_risk_scan(resumes: pd.DataFrame) -> pd.DataFrame:
    """How many resumes contain each maskable category. Zero means the supplied data cannot test that pathway (it is still masked)."""
    counts = {c: 0 for c in sorted(ALL_CATEGORIES)}
    for t in resumes["resume_text"]:
        for c in counts:
            if mask_personal_information(t, {c})[1].get(c, 0):
                counts[c] += 1
    return pd.DataFrame([{"category": c, "resumes_containing": n, "of_resumes": len(resumes),
                          "testable_with_supplied_twins": c in {"name", "email", "location", "tier"}} for c, n in counts.items()])


def proxy_scan(resumes: pd.DataFrame, protected: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Does a job-relevant attribute (the job a resume targets) correlate with gender / tier? If so, skill-based scoring will show group
    differences that are NOT discrimination; those skills must not be removed - they must be monitored."""
    if protected is None:
        return pd.DataFrame()
    m = resumes[["resume_id", "target_job_family"]].merge(protected, on="resume_id")
    rows = []
    for attr in ("inferred_gender", "college_tier"):
        ct = pd.crosstab(m["target_job_family"], m[attr])
        chi2, p, dof, _ = chi2_contingency(ct)
        rows.append({"attribute": attr, "versus": "target_job_family", "chi2": chi2, "dof": dof, "p_value": p,
                     "cramers_v": (chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1))) ** 0.5,
                     "note": "small expected counts; descriptive only"})
    return pd.DataFrame(rows)
