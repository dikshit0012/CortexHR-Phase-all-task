"""SYNTHETIC stress test (NOT part of the 25 supplied pairs, not an audit result about real candidates).

The supplied twins differ only in name, e-mail and institution tier, so they cannot exercise age, date of birth, graduation years,
gendered titles or pronouns. Here a sample of the real resumes gets two variants that differ ONLY in one sensitive token set
(a swap such as 'He'/'She', or disclosure vs non-disclosure of a personal-details block); the same resume content, skills and experience are in both. The gap between
the two variants is then measured with and without masking. Name pathways are covered by the real twins, not here."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import pandas as pd

from . import paths
from .screeners import FAIR, TEXT_ONLY, run_screener

paths.ensure_task11_importable()

_same = lambda t: t
PERTURBATIONS: Dict[str, Tuple[Callable[[str], str], Callable[[str], str]]] = {
    # disclosure vs non-disclosure: identical resume, X omits the personal block, Y includes it
    "age_and_date_of_birth_disclosed": (_same, lambda t: t + "\n\nPERSONAL\nDate of birth: 14 March 1999\nAge: 25 years old\n"),
    "gender_and_marital_status_disclosed": (_same, lambda t: t + "\n\nGender: Female\nMarital status: Married\n"),
    # swaps: only the sensitive token differs
    "graduation_year_old_vs_recent": (lambda t: t + "\n\nGraduated May 2022, class of 2022\n", lambda t: t + "\n\nGraduated June 2003, class of 2003\n"),
    "gendered_title_Mr_vs_Ms": (lambda t: "Mr. " + t, lambda t: "Ms. " + t),
    "pronouns_he_vs_she": (lambda t: t + "\n\nHe prefers collaborative teams. His availability is flexible.\n",
                           lambda t: t + "\n\nShe prefers collaborative teams. Her availability is flexible.\n"),
}


def run_stress_test(resumes: pd.DataFrame, jobs: pd.DataFrame, screeners=(TEXT_ONLY, FAIR), step: int = 6) -> pd.DataFrame:
    """One resume at a time (every `step`-th resume, deterministic) is given variant X or Y while the rest of the corpus is unchanged, so the
    sensitive tokens are rare in the corpus exactly as for a single real candidate. Gap = |score(X) - score(Y)| for the resume's own target job."""
    base = resumes[["resume_id", "resume_text", "target_job_family"]].reset_index(drop=True)
    sample = list(range(0, len(base), step))
    rows: List[Dict] = []
    for label, (fx, fy) in PERTURBATIONS.items():
        gaps = {sc: [] for sc in screeners}
        for i in sample:
            tgt, rid = base.at[i, "target_job_family"], base.at[i, "resume_id"]
            vx, vy = base[["resume_id", "resume_text"]].copy(), base[["resume_id", "resume_text"]].copy()
            vx.at[i, "resume_text"] = fx(base.at[i, "resume_text"])
            vy.at[i, "resume_text"] = fy(base.at[i, "resume_text"])
            for sc in screeners:
                sx = run_screener(sc, vx, jobs).set_index(["job_id", "resume_id"])["final_score"].loc[(tgt, rid)]
                sy = run_screener(sc, vy, jobs).set_index(["job_id", "resume_id"])["final_score"].loc[(tgt, rid)]
                gaps[sc].append(abs(sx - sy))
        for sc in screeners:
            g = pd.Series(gaps[sc])
            rows.append({"perturbation": label, "screener": sc, "n_resumes": len(g), "mean_gap": g.mean(), "max_gap": g.max(),
                         "non_zero": int((g > 1e-9).sum()), "above_tolerance_0.01": int((g > 0.01).sum())})
    return pd.DataFrame(rows)
