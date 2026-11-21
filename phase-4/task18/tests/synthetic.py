"""SYNTHETIC test data only. Nothing here is real employee, candidate or salary data, and no result computed from it may be
presented as a real monitoring result. These generators let the tests drive the REAL Task 12 / Task 15 code paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SKILLS = ["Python", "SQL", "Excel", "Recruitment", "Payroll", "Communication", "Statistics", "Java"]


# ------------------------------------------------------------------------------------------------ attrition-like data (drift)
def attrition_frame(n: int, seed: int = 0, shift: float = 0.0, remote_share: float = 0.2, missing_engagement: float = 0.2, source: str = None) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    eng = r.normal(3.5 + shift, 0.8, n).clip(1, 5).round(1)
    eng[r.random(n) < missing_engagement] = np.nan
    mode = r.choice(["Onsite", "Hybrid", "Remote"], n, p=[(1 - remote_share) / 2, (1 - remote_share) / 2, remote_share])
    df = pd.DataFrame({
        "employee_id": [f"X-{seed}-{i}" for i in range(n)],
        "tenure_months": r.gamma(3, 14 * (1 + shift), n).round(),
        "annual_fixed_pay": r.lognormal(13.2 + shift, 0.4, n).round(),
        "comp_ratio": r.normal(1.0 + shift / 2, 0.15, n),
        "attendance_rate_3m": r.beta(20, 2, n),
        "last_engagement_score": eng,
        "avg_overtime_hours_per_month_3m": np.where(r.random(n) < 0.8, 0.0, r.gamma(2, 6, n)),
        "job_level": r.choice([1, 2, 3, 4, 5, 6], n, p=[.3, .3, .2, .12, .06, .02]),
        "work_mode": mode,
    })
    if source:
        df["source"] = source
    return df


# ------------------------------------------------------------------------------------------------ Task 11/12 twin resumes
def write_twin_data(root: Path, n_pairs: int = 25, bias: float = 0.0) -> Path:
    """Writes Task 11 data (jobs, resumes) and the Task 12 pairs CSV. `bias` > 0 appends text to one twin so its score changes."""
    data = root / "task11_data"
    data.mkdir(parents=True, exist_ok=True)
    jobs = pd.DataFrame([{"job_id": f"J{j:02d}", "job_title": t, "min_experience_years": 2, "required_skills": req, "nice_to_have_skills": nice,
                          "description_text": f"We need a {t} with strong {req} skills."} for j, (t, req, nice) in
                         enumerate([("Data Analyst", "Python; SQL", "Statistics"), ("HR Executive", "Recruitment; Communication", "Excel"),
                                    ("Payroll Specialist", "Payroll; Excel", "SQL")], 1)])
    rows, pairs = [], []
    for i in range(n_pairs):
        fam = f"J{(i % 3) + 1:02d}"
        skills = list(np.random.default_rng(i).choice(SKILLS, 4, replace=False))
        body = f"Experienced professional with {2 + i % 6} years. Skills: {', '.join(skills)}. Worked on projects using {skills[0]} and {skills[1]}."
        for tag, name in (("A", f"Person Alpha{i}"), ("B", f"Person Beta{i}")):
            extra = " Additional certification in advanced analytics and leadership programmes." * (1 if (tag == "B" and bias) else 0)
            rows.append({"resume_id": f"R{i:03d}{tag}", "candidate_name": name, "email": f"{name.split()[1].lower()}@example.org", "city": "Pune",
                         "total_experience_years": 2 + i % 6, "current_title": "Analyst", "current_company": "Acme", "highest_education": "B.Tech",
                         "college_tier": 2, "skills": "; ".join(skills), "certifications": "", "source_format": "pdf", "layout_difficulty": "easy",
                         "target_job_family": fam, "resume_text": f"{name}\n{name.split()[1].lower()}@example.org\n{body}{extra}"})
        pairs.append({"pair_id": f"P{i:02d}", "resume_id_original": f"R{i:03d}A", "resume_id_twin": f"R{i:03d}B", "attribute_changed": "name",
                      "skills_identical": "Y", "experience_identical": "Y", "expected_score_difference": 0})
    jobs.to_csv(data / "job_descriptions.csv", index=False)
    pd.DataFrame(rows).to_csv(data / "resumes.csv", index=False)
    pd.DataFrame([{"resume_id": r["resume_id"], "job_id": r["target_job_family"], "human_suitability_score": 3, "shortlist": "N"} for r in rows]
                 ).to_csv(data / "resume_job_labels.csv", index=False)
    pairs_csv = root / "resume_bias_pairs.csv"
    pd.DataFrame(pairs).to_csv(pairs_csv, index=False)
    return pairs_csv


# ------------------------------------------------------------------------------------------------ Task 15 salary data
def write_salary_data(root: Path, n: int = 700, seed: int = 3, gap_penalty: float = 0.0) -> tuple[Path, Path]:
    """Synthetic salary_training CSV + unlock_CX-15 file. `gap_penalty` lowers pay of career-gap hires (a synthetic injected gap)."""
    r = np.random.default_rng(seed)
    role = r.choice([f"R{i:02d}" for i in range(8)], n)
    level = r.integers(1, 6, n)
    exp = r.integers(0, 15, n)
    mkt = 1 + r.random(n) * 0.3
    gap = r.choice([0, 0, 0, 0, 0, 3, 6, 12, 24], n)
    base = (400_000 + 120_000 * level + 15_000 * exp + 20_000 * pd.Series(role).str[1:].astype(int)) * mkt
    pay = base * (1 + r.normal(0, 0.04, n)) * np.where(gap >= 12, 1 - gap_penalty, 1)
    band = base * r.uniform(0.97, 1.03, n) / mkt
    df = pd.DataFrame({"hire_id": [f"H{i}" for i in range(n)], "role_id": role, "job_level": level, "department_id": [f"D{int(x[1:]) % 4:02d}" for x in role],
                       "prior_experience_years": exp, "market_index": mkt, "work_location": r.choice(["Pune", "Delhi", "Mumbai"], n),
                       "highest_education": r.choice(["B.Tech", "MBA", "B.Com"], n), "career_gap_months": gap,
                       "band_min_2026": band * 0.9, "band_mid_2026": band, "band_max_2026": band * 1.1,
                       "hire_date": pd.date_range("2018-01-01", periods=n, freq="D"), "hire_year": 2018, "skills_count": r.integers(1, 9, n),
                       "source": r.choice(["real", "extension"], n), "offered_annual_fixed": pay.round()})
    csv = root / "salary_training.csv"
    df.to_csv(csv, index=False)
    unlock_dir = root / "private"
    unlock_dir.mkdir(exist_ok=True)
    unlock = unlock_dir / "unlock_CX-15_protected_attributes.csv"
    pd.DataFrame({"person_ref": df["hire_id"], "record_type": "hire", "gender": r.choice(["Female", "Male", "Other"], n, p=[.45, .5, .05]),
                  "age_years": 30, "age_band": "30-39"}).to_csv(unlock, index=False)
    return csv, unlock
