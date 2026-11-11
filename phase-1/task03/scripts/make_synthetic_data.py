"""Create a STAND-IN workbook so Task 3 runs when the real workbook is absent.

It is synthetic. It is NOT your real CortexHR data and the benchmark sheet is
NOT the real one. Replace data/raw/cortexhr_dataset.xlsx with your own workbook
(sheets: `employees` and `baseline_benchmarks`) and re-run the notebook.

Honesty note: the strength of the signal in the fake data was tuned so that a
logistic regression lands near ROC-AUC 0.71. That match is by construction and
proves nothing about your real data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexhr.paths import BENCHMARK_SHEET, DATA_RAW, DATA_SHEET, WORKBOOK_NAME  # noqa: E402

N_ROWS = 2400
TARGET_RATE = 0.147
SIGNAL = 1.15  # tuned so logistic regression is near 0.71 ROC-AUC


def build_employees(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = N_ROWS
    age = np.clip(rng.normal(37, 9, n), 20, 60).round()
    job_level = rng.choice([1, 2, 3, 4, 5], n, p=[0.30, 0.30, 0.20, 0.13, 0.07])
    monthly_income = (2500 + 1800 * job_level + rng.normal(0, 900, n)).clip(1500).round()
    years_at_company = np.clip(rng.gamma(2.0, 3.0, n), 0, 35).round(1)
    years_since_promo = np.clip(years_at_company * rng.uniform(0, 0.8, n), 0, 15).round(1)
    distance = rng.integers(1, 30, n).astype(float)
    overtime = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    job_sat = rng.choice([1, 2, 3, 4], n, p=[0.19, 0.20, 0.31, 0.30]).astype(float)
    wlb = rng.choice([1, 2, 3, 4], n, p=[0.06, 0.22, 0.52, 0.20])
    n_companies = rng.integers(0, 10, n)
    performance = rng.choice([3, 4], n, p=[0.85, 0.15])
    training_hours = rng.integers(0, 60, n)
    department = rng.choice(["Sales", "R&D", "Engineering", "HR", "Finance"], n,
                            p=[0.30, 0.25, 0.25, 0.08, 0.12])

    z = (0.85 * (overtime == "Yes")
         - 0.45 * (job_sat - 2.5)
         - 0.035 * (age - 37)
         - 0.25 * (wlb - 2.5)
         + 0.06 * years_since_promo
         + 0.020 * (distance - 15)
         + 0.07 * (n_companies - 4)
         - 0.00018 * (monthly_income - 7000)
         + 0.35 * (department == "Sales")) * SIGNAL
    z = z + rng.normal(0, 1.0, n)
    lo, hi = -10.0, 10.0
    for _ in range(60):  # bisection on the intercept to hit the target rate
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(z + mid)))).mean()
        lo, hi = (mid, hi) if rate < TARGET_RATE else (lo, mid)
    p = 1 / (1 + np.exp(-(z + mid)))
    left = (rng.uniform(size=n) < p).astype(int)

    # Leaky column: only exists AFTER someone leaves. Task 2 must exclude it.
    exit_iv = np.where(left == 1, rng.uniform(size=n) < 0.92, rng.uniform(size=n) < 0.02)

    df = pd.DataFrame({
        "employee_id": [f"E{i:05d}" for i in range(1, n + 1)],
        "age": age, "department": department, "job_level": job_level,
        "monthly_income": monthly_income, "years_at_company": years_at_company,
        "years_since_last_promotion": years_since_promo, "distance_from_home_km": distance,
        "overtime": overtime, "job_satisfaction": job_sat, "work_life_balance": wlb,
        "num_companies_worked": n_companies, "performance_rating": performance,
        "training_hours_last_year": training_hours,
        "exit_interview_completed": np.where(exit_iv, "Yes", "No"),
        "left_within_12m": left,
    })
    for col, frac in [("monthly_income", 0.03), ("job_satisfaction", 0.04),
                      ("distance_from_home_km", 0.02)]:
        df.loc[rng.uniform(size=n) < frac, col] = np.nan
    return df


def build_benchmarks() -> pd.DataFrame:
    return pd.DataFrame({
        "model": ["DummyClassifier", "LogisticRegression"],
        "roc_auc": [0.50, 0.71],
        "source": ["PLACEHOLDER: constant scores give 0.5 by definition",
                   "PLACEHOLDER: approximate value stated in the task text"],
    })


def main() -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / WORKBOOK_NAME
    readme = pd.DataFrame({"note": [
        "SYNTHETIC STAND-IN DATA. Not the real CortexHR workbook.",
        "Replace this file with your real workbook (same sheet names) and re-run.",
        "The benchmark sheet here is a placeholder, not your real benchmark sheet.",
    ]})
    with pd.ExcelWriter(out) as xl:
        readme.to_excel(xl, sheet_name="README", index=False)
        build_employees().to_excel(xl, sheet_name=DATA_SHEET, index=False)
        build_benchmarks().to_excel(xl, sheet_name=BENCHMARK_SHEET, index=False)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
