"""Central configuration for Task 18 monitoring.

Everything that is a *choice* (features, thresholds, baselines) lives here, each with its source. Nothing in this module is
a computed result: every monitored value is calculated at run time from the data.

Private data is NEVER committed. Locations come from environment variables (same names as Tasks 12/14/15/17 where they exist).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

TASK18_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ["CORTEXHR_REPO_ROOT"]).expanduser() if os.environ.get("CORTEXHR_REPO_ROOT") else TASK18_ROOT.parents[1]

# ------------------------------------------------------------------------------------------------ datasets
REFERENCE_SHEET = "attrition_dataset"          # reference: what the attrition-v2 model was trained on (Task 1-4, 13)
SCORING_SHEET = "attrition_scoring_2026"       # current: the employees being scored now (Task 14)
ID_COLUMN = "employee_id"                      # used only to COUNT duplicates; never written to any output
MIN_ROWS = 30                                  # below this PSI is not meaningful -> NOT_EVALUATED
EXPECTED_ROWS = {"reference": 2389, "scoring": 430}   # informational (Task 1 audit / Task 14 README); a mismatch is a warning, not a failure


def _env_path(name: str) -> Optional[Path]:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


def output_dir() -> Path:
    return _env_path("CORTEXHR_MONITORING_DIR") or (TASK18_ROOT / "outputs" / "monitoring")


def workbook_candidates(specific_env: Optional[str] = None) -> Tuple[Path, ...]:
    """Workbook (never committed). The workbooks differ between tasks (Task 1-4: 02_CortexHR_AI_ML_Dataset.xlsx, Task 14/15:
    CortexHR_Dataset.xlsx) while CORTEXHR_WORKBOOK is shared, so each dataset can have its own override, e.g.
    CORTEXHR_REFERENCE_WORKBOOK, CORTEXHR_SCORING_WORKBOOK, CORTEXHR_PAYGAP_WORKBOOK; then CORTEXHR_WORKBOOK; then ./data."""
    for name in ([specific_env] if specific_env else []) + ["CORTEXHR_WORKBOOK"]:
        env = _env_path(name)
        if env:
            return (env,)
    d = TASK18_ROOT / "data"
    return (d / "CortexHR_Dataset.xlsx", d / "02_CortexHR_AI_ML_Dataset.xlsx")


# ------------------------------------------------------------------------------------------------ PSI
PSI_WATCH = 0.10        # PSI <  0.10            -> OK       (the thresholds required by the Task 18 brief,
PSI_ALERT = 0.25        # 0.10 <= PSI <= 0.25    -> WATCH     identical to the docstring of the course psi() helper)
                        # PSI >  0.25            -> ALERT
PSI_BINS = 10           # reference-quantile bins for numeric features (same as the original helper)
PSI_EPS = 1e-6          # zero-frequency floor (same as the original helper)
PSI_DECIMALS = 6        # PSI is rounded BEFORE classification so the printed value and the verdict can never disagree
MAX_DISCRETE = PSI_BINS  # numeric feature with <= this many distinct reference values is compared value-by-value


# POPULATION-MATCHED REFERENCE. The real reference sheet mixes two populations: 2,000 simulated `historical_extension` rows (2021-2025, ids HIST-*) and 389
# `master_employee_snapshot` rows (real employees observed 2025-09-30, exactly 12 months before the scoring date; 333 of the 430 scoring employees are the same
# people). Scoring rows are built like the snapshot rows, so the snapshot rows are the population-matched reference and drive the verdict.
# The model, however, was trained on ALL rows, so PSI against the full "training" reference is kept for every feature and is never hidden:
#   * training status worse than matched status, and the two sub-populations differ on that feature (internal PSI >= PSI_WATCH) -> composition effect:
#     an ALERT is downgraded to WATCH (never to OK), a WATCH adds only a note;
#   * training status worse, but the sub-populations do NOT differ (internal PSI < PSI_WATCH) -> unexplained disagreement: the stricter status stands.
# No age adjustment of tenure is applied: tested on the real data it made tenure PSI worse (0.12 -> 0.28) because the workforce also turned over.
MATCH_COLUMN = "source"
MATCH_VALUE = "master_employee_snapshot"
MIN_MATCHED_ROWS = 100            # below this the matched reference is too small; fall back to the full reference (reported, never silent)


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str      # "numeric" (quantile bins) or "categorical" (one bin per value; also used for ordinal levels)
    why: str


# Eight features, chosen from the 33 columns the production model attrition-v2 actually consumes
# (artifacts/attrition-v2.metadata.json -> feature_list), so every one is (a) available at scoring time, (b) a real model input,
# (c) expected in both sheets. Leakage columns and dept_attrition_rate_prior_12m (excluded from the model) are NOT eligible.
FEATURES: Tuple[Feature, ...] = (
    Feature("tenure_months", "numeric", "workforce experience mix; attrition is strongly tenure-dependent"),
    Feature("annual_fixed_pay", "numeric", "pay level (INR); a pay-structure or inflation shift moves attrition risk"),
    Feature("comp_ratio", "numeric", "pay vs role mid-point; below 1 = underpaid, a core flight-risk driver"),
    Feature("attendance_rate_3m", "numeric", "behavioural signal measured by the attendance system"),
    Feature("last_engagement_score", "numeric", "survey-based; ~22% missing in training, so missing-rate drift matters"),
    Feature("avg_overtime_hours_per_month_3m", "numeric", "workload / burnout proxy; zero-inflated (~81% zeros in training)"),
    Feature("job_level", "categorical", "seniority mix (ordinal 1-6, compared level-by-level)"),
    Feature("work_mode", "categorical", "Onsite / Hybrid / Remote mix; a policy change shows up here first"),
)
assert len(FEATURES) == 8

# ------------------------------------------------------------------------------------------------ fairness: Task 12 (twin resumes)
TWIN_EXPECTED_PAIRS = 25                       # Task 12 data_loading: load_pairs(expected_count=25)
TWIN_WATCH_GAP = 0.001                         # MONITORING CHOICE (not company policy): 10% of the Task 12 tolerance and above the
                                               # 0.000677 max gap Task 12 measured for the text-only screener (i.e. above its noise)
# The ALERT threshold is Task 12's own TOLERANCE (0.01), imported at run time from cortexhr_task12.fairness_metrics, never copied.
# Task 12 README section 4 ("Task 11 as shipped", 25 pairs, primary job). The monitored production screener is "as shipped"
# because the Task 17 /v1/resume/rank endpoint runs that exact code.
TWIN_BASELINE = {"source": "Task 12 README s4, screener task11_as_shipped", "pairs": 25, "average_gap": 0.0, "maximum_gap": 0.0}

# ------------------------------------------------------------------------------------------------ fairness: Task 15 (pay gap)
# Task 15 defines NO pass/fail threshold (only MIN_GROUP = 30). The rules below are MONITORING CHOICES, documented in the
# README, and are not company policy. They flag *change versus the Task 15 baseline*, not discrimination.
PAY_WATCH_REL, PAY_ALERT_REL = 0.10, 0.25      # relative change of a gap versus its baseline
PAY_MATERIALITY = {"ratio": 0.005, "inr": 5_000.0}   # ignore changes smaller than this (noise floor); ratio = pay / role band
PAY_MODEL_QUANTITIES = {
    "predicted pay vs role band (ratio)": "ratio",
    "signed error (predicted - actual)": "inr",
}
# Task 15 README s7.1 (gender) and s7.2 (career gap): selected model, held-out test set, official group_gap().
PAY_BASELINE: Dict[str, Dict[str, float]] = {
    "gender": {"predicted pay vs role band (ratio)": 0.017, "signed error (predicted - actual)": 23_757.0},
    "career_gap": {"predicted pay vs role band (ratio)": 0.035, "signed error (predicted - actual)": 55_549.0},
}
PAY_BASELINE_SOURCE = "Task 15 README s7.1-7.2 (selected model, held-out test set, official group_gap())"
PAY_MIN_GROUP = 30                             # Task 15 config.MIN_GROUP; smaller groups are flagged unreliable

# ------------------------------------------------------------------------------------------------ aggregation
# A check listed here that comes back NOT_EVALUATED makes the overall status ALERT; any other unevaluated check -> WATCH.
REQUIRED_CHECKS = ("dataset_validation", "drift", "twin_resume", "pay_gap")


def required_checks() -> Tuple[str, ...]:
    """Override with CORTEXHR_MONITOR_OPTIONAL_CHECKS=twin_resume,pay_gap to make a check optional (unavailable -> WATCH)."""
    opt = {c.strip() for c in os.environ.get("CORTEXHR_MONITOR_OPTIONAL_CHECKS", "").split(",") if c.strip()}
    return tuple(c for c in REQUIRED_CHECKS if c not in opt)


# ------------------------------------------------------------------------------------------------ API
API_MAX_AGE_HOURS = float(os.environ.get("CORTEXHR_MONITORING_MAX_AGE_HOURS", "48"))   # older report -> flagged "stale"
