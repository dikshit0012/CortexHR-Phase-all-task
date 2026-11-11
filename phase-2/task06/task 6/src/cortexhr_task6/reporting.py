"""Task 6 - data dictionary, diagnostics, masked examples and the Markdown report.

Every number in the report is computed from the result tables passed in; the only
static text is structural (what a rule or error type is), never a measured claim.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .payroll_evaluation import ALL_ERRORS, ID_COL, LABEL_TYPE_COL
from .payroll_features import EMPLOYEE_COL, FEATURE_DEFINITIONS, MONTH_COL, NUMERIC_COLUMNS
from .payroll_rules import RULE_COLUMNS
from .privacy import mask_ids

# column -> (used for detection?, reason)
COLUMN_USAGE: Dict[str, tuple] = {
    "payroll_item_id": ("No (join key only)", "Unique row id. The 'PI-D' prefix reveals the double-payment label, so it must never be a feature."),
    "run_id": ("No", "Repeats the month (PR-YYYY-MM); identifier only."),
    "employee_id": ("Key only", "Groups payslips for the duplicate rule and for previous-month history. Not a model feature; masked in saved outputs."),
    "month": ("Key only", "Orders history, gives days-in-month, defines the duplicate pay period. Not a model feature."),
    "department_id": ("No", "Category, not used in v1; role_id is the pay reference."),
    "role_id": ("Reference key", "Groups the previous-calendar-month median for fixed_gross_vs_role_median. Not a model feature itself."),
    "job_level": ("No", "Not needed; role median already captures grade."),
    "overtime_eligible": ("Rule input", "Overtime for a non-eligible role is a rule violation."),
    "monthly_fixed_gross": ("Yes", "Denominator/input of gross_to_fixed_gross, role and history features, gross check."),
    "payable_days": ("Rule input", "Earned fixed pay = fixed * payable_days / days_in_month; zero-pay rule context."),
    "lop_days": ("Validity only", "Loss-of-pay days; checked for invalid values."),
    "overtime_hours_paid": ("Yes", "Model feature and overtime-cap rule."),
    "overtime_pay": ("Yes", "Expected-gross check and overtime_pay_share."),
    "variable_pay": ("Rule input", "Part of expected gross."),
    "variable_pay_type": ("No", "Description of the bonus; blank simply means no bonus (not an error)."),
    "gross_pay": ("Yes", "Numerator/denominator of most ratios and the gross reconciliation rule."),
    "pf_employee": ("Yes", "pf_vs_prev_month and missed-PF rule."),
    "esi_employee": ("Validity only", "Part of total_deductions; checked for invalid values."),
    "professional_tax": ("Validity only", "Part of total_deductions; checked for invalid values."),
    "tds": ("Validity only", "Part of total_deductions; checked for invalid values."),
    "other_deductions": ("Validity only", "Part of total_deductions; checked for invalid values."),
    "total_deductions": ("Yes", "Numerator of deduction_ratio."),
    "net_pay": ("Yes", "Numerator of net_pay_ratio; negative/zero net rules."),
}


def build_data_dictionary(raw: pd.DataFrame, workbook_dictionary: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Column | Meaning | Data type | Missing | Used for detection? | Reason."""
    meaning = {}
    if workbook_dictionary is not None:
        sub = workbook_dictionary[workbook_dictionary["sheet"] == "payroll_monthly_features"]
        meaning = dict(zip(sub["column"], sub["description"]))
    rows = []
    for c in raw.columns:
        used, why = COLUMN_USAGE.get(c, ("No", "Not reviewed"))
        rows.append({
            "Column": c, "Meaning": meaning.get(c, ""), "Data type": str(raw[c].dtype),
            "Missing": int(raw[c].isna().sum()), "Used for detection?": used, "Reason": why,
        })
    return pd.DataFrame(rows)


def feature_definition_table() -> pd.DataFrame:
    return pd.DataFrame([{"Feature": k, "Formula": v["formula"], "Why it helps": v["why"]}
                         for k, v in FEATURE_DEFINITIONS.items()])


def data_quality_summary(raw: pd.DataFrame, feats: pd.DataFrame) -> Dict[str, int]:
    """Label-free data-quality facts used in the report."""
    dup_groups = feats.duplicated([EMPLOYEE_COL, MONTH_COL], keep=False)
    return {
        "rows": int(len(raw)),
        "employees": int(raw[EMPLOYEE_COL].nunique()),
        "months": int(raw[MONTH_COL].nunique()),
        "duplicate_item_ids": int(raw[ID_COL].duplicated().sum()),
        "invalid_numeric_values": int(feats[NUMERIC_COLUMNS].isna().sum().sum() - raw[NUMERIC_COLUMNS].isna().sum().sum()),
        "missing_numeric_values": int(raw[NUMERIC_COLUMNS].isna().sum().sum()),
        "rows_in_employee_month_duplicate_groups": int(dup_groups.sum()),
        "negative_net_rows": int((feats["net_pay"] < 0).sum()),
        "zero_net_rows": int((feats["net_pay"] == 0).sum()),
        "zero_gross_rows": int((feats["gross_pay"] == 0).sum()),
    }


# --------------------------------------------------------------------------
# Diagnostics (explanation only - they never change a rule or the model)
# --------------------------------------------------------------------------
def diagnostics(feats: pd.DataFrame, flags: pd.DataFrame, labels: pd.DataFrame) -> Dict:
    f = feats.copy()
    f.index = f[ID_COL].values
    fl = flags.copy()
    fl.index = f.index
    f["label"] = labels.set_index(ID_COL)[LABEL_TYPE_COL].reindex(f.index)
    out: Dict = {}

    # salary-drop flags: how many are the month directly after a ROLE_MISMATCH row of the same employee?
    drops = f[fl["rule_salary_drop"] == 1][[EMPLOYEE_COL, MONTH_COL, "label"]]
    rm = f[f["label"] == "ROLE_MISMATCH"][[EMPLOYEE_COL, MONTH_COL]].copy()
    rm[MONTH_COL] = rm[MONTH_COL] + pd.offsets.MonthBegin(1)
    after = drops.merge(rm, on=[EMPLOYEE_COL, MONTH_COL], how="inner")
    out["salary_drop_flags"] = int(len(drops))
    out["salary_drop_flags_labelled"] = int(drops["label"].notna().sum())
    out["salary_drop_flags_month_after_role_mismatch"] = int(len(after))

    # Isolation Forest false alarms: share with a bonus/variable pay vs overall
    clean_if = f[(fl["if_flag"] == 1) & f["label"].isna()]
    out["if_false_alarms"] = int(len(clean_if))
    out["if_false_alarm_share_with_variable_pay"] = float((clean_if["variable_pay"] > 0).mean()) if len(clean_if) else np.nan
    out["overall_share_with_variable_pay"] = float((f["variable_pay"] > 0).mean())

    # rows missed by the combined detector
    missed = f[(fl["combined_flag"] == 0) & f["label"].notna()].copy()
    cols = ["label", "overtime_hours_paid", "overtime_eligible", "pf_employee", "prev_pf_employee",
            "fixed_gross_vs_role_median", "gross_vs_expected"]
    missed_out = missed[cols].copy()
    # employee's other months' overtime range for missed overtime spikes
    own_max, own_med = [], []
    for pid, row in missed.iterrows():
        others = f[(f[EMPLOYEE_COL] == row[EMPLOYEE_COL]) & (f.index != pid)]["overtime_hours_paid"]
        own_max.append(others.max() if len(others) else np.nan)
        own_med.append(others.median() if len(others) else np.nan)
    missed_out["employee_other_months_max_ot_hours"] = own_max
    missed_out["employee_other_months_median_ot_hours"] = own_med
    clean_ot = f[f["label"].isna() & (f["overtime_hours_paid"] > 0)]["overtime_hours_paid"]
    out["clean_rows_with_overtime"] = int(len(clean_ot))
    out["clean_overtime_hours_median"] = float(clean_ot.median()) if len(clean_ot) else np.nan
    out["clean_overtime_hours_max"] = float(clean_ot.max()) if len(clean_ot) else np.nan
    missed_out.insert(0, "record_ref", mask_ids(pd.Series(missed_out.index, index=missed_out.index), "REC").values)
    out["missed_rows"] = missed_out.reset_index(drop=True)

    # stale-salary mechanism
    st = f[f["label"] == "STALE_SALARY"]
    out["stale_gross_vs_expected_min"] = float(st["gross_vs_expected"].min()) if len(st) else np.nan
    out["stale_gross_vs_expected_max"] = float(st["gross_vs_expected"].max()) if len(st) else np.nan
    out["stale_fixed_gross_change_median"] = float(st["fixed_gross_change_vs_prev"].median()) if len(st) else np.nan
    return out


def masked_examples(feats: pd.DataFrame, flags: pd.DataFrame, labels: pd.DataFrame, per_type: int = 2) -> pd.DataFrame:
    """The `per_type` earliest (by month, then item id) rules-caught-but-model-missed payslips of each
    error type - deterministic and independent of row order, not cherry-picked. Identifiers are masked."""
    f = feats.copy()
    f.index = f[ID_COL].values
    fl = flags.copy()
    fl.index = f.index
    lab = labels.set_index(ID_COL)[LABEL_TYPE_COL].reindex(f.index)
    sel = (fl["rules_only_flag"] == 1) & (fl["if_flag"] == 0) & lab.notna()
    ex = f[sel].assign(error_type=lab[sel]).join(fl[RULE_COLUMNS + ["detection_reasons"]], rsuffix="_flag")
    ex = ex.sort_values([MONTH_COL, ID_COL]).groupby("error_type", sort=True).head(per_type)   # earliest months first: independent of row order
    ex = ex.sort_values(["error_type", MONTH_COL, ID_COL])
    out = pd.DataFrame({
        "record_ref": mask_ids(ex[ID_COL], "REC").values,
        "employee_ref": mask_ids(ex[EMPLOYEE_COL], "EMP").values,
        "month": ex[MONTH_COL].dt.strftime("%Y-%m").values,
        "error_type_label": ex["error_type"].values,
        "gross_pay": ex["gross_pay"].values,
        "net_pay": ex["net_pay"].values,
        "gross_vs_expected": ex["gross_vs_expected"].round(3).values,
        "fixed_gross_vs_role_median": ex["fixed_gross_vs_role_median"].round(3).values,
        "detection_reasons": ex["detection_reasons"].values,
    })
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------
def _fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}f}"
    return str(v)


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], digits).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


TYPE_NOTES = {
    "DOUBLE_PAYMENT": "an exact repeat of a normal payslip - every ratio is ordinary, so only a same-employee/same-month check can see it; a person sees the same payslip twice at once.",
    "WRONG_DECIMAL": "gross/net are ~10x too high, breaking the 'gross = fixed + overtime + variable' identity and the gross-to-fixed ratio.",
    "MISSED_DEDUCTION": "PF is 0; invisible without knowing the employee normally pays PF (first-month cases have no history).",
    "ROLE_MISMATCH": "fixed pay roughly 3x the role norm and a large jump over the previous month.",
    "OVERTIME_SPIKE": "overtime is inflated, but only some spikes exceed the 50-hour policy cap; below the cap they resemble ordinary overtime.",
    "STALE_SALARY": "gross and net are 12% too low while fixed gross is unchanged; a statistical outlier detector sees an ordinary-looking ratio, but the identity check does not.",
    "NEGATIVE_NET": "net pay below zero - an explicit rule and an extreme net_pay_ratio.",
    "ZERO_PAY": "gross and net exactly 0 despite payable days - an explicit rule; only 5 cases, so recall figures rest on very few records.",
}


def hardest_types(per_type: pd.DataFrame, approach: str) -> pd.DataFrame:
    sub = per_type[(per_type["approach"] == approach) & (per_type["error_type"] != ALL_ERRORS)]
    return sub[sub["recall"] == sub["recall"].min()].sort_values("error_type")
