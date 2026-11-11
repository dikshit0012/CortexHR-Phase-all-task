"""Task 7 - independent verification of the saved outputs.

Recomputes every payroll `money_at_risk` from the RAW payslips with plain pandas (it does not import the
Task 7 code), and checks identifiers, schema, ranking and the 20-row sample.
Exit code 0 = everything reconciles, 1 = at least one problem.

Usage:
    python "task07/task 7/scripts/verify_task7_reports.py" [reports_dir] [workbook] [--sample-size 20]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["result_id", "source_dataset", "record_id", "employee_id", "period", "anomaly_score", "flagged",
           "reason_code", "reason_text", "money_at_risk", "model_version", "scored_at"]
PAY, ATT = "payroll_monthly_features", "attendance_monthly_features"
NO_AMOUNT = {"STATISTICAL_OUTLIER", "INVALID_PAYROLL_DATA", "NO_ATTENDANCE", "LONG_UNPAID_ABSENCE", "IMPLAUSIBLE_HOURS", "PATTERN_SHIFT"}
CAP = 50.0
REASON_PRIORITY = [
    "DUPLICATE_PAYMENT", "NEGATIVE_NET_PAY", "ZERO_PAY", "GROSS_ABOVE_NORM", "GROSS_BELOW_EXPECTED",
    "OVERTIME_ABOVE_CAP", "OVERTIME_NOT_ELIGIBLE", "PF_NOT_DEDUCTED", "PAY_ABOVE_ROLE_NORM", "SALARY_JUMP",
    "SALARY_DROP", "INVALID_PAYROLL_DATA", "NO_ATTENDANCE", "LONG_UNPAID_ABSENCE", "IMPLAUSIBLE_HOURS",
    "STATISTICAL_OUTLIER", "PATTERN_SHIFT",
]
REASON_SOURCE = {
    **{code: "payroll" for code in REASON_PRIORITY[:12]},
    **{code: "attendance" for code in REASON_PRIORITY[12:15]},
    "STATISTICAL_OUTLIER": "payroll",
    "PATTERN_SHIFT": "attendance",
}
DETAIL_COLUMNS = {
    "result_id", "source_dataset", "record_id", "primary_reason_code", "signal_count",
    "rule_count", "model_flag", "anomaly_score_raw", "anomaly_score",
}


def _find_workbook(explicit=None):
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    for p in [explicit, repo_root / "task05" / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx",
              repo_root / "task06" / "task 6" / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"]:
        if p and Path(p).exists():
            return Path(p)
    raise FileNotFoundError("workbook not found; pass its path as the second argument")


def expected_amounts(raw: pd.DataFrame, official: pd.DataFrame) -> pd.Series:
    """Independent re-computation of money_at_risk for every payroll row of `official` (index = record_id)."""
    raw = raw.copy()
    raw["month"] = pd.to_datetime(raw["month"])
    raw["period"] = raw["month"].dt.to_period("M")
    raw["dim"] = raw["month"].dt.days_in_month
    raw["expected_gross"] = raw["monthly_fixed_gross"] * raw["payable_days"] / raw["dim"] + raw["overtime_pay"] + raw["variable_pay"]
    raw["ratio"] = (raw["total_deductions"] / raw["gross_pay"].where(raw["gross_pay"] != 0))
    global_share = raw["ratio"].median()
    by_month = raw.groupby(["employee_id", "period"]).agg(fixed=("monthly_fixed_gross", "median"), pf=("pf_employee", "max"),
                                                          ratio=("ratio", "median"))
    role_median = raw.groupby("role_id")["monthly_fixed_gross"].median()
    pay = official[official["source_dataset"] == PAY].set_index("record_id")
    src = raw.drop_duplicates("payroll_item_id").set_index("payroll_item_id")
    out = {}
    dup_counts = pay[pay["reason_code"] == "DUPLICATE_PAYMENT"].groupby(["employee_id", "period"]).size()
    for rid, o in pay.iterrows():
        if rid not in src.index:                       # reported as "not in source" by verify(); never crash on it
            out[rid] = np.nan
            continue
        r, code = src.loc[rid], o["reason_code"]
        per = r["month"].to_period("M")
        prev = by_month.loc[(r["employee_id"], per - 1)] if (r["employee_id"], per - 1) in by_month.index else None
        prorate = r["payable_days"] / r["dim"]
        if code in NO_AMOUNT:
            out[rid] = np.nan
            continue
        if code == "GROSS_ABOVE_NORM":
            v = r["gross_pay"] - r["expected_gross"]
        elif code == "GROSS_BELOW_EXPECTED":
            v = r["expected_gross"] - r["gross_pay"]
        elif code == "OVERTIME_ABOVE_CAP":
            v = r["overtime_pay"] * (r["overtime_hours_paid"] - CAP) / r["overtime_hours_paid"]
        elif code == "OVERTIME_NOT_ELIGIBLE":
            v = r["overtime_pay"]
        elif code == "PF_NOT_DEDUCTED":
            v = prev["pf"] if prev is not None and prev["pf"] > 0 else np.nan
        elif code == "PAY_ABOVE_ROLE_NORM":
            v = (r["monthly_fixed_gross"] - role_median[r["role_id"]]) * prorate
        elif code == "SALARY_JUMP":
            v = (r["monthly_fixed_gross"] - prev["fixed"]) * prorate if prev is not None else np.nan
        elif code == "SALARY_DROP":
            v = (prev["fixed"] - r["monthly_fixed_gross"]) * prorate if prev is not None else np.nan
        elif code == "DUPLICATE_PAYMENT":
            same = raw[(raw["employee_id"] == r["employee_id"]) & (raw["month"] == r["month"]) & (raw["gross_pay"] == r["gross_pay"])
                       & (raw["total_deductions"] == r["total_deductions"]) & (raw["net_pay"] == r["net_pay"])]
            flagged = int(dup_counts.get((o["employee_id"], o["period"]), 1))
            v = (len(same) - 1) * max(r["net_pay"], 0) / flagged
        elif code in ("NEGATIVE_NET_PAY", "ZERO_PAY"):
            hist = by_month.loc[r["employee_id"]]
            hist = hist[hist.index < per]["ratio"].dropna()
            share = hist.median() if len(hist) else global_share
            share = share if 0 <= share < 1 else np.nan
            if code == "NEGATIVE_NET_PAY":
                excess = r["total_deductions"] - share * max(r["gross_pay"], 0)
                cands = [x for x in (excess, -r["net_pay"]) if pd.notna(x)]
                v = max(cands) if cands else np.nan
            else:
                v = r["expected_gross"] * (1 - share) - max(r["net_pay"], 0) if r["expected_gross"] > 0 else np.nan
        else:
            v = np.nan
        out[rid] = round(max(v, 0), 2) if pd.notna(v) and np.isfinite(v) else np.nan
    return pd.Series(out)


def _expected_sample_ids(official: pd.DataFrame, sample_size: int) -> list:
    """Rebuild the documented reason-coverage sample, then fill it in ranking order."""
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if len(official) <= sample_size:
        return list(official["result_id"])

    first_by_code = official.groupby("reason_code", sort=False)["result_id"].first()
    chosen = [first_by_code[code] for code in REASON_PRIORITY if code in first_by_code.index][:sample_size]
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < sample_size:
        with_amount = official[
            official["money_at_risk"].notna() & ~official["result_id"].isin(chosen)
        ]
        chosen.extend(with_amount["result_id"].tolist()[: sample_size - len(chosen)])
    if len(chosen) < sample_size:
        remaining = official[~official["result_id"].isin(chosen)]
        chosen.extend(remaining["result_id"].tolist()[: sample_size - len(chosen)])
    return sorted(chosen)


def _check_ranking(official: pd.DataFrame, detail: pd.DataFrame) -> list:
    """Check all documented tie-breaks using detail values reconciled to official rows."""
    problems = []
    priority = pd.Series({code: i for i, code in enumerate(REASON_PRIORITY)})
    missing_priorities = set(official["reason_code"]) - set(priority.index)
    if missing_priorities:
        return [f"reason codes missing from the priority dictionary: {sorted(missing_priorities)}"]

    if detail["result_id"].duplicated().any():
        return ["detail report has duplicate result_id values"]
    if set(detail["result_id"]) != set(official["result_id"]):
        return ["detail report result_id values do not match the official output"]

    joined = official[
        ["result_id", "source_dataset", "record_id", "reason_code", "anomaly_score", "money_at_risk"]
    ].merge(
        detail[list(DETAIL_COLUMNS)], on="result_id", how="left", suffixes=("_o", "_d"), indicator=True,
    )
    if (joined["_merge"] != "both").any():
        return ["detail report is missing one or more official result_id values"]
    for out_col, detail_col in (
        ("source_dataset_o", "source_dataset_d"),
        ("record_id_o", "record_id_d"),
        ("reason_code", "primary_reason_code"),
    ):
        if not (joined[out_col].astype(str).to_numpy() == joined[detail_col].astype(str).to_numpy()).all():
            problems.append(f"detail report does not match official {out_col[:-2]}")
    detail_score = pd.to_numeric(joined["anomaly_score_d"], errors="coerce")
    official_score = pd.to_numeric(joined["anomaly_score_o"], errors="coerce")
    if detail_score.isna().any() or not np.allclose(detail_score, official_score, atol=0.0005, rtol=0):
        problems.append("detail report anomaly_score does not match the official output")

    signal_count = pd.to_numeric(joined["signal_count"], errors="coerce")
    rule_count = pd.to_numeric(joined["rule_count"], errors="coerce")
    model_flag = pd.to_numeric(joined["model_flag"], errors="coerce")
    raw_score = pd.to_numeric(joined["anomaly_score_raw"], errors="coerce")
    if signal_count.isna().any() or rule_count.isna().any() or model_flag.isna().any():
        return problems + ["detail report has invalid signal counts"]
    if not (signal_count == rule_count + model_flag).all():
        problems.append("detail report signal_count does not equal rule_count plus model_flag")
    if raw_score.isna().any() or not np.isfinite(raw_score).all():
        return problems + ["detail report has invalid raw anomaly scores"]

    joined["_money"] = pd.to_numeric(joined["money_at_risk"], errors="coerce")
    joined["_priority"] = joined["reason_code"].map(priority)
    joined["_signals"] = signal_count.to_numpy()
    joined["_raw_score"] = raw_score.to_numpy()
    expected_order = joined.sort_values(
        ["_money", "_priority", "_signals", "_raw_score", "record_id_o"],
        ascending=[False, True, False, False, True],
        na_position="last",
        kind="mergesort",
    )["result_id"].tolist()
    if expected_order != official["result_id"].tolist():
        problems.append("rows do not follow the complete documented ranking and tie-break order")
    return problems


def verify(reports_dir, workbook=None, sample_size=20) -> list:
    d = Path(reports_dir)
    wb = _find_workbook(workbook)
    official = pd.read_csv(d / "task7_anomaly_results.csv")
    sample = pd.read_csv(d / "task7_sample_20_flags.csv")
    detail = pd.read_csv(d / "task7_flag_details.csv")
    dictionary = pd.read_csv(d / "task7_reason_code_dictionary.csv")
    raw_pay = pd.read_excel(wb, sheet_name=PAY)
    raw_att = pd.read_excel(wb, sheet_name=ATT)
    problems = []

    sheet_cols = list(pd.read_excel(wb, sheet_name="sample_anomaly_results", nrows=0).columns)
    if sheet_cols != COLUMNS:
        problems.append("the sample sheet header differs from the columns this verifier expects")
    for name, df in (("task7_anomaly_results.csv", official), ("task7_sample_20_flags.csv", sample)):
        if list(df.columns) != sheet_cols:
            problems.append(f"{name}: columns differ from the sample sheet (names or order)")
    if not DETAIL_COLUMNS <= set(detail.columns):
        problems.append(f"task7_flag_details.csv: missing required columns {sorted(DETAIL_COLUMNS - set(detail.columns))}")
    if not {"reason_code", "priority", "source"} <= set(dictionary.columns):
        problems.append("task7_reason_code_dictionary.csv: missing reason_code, priority or source")
    if problems:
        return problems

    valid_codes = set(dictionary["reason_code"])
    if dictionary["reason_code"].duplicated().any():
        problems.append("reason-code dictionary contains duplicate codes")
    if valid_codes != set(REASON_PRIORITY):
        problems.append("reason-code dictionary does not match the verifier's documented vocabulary")
    expected_priorities = {code: i + 1 for i, code in enumerate(REASON_PRIORITY)}
    actual_priorities = pd.to_numeric(
        dictionary.drop_duplicates("reason_code").set_index("reason_code")["priority"], errors="coerce"
    ).to_dict()
    if actual_priorities != expected_priorities:
        problems.append("reason-code dictionary priorities do not match the documented order")
    if not set(official["reason_code"]) <= valid_codes:
        problems.append(f"unknown reason codes: {set(official['reason_code']) - valid_codes}")
    known_sources = {PAY, ATT}
    if not set(official["source_dataset"]) <= known_sources:
        problems.append(f"unknown source datasets: {set(official['source_dataset']) - known_sources}")
    else:
        expected_source = official["reason_code"].map(REASON_SOURCE)
        if not (expected_source.to_numpy() == official["source_dataset"].map(
            {PAY: "payroll", ATT: "attendance"}
        ).to_numpy()).all():
            problems.append("a reason code is assigned to the wrong source dataset")
    if official["reason_text"].isna().any() or (official["reason_text"].astype(str).str.strip() == "").any():
        problems.append("empty reason_text")
    if not (official["flagged"] == "Y").all():
        problems.append("flagged is not Y everywhere")
    if official["record_id"].duplicated().any():
        problems.append("duplicate record_id")

    # identifiers exist in the source and match employee and period
    for src_name, raw, id_col in ((PAY, raw_pay, "payroll_item_id"), (ATT, raw_att, "attendance_feature_id")):
        part = official[official["source_dataset"] == src_name]
        r = raw.drop_duplicates(id_col).set_index(id_col)
        missing = set(part["record_id"]) - set(r.index)
        if missing:
            problems.append(f"{src_name}: {len(missing)} record_id not in source")
            continue
        j = r.loc[part["record_id"]]
        if not (j["employee_id"].to_numpy() == part["employee_id"].to_numpy()).all():
            problems.append(f"{src_name}: employee_id mismatch")
        if not (pd.to_datetime(j["month"]).dt.strftime("%Y-%m-%d").to_numpy() == part["period"].to_numpy()).all():
            problems.append(f"{src_name}: period mismatch")

    # amounts recomputed independently
    exp = expected_amounts(raw_pay, official)
    got = official[official["source_dataset"] == PAY].set_index("record_id")["money_at_risk"]
    both_nan = exp.isna() & got.reindex(exp.index).isna()
    diff = (exp - got.reindex(exp.index)).abs()
    bad = exp.index[~both_nan & ~(diff <= 0.011)]
    if len(bad):
        problems.append(f"{len(bad)} payroll amount(s) differ from the independent recomputation, e.g. {list(bad[:3])}")
    if (official.loc[official["source_dataset"] == ATT, "money_at_risk"].notna()).any():
        problems.append("attendance rows must not carry a money estimate")
    if (pd.to_numeric(official["money_at_risk"]) < 0).any():
        problems.append("negative money_at_risk")

    # ranking: largest first, empty last, then every documented deterministic tie-break
    money = official["money_at_risk"]
    n = int(money.notna().sum())
    if not (money.iloc[:n].notna().all() and money.iloc[:n].is_monotonic_decreasing and money.iloc[n:].isna().all()):
        problems.append("rows are not ranked by money_at_risk (largest first, empty last)")
    if list(official["result_id"]) != [f"AN-{i:04d}" for i in range(1, len(official) + 1)]:
        problems.append("result_id is not sequential in ranking order")
    problems.extend(_check_ranking(official, detail))

    # sample: exact deterministic selection, not merely a subset of real rows
    merged = sample.merge(official, on="result_id", suffixes=("_s", "_o"), how="left", indicator=True)
    if (merged["_merge"] != "both").any():
        problems.append("sample contains rows that are not in the official output")
    else:
        for col in ("record_id", "reason_code", "reason_text", "money_at_risk", "anomaly_score"):
            a, b = merged[f"{col}_s"], merged[f"{col}_o"]
            if not ((a == b) | (a.isna() & b.isna())).all():
                problems.append(f"sample column {col} differs from the official output")
        if not sample["result_id"].is_monotonic_increasing:
            problems.append("sample is not in ranking order")
    try:
        expected_sample = _expected_sample_ids(official, sample_size)
        if sample["result_id"].tolist() != expected_sample:
            problems.append(
                f"sample does not match the documented {sample_size}-row selection rule"
            )
    except ValueError as exc:
        problems.append(str(exc))
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independently verify Task 7 reports.")
    parser.add_argument("reports_dir", nargs="?", default=Path(__file__).resolve().parents[1] / "reports")
    parser.add_argument("workbook", nargs="?", default=None)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()
    issues = verify(args.reports_dir, args.workbook, args.sample_size)
    if issues:
        print("PROBLEMS FOUND:")
        for i in issues:
            print(" -", i)
        sys.exit(1)
    print(f"OK: outputs in {target} reconcile with an independent recomputation (schema, ids, amounts, ranking, sample).")
