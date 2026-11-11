"""Task 7 - end to end: Task 6 + Task 5 detections -> explained, ranked, official-format output."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from cortexhr_task5.attendance_anomaly_detector import run_detector
from cortexhr_task6.detection import run_detection
from cortexhr_task6.payroll_rules import RuleConfig
from cortexhr_task6.payroll_model import ModelConfig
from cortexhr_task6.privacy import mask_ids

from .explain import ATTENDANCE_SOURCE, PAYROLL_SOURCE, explain_attendance, explain_payroll
from .official_output import DEFAULT_SCORED_AT, to_official, validate_official_values
from .ranking import rank_flags

DETAIL_COLUMNS = ["result_id", "source_dataset", "record_id", "employee_ref", "period", "primary_reason_code",
                  "all_reason_codes", "signal_count", "rule_count", "model_flag", "anomaly_score", "anomaly_score_raw",
                  "money_at_risk", "money_basis", "candidate_amounts", "reason_text"]


def assemble(payroll_expl: pd.DataFrame, attendance_expl: pd.DataFrame, scored_at: str = DEFAULT_SCORED_AT
             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combine explained records, rank them, and return (official output, internal detail report)."""
    ranked = rank_flags(pd.concat([payroll_expl, attendance_expl], ignore_index=True))
    official = to_official(ranked, scored_at)
    problems = validate_official_values(official)
    if problems:
        raise ValueError(f"official output failed validation: {problems}")
    detail = ranked.copy()
    detail.insert(0, "result_id", official["result_id"].to_numpy())
    detail["employee_ref"] = mask_ids(detail["employee_id"], "EMP").to_numpy()
    detail["period"] = pd.to_datetime(detail["period"]).dt.strftime("%Y-%m-%d")
    return official, detail[DETAIL_COLUMNS]


def verify_integrity(official: pd.DataFrame, payroll_raw: pd.DataFrame, attendance_raw: pd.DataFrame,
                     expected_counts: Dict[str, int]) -> None:
    """No flag silently dropped, none invented, every record_id/employee_id/period matches its source row."""
    for source, raw, id_col in ((PAYROLL_SOURCE, payroll_raw, "payroll_item_id"),
                                (ATTENDANCE_SOURCE, attendance_raw, "attendance_feature_id")):
        out = official[official["source_dataset"] == source]
        if len(out) != expected_counts[source]:
            raise ValueError(f"{source}: {len(out)} output rows but {expected_counts[source]} flags were produced (a flag was dropped or invented)")
        if out["record_id"].duplicated().any():
            raise ValueError(f"{source}: a record appears more than once")
        src = raw.drop_duplicates(id_col).set_index(id_col)
        unknown = sorted(set(out["record_id"]) - set(src.index))
        if unknown:
            raise ValueError(f"{source}: {len(unknown)} output record_id(s) not found in the source data, e.g. {unknown[:3]}")
        joined = src.loc[out["record_id"]]
        if not (joined["employee_id"].to_numpy() == out["employee_id"].to_numpy()).all():
            raise ValueError(f"{source}: employee_id does not match the source record")
        src_period = pd.to_datetime(joined["month"]).dt.strftime("%Y-%m-%d").to_numpy()
        if not (src_period == out["period"].to_numpy()).all():
            raise ValueError(f"{source}: period does not match the source record")


def build_official_output(payroll_raw: pd.DataFrame, attendance_raw: pd.DataFrame, scored_at: str = DEFAULT_SCORED_AT,
                          rule_cfg: Optional[RuleConfig] = None, model_cfg: Optional[ModelConfig] = None
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Run the existing Task 6 (payroll) and Task 5 (attendance) detectors UNCHANGED and explain their flags.

    Returns (official output, detail report, run info). No labels are read and no model is refitted for the
    explanations: the explanations only read the detectors' flags and the payroll/attendance columns.
    """
    feats6, flags6, _ = run_detection(payroll_raw, rule_cfg, model_cfg)
    feats5, flags5, _ranked5, _det = run_detector(attendance_raw)
    payroll_expl = explain_payroll(feats6, flags6, rule_cfg)
    attendance_expl = explain_attendance(feats5, flags5)
    official, detail = assemble(payroll_expl, attendance_expl, scored_at)
    counts = {PAYROLL_SOURCE: int(flags6["combined_flag"].sum()), ATTENDANCE_SOURCE: int(flags5["combined_anomaly"].sum())}
    verify_integrity(official, payroll_raw, attendance_raw, counts)
    return official, detail, {"expected_counts": counts, "payroll_rows": len(payroll_raw), "attendance_rows": len(attendance_raw)}
