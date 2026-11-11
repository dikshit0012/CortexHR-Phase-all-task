"""Scoring service: runs the EXISTING Task 5 / 6 detectors (already fitted, loaded once) and the EXISTING Task 7
explanation, money-at-risk, ranking and official-output code. Nothing is refitted and no Task 5-7 logic is copied.

How history is handled (reference-workbook design)
    The Task 5 / 6 features need earlier months of the same employee, the previous month's role median, other
    payslips of the same employee-month (duplicates) and a dataset-wide median deduction share. For every request:
        combined = reference rows (minus any row whose id is being re-scored) + the request rows
    The Task 5 / 6 / 7 functions are run on `combined`, and only the request rows are reported. Explanations are
    computed over ALL rows exactly as Task 7 does (its money-at-risk logic looks at other rows of the same
    employee-month, e.g. the other copies of a duplicate payslip, so a duplicate splits its amount over the copies
    and the same money is never counted twice). The fitted forest only PREDICTS (`predict`, never `fit`). A request row whose id already exists in the reference REPLACES that
    reference row, so a corrected record can be re-scored.

How scores and ranking stay consistent across requests
    * anomaly_score: Task 7 min-max scales the raw forest score over all scored rows. Here the min and max of the
      reference scoring are stored in the artifact and used for every request (`ScoreScale`), so the same record
      gets the same score alone, in a batch, or in another request. Unseen rows are clipped to 0..1.
    * ranking: Task 7 `rank_flags` (money at risk, reason priority, signal count, raw score, record_id) is applied
      to the flagged records of the response; result_id AN-0001.. follows that rank. Un-flagged records follow
      after them, ordered by raw score then record_id. `rank_flags` uses the RAW score, which does not depend on
      the scaling. The list itself is returned in INPUT order; sort by result_id to get the ranking.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401
from cortexhr_task5.attendance_anomaly_detector import (
    ID_COL as ATTENDANCE_ID, apply_rules as apply_attendance_rules, build_feature_frame,
    combine_flags as combine_attendance_flags,
)
from cortexhr_task6.detection import combine_flags as combine_payroll_flags
from cortexhr_task6.payroll_features import ID_COL as PAYROLL_ID, build_payroll_features
from cortexhr_task6.payroll_rules import apply_rules as apply_payroll_rules
from cortexhr_task7.explain import ATTENDANCE_SOURCE, PAYROLL_SOURCE, explain_attendance, explain_payroll
from cortexhr_task7.official_output import MODEL_VERSIONS, to_official, validate_official_values
from cortexhr_task7.ranking import rank_flags

from .artifacts import ATTENDANCE, PAYROLL, LoadedDetector
from .reference import ReferenceData


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _none_if_nan(x: Any) -> Optional[float]:
    return None if x is None or pd.isna(x) else float(x)


class AnomalyService:
    """Holds the loaded detectors and the reference history. Read-only after construction (safe across requests)."""

    def __init__(self, attendance: LoadedDetector, payroll: LoadedDetector, reference: ReferenceData,
                 max_batch_size: int, clock: Callable[[], datetime] = _utc_now):
        self.attendance = attendance
        self.payroll = payroll
        self.reference = reference
        self.max_batch_size = max_batch_size
        self._clock = clock

    # ------------------------------------------------------------------ info
    def info(self) -> Dict[str, Any]:
        """Safe metadata for /v1/models: versions and counts only - no paths, no employee data."""
        def one(d: LoadedDetector, n_ref: int) -> Dict[str, Any]:
            return {"artifact_version": d.version, "response_model_version": MODEL_VERSIONS[
                        ATTENDANCE_SOURCE if d.kind == ATTENDANCE else PAYROLL_SOURCE],
                    "created_at": d.metadata["created_at"], "reference_rows": n_ref,
                    "model_features": d.metadata["kept_features"], "sklearn_version": d.metadata["sklearn_version"],
                    "dependency_versions": {name: d.metadata[f"{name}_version"] for name in
                                             ("python", "numpy", "pandas", "scipy", "sklearn", "joblib", "openpyxl")},
                    "score_scale": d.metadata["score_scale"]}
        return {"max_batch_size": self.max_batch_size,
                "detectors": {ATTENDANCE: one(self.attendance, len(self.reference.attendance)),
                              PAYROLL: one(self.payroll, len(self.reference.payroll))}}

    # --------------------------------------------------------------- scoring
    def score_attendance(self, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        req = self._request_frame(records, ATTENDANCE_ID)
        ids = set(req[ATTENDANCE_ID])
        ref = self.reference.attendance
        combined = pd.concat([ref[~ref[ATTENDANCE_ID].isin(ids)], req], ignore_index=True)

        feats = build_feature_frame(combined)                                       # Task 5 features (history from reference)
        model_out = self.attendance.detector.predict(feats)                         # predict only - never fit
        flags = combine_attendance_flags(model_out, apply_attendance_rules(feats))  # Task 5 rules + combined flag
        is_req = feats[ATTENDANCE_ID].isin(ids)
        expl = explain_attendance(feats, flags)                                     # Task 7 reasons / text / money (NaN)
        expl = expl[expl["record_id"].isin(ids)]                                    # report the request rows only
        return self._assemble(ATTENDANCE_SOURCE, ATTENDANCE_ID, list(req[ATTENDANCE_ID]), feats, flags,
                              "combined_anomaly", is_req, expl, self.attendance)

    def score_payroll(self, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        req = self._request_frame(records, PAYROLL_ID)
        ids = set(req[PAYROLL_ID])
        ref = self.reference.payroll
        combined = pd.concat([ref[~ref[PAYROLL_ID].isin(ids)], req], ignore_index=True)

        feats = build_payroll_features(combined)                                    # Task 6 features (history from reference)
        model_out = self.payroll.detector.predict(feats)
        flags = combine_payroll_flags(model_out, apply_payroll_rules(feats, None))  # Task 6 rules + combined flag
        is_req = feats[PAYROLL_ID].isin(ids)
        expl = explain_payroll(feats, flags, None)                                  # Task 7 reasons / text / money at risk
        expl = expl[expl["record_id"].isin(ids)]                                    # report the request rows only
        return self._assemble(PAYROLL_SOURCE, PAYROLL_ID, list(req[PAYROLL_ID]), feats, flags,
                              "combined_flag", is_req, expl, self.payroll)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _request_frame(records: Sequence[Mapping[str, Any]], id_col: str) -> pd.DataFrame:
        if not records:
            raise ValueError("no records to score")
        req = pd.DataFrame(list(records))
        req["month"] = pd.to_datetime(req["month"])
        if req[id_col].duplicated().any():
            raise ValueError(f"duplicate {id_col} in one request")
        return req

    def _assemble(self, source: str, id_col: str, req_order: List[str], feats: pd.DataFrame, flags: pd.DataFrame,
                  flag_col: str, is_req: pd.Series, expl: pd.DataFrame, det: LoadedDetector) -> List[Dict[str, Any]]:
        scored_at = self._clock().strftime("%Y-%m-%d %H:%M:%S")
        by_id: Dict[str, Dict[str, Any]] = {}
        next_n = 1

        # --- flagged records: Task 7 explanation -> Task 7 ranking -> Task 7 official output
        if len(expl):
            expl = expl.copy()
            expl["anomaly_score"] = det.scale.apply(expl["anomaly_score_raw"])      # fixed reference scale (see module doc)
            ranked = rank_flags(expl)
            official = to_official(ranked, scored_at)
            problems = validate_official_values(official)
            if problems:
                raise ValueError(f"official output failed validation: {problems}")
            for rec in official.to_dict("records"):
                rec["money_at_risk"] = _none_if_nan(rec["money_at_risk"])
                by_id[rec["record_id"]] = rec
            next_n = len(official) + 1

        # --- records that were scored but not flagged: same columns, flagged 'N', no reason, no amount
        unflagged = feats[is_req & (flags[flag_col] == 0)]
        if len(unflagged):
            raw = flags.loc[unflagged.index, "if_anomaly_score"]
            order = pd.DataFrame({"id": unflagged[id_col].to_numpy(), "raw": raw.to_numpy()}, index=unflagged.index)
            order = order.sort_values(["raw", "id"], ascending=[False, True], kind="mergesort")
            scaled = det.scale.apply(order["raw"])
            for n, (idx, row) in enumerate(order.iterrows(), start=next_n):
                by_id[row["id"]] = {
                    "result_id": f"AN-{n:04d}", "source_dataset": source, "record_id": row["id"],
                    "employee_id": unflagged.at[idx, "employee_id"],
                    "period": pd.Timestamp(unflagged.at[idx, "month"]).strftime("%Y-%m-%d"),
                    "anomaly_score": float(scaled.at[idx]), "flagged": "N", "reason_code": None, "reason_text": None,
                    "money_at_risk": None, "model_version": MODEL_VERSIONS[source], "scored_at": scored_at}

        missing = [rid for rid in req_order if rid not in by_id]
        if missing:                                    # a record may never be silently dropped
            raise ValueError(f"{len(missing)} request record(s) produced no result")
        return [by_id[rid] for rid in req_order]       # input order
