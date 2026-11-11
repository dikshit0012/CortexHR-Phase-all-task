"""OPTIONAL: runs only when CORTEXHR_WORKBOOK points at the real reference workbook (it is never committed).

Checks the service against a FRESH run of the existing Task 7 pipeline on the same workbook and the same library
versions: every flagged record must come back with identical score, reason code, reason text and money at risk.
(A committed Task 7 CSV can differ slightly across scikit-learn / pandas versions, so it is not the oracle here.)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.artifacts import build_artifacts
from app.bootstrap import load_service
from app.config import Settings
from app.reference import ATTENDANCE_SHEET, PAYROLL_SHEET, load_reference
from app.schemas import AttendanceRecord, PayrollRecord
from cortexhr_task7.pipeline import build_official_output

WB = os.environ.get("CORTEXHR_WORKBOOK")
pytestmark = pytest.mark.skipif(not WB or not Path(WB).is_file(), reason="set CORTEXHR_WORKBOOK to run the real-data checks")


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    ref = load_reference(Path(WB))
    d = tmp_path_factory.mktemp("real_models")
    build_artifacts(ref, d)
    svc = load_service(Settings(workbook_path=Path(WB), model_dir=d, max_batch_size=100))
    official, _, _ = build_official_output(pd.read_excel(WB, sheet_name=PAYROLL_SHEET), pd.read_excel(WB, sheet_name=ATTENDANCE_SHEET),
                                           scored_at="2026-10-01 02:00:00")
    return ref, svc, official


def _records(df, model, ids):
    out = []
    for r in df[df.iloc[:, 0].isin(ids)].to_dict("records"):
        out.append(model.model_validate({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}).model_dump())
    return out


def test_every_real_row_is_accepted_by_the_request_schemas(real):
    ref, _, _ = real
    assert len(_records(ref.attendance, AttendanceRecord, set(ref.attendance.iloc[:, 0]))) == len(ref.attendance)
    assert len(_records(ref.payroll, PayrollRecord, set(ref.payroll.iloc[:, 0]))) == len(ref.payroll)


@pytest.mark.parametrize("src, idc, attr, model, method", [
    (ATTENDANCE_SHEET, "attendance_feature_id", "attendance", AttendanceRecord, "score_attendance"),
    (PAYROLL_SHEET, "payroll_item_id", "payroll", PayrollRecord, "score_payroll"),
])
def test_service_reproduces_fresh_task7_output(real, src, idc, attr, model, method):
    ref, svc, official = real
    frame = getattr(ref, attr)
    expected_source = official[official.source_dataset == src]
    expected = expected_source.set_index("record_id")
    rng = np.random.default_rng(0)
    unflagged = [i for i in frame[idc] if i not in expected.index]
    sample = set(rng.choice(unflagged, 200, replace=False))
    res = pd.DataFrame(getattr(svc, method)(_records(frame, model, set(expected.index) | sample))).set_index("record_id")
    expected_ranking = expected_source["record_id"].tolist()
    actual_ranking = res[res["flagged"] == "Y"].sort_values("result_id").index.tolist()
    assert actual_ranking == expected_ranking                         # Task 7 tie-break order within this detector
    for rid, exp in expected.iterrows():
        got = res.loc[rid]
        assert got["flagged"] == "Y"
        for col in ("employee_id", "period", "anomaly_score", "reason_code", "reason_text", "model_version"):
            assert got[col] == exp[col], (rid, col)
        assert (pd.isna(got["money_at_risk"]) and pd.isna(exp["money_at_risk"])) or abs(got["money_at_risk"] - exp["money_at_risk"]) < 1e-9, rid
    assert (res.loc[list(sample), "flagged"] == "N").all()               # nothing extra is flagged
