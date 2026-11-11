"""The independent verifier must accept a correct run and reject tampering."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from test_pipeline_smoke import load_runner, write_workbook

VERIFY = Path(__file__).resolve().parents[1] / "scripts" / "verify_task7_reports.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("verify_task7_reports", VERIFY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def run(tmp_path, planted_panel, planted_attendance):
    wb = write_workbook(tmp_path / "wb.xlsx", planted_panel[0], planted_attendance[0])
    out = tmp_path / "reports"
    load_runner().main(["--workbook", str(wb), "--reports-dir", str(out)])
    return wb, out


def test_verifier_accepts_a_correct_run(run):
    wb, out = run
    assert load_verifier().verify(out, wb) == []


def _edit(out, fn):
    path = out / "task7_anomaly_results.csv"
    df = pd.read_csv(path)
    fn(df)
    df.to_csv(path, index=False)


@pytest.mark.parametrize("name,tamper,expected", [
    ("wrong amount", lambda d: d.__setitem__("money_at_risk", d["money_at_risk"].where(d.index != 0, d["money_at_risk"] + 500)), "differ from the independent recomputation"),
    ("wrong column order", lambda d: d.insert(0, "record_id", d.pop("record_id")), "columns differ"),
    ("unknown reason code", lambda d: d.__setitem__("reason_code", d["reason_code"].where(d.index != 0, "MADE_UP")), "unknown reason codes"),
    ("empty text", lambda d: d.__setitem__("reason_text", d["reason_text"].where(d.index != 0, "")), "empty reason_text"),
    ("invented record", lambda d: d.__setitem__("record_id", d["record_id"].where(d.index != 0, "PI-NOT-REAL")), "not in source"),
    ("wrong employee", lambda d: d.__setitem__("employee_id", d["employee_id"].where(d.index != 0, "E-WRONG")), "employee_id mismatch"),
    ("broken ranking", lambda d: d.__setitem__("money_at_risk", d["money_at_risk"].iloc[::-1].to_numpy()), "not ranked"),
])
def test_verifier_rejects_tampering(run, name, tamper, expected):
    wb, out = run
    _edit(out, tamper)
    problems = load_verifier().verify(out, wb)
    assert any(expected in p for p in problems), (name, problems)


def test_verifier_rejects_a_sample_that_is_not_part_of_the_output(run):
    wb, out = run
    path = out / "task7_sample_20_flags.csv"
    s = pd.read_csv(path)
    s.loc[0, "result_id"] = "AN-9999"
    s.to_csv(path, index=False)
    assert any("not in the official output" in p for p in load_verifier().verify(out, wb))
