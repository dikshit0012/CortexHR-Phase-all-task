"""Fairness monitor. The re-execution tests run the REAL Task 12 / Task 15 code on SYNTHETIC data (values are not real results)."""
import json

import pytest

import synthetic
from monitoring import fairness_monitor as fm
from monitoring import monitoring_config as cfg


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("fair")
    pairs = synthetic.write_twin_data(root)
    csv, unlock = synthetic.write_salary_data(root)
    return {"root": root, "pairs": pairs, "csv": csv, "unlock": unlock}


@pytest.fixture
def twin_env(data, monkeypatch):
    monkeypatch.setenv("CORTEXHR_TASK11_DATA", str(data["root"] / "task11_data"))
    monkeypatch.setenv("CORTEXHR_TASK12_PAIRS", str(data["pairs"]))


@pytest.fixture
def pay_env(data, monkeypatch):
    monkeypatch.setenv("CORTEXHR_SALARY_CSV", str(data["csv"]))
    monkeypatch.setenv("CORTEXHR_TASK15_PRIVATE", str(data["unlock"].parent))


@pytest.fixture(scope="module")
def pay_result(data):
    import os
    os.environ.update(CORTEXHR_SALARY_CSV=str(data["csv"]), CORTEXHR_TASK15_PRIVATE=str(data["unlock"].parent))
    try:
        return fm.evaluate_pay_gap()
    finally:
        os.environ.pop("CORTEXHR_SALARY_CSV"), os.environ.pop("CORTEXHR_TASK15_PRIVATE")


# ---------------------------------------------------------------- Task 12
def test_twin_resume_reexecution_uses_task12_methodology(twin_env):
    r = fm.evaluate_twin_resume()
    assert r["status"] == "OK" and r["valid_pairs"] == 25 == r["expected_pairs"]
    assert r["average_gap"] == 0.0 and r["maximum_gap"] == 0.0 and r["pairs_exceeding_tolerance"] == 0
    assert r["thresholds"]["alert_gap"] == 0.01                        # Task 12's own TOLERANCE, imported not copied
    assert r["baseline"]["pairs"] == 25 and r["delta_vs_baseline"] == {"average_gap": 0.0, "maximum_gap": 0.0}


def test_twin_gap_is_detected_when_one_twin_is_scored_differently(tmp_path, monkeypatch):
    pairs = synthetic.write_twin_data(tmp_path, bias=1.0)
    monkeypatch.setenv("CORTEXHR_TASK11_DATA", str(tmp_path / "task11_data"))
    monkeypatch.setenv("CORTEXHR_TASK12_PAIRS", str(pairs))
    r = fm.evaluate_twin_resume()
    assert r["status"] in ("WATCH", "ALERT") and r["maximum_gap"] > cfg.TWIN_WATCH_GAP and r["non_zero_gap_pairs"] > 0


@pytest.mark.parametrize("gap,expected", [(0.0, "OK"), (0.001, "OK"), (0.0010001, "WATCH"), (0.01, "WATCH"), (0.0100001, "ALERT"),
                                          (float("nan"), "NOT_EVALUATED"), (None, "NOT_EVALUATED")])
def test_twin_threshold_boundaries(gap, expected):
    assert fm.classify_twin_gap(gap, 0.01) == expected


def test_twin_missing_data_is_not_evaluated_never_ok():
    r = fm.evaluate_twin_resume()
    assert r["status"] == "NOT_EVALUATED" and r["valid_pairs"] == 0 and r["maximum_gap"] is None


def test_twin_invalid_inputs_are_an_error_without_leaking_ids(data, monkeypatch):
    monkeypatch.setenv("CORTEXHR_TASK11_DATA", str(data["root"] / "task11_data"))
    monkeypatch.setenv("CORTEXHR_TASK12_PAIRS", str(data["pairs"]))
    r = fm.evaluate_twin_resume(expected_pairs=24)                     # wrong pair count
    assert r["status"] == "ERROR" and "R000" not in json.dumps(r) and str(data["root"]) not in json.dumps(r)


def test_twin_pairs_referencing_unknown_resumes_is_an_error(data, tmp_path, monkeypatch):
    import pandas as pd
    p = pd.read_csv(data["pairs"])
    p.loc[0, "resume_id_twin"] = "R999Z"
    bad = tmp_path / "bad_pairs.csv"
    p.to_csv(bad, index=False)
    monkeypatch.setenv("CORTEXHR_TASK11_DATA", str(data["root"] / "task11_data"))
    monkeypatch.setenv("CORTEXHR_TASK12_PAIRS", str(bad))
    r = fm.evaluate_twin_resume()
    assert r["status"] == "ERROR" and "R999Z" not in json.dumps(r)


# ---------------------------------------------------------------- Task 15
def test_pay_gap_reexecution_runs_the_real_task15_pipeline(pay_result):
    r = pay_result
    assert r["status"] in ("OK", "WATCH", "ALERT") and set(r["components"]) == {"gender", "career_gap"}
    g = r["components"]["gender"]
    assert {x["group"] for x in g["groups"]} == {"Female", "Male", "Other"} and sum(x["n"] for x in g["groups"]) > 100
    assert any(not x["reliable"] for x in g["groups"])                  # 'Other' is below the reliability minimum
    assert set(g["metrics"]) == set(cfg.PAY_MODEL_QUANTITIES) and r["groups_evaluated"] == 6
    assert r["model"] in ("Linear Regression", "Gradient Boosting Regression")


def test_pay_gap_output_has_no_individual_records(pay_result, data):
    text = json.dumps(pay_result)
    assert "H12" not in text and "hire_id" not in text and "person_ref" not in text and str(data["root"]) not in text


def test_pay_gap_missing_salary_data_is_not_evaluated():
    r = fm.evaluate_pay_gap()
    assert r["status"] == "NOT_EVALUATED"


def test_pay_gap_missing_task15_code_is_an_error_not_a_fabricated_result():
    def missing_code():
        raise ImportError("Task 15 is unavailable")

    r = fm.evaluate_pay_gap(missing_code)
    assert r["status"] == "ERROR" and "fallback" not in json.dumps(r).lower()


def test_pay_gap_without_protected_labels_cannot_be_ok(data, tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEXHR_SALARY_CSV", str(data["csv"]))
    monkeypatch.setenv("CORTEXHR_TASK15_PRIVATE", str(tmp_path))       # empty folder: no unlock_CX-15
    r = fm.evaluate_pay_gap()
    assert r["components"]["gender"]["status"] == "NOT_EVALUATED" and r["components"]["career_gap"]["status"] != "NOT_EVALUATED"
    assert r["status"] != "OK"


def test_pay_gap_invalid_input_is_an_error(data, tmp_path, monkeypatch):
    import pandas as pd
    d = pd.read_csv(data["csv"]).drop(columns=["offered_annual_fixed"])
    bad = tmp_path / "salary_training.csv"
    d.to_csv(bad, index=False)
    monkeypatch.setenv("CORTEXHR_SALARY_CSV", str(bad))
    r = fm.evaluate_pay_gap()
    assert r["status"] == "ERROR" and "H1" not in json.dumps(r)


@pytest.mark.parametrize("cur,base,kind,expected", [
    (0.017, 0.017, "ratio", "OK"), (0.010, 0.017, "ratio", "OK"),          # smaller gap = improvement, not flagged
    (0.021, 0.017, "ratio", "OK"),                                          # +0.004 below materiality 0.005
    (0.0221, 0.017, "ratio", "ALERT"),                                      # +0.0051 passes materiality; 0.0051/0.017 = 30% > 25%
    (0.041, 0.035, "ratio", "WATCH"),                                       # +0.006 and +17%
    (0.0385, 0.035, "ratio", "OK"),                                         # +10.0% exactly is not above the watch limit... 0.0035 < materiality anyway
    (0.045, 0.035, "ratio", "ALERT"), (0.0191, 0.017, "ratio", "OK"),
    (26_000, 23_757, "inr", "OK"), (29_000, 23_757, "inr", "WATCH"), (31_000, 23_757, "inr", "ALERT"),
    (float("nan"), 0.017, "ratio", "NOT_EVALUATED"), (None, 0.017, "ratio", "NOT_EVALUATED")])
def test_pay_gap_change_rules(cur, base, kind, expected):
    assert fm.classify_gap_change(cur, base, kind)["status"] == expected


def test_pay_gap_aggregation_with_fake_runner():
    import pandas as pd
    q = list(cfg.PAY_MODEL_QUANTITIES)

    def res(gender_ratio, include_gender=True, career=0.035):
        def summ(ratio, inr):
            return pd.DataFrame({"model": "M", "quantity": q, "max_gap_between_group_means": [ratio, inr]})
        tab = pd.DataFrame({"model": "M", "group": ["A", "B"], "n": [50, 12]})
        out = {"gap_summaries": {"career_gap": summ(career, 55_549.0)}, "group_tables": {"career_gap": tab}, "selected": "m"}
        if include_gender:
            out["gap_summaries"]["gender"] = summ(gender_ratio, 23_757.0)
            out["group_tables"]["gender"] = tab
        return out
    name = lambda r: "M"  # noqa: E731
    assert fm.evaluate_pay_gap(lambda: res(0.017), name)["status"] == "OK"
    assert fm.evaluate_pay_gap(lambda: res(0.0195), name)["status"] == "OK"
    assert fm.evaluate_pay_gap(lambda: res(0.040), name)["status"] == "ALERT"     # gender baseline 0.017 -> +135%
    assert fm.evaluate_pay_gap(lambda: res(0.017, career=0.041), name)["status"] == "WATCH"
    assert fm.evaluate_pay_gap(lambda: res(0.017, career=0.050), name)["status"] == "ALERT"
    assert fm.evaluate_pay_gap(lambda: res(0.017, include_gender=False), name)["status"] == "NOT_EVALUATED"
    assert fm.evaluate_pay_gap(lambda: (_ for _ in ()).throw(FileNotFoundError("/secret/path.xlsx")), name)["status"] == "NOT_EVALUATED"
    boom = fm.evaluate_pay_gap(lambda: (_ for _ in ()).throw(ValueError("hire H77 has bad salary")), name)
    assert boom["status"] == "ERROR" and "H77" not in json.dumps(boom)


# ---------------------------------------------------------------- unlock file with two record types (found in the real file)
def _unlock_variant(data, tmp_path, extra_snapshot_rows=True, dup_within_hires=False):
    import pandas as pd
    u = pd.read_csv(data["unlock"])
    u["record_type"] = "salary_hire"
    parts = [u]
    if extra_snapshot_rows:                    # the same people again as 'attrition_snapshot' (as in the real file): same gender, other age
        snap = u.head(60).copy()
        snap["record_type"], snap["age_years"] = "attrition_snapshot", 45.0
        parts.append(snap)
    if dup_within_hires:
        parts.append(u.head(2))
    d = tmp_path / "private"
    d.mkdir()
    pd.concat(parts, ignore_index=True).to_csv(d / data["unlock"].name, index=False)
    return d


def test_unlock_with_attrition_snapshot_rows_is_filtered_to_salary_hire(data, tmp_path, monkeypatch, pay_result):
    monkeypatch.setenv("CORTEXHR_SALARY_CSV", str(data["csv"]))
    monkeypatch.setenv("CORTEXHR_TASK15_PRIVATE", str(_unlock_variant(data, tmp_path)))
    r = fm.evaluate_pay_gap()
    assert r["status"] in ("OK", "WATCH", "ALERT") and r["components"]["gender"]["status"] != "NOT_EVALUATED"
    assert r["components"]["gender"]["groups"] == pay_result["components"]["gender"]["groups"]       # same groups/sizes as the clean file


def test_duplicate_person_ref_among_hire_records_is_still_rejected(data, tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEXHR_SALARY_CSV", str(data["csv"]))
    monkeypatch.setenv("CORTEXHR_TASK15_PRIVATE", str(_unlock_variant(data, tmp_path, dup_within_hires=True)))
    assert fm.evaluate_pay_gap()["status"] == "ERROR"
