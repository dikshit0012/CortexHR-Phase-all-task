"""Unit tests for Task 5. SYNTHETIC data only - real evaluation lives in the run script."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexhr_task5 import attendance_anomaly_detector as m  # noqa: E402
from cortexhr_task5.evaluation import catch_rate, summarise  # noqa: E402


def make_frame(n_emp=60, months=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    k = 0
    for e in range(n_emp):
        for mo in range(months):
            k += 1
            sched = 20
            unpaid = int(rng.choice([0, 0, 0, 1]))
            leave = int(rng.choice([0, 0, 1]))
            present = sched - unpaid - leave
            rows.append({
                "attendance_feature_id": f"AF-{k:05d}", "employee_id": f"EMP-{e:04d}",
                "month": pd.Timestamp(2025, 4 + mo, 1), "scheduled_days": sched,
                "present_days": present, "half_days_unpaid": 0, "paid_leave_days": leave,
                "unpaid_absence_days": unpaid, "longest_unpaid_streak_days": min(unpaid, 1),
                "late_arrivals": int(rng.integers(0, 4)),
                "avg_hours_worked": float(rng.normal(8.5, 0.3)), "overtime_hours": 0.0,
                "attendance_rate": present / sched,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def frame():
    return make_frame()


# 1. valid numeric features accepted
def test_detector_accepts_valid_features(frame):
    feats = m.build_feature_frame(frame)
    out = m.AttendanceAnomalyDetector().fit_predict(feats)
    assert len(out) == len(feats) and {"if_flag", "if_anomaly_score"} <= set(out.columns)


# 2. missing / invalid / infinite values handled
def test_missing_and_invalid_values_handled(frame):
    bad = frame.copy()
    bad.loc[3, "late_arrivals"] = np.nan
    bad.loc[4, "avg_hours_worked"] = np.inf
    bad["overtime_hours"] = bad["overtime_hours"].astype(object)  # allow a junk string
    bad.loc[5, "overtime_hours"] = "n/a"
    out = m.AttendanceAnomalyDetector().fit_predict(m.build_feature_frame(bad))
    assert out["if_anomaly_score"].notna().all()


def test_missing_required_column_raises(frame):
    with pytest.raises(ValueError, match="missing required columns"):
        m.build_feature_frame(frame.drop(columns=["late_arrivals"]))


def test_constant_feature_is_dropped(frame):
    feats = m.build_feature_frame(frame)
    det = m.AttendanceAnomalyDetector().fit(feats)
    assert "half_days_unpaid" in det.dropped_constant_  # constant 0 in the fixture
    assert "half_days_unpaid" not in det.kept_features_


# 3. output format: 1 anomaly / 0 normal, not sklearn's -1 / 1
def test_flag_format_is_zero_one(frame):
    out = m.AttendanceAnomalyDetector().fit_predict(m.build_feature_frame(frame))
    assert set(out["if_flag"].unique()) <= {0, 1}


# contamination roughly controls the share flagged on the training data
def test_contamination_controls_share_flagged(frame):
    out = m.AttendanceAnomalyDetector().fit_predict(m.build_feature_frame(frame))
    assert 0.01 <= out["if_flag"].mean() <= 0.06


# 4 + 5. rules, including legitimate zero attendance
def test_rules_edge_cases():
    base = {"scheduled_days": 20, "present_days": 20, "paid_leave_days": 0,
            "longest_unpaid_streak_days": 0, "avg_hours_worked": 8.5}
    rows = [
        {**base, "present_days": 0, "paid_leave_days": 20},                 # fully on paid leave -> OK
        {**base, "present_days": 0, "paid_leave_days": 0},                  # no presence, no cover -> flag
        {**base, "scheduled_days": 0, "present_days": 0},                   # nothing scheduled -> OK
        {**base, "longest_unpaid_streak_days": 6, "present_days": 14},      # 6-day streak -> flag
        {**base, "longest_unpaid_streak_days": 4, "present_days": 16},      # 4-day streak -> OK
        {**base, "avg_hours_worked": 18.9},                                 # impossible hours -> flag
        {**base, "present_days": 0, "avg_hours_worked": 0.0, "paid_leave_days": 5},  # partial cover only -> flag
    ]
    r = m.apply_rules(pd.DataFrame(rows))
    assert r["rule_unexcused_absence"].tolist() == [0, 1, 0, 1, 0, 0, 1]
    assert r["rule_implausible_hours"].tolist() == [0, 0, 0, 0, 0, 1, 0]


# 6. combined flag behaves as OR and keeps the sources
def test_combined_flag_is_or():
    mo = pd.DataFrame({"if_flag": [1, 0, 0, 0], "if_anomaly_score": [0.2, 0.0, 0.1, -0.1]})
    ru = pd.DataFrame({"rule_unexcused_absence": [0, 1, 0, 0], "rule_implausible_hours": [0, 0, 0, 0]})
    c = m.combine_flags(mo, ru)
    assert c["combined_anomaly"].tolist() == [1, 1, 0, 0]
    assert "isolation forest" in c.loc[0, "anomaly_reasons"]
    assert "unexcused" in c.loc[1, "anomaly_reasons"]
    assert c.loc[2, "anomaly_reasons"] == ""


# 7. ranking: more abnormal first, score direction correct
def test_ranking_puts_more_abnormal_first():
    flags = pd.DataFrame({
        "if_flag": [1, 1, 0, 1],
        "if_anomaly_score": [0.05, 0.30, 0.0, 0.15],
        "rule_unexcused_absence": [0, 1, 1, 0], "rule_implausible_hours": [0, 0, 0, 0],
        "rule_count": [0, 1, 1, 0], "combined_anomaly": [1, 1, 1, 1],
    })
    ranked = m.rank_for_review(flags)
    assert ranked.index[0] == 1                       # IF + rule + highest score
    assert ranked.loc[1, "priority"] == "High"
    assert ranked.index.tolist().index(3) < ranked.index.tolist().index(0)  # 0.15 outranks 0.05


def test_if_score_direction_higher_is_more_abnormal(frame):
    f = frame.copy()
    f.loc[10, ["unpaid_absence_days", "longest_unpaid_streak_days", "present_days", "attendance_rate"]] = [18, 18, 2, 0.1]
    feats = m.build_feature_frame(f)
    out = m.AttendanceAnomalyDetector().fit_predict(feats)
    assert out["if_anomaly_score"].idxmax() == 10


# 8. reproducible with a fixed seed
def test_reproducible_with_seed(frame):
    feats = m.build_feature_frame(frame)
    a = m.AttendanceAnomalyDetector(m.DetectorConfig(random_state=7)).fit_predict(feats)
    b = m.AttendanceAnomalyDetector(m.DetectorConfig(random_state=7)).fit_predict(feats)
    pd.testing.assert_frame_equal(a, b)


# 9. labels are not part of any fit / predict interface
def test_no_label_arguments_anywhere():
    for fn in (m.AttendanceAnomalyDetector.fit, m.AttendanceAnomalyDetector.predict,
               m.AttendanceAnomalyDetector.fit_predict, m.run_detector, m.apply_rules, m.rank_for_review):
        names = " ".join(inspect.signature(fn).parameters)
        assert "label" not in names and "truth" not in names and "y" not in inspect.signature(fn).parameters


def test_history_feature_uses_past_only():
    f = make_frame(n_emp=1, months=4)
    f["attendance_rate"] = [1.0, 0.5, 0.9, 0.9]
    out = m.add_history_features(m.validate_attendance_frame(f))
    assert out["rate_change_vs_own_history"].iloc[0] == 0.0
    assert out["rate_change_vs_own_history"].iloc[1] == pytest.approx(0.5 - 1.0)
    assert out["rate_change_vs_own_history"].iloc[2] == pytest.approx(0.9 - 0.75)


# evaluation helper
def test_catch_rate_and_summary():
    c = catch_rate({"a", "b", "c", "d"}, {"a", "b", "x"})
    assert c["precision"] == 0.5 and c["recall"] == pytest.approx(round(2 / 3, 3))
    s = summarise("t", {"a", "b", "c", "d"}, {"a", "b", "x"}, 100)
    assert s["false_positives"] == 2 and s["false_negatives"] == 1 and s["pct_flagged"] == 4.0
