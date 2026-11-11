import numpy as np
import pandas as pd
import pytest

from cortexhr_task6.payroll_evaluation import (
    ALL_ERRORS, baseline_replica_flag, clean_overlap, load_payroll_labels, metrics_from_counts, overlap_table,
    per_type_metrics, results_table, rule_hit_table, validate_label_join,
)
from cortexhr_task6.privacy import mask_ids


def labels_df():
    return pd.DataFrame({"payroll_item_id": ["a1", "a2", "a3", "b1", "b2"],
                         "anomaly_type": ["TYPE_A", "TYPE_A", "TYPE_A", "TYPE_B", "TYPE_B"]})


def flag_series(flagged, n_clean=10):
    ids = ["a1", "a2", "a3", "b1", "b2"] + [f"c{i}" for i in range(n_clean)]
    return pd.Series([1 if i in flagged else 0 for i in ids], index=ids)


def row(df, approach, t):
    return df[(df.approach == approach) & (df.error_type == t)].iloc[0]


# ---------------------------------------------------------------- metric maths
def test_metrics_from_counts_known_values():
    m = metrics_from_counts(tp=3, fp=1, fn=1)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75) and m["f1"] == pytest.approx(0.75)


def test_zero_denominators_are_nan_not_zero():
    assert np.isnan(metrics_from_counts(0, 0, 5)["precision"])        # nothing predicted
    assert np.isnan(metrics_from_counts(0, 3, 0)["recall"])           # no actual cases
    assert np.isnan(metrics_from_counts(0, 0, 5)["f1"])
    assert metrics_from_counts(0, 3, 2)["f1"] == 0.0                  # both defined and zero


def test_per_type_metrics_hand_calculated():
    flag = flag_series({"a1", "a2", "b1", "c0", "c1"})      # 2/3 of A, 1/2 of B, 2 false alarms
    pt = per_type_metrics("X", flag, labels_df())
    a = row(pt, "X", "TYPE_A")
    assert (a.actual_cases, a.tp, a.fn, a.fp, a.tn) == (3, 2, 1, 2, 8)
    assert a.precision == pytest.approx(2 / 4) and a.recall == pytest.approx(2 / 3)
    assert a.flagged_on_other_error_types == 1                         # b1 is another type: neither TP nor FP
    assert a.precision_strict == pytest.approx(2 / 5)                  # strict counts b1 as wrong
    b = row(pt, "X", "TYPE_B")
    assert (b.tp, b.fn, b.fp) == (1, 1, 2) and b.support == 2
    allr = row(pt, "X", ALL_ERRORS)
    assert (allr.tp, allr.fp, allr.fn, allr.tn) == (3, 2, 2, 8)
    assert allr.precision == pytest.approx(3 / 5) and allr.recall == pytest.approx(3 / 5)


def test_all_error_types_are_reported_even_if_never_caught():
    pt = per_type_metrics("X", flag_series(set()), labels_df())
    assert set(pt.error_type) == {"TYPE_A", "TYPE_B", ALL_ERRORS}
    assert row(pt, "X", "TYPE_A").recall == 0.0
    assert np.isnan(row(pt, "X", "TYPE_A").precision)                  # no predictions -> undefined, not 0


def test_results_table_has_model_rules_combined_columns():
    pt = pd.concat([per_type_metrics(n, flag_series({"a1"}), labels_df()) for n in ("M", "R", "C")])
    wide = results_table(pt, {"Model": "M", "Rules": "R", "Combined": "C"})
    assert {"Model precision", "Model recall", "Rules precision", "Rules recall", "Combined precision", "Combined recall"} <= set(wide.columns)
    assert len(wide) == 3


# ---------------------------------------------------------------- rules vs model
def test_overlap_table_buckets_sum_to_actual_cases():
    rules = flag_series({"a1", "a2", "b1"})
    model = flag_series({"a2", "a3", "b1"})
    ov = overlap_table(rules, model, labels_df()).set_index("error_type")
    a = ov.loc["TYPE_A"]
    assert (a.caught_by_rules_only, a.caught_by_model_only, a.caught_by_both, a.missed_by_both) == (1, 1, 1, 0)
    for t in ov.index:
        r = ov.loc[t]
        assert r.caught_by_rules_only + r.caught_by_model_only + r.caught_by_both + r.missed_by_both == r.actual_cases


def test_clean_overlap_counts_false_alarms():
    co = clean_overlap(flag_series({"c0", "c1", "a1"}), flag_series({"c1", "c2"}), labels_df())
    assert co == {"clean_rows": 10, "flagged_by_rules_only": 1, "flagged_by_model_only": 1, "flagged_by_both": 1, "flagged_by_neither": 7}


def test_rule_hit_table():
    flags = pd.DataFrame({"r1": [1, 1, 0, 1], "r2": [0, 0, 1, 0]}, index=["a1", "c0", "b1", "a2"])
    hits = rule_hit_table(flags, labels_df(), ["r1", "r2"]).set_index("rule")
    assert hits.loc["r1", "total_flags"] == 3 and hits.loc["r1", "flags_on_clean_rows"] == 1
    assert hits.loc["r1", "hits_TYPE_A"] == 2 and hits.loc["r2", "hits_TYPE_B"] == 1


# ---------------------------------------------------------------- record matching
def test_join_validation_reports_counts():
    rep = validate_label_join(pd.Series(["a1", "a2", "a3", "b1", "b2", "c0"]), labels_df())
    assert rep["label_rows"] == 5 and rep["matched_rows"] == 5 and rep["unmatched_rows"] == 0
    assert rep["evaluation_rows_used"] == 5 and rep["feature_rows"] == 6


def test_join_rejects_unmatched_labels():
    with pytest.raises(ValueError, match="missing from the feature data"):
        validate_label_join(pd.Series(["a1", "a2"]), labels_df())


def test_join_rejects_duplicate_keys():
    dup_labels = pd.concat([labels_df(), labels_df().iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate join keys"):
        validate_label_join(pd.Series(["a1", "a2", "a3", "b1", "b2"]), dup_labels)
    with pytest.raises(ValueError, match="duplicate join keys"):
        validate_label_join(pd.Series(["a1", "a1", "a2", "a3", "b1", "b2"]), labels_df())


def test_label_loader_prefers_unlock_sheet_then_falls_back(tmp_path):
    base = labels_df()
    f1 = tmp_path / "with_unlock.xlsx"
    with pd.ExcelWriter(f1) as w:
        base.to_excel(w, sheet_name="payroll_anomaly_labels", index=False)
        base.iloc[:2].to_excel(w, sheet_name="unlock_CX-06", index=False)
    df, sheet = load_payroll_labels(f1)
    assert sheet == "unlock_CX-06" and len(df) == 2
    f2 = tmp_path / "fallback.xlsx"
    with pd.ExcelWriter(f2) as w:
        base.to_excel(w, sheet_name="payroll_anomaly_labels", index=False)
    assert load_payroll_labels(f2)[1] == "payroll_anomaly_labels"
    f3 = tmp_path / "none.xlsx"
    with pd.ExcelWriter(f3) as w:
        base.to_excel(w, sheet_name="other", index=False)
    with pytest.raises(ValueError, match="no labels sheet"):
        load_payroll_labels(f3)


def test_baseline_replica_returns_flags_indexed_by_item_id(base_frame):
    flags = baseline_replica_flag(base_frame, contamination=0.1)
    assert list(flags.index) == list(base_frame.payroll_item_id)
    assert set(flags.unique()) <= {0, 1}


# ---------------------------------------------------------------- privacy
def test_masked_ids_are_stable_and_do_not_contain_the_original():
    s = pd.Series(["EMP-0001", "EMP-0002", "EMP-0001"])
    m = mask_ids(s, "EMP")
    assert m[0] == m[2] and m[0] != m[1]
    assert m[0].startswith("EMP-") and m[0] != "EMP-0001"      # prefix kept, original id replaced by a hash
    assert m.str.fullmatch(r"EMP-[0-9a-f]{16}").all()
    assert not m.str.contains("EMP-0001").any()


def test_custom_mask_salt_changes_the_pseudonyms(monkeypatch):
    values = pd.Series(["EMP-0001"])
    monkeypatch.setenv("CORTEXHR_ID_SALT", "test-secret-one")
    first = mask_ids(values, "EMP")
    monkeypatch.setenv("CORTEXHR_ID_SALT", "test-secret-two")
    second = mask_ids(values, "EMP")
    assert first.iloc[0] != second.iloc[0]


# ---------------------------------------------------------------- uncertainty (small samples)
def test_wilson_interval_known_values():
    from cortexhr_task6.payroll_evaluation import wilson_interval
    lo, hi = wilson_interval(5, 5)
    assert lo == pytest.approx(5 / (5 + 1.96**2), abs=1e-6) and hi == pytest.approx(1.0)   # 5/5 still allows ~0.57
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35
    assert all(np.isnan(v) for v in wilson_interval(0, 0))                                   # no trials -> undefined
    lo, hi = wilson_interval(7, 20)
    assert lo < 7 / 20 < hi


def test_per_type_metrics_include_confidence_intervals():
    pt = per_type_metrics("X", flag_series({"a1", "a2", "b1", "c0"}), labels_df())
    a = row(pt, "X", "TYPE_A")
    assert a.recall_ci95_low < a.recall < a.recall_ci95_high
    assert a.precision_ci95_low < a.precision < a.precision_ci95_high


def test_tiered_alerts_add_up_to_the_combined_detector():
    from cortexhr_task6.experiments import tiered_alert_table
    ids = ["a1", "a2", "a3", "b1", "b2", "c0", "c1", "c2"]
    flags = pd.DataFrame({"rules_only_flag": [1, 1, 0, 0, 0, 1, 0, 0], "if_flag": [0, 1, 1, 0, 0, 0, 1, 1]})
    t = tiered_alert_table(flags, ids, labels_df()).set_index("alert_tier")
    t1, t2, both = t.iloc[0], t.iloc[1], t.iloc[2]
    assert (t1.alerts, t1.true_errors_found, t1.false_alarms) == (3, 2, 1)       # a1, a2, c0
    assert (t2.alerts, t2.true_errors_found, t2.false_alarms) == (3, 1, 2)       # a3, c1, c2  (model only)
    assert both.alerts == t1.alerts + t2.alerts and both.true_errors_found == 3
