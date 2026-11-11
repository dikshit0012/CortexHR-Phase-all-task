import inspect

import numpy as np
import pandas as pd
import pytest

from conftest import payslip
from cortexhr_task6 import detection
from cortexhr_task6.detection import combine_flags, run_detection
from cortexhr_task6.payroll_features import FORBIDDEN_MODEL_COLUMNS, MODEL_FEATURES
from cortexhr_task6.payroll_model import ModelConfig, PayrollAnomalyModel, assert_no_forbidden_columns
from cortexhr_task6.payroll_rules import RULE_COLUMNS


@pytest.fixture
def bigger_frame():
    """40 clean employees x 6 months plus a couple of planted problems (synthetic)."""
    rows, n = [], 0
    rng = np.random.default_rng(0)
    for e in range(40):
        for m in range(1, 7):
            n += 1
            rows.append(payslip(f"P{n:04d}", emp=f"E{e:02d}", month=f"2025-{m:02d}-01", role=f"R{e % 4}",
                                fixed=float(rng.integers(30, 90) * 1000)))
    df = pd.DataFrame(rows)
    df.loc[5, ["gross_pay", "net_pay"]] = [df.loc[5, "gross_pay"] * 10, df.loc[5, "net_pay"] * 10]   # shifted decimal
    df.loc[17, "net_pay"] = -1000.0
    return df


def test_combined_flag_is_exactly_rules_or_model(bigger_frame):
    feats, flags, _ = run_detection(bigger_frame)
    expected = ((flags["rules_only_flag"] == 1) | (flags["if_flag"] == 1)).astype(int)
    assert (flags["combined_flag"] == expected).all()
    assert (flags["rules_only_flag"] == (flags[RULE_COLUMNS].sum(axis=1) > 0).astype(int)).all()


def test_combine_flags_on_hand_made_inputs():
    model_out = pd.DataFrame({"if_flag": [0, 1, 0, 1], "if_anomaly_score": [0.1, 0.9, 0.2, 0.8]})
    rules = pd.DataFrame(0, index=range(4), columns=RULE_COLUMNS)
    rules.loc[[2, 3], "rule_negative_net"] = 1
    out = combine_flags(model_out, rules)
    assert out["combined_flag"].tolist() == [0, 1, 1, 1]
    assert out["rules_only_flag"].tolist() == [0, 0, 1, 1]
    assert "negative net pay" in out.loc[2, "detection_reasons"]
    assert "isolation forest" in out.loc[1, "detection_reasons"]
    assert out.loc[0, "detection_reasons"] == ""


def test_output_schema_keeps_every_rule_flag_and_model_flag(bigger_frame):
    _, flags, _ = run_detection(bigger_frame)
    required = set(RULE_COLUMNS) | {"if_flag", "if_anomaly_score", "rule_count", "rules_only_flag", "combined_flag", "detection_reasons"}
    assert required <= set(flags.columns)
    assert set(flags["combined_flag"].unique()) <= {0, 1}
    assert len(flags) == len(bigger_frame)


def test_planted_errors_are_flagged_by_rules(bigger_frame):
    _, flags, _ = run_detection(bigger_frame)
    assert flags.loc[5, "rule_gross_mismatch"] == 1
    assert flags.loc[17, "rule_negative_net"] == 1


# ---------------------------------------------------------------- leakage prevention
def test_model_features_contain_no_identifier_or_label_columns():
    assert not set(MODEL_FEATURES) & set(FORBIDDEN_MODEL_COLUMNS)
    assert_no_forbidden_columns(MODEL_FEATURES)


@pytest.mark.parametrize("bad", ["payroll_item_id", "employee_id", "anomaly_type", "injection_note", "rule_negative_net", "label_x"])
def test_forbidden_model_feature_is_rejected(bad):
    with pytest.raises(ValueError, match="forbidden"):
        PayrollAnomalyModel(ModelConfig(features=MODEL_FEATURES + [bad]))


def test_detection_functions_accept_no_labels():
    for fn in (detection.run_detection, PayrollAnomalyModel.fit, PayrollAnomalyModel.predict):
        names = set(inspect.signature(fn).parameters)
        assert not any("label" in n or "target" in n or n == "y" for n in names), fn


def test_predictions_do_not_change_when_label_columns_are_present(bigger_frame):
    a = run_detection(bigger_frame)[1]
    with_labels = bigger_frame.assign(anomaly_type="DOUBLE_PAYMENT", is_injected=1, injection_note="x")
    b = run_detection(with_labels)[1]
    pd.testing.assert_frame_equal(a, b)


def test_id_prefix_cannot_influence_predictions(bigger_frame):
    a = run_detection(bigger_frame)[1]
    renamed = bigger_frame.assign(payroll_item_id=["PI-D" + p[1:] if i % 2 else p for i, p in enumerate(bigger_frame.payroll_item_id)])
    b = run_detection(renamed)[1]
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- model behaviour
def test_same_seed_gives_identical_predictions(bigger_frame):
    a = run_detection(bigger_frame, model_cfg=ModelConfig(random_state=3))[1]
    b = run_detection(bigger_frame, model_cfg=ModelConfig(random_state=3))[1]
    pd.testing.assert_frame_equal(a, b)


def test_contamination_controls_flag_share(bigger_frame):
    flags = run_detection(bigger_frame, model_cfg=ModelConfig(contamination=0.05))[1]
    assert flags["if_flag"].mean() == pytest.approx(0.05, abs=0.02)


def test_score_threshold_overrides_contamination(bigger_frame):
    _, flags, _ = run_detection(bigger_frame, model_cfg=ModelConfig(score_threshold=1e9))
    assert flags["if_flag"].sum() == 0
    _, flags2, _ = run_detection(bigger_frame, model_cfg=ModelConfig(score_threshold=-1e9))
    assert flags2["if_flag"].sum() == len(bigger_frame)


def test_nan_features_do_not_crash_the_model(bigger_frame):
    df = bigger_frame.copy()
    df.loc[3, "gross_pay"] = np.nan
    _, flags, _ = run_detection(df)
    assert not flags["if_anomaly_score"].isna().any()
    assert flags.loc[3, "rule_invalid_amounts"] == 1


def test_predict_before_fit_raises(bigger_frame):
    with pytest.raises(RuntimeError):
        PayrollAnomalyModel().predict(bigger_frame)
