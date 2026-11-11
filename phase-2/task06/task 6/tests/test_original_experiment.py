"""Integration test on the real workbook (skipped when it is not available): the FIRST experiment must
remain reproducible with --original-experiment settings, and the corrected default must be order independent."""
import pandas as pd
import pytest

from cortexhr_task6.data_loading import find_workbook, load_payroll_features
from cortexhr_task6.detection import run_detection
from cortexhr_task6.experiments import original_experiment_configs
from cortexhr_task6.payroll_evaluation import ALL_ERRORS, load_payroll_labels, per_type_metrics
from cortexhr_task6.payroll_model import ModelConfig
from cortexhr_task6.payroll_features import RoleReference
from cortexhr_task6.payroll_rules import RuleConfig

# frozen from reports/original_experiment/task6_metrics_per_type.csv (ALL_ERRORS rows): flagged, TP
ORIGINAL_V1 = {"rules_only_flag": (105, 92), "if_flag": (184, 56), "combined_flag": (238, 97)}


@pytest.fixture(scope="module")
def real():
    try:
        wb = find_workbook()
        raw = load_payroll_features(wb)
        labels, sheet = load_payroll_labels(wb)
    except Exception as exc:                       # workbook missing -> skip, never fail
        pytest.skip(f"workbook not available: {exc}")
    return raw, labels


def overall(raw, labels, rule_cfg, model_cfg, role_reference=None):
    feats, flags, _ = run_detection(raw, rule_cfg, model_cfg, role_reference=role_reference)
    ids = feats["payroll_item_id"].values
    out = {}
    for col in ORIGINAL_V1:
        r = per_type_metrics("x", pd.Series(flags[col].values, index=ids), labels).query("error_type == @ALL_ERRORS").iloc[0]
        out[col] = (int(r.predicted_positive), int(r.tp))
    return out


def test_original_experiment_is_still_reproducible(real):
    raw, labels = real
    rule_cfg, model_cfg = original_experiment_configs(RuleConfig(), ModelConfig())
    assert overall(raw, labels, rule_cfg, model_cfg, RoleReference().fit(raw)) == ORIGINAL_V1


def test_corrected_default_is_row_order_independent_on_real_data(real):
    raw, labels = real
    a = overall(raw, labels, RuleConfig(), ModelConfig())
    b = overall(raw.sample(frac=1, random_state=11).reset_index(drop=True), labels, RuleConfig(), ModelConfig())
    assert a == b
