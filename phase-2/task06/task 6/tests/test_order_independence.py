"""Results must not depend on the order of the input rows (original-experiment review)."""
import numpy as np
import pandas as pd
import pytest

from conftest import payslip
from cortexhr_task6.detection import run_detection
from cortexhr_task6.payroll_features import build_payroll_features
from cortexhr_task6.payroll_model import ModelConfig
from cortexhr_task6.payroll_rules import RULE_COLUMNS, RuleConfig, apply_rules, detect_duplicate_payments


@pytest.fixture
def panel():
    """40 employees x 6 months, plus a duplicated payslip, a missing month and a negative net (synthetic)."""
    rows, n = [], 0
    rng = np.random.default_rng(0)
    for e in range(40):
        for m in range(1, 7):
            if e == 7 and m == 3:
                continue                                          # employee 7 has no March payslip
            n += 1
            rows.append(payslip(f"P{n:04d}", emp=f"E{e:02d}", month=f"2025-{m:02d}-01", role=f"R{e % 4}",
                                fixed=float(rng.integers(30, 90) * 1000)))
    df = pd.DataFrame(rows)
    df.loc[17, "net_pay"] = -1000.0
    dup = df.iloc[[60]].copy()
    dup["payroll_item_id"] = "P9999"
    return pd.concat([df, dup], ignore_index=True)


def flags_by_id(df, rule_cfg=None, model_cfg=None):
    feats, flags, _ = run_detection(df, rule_cfg, model_cfg)
    out = flags.copy()
    out.index = feats["payroll_item_id"].values
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_whole_pipeline_is_identical_after_shuffling_rows(panel, seed):
    base = flags_by_id(panel)
    shuffled = flags_by_id(panel.sample(frac=1, random_state=seed).reset_index(drop=True)).reindex(base.index)
    pd.testing.assert_frame_equal(base, shuffled)          # rule flags, model flag, scores, reasons - everything


def test_whole_pipeline_is_identical_after_reversing_rows(panel):
    base = flags_by_id(panel)
    rev = flags_by_id(panel.iloc[::-1].reset_index(drop=True)).reindex(base.index)
    pd.testing.assert_frame_equal(base, rev)


@pytest.mark.parametrize("selection", ["all", "later_by_item_id"])
def test_duplicate_rule_flags_same_payslips_after_shuffle(panel, selection):
    cfg = RuleConfig(duplicate_selection=selection)
    a = flags_by_id(panel, cfg)["rule_duplicate_payment"]
    b = flags_by_id(panel.sample(frac=1, random_state=9).reset_index(drop=True), cfg)["rule_duplicate_payment"].reindex(a.index)
    assert a.equals(b)
    assert a.sum() == (2 if selection == "all" else 1)


def test_original_experiment_duplicate_selection_is_order_dependent_as_documented(panel):
    """file_order is kept ONLY to reproduce the first experiment; this test records why it was replaced."""
    cfg = RuleConfig(duplicate_selection="file_order")
    a = flags_by_id(panel, cfg)["rule_duplicate_payment"]
    b = flags_by_id(panel.iloc[::-1].reset_index(drop=True), cfg)["rule_duplicate_payment"].reindex(a.index)
    assert not a.equals(b)


def test_history_features_do_not_depend_on_row_order(panel):
    cols = [
        "prev_monthly_fixed_gross", "prev_pf_employee", "prev2_monthly_fixed_gross",
        "fixed_gross_change_vs_prev", "pf_vs_prev_month",
        "role_median_fixed_gross", "fixed_gross_vs_role_median",
    ]
    a = build_payroll_features(panel).set_index("payroll_item_id")[cols]
    b = build_payroll_features(panel.sample(frac=1, random_state=4)).set_index("payroll_item_id")[cols].reindex(a.index)
    pd.testing.assert_frame_equal(a, b)


def test_missing_month_gives_no_history_instead_of_reaching_back(panel):
    f = build_payroll_features(panel)
    rules = apply_rules(f)
    is_e07 = f["employee_id"] == "E07"
    april = f[is_e07 & (f["month"] == "2025-04-01")].iloc[0]       # March is missing for E07
    assert np.isnan(april["prev_monthly_fixed_gross"]) and np.isnan(april["fixed_gross_change_vs_prev"])
    feb_fixed = f[is_e07 & (f["month"] == "2025-02-01")]["monthly_fixed_gross"].iloc[0]
    assert april["prev2_monthly_fixed_gross"] == feb_fixed         # two calendar months back is February
    assert rules.loc[april.name, "rule_salary_drop"] == 0          # undefined comparison -> no flag


# ---------------------------------------------------------------- the approaches are independent of each other
def test_rules_only_flags_do_not_depend_on_model_settings(panel):
    a = flags_by_id(panel, model_cfg=ModelConfig(contamination=0.01, random_state=1))
    b = flags_by_id(panel, model_cfg=ModelConfig(contamination=0.10, random_state=99))
    pd.testing.assert_frame_equal(a[RULE_COLUMNS + ["rules_only_flag"]], b[RULE_COLUMNS + ["rules_only_flag"]])
    assert not a["if_flag"].equals(b["if_flag"])               # sanity: the model settings did change something


def test_model_only_flags_do_not_depend_on_rule_settings(panel):
    a = flags_by_id(panel, rule_cfg=RuleConfig())
    b = flags_by_id(panel, rule_cfg=RuleConfig(overtime_cap_hours=1, role_ratio_limit=1.01, duplicate_selection="later_by_item_id",
                                               suppress_drop_after_flagged_spike=True))
    pd.testing.assert_frame_equal(a[["if_flag", "if_anomaly_score"]], b[["if_flag", "if_anomaly_score"]])


@pytest.fixture
def noisy_panel():
    """Larger synthetic panel with varied overtime/variable pay, so the forest's fit is sensitive to row order."""
    rng = np.random.default_rng(7)
    rows, n = [], 0
    for e in range(120):
        for m in range(1, 7):
            n += 1
            eligible = "Y" if e % 3 == 0 else "N"
            rows.append(payslip(f"N{n:05d}", emp=f"E{e:03d}", month=f"2025-{m:02d}-01", role=f"R{e % 5}",
                                fixed=float(rng.integers(25, 120) * 1000),
                                ot_hours=float(rng.integers(0, 25)) if eligible == "Y" else 0.0, ot_eligible=eligible,
                                variable=float(rng.choice([0.0, 0.0, 0.0, 5000.0, 20000.0]))))
    return pd.DataFrame(rows)


def test_model_flags_do_not_depend_on_row_order_on_a_noisy_panel(noisy_panel):
    a = flags_by_id(noisy_panel)
    b = flags_by_id(noisy_panel.sample(frac=1, random_state=21).reset_index(drop=True)).reindex(a.index)
    assert a["if_flag"].sum() > 0
    pd.testing.assert_series_equal(a["if_flag"], b["if_flag"])
    pd.testing.assert_series_equal(a["if_anomaly_score"], b["if_anomaly_score"])


def test_file_order_fit_really_is_order_dependent_on_a_noisy_panel(noisy_panel):
    """Records why canonical ordering exists: the original (file-order) fit changes when rows are shuffled."""
    cfg = ModelConfig(canonical_row_order=False)
    a = flags_by_id(noisy_panel, model_cfg=cfg)
    b = flags_by_id(noisy_panel.sample(frac=1, random_state=21).reset_index(drop=True), model_cfg=cfg).reindex(a.index)
    assert not a["if_anomaly_score"].equals(b["if_anomaly_score"])
