import numpy as np
import pandas as pd
import pytest

from conftest import payslip
from cortexhr_task6.payroll_features import (
    MODEL_FEATURES, RoleReference, build_payroll_features, safe_divide, validate_payroll_frame,
)


def test_ratio_features_are_calculated_correctly(base_frame):
    f = build_payroll_features(base_frame)
    row = f.iloc[0]                                  # E1, April, clean
    assert row["gross_to_fixed_gross"] == pytest.approx(1.0)
    assert row["deduction_ratio"] == pytest.approx(5000 / 50000)
    assert row["net_pay_ratio"] == pytest.approx(45000 / 50000)
    assert row["gross_vs_expected"] == pytest.approx(1.0)


def test_division_by_zero_gives_nan_not_zero_or_inf():
    out = safe_divide(pd.Series([1.0, 5.0, 0.0]), pd.Series([0.0, 2.0, 0.0]))
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 2.5 and np.isnan(out.iloc[2])
    assert not np.isinf(out).any()


def test_zero_gross_payslip_has_nan_ratios_but_zero_gross_to_fixed(make_frame):
    df = make_frame([payslip("P1", gross_pay=0.0, net_pay=0.0, total_deductions=0.0)])
    f = build_payroll_features(df).iloc[0]
    assert np.isnan(f["deduction_ratio"]) and np.isnan(f["net_pay_ratio"])
    assert f["gross_to_fixed_gross"] == 0.0


def test_missing_and_infinite_values_become_nan(make_frame):
    df = make_frame([payslip("P1", gross_pay=np.inf), payslip("P2", emp="E2", net_pay="oops"),
                     payslip("P3", emp="E3", tds=np.nan)])
    v = validate_payroll_frame(df)
    assert np.isnan(v.loc[0, "gross_pay"]) and np.isnan(v.loc[1, "net_pay"]) and np.isnan(v.loc[2, "tds"])
    f = build_payroll_features(df)                   # must not raise or create inf
    assert not np.isinf(f[MODEL_FEATURES].to_numpy(dtype=float)).any()


def test_negative_net_pay_keeps_its_sign(make_frame):
    f = build_payroll_features(make_frame([payslip("P1", net_pay=-500.0)])).iloc[0]
    assert f["net_pay_ratio"] < 0                    # not clipped, not replaced


def test_history_features_use_previous_month_only(base_frame):
    df = base_frame.copy()
    df.loc[(df.employee_id == "E1") & (df.month == "2025-06-01"), "monthly_fixed_gross"] = 100000.0
    f = build_payroll_features(df)
    e1 = f[f.employee_id == "E1"].sort_values("month")
    assert np.isnan(e1.iloc[0]["fixed_gross_change_vs_prev"])            # first month: no history
    assert e1.iloc[1]["fixed_gross_change_vs_prev"] == pytest.approx(1.0)
    assert e1.iloc[2]["fixed_gross_change_vs_prev"] == pytest.approx(2.0)  # June vs May
    assert e1.iloc[3]["fixed_gross_change_vs_prev"] == pytest.approx(0.5)  # July vs June (prior only)


def test_changing_a_future_month_does_not_change_earlier_features(base_frame):
    a = build_payroll_features(base_frame)
    changed = base_frame.copy()
    changed.loc[changed.month == "2025-07-01", "monthly_fixed_gross"] *= 3
    b = build_payroll_features(changed)
    early = base_frame.month < "2025-07-01"
    cols = [
        "fixed_gross_change_vs_prev", "pf_vs_prev_month",
        "role_median_fixed_gross", "fixed_gross_vs_role_median",
    ]
    pd.testing.assert_frame_equal(a.loc[early, cols], b.loc[early, cols])


def test_default_role_reference_uses_only_the_previous_calendar_month(base_frame):
    f = build_payroll_features(base_frame)
    e1 = f[f.employee_id == "E1"].sort_values("month")
    assert np.isnan(e1.iloc[0]["role_median_fixed_gross"])
    assert e1.iloc[1]["role_median_fixed_gross"] == pytest.approx(51000.0)
    assert e1.iloc[1]["fixed_gross_vs_role_median"] == pytest.approx(50000 / 51000)


def test_role_reference_is_fit_then_applied(base_frame):
    ref = RoleReference().fit(base_frame)
    f = build_payroll_features(base_frame, role_reference=ref)
    r1 = f[f.employee_id == "E1"].iloc[0]
    assert r1["role_median_fixed_gross"] == pytest.approx(51000.0)       # median of R1 (50000, 52000)
    with pytest.raises(RuntimeError):
        RoleReference().transform(base_frame)


def test_required_columns_and_duplicate_ids_are_rejected(base_frame):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_payroll_frame(base_frame.drop(columns=["net_pay"]))
    dup = pd.concat([base_frame, base_frame.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate values"):
        validate_payroll_frame(dup)


@pytest.mark.parametrize(
    "column,value,error",
    [
        ("employee_id", None, "missing or blank key values"),
        ("run_id", " ", "missing or blank key values"),
        ("month", "2025-04-02", "first day"),
        ("run_id", "PR-2025-05", "must match their month"),
    ],
)
def test_invalid_payroll_keys_and_months_are_rejected(base_frame, column, value, error):
    df = base_frame.copy()
    df.loc[df.index[0], column] = value
    with pytest.raises(ValueError, match=error):
        validate_payroll_frame(df)


def test_original_columns_and_row_order_are_preserved(base_frame):
    f = build_payroll_features(base_frame)
    assert list(f["payroll_item_id"]) == list(base_frame["payroll_item_id"])
    assert set(base_frame.columns) <= set(f.columns)
