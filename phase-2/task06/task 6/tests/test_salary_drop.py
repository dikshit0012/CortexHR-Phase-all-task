"""rule_salary_drop: behaviour of the original rule and of the post-evaluation suppression hypothesis."""
import numpy as np
import pandas as pd
import pytest

from conftest import payslip
from cortexhr_task6.payroll_features import build_payroll_features
from cortexhr_task6.payroll_rules import RuleConfig, apply_rules

ORIGINAL = RuleConfig()
EXPERIMENT = RuleConfig(suppress_drop_after_flagged_spike=True)       # post-evaluation hypothesis, off by default


def series(emp, fixed_by_month, first_month=1, role="R1"):
    """One employee: fixed gross per month (None = no payslip that month, np.nan = payslip with missing value)."""
    rows = []
    for i, fx in enumerate(fixed_by_month):
        if fx is None:
            continue
        m = first_month + i
        row = payslip(f"{emp}-{m:02d}", emp=emp, month=f"2025-{m:02d}-01", role=role, fixed=50000.0 if np.isnan(fx) else fx)
        if np.isnan(fx):
            row["monthly_fixed_gross"] = np.nan
        rows.append(row)
    return rows


def drop_flags(rows, cfg):
    f = build_payroll_features(pd.DataFrame(rows))
    r = apply_rules(f, cfg)
    return pd.Series(r["rule_salary_drop"].to_numpy(), index=f["payroll_item_id"].to_numpy())


def test_default_is_the_original_rule():
    assert RuleConfig().suppress_drop_after_flagged_spike is False


def test_genuine_unexpected_decrease_is_flagged():
    rows = series("E1", [50000, 50000, 44000]) + series("E2", [50000] * 3, role="R1")
    for cfg in (ORIGINAL, EXPERIMENT):
        assert drop_flags(rows, cfg)["E1-03"] == 1


def test_decrease_with_no_previous_spike_is_flagged_even_by_the_experiment():
    rows = series("E1", [50000, 50000, 50000, 45000])
    assert drop_flags(rows, EXPERIMENT)["E1-04"] == 1


def test_normal_increase_followed_by_return_to_baseline_is_flagged():
    rows = series("E1", [50000, 55000, 50000])                       # +10% is not a spike, so the later fall is not suppressed
    for cfg in (ORIGINAL, EXPERIMENT):
        assert drop_flags(rows, cfg)["E1-03"] == 1


def test_flagged_spike_then_return_is_flagged_by_original_but_suppressed_by_experiment():
    rows = series("E1", [50000, 50000, 160000, 50000])
    assert drop_flags(rows, ORIGINAL)["E1-04"] == 1
    assert drop_flags(rows, EXPERIMENT)["E1-04"] == 0


def test_experiment_would_hide_a_stale_salary_after_a_genuine_large_raise():
    """WHY the suppression is not adopted: a real 80% raise (flagged as a jump) followed by payroll
    reverting to the OLD salary is a genuine stale-salary error, and the suppression would hide it."""
    rows = series("E1", [50000, 50000, 90000, 50000])
    assert drop_flags(rows, ORIGINAL)["E1-04"] == 1
    assert drop_flags(rows, EXPERIMENT)["E1-04"] == 0                # hidden error - documented risk


def test_drop_to_a_different_level_after_a_spike_is_still_flagged_by_the_experiment():
    rows = series("E1", [50000, 50000, 160000, 100000])               # not back to the pre-spike level
    assert drop_flags(rows, EXPERIMENT)["E1-04"] == 1


def test_missing_previous_month_record_gives_no_flag():
    rows = series("E1", [50000, None, 40000])                         # no February payslip
    for cfg in (ORIGINAL, EXPERIMENT):
        assert drop_flags(rows, cfg)["E1-03"] == 0


def test_missing_salary_value_gives_no_flag_and_no_crash():
    rows = series("E1", [50000, np.nan, 50000, 40000])
    for cfg in (ORIGINAL, EXPERIMENT):
        flags = drop_flags(rows, cfg)
        assert flags["E1-02"] == 0 and flags["E1-03"] == 0           # NaN current / NaN previous -> undefined, not flagged
        assert flags["E1-04"] == 1


@pytest.mark.parametrize("cfg", [ORIGINAL, EXPERIMENT])
def test_out_of_order_rows_give_identical_results(cfg):
    rows = series("E1", [50000, 50000, 160000, 50000]) + series("E2", [60000, 60000, 54000, 54000])
    a = drop_flags(rows, cfg)
    rng = np.random.default_rng(3)
    b = drop_flags([rows[i] for i in rng.permutation(len(rows))], cfg).reindex(a.index)
    assert a.equals(b)


@pytest.mark.parametrize("cfg", [ORIGINAL, EXPERIMENT])
def test_interleaved_employees_do_not_contaminate_each_other(cfg):
    e1 = series("E1", [50000, 50000, 160000, 50000])                  # spike then return
    e2 = series("E2", [60000, 60000, 60000, 54000])                   # plain 10% drop, no spike
    e3 = series("E3", [70000, 70000, 70000, 70000])                   # stable
    interleaved = [r for quad in zip(e1, e2, e3) for r in quad]
    flags = drop_flags(interleaved, cfg)
    assert flags["E2-04"] == 1                                        # E1's spike must not suppress E2's drop
    assert flags["E3-04"] == 0 and flags["E3-03"] == 0
    assert flags["E1-04"] == (1 if cfg is ORIGINAL else 0)
