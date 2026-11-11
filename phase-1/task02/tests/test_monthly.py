"""Tests for src/cortexhr/monthly.py: windows, no look-ahead, no row multiplication, duplicates.

All data here is SYNTHETIC (small hand-made tables with the real column names).
Real-data checks of the same rules live in tests/test_features.py::TestRealDataset.
"""
import numpy as np
import pandas as pd
import pytest

from cortexhr.monthly import add_monthly_features, drop_duplicate_employee_months, window_bounds


def _att(emp, rates, start="2025-07-01", unpaid=0.0):
    months = pd.date_range(start, periods=len(rates), freq="MS")
    return pd.DataFrame({"attendance_feature_id": [f"A-{emp}-{i}" for i in range(len(rates))], "employee_id": emp,
                         "month": months, "attendance_rate": rates, "unpaid_absence_days": unpaid})


def _pay(emp, variable, start="2025-07-01"):
    months = pd.date_range(start, periods=len(variable), freq="MS")
    return pd.DataFrame({"payroll_item_id": [f"P-{emp}-{i}" for i in range(len(variable))], "employee_id": emp,
                         "month": months, "variable_pay": variable})


def _attrition(*ids):
    return pd.DataFrame({"employee_id": list(ids), "observation_date": pd.Timestamp("2025-09-30"),
                         "left_within_12m": [0, 1, 0][:len(ids)]})


class TestWindow:
    def test_last_day_of_month_includes_that_month(self):
        assert window_bounds("2025-09-30") == (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-01"))

    def test_mid_month_uses_only_completed_months(self):
        assert window_bounds("2025-10-15") == (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-01"))
        assert window_bounds("2025-10-31") == (pd.Timestamp("2025-08-01"), pd.Timestamp("2025-10-01"))

    def test_lag_moves_window_back(self):
        assert window_bounds("2025-09-30", lag_months=1) == (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-08-01"))

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            window_bounds("2025-09-30", window_months=0)


class TestJoin:
    def test_future_months_are_ignored(self):
        att = _att("E1", [0.90, 0.92, 0.95, 0.10, 0.10, 0.10])           # Jul-Sep real, Oct-Dec in the future
        pay = _pay("E1", [100, 200, 300, 9999, 9999, 9999])
        out, diag = add_monthly_features(_attrition("E1"), att, pay)
        assert out.loc[0, "avg_attendance_rate_3m"] == pytest.approx((0.90 + 0.92 + 0.95) / 3)
        assert out.loc[0, "avg_variable_pay_3m"] == pytest.approx(200)
        assert diag["latest_month_used"] == "2025-09-01" and diag["no_look_ahead"]

    def test_change_is_latest_minus_earliest(self):
        out, _ = add_monthly_features(_attrition("E1"), _att("E1", [0.90, 0.92, 0.95]), _pay("E1", [1, 1, 1]))
        assert out.loc[0, "attendance_rate_change_3m"] == pytest.approx(0.05)

    def test_one_month_gives_average_but_no_trend(self):
        out, _ = add_monthly_features(_attrition("E1"), _att("E1", [0.9], start="2025-09-01"), _pay("E1", [5], start="2025-09-01"))
        assert out.loc[0, "avg_attendance_rate_3m"] == pytest.approx(0.9)
        assert np.isnan(out.loc[0, "attendance_rate_change_3m"])
        assert out.loc[0, "attendance_months_in_window"] == 1

    def test_missing_employee_is_nan_not_zero(self):
        out, diag = add_monthly_features(_attrition("E1", "HIST-1"), _att("E1", [0.9, 0.9, 0.9]), _pay("E1", [1, 1, 1]))
        row = out[out.employee_id == "HIST-1"].iloc[0]
        assert np.isnan(row["avg_attendance_rate_3m"]) and np.isnan(row["avg_variable_pay_3m"])
        assert row["attendance_months_in_window"] == 0            # a true count of months, not a guessed value
        assert diag["rows_without_attendance_data"] == 1

    def test_no_row_multiplication_and_target_untouched(self):
        attrition = _attrition("E1", "E2", "E3")
        att = pd.concat([_att("E1", [0.9] * 18, start="2025-04-01"), _att("E2", [0.8] * 18, start="2025-04-01")])
        pay = pd.concat([_pay("E1", [1] * 18, start="2025-04-01"), _pay("E2", [2] * 18, start="2025-04-01")])
        out, diag = add_monthly_features(attrition, att, pay)
        assert len(out) == 3 and out["employee_id"].tolist() == ["E1", "E2", "E3"]
        assert out["left_within_12m"].tolist() == attrition["left_within_12m"].tolist()
        assert diag["rows_in"] == diag["rows_out"] == 3
        assert out["attendance_months_in_window"].tolist() == [3, 3, 0]

    def test_partial_history_is_not_padded(self):
        out, _ = add_monthly_features(_attrition("E1"), _att("E1", [0.9, 0.8], start="2025-08-01"), _pay("E1", [1, 2], start="2025-08-01"))
        assert out.loc[0, "attendance_months_in_window"] == 2 and out.loc[0, "avg_attendance_rate_3m"] == pytest.approx(0.85)

    def test_existing_column_clash_is_refused(self):
        attrition = _attrition("E1").assign(avg_variable_pay_3m=1.0)
        with pytest.raises(ValueError, match="already has columns"):
            add_monthly_features(attrition, _att("E1", [0.9]), _pay("E1", [1]))

    def test_non_unique_index_is_refused(self):
        attrition = pd.concat([_attrition("E1"), _attrition("E1")])
        with pytest.raises(ValueError, match="index"):
            add_monthly_features(attrition, _att("E1", [0.9]), _pay("E1", [1]))


class TestDuplicates:
    def test_exact_duplicate_months_are_removed(self):
        pay = _pay("E1", [100, 200, 300])
        dup = pay.iloc[[1]].assign(payroll_item_id="P-D0001")      # same content, different row id
        clean, n = drop_duplicate_employee_months(pd.concat([pay, dup], ignore_index=True), "payroll_monthly_features")
        assert n == 1 and len(clean) == 3

    def test_conflicting_duplicates_raise(self):
        pay = _pay("E1", [100, 200, 300])
        conflict = pay.iloc[[1]].assign(payroll_item_id="P-D0001", variable_pay=999)
        with pytest.raises(ValueError, match="DIFFERENT values"):
            drop_duplicate_employee_months(pd.concat([pay, conflict], ignore_index=True), "payroll_monthly_features")

    def test_duplicates_do_not_inflate_the_average(self):
        pay = _pay("E1", [100, 200, 300])
        pay = pd.concat([pay, pay.iloc[[2]].assign(payroll_item_id="P-D0001")], ignore_index=True)
        out, diag = add_monthly_features(_attrition("E1"), _att("E1", [0.9, 0.9, 0.9]), pay)
        assert out.loc[0, "avg_variable_pay_3m"] == pytest.approx(200) and out.loc[0, "payroll_months_in_window"] == 3
        assert diag["duplicate_rows_removed"]["payroll_monthly_features"] == 1
