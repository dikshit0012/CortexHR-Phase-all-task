"""The output must match the official sample sheet: exact column names, exact order, sensible value formats."""
import pandas as pd
import pytest

from cortexhr_task6.data_loading import find_workbook
from cortexhr_task7.official_output import (
    DEFAULT_SCORED_AT, OFFICIAL_COLUMNS, SAMPLE_SHEET, sample_columns, to_official, validate_official_schema,
    validate_official_values,
)
from cortexhr_task7.reason_codes import REASON_CODES

# Copied by hand from the sample sheet header (sheet `sample_anomaly_results`).
EXPECTED = ["result_id", "source_dataset", "record_id", "employee_id", "period", "anomaly_score", "flagged",
            "reason_code", "reason_text", "money_at_risk", "model_version", "scored_at"]


def internal(n=3):
    return pd.DataFrame({
        "source_dataset": ["payroll_monthly_features"] * n, "record_id": [f"PI-{i:06d}" for i in range(n)],
        "employee_id": [f"EMP-{i:04d}" for i in range(n)], "period": pd.to_datetime(["2025-06-01"] * n),
        "anomaly_score": [0.9, 0.5, 0.1][:n], "primary_reason_code": ["ZERO_PAY"] * n,
        "reason_text": ["Please check."] * n, "money_at_risk": [100.0, 50.0, float("nan")][:n],
        "extra_internal_column": ["x"] * n, "signal_count": [1] * n,
    })


def test_official_columns_constant_is_the_expected_list_in_order():
    assert OFFICIAL_COLUMNS == EXPECTED


def test_official_columns_match_the_real_sample_sheet_when_the_workbook_is_available():
    try:
        wb = find_workbook()
        cols = sample_columns(wb)
    except Exception as exc:
        pytest.skip(f"workbook not available: {exc}")
    assert SAMPLE_SHEET == "sample_anomaly_results"
    assert cols == OFFICIAL_COLUMNS


def test_to_official_returns_exactly_the_official_columns_and_drops_internal_ones():
    out = to_official(internal())
    assert list(out.columns) == EXPECTED
    assert "extra_internal_column" not in out.columns and "signal_count" not in out.columns


@pytest.mark.parametrize("mutate", [
    lambda d: d.drop(columns=["money_at_risk"]),                                   # missing column
    lambda d: d.assign(extra=1),                                                   # extra column
    lambda d: d[[*d.columns[1:], d.columns[0]]],                                   # wrong order
    lambda d: d.rename(columns={"reason_code": "reason"}),                         # wrong name
])
def test_schema_validation_rejects_any_deviation(mutate):
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_official_schema(mutate(to_official(internal())))


def test_value_formats_follow_the_sample():
    out = to_official(internal())
    assert out["result_id"].tolist() == ["AN-0001", "AN-0002", "AN-0003"]
    assert (out["flagged"] == "Y").all()
    assert out["period"].tolist() == ["2025-06-01"] * 3
    assert (out["scored_at"] == DEFAULT_SCORED_AT).all() and DEFAULT_SCORED_AT == "2026-10-01 02:00:00"
    assert out["anomaly_score"].between(0, 1).all()
    assert pd.api.types.is_float_dtype(out["money_at_risk"]) and pd.api.types.is_float_dtype(out["anomaly_score"])
    assert out["model_version"].eq("iforest+rules-payroll-task6").all()
    assert validate_official_values(out) == []


def test_empty_money_stays_empty_not_zero():
    out = to_official(internal())
    assert pd.isna(out["money_at_risk"].iloc[2])


@pytest.mark.parametrize("column,bad,message", [
    ("money_at_risk", -5.0, "negative"), ("anomaly_score", 1.5, "outside 0..1"), ("reason_text", "", "empty reason_text"),
    ("reason_code", "", "empty reason_code"), ("flagged", "N", "flagged"), ("period", "June 2025", "YYYY-MM-DD"),
])
def test_value_validation_catches_bad_values(column, bad, message):
    out = to_official(internal())
    out.loc[0, column] = bad
    assert any(message in p for p in validate_official_values(out))


def test_duplicate_record_or_result_id_is_caught():
    out = to_official(internal())
    out.loc[1, "record_id"] = out.loc[0, "record_id"]
    out.loc[2, "result_id"] = out.loc[0, "result_id"]
    problems = validate_official_values(out)
    assert any("record appears more than once" in p for p in problems) and any("duplicate result_id" in p for p in problems)


def test_bad_scored_at_is_rejected():
    with pytest.raises(ValueError, match="scored_at"):
        to_official(internal(), scored_at="yesterday")


def test_reused_reason_codes_exist_in_the_official_sample_vocabulary_when_the_workbook_is_available():
    try:
        wb = find_workbook()
        vocab = set(pd.read_excel(wb, sheet_name=SAMPLE_SHEET)["reason_code"])
    except Exception as exc:
        pytest.skip(f"workbook not available: {exc}")
    claimed = {c for c, rc in REASON_CODES.items() if rc.in_official_sample}
    assert claimed <= vocab, f"codes claimed as official but not in the sample: {claimed - vocab}"
