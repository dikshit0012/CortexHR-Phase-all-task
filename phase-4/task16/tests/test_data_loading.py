import pandas as pd
import pytest

from cortexhr_task16 import config, data_loader as dl


def test_missing_workbook_is_reported_clearly(tmp_path):
    with pytest.raises(dl.DataError, match="Workbook not found"):
        dl.load_headcount_history(tmp_path / "nope.xlsx")


def test_missing_columns_rejected(tmp_path):
    p = tmp_path / "bad.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame({"month": [1]}).to_excel(w, sheet_name=config.HEADCOUNT_SHEET, index=False)
    with pytest.raises(dl.DataError, match="missing required columns"):
        dl.load_headcount_history(p)


def test_missing_sheet_rejected(tmp_path):
    p = tmp_path / "other.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="other", index=False)
    with pytest.raises(dl.DataError, match="not found"):
        dl.load_workforce_events(p)


def test_real_inputs_have_documented_structure(real_available):
    if not real_available:
        pytest.skip("workbook not available")
    h, e, s = dl.load_headcount_history(), dl.load_workforce_events(), dl.load_sample_forecast()
    assert set(config.REQUIRED_HEADCOUNT_COLS) <= set(h.columns)
    assert list(e.columns) == config.REQUIRED_EVENT_COLS
    assert list(s.columns) == config.SAMPLE_COLUMNS and len(s) == 72
