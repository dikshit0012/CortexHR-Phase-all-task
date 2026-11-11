import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortexhr_task5.evaluation import find_workbook


def test_find_workbook_accepts_explicit_path(tmp_path):
    workbook = tmp_path / "approved.xlsx"
    workbook.touch()

    assert find_workbook(workbook) == workbook.resolve()


def test_find_workbook_uses_environment_override(tmp_path, monkeypatch):
    workbook = tmp_path / "approved.xlsx"
    workbook.touch()
    monkeypatch.setenv("CORTEXHR_WORKBOOK", str(workbook))

    assert find_workbook() == workbook.resolve()


def test_find_workbook_reports_missing_configured_path(tmp_path, monkeypatch):
    missing = tmp_path / "missing.xlsx"
    monkeypatch.setenv("CORTEXHR_WORKBOOK", str(missing))

    with pytest.raises(FileNotFoundError, match="CortexHR workbook not found"):
        find_workbook()
