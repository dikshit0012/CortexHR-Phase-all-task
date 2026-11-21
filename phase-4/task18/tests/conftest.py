import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

PRIVATE_ENV = ("CORTEXHR_WORKBOOK", "CORTEXHR_REFERENCE_WORKBOOK", "CORTEXHR_SCORING_WORKBOOK", "CORTEXHR_PAYGAP_WORKBOOK", "CORTEXHR_REFERENCE_CSV",
               "CORTEXHR_SCORING_CSV", "CORTEXHR_TASK11_DATA", "CORTEXHR_TASK12_PAIRS", "CORTEXHR_TASK12_PRIVATE", "CORTEXHR_SALARY_CSV",
               "CORTEXHR_TASK15_PRIVATE", "CORTEXHR_PAYGAP_UNLOCK", "CORTEXHR_MONITOR_OPTIONAL_CHECKS", "CORTEXHR_MONITORING_DIR")


@pytest.fixture(autouse=True)
def no_private_data(monkeypatch):
    """Every test starts with NO private data configured, so a test can never silently use someone's real files."""
    for name in PRIVATE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def drift_csvs(tmp_path, monkeypatch):
    import synthetic

    def make(ref_kwargs=None, cur_kwargs=None, n_ref=2000, n_cur=1000):
        synthetic.attrition_frame(n_ref, seed=1, **(ref_kwargs or {})).to_csv(tmp_path / "ref.csv", index=False)
        synthetic.attrition_frame(n_cur, seed=2, **(cur_kwargs or {})).to_csv(tmp_path / "cur.csv", index=False)
        monkeypatch.setenv("CORTEXHR_REFERENCE_CSV", str(tmp_path / "ref.csv"))
        monkeypatch.setenv("CORTEXHR_SCORING_CSV", str(tmp_path / "cur.csv"))
    return make
