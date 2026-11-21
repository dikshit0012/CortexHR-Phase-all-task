import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortexhr_task16 import config  # noqa: E402


def make_history(n=60, depts=("D01", "D02"), seed=0) -> pd.DataFrame:
    """SYNTHETIC fixture used only to test code logic (never as training data for any reported result)."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2021-01-01", periods=n, freq="MS")
    rows, per = [], {}
    for d in depts:
        start, hc = 20, []
        hires = rng.integers(2, 6, n); exits = rng.integers(0, 3, n)
        cur = start
        for i in range(n):
            s = cur; cur = s + hires[i] - exits[i]
            hc.append(dict(month=months[i], department_id=d, department_name=d, headcount_end=cur, hires=int(hires[i]), exits=int(exits[i]),
                           voluntary_exits=int(exits[i]), transfers_in=0, transfers_out=0, headcount_start=s, hiring_drive_flag=0,
                           attrition_rate_monthly_pct=round(exits[i] / s * 100, 2), appraisal_cycle_flag=0, financial_year_end_flag=0))
        per[d] = pd.DataFrame(hc)
    tot = sum(per[d].set_index("month")[["headcount_end", "hires", "exits", "headcount_start"]] for d in depts).reset_index()
    allrows = pd.DataFrame({"month": tot["month"], "department_id": "ALL", "department_name": "Company total", "headcount_end": tot["headcount_end"],
                            "hires": tot["hires"], "exits": tot["exits"], "voluntary_exits": tot["exits"], "transfers_in": 0, "transfers_out": 0,
                            "headcount_start": tot["headcount_start"], "hiring_drive_flag": 0,
                            "attrition_rate_monthly_pct": (tot["exits"] / tot["headcount_start"] * 100).round(2), "appraisal_cycle_flag": 0, "financial_year_end_flag": 0})
    return pd.concat([allrows] + list(per.values()), ignore_index=True)


@pytest.fixture
def synth():
    return make_history()


@pytest.fixture
def events_df():
    return pd.DataFrame({
        "event_id": ["EV-A", "EV-B", "EV-C"], "start_date": ["2021-04-01", "2022-01-01", "2022-06-01"], "end_date": ["2021-05-31", "2022-02-28", "2022-06-30"],
        "department_id": ["ALL", "D01", "ALL"], "event_name": ["Hiring freeze", "Engineering hiring drive", "Bonus cycle"],
        "description": ["Hiring stopped", "Hiring doubled", "Bonus paid"]})


@pytest.fixture
def trend_series():
    idx = pd.date_range("2020-01-01", periods=48, freq="MS")
    return pd.Series(100 + 2.0 * np.arange(48), index=idx, name="headcount_end")


@pytest.fixture(scope="session")
def real_available():
    return config.WORKBOOK.is_file()


@pytest.fixture(scope="session")
def result(real_available):
    if not real_available:
        pytest.skip("real-workbook tests need CORTEXHR_WORKBOOK or data/CortexHR_Dataset.xlsx")
    from cortexhr_task16.forecasting import run
    return run(write=False)
