import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def df():
    from cortexhr_task15 import config
    from cortexhr_task15.data import load_salary_training
    if not config.SALARY_CSV.is_file() and not config.WORKBOOK.is_file():
        pytest.skip(
            "Task 15 data tests require salary_training.csv or the authorized workbook; "
            "set CORTEXHR_SALARY_CSV or CORTEXHR_WORKBOOK."
        )
    return load_salary_training()


@pytest.fixture(scope="session")
def result(tmp_path_factory):
    from cortexhr_task15 import config
    from cortexhr_task15.pipeline import run
    if not config.SALARY_CSV.is_file() and not config.WORKBOOK.is_file():
        pytest.skip(
            "Task 15 pipeline tests require salary_training.csv or the authorized workbook; "
            "set CORTEXHR_SALARY_CSV or CORTEXHR_WORKBOOK."
        )
    out = tmp_path_factory.mktemp("rep15")
    config.MODELS = out / "models"                       # tests never overwrite the committed model files
    return run(reports=out) | {"dir": out}
