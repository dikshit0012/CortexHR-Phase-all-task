import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortexhr_task11 import paths  # noqa: E402
from cortexhr_task11.pipeline import run  # noqa: E402


@pytest.fixture(scope="session")
def reports_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("reports")


@pytest.fixture(scope="session")
def result(reports_dir):
    """One real run on the supplied data; reports are written to a temp folder (the committed reports/ are untouched)."""
    required = (paths.JOBS_CSV, paths.RESUMES_CSV, paths.LABELS_CSV)
    missing = [path for path in required if not path.is_file()]
    if missing:
        pytest.skip(f"Task 11 end-to-end tests require approved inputs: {', '.join(map(str, missing))}")
    return run(paths.DATA_DIR, reports_dir, write=True)


@pytest.fixture()
def data_copy(tmp_path):
    required = (paths.JOBS_CSV, paths.RESUMES_CSV, paths.LABELS_CSV)
    if any(not path.is_file() for path in required):
        pytest.skip("Task 11 data-copy tests require all three approved input CSVs in task11/data.")
    d = tmp_path / "data"
    shutil.copytree(paths.DATA_DIR, d)
    return d
