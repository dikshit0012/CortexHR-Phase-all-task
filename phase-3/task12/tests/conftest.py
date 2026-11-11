import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def audit(tmp_path_factory):
    from cortexhr_task12.fairness_audit import run
    out = tmp_path_factory.mktemp("reports12")
    return {"res": run(out, write=True, stress_step=100), "dir": out}
