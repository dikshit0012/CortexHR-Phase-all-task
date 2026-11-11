import os
import shutil
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortexhr_task14 import config  # noqa: E402


@pytest.fixture(scope="session")
def loaded():
    task13_root = Path(os.environ.get("CORTEXHR_TASK13_ROOT", config.TASK14_ROOT.parent / "task13"))
    artifact = task13_root / "model_artifacts" / "attrition-v2" / "model.joblib"
    if not artifact.is_file():
        pytest.skip("Task 14 tests require the Task 13 attrition-v2 model artifact.")
    from cortexhr_task14.model_loader import load_attrition_v2
    return load_attrition_v2()


@pytest.fixture(scope="session")
def inputs(loaded):
    if not config.locate_workbook():
        pytest.skip("Task 14 scoring tests require the authorized CortexHR workbook.")
    from cortexhr_task14.employee_scoring import build_scoring_inputs
    return build_scoring_inputs(loaded)


@pytest.fixture(scope="session")
def result(tmp_path_factory):
    """Full pipeline on the real data. Reuses cached REAL SHAP values from reports/internal when present (computing them takes ~12 min)."""
    if not config.locate_workbook():
        pytest.skip("Task 14 pipeline tests require the authorized CortexHR workbook.")
    from cortexhr_task14.pipeline import run
    out = tmp_path_factory.mktemp("reports14")
    cache = config.TASK14_ROOT / "reports" / "internal"
    if (cache / "shap_values.csv").exists():
        (out / "internal").mkdir()
        for f in ("shap_values.csv", "shap_values.meta.json"):
            shutil.copy(cache / f, out / "internal" / f)
    return run(reports_dir=out, stability_check=False, progress=False) | {"dir": out}
