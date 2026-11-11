"""Project-relative paths, so no notebook or script needs an absolute path."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
REPORTS = PROJECT_ROOT / "reports"
NOTEBOOKS = PROJECT_ROOT / "notebooks"

# The 2,400-row cortexhr_dataset.xlsx in this folder is a synthetic stand-in.
# Official Task 3 uses attrition_dataset (2,389 rows) in 02_CortexHR_AI_ML_Dataset.xlsx.
# See task 4/files/cortexhr-ml (notebooks/02_baselines.ipynb) for the real run.
WORKBOOK_NAME = "cortexhr_dataset.xlsx"
DATA_SHEET = "employees"
BENCHMARK_SHEET = "baseline_benchmarks"
