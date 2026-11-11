"""Fit the Task 5 and Task 6 detectors ONCE on the reference workbook and save the artifacts.

    python scripts/build_artifacts.py --workbook path\to\02_CortexHR_AI_ML_Dataset.xlsx
    (or set CORTEXHR_WORKBOOK)   -> task 8/models/attendance-anomaly-v1.joblib, payroll-anomaly-v1.joblib (+ metadata json)

This uses the existing training pipelines (no labels are read). It is the only place a model is fitted; the API only loads.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.artifacts import build_artifacts  # noqa: E402
from app.config import DEFAULT_MODEL_DIR, ENV_MODEL_DIR, ENV_WORKBOOK, StartupError  # noqa: E402
from app.reference import load_reference  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workbook", default=os.environ.get(ENV_WORKBOOK), help=f"reference workbook (default: ${ENV_WORKBOOK})")
    p.add_argument("--model-dir", default=os.environ.get(ENV_MODEL_DIR) or str(DEFAULT_MODEL_DIR))
    args = p.parse_args()
    try:
        reference = load_reference(Path(args.workbook) if args.workbook else None)
        meta = build_artifacts(reference, Path(args.model_dir))
    except StartupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Saved artifacts to {args.model_dir}")
    for kind, m in meta.items():
        print(f"  {m['model_version']}: reference rows={m['reference']['rows']}, features kept={len(m['kept_features'])}, "
              f"score scale=[{m['score_scale']['min']:.6f}, {m['score_scale']['max']:.6f}], "
              f"Python={m['python_version']}, NumPy={m['numpy_version']}, pandas={m['pandas_version']}, "
              f"SciPy={m['scipy_version']}, scikit-learn={m['sklearn_version']}, "
              f"joblib={m['joblib_version']}, openpyxl={m['openpyxl_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
