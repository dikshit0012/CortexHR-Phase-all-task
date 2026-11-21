"""Run the whole Task 16 pipeline: python scripts/run_task16.py [--workbook PATH]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortexhr_task16.forecasting import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
