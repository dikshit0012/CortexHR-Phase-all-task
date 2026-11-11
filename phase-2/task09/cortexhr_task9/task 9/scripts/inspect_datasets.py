"""Inspect Task 9 training, test and FAQ data without modifying any source dataset."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortexhr_task9.inspection import inspect_datasets  # noqa: E402
from cortexhr_task9.paths import DEFAULT_FAQ, REPORTS, TEST_CSV, TRAIN_CSV  # noqa: E402


def main() -> None:
    report = inspect_datasets(TRAIN_CSV, TEST_CSV, DEFAULT_FAQ)
    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "dataset_inspection.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {output.name}")
    print(json.dumps({key: report[key] for key in ("training", "test", "faq", "training_test_overlap")}, indent=2))


if __name__ == "__main__":
    main()
