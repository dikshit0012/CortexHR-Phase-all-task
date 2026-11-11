"""Verify Task 13 is unchanged, run the test suite, and record the result in reports/internal/test_summary.json."""
from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_task13_unchanged import verify  # noqa: E402


def main() -> int:
    t13 = verify()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    res = {"passed": passed, "failed": failed, "pytest_summary_line": tail, "when": datetime.date.today().isoformat(),
           "task13_unchanged": t13["unchanged"], "task13_files_checked": t13["files_now"]}
    (ROOT / "reports" / "internal" / "test_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(proc.stdout[-600:] if proc.returncode else "")
    return 0 if (proc.returncode == 0 and t13["unchanged"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
