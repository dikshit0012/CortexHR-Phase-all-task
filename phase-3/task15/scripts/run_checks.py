"""Run the test suite and record exact passed / failed / skipped counts in reports/test_summary.json."""
from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs"], cwd=ROOT, capture_output=True, text=True)
tail = [l for l in p.stdout.strip().splitlines() if re.search(r"\d+ (passed|failed|skipped)", l)]
line = tail[-1] if tail else ""
get = lambda w: int(m.group(1)) if (m := re.search(rf"(\d+) {w}", line)) else 0
res = {"passed": get("passed"), "failed": get("failed"), "skipped": get("skipped"), "summary_line": line, "when": datetime.date.today().isoformat()}
(ROOT / "reports" / "test_summary.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2)); raise SystemExit(p.returncode)
