"""Proves Task 14 uses the Task 13 attrition-v2 artifact UNCHANGED: SHA-256 of the three artifact files and of every Task 13 source file
must equal the values recorded in docs/task13_sha256.json (taken from the delivered task_13.zip). __pycache__ is ignored.
    python scripts/verify_task13_unchanged.py            # verify
    python scripts/verify_task13_unchanged.py --record   # (re)record from the current Task 13 folder - only for a deliberate new baseline"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexhr_task14 import config  # noqa: E402

RECORD = config.TASK14_ROOT / "docs" / "task13_sha256.json"


def hashes(root: Path) -> dict:
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def verify(root: Path | None = None) -> dict:
    root = root or config.locate_task13_root()
    expected, now = json.loads(RECORD.read_text())["files"], hashes(root)
    changed = sorted(k for k in expected if k in now and expected[k] != now[k])
    missing = sorted(set(expected) - set(now))
    added = sorted(set(now) - set(expected))
    return {"task13_root": str(root), "files_recorded": len(expected), "files_now": len(now), "changed": changed, "missing": missing, "added": added,
            "model_joblib_sha256": now.get("model_artifacts/attrition-v2/model.joblib"), "unchanged": not (changed or missing or added)}


if __name__ == "__main__":
    if "--record" in sys.argv:
        RECORD.write_text(json.dumps({"source": "delivered task_13.zip", "files": hashes(config.locate_task13_root())}, indent=1))
        print("recorded", RECORD)
    else:
        r = verify()
        print(json.dumps(r, indent=1))
        raise SystemExit(0 if r["unchanged"] else 1)
