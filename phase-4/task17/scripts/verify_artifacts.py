#!/usr/bin/env python3
"""Check that every required artifact exists, loads with the installed libraries, and matches its metadata. Exit 1 on any problem.

    python scripts/verify_artifacts.py            # uses the same settings/env as the API
"""
import hashlib
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cortexhr_api.config import Settings  # noqa: E402
from cortexhr_api.registry import ModelRegistry  # noqa: E402


def main() -> int:
    s = Settings.from_env()
    warnings.filterwarnings("ignore")
    print(f"Artifact directory: {s.artifact_dir}")
    for f in sorted(s.artifact_dir.glob("*.joblib")):
        print(f"  {f.name:48s} {f.stat().st_size:>9,d} bytes  sha256={hashlib.sha256(f.read_bytes()).hexdigest()[:16]}")
    reg = ModelRegistry().load_all(s)
    print()
    bad = 0
    for n, e in reg.entries.items():
        mark = "OK  " if e.loaded else "MISS"
        bad += not e.loaded
        print(f"[{mark}] {n:20s} {e.version or '-':32s} {e.mode or ''}")
        if e.detail:
            print(f"        {e.detail}")
        if e.compatibility:
            print(f"        compat: {e.compatibility}")
    print(f"\n{7 - bad}/7 capabilities available")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
