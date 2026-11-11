"""Make the Task 5 and Task 6 packages importable without installing them."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def ensure_upstream_on_path() -> None:
    for folder in ("task05/src", "task06/task 6/src"):
        src = str(_REPO_ROOT / folder)
        if src not in sys.path:
            sys.path.insert(0, src)
