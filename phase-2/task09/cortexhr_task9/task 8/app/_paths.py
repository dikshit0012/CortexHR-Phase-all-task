"""Make the Task 5, 6 and 7 packages importable without installing them.

The task directories use zero-padded names at the repository root; Tasks 6 and 7
keep their source under a nested project folder.
"""
import os
import sys
from pathlib import Path

TASK8_ROOT = Path(__file__).resolve().parents[1]
ENV_TASKS_ROOT = "CORTEXHR_TASKS_ROOT"
_TASK_ALIASES = {
    "task 5": ("task05", "task 5", "task5"),
    "task 6": ("task06", "task 6", "task6"),
    "task 7": ("task07", "task 7", "task7"),
}


def _source_root() -> Path:
    configured = os.environ.get(ENV_TASKS_ROOT)
    if configured:
        roots = [Path(configured).expanduser().resolve()]
    else:
        roots = []
        current = TASK8_ROOT
        for _ in range(6):
            roots.append(current)
            current = current.parent

    def has_sources(root: Path) -> bool:
        if not root.exists():
            return False
        for folder, aliases in _TASK_ALIASES.items():
            for alias in aliases:
                for base in (root / alias, root / alias.replace(" ", ""), root / folder):
                    for candidate in (base / "src", base / folder / "src", base / alias / "src"):
                        if candidate.is_dir():
                            return True
        return False

    for root in roots:
        if has_sources(root):
            return root

    missing = [folder for folder in _TASK_ALIASES]
    raise RuntimeError(
        f"Task source folders {missing} were not found under the configured project root. "
        f"Set {ENV_TASKS_ROOT} to the repository root containing task05/src, task06/task 6/src, "
        "and task07/task 7/src."
    )


REPO_ROOT = _source_root()


def ensure_upstream_on_path() -> None:
    for folder, aliases in _TASK_ALIASES.items():
        for alias in aliases:
            for base in (REPO_ROOT / alias, REPO_ROOT / alias.replace(" ", ""), REPO_ROOT / folder):
                for candidate in (base / "src", base / folder / "src", base / alias / "src"):
                    if candidate.is_dir():
                        src = str(candidate)
                        if src not in sys.path:
                            sys.path.insert(0, src)
                        break
                else:
                    continue
                break


ensure_upstream_on_path()
