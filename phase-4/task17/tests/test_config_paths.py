import importlib
import os
import sys
from pathlib import Path

from cortexhr_api.config import PHASE4_ROOT, REPO_ROOT, Settings
from cortexhr_api.registry import _add_paths, load_task8_package


def test_default_paths_distinguish_repository_and_phase4_roots():
    settings = Settings()
    assert settings.repo_root == REPO_ROOT
    assert settings.precomputed_forecast == PHASE4_ROOT / "task16" / "outputs" / "headcount_forecast.csv"
    assert (REPO_ROOT / "phase-2" / "task08" / "task 8" / "app" / "__init__.py").is_file()


def test_task8_package_resolves_from_repository_root(monkeypatch):
    monkeypatch.delenv("CORTEXHR_TASKS_ROOT", raising=False)

    load_task8_package(REPO_ROOT)
    module = importlib.import_module("task8_app._paths")

    assert Path(module.__file__).resolve() == REPO_ROOT / "phase-2" / "task08" / "task 8" / "app" / "_paths.py"
    assert module.REPO_ROOT.resolve() == REPO_ROOT.resolve()


def test_upstream_paths_use_monorepo_layout(monkeypatch):
    monkeypatch.delenv("CORTEXHR_TASKS_ROOT", raising=False)

    _add_paths(Settings())

    assert os.environ["CORTEXHR_TASKS_ROOT"] == str(REPO_ROOT)
    assert str(PHASE4_ROOT / "task16" / "src") in sys.path
