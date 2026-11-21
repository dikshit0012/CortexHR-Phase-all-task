"""Repository-root Vercel entrypoint for the Task 16 forecast API."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_task16_app_path = (
    Path(__file__).resolve().parent / "phase-4" / "task16" / "app.py"
)
_task16_spec = spec_from_file_location("task16_vercel_app", _task16_app_path)
if _task16_spec is None or _task16_spec.loader is None:
    raise ImportError(f"Unable to load Task 16 API entrypoint at {_task16_app_path}")

_task16_module = module_from_spec(_task16_spec)
_task16_spec.loader.exec_module(_task16_module)
app = _task16_module.app
