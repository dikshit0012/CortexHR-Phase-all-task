"""Service settings, read from environment variables. Nothing here touches the data or the models."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from ._paths import TASK8_ROOT

ENV_WORKBOOK = "CORTEXHR_WORKBOOK"            # same variable name as Task 6 (cortexhr_task6.data_loading.ENV_VAR)
ENV_MODEL_DIR = "CORTEXHR_MODEL_DIR"
ENV_MAX_BATCH = "CORTEXHR_MAX_BATCH_SIZE"
DEFAULT_MODEL_DIR = TASK8_ROOT / "models"
DEFAULT_MAX_BATCH_SIZE = 100
MAX_BATCH_CEILING = 1000                       # a configured value above this is rejected at startup


class StartupError(RuntimeError):
    """A required file, artifact or setting is missing or unusable. The service must not start."""


@dataclass(frozen=True)
class Settings:
    workbook_path: Optional[Path]
    model_dir: Path = DEFAULT_MODEL_DIR
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.max_batch_size, int) or not 1 <= self.max_batch_size <= MAX_BATCH_CEILING:
            raise StartupError(f"{ENV_MAX_BATCH} must be an integer between 1 and {MAX_BATCH_CEILING}, got {self.max_batch_size!r}")

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if environ is None else environ
        raw_batch = env.get(ENV_MAX_BATCH, str(DEFAULT_MAX_BATCH_SIZE))
        try:
            batch = int(raw_batch)
        except ValueError:
            raise StartupError(f"{ENV_MAX_BATCH} must be an integer, got {raw_batch!r}") from None
        workbook = env.get(ENV_WORKBOOK)
        return cls(
            workbook_path=Path(workbook) if workbook else None,
            model_dir=Path(env[ENV_MODEL_DIR]) if env.get(ENV_MODEL_DIR) else DEFAULT_MODEL_DIR,
            max_batch_size=batch,
        )
