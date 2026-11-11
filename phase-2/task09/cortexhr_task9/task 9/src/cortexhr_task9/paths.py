"""Default locations (relative to the `task 9` folder, so they work on Windows and Linux) and environment overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

TASK9_ROOT = Path(__file__).resolve().parents[2]
ENV_MODEL = "CORTEXHR_INTENT_MODEL"            # full path of the .joblib artifact
ENV_FAQ = "CORTEXHR_FAQ_FILE"                  # path of hr_faq.csv
ENV_THRESHOLD = "CORTEXHR_CHAT_THRESHOLD"      # overrides the threshold stored in the artifact (0 < t < 1)

DEFAULT_MODEL = TASK9_ROOT / "models" / "hr_intent_classifier.joblib"
DEFAULT_FAQ = TASK9_ROOT / "data" / "hr_faq.csv"
TRAIN_CSV = TASK9_ROOT / "data" / "chatbot_nlu_training.csv"
TEST_CSV = TASK9_ROOT / "data" / "chatbot_test_set.csv"
REPORTS = TASK9_ROOT / "reports"
PACKAGE_DIR = Path(__file__).resolve().parent
INTENT_FAQ_MAP = PACKAGE_DIR / "intent_faq_map.json"


def env_path(name: str, default: Path, environ: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if environ is None else environ
    return Path(env[name]).expanduser() if env.get(name) else default
