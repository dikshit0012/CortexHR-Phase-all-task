"""Runtime settings, read from environment variables. Nothing secret has a default value."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

API_VERSION = "1.0.0"
PKG_DIR = Path(__file__).resolve().parent
TASK17_ROOT = PKG_DIR.parent
REPO_ROOT = TASK17_ROOT.parent


def _bool(v: Optional[str], default: bool) -> bool:
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _path(env: Mapping[str, str], name: str) -> Optional[Path]:
    v = env.get(name)
    return Path(v).expanduser() if v else None


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    require_auth: bool = True
    api_keys: tuple = ()                       # never logged, never returned
    cors_origins: tuple = ()
    max_body_bytes: int = 1_000_000
    rate_limit_per_minute: int = 120
    max_batch: int = 100
    artifact_dir: Path = TASK17_ROOT / "artifacts"
    # Secure data locations (Render secret files / mounted disk). None = not provided.
    workbook_path: Optional[Path] = None        # master workbook: anomaly reference history (never committed)
    headcount_csv: Optional[Path] = None        # monthly_headcount_history export (never committed)
    payband_calibration: Optional[Path] = None  # optional JSON with the Task 15 band ratios
    workbook_url: Optional[str] = None          # optional https URL (private storage, signed) to fetch the workbook at startup
    workbook_sha256: Optional[str] = None       # required when workbook_url is used: integrity check of the download
    precomputed_forecast: Path = REPO_ROOT / "task16" / "outputs" / "headcount_forecast.csv"
    events_csv: Path = REPO_ROOT / "task16" / "outputs" / "workforce_event_adjustments.csv"
    faq_file: Path = REPO_ROOT / "task16" / "data" / "hr_faq.csv"
    repo_root: Path = REPO_ROOT
    docs_enabled: bool = True

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if env is None else env
        keys = tuple(k.strip() for k in env.get("CORTEXHR_API_KEYS", "").split(",") if k.strip())
        origins = tuple(o.strip() for o in env.get("CORTEXHR_CORS_ORIGINS", "").split(",") if o.strip())
        d = cls()
        return cls(
            environment=env.get("CORTEXHR_ENV", d.environment),
            require_auth=_bool(env.get("CORTEXHR_REQUIRE_AUTH"), True),
            api_keys=keys, cors_origins=origins,
            max_body_bytes=int(env.get("CORTEXHR_MAX_BODY_BYTES", d.max_body_bytes)),
            rate_limit_per_minute=int(env.get("CORTEXHR_RATE_LIMIT_PER_MINUTE", d.rate_limit_per_minute)),
            max_batch=int(env.get("CORTEXHR_MAX_BATCH", d.max_batch)),
            artifact_dir=_path(env, "CORTEXHR_ARTIFACT_DIR") or d.artifact_dir,
            workbook_path=_path(env, "CORTEXHR_WORKBOOK"),
            headcount_csv=_path(env, "CORTEXHR_HEADCOUNT_CSV"),
            payband_calibration=_path(env, "CORTEXHR_PAYBAND_CALIBRATION"),
            workbook_url=env.get("CORTEXHR_WORKBOOK_URL") or None, workbook_sha256=(env.get("CORTEXHR_WORKBOOK_SHA256") or "").lower() or None,
            precomputed_forecast=_path(env, "CORTEXHR_PRECOMPUTED_FORECAST") or d.precomputed_forecast,
            events_csv=_path(env, "CORTEXHR_EVENTS_CSV") or d.events_csv,
            faq_file=_path(env, "CORTEXHR_FAQ_FILE") or d.faq_file,
            docs_enabled=_bool(env.get("CORTEXHR_DOCS_ENABLED"), True),
        )
