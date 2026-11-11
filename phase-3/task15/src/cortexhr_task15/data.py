"""Loading, target detection, validation and the dataset audit for `salary_training`."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config


class DataError(ValueError):
    pass


def load_salary_training(workbook: Optional[Path] = None, csv: Optional[Path] = None) -> pd.DataFrame:
    csv, workbook = Path(csv or config.SALARY_CSV), Path(workbook or config.WORKBOOK)
    if csv.exists():
        return pd.read_csv(csv, parse_dates=["hire_date"])
    if workbook.exists():
        return pd.read_excel(workbook, sheet_name=config.SHEET)
    raise FileNotFoundError(f"salary_training not found. Looked for {csv} and sheet '{config.SHEET}' in {workbook}. "
                            "Set CORTEXHR_WORKBOOK (master workbook) or CORTEXHR_SALARY_CSV.")


def detect_target(df: pd.DataFrame, descriptions: Optional[Dict[str, str]] = None) -> str:
    """The column documented as TARGET in the data dictionary; otherwise a single numeric column named like a salary offer. Never guessed silently."""
    if descriptions:
        marked = [c for c, d in descriptions.items() if c in df.columns and str(d).strip().upper().startswith("TARGET")]
        if len(marked) == 1:
            return marked[0]
    cands = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and any(k in c.lower() for k in ("offered", "salary", "ctc")) and "band" not in c.lower()]
    if len(cands) == 1:
        return cands[0]
    raise DataError(f"cannot identify a single salary target column (candidates: {cands})")


def validate(df: pd.DataFrame, target: str = config.TARGET) -> Dict:
    if target not in df.columns:
        raise DataError(f"target column {target!r} missing")
    y = pd.to_numeric(df[target], errors="coerce")
    problems = []
    if y.isna().any():
        problems.append(f"{int(y.isna().sum())} missing/non-numeric salaries")
    if (y <= 0).any():
        problems.append(f"{int((y <= 0).sum())} zero/negative salaries")
    if not np.isfinite(y.dropna()).all():
        problems.append("non-finite salaries")
    if config.ID_COL in df.columns and df[config.ID_COL].duplicated().any():
        problems.append(f"{int(df[config.ID_COL].duplicated().sum())} duplicate ids")
    missing = [c for c in config.ALLOWED if c not in df.columns]
    if missing:
        problems.append(f"missing feature columns {missing}")
    if problems:
        raise DataError("; ".join(problems))
    return {"rows": len(df), "duplicate_rows": int(df.duplicated().sum()), "missing_by_column": {c: int(n) for c, n in df.isna().sum().items() if n}}


def audit(df: pd.DataFrame) -> Dict:
    y = df[config.TARGET]
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    band_ratio = y / (df["band_mid_2026"] * df["market_index"]) if {"band_mid_2026", "market_index"} <= set(df.columns) else None
    counts = lambda column: {str(k): int(v) for k, v in df[column].value_counts(dropna=False).items()} if column in df else {}
    return {
        "rows": len(df), "columns": list(df.columns), "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_by_column": {c: int(n) for c, n in df.isna().sum().items() if n}, "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[config.ID_COL].duplicated().sum()), "target": config.TARGET,
        "target_summary": {k: float(v) for k, v in y.describe().items()}, "target_skew": float(y.skew()), "target_log_skew": float(np.log(y).skew()),
        "iqr_outliers": int(((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum()),
        "iqr_outlier_bounds": {"low": float(q1 - 1.5 * iqr), "high": float(q3 + 1.5 * iqr)}, "non_positive": int((y <= 0).sum()),
        "roles": int(df["role_id"].nunique()), "departments": int(df["department_id"].nunique()), "min_rows_per_role": int(df["role_id"].value_counts().min()),
        "role_distribution": counts("role_id"), "department_distribution": counts("department_id"),
        "job_levels": {int(k): int(v) for k, v in df["job_level"].value_counts().sort_index().items()},
        "work_locations": counts("work_location"), "employment_type_present": "employment_type" in df.columns,
        "employment_type_distribution": counts("employment_type"), "source_counts": counts("source"),
        "hire_year_range": [int(df["hire_year"].min()), int(df["hire_year"].max())] if "hire_year" in df else None,
        "prior_experience_summary": {k: float(v) for k, v in df["prior_experience_years"].describe().items()},
        "career_gap_months_counts": {int(k): int(v) for k, v in df["career_gap_months"].value_counts().sort_index().items()},
        "career_gap_share": float((df["career_gap_months"] > 0).mean()),
        "band_circularity": None if band_ratio is None else {"mean_target_over_band_mid_x_index": float(band_ratio.mean()), "sd": float(band_ratio.std()),
                                                            "log_correlation": float(np.corrcoef(np.log(y), np.log(df["band_mid_2026"] * df["market_index"]))[0, 1])},
        "gender_in_table": any("gender" in c.lower() for c in df.columns), "age_in_table": any(("age" in c.lower().split("_") or "dob" in c.lower()) for c in df.columns),
        "sensitive_columns_in_table": [c for c in df.columns if c.lower() in config.SENSITIVE_NEVER_FEATURES],
        "historical_salary_field": [c for c in df.columns if "prior_salary" in c.lower() or "last_salary" in c.lower() or "current_salary" in c.lower()],
    }
