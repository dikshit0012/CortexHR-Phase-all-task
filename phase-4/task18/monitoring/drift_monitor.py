"""Dataset validation and eight-feature PSI drift monitoring (reference = attrition_dataset, current = attrition_scoring_2026).

Privacy: only the monitored feature columns and the id column (used to count duplicates, never stored) are read from disk, and
only aggregates leave this module. No path, id or row value appears in any result or message.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from . import monitoring_config as cfg
from .psi import PSIError, classify_psi, compute_psi
from .status import ALERT, ERROR, NOT_EVALUATED, OK, WATCH, worst


class DatasetUnavailable(Exception):
    """A dataset could not be loaded. `code` is a stable machine-readable reason; there is never a path in it."""

    def __init__(self, which: str, code: str):
        super().__init__(f"{which} dataset unavailable: {code}")
        self.which, self.code = which, code


def _needed_columns(features: Sequence[cfg.Feature]) -> set:
    return {f.name for f in features} | {cfg.ID_COLUMN, cfg.MATCH_COLUMN}


def load_dataset(which: str, sheet: str, csv_env: str, workbook_env: Optional[str] = None,
                 features: Sequence[cfg.Feature] = cfg.FEATURES) -> pd.DataFrame:
    """CSV override (CORTEXHR_REFERENCE_CSV / CORTEXHR_SCORING_CSV, handy in CI) else a sheet of the workbook."""
    needed = _needed_columns(features)
    keep = lambda c: str(c) in needed  # noqa: E731 - only the columns we need are ever loaded
    csv = os.environ.get(csv_env)
    try:
        if csv:
            p = Path(csv).expanduser()
            if not p.is_file():
                raise DatasetUnavailable(which, "csv_not_found")
            return pd.read_csv(p, usecols=keep)
        wb = next((p for p in cfg.workbook_candidates(workbook_env) if p.is_file()), None)
        if wb is None:
            raise DatasetUnavailable(which, "workbook_not_found")
        return pd.read_excel(wb, sheet_name=sheet, usecols=keep)
    except DatasetUnavailable:
        raise
    except ValueError as exc:                                   # pandas: "Worksheet named ... not found"
        raise DatasetUnavailable(which, "sheet_not_found" if "orksheet" in str(exc) else "unreadable") from None
    except Exception:                                           # corrupt file, permission problem, ... (details deliberately dropped)
        raise DatasetUnavailable(which, "unreadable") from None


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (load_dataset("reference", cfg.REFERENCE_SHEET, "CORTEXHR_REFERENCE_CSV", "CORTEXHR_REFERENCE_WORKBOOK"),
            load_dataset("scoring", cfg.SCORING_SHEET, "CORTEXHR_SCORING_CSV", "CORTEXHR_SCORING_WORKBOOK"))


# ------------------------------------------------------------------------------------------------ validation (step 1)
def validate_datasets(reference: Optional[pd.DataFrame], scoring: Optional[pd.DataFrame],
                      features: Sequence[cfg.Feature] = cfg.FEATURES, unavailable: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Structural checks only. Returns {status, message, reference_rows, scoring_rows, checks[], notes[]}."""
    unavailable = unavailable or {}
    checks: List[Dict[str, str]] = []
    notes: List[str] = []
    rows = {"reference": None, "scoring": None}
    for which, df in (("reference", reference), ("scoring", scoring)):
        if df is None:
            checks.append({"name": f"{which}_loaded", "status": NOT_EVALUATED, "detail": unavailable.get(which, "not_loaded")})
            continue
        rows[which] = int(len(df))
        checks.append({"name": f"{which}_loaded", "status": OK, "detail": f"{len(df)} rows"})
        if len(df) < cfg.MIN_ROWS:
            checks.append({"name": f"{which}_min_rows", "status": ALERT, "detail": f"{len(df)} rows < minimum {cfg.MIN_ROWS}"})
        missing = [f.name for f in features if f.name not in df.columns]
        if missing:
            checks.append({"name": f"{which}_monitored_columns", "status": ALERT, "detail": f"missing monitored column(s): {', '.join(missing)}"})
        else:
            checks.append({"name": f"{which}_monitored_columns", "status": OK, "detail": f"all {len(features)} present"})
        if cfg.ID_COLUMN in df.columns:
            dup = int(df[cfg.ID_COLUMN].dropna().duplicated().sum())
            if dup and which == "scoring":                     # a scoring batch must have one row per employee
                checks.append({"name": "scoring_duplicate_ids", "status": WATCH, "detail": f"{dup} duplicated id value(s)"})
        expected = cfg.EXPECTED_ROWS[which]
        if len(df) != expected:
            notes.append(f"{which} has {len(df)} rows; Task 1/14 documented {expected} (informational: data may have been refreshed)")
    status = worst(c["status"] for c in checks)
    msg = {OK: "datasets loaded and structurally valid", WATCH: "datasets usable with warnings", ALERT: "dataset structure check failed",
           NOT_EVALUATED: "a required dataset could not be loaded"}.get(status, "dataset validation did not complete")
    bad = [c for c in checks if c["status"] != OK]
    if bad:
        msg += ": " + "; ".join(f"{c['name']} ({c['detail']})" for c in bad)
    return {"status": status, "message": msg, "reference_rows": rows["reference"], "scoring_rows": rows["scoring"], "checks": checks, "notes": notes}


# ------------------------------------------------------------------------------------------------ PSI run (steps 2-3)
def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


_RANK = {OK: 0, WATCH: 1, ALERT: 2}


def _reconcile(primary: str, training: Optional[str], internal: Optional[float]):
    """Combine the matched-reference status (primary) with the training-reference status. Returns (final_status, note). Never hides an
    unexplained disagreement and never downgrades below WATCH when the training reference says ALERT."""
    if training is None or primary not in _RANK or training not in _RANK or _RANK[training] <= _RANK[primary]:
        return primary, ""
    explained = internal is not None and internal >= cfg.PSI_WATCH
    if not explained:
        return training, f"training-reference status {training} not explained by reference composition (internal PSI {internal if internal is None else round(internal, 4)}): stricter status kept"
    if training == ALERT:
        return (WATCH if _RANK[primary] < 1 else primary), "training-serving skew: PSI vs the full training reference is ALERT, matched reference differs (composition effect); held at WATCH, not hidden"
    return primary, f"training reference {training}; reference sub-populations differ on this feature (composition)"


def run_drift(reference: Optional[pd.DataFrame], scoring: Optional[pd.DataFrame],
              features: Sequence[cfg.Feature] = cfg.FEATURES, reason: str = "datasets not available") -> Dict[str, Any]:
    """One row per feature. `psi`/`status` use the population-matched reference when available (else the full reference, flagged);
    `psi_training_reference` is always the full reference. A feature that cannot be evaluated says so explicitly."""
    rows: List[Dict[str, Any]] = []
    matched = other = None
    if reference is not None and cfg.MATCH_COLUMN in reference.columns:
        m = reference[cfg.MATCH_COLUMN] == cfg.MATCH_VALUE
        matched = reference[m] if int(m.sum()) >= cfg.MIN_MATCHED_ROWS else None
        other = reference[~m] if matched is not None and int((~m).sum()) >= cfg.MIN_ROWS else None
    primary_ref = matched if matched is not None else reference
    for f in features:
        row: Dict[str, Any] = {"feature": f.name, "kind": f.kind, "method": None, "reference_used": None, "reference_n": None, "current_n": None,
                               "training_reference_n": None, "reference_missing_pct": None, "current_missing_pct": None, "psi": None,
                               "status": NOT_EVALUATED, "psi_training_reference": None, "status_training_reference": None,
                               "reference_internal_psi": None, "top_shift_bin": None, "top_shift_reference_pct": None,
                               "top_shift_current_pct": None, "message": ""}
        if reference is None or scoring is None:
            row["message"] = reason
        elif f.name not in reference.columns or f.name not in scoring.columns:
            row["message"] = "monitored column missing from a dataset"
        else:
            try:
                r = compute_psi(primary_ref[f.name], scoring[f.name], f.kind)
                full = r if matched is None else compute_psi(reference[f.name], scoring[f.name], f.kind)
                internal = None
                if matched is not None and other is not None:
                    try:
                        internal = compute_psi(other[f.name], matched[f.name], f.kind).psi
                    except PSIError:
                        internal = None
            except PSIError as exc:
                row.update(status=ERROR, message=f"PSI could not be computed: {exc}")
            except Exception as exc:                            # never let one feature stop the others, never report it as OK
                row.update(status=ERROR, message=f"unexpected {type(exc).__name__} while computing PSI")
            else:
                primary_status = classify_psi(r.psi)
                training_status = classify_psi(full.psi)
                final, note = _reconcile(primary_status, training_status if matched is not None else None, internal)
                row.update(method=r.method, reference_used="matched" if matched is not None else "full", reference_n=r.reference_n,
                           current_n=r.current_n, training_reference_n=full.reference_n, reference_missing_pct=_pct(r.reference_missing, r.reference_n),
                           current_missing_pct=_pct(r.current_missing, r.current_n), psi=r.psi, status=final, psi_training_reference=full.psi,
                           status_training_reference=training_status, reference_internal_psi=internal, top_shift_bin=r.top_bin,
                           top_shift_reference_pct=r.top_bin_reference_pct, top_shift_current_pct=r.top_bin_current_pct)
                msg = []
                if final in (WATCH, ALERT) or note:
                    msg.append(f"PSI {r.psi:.4f} vs {row['reference_used']} reference; largest shift in bin {r.top_bin}: {r.top_bin_reference_pct}% -> {r.top_bin_current_pct}%")
                if matched is not None and training_status != primary_status:
                    msg.append(f"vs training reference {full.psi:.4f} {training_status}")
                if note:
                    msg.append(note)
                row["message"] = "; ".join(msg)
        rows.append(row)
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in (OK, WATCH, ALERT, NOT_EVALUATED, ERROR)}
    evaluated = [r for r in rows if r["psi"] is not None]
    top = max(evaluated, key=lambda r: r["psi"]) if evaluated else None
    return {"status": worst(r["status"] for r in rows), "features_checked": len(rows), "counts": counts, "features": rows,
            "max_psi": top["psi"] if top else None, "max_psi_feature": top["feature"] if top else None,
            "reference_policy": {"primary": "matched" if matched is not None else "full",
                                 "matched_rows": None if matched is None else int(len(matched)), "training_rows": None if reference is None else int(len(reference)),
                                 "matched_rule": f"{cfg.MATCH_COLUMN} = {cfg.MATCH_VALUE}", "min_matched_rows": cfg.MIN_MATCHED_ROWS},
            "thresholds": {"ok_below": cfg.PSI_WATCH, "watch_up_to_inclusive": cfg.PSI_ALERT, "alert_above": cfg.PSI_ALERT}}
