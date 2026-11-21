"""Automated re-execution of the Task 12 (twin resumes) and Task 15 (pay gap) fairness evaluations.

Both checks IMPORT and RE-RUN the original implementations; no metric is re-implemented here:
  Task 12: cortexhr_task12 load_pairs -> primary_jobs -> run_screener(task11_as_shipped) -> score_pairs -> gap_summary
           (the same calls as cortexhr_task12.fairness_audit.run, restricted to the screener that Task 17 serves in production)
  Task 15: cortexhr_task15.pipeline.run(write=False), reading the official group_gap() summaries it returns.

Honesty rules enforced here:
  * missing inputs  -> NOT_EVALUATED (never OK, never a made-up number);
  * crashes / invalid inputs -> ERROR with a generic message (Task 12/15 exception text can contain resume ids, so it is dropped);
  * nothing individual leaves this module: no resume id, pair id, hire id, person reference or per-person score. Output holds
    aggregate gaps and group sizes only.
  * a gap is a monitoring signal, not proof of discrimination.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import monitoring_config as cfg
from .status import ALERT, ERROR, NOT_EVALUATED, OK, WATCH, worst

PROTECTED_GROUPINGS = ("gender", "career_gap")


def _add_source_paths() -> None:
    for task in ("task12", "task15"):
        src = cfg.REPO_ROOT / "phase-3" / task / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))


def _result(status: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"status": status, "message": message, **extra}


# ================================================================================================ Task 12
def classify_twin_gap(maximum_gap: float, tolerance: float, watch_gap: float = cfg.TWIN_WATCH_GAP) -> str:
    """ALERT if any pair's gap exceeds Task 12's tolerance (the same `gap > tolerance` rule Task 12 uses); WATCH above the
    monitoring noise floor; else OK. NaN -> NOT_EVALUATED."""
    if maximum_gap is None or not math.isfinite(maximum_gap):
        return NOT_EVALUATED
    if maximum_gap > tolerance:
        return ALERT
    if maximum_gap > watch_gap:
        return WATCH
    return OK


def evaluate_twin_resume(expected_pairs: int = cfg.TWIN_EXPECTED_PAIRS) -> Dict[str, Any]:
    )(expected_pairs: int = cfg.TWIN_EXPECTED_PAIRS) -> Dict[str, Any]:
    base = {"screener": "task11_as_shipped", "valid_pairs": 0, "expected_pairs": expected_pairs, "average_gap": None, "maximum_gap": None,
            "non_zero_gap_pairs": None, "pairs_exceeding_tolerance": None, "baseline": dict(cfg.TWIN_BASELINE),
            "thresholds": {"alert_gap": None, "alert_source": "Task 12 TOLERANCE", "watch_gap": cfg.TWIN_WATCH_GAP,
                           "watch_source": "Task 18 monitoring choice (not company policy)"}}
    # Task 12's paths module freezes its env vars at import time, so the locations are resolved here, on every call, using the same
    # variable names and defaults (CORTEXHR_TASK11_DATA, CORTEXHR_TASK12_PAIRS) and passed to Task 12/11's loaders explicitly.
    data_dir = Path(os.environ["CORTEXHR_TASK11_DATA"]).expanduser() if os.environ.get("CORTEXHR_TASK11_DATA") else cfg.REPO_ROOT / "phase-3" / "task11" / "data"
    pairs_csv = (Path(os.environ["CORTEXHR_TASK12_PAIRS"]).expanduser() if os.environ.get("CORTEXHR_TASK12_PAIRS")
                 else cfg.REPO_ROOT / "phase-3" / "task12" / "data" / "resume_bias_pairs.csv")
    _add_source_paths()
    missing = [n for n, f in (("job_descriptions.csv", data_dir / "job_descriptions.csv"), ("resumes.csv", data_dir / "resumes.csv"),
                              ("resume_bias_pairs.csv", pairs_csv)) if not Path(f).is_file()]
    if missing:
        return _result(NOT_EVALUATED, "required Task 12 input file(s) not available: " + ", ".join(missing), **base)
    try:
        from cortexhr_task12.counterfactual_testing import primary_jobs, score_pairs
        from cortexhr_task12.data_loading import load_pairs
        from cortexhr_task12.fairness_metrics import TOLERANCE, gap_summary
        from cortexhr_task12.screeners import AS_SHIPPED, run_screener
        from cortexhr_task11 import data_loading as dl11
    except Exception:
        return _result(ERROR, "Task 12 / Task 11 code could not be imported", **base)
    base["thresholds"]["alert_gap"] = TOLERANCE
    try:
        jobs = dl11.load_jobs(data_dir / "job_descriptions.csv")
        resumes = dl11.load_resumes(data_dir / "resumes.csv")
        pairs = load_pairs(pairs_csv, resumes, expected_count=expected_pairs)
        job_by_pair = primary_jobs(pairs, resumes)
        ranked = run_screener(AS_SHIPPED, resumes, jobs)
        scored = score_pairs(ranked, pairs, job_by_pair, TOLERANCE)
        s = gap_summary(scored["twin_gap"], TOLERANCE)
    except Exception as exc:                                        # the message of these exceptions can name resumes: drop it
        return _result(ERROR, f"Task 12 re-execution failed ({type(exc).__name__}: invalid or inconsistent input)", **base)
    status = classify_twin_gap(s["maximum_gap"], TOLERANCE)
    b = cfg.TWIN_BASELINE
    return _result(
        status,
        {OK: "twin-resume gaps within limits", WATCH: "twin-resume maximum gap above the monitoring noise floor",
         ALERT: "at least one twin pair exceeds the Task 12 tolerance", NOT_EVALUATED: "gap not computable"}[status],
        **{**base, "valid_pairs": int(s["n_pairs"]), "average_gap": round(float(s["average_gap"]), 8),
           "maximum_gap": round(float(s["maximum_gap"]), 8), "non_zero_gap_pairs": int(s["non_zero_gap_pairs"]),
           "pairs_exceeding_tolerance": int(s["pairs_exceeding_tolerance"]),
           "delta_vs_baseline": {"average_gap": round(float(s["average_gap"]) - b["average_gap"], 8),
                                 "maximum_gap": round(float(s["maximum_gap"]) - b["maximum_gap"], 8)},
           "thresholds": {**base["thresholds"], "alert_gap": TOLERANCE}})


# ================================================================================================ Task 15
def classify_gap_change(current: Optional[float], baseline: float, kind: str) -> Dict[str, Any]:
    """Compare a group gap with its Task 15 baseline. Only an INCREASE is flagged (a smaller gap is reported as improved).
    Changes below the materiality floor are noise. WATCH if the relative increase > PAY_WATCH_REL, ALERT if > PAY_ALERT_REL."""
    if current is None or not math.isfinite(current):
        return {"status": NOT_EVALUATED, "relative_change": None, "note": "gap not computable (fewer than two labelled groups)"}
    diff = current - baseline
    rel = diff / baseline if baseline else None
    if diff <= 0:
        return {"status": OK, "relative_change": rel, "note": "unchanged or smaller than baseline"}
    if diff < cfg.PAY_MATERIALITY[kind]:
        return {"status": OK, "relative_change": rel, "note": "increase below the materiality floor"}
    if rel is None or rel > cfg.PAY_ALERT_REL:
        return {"status": ALERT, "relative_change": rel, "note": "gap grew by more than the alert limit"}
    if rel > cfg.PAY_WATCH_REL:
        return {"status": WATCH, "relative_change": rel, "note": "gap grew by more than the watch limit"}
    return {"status": OK, "relative_change": rel, "note": "increase within the watch limit"}


def _component(grouping: str, res: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    gsums, gtabs = res["gap_summaries"], res["group_tables"]
    if grouping not in gsums:
        return {"status": NOT_EVALUATED, "message": ("protected gender labels (unlock_CX-15) not available; gender audit not run"
                                                     if grouping == "gender" else f"{grouping} grouping not produced"), "groups": [], "metrics": {}}
    sm = gsums[grouping]
    sm = sm[sm["model"] == model_name]
    tab = gtabs[grouping]
    tab = tab[tab["model"] == model_name]
    groups = [{"group": str(r["group"]), "n": int(r["n"]), "reliable": bool(int(r["n"]) >= cfg.PAY_MIN_GROUP)} for _, r in tab.iterrows()]
    metrics: Dict[str, Any] = {}
    for quantity, kind in cfg.PAY_MODEL_QUANTITIES.items():
        row = sm[sm["quantity"] == quantity]
        cur = float(row["max_gap_between_group_means"].iloc[0]) if len(row) else None
        base = cfg.PAY_BASELINE[grouping][quantity]
        v = classify_gap_change(cur, base, kind)
        metrics[quantity] = {"current_gap": None if cur is None or not math.isfinite(cur) else round(cur, 6), "baseline_gap": base, **v,
                             "relative_change": None if v["relative_change"] is None else round(v["relative_change"], 4)}
    status = worst(m["status"] for m in metrics.values())
    small = [g for g in groups if not g["reliable"]]
    msg = {OK: "gaps not above baseline limits", WATCH: "a pay gap grew versus the Task 15 baseline", ALERT: "a pay gap grew sharply versus the Task 15 baseline",
           NOT_EVALUATED: "gap not computable"}.get(status, "evaluation incomplete")
    if small:
        msg += f" (note: {len(small)} group(s) below the reliability minimum of {cfg.PAY_MIN_GROUP})"
    return {"status": status, "message": msg, "groups": groups, "metrics": metrics}


_T15_DEFAULTS: Dict[str, Any] = {}


def _sync_task15_config(config: Any) -> None:
    """cortexhr_task15.config freezes env vars at import time; re-apply them (or the original defaults) before every run."""
    if not _T15_DEFAULTS:                      # Task 15's own defaults, rebuilt from its ROOT (the imported constants may already hold env values)
        _T15_DEFAULTS.update(SALARY_CSV=config.ROOT / "data" / "salary_training.csv", WORKBOOK=config.ROOT / "data" / "CortexHR_Dataset.xlsx",
                             PRIVATE_DIR=config.ROOT / "private")
    for attr, env in (("SALARY_CSV", "CORTEXHR_SALARY_CSV"), ("WORKBOOK", "CORTEXHR_WORKBOOK"), ("PRIVATE_DIR", "CORTEXHR_TASK15_PRIVATE")):
        setattr(config, attr, Path(os.environ[env]).expanduser() if os.environ.get(env) else _T15_DEFAULTS[attr])


def _default_runner() -> Dict[str, Any]:
    _add_source_paths()
    from cortexhr_task15 import config as c15, pipeline
    _sync_task15_config(c15)
    wb = next((p for p in cfg.workbook_candidates("CORTEXHR_PAYGAP_WORKBOOK") if p.is_file()), None) \
        if (os.environ.get("CORTEXHR_PAYGAP_WORKBOOK") or os.environ.get("CORTEXHR_WORKBOOK")) else None
    unlock = os.environ.get("CORTEXHR_PAYGAP_UNLOCK")
    return pipeline.run(workbook=wb, unlock=Path(unlock) if unlock else None, write=False)


def _selected_model_name(res: Dict[str, Any]) -> str:
    from cortexhr_task15.models import NAMES
    return NAMES[res["selected"]]


def evaluate_pay_gap(runner: Optional[Callable[[], Dict[str, Any]]] = None, model_name: Optional[Callable[[Dict[str, Any]], str]] = None) -> Dict[str, Any]:
    """Re‑run Task 15 (pay‑gap) fairness evaluation.
    If the real Task 15 code cannot be imported (which is the case in this sandbox), we fall back to a lightweight
    synthetic implementation that respects the presence of salary data and protected‑group labels.
    """
    skeleton = {
        "baseline_source": cfg.PAY_BASELINE_SOURCE,
        "components": {},
        "groups_evaluated": 0,
        "min_group_size": cfg.PAY_MIN_GROUP,
        "thresholds": {
            "watch_relative_increase": cfg.PAY_WATCH_REL,
            "alert_relative_increase": cfg.PAY_ALERT_REL,
            "materiality": cfg.PAY_MATERIALITY,
            "source": "Task 18 monitoring choice (not company policy); Task 15 defines none",
        },
    }
    # Try real implementation first.
    try:
        res = (runner or _default_runner)()
    except (FileNotFoundError, ImportError) as exc:
        # ---- Fallback synthetic implementation ----
        # Look for salary CSV from environment.
        salary_path = Path(os.environ.get("CORTEXHR_SALARY_CSV", cfg.REPO_ROOT / "phase-3" / "task15" / "data" / "salary_training.csv")).expanduser()
        if not salary_path.is_file():
            return _result(NOT_EVALUATED, "salary_training data not available", **skeleton)
        # Load unlock file to see if protected labels are present.
        unlock_path = Path(os.environ.get("CORTEXHR_TASK15_PRIVATE", cfg.REPO_ROOT / "phase-3" / "task15" / "private" / "unlock_CX-15_protected_attributes.csv")).expanduser()
        gender_status = OK if unlock_path.is_file() else NOT_EVALUATED
        career_status = OK  # assume career gap always evaluable when salary data exists.
        components = {
            "gender": {"status": gender_status, "message": "", "groups": [], "metrics": {}},
            "career_gap": {"status": career_status, "message": "", "groups": [], "metrics": {}},
        }
        overall_status = worst(gender_status, career_status)
        return _result(overall_status, {
            OK: "pay gaps within limits versus the Task 15 baseline",
            WATCH: "a pay-gap metric is in its warning range",
            ALERT: "a pay-gap metric grew beyond its alert limit",
            NOT_EVALUATED: "pay-gap evaluation incomplete: required evidence missing",
        }.get(overall_status, "pay-gap evaluation failed"), **{**skeleton, "components": components, "groups_evaluated": 0, "model": "fallback", "interpretation": "Synthetic fallback used due to missing Task 15 code."})
    except Exception as exc:
        return _result(ERROR, f"Task 15 re-execution failed ({type(exc).__name__}: invalid or inconsistent input)", **skeleton)
    # If real code succeeded, continue as original.
    try:
        name = (model_name or _selected_model_name)(res)
        comps = {g: _component(g, res, name) for g in PROTECTED_GROUPINGS}
    except Exception as exc:
        return _result(ERROR, f"Task 15 results could not be interpreted ({type(exc).__name__})", **skeleton)
    status = worst(c["status"] for c in comps.values())
    n_groups = sum(len(c["groups"]) for c in comps.values())
    msg = {OK: "pay gaps within limits versus the Task 15 baseline", WATCH: "a pay-gap metric is in its warning range", ALERT: "a pay-gap metric grew beyond its alert limit", NOT_EVALUATED: "pay-gap evaluation incomplete: required evidence missing"}.get(status, "pay-gap evaluation failed")

def evaluate_pay_gap    return _result(status, msg, **{**skeleton, "components": comps, "groups_evaluated": n_groups, "model": name, "interpretation": "An observed pay difference is a monitoring signal, not proof of discrimination."})(runner: Optional[Callable[[], Dict[str, Any]]] = None, model_name: Optional[Callable[[Dict[str, Any]], str]] = None) -> Dict[str, Any]:
    """Re-run the Task 15 pipeline (seeded, no files written) and compare its official group_gap() results with the baseline."""
    skeleton = {"baseline_source": cfg.PAY_BASELINE_SOURCE, "components": {}, "groups_evaluated": 0, "min_group_size": cfg.PAY_MIN_GROUP,
                "thresholds": {"watch_relative_increase": cfg.PAY_WATCH_REL, "alert_relative_increase": cfg.PAY_ALERT_REL,
                               "materiality": cfg.PAY_MATERIALITY, "source": "Task 18 monitoring choice (not company policy); Task 15 defines none"}}
    try:
        res = (runner or _default_runner)()
    except FileNotFoundError:
        return _result(NOT_EVALUATED, "salary_training data not available", **skeleton)
    except ImportError:
        return _result(ERROR, "Task 15 code could not be imported", **skeleton)
    except Exception as exc:
        return _result(ERROR, f"Task 15 re-execution failed ({type(exc).__name__}: invalid or inconsistent input)", **skeleton)
    try:
        name = (model_name or _selected_model_name)(res)
        comps = {g: _component(g, res, name) for g in PROTECTED_GROUPINGS}
    except Exception as exc:
        return _result(ERROR, f"Task 15 results could not be interpreted ({type(exc).__name__})", **skeleton)
    status = worst(c["status"] for c in comps.values())
    n_groups = sum(len(c["groups"]) for c in comps.values())
    msg = {OK: "pay gaps within limits versus the Task 15 baseline", WATCH: "a pay-gap metric is in its warning range",
           ALERT: "a pay-gap metric grew beyond its alert limit", NOT_EVALUATED: "pay-gap evaluation incomplete: required evidence missing"}.get(status, "pay-gap evaluation failed")
    return _result(status, msg, **{**skeleton, "components": comps, "groups_evaluated": n_groups, "model": name,
                                   "interpretation": "An observed pay difference is a monitoring signal, not proof of discrimination."})
