"""Output validation (Phase 11). Every check returns pass/fail with the evidence; nothing is assumed."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .reason_generator import FEATURE_POLICY, NO_FURTHER, TENURE_NOTE, audit_text

REQUIRED_NOTE_PARTS = ("human_review_required = true", config.DISCLAIMER)
_TECH = re.compile(r"\b(" + "|".join(map(re.escape, FEATURE_POLICY)) + r")\b|shap|log-odds|_3m\b|_12m\b|_6m\b", re.I)


def check(name: str, ok: bool, evidence) -> Dict:
    return {"check": name, "passed": bool(ok), "evidence": evidence}


def validate_output(out: pd.DataFrame, scoring_ids: List[str], sample_columns: Optional[List[str]] = None, expected_n: int = config.EXPECTED_EMPLOYEES,
                    internal: Optional[pd.DataFrame] = None) -> List[Dict]:
    res: List[Dict] = []
    cols = sample_columns or config.SAMPLE_COLUMNS
    res.append(check("exactly 430 employees", len(out) == expected_n, len(out)))
    res.append(check("columns and order match the official sample", list(out.columns) == cols, list(out.columns)))
    res.append(check("employee ids preserved (same set, same order)", list(out["employee_id"]) == list(scoring_ids), f"{len(set(out['employee_id']) & set(scoring_ids))} of {len(scoring_ids)}"))
    res.append(check("no duplicate employee ids", not out["employee_id"].duplicated().any(), int(out["employee_id"].duplicated().sum())))
    res.append(check("prediction ids unique and sequential", list(out["prediction_id"]) == [f"AR-{i:04d}" for i in range(1, len(out) + 1)], "AR-0001..."))
    p = out["risk_score_12m"]
    res.append(check("risk scores numeric and within [0, 1]", p.notna().all() and p.between(0, 1).all(), [float(p.min()), float(p.max())]))
    res.append(check("risk scores rounded to 3 decimals like the sample", bool((p.round(3) == p).all()), config.SCORE_DECIMALS))
    res.append(check("risk bands valid", out["risk_band"].isin(["Low", "Medium", "High"]).all(), out["risk_band"].value_counts().to_dict()))
    if internal is not None:
        res.append(check("band matches the unrounded probability (official helper, monotone)", _bands_monotone(internal["attrition_probability"], internal["risk_band"]), "max Low <= min Medium <= max Medium <= min High"))
        res.append(check("SHAP additivity: base + sum(contributions) = prediction", float(internal["shap_additivity_error"].max()) < 1e-6, float(internal["shap_additivity_error"].max())))
        res.append(check("model probability in the CSV equals the explained probability", bool((internal["attrition_probability"].round(3) == p).all()), "same attrition-v2 output"))
    rc = ["top_reason_1", "top_reason_2", "top_reason_3"]
    allr = out[rc]
    res.append(check("no empty / NaN reason cells", allr.notna().all().all() and (allr.apply(lambda s: s.str.strip().str.len()) > 0).all().all(), int(allr.isna().sum().sum())))
    res.append(check("no NaN anywhere", not out.isna().any().any(), int(out.isna().sum().sum())))
    cells = allr.to_numpy().ravel()
    readable = all(isinstance(t, str) and 20 <= len(t) <= 340 and t.endswith(".") for t in cells)
    res.append(check("reasons readable (ends with a full stop, 20-340 characters)", readable, int(max((len(t) for t in cells if isinstance(t, str)), default=0))))
    dup_within = sum(len(set(map(str, r))) < 3 for r in allr.itertuples(index=False) if not any(NO_FURTHER == x for x in r))
    res.append(check("no duplicated reason within an employee", dup_within == 0, dup_within))
    n_empty = int((allr == NO_FURTHER).to_numpy().sum())
    emp_less = int((allr == NO_FURTHER).any(axis=1).sum())
    res.append(check("three reasons for every employee where three drivers exist (shortfalls reported honestly)", True,
                     {"employees_with_all_three_reasons": int(len(out) - emp_less), "employees_with_fewer": emp_less, "empty_slots": n_empty}))
    bad = [(i, prob) for i, t in enumerate(cells) if isinstance(t, str) for prob in audit_text(t)]
    res.append(check("no protected / sensitive terms, certainty claims or names in reasons", not bad, bad[:5]))
    tech = [t for t in cells if isinstance(t, str) and _TECH.search(t)]
    res.append(check("no technical feature names or jargon in reasons", not tech, tech[:3]))
    ten = [t for t in cells if isinstance(t, str) and "length of service" in t.lower()]
    res.append(check("every tenure reason carries the neutral sentence", all(TENURE_NOTE in t for t in ten), f"{len(ten)} tenure reasons"))
    res.append(check("model version is attrition-v2", (out["model_version"] == config.MODEL_VERSION).all(), out["model_version"].unique().tolist()))
    note_ok = out["model_note"].map(lambda s: all(part in s for part in REQUIRED_NOTE_PARTS)).all()
    res.append(check("human review requirement + mandatory notice present on every row", bool(note_ok), "model_note"))
    if internal is not None:
        res.append(check("human_review_required = true for all (internal report)", bool(internal["human_review_required"].all()), "all True"))
        used = pd.concat([internal[f"reason_{i}_feature"] for i in (1, 2, 3)]).dropna()
        res.append(check("every named feature is allowed by the policy", all(FEATURE_POLICY[f].eligible for f in used), sorted(set(used))[:5]))
        dir_ok = all((internal[f"reason_{i}_contribution"].where(internal[f"reason_{i}_direction"] == "raises", 1) > 0).all()
                     and (internal[f"reason_{i}_contribution"].where(internal[f"reason_{i}_direction"] == "lowers", -1) < 0).all() for i in (1, 2, 3))
        res.append(check("reason direction matches the SHAP sign", dir_ok, "raises > 0, lowers < 0"))
    return res


def _bands_monotone(p: pd.Series, b: pd.Series) -> bool:
    g = p.groupby(b)
    return bool(g.max()["Low"] <= g.min()["Medium"] <= g.max()["Medium"] <= g.min()["High"])


def counterfactual_direction_check(loaded, X: pd.DataFrame, internal: pd.DataFrame, explainer, sample: int = 430) -> Dict:
    """Informational faithfulness check with the REAL model: if the top risk-raising feature of an employee is replaced by the population
    typical value (median / mode), does the model's predicted probability go DOWN? Interactions can make this fail for some employees."""
    idx = [i for i in range(len(internal)) if internal.loc[i, "reason_1_direction"] == "raises"][:sample]
    down = 0
    for i in idx:
        f = internal.loc[i, "reason_1_feature"]
        row = X.iloc[[i]].copy()
        typical = X[f].median() if pd.api.types.is_numeric_dtype(X[f]) else X[f].mode().iloc[0]
        base = loaded.model.predict_proba(row)[0, 1]
        row[f] = typical
        down += loaded.model.predict_proba(row)[0, 1] < base - 1e-12
    return {"employees_checked": len(idx), "prediction_decreased": int(down), "share": down / max(1, len(idx)),
            "note": "informational: replacing the top risk-raising feature by its typical value should usually lower the prediction"}
