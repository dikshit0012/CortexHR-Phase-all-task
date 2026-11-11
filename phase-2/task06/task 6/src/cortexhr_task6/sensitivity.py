"""Task 6 - POST-HOC sensitivity checks. Reporting only: nothing here selects or tunes
the detector. The headline results always use the pre-specified default configuration."""
from __future__ import annotations

from dataclasses import replace
from typing import List

import pandas as pd

from .detection import run_detection
from .payroll_evaluation import ALL_ERRORS, per_type_metrics
from .payroll_features import ID_COL, RoleReference
from .payroll_model import ModelConfig
from .payroll_rules import RULE_COLUMNS, RuleConfig


def _overall(name: str, flag: pd.Series, labels: pd.DataFrame, approach: str) -> dict:
    r = per_type_metrics(name, flag, labels).query("error_type == @ALL_ERRORS").iloc[0]
    return {"variant": name, "approach": approach, "flagged": int(flag.sum()), "tp": int(r.tp), "fp": int(r.fp),
            "fn": int(r.fn), "tn": int(r.tn), "precision": r.precision, "recall": r.recall, "f1": r.f1}


def _three(name: str, feats: pd.DataFrame, flags: pd.DataFrame, labels: pd.DataFrame) -> List[dict]:
    ids = feats[ID_COL].values
    out = []
    for approach, col in [("rules", "rules_only_flag"), ("model", "if_flag"), ("combined", "combined_flag")]:
        out.append(_overall(name, pd.Series(flags[col].values, index=ids), labels, approach))
    return out


def run_sensitivity(raw: pd.DataFrame, labels: pd.DataFrame, base_model: ModelConfig, base_rules: RuleConfig) -> pd.DataFrame:
    rows: List[dict] = []
    for c in (0.01, 0.025, 0.05):
        f, fl, _ = run_detection(raw, base_rules, replace(base_model, contamination=c))
        rows += _three(f"contamination={c}", f, fl, labels)
    for s in (1, 7, 42, 99):
        f, fl, _ = run_detection(raw, base_rules, replace(base_model, random_state=s))
        rows += _three(f"seed={s}", f, fl, labels)
    for sel in ("later_by_item_id", "file_order"):
        f, fl, _ = run_detection(raw, replace(base_rules, duplicate_selection=sel), base_model)
        rows += _three(f"duplicate selection = {sel}", f, fl, labels)
    f, fl, _ = run_detection(raw.iloc[::-1].copy(), base_rules, base_model)
    rows += _three("input row order REVERSED (current config)", f, fl, labels)
    legacy_rules = replace(base_rules, duplicate_selection="file_order")
    legacy_model = replace(base_model, canonical_row_order=False)
    f, fl, _ = run_detection(
        raw.iloc[::-1].copy(), legacy_rules, legacy_model, role_reference=RoleReference().fit(raw)
    )
    rows += _three("input row order REVERSED (ORIGINAL-experiment settings)", f, fl, labels)
    # ablation: drop one rule at a time from the rules-only / combined flags
    f, fl, _ = run_detection(raw, base_rules, base_model)
    for rule in RULE_COLUMNS:
        keep = [r for r in RULE_COLUMNS if r != rule]
        fl2 = fl.copy()
        fl2["rules_only_flag"] = (fl2[keep].sum(axis=1) > 0).astype(int)
        fl2["combined_flag"] = ((fl2["if_flag"] == 1) | (fl2["rules_only_flag"] == 1)).astype(int)
        rows += _three(f"ablation: without {rule}", f, fl2, labels)
    return pd.DataFrame(rows)
