"""Task 6 - label-free detection pipeline: features -> rules -> model -> combined flag."""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from .payroll_features import RoleReference, build_payroll_features
from .payroll_model import ModelConfig, PayrollAnomalyModel
from .payroll_rules import RULE_COLUMNS, RULE_REASONS, RuleConfig, apply_rules


def combine_flags(model_out: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """combined_flag = Isolation Forest flag OR any rule flag. Every source column is kept.

    rules_only_flag : 1 if at least one rule fired        (Approach 1)
    if_flag         : 1 if Isolation Forest flagged       (Approach 2)
    combined_flag   : rules_only_flag OR if_flag          (Approach 3)
    """
    out = pd.concat([model_out, rules], axis=1)
    out["rule_count"] = out[RULE_COLUMNS].sum(axis=1)
    out["rules_only_flag"] = (out["rule_count"] > 0).astype(int)
    out["combined_flag"] = ((out["if_flag"] == 1) | (out["rules_only_flag"] == 1)).astype(int)

    def reasons(row) -> str:
        parts = []
        if row["if_flag"] == 1:
            parts.append("isolation forest: unusual combination of payroll ratios")
        parts += [RULE_REASONS[c] for c in RULE_COLUMNS if row[c] == 1]
        return "; ".join(parts)

    out["detection_reasons"] = out.apply(reasons, axis=1)
    return out


def run_detection(
    raw: pd.DataFrame,
    rule_cfg: Optional[RuleConfig] = None,
    model_cfg: Optional[ModelConfig] = None,
    role_reference: Optional[RoleReference] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, PayrollAnomalyModel]:
    """End to end and label-free. Returns (feature_frame, flags, fitted_model)."""
    feats = build_payroll_features(raw, role_reference=role_reference)
    model = PayrollAnomalyModel(model_cfg)
    model_out = model.fit_predict(feats)
    flags = combine_flags(model_out, apply_rules(feats, rule_cfg))
    return feats, flags, model
