"""Task 6 - labelled comparisons between named variants of the detector.

Everything here is EVALUATION-SIDE reporting. Variants that were created after the evaluation labels had
been seen are named "post-evaluation" and are never used to choose the headline configuration.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List

import pandas as pd

from .detection import run_detection
from .payroll_evaluation import (
    ALL_ERRORS, ID_COL, LABEL_TYPE_COL, baseline_replica_flag, metrics_from_counts, per_type_metrics,
)
from .payroll_model import ModelConfig
from .payroll_features import RoleReference
from .payroll_rules import RuleConfig
from .sensitivity import _three

ORIGINAL_NAME = "Original experiment (v1: file-order duplicate rule, file-order model fit)"
THIS_RUN_NAME = "Current configuration (order-independent duplicate rule, canonical model fit)"
EXPLORATORY_NAME = "POST-EVALUATION exploratory: current + salary-drop suppression (NOT adopted)"


def original_experiment_configs(rule_cfg: RuleConfig, model_cfg: ModelConfig):
    """The v1 behaviour: the only differences are the two order-dependent choices."""
    return replace(rule_cfg, duplicate_selection="file_order"), replace(model_cfg, canonical_row_order=False)


def variant_comparison(raw: pd.DataFrame, labels: pd.DataFrame, rule_cfg: RuleConfig, model_cfg: ModelConfig) -> pd.DataFrame:
    """Overall metrics of rules / model / combined for original, current and exploratory variants."""
    rows: List[dict] = []
    o_rules, o_model = original_experiment_configs(rule_cfg, model_cfg)
    for name, rc, mc in [
        (ORIGINAL_NAME, o_rules, o_model),
        (THIS_RUN_NAME, rule_cfg, model_cfg),
        (EXPLORATORY_NAME, replace(rule_cfg, suppress_drop_after_flagged_spike=True), model_cfg),
    ]:
        ref = RoleReference().fit(raw) if name == ORIGINAL_NAME else None
        feats, flags, _ = run_detection(raw, rc, mc, role_reference=ref)
        rows += _three(name, feats, flags, labels)
    return pd.DataFrame(rows)


def salary_drop_review(raw: pd.DataFrame, labels: pd.DataFrame, rule_cfg: RuleConfig, model_cfg: ModelConfig) -> pd.DataFrame:
    """Before/after table for the salary-drop review (post-evaluation analysis).

    'Original' = salary-drop rule exactly as specified; 'Revised' = same code with the suppression option on.
    Rows: detector-level metrics (rules only, combined) and the salary-drop rule on its own, judged against
    its intended target STALE_SALARY.
    """
    base_cfg = replace(rule_cfg, suppress_drop_after_flagged_spike=False)
    rev_cfg = replace(rule_cfg, suppress_drop_after_flagged_spike=True)
    out = []
    lab = labels.set_index(ID_COL)[LABEL_TYPE_COL]
    for scope in ("rule_salary_drop alone (target: STALE_SALARY)", "Rules only (all rules)", "Combined (rules OR model)"):
        for label, cfg in (("Original implementation", base_cfg), ("Revised (post-evaluation)", rev_cfg)):
            feats, flags, _ = run_detection(raw, cfg, model_cfg)
            ids = feats[ID_COL].values
            if scope.startswith("rule_salary_drop"):
                flagged = set(ids[flags["rule_salary_drop"].to_numpy() == 1])
                pos = set(lab.index[lab == "STALE_SALARY"])
                clean = set(ids) - set(lab.index)
                tp, fp, fn = len(flagged & pos), len(flagged & clean), len(pos - flagged)
                tn = len(clean - flagged)
            else:
                col = "rules_only_flag" if scope.startswith("Rules") else "combined_flag"
                flagged = set(ids[flags[col].to_numpy() == 1])
                pos = set(lab.index)
                tp, fn = len(flagged & pos), len(pos - flagged)
                fp = len(flagged - pos)
                tn = len(set(ids) - pos - flagged)
            m = metrics_from_counts(tp, fp, fn)
            out.append({"scope": scope, "implementation": label, "flagged_records": len(flagged), "true_positives": tp,
                        "false_positives": fp, "false_negatives": fn, "true_negatives": tn, **m})
    return pd.DataFrame(out)


def baseline_replica_seed_table(raw: pd.DataFrame, labels: pd.DataFrame, seeds=range(10)) -> pd.DataFrame:
    """How much the baseline replica's true positives move with the random seed alone."""
    rows = []
    for sd in seeds:
        flag = baseline_replica_flag(raw, contamination=0.025, seed=sd, dup_keep=False)
        r = per_type_metrics("replica", flag, labels).query("error_type == @ALL_ERRORS").iloc[0]
        rows.append({"seed": sd, "flagged": int(flag.sum()), "tp": int(r.tp), "tn": int(r.tn),
                     "precision": r.precision, "recall": r.recall})
    return pd.DataFrame(rows)


def tiered_alert_table(flags: pd.DataFrame, ids, labels: pd.DataFrame) -> pd.DataFrame:
    """EXPERIMENTAL review policy (does not replace any detector): split the combined alerts into
    Tier 1 = at least one business rule fired (explicit, explainable) and
    Tier 2 = flagged ONLY by Isolation Forest (statistical, needs a human to judge).
    Each tier is scored on its own; tier 1 + tier 2 equals the combined detector exactly.
    """
    lab = set(labels[ID_COL])
    ids = pd.Series(ids)
    rules = flags["rules_only_flag"].to_numpy() == 1
    model_only = (flags["if_flag"].to_numpy() == 1) & ~rules
    rows = []
    for name, mask in (("Tier 1: rule alerts (high-confidence, explainable)", rules),
                       ("Tier 2: model-only alerts (review queue)", model_only),
                       ("Tier 1 + Tier 2 (= combined detector)", rules | model_only)):
        flagged = set(ids[mask])
        tp = len(flagged & lab)
        m = metrics_from_counts(tp, len(flagged) - tp, len(lab) - len(flagged & lab))
        rows.append({"alert_tier": name, "alerts": len(flagged), "true_errors_found": tp, "false_alarms": len(flagged) - tp,
                     "precision": m["precision"], "share_of_all_100_errors_found": tp / len(lab) if lab else float("nan")})
    return pd.DataFrame(rows)
