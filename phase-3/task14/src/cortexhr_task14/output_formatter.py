"""Final CSV in the exact `sample_attrition_risk` schema, the 10-row sample, the internal report and the SHAP charts."""
from __future__ import annotations

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import shap                       # noqa: E402

from . import config                                   # noqa: E402
from .reason_generator import FEATURE_POLICY           # noqa: E402


def read_sample_schema(workbook) -> List[str]:
    """Column names/order of the official sample sheet (the source of truth); re-checked against config.SAMPLE_COLUMNS."""
    cols = list(pd.read_excel(workbook, sheet_name=config.SAMPLE_SHEET, nrows=0).columns)
    if cols != config.SAMPLE_COLUMNS:
        raise ValueError(f"official sample columns changed: {cols}")
    return cols


def model_note() -> str:
    """The caveat column (the sample's `model_note`, 'Caveat shown with the prediction') carries the human-review requirement."""
    return (f"attrition-v2 (calibrated logistic regression). HUMAN REVIEW REQUIRED (human_review_required = true). "
            f"{config.RISK_BAND_NOTE} Reasons name only job-related factors; the model also uses other inputs that are deliberately never named. "
            f"{config.DISCLAIMER}")


def build_output(meta: pd.DataFrame, scores: pd.DataFrame, bands: pd.Series, reasons: List[List[str]]) -> pd.DataFrame:
    n = len(meta)
    if not (len(scores) == len(bands) == len(reasons) == n):
        raise ValueError("components have different lengths")
    out = pd.DataFrame({
        "prediction_id": [f"AR-{i:04d}" for i in range(1, n + 1)],                 # sequential, as in the sample
        "employee_id": meta["employee_id"].to_numpy(),
        "department_id": meta["department_id"].to_numpy(),
        "as_of_date": meta["as_of_date"].to_numpy(),
        "risk_score_12m": scores["attrition_probability"].round(config.SCORE_DECIMALS).to_numpy(),
        "risk_band": bands.to_numpy(),
        "top_reason_1": [r[0] for r in reasons], "top_reason_2": [r[1] for r in reasons], "top_reason_3": [r[2] for r in reasons],
        "model_version": scores["model_version"].to_numpy(),
        "model_note": model_note(),
    })
    return out[config.SAMPLE_COLUMNS]


def pick_sample(out: pd.DataFrame, quotas: Dict[str, int] = None) -> pd.DataFrame:
    """10 representative rows across bands (High 4 / Medium 3 / Low 3), evenly spaced by score inside each band. Deterministic."""
    quotas = quotas or {"High": 4, "Medium": 3, "Low": 3}
    parts = []
    for band, q in quotas.items():
        g = out[out["risk_band"] == band].sort_values(["risk_score_12m", "employee_id"], ascending=[False, True])
        if len(g) == 0:
            continue
        pos = np.unique(np.linspace(0, len(g) - 1, min(q, len(g))).round().astype(int))
        parts.append(g.iloc[pos])
    return pd.concat(parts).reset_index(drop=True)


def internal_report(meta, scores, bands, shap_res, selections: List[Dict], X: pd.DataFrame) -> pd.DataFrame:
    """Technical detail kept OUT of the official CSV."""
    df = pd.DataFrame({"employee_id": meta["employee_id"].to_numpy(), "attrition_probability": scores["attrition_probability"].to_numpy(),
                       "model_version": scores["model_version"].to_numpy(), "risk_band": bands.to_numpy(), "human_review_required": True,
                       "shap_base_value": shap_res.base_value, "shap_additivity_error": shap_res.additivity_error})
    df["n_risk_raising_reasons"] = [s["n_risk_raising"] for s in selections]
    df["n_risk_lowering_reasons"] = [s["n_risk_lowering"] for s in selections]
    df["n_empty_reason_slots"] = [s["n_empty"] for s in selections]
    df["suppressed_abs_contribution"] = [s["suppressed_abs_contribution"] for s in selections]
    df["data_gap_features_not_named"] = ["; ".join(s["data_gap_features"]) for s in selections]
    for i in range(3):
        df[f"reason_{i + 1}_feature"] = [s["reasons"][i].feature for s in selections]
        df[f"reason_{i + 1}_direction"] = [s["reasons"][i].direction for s in selections]
        df[f"reason_{i + 1}_contribution"] = [s["reasons"][i].contribution for s in selections]
    shap_cols = shap_res.values.add_prefix("shap__").reset_index(drop=True)
    return pd.concat([df.reset_index(drop=True), shap_cols], axis=1)


def feature_importance(shap_values: pd.DataFrame) -> pd.DataFrame:
    imp = shap_values.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index, "label": [FEATURE_POLICY[f].label for f in imp.index], "mean_abs_shap": imp.values,
                         "mean_abs_shap_percentage_points": imp.values * 100,
                         "named_in_reasons": [FEATURE_POLICY[f].eligible for f in imp.index],
                         "share_of_total": imp.values / imp.sum()})


def _display(f: str) -> str:
    s = FEATURE_POLICY[f]
    return s.label + ("" if s.eligible else " *")


def plot_summary(shap_values: pd.DataFrame, X: pd.DataFrame, path) -> None:
    """SHAP beeswarm for the numeric features (colour = feature value). Categorical features appear in the bar chart."""
    num = [c for c in shap_values.columns if pd.api.types.is_numeric_dtype(X[c])]
    top = shap_values[num].abs().mean().sort_values(ascending=False).index[:20]
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[top].to_numpy(), X[top].astype(float).to_numpy(), feature_names=[_display(f) for f in top],
                      max_display=20, show=False, plot_size=None)
    plt.xlabel("SHAP value: effect on the predicted 12-month attrition probability\n(0.01 = 1 percentage point; right = raises the prediction)", fontsize=10)
    plt.title("What influences attrition-v2 across the 430 active employees\n(numeric features; * = used by the model but never named in employee reasons)", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_bar(shap_values: pd.DataFrame, path, top_n: int = 20) -> None:
    imp = feature_importance(shap_values).head(top_n).iloc[::-1]
    colors = ["#1f77b4" if e else "#b0b0b0" for e in imp["named_in_reasons"]]
    plt.figure(figsize=(10, 7))
    plt.barh([_display(f) for f in imp["feature"]], imp["mean_abs_shap_percentage_points"], color=colors)
    plt.xlabel("Average absolute SHAP value\n(percentage points of predicted attrition probability)", fontsize=10)
    plt.title("Most influential features (top 20 of the 33 model inputs)\nblue = may be named in reasons, grey * = in the model but never named (sensitive or proxy)", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
