"""Evidence for the `total_experience_years` review (README section 'total_experience_years'). Uses the REAL attrition-v2 model and the
REAL SHAP values already computed by run_task14.py. Writes reports/internal/total_experience_analysis.json.

Everything here is about the MODEL's behaviour. The dataset has no age / date-of-birth column, so any link to age is NOT measured."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from cortexhr_task14 import config
from cortexhr_task14.employee_scoring import build_scoring_inputs
from cortexhr_task14.model_loader import load_attrition_v2, mv
from cortexhr_task14.reason_generator import FEATURE_POLICY
from cortexhr_task14.risk_band_assignment import assign_risk_bands

F = "total_experience_years"


def main(reports: Path = config.REPORTS) -> dict:
    loaded = load_attrition_v2()
    inputs = build_scoring_inputs(loaded)
    X = inputs.X
    internal = pd.read_csv(reports / "internal" / "attrition_scoring_internal.csv")
    shap = internal.filter(like="shap__").rename(columns=lambda c: c.replace("shap__", ""))
    base = loaded.model.predict_proba(X)[:, 1]
    out: dict = {"feature": F, "employees": len(X)}

    # 1. importance
    imp = shap.abs().mean().sort_values(ascending=False)
    out["importance"] = {"rank_of_33": int(list(imp.index).index(F) + 1), "mean_abs_shap_pp": float(imp[F] * 100), "share_of_total_abs_shap": float(imp[F] / imp.sum()),
                         "top_features_pp": {k: float(v * 100) for k, v in imp.head(5).items()},
                         "direction": {"spearman_value_vs_shap": float(spearmanr(X[F], shap[F])[0]), "meaning": "positive = more recorded experience -> higher model risk"}}

    # 2. what it is correlated with (in the scoring data)
    tenure_years = X["tenure_months"] / 12
    prev_years = X[F] - tenure_years
    out["relationships_in_scoring_data"] = {
        "spearman_with_tenure_months": float(spearmanr(X[F], X["tenure_months"])[0]), "spearman_with_job_level": float(spearmanr(X[F], X["job_level"])[0]),
        "spearman_with_annual_fixed_pay": float(spearmanr(X[F], X["annual_fixed_pay"])[0]), "spearman_with_previous_employers_count": float(spearmanr(X[F], X["previous_employers_count"])[0]),
        "experience_less_than_tenure_rows": int((prev_years < -1e-9).sum()), "min_max_total_experience_years": [float(X[F].min()), float(X[F].max())],
        "median_total_experience_years": float(X[F].median())}

    # 3. coefficient inside each of the 5 members (per 1 SD of the scaled feature) vs the other features
    coefs = []
    for m in loaded.model.calibrated_classifiers_:
        pipe = m.estimator
        ct = pipe.named_steps["features"].steps[-1][1]
        names = list(ct.get_feature_names_out())
        w = pd.Series(pipe.named_steps["model"].coef_[0], index=names)
        coefs.append(w[[n for n in names if F in n and "missing" not in n]].sum())
    out["logistic_coefficient_per_sd_by_member"] = [float(c) for c in coefs]

    # 4. influence on predictions: set the feature to the cohort median for everyone (the model itself is untouched; this is a what-if)
    def what_if(col):
        Z = X.copy()
        Z[col] = X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else X[col].mode().iloc[0]
        return loaded.model.predict_proba(Z)[:, 1]
    p_neutral = what_if(F)
    b0, b1 = assign_risk_bands(pd.Series(base)), assign_risk_bands(pd.Series(p_neutral))
    top0, top1 = set(np.argsort(-base)[:43]), set(np.argsort(-p_neutral)[:43])
    out["what_if_everyone_at_median_experience"] = {
        "mean_abs_change_pp": float(np.abs(p_neutral - base).mean() * 100), "max_abs_change_pp": float(np.abs(p_neutral - base).max() * 100),
        "spearman_scores": float(spearmanr(base, p_neutral)[0]), "employees_whose_band_changes": int((b0.values != b1.values).sum()),
        "top43_overlap": len(top0 & top1), "note": "model unchanged; only the input is altered for this what-if"}
    ranking = {c: float(np.abs(what_if(c) - base).mean() * 100) for c in X.columns}
    order = sorted(ranking, key=lambda k: -ranking[k])
    out["what_if_mean_abs_change_pp_all_features_top8"] = {k: ranking[k] for k in order[:8]}
    out["what_if_rank_of_feature"] = order.index(F) + 1

    # 5. exposure in explanations
    raising = shap.where(shap > 0)
    top3 = raising.apply(lambda r: set(r.nlargest(3).dropna().index), axis=1)
    top1 = raising.idxmax(axis=1)
    cand = shap.drop(columns=[F])
    out["if_it_were_not_suppressed"] = {"employees_with_it_in_top3_risk_raising": int(sum(F in s for s in top3)), "employees_with_it_as_top_driver": int((top1 == F).sum()),
                                        "employees_where_it_is_a_risk_raising_contribution_of_at_least_0.5pp": int((shap[F] >= config.MATERIALITY).sum()),
                                        "mean_abs_contribution_pp": float(shap[F].abs().mean() * 100)}
    excl = [f for f, s in FEATURE_POLICY.items() if not s.eligible]
    out["all_excluded_features"] = {"features": excl, "mean_abs_shap_pp_total": float(shap[excl].abs().sum(axis=1).mean() * 100),
                                    "employees_where_largest_contributor_is_excluded": int(shap.abs().idxmax(axis=1).isin(excl).sum()),
                                    "of_which_total_experience_years": int((shap.abs().idxmax(axis=1) == F).sum())}

    # 6. association in the TRAINING table (not causal): univariate AUC and the sign after controlling for tenure and level
    train = pd.read_excel(mv.config.WORKBOOK, sheet_name="attrition_dataset")
    train = train[train["left_within_12m"].notna() & (train["tenure_months"] >= 0)].drop_duplicates(subset=[c for c in train.columns if c not in ("row_id",)])
    y = train["left_within_12m"].astype(int)
    cols = ["tenure_months", "job_level", F]
    Z = StandardScaler().fit_transform(train[cols].fillna(train[cols].median()))
    lr = LogisticRegression(max_iter=1000).fit(Z, y)
    rng = np.random.RandomState(0)
    boot = []
    for _ in range(300):
        i = rng.randint(0, len(y), len(y))
        boot.append(LogisticRegression(max_iter=1000).fit(Z[i], y.iloc[i]).coef_[0])
    boot = np.array(boot)
    out["training_table_association"] = {
        "rows": int(len(y)), "leavers": int(y.sum()), "univariate_auc_total_experience_years": float(roc_auc_score(y, train[F].fillna(train[F].median()))),
        "coef_per_sd_with_tenure_and_job_level": dict(zip(cols, map(float, lr.coef_[0]))),
        "bootstrap_95ci_experience_coef": [float(np.percentile(boot[:, 2], 2.5)), float(np.percentile(boot[:, 2], 97.5))],
        "note": "association in the training table, not causation; tenure and job level are correlated with experience"}
    out["age_data_available"] = False
    (reports / "internal" / "total_experience_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
