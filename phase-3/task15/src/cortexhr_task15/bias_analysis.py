"""Historical-pay-bias investigation and mitigation experiments. Everything is computed; nothing is assumed."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config
from .fairness import career_gap_groups, gap_summary, group_gap_checked, proxy_check
from .models import BOOSTING, build_model, cv_oof, make_preprocessor, metrics


def _design(df: pd.DataFrame, extra: pd.DataFrame, numeric=config.NUMERIC, categorical=config.CATEGORICAL) -> np.ndarray:
    prep = make_preprocessor(numeric, categorical)
    Z = prep.fit_transform(df[numeric + categorical])
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    return np.hstack([Z, extra.to_numpy(float)])


def historical_pay_gap(df: pd.DataFrame, group: pd.Series, reference: str, n_boot: int = 300) -> pd.DataFrame:
    """DESCRIPTIVE: OLS of log(historical pay) on the job factors plus group dummies. exp(coef)-1 = pay difference of the group versus the reference
    group among people with the same role, level, department, experience, location, education and hiring-year price level. Bootstrap 95% interval."""
    ok = group.notna()
    d, g = df[ok], group[ok]
    levels = [x for x in sorted(g.unique()) if x != reference]
    extra = pd.DataFrame({f"is_{x}": (g == x).astype(float) for x in levels}, index=d.index)
    X, y = _design(d, extra), np.log(d[config.TARGET].to_numpy(float))
    k = len(levels)
    base = LinearRegression().fit(X, y).coef_[-k:]
    rng = np.random.RandomState(config.SEED)
    boots = np.array([LinearRegression().fit(X[i], y[i]).coef_[-k:] for i in (rng.randint(0, len(y), len(y)) for _ in range(n_boot))])
    rows = []
    for j, lv in enumerate(levels):
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        rows.append({"group": lv, "reference": reference, "n_group": int((g == lv).sum()), "n_reference": int((g == reference).sum()),
                     "pay_difference_pct": (np.exp(base[j]) - 1) * 100, "ci95_low_pct": (np.exp(lo) - 1) * 100, "ci95_high_pct": (np.exp(hi) - 1) * 100,
                     "interval_excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def mitigation_variants(train: pd.DataFrame, test: pd.DataFrame, frame_by_group: Dict[str, pd.Series], band_ref_test: pd.Series) -> pd.DataFrame:
    """Re-train / re-evaluate. Every variant is scored against the ORIGINAL (historical) test labels so the cost of each change stays visible.
      M0 original                : allowed features, historical labels
      M1 label correction        : estimate the career-gap pay difference on TRAIN only and add it back to the training labels of gap hires
                                   (raise only); features unchanged. Uses the audit attribute at TRAINING time only; never at prediction time.
      M2 minimal feature set     : drop work_location and highest_education (possible proxies)"""
    y_tr = train[config.TARGET].to_numpy(float)
    X_tr, X_te = train[config.ALLOWED], test[config.ALLOWED]
    grp_tr = career_gap_groups(train["career_gap_months"])
    levels = [g for g in sorted(grp_tr.dropna().unique()) if g != "No career gap"]
    extra = pd.DataFrame({f"is_{x}": (grp_tr == x).astype(float) for x in levels}, index=train.index)
    coefs = LinearRegression().fit(_design(train, extra), np.log(y_tr)).coef_[-len(levels):]
    factor = pd.Series({x: float(np.exp(-min(c, 0.0))) for x, c in zip(levels, coefs)})
    y_adj = y_tr * grp_tr.map(factor).fillna(1.0).astype(float).to_numpy()
    runs = {
        "M0 original (historical labels)": (build_model(BOOSTING).fit(X_tr, y_tr), config.ALLOWED, None),
        "M1 label correction for career-gap pay difference": (build_model(BOOSTING).fit(X_tr, y_adj), config.ALLOWED, factor),
        "M2 drop work_location and highest_education": (build_model(BOOSTING, config.NUMERIC, ["role_id", "department_id"]).fit(X_tr[config.NUMERIC + ["role_id", "department_id"]], y_tr),
                                                      config.NUMERIC + ["role_id", "department_id"], None),
    }
    rows: List[Dict] = []
    for name, (model, cols, fac) in runs.items():
        pred = model.predict(X_te[cols])
        m = metrics(test[config.TARGET], pred)
        cv_mae = float(np.mean(np.abs(cv_oof(BOOSTING, X_tr[cols], y_tr, numeric=[c for c in cols if c in config.NUMERIC], categorical=[c for c in cols if c in config.CATEGORICAL]) - y_tr))) if fac is None else float("nan")
        row = {"variant": name, "cv_mae_train_only": cv_mae, "mae": m["mae"], "r2": m["r2"], "label_correction_factors": "" if fac is None else "; ".join(f"{k}: x{v:.4f}" for k, v in fac.items())}
        for gname, labels in frame_by_group.items():
            fr = pd.DataFrame({"g": labels.to_numpy(), "actual": test[config.TARGET].to_numpy(float), "predicted": pred, "band_ref": band_ref_test.to_numpy(float)})
            err = group_gap_checked((fr["predicted"] - fr["actual"]).to_numpy(), fr["g"].to_numpy())
            prd = group_gap_checked((fr["predicted"] / fr["band_ref"]).to_numpy(), fr["g"].to_numpy())
            row[f"{gname}: signed-error gap (INR)"] = err["max_gap"]
            row[f"{gname}: predicted-vs-role-band gap (ratio)"] = prd["max_gap"]
        rows.append(row)
    return pd.DataFrame(rows)
