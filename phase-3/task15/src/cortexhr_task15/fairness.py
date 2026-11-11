"""Group-level evaluation with Task 12's official `group_gap()` helper.

group_gap(scores, groups) -> (dict of mean score per group, largest gap between group means). It is a warning signal, not proof.
Important distinctions reported separately:
  actual pay gap        = difference in HISTORICAL salaries between groups (may contain past inequity; not caused by the model)
  prediction gap        = difference in the MODEL's predicted salaries
  error gap             = difference in signed error (predicted - actual) and in absolute error (MAE)
  band gap              = difference in the recommended Low / Mid / High
'Relative to role band' compares pay with each role's 2026 band mid x market index (an audit yardstick only, never a model input)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config
from .cortexhr_helpers import group_gap as official_group_gap
from .models import make_preprocessor

LABELS = {0: "No career gap", 1: "Career gap 3-9 months", 2: "Career gap 12+ months"}


def career_gap_groups(months: pd.Series) -> pd.Series:
    """Fixed, documented buckets of the months of career break before joining."""
    m = pd.to_numeric(months, errors="coerce")
    out = pd.Series(pd.NA, index=months.index, dtype="object")
    out[m == 0] = LABELS[0]
    out[(m >= 1) & (m <= 11)] = LABELS[1]
    out[m >= 12] = LABELS[2]
    return out


def group_gap_checked(values, groups) -> Dict:
    """Official helper, with missing group labels / scores removed first (and counted), so the helper never silently mixes NaN groups."""
    s, g = pd.Series(np.asarray(values, float)), pd.Series(np.asarray(groups, object))
    blank = g.map(lambda value: isinstance(value, str) and not value.strip())
    g = g.mask(blank)
    ok = s.notna() & g.notna()
    if ok.sum() == 0 or g[ok].nunique() < 2:
        return {"group_means": {}, "max_gap": float("nan"), "dropped_missing": int((~ok).sum()), "note": "fewer than two labelled groups"}
    means, gap = official_group_gap(s[ok].to_numpy(), g[ok].to_numpy())
    return {"group_means": {k: float(v) for k, v in means.items()}, "max_gap": float(gap), "dropped_missing": int((~ok).sum())}


def group_table(frame: pd.DataFrame, group_col: str, model: str) -> pd.DataFrame:
    """One row per group. `frame` needs: actual, predicted, rec_low, rec_mid, rec_high, band_ref (2026 band mid x market index)."""
    f = frame.dropna(subset=[group_col]).copy()
    f["err"] = f["predicted"] - f["actual"]
    f["rel_actual"], f["rel_pred"] = f["actual"] / f["band_ref"], f["predicted"] / f["band_ref"]
    rows = []
    for g, d in f.groupby(group_col):
        rows.append({"model": model, "grouping": group_col, "group": g, "n": len(d), "reliable_n": len(d) >= config.MIN_GROUP,
                     "mean_actual_salary": d["actual"].mean(), "mean_predicted_salary": d["predicted"].mean(), "mean_signed_error": d["err"].mean(),
                     "mean_signed_error_pct_of_actual": (d["err"] / d["actual"]).mean() * 100, "mae": d["err"].abs().mean(),
                     "mean_actual_vs_role_band": d["rel_actual"].mean(), "mean_predicted_vs_role_band": d["rel_pred"].mean(),
                     "mean_rec_low": d["rec_low"].mean() if "rec_low" in d else np.nan, "mean_rec_mid": d["rec_mid"].mean() if "rec_mid" in d else np.nan,
                     "mean_rec_high": d["rec_high"].mean() if "rec_high" in d else np.nan})
    return pd.DataFrame(rows)


def gap_summary(frame: pd.DataFrame, group_col: str, model: str) -> pd.DataFrame:
    """The official group_gap() applied to each quantity. Scores per group are means; `max_gap` is the largest difference between group means."""
    f = frame.dropna(subset=[group_col]).copy()
    q = {"actual salary (historical pay)": f["actual"], "predicted salary (model)": f["predicted"], "signed error (predicted - actual)": f["predicted"] - f["actual"],
         "absolute error": (f["predicted"] - f["actual"]).abs(), "actual pay vs role band (ratio)": f["actual"] / f["band_ref"],
         "predicted pay vs role band (ratio)": f["predicted"] / f["band_ref"]}
    if "rec_mid" in f:
        q["recommended mid"] = f["rec_mid"]; q["recommended low"] = f["rec_low"]; q["recommended high"] = f["rec_high"]
    rows = []
    for name, v in q.items():
        r = group_gap_checked(v.to_numpy(), f[group_col].to_numpy())
        rows.append({"model": model, "grouping": group_col, "quantity": name, "max_gap_between_group_means": r["max_gap"],
                     "group_means": "; ".join(f"{k}={x:,.3f}" if abs(x) < 100 else f"{k}={x:,.0f}" for k, x in r["group_means"].items()), "dropped_missing": r["dropped_missing"]})
    return pd.DataFrame(rows)


def proxy_check(X: pd.DataFrame, binary_group: pd.Series) -> Dict:
    """Can the ALLOWED features predict the protected group? AUC ~0.5 = no proxy channel detectable; clearly above 0.5 = proxies exist."""
    ok = binary_group.notna()
    y = binary_group[ok].astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 10:
        return {"auc": float("nan"), "note": "too few members of a group"}
    pipe_prep = make_preprocessor(config.NUMERIC, config.CATEGORICAL)
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("prep", pipe_prep), ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=config.SEED))])
    p = cross_val_predict(pipe, X[ok], y, cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED), method="predict_proba")[:, 1]
    return {"auc": float(roc_auc_score(y, p)), "n": int(ok.sum()), "positives": int(y.sum())}


# ------------------------------------------------------------------ the confidential unlock_CX-15 table (gender, age)
class UnlockMissing(FileNotFoundError):
    pass


def find_unlock(path: Optional[Path] = None) -> Optional[Path]:
    if path:
        return Path(path) if Path(path).exists() else None
    for n in config.UNLOCK_NAMES:
        p = config.PRIVATE_DIR / n
        if p.exists():
            return p
    return None


def load_unlock(path: Optional[Path] = None, required: bool = False) -> Optional[pd.DataFrame]:
    """`protected_attributes` = person_ref, record_type, gender, age_years, age_band. Returns None (never invents data) if the file is absent."""
    p = find_unlock(path)
    if p is None:
        if required:
            raise UnlockMissing(f"unlock_CX-15 (protected_attributes) not found in {config.PRIVATE_DIR}; expected one of {config.UNLOCK_NAMES}")
        return None
    d = pd.read_csv(p)
    miss = [c for c in ("person_ref", "gender") if c not in d.columns]
    if miss:
        raise ValueError(f"unlock_CX-15 lacks columns {miss}; found {list(d.columns)}")
    if d["person_ref"].isna().any():
        raise ValueError("unlock_CX-15 contains missing person_ref values")
    if d["person_ref"].duplicated().any():
        raise ValueError("unlock_CX-15 contains duplicate person_ref values")
    return d


def attach_gender(df: pd.DataFrame, unlock: pd.DataFrame) -> pd.DataFrame:
    """Adds `gender` (audit only) by hire_id == person_ref. Unmatched hires keep NaN and are reported, never guessed."""
    g = unlock.set_index("person_ref")["gender"]
    out = df.copy()
    out["gender"] = out[config.ID_COL].map(g)
    return out
