"""Descriptive disparate-outcome audit. Protected / proxy columns are used HERE ONLY to look at outcomes; they are never scoring
inputs. The dataset has no gender, age or date-of-birth column, so those cannot be audited; college_tier, layout / file format are
audited as available proxies. Groups are small - treat as a screening signal, not evidence of (un)fairness."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def audit_groups(pool: pd.DataFrame, resumes: pd.DataFrame, processed: pd.DataFrame, attrs=("college_tier", "layout_difficulty", "source_format")) -> pd.DataFrame:
    p = pool.merge(resumes[["resume_id", *attrs]], on="resume_id", how="left")
    p["R"] = p.groupby("job_id")["relevant"].transform("sum")
    p["in_top_R"] = (p["pool_rank"] <= p["R"]).astype(int)
    rows = []
    for a in attrs:
        for grp, g in p.groupby(a, dropna=False):
            rel = g[g["relevant"] == 1]
            rows.append({"attribute": a, "group": grp, "labelled_pairs": len(g), "human_shortlisted": len(rel),
                         "mean_final_score": g["final_score"].mean(), "mean_human_score": g["human_suitability_score"].mean(),
                         "selection_rate_top_R": g["in_top_R"].mean(),
                         "human_shortlisted_captured_in_top_R": rel["in_top_R"].mean() if len(rel) else np.nan})
    return pd.DataFrame(rows)


def experience_dependence(pool: pd.DataFrame) -> dict:
    """Spearman correlation between final score and extracted experience (should be weak: experience is not in the formula)."""
    g = pool.dropna(subset=["experience_years"])
    rho, pval = spearmanr(g["final_score"], g["experience_years"])
    return {"spearman_rho_score_vs_experience": float(rho), "p_value": float(pval), "n": int(len(g))}
