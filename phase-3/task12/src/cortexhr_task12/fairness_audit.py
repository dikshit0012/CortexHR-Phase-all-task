"""Orchestrates the Task 12 audit: baseline, protected-attribute analysis, bias sources, fair screener, comparison, quality, reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import paths
from .bias_detection import VARIANTS, ablation, proxy_scan, skill_extraction_consistency, text_differences, text_risk_scan
from .counterfactual_testing import primary_jobs, score_pairs
from .data_loading import load_pairs, load_protected
from .fairness_metrics import TOLERANCE, compare_summaries, gap_summary, group_gap_report, reduction_pct
from .ranking_comparison import quality_row
from .screeners import AS_SHIPPED, FAIR, SCREENERS, TEXT_ONLY, run_screener

paths.ensure_task11_importable()
from cortexhr_task11 import data_loading as dl11  # noqa: E402

BEFORE = (AS_SHIPPED, TEXT_ONLY)
FMT = dict(index=False, float_format="%.8f")


def _protected_analysis(prim: Dict[str, pd.DataFrame], pairs: pd.DataFrame, prot: pd.DataFrame):
    """Returns (aggregated public table, per-pair private table)."""
    g = prot.set_index("resume_id")
    meta = pairs[["pair_id", "resume_id_original", "resume_id_twin"]].copy()
    meta["gender_a"] = g.loc[meta["resume_id_original"], "inferred_gender"].to_numpy()
    meta["gender_b"] = g.loc[meta["resume_id_twin"], "inferred_gender"].to_numpy()
    meta["tier_a"] = g.loc[meta["resume_id_original"], "college_tier"].to_numpy()
    meta["tier_b"] = g.loc[meta["resume_id_twin"], "college_tier"].to_numpy()
    meta["gender_flipped"] = meta["gender_a"] != meta["gender_b"]
    meta["tier_changed"] = meta["tier_a"] != meta["tier_b"]
    meta["tier_direction"] = np.where(~meta["tier_changed"], "unchanged", np.where(meta["tier_b"] < meta["tier_a"], "twin higher tier", "twin lower tier"))
    meta["gender_direction"] = meta["gender_a"] + "->" + meta["gender_b"]
    agg_rows, private = [], meta.copy()
    for name, d in prim.items():
        d = d.merge(meta, on="pair_id")
        d["signed_gap_b_minus_a"] = d["score_b"] - d["score_a"]
        private[f"{name}__gap"] = d["twin_gap"].to_numpy()
        private[f"{name}__higher_scored"] = d["higher_scored"].to_numpy()
        for (attr, gd, td), s in d.groupby(["attribute_changed", "gender_direction", "tier_direction"]):
            agg_rows.append({"screener": name, "attribute_changed": attr, "gender_direction_orig_to_twin": gd, "tier_direction": td, "n_pairs": len(s),
                             "mean_signed_gap_twin_minus_original": s["signed_gap_b_minus_a"].mean(), "mean_abs_gap": s["twin_gap"].mean(),
                             "max_abs_gap": s["twin_gap"].max(), "pairs_exceeding_tolerance": int(s["exceeds_tolerance"].sum())})
    return pd.DataFrame(agg_rows), private


def job_stratified_group_analysis(
    ranked: pd.DataFrame,
    resumes: pd.DataFrame,
    labels: pd.DataFrame,
    protected: pd.DataFrame,
    min_group_size: int = 3,
):
    """Compare protected groups within each resume's target job; attributes are audit-only."""
    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")
    attrs = ["inferred_gender", "college_tier"]
    missing = set(["resume_id", *attrs]) - set(protected.columns)
    if missing:
        raise ValueError(f"protected attributes lack columns: {sorted(missing)}")
    target = resumes[["resume_id", "target_job_family"]].merge(
        protected[["resume_id", *attrs]], on="resume_id", how="left", validate="one_to_one"
    )
    if target[attrs].isna().any().any():
        raise ValueError("protected attributes are missing for one or more resumes")
    scored = ranked.merge(target, on="resume_id", how="inner", validate="many_to_one")
    scored = scored.loc[scored["job_id"].eq(scored["target_job_family"])].copy()
    if len(scored) != len(resumes) or scored["resume_id"].duplicated().any():
        raise ValueError("each resume must have exactly one score for its target job")
    scored = scored.merge(labels, on=["resume_id", "job_id"], how="left", validate="one_to_one")
    scored["score_centered_within_job"] = scored["final_score"] - scored.groupby("job_id")["final_score"].transform("mean")
    scored["skill_overlap_centered_within_job"] = scored["skill_overlap"] - scored.groupby("job_id")["skill_overlap"].transform("mean")

    rows = []
    summary = {}
    for attr in attrs:
        valid_job_gaps = []
        largest_gap = None
        for job_id, job in scored.groupby("job_id", sort=True):
            counts = job[attr].value_counts()
            eligible = counts[counts >= min_group_size].index
            eligible_scores = job.loc[job[attr].isin(eligible)]
            group_means = eligible_scores.groupby(attr, observed=True)["final_score"].mean()
            gap = float(group_means.max() - group_means.min()) if len(group_means) >= 2 else np.nan
            if np.isfinite(gap):
                valid_job_gaps.append(gap)
                if largest_gap is None or gap > largest_gap["max_group_gap"]:
                    largest_gap = {"job_id": str(job_id), "max_group_gap": gap}
            for group, members in job.groupby(attr, observed=True, sort=True):
                labeled = members.dropna(subset=["human_suitability_score"])
                rows.append({
                    "attribute": attr,
                    "job_id": job_id,
                    "group": group,
                    "n_candidates": len(members),
                    "n_labeled": len(labeled),
                    "mean_final_score": float(members["final_score"].mean()),
                    "mean_score_centered_within_job": float(members["score_centered_within_job"].mean()),
                    "mean_tfidf_similarity": float(members["tfidf_similarity"].mean()),
                    "mean_skill_overlap": float(members["skill_overlap"].mean()),
                    "mean_skill_overlap_centered_within_job": float(members["skill_overlap_centered_within_job"].mean()),
                    "mean_human_suitability_score": float(labeled["human_suitability_score"].mean()) if len(labeled) else np.nan,
                    "mean_required_skill_coverage": float(labeled["required_skill_coverage"].mean()) if len(labeled) else np.nan,
                    "human_shortlist_rate": float(labeled["shortlist"].eq("Y").mean()) if len(labeled) else np.nan,
                    "model_top5_rate": float(members["rank"].le(5).mean()),
                    "within_job_score_gap": gap,
                    "groups_in_gap": int(len(group_means)),
                    "min_group_size_for_gap": min_group_size,
                })
        centered_scores = scored.groupby(attr, observed=True)["score_centered_within_job"].mean().to_dict()
        centered_skills = scored.groupby(attr, observed=True)["skill_overlap_centered_within_job"].mean().to_dict()
        labeled_scored = scored.dropna(subset=["human_suitability_score"]).copy()
        labeled_scored["human_score_centered_within_job"] = (
            labeled_scored["human_suitability_score"]
            - labeled_scored.groupby("job_id")["human_suitability_score"].transform("mean")
        )
        centered_human = labeled_scored.groupby(attr, observed=True)["human_score_centered_within_job"].mean().to_dict()
        summary[attr] = {
            "min_group_size": min_group_size,
            "jobs_with_comparable_groups": len(valid_job_gaps),
            "jobs_without_comparable_groups": int(scored["job_id"].nunique() - len(valid_job_gaps)),
            "equal_job_weighted_mean_gap": float(np.mean(valid_job_gaps)) if valid_job_gaps else None,
            "maximum_within_job_gap": largest_gap,
            "mean_job_centered_score_by_group": {str(k): float(v) for k, v in centered_scores.items()},
            "mean_job_centered_skill_overlap_by_group": {str(k): float(v) for k, v in centered_skills.items()},
            "mean_job_centered_human_score_by_group": {str(k): float(v) for k, v in centered_human.items()},
        }
    return pd.DataFrame(rows), summary


def run(reports_dir: Optional[Path] = None, write: bool = True, tolerance: float = TOLERANCE, stress_step: int = 6) -> Dict[str, Any]:
    reports_dir = Path(reports_dir) if reports_dir else paths.REPORTS_DIR
    jobs = dl11.load_jobs(paths.TASK11_DATA / "job_descriptions.csv")
    resumes = dl11.load_resumes(paths.TASK11_DATA / "resumes.csv")
    labels = dl11.load_labels(paths.TASK11_DATA / "resume_job_labels.csv")
    pairs = load_pairs(paths.PAIRS_CSV, resumes, expected_count=25)
    protected = load_protected()                       # confidential, audit only; may be None
    job_by_pair = primary_jobs(pairs, resumes)

    # -- 1. score with every screener (labels are not involved) --
    ranked = {n: run_screener(n, resumes, jobs) for n in SCREENERS}
    prim = {n: score_pairs(ranked[n], pairs, job_by_pair, tolerance) for n in SCREENERS}
    allj = {n: score_pairs(ranked[n], pairs, None, tolerance) for n in SCREENERS}
    for n in SCREENERS:
        for d in (prim[n], allj[n]):
            d.insert(0, "screener", n)
    summ = {n: gap_summary(prim[n]["twin_gap"], tolerance) for n in SCREENERS}
    summ_all = {n: gap_summary(allj[n]["twin_gap"], tolerance) for n in SCREENERS}

    # -- 2. comparisons --
    comps = []
    for scope, S in (("primary_job_25_pairs", summ), ("all_10_jobs_250_comparisons", summ_all)):
        for b in BEFORE:
            t = compare_summaries(S[b], S[FAIR])
            t.insert(0, "after_screener", FAIR); t.insert(0, "before_screener", b); t.insert(0, "scope", scope)
            comps.append(t)
    comparison = pd.concat(comps, ignore_index=True)

    # -- 3. ranking quality (identical data & methodology; labels used only here) --
    q = {n: quality_row(ranked[n], labels) for n in SCREENERS}
    qrows = []
    for key, label in [("precision_at_5", "Precision@5"), ("recall_at_5", "Recall@5"), ("ndcg_at_5", "NDCG@5"), ("top5_hits", "Top-5 Hits")]:
        qrows.append({"metric": label, **{n: q[n][key] for n in SCREENERS},
                      "difference_fair_minus_as_shipped": q[FAIR][key] - q[AS_SHIPPED][key],
                      "difference_fair_minus_text_only": q[FAIR][key] - q[TEXT_ONLY][key]})
    quality = pd.DataFrame(qrows)
    from cortexhr_task11.evaluation import labeled_pool
    tops = {n: {j: frozenset(g.nsmallest(5, "pool_rank")["resume_id"]) for j, g in labeled_pool(ranked[n], labels).groupby("job_id")} for n in SCREENERS}
    top5_changed = {f"{n}_vs_{AS_SHIPPED}": int(sum(tops[n][j] != tops[AS_SHIPPED][j] for j in tops[n])) for n in SCREENERS if n != AS_SHIPPED}

    # -- 4. bias-source evidence --
    abl = ablation(resumes, jobs, pairs, job_by_pair)
    diffs = text_differences(resumes, pairs)
    consistency = {n: skill_extraction_consistency(ranked[n], pairs) for n in SCREENERS}
    risk = text_risk_scan(resumes)
    component = {n: {"mean_tfidf_gap": float(allj[n]["tfidf_gap"].mean()), "mean_skill_overlap_gap": float(allj[n]["skill_overlap_gap"].mean())} for n in SCREENERS}

    # -- 5. protected-attribute analysis (audit only) --
    prot_tbl, private_tbl, group_tbl, proxy = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    stratified_group_tbl, stratified_group_summary = pd.DataFrame(), {}
    if protected is not None:
        prot_tbl, private_tbl = _protected_analysis(prim, pairs, protected)
        stratified_group_tbl, stratified_group_summary = job_stratified_group_analysis(
            ranked[FAIR], resumes, labels, protected
        )
        g = protected.set_index("resume_id")
        tgt = resumes.set_index("resume_id")["target_job_family"]
        rows = []
        twin_ids = set(pairs["resume_id_original"]) | set(pairs["resume_id_twin"])
        for n in SCREENERS:
            sc = ranked[n].set_index(["job_id", "resume_id"])["final_score"]
            own = pd.Series({rid: sc.loc[(tgt.loc[rid], rid)] for rid in resumes["resume_id"]})
            for scope, ids in (("twin_resumes_only (50)", sorted(twin_ids)), ("all_resumes (185)", list(resumes["resume_id"]))):
                for attr in ("inferred_gender", "college_tier"):
                    rep = group_gap_report(own.loc[ids].to_numpy(), g.loc[ids, attr].to_numpy())
                    rows.append({"screener": n, "scope": scope, "attribute": attr, "group_means": json.dumps({k: round(v, 6) for k, v in rep["group_means"].items()}),
                                 "max_group_gap": rep["max_gap"]})
        group_tbl = pd.DataFrame(rows)
        proxy = proxy_scan(resumes, protected)

    gender_dir = {}
    if protected is not None:
        from scipy.stats import binomtest
        for n in SCREENERS:
            d = prim[n].merge(private_tbl[["pair_id", "gender_a"]], on="pair_id")
            male = np.where(d["gender_a"] == "Male", d["score_a"], d["score_b"])
            female = np.where(d["gender_a"] == "Male", d["score_b"], d["score_a"])
            diff = male - female
            hi, lo = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
            gender_dir[n] = {"mean_male_minus_female_score": float(diff.mean()), "pairs_male_coded_higher": hi, "pairs_female_coded_higher": lo,
                             "pairs_tied": int(len(diff) - hi - lo), "sign_test_p": float(binomtest(hi, hi + lo).pvalue) if hi + lo else None}
    from .stress_test import run_stress_test
    stress = run_stress_test(resumes, jobs, step=stress_step)
    summary = {
        "tolerance": tolerance, "n_pairs": len(pairs), "attributes_in_pairs": pairs["attribute_changed"].value_counts().to_dict(),
        "protected_attributes_available": None if protected is None else [c for c in protected.columns if c != "resume_id"],
        "primary_job_gap_summary": summ, "all_jobs_gap_summary": summ_all, "gap_reduction_pct_primary": {
            b: reduction_pct(summ[b]["average_gap"], summ[FAIR]["average_gap"]) for b in BEFORE},
        "gap_reduction_pct_all_jobs": {b: reduction_pct(summ_all[b]["average_gap"], summ_all[FAIR]["average_gap"]) for b in BEFORE},
        "ranking_quality": q, "top5_sets_changed_vs_as_shipped": top5_changed, "component_gaps_all_jobs": component,
        "job_stratified_group_analysis": stratified_group_summary,
        "extraction_consistency": consistency, "ablation": abl.to_dict("records"),
        "unintended_text_differences_total": int(diffs["other (unintended difference)"].sum()),
        "gender_direction": gender_dir,
    }
    result = {"ranked": ranked, "primary": prim, "all_jobs": allj, "summaries": summ, "summaries_all": summ_all, "comparison": comparison,
              "quality": quality, "ablation": abl, "diffs": diffs, "risk": risk, "protected_analysis": prot_tbl, "private_details": private_tbl,
              "job_stratified_group": stratified_group_tbl,
              "group_gap": group_tbl, "proxy": proxy, "stress": stress, "summary": summary, "pairs": pairs, "labels": labels, "resumes": resumes, "jobs": jobs}
    if write:
        from .reporting import write_reports
        write_reports(result, reports_dir)
    return result
