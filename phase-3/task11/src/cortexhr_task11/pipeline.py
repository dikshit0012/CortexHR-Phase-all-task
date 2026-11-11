"""End-to-end Task 11 pipeline. Order matters for leakage control: rank first from resumes + jobs only, load labels afterwards."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import data_loading as dl
from . import paths
from .evaluation import K_DEFAULT, ablation, evaluate_micro, evaluate_pool, labeled_pool, tie_sensitivity
from .extraction_check import summarize_extraction, validate_extraction
from .fairness_audit import audit_groups, experience_dependence
from .job_processor import process_jobs
from .misranking_analysis import diagnose, find_misrankings
from .ranking_engine import RankingConfig, count_ties, rank_resumes
from .resume_processor import process_resumes
from .skill_extractor import build_default_extractor

HUMAN_COLUMNS = {"human_suitability_score": "eval_only_human_suitability_score", "shortlist": "eval_only_shortlist"}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is pd.NA:
        return None
    raise TypeError(type(o))


def run(data_dir: Optional[Path] = None, reports_dir: Optional[Path] = None, config: RankingConfig = RankingConfig(), k: int = K_DEFAULT,
        write: bool = True) -> Dict[str, Any]:
    data_dir = Path(data_dir) if data_dir else paths.DATA_DIR
    reports_dir = Path(reports_dir) if reports_dir else paths.REPORTS_DIR
    jobs_df = dl.load_jobs(data_dir / "job_descriptions.csv")
    resumes_df = dl.load_resumes(data_dir / "resumes.csv")

    # ---- 1. scoring: resumes + jobs ONLY (labels are not loaded yet) ----
    extractor = build_default_extractor(jobs_df)
    processed = process_resumes(resumes_df, extractor)
    profiles = process_jobs(jobs_df, extractor)
    ranked = rank_resumes(processed, profiles, config)

    # ---- 2. evaluation: labels enter here, after every score is final ----
    labels = dl.load_labels(data_dir / "resume_job_labels.csv")
    integrity = dl.check_referential_integrity(jobs_df, resumes_df, labels)
    inspection = dl.inspect_datasets(jobs_df, resumes_df, labels)
    pool = labeled_pool(ranked, labels)
    quality = evaluate_pool(pool, k)
    abl = ablation(ranked, labels, k)
    ties_sens = tie_sensitivity(ranked, labels, k)
    cases = find_misrankings(pool)
    misranks = diagnose(cases, pool, resumes_df, processed, profiles, extractor, config)
    audit = audit_groups(pool, resumes_df, processed)
    exp_dep = experience_dependence(pool)
    validation_extractor = build_default_extractor(jobs_df, resumes_df)
    validation_processed = process_resumes(resumes_df, validation_extractor)
    extraction = validate_extraction(resumes_df, validation_processed, validation_extractor)
    extraction_summary = summarize_extraction(extraction)

    ranked_out = ranked.merge(labels[["job_id", "resume_id", *HUMAN_COLUMNS]].rename(columns=HUMAN_COLUMNS), on=["job_id", "resume_id"], how="left")
    top5 = ranked[ranked["rank"] <= k]

    pool_top = pool[pool["pool_rank"] <= k]
    eff = ranked.groupby("job_id").apply(lambda g: pd.Series({"std_tfidf_term": (config.tfidf_weight * g["tfidf_similarity"]).std(),
                                                              "std_skill_term": (config.skill_weight * g["skill_overlap"]).std()}), include_groups=False)
    summary = {
        "config": {**asdict(config), "ngram_range": list(config.ngram_range), "k": k},
        "counts": {"jobs": len(jobs_df), "resumes": len(resumes_df), "scored_pairs": len(ranked), "possible_pairs": len(jobs_df) * len(resumes_df),
                   "labelled_pairs_evaluated": len(pool), "duplicate_pairs": int(ranked.duplicated(["job_id", "resume_id"]).sum()),
                   "resumes_with_empty_text": int(processed["empty_text"].sum()),
                   "resumes_without_experience_statement": int((processed["experience_source"] == "not_found").sum())},
        "integrity": integrity,
        "ties": {"rows_with_exactly_tied_final_score_all_pairs": count_ties(ranked),
                 "rows_tied_inside_labelled_top_k": int(pool_top.duplicated(["job_id", "final_score"], keep=False).sum())},
        "effective_weight": {"mean_std_of_0.6xtfidf_term": float(eff["std_tfidf_term"].mean()), "mean_std_of_0.4xoverlap_term": float(eff["std_skill_term"].mean()),
                             "mean_tfidf_similarity": float(ranked["tfidf_similarity"].mean()), "max_tfidf_similarity": float(ranked["tfidf_similarity"].max())},
        "experience_dependence": exp_dep,
        "misrankings": {"total": len(misranks), "by_category": misranks["category"].value_counts().to_dict() if len(misranks) else {},
                        "false_shortlist_with_human_score_le_2": int(((misranks["category"] == "false_shortlist") & (misranks["human_suitability_score"] <= 2)).sum()) if len(misranks) else 0},
        "ranking_quality_macro": quality.iloc[-1].to_dict(),
        "ranking_quality_micro": evaluate_micro(pool, k),
        "extraction_overall": extraction_summary[extraction_summary["layout_difficulty"] == "ALL"].iloc[0].to_dict(),
        "inspection": inspection,
    }
    result = {"ranked": ranked, "ranked_out": ranked_out, "top5": top5, "quality": quality, "ablation": abl, "tie_sensitivity": ties_sens, "misrankings": misranks, "fairness_audit": audit,
              "extraction": extraction, "extraction_summary": extraction_summary, "summary": summary, "pool": pool, "processed": processed,
              "profiles": profiles, "extractor": extractor, "resumes": resumes_df, "jobs": jobs_df, "labels": labels}
    if write:
        reports_dir.mkdir(parents=True, exist_ok=True)
        fmt = dict(index=False, float_format="%.6f")
        ranked_out.to_csv(reports_dir / "ranked_candidates.csv", **fmt)
        top5.to_csv(reports_dir / "top5_per_job.csv", **fmt)
        quality.to_csv(reports_dir / "ranking_quality.csv", **fmt)
        abl.to_csv(reports_dir / "ranking_quality_ablation.csv", **fmt)
        ties_sens.to_csv(reports_dir / "tie_sensitivity.csv", **fmt)
        misranks.to_csv(reports_dir / "misranking_analysis.csv", **fmt)
        audit.to_csv(reports_dir / "fairness_audit.csv", **fmt)
        extraction_summary.to_csv(reports_dir / "extraction_validation.csv", **fmt)
        (reports_dir / "dataset_inspection.json").write_text(json.dumps(inspection, indent=2, default=_json_default), encoding="utf-8")
        (reports_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return result
