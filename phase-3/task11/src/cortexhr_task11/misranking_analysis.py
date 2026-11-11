"""Find and explain genuine misrankings (post-hoc analysis; human labels and some structured resume fields are used here for
DIAGNOSIS ONLY - none of them feed the score).

Definition (size-matched cutoff). For each job let R = number of human-shortlisted candidates in the labelled pool (score >= 4).
A perfect ranker would place exactly those R candidates first, so
  * missed_shortlist : shortlisted (score >= 4) candidate ranked OUTSIDE the top R
  * false_shortlist  : non-shortlisted (score <= 3) candidate ranked INSIDE the top R
These are counted in equal numbers by construction. Nothing is fabricated: the cases below are exactly the pairs that satisfy the
definition, and the number found is reported whatever it is. Severity (to order the cases):
  missed : (rank - R) / (N - R) * (score - 3) / 2          false : (R - rank + 1) / R * (4 - score) / 3
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd

from .job_processor import JobProfile
from .ranking_engine import RankingConfig, explain_tfidf, fit_tfidf
from .resume_processor import SCRUB_COLUMNS
from .preprocessing import scrub_identity
from .skill_extractor import SkillExtractor, split_skill_list

# Diagnostic-only phrases: a skill described indirectly in a bullet. NOT used for scoring (scoring is lexical).
INDIRECT_EVIDENCE: Dict[str, List[str]] = {
    "model deployment": ["deployed a model", "model serving"],
    "rest apis": ["rest services"],
    "data visualization": ["built dashboards"],
    "lead generation": ["qualified leads"],
    "seo": ["organic traffic"],
    "email campaigns": ["email campaigns reaching"],
    "bank reconciliation": ["bank reconciliation"],
    "unit testing": ["unit tests"],
    "ci/cd": ["deployment time"],
    "recruitment": ["closed positions", "positions in a year"],
    "employee onboarding": ["onboarded"],
}

CASE_COLUMNS = ["case_id", "job_id", "job_title", "resume_id", "category", "predicted_rank", "n_labeled_candidates",
                "size_matched_cutoff_R", "predicted_rank_among_all_resumes", "human_suitability_score", "human_label", "final_score",
                "tfidf_similarity", "skill_overlap", "matched_skills", "missing_skills", "severity", "layout_difficulty",
                "skills_heading_in_text", "structured_required_coverage_eval_only", "extraction_gap_skills", "credited_beyond_profile",
                "indirect_evidence_for_missing", "tfidf_vs_topR_median", "preferred_skills_matched_vs_topR_median", "top_tfidf_terms", "generic_share_of_tfidf",
                "n_ranked_above_with_lower_human_score", "reason_for_misranking", "possible_improvement"]


def find_misrankings(pool: pd.DataFrame) -> pd.DataFrame:
    """Return the pool rows that violate the size-matched cutoff, with category and severity."""
    p = pool.copy()
    p["R"] = p.groupby("job_id")["relevant"].transform("sum")
    p["N"] = p["n_pool"]
    missed = p[(p["relevant"] == 1) & (p["pool_rank"] > p["R"])].copy()
    false = p[(p["relevant"] == 0) & (p["pool_rank"] <= p["R"])].copy()
    missed["category"] = "missed_shortlist"
    missed["severity"] = (missed["pool_rank"] - missed["R"]) / (missed["N"] - missed["R"]).clip(lower=1) * (missed["human_suitability_score"] - 3) / 2
    false["category"] = "false_shortlist"
    false["severity"] = (false["R"] - false["pool_rank"] + 1) / false["R"].clip(lower=1) * (4 - false["human_suitability_score"]) / 3
    return pd.concat([missed, false]).sort_values("severity", ascending=False, kind="mergesort").reset_index(drop=True)


def _contains(text: str, phrases: List[str]) -> bool:
    t = " ".join(str(text).lower().split())
    return any(ph in t for ph in phrases)


def diagnose(cases: pd.DataFrame, pool: pd.DataFrame, resumes: pd.DataFrame, processed: pd.DataFrame, jobs: List[JobProfile],
             extractor: SkillExtractor, config: RankingConfig = RankingConfig()) -> pd.DataFrame:
    """Add measured diagnostics and a rule-composed explanation to each case. Every statement is derived from values in the row."""
    vec, _, _ = fit_tfidf(processed, jobs, config)
    job_by_id = {j.job_id: j for j in jobs}
    res_by_id = resumes.set_index("resume_id")
    proc_by_id = processed.set_index("resume_id")
    out: List[Dict] = []
    for i, c in enumerate(cases.to_dict("records"), start=1):
        job, rid = job_by_id[c["job_id"]], c["resume_id"]
        raw = res_by_id.loc[rid]
        text = scrub_identity(
            raw["resume_text"],
            [raw.get(col) for col in SCRUB_COLUMNS if col in raw.index],
        )
        keys_text = set(proc_by_id.loc[rid, "skill_keys"])
        structured = {extractor.canonical_key(s) for s in split_skill_list(raw.get("skills"))}
        req = [(d, k) for d, k in job.required]
        gap = [d for d, k in req if k in structured and k not in keys_text]               # in profile, not extracted from text
        credited = [(d, k) for d, k in req if k in keys_text and k not in structured]     # extracted from text, not in profile
        credited_txt = "; ".join(f"{d} <- '{extractor.evidence(text, k, 60)}'" for d, k in credited)
        missing = [(d, k) for d, k in req if k not in keys_text]
        indirect = [d for d, k in missing if k in INDIRECT_EVIDENCE and _contains(text, INDIRECT_EVIDENCE[k])]
        has_heading = bool(re.search(r"\bskills\b", text.lower()))
        g = pool[pool["job_id"] == c["job_id"]]
        R = int(c["R"])
        top_r_tfidf = g[g["pool_rank"] <= R]["tfidf_similarity"]
        med = float(top_r_tfidf.median()) if len(top_r_tfidf) else float("nan")
        count_pref = lambda v: 0 if (v is None or v != v or not str(v).strip()) else len(str(v).split("; "))
        pref_n = count_pref(c.get("preferred_skills_matched"))
        pref_med = float(g[g["pool_rank"] <= R]["preferred_skills_matched"].map(count_pref).median()) if R else float("nan")
        ex = explain_tfidf(vec, job, proc_by_id.loc[rid, "scoring_text"])
        above = g[g["pool_rank"] < c["pool_rank"]]
        n_lower_above = int((above["human_suitability_score"] < c["human_suitability_score"]).sum())
        n_higher_below = int((g[g["pool_rank"] > c["pool_rank"]]["human_suitability_score"] > c["human_suitability_score"]).sum())
        hcov = float(c["required_skill_coverage"])
        reasons, fixes = [], []
        if c["category"] == "missed_shortlist":
            if gap:
                reasons.append(f"Skill extraction gap: {', '.join(gap)} appear in the candidate's structured skill profile (structured-profile required coverage "
                               f"{hcov:.1f}) but not in the resume text that is scored (layout '{raw.get('layout_difficulty')}', "
                               f"{'SKILLS heading present' if has_heading else 'no SKILLS section in the text'}), so overlap was {c['skill_overlap']:.1f}.")
                fixes.append("Section-aware parsing; when the SKILLS section is missing fall back to the candidate's structured skills or route the resume to manual review instead of silently under-scoring it.")
            if indirect:
                reasons.append(f"{', '.join(indirect)} is evidenced only indirectly in the experience bullets, which lexical matching cannot see.")
                ex_skill = indirect[0]
                ex_phrase = INDIRECT_EVIDENCE.get(extractor.canonical_key(ex_skill), [""])[0]
                fixes.append(f"Phrase-level or semantic matching of experience bullets to skills (e.g. '{ex_phrase}' -> {ex_skill}).")
            if not gap and c["tfidf_similarity"] < med:
                reasons.append(f"Skill overlap ({c['skill_overlap']:.1f}) equals the human-reviewed coverage ({hcov:.1f}), so extraction is not the cause; "
                               f"its TF-IDF similarity ({c['tfidf_similarity']:.3f}) is below the median ({med:.3f}) of the candidates the system placed in the top {R}. "
                               f"Top matching terms: {', '.join(ex['top_terms'][:3])}; {ex['generic_share']:.0%} of the similarity comes from non-skill words. "
                               f"It matches {pref_n} preferred (nice-to-have) skill(s) versus a top-{R} median of {pref_med:.0f}"
                               f"{', which lowers its similarity because preferred skills are part of the job text' if pref_n < pref_med else ''}.")
                fixes.append("Reduce the weight of generic text (remove job-description boilerplate, weight skill terms) or add a semantic / job-title-family feature.")
            if n_lower_above:
                reasons.append(f"{n_lower_above} candidate(s) ranked above it received a lower human score.")
            if not gap and c["skill_overlap"] < 1.0 and abs(c["skill_overlap"] - hcov) < 1e-9:
                reasons.append("Reviewers shortlisted it with partial required coverage; factors outside the scored features (e.g. role / title fit) may explain this - a hypothesis, not verified.")
                fixes.append("Add an explainable role-title-family similarity feature (job-related, not a protected attribute) and re-audit.")
        else:  # false_shortlist
            if credited:
                reasons.append(f"Over-credit: skill overlap {c['skill_overlap']:.1f} exceeds the structured-profile coverage {hcov:.1f}; credited without being in the profile skill list: {credited_txt}.")
                fixes.append("Give lower weight to skills evidenced only by a certification line, a bullet or a synonym, and show them as 'indirect' to the recruiter.")
            if c["tfidf_similarity"] >= med:
                reasons.append(f"Its TF-IDF similarity ({c['tfidf_similarity']:.3f}) is at/above the top-{R} median ({med:.3f}); top terms: {', '.join(ex['top_terms'][:3])}.")
            if pref_n > pref_med:
                reasons.append(f"It matches {pref_n} preferred (nice-to-have) skill(s) versus a top-{R} median of {pref_med:.0f}; preferred skills are part of the job text, which raises its TF-IDF similarity although they are not required.")
            if n_higher_below:
                reasons.append(f"{n_higher_below} candidate(s) ranked below it received a higher human score.")
            if not credited:
                reasons.append(f"Overlap ({c['skill_overlap']:.1f}) matches human coverage ({hcov:.1f}); the reviewer rated it a partial match (score {c['human_suitability_score']}), so the disagreement is about degree of fit, not a clear error.")
                fixes.append("Calibrate or add features for degree of fit (e.g. nice-to-have skills, role family); verify with reviewers.")
        out.append({
            "case_id": i, "job_id": c["job_id"], "job_title": c["job_title"], "resume_id": rid, "category": c["category"],
            "predicted_rank": int(c["pool_rank"]), "n_labeled_candidates": int(c["n_pool"]), "size_matched_cutoff_R": R,
            "predicted_rank_among_all_resumes": int(c["rank_all_resumes"]), "human_suitability_score": int(c["human_suitability_score"]),
            "human_label": f"{c['shortlist']} / {c['reviewer_note']}", "final_score": round(float(c["final_score"]), 4),
            "tfidf_similarity": round(float(c["tfidf_similarity"]), 4), "skill_overlap": round(float(c["skill_overlap"]), 4),
            "matched_skills": c["matched_skills"], "missing_skills": c["missing_skills"], "severity": round(float(c["severity"]), 4),
            "layout_difficulty": raw.get("layout_difficulty"), "skills_heading_in_text": has_heading,
            "structured_required_coverage_eval_only": hcov, "extraction_gap_skills": "; ".join(gap), "credited_beyond_profile": credited_txt,
            "indirect_evidence_for_missing": "; ".join(indirect), "tfidf_vs_topR_median": f"{c['tfidf_similarity']:.3f} vs {med:.3f}",
            "preferred_skills_matched_vs_topR_median": f"{pref_n} vs {pref_med:.0f}",
            "top_tfidf_terms": "; ".join(ex["top_terms"]), "generic_share_of_tfidf": round(float(ex["generic_share"]), 3),
            "n_ranked_above_with_lower_human_score": n_lower_above,
            "reason_for_misranking": " ".join(reasons) or "No single measured cause; see TF-IDF and overlap columns.",
            "possible_improvement": " ".join(dict.fromkeys(fixes)) or "Collect more reviewer rationale to identify the missing signal.",
        })
    return pd.DataFrame(out, columns=CASE_COLUMNS)
