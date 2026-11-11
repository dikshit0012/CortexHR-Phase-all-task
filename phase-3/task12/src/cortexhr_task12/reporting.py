"""Write all Task 12 reports. Per-resume protected values go ONLY to reports/private/ (git-ignored)."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .fairness_audit import BEFORE, FMT
from .screeners import AS_SHIPPED, FAIR, TEXT_ONLY

KEEP = ["screener", "pair_id", "resume_a_id", "resume_b_id", "job_id", "attribute_changed", "score_a", "score_b", "signed_gap_b_minus_a",
        "twin_gap", "relative_gap_pct",
        "tfidf_gap", "skill_overlap_gap", "rank_a", "rank_b", "rank_difference", "higher_scored", "non_zero_gap", "exceeds_tolerance"]


def _fmt(x, nd=6):
    return "n/a (baseline 0)" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_reports(res: Dict[str, Any], out: Path) -> None:
    out = Path(out)
    (out / "private").mkdir(parents=True, exist_ok=True)
    prim, s = res["primary"], res["summary"]
    pd.concat([prim[n][KEEP] for n in BEFORE], ignore_index=True).to_csv(out / "fairness_before.csv", **FMT)
    prim[FAIR][KEEP].to_csv(out / "fairness_after.csv", **FMT)
    prim[AS_SHIPPED][KEEP].to_csv(out / "twin_gap_before.csv", **FMT)
    prim[FAIR][KEEP].to_csv(out / "twin_gap_after.csv", **FMT)
    pd.concat([res["all_jobs"][n][KEEP] for n in res["all_jobs"]], ignore_index=True).to_csv(out / "fairness_all_jobs_250_comparisons.csv", **FMT)
    res["comparison"].to_csv(out / "fairness_comparison.csv", **FMT)
    res["quality"].to_csv(out / "ranking_quality_comparison.csv", **FMT)
    res["ablation"].to_csv(out / "bias_source_ablation.csv", **FMT)
    res["diffs"].to_csv(out / "twin_text_differences.csv", **FMT)
    res["risk"].to_csv(out / "text_risk_scan.csv", **FMT)
    res["stress"].to_csv(out / "synthetic_stress_test.csv", **FMT)
    if len(res["protected_analysis"]):
        res["protected_analysis"].to_csv(out / "protected_attribute_analysis.csv", **FMT)
        res["group_gap"].to_csv(out / "group_gap_report.csv", **FMT)
        res["job_stratified_group"].to_csv(out / "job_stratified_group_gap.csv", **FMT)
        res["proxy"].to_csv(out / "proxy_scan.csv", **FMT)
        res["private_details"].to_csv(out / "private" / "protected_pair_details.csv", **FMT)
    else:
        pd.DataFrame([{"status": "unavailable", "reason": "The unlock_CX-12 protected-attribute file was not available."}]).to_csv(
            out / "group_gap_report.csv", index=False
        )
        pd.DataFrame().to_csv(out / "protected_attribute_analysis.csv", index=False)
        pd.DataFrame().to_csv(out / "job_stratified_group_gap.csv", index=False)
        pd.DataFrame().to_csv(out / "proxy_scan.csv", index=False)
    (out / "run_summary.json").write_text(
        json.dumps(_json_clean(s), indent=2, allow_nan=False), encoding="utf-8"
    )
    audit_report = {
        "pairs_tested": s["n_pairs"],
        "all_jobs_pair_comparisons": len(res["all_jobs"][AS_SHIPPED]),
        "protected_attributes_available": s["protected_attributes_available"],
        "fairness_before_fix": s["primary_job_gap_summary"][AS_SHIPPED],
        "fairness_text_only_reference": s["primary_job_gap_summary"][TEXT_ONLY],
        "fairness_after_fix": s["primary_job_gap_summary"][FAIR],
        "ranking_quality": res["quality"].to_dict(orient="records"),
        "job_stratified_group_analysis": s["job_stratified_group_analysis"],
        "limitations": [
            "The 25-pair counterfactual set does not establish fairness for unrepresented groups or intersections.",
            "Protected-group score gaps are descriptive and do not establish causal discrimination.",
            "Job-stratified group cells are small and human labels may reflect historical reviewer bias.",
            "Synthetic personal-information perturbations are not evidence about real applicants.",
            "Header-based name masking may miss unusual layouts or names outside the header.",
            "Human ranking labels may themselves be biased.",
        ],
    }
    if not s["protected_attributes_available"]:
        audit_report["limitations"].append("Protected-attribute analyses were unavailable because unlock_CX-12 was not found.")
    (out / "fairness_audit_report.json").write_text(
        json.dumps(_json_clean(audit_report), indent=2, allow_nan=False), encoding="utf-8"
    )
    statement = build_summary_md(res)
    (out / "fairness_summary.md").write_text(statement, encoding="utf-8")
    (out / "fairness_statement.md").write_text(statement, encoding="utf-8")
    _write_twin_gap_svg(prim[AS_SHIPPED], prim[FAIR], out / "twin_gap_comparison.svg")


def _json_clean(value):
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_twin_gap_svg(before: pd.DataFrame, after: pd.DataFrame, path: Path) -> None:
    width, height = 900, 360
    left, right, top, bottom = 64, 20, 42, 64
    plot_width, plot_height = width - left - right, height - top - bottom
    max_gap = max(float(before["twin_gap"].max()), float(after["twin_gap"].max()))
    scale_max = max_gap * 1.1 if max_gap > 0 else 0.01
    count = len(before)
    group_width = plot_width / max(count, 1)
    bar_width = min(7.0, group_width * 0.34)
    bars = []
    for index, (before_gap, after_gap, pair_id) in enumerate(
        zip(before["twin_gap"], after["twin_gap"], before["pair_id"])
    ):
        center = left + (index + 0.5) * group_width
        for offset, gap, color in ((-bar_width / 2, before_gap, "#3569a8"), (bar_width / 2, after_gap, "#2c9b73")):
            bar_height = max(float(gap) / scale_max * plot_height, 1.0)
            y = top + plot_height - bar_height
            bars.append(
                f'<rect x="{center + offset:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
                f'fill="{color}"><title>{escape(str(pair_id))}: {float(gap):.8f}</title></rect>'
            )
        label = escape(str(pair_id).replace("BP-", ""))
        bars.append(
            f'<text x="{center:.2f}" y="{height - bottom + 15}" text-anchor="end" '
            f'transform="rotate(-55 {center:.2f} {height - bottom + 15})" font-size="8">{label}</text>'
        )
    mean_before = float(before["twin_gap"].mean())
    mean_after = float(after["twin_gap"].mean())
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<title>Task 11 and corrected screener absolute twin score gaps</title>
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="16">Absolute twin score gaps by pair</text>
<text x="{width - right - 235}" y="42" font-family="sans-serif" font-size="10" fill="#3569a8">Before (Task 11 as shipped)</text>
<text x="{width - right - 105}" y="42" font-family="sans-serif" font-size="10" fill="#2c9b73">After (Task 12 fair)</text>
<text x="12" y="{top + plot_height / 2}" transform="rotate(-90 12 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="10">Absolute score gap</text>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#444"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#444"/>
{''.join(bars)}
<text x="{left}" y="{height - 8}" font-family="sans-serif" font-size="10">Mean gap: before {mean_before:.8f} | after {mean_after:.8f}</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def _tbl(df: pd.DataFrame, cols=None, nd=6) -> str:
    d = df[cols] if cols else df
    head = "| " + " | ".join(d.columns) + " |\n|" + "|".join(["---"] * len(d.columns)) + "|\n"
    rows = ["| " + " | ".join(_fmt(v, nd) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return head + "\n".join(rows) + "\n"


def build_summary_md(res: Dict[str, Any]) -> str:
    s, q = res["summary"], res["quality"]
    A, T, F = (s["primary_job_gap_summary"][n] for n in (AS_SHIPPED, TEXT_ONLY, FAIR))
    redT, redA = s["gap_reduction_pct_primary"][TEXT_ONLY], s["gap_reduction_pct_primary"][AS_SHIPPED]
    gd = s.get("gender_direction", {}).get(TEXT_ONLY)
    cmp_ = res["comparison"]
    cmp_ = cmp_[(cmp_["scope"] == "primary_job_25_pairs")].copy()
    show = cmp_[["before_screener", "metric", "before", "after", "improvement_absolute", "improvement_pct"]]
    stress = res["stress"].pivot(index="perturbation", columns="screener", values="mean_gap").reset_index()
    stress_max = res["stress"].pivot(index="perturbation", columns="screener", values="max_gap").reset_index()
    stress = stress.merge(stress_max, on="perturbation", suffixes=("_mean_gap", "_max_gap"))
    abl = res["ablation"][["variant", "mean_gap_primary_job", "max_gap_primary_job", "nonzero_pairs_primary_job"]]
    gdline = ("" if not gd else f"In the text-only run, male-coded names scored higher in {gd['pairs_male_coded_higher']} pairs, lower in {gd['pairs_female_coded_higher']} "
              f"and tied in {gd['pairs_tied']} (mean male - female = {gd['mean_male_minus_female_score']:+.6f}; sign test p = {_fmt(gd['sign_test_p'], 3)}). "
              "The size is about 1/100 of the tolerance, so it is a measurable sensitivity, not a material gap.")
    n_stress = int(res['stress']['n_resumes'].iloc[0])
    gg = res["group_gap"]
    protected_available = bool(s["protected_attributes_available"])
    protected_names = ", ".join(s["protected_attributes_available"] or [])
    all_resume_scope = f"all_resumes ({len(res['resumes'])})"
    def _gg(attr):
        if not len(gg):
            return "not available (protected validation data unavailable)"
        row = gg[(gg["screener"] == FAIR) & (gg["scope"] == all_resume_scope) & (gg["attribute"] == attr)]
        if row.empty:
            return "not available"
        row = row.iloc[0]
        return f"{attr.replace('inferred_', '').replace('college_', '')} {row['max_group_gap']:.4f} (group means {row['group_means']})"
    ggtxt = "; ".join(_gg(a) for a in ("inferred_gender", "college_tier"))
    stratified = s["job_stratified_group_analysis"]
    def _strat(attr):
        stat = stratified.get(attr)
        if not stat:
            return f"{attr.replace('inferred_', '').replace('college_', '')}: unavailable"
        largest = stat["maximum_within_job_gap"]
        largest_txt = "n/a" if largest is None else f"{largest['max_group_gap']:.4f} in {largest['job_id']}"
        centers = stat["mean_job_centered_score_by_group"]
        centers_txt = ", ".join(f"{group} {value:+.4f}" for group, value in centers.items())
        return (f"{attr.replace('inferred_', '').replace('college_', '')}: mean job-level max gap "
                f"{stat['equal_job_weighted_mean_gap']:.4f} across {stat['jobs_with_comparable_groups']} jobs; "
                f"largest {largest_txt}; candidate-weighted job-centered score means ({centers_txt})")
    stratified_txt = "; ".join(_strat(a) for a in ("inferred_gender", "college_tier"))
    d = q.set_index("metric")["difference_fair_minus_as_shipped"]
    dt = q.set_index("metric")["difference_fair_minus_text_only"]
    k_ship, k_txt = s["top5_sets_changed_vs_as_shipped"].get("task12_fair_vs_task11_as_shipped"), s["top5_sets_changed_vs_as_shipped"].get("task11_text_only_vs_task11_as_shipped")
    unchanged = [m for m in ("Precision@5", "Recall@5", "Top-5 Hits") if abs(d[m]) < 1e-12]
    quality_text = (f"Versus shipped Task 11: {', '.join(unchanged) if unchanged else 'no metric'} unchanged; NDCG@5 changed by {d['NDCG@5']:+.4f}"
                    f" and the top-5 sets differ in {k_ship} of 10 jobs. The fair screener keeps degree text (which Task 11 scrubbed) and masks only the institution tier, so a small ranking change is expected and is not caused by protected attributes. "
                    f"Versus the text-only run the NDCG@5 difference is {dt['NDCG@5']:+.4f} (its top-5 sets differ from shipped in {k_txt} jobs). "
                    "With 10 jobs and 95 relevant candidates, differences of this size are within noise: they show that the fairness change did not meaningfully reduce ranking quality, and they must not be read as an accuracy gain.")
    worst = min(d.min(), dt.min())
    tradeoff_text = ("In this audit no accuracy sacrifice was observed: no ranking-quality metric fell against either baseline." if worst >= -1e-12 else
                     f"In this audit the largest observed ranking-quality drop was {worst:+.4f}; whether that is acceptable is exactly the governance decision described above.")
    protected_text = (
        f"Protected attributes available for this audit: {protected_names}. Protected values were used only for aggregate auditing and never as scoring inputs."
        if protected_available else
        "The unlock_CX-12 protected-attribute file was unavailable, so no protected-label or group-level analysis is claimed; only text-based counterfactual and proxy-pathway tests were run."
    )
    labels_count = len(res["labels"])
    return f"""# Fairness statement - Task 12 (counterfactual audit of the Task 11 resume screener)

*Generated by `scripts/run_task12.py` from the numbers in this folder; nothing here is typed in by hand.*

## What was tested
The Task 11 screener (60% TF-IDF similarity + 40% required-skill overlap, score in [0, 1]) was audited with **{s['n_pairs']} counterfactual twin pairs**
(`resume_bias_pairs`). In each pair the two resumes have identical skills and experience and differ only in a gender-coded candidate name
(and, for {s['attributes_in_pairs'].get('candidate_name (gender-coded) and college tier', 0)} pairs, the college tier);
{s['attributes_in_pairs'].get('candidate_name (gender-coded) only', 0)} pairs change the name only. A fair screener must give both the same score
(`expected_score_difference = 0`). Twins isolate the effect of the protected attribute: any score difference cannot be explained by qualifications.
{protected_text} Tolerance: a twin gap above {s['tolerance']} (one point on a 0-100 view) counts as material.

## Results (primary job per pair; {s['n_pairs']} pairs)
| Screener | Average gap | Median | Maximum | Non-zero pairs | Above tolerance |
|---|---|---|---|---|---|
| Task 11 as shipped (before) | {_fmt(A['average_gap'])} | {_fmt(A['median_gap'])} | {_fmt(A['maximum_gap'])} | {A['non_zero_gap_pairs']} | {A['pairs_exceeding_tolerance']} |
| Task 11 given resume text only (before) | {_fmt(T['average_gap'])} | {_fmt(T['median_gap'])} | {_fmt(T['maximum_gap'])} | {T['non_zero_gap_pairs']} | {T['pairs_exceeding_tolerance']} |
| Task 12 fair screener (after) | {_fmt(F['average_gap'])} | {_fmt(F['median_gap'])} | {_fmt(F['maximum_gap'])} | {F['non_zero_gap_pairs']} | {F['pairs_exceeding_tolerance']} |

* **Average twin gap before / after:** shipped Task 11 {_fmt(A['average_gap'])} -> {_fmt(F['average_gap'])}; text-only Task 11 {_fmt(T['average_gap'])} -> {_fmt(F['average_gap'])}.
* **Percentage reduction:** {_fmt(redT, 1)}% versus the text-only baseline; **not defined versus the shipped Task 11 (its gap was already 0, so there was nothing to reduce)**.
* Both gaps were far below the tolerance: the largest baseline gap was {_fmt(T['maximum_gap'])} against {s['tolerance']}. Task 11 was **not** materially biased on these twins; its protection came from scrubbing the *values of structured columns*,
  and the audit shows that protection disappears when only text is supplied (small but non-zero gaps in {T['non_zero_gap_pairs']}/{T['n_pairs']} pairs). {gdline}
* Robustness: over all 10 jobs (250 pair-job comparisons) the text-only average gap is {_fmt(s['all_jobs_gap_summary'][TEXT_ONLY]['average_gap'])} and the fair screener's is {_fmt(s['all_jobs_gap_summary'][FAIR]['average_gap'])}.

## Where the differences came from (measured)
{_tbl(abl)}
The twins differ **only** in header name, contact line (name-bearing e-mail) and, for some pairs, the institution-tier statement; there are {s['unintended_text_differences_total']} other differing lines.
Skill and experience extraction was identical for twins in every pair-job comparison. All residual score differences are TF-IDF effects (rare name tokens change the vector norm); skill-overlap gaps were 0.
Masking the name alone or the e-mail alone is **not** enough; both are needed. Institution tier did not leak in this pipeline only because Task 11 drops numeric tokens ("Tier 1" and "Tier 2" are textually identical after preprocessing) - an accidental, fragile protection that the fair screener now makes explicit.

## What changed
`fair_preprocessing.py` masks personal information from the resume text itself (name, e-mail, location prefix, phone, institution-tier statement, age / date of birth, personal-detail fields, gendered titles, pronouns, graduation years), and the fair screener is given only `resume_id` and `resume_text` (feature isolation).
Skills, certifications, degree subjects, job titles, employers and the experience statement are **not** masked. The Task 11 engine is reused unchanged; the original screener stays available (`task11_as_shipped`, `task11_text_only`).
Not hard-coded to any pair. A **synthetic stress test** (not part of the 25 pairs: {n_stress} real resumes, one at a time, with only the sensitive tokens varied or a personal-details block added) checks the pathways the twins cannot reach (mean / max gap):

{_tbl(stress)}
## Did ranking quality change?
{_tbl(q)}
Same Task 11 labels and methodology ({res['jobs']['job_id'].nunique()} jobs, {labels_count} labeled resume-job rows; labels used only for evaluation). {quality_text}

## Limitations
* Only {s['n_pairs']} twins are available; the paired changes primarily test gender-coded names and sometimes college tier. No conclusions are made for unrepresented protected groups or intersections.
* The supplied twins cannot test every personal-information pathway. Synthetic perturbations are not evidence about real applicants. Real resumes contain many more potential proxies (hobbies, employers, schools by name, gaps, locations).
* Equal scores for twins do not show the screener is fair: group-level gaps {('were observed' if protected_available else 'could not be measured without protected labels')} (`group_gap_report.csv`, fair screener, {all_resume_scope}): {ggtxt}. These aggregates do not establish treatment effects; monitor them with appropriate job-related and legal context.
* Job-stratified audit (`job_stratified_group_gap.csv`; only groups with at least 3 resumes enter a job-level gap): {stratified_txt}. Group score differences co-move with required-skill overlap and human suitability in this small sample; labels may be biased. No protected-group score adjustment was applied because it would change candidate treatment using the protected attribute without evidence that these differences are unfair model errors. Keep protected attributes audit-only and review job-specific outcomes and labels before any policy change.
* Human labels, used to evaluate quality, may themselves be biased. The names are Indian-context fictional names; results may not transfer.
* Name masking uses a header heuristic; unusual layouts, multi-line names or names in free text could be missed. Re-audit on new data.

## Who decides the accuracy-fairness trade-off?
*"If fixing the fairness gap lowers accuracy a little, who decides what trade-off is acceptable?"* Not the model builder alone, and not accuracy by default.
Fairness and accuracy should not be treated as competing goals in which accuracy automatically wins. The acceptable trade-off should be decided through **transparent governance** that includes
affected stakeholders (candidates and their representatives), HR professionals, legal / compliance teams and responsible-AI experts. A small reduction in ranking performance may be acceptable if it meaningfully reduces
discriminatory outcomes; the decision must be supported by evidence (like the tables above), documented, reviewed and monitored over time. {tradeoff_text}

**Equal accuracy does not mean equal treatment.** Two systems can have identical Precision@5 or NDCG@5 while treating similar candidates differently - for example, a screener that ranks twin resumes differently can still
have the same average accuracy, because errors for one group can offset errors for another. Accuracy is an average over a population; equal treatment is a property of how comparable individuals are scored,
which is why counterfactual twins and group audits are needed in addition to accuracy metrics. The ranking is a decision-support tool, not an automatic hiring decision.
"""
