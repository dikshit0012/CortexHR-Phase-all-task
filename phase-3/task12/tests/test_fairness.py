"""Task 12 tests. General behaviour and properties only - no expected values are copied from the validation pairs."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cortexhr_task12 import paths
from cortexhr_task12.counterfactual_testing import competition_rank, primary_jobs, score_pairs
from cortexhr_task12.cortexhr_helpers import group_gap
from cortexhr_task12.data_loading import AuditDataError, load_pairs, load_protected
from cortexhr_task12.fair_preprocessing import AUDIT_ONLY_COLUMNS, mask_personal_information as mask
from cortexhr_task12.fairness_metrics import compare_summaries, gap_summary, group_gap_report, reduction_pct, relative_gap
from cortexhr_task12.screeners import AS_SHIPPED, FAIR, TEXT_ONLY, process_resumes_fair, run_screener, text_only_view

from cortexhr_task11 import data_loading as dl11  # noqa: E402  (importable once cortexhr_task12.paths ran)
from cortexhr_task11.skill_extractor import build_default_extractor  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src" / "cortexhr_task12"
required_inputs = (
    paths.TASK11_DATA / "job_descriptions.csv",
    paths.TASK11_DATA / "resumes.csv",
    paths.PAIRS_CSV,
)
missing_inputs = [path for path in required_inputs if not path.is_file()]
if missing_inputs:
    pytest.skip(
        "Task 12 test module requires the approved Task 11 and twin-pair CSVs: "
        + ", ".join(map(str, missing_inputs)),
        allow_module_level=True,
    )
JOBS = dl11.load_jobs(paths.TASK11_DATA / "job_descriptions.csv")
RESUMES = dl11.load_resumes(paths.TASK11_DATA / "resumes.csv")


def _text(name="Anita Deshpande", email="anita.deshpande@mail-demo.example", extra=""):
    return (f"{name.upper()}\nPune | {email}\n\nSUMMARY\nExperienced analyst with 4 years of experience. Strengths include Python, SQL, MS Excel.\n\n"
            f"EXPERIENCE\nData Analyst, Acme Corp (2.2 yrs)\n- Built dashboards with Power BI\n\nEDUCATION\nMBA, Tier 2 institution\n{extra}")


# ------------------------------------------------------------------ masking rules
def test_name_email_location_tier_are_masked():
    out, counts = mask(_text())
    low = out.lower()
    for tok in ("anita", "deshpande", "mail-demo", "@", "tier", "institution", "pune"):
        assert tok not in low
    assert "MBA" in out                                   # the degree itself is job-relevant and is kept
    assert {"name", "email", "location", "tier"} <= set(counts)


def test_two_column_header_name_is_masked():
    t = "FATHIMA FERNANDES                         EXPERIENCE\nCoimbatore | fathima.fernandes@mail-demo.exampleSales Rep (0.3 yrs)\nSKILLS\nCRM"
    out, _ = mask(t)
    assert "fathima" not in out.lower() and "fernandes" not in out.lower() and "Sales Rep" in out and "CRM" in out


def test_age_dob_personal_fields_titles_pronouns_graduation_masked():
    t = ("Mr. Rahul Verma\nDOB: 12 May 1990\nAge: 34 years old. Gender: Male\nMarital status: Single\nHe led Python projects. His SQL is strong. "
         "Graduated May 2015, class of 2015. Call +91 98765 43210.")
    out, c = mask(t)
    low = out.lower()
    for tok in ("rahul", "1990", "34", "male", "single", "2015", "98765", " he ", " his ", "mr."):
        assert tok not in low, tok
    assert "Python" in out and "SQL" in out
    assert {"age", "personal", "pronoun", "graduation", "phone"} <= set(c)


def test_job_relevant_content_is_not_masked():
    t = "Skills: MS Excel, MS Office, Python\nPersonal projects in Machine Learning\nWith 4.8 years of experience. Engineer (2.2 yrs) 2.2 yrs\nMs. Rao referred me"
    out, _ = mask(t)
    for keep in ("MS Excel", "MS Office", "Personal projects in Machine Learning", "4.8 years of experience", "2.2 yrs"):
        assert keep in out
    assert "Ms." not in out


@pytest.mark.parametrize("bad", [None, float("nan"), "", "   ", pd.NA])
def test_masking_handles_missing_values(bad):
    out, c = mask(bad)
    assert isinstance(out, str)


def test_unknown_category_rejected():
    with pytest.raises(ValueError):
        mask("x", {"nonsense"})


def test_non_name_first_line_is_not_masked_as_a_name():
    out, c = mask("Software Engineer\nPython, SQL")
    assert "Software Engineer" in out and "name" not in c


# ------------------------------------------------------------------ equal treatment on synthetic counterfactuals
def _score(texts):
    df = pd.DataFrame({"resume_id": [f"R{i}" for i in range(len(texts))], "resume_text": texts})
    jobs = JOBS.iloc[:1].copy()
    out = run_screener(FAIR, df, jobs)
    return out.sort_values("resume_id")["final_score"].to_numpy()


def _corpus(variant_text):
    others = [_text(f"Person{i} Surname{i}", f"p{i}.s{i}@mail-demo.example", extra=f"Extra{i}") for i in range(6)]
    return [variant_text] + others


def test_names_do_not_influence_fair_score():
    a = _score(_corpus(_text("Anita Deshpande", "anita.deshpande@mail-demo.example")))[0]
    b = _score(_corpus(_text("Rohit Verma", "rohit.verma@mail-demo.example")))[0]
    assert a == pytest.approx(b, abs=1e-12)


def test_gender_terms_do_not_influence_fair_score():
    base = _score(_corpus(_text()))[0]
    for extra in ("He prefers teams. His availability is flexible.", "She prefers teams. Her availability is flexible.", "Gender: Female", "Gender: Male"):
        v = _score(_corpus(_text(extra=extra)))[0]
        no_pron = _score(_corpus(_text(extra="prefers teams. availability is flexible." if "prefers" in extra else "")))[0]
        assert v == pytest.approx(no_pron, abs=1e-12)
    assert base == pytest.approx(_score(_corpus(_text(extra="Gender: Female")))[0], abs=1e-12)


def test_age_and_dob_do_not_influence_fair_score():
    a = _score(_corpus(_text()))[0]
    b = _score(_corpus(_text(extra="Date of birth: 14 March 1999\nAge: 25 years old")))[0]
    c = _score(_corpus(_text(extra="Date of birth: 3 October 1969\nAge: 55 years old")))[0]
    assert a == pytest.approx(b, abs=1e-12) and a == pytest.approx(c, abs=1e-12)


def test_graduation_year_does_not_influence_fair_score():
    a = _score(_corpus(_text(extra="Graduated May 2003, class of 2003")))[0]
    b = _score(_corpus(_text(extra="Graduated June 2022, class of 2022")))[0]
    assert a == pytest.approx(b, abs=1e-12)


def test_job_relevant_skill_still_changes_the_score():
    """Fairness masking must not blind the screener: genuinely different skills still give different scores."""
    with_skill = _score(_corpus(_text().replace("Python, SQL", "Python, SQL, Docker, Git")))[0]
    without = _score(_corpus(_text()))[0]
    assert with_skill != pytest.approx(without, abs=1e-9)


# ------------------------------------------------------------------ real data properties
def test_job_relevant_skills_survive_masking_for_every_resume():
    ex = build_default_extractor(JOBS)
    fair = process_resumes_fair(RESUMES, ex).set_index("resume_id")
    for rid, text in zip(RESUMES["resume_id"], RESUMES["resume_text"]):
        assert set(fair.loc[rid, "skill_keys"]) == set(ex.extract_keys(text)), rid


def test_experience_statement_survives_masking():
    from cortexhr_task11.preprocessing import extract_experience_years

    ex = build_default_extractor(JOBS)
    fair = process_resumes_fair(RESUMES, ex).set_index("resume_id")
    for rid, text in zip(RESUMES["resume_id"], RESUMES["resume_text"]):
        raw = extract_experience_years(text)[0]
        got = fair.loc[rid, "experience_years"]
        if raw is None:
            assert pd.isna(got), f"{rid}: expected no extracted experience, got {got!r}"
        else:
            assert got == pytest.approx(raw), f"{rid}: expected {raw} years, got {got!r}"


def test_twins_get_identical_preprocessing_rules():
    pairs = load_pairs(paths.PAIRS_CSV, RESUMES)
    ex = build_default_extractor(JOBS)
    fair = process_resumes_fair(RESUMES, ex).set_index("resume_id")
    for p in pairs.itertuples():
        a, b = fair.loc[p.resume_id_original], fair.loc[p.resume_id_twin]
        assert a["scoring_text"] == b["scoring_text"] and a["skill_keys"] == b["skill_keys"]
        assert a["masked_items"] == b["masked_items"]


def test_feature_isolation_columns():
    assert list(text_only_view(RESUMES).columns) == ["resume_id", "resume_text"]
    with pytest.raises(ValueError):
        text_only_view(RESUMES.drop(columns=["resume_text"]))
    assert {"inferred_gender", "age", "college_tier", "candidate_name"} <= AUDIT_ONLY_COLUMNS


def test_protected_and_structured_columns_cannot_change_fair_scores():
    base = run_screener(FAIR, RESUMES, JOBS)
    shuffled = RESUMES.copy()
    rng = np.random.default_rng(3)
    for c in ["candidate_name", "email", "city", "college_tier", "highest_education", "skills", "total_experience_years", "target_job_family"]:
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
    shuffled["inferred_gender"] = rng.choice(["Male", "Female"], len(shuffled))
    pd.testing.assert_frame_equal(base, run_screener(FAIR, shuffled, JOBS))


def test_scoring_code_never_touches_protected_data_labels_or_pair_ids():
    for module in ("fair_preprocessing.py", "screeners.py"):
        code = re.sub(r'""".*?"""', "", (SRC / module).read_text(encoding="utf-8"), flags=re.S)
        code = re.sub(r"AUDIT_ONLY_COLUMNS = frozenset\(\{.*?\}\)", "", code, flags=re.S)       # a blocklist of names, not a data access
        code = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("#"))
        for forbidden in ("unlock", "inferred_gender", "PROTECTED_CSV", "load_protected", "resume_job_labels", "human_suitability", "shortlist", "load_labels"):
            assert forbidden not in code, f"{forbidden} in {module}"
    for f in SRC.glob("*.py"):
        assert not re.search(r"BP-\d\d", f.read_text(encoding="utf-8")), f"pair id hard-coded in {f.name}"
        assert not re.search(r"RES-\d{4}", f.read_text(encoding="utf-8")), f"resume id hard-coded in {f.name}"


def test_as_shipped_reproduces_committed_task11_scores():
    rep = pd.read_csv(paths.TASK11_ROOT / "reports" / "ranked_candidates.csv")
    cur = run_screener(AS_SHIPPED, RESUMES, JOBS).merge(rep[["job_id", "resume_id", "final_score"]], on=["job_id", "resume_id"], suffixes=("", "_rep"))
    assert len(cur) == 1850 and (cur["final_score"] - cur["final_score_rep"]).abs().max() < 2e-6


# ------------------------------------------------------------------ audit results (properties, not copied numbers)
def test_all_25_pairs_scored_with_both_twins(audit):
    r = audit["res"]
    assert r["summary"]["n_pairs"] == 25
    for n in (AS_SHIPPED, TEXT_ONLY, FAIR):
        p = r["primary"][n]
        assert len(p) == 25 and p["pair_id"].nunique() == 25
        assert p[["score_a", "score_b"]].notna().all().all() and p[["score_a", "score_b"]].apply(lambda s: s.between(0, 1)).all().all()
        assert len(r["all_jobs"][n]) == 250
    ids = set(r["primary"][FAIR]["resume_a_id"]) | set(r["primary"][FAIR]["resume_b_id"])
    assert len(ids) == 50


def test_gap_equals_absolute_score_difference(audit):
    for p in audit["res"]["primary"].values():
        np.testing.assert_allclose(p["twin_gap"], (p["score_a"] - p["score_b"]).abs(), atol=1e-15)


def test_fair_screener_is_not_less_fair_and_within_tolerance(audit):
    s = audit["res"]["summaries"]
    assert s[FAIR]["average_gap"] <= s[TEXT_ONLY]["average_gap"] + 1e-15
    assert s[FAIR]["average_gap"] <= s[AS_SHIPPED]["average_gap"] + 1e-9
    assert s[FAIR]["pairs_exceeding_tolerance"] == 0


def test_fix_does_not_destroy_ranking_quality(audit):
    q = audit["res"]["quality"].set_index("metric")
    assert abs(q.loc["NDCG@5", "difference_fair_minus_as_shipped"]) < 0.02           # governance threshold for this regression test
    assert abs(q.loc["Precision@5", "difference_fair_minus_as_shipped"]) < 0.02
    assert set(q.index) == {"Precision@5", "Recall@5", "NDCG@5", "Top-5 Hits"}
    assert q.loc[["Precision@5", "Recall@5", "NDCG@5"], [AS_SHIPPED, FAIR]].apply(lambda s: s.between(0, 1)).all().all()


def test_twins_differ_only_in_name_email_tier(audit):
    d = audit["res"]["diffs"]
    assert (d["other (unintended difference)"] == 0).all()


def test_skill_extraction_consistent_across_twins(audit):
    for c in audit["res"]["summary"]["extraction_consistency"].values():
        assert c["inconsistent_skill_extraction"] == 0 and c["inconsistent_experience_extraction"] == 0


def test_stress_test_fair_not_worse_than_text_only(audit):
    st = audit["res"]["stress"].pivot(index="perturbation", columns="screener", values="mean_gap")
    assert (st[FAIR] <= st[TEXT_ONLY] + 1e-12).all() and (st[FAIR] < 0.01).all()


# ------------------------------------------------------------------ reports
def test_reports_generated_and_protected_values_stay_private(audit):
    d = audit["dir"]
    for f in ["fairness_before.csv", "fairness_after.csv", "fairness_comparison.csv", "ranking_quality_comparison.csv", "fairness_summary.md",
              "bias_source_ablation.csv", "run_summary.json", "synthetic_stress_test.csv", "group_gap_report.csv",
              "job_stratified_group_gap.csv",
              "twin_gap_before.csv", "twin_gap_after.csv", "fairness_audit_report.json", "fairness_statement.md",
              "twin_gap_comparison.svg"]:
        assert (d / f).exists() and (d / f).stat().st_size > 0, f
    before, after = pd.read_csv(d / "fairness_before.csv"), pd.read_csv(d / "fairness_after.csv")
    assert len(before) == 50 and set(before["screener"]) == {AS_SHIPPED, TEXT_ONLY} and len(after) == 25 and set(after["screener"]) == {FAIR}
    twin_before, twin_after = pd.read_csv(d / "twin_gap_before.csv"), pd.read_csv(d / "twin_gap_after.csv")
    assert len(twin_before) == len(twin_after) == 25
    assert "signed_gap_b_minus_a" in twin_before and "twin_gap" in twin_before
    report = json.loads((d / "fairness_audit_report.json").read_text(encoding="utf-8"))
    assert report["pairs_tested"] == 25 and report["fairness_before_fix"]["n_pairs"] == 25
    assert report["fairness_after_fix"]["n_pairs"] == 25
    stratified = pd.read_csv(d / "job_stratified_group_gap.csv")
    assert {"attribute", "job_id", "group", "n_candidates", "within_job_score_gap"} <= set(stratified.columns)
    assert not {"resume_id", "candidate_name", "email"} & set(stratified.columns)
    assert report["limitations"] and len(report["limitations"]) >= 5
    assert "accuracy-fairness trade-off" in (d / "fairness_statement.md").read_text(encoding="utf-8")
    for df in (before, after, pd.read_csv(d / "fairness_all_jobs_250_comparisons.csv")):
        assert not {"gender_a", "gender_b", "tier_a", "tier_b", "inferred_gender", "college_tier"} & set(df.columns)
    assert (d / "private" / "protected_pair_details.csv").exists()
    assert json.loads((d / "run_summary.json").read_text())["n_pairs"] == 25


def test_summary_numbers_match_reports(audit):
    d, r = audit["dir"], audit["res"]
    after = pd.read_csv(d / "fairness_after.csv")
    assert after["twin_gap"].mean() == pytest.approx(r["summaries"][FAIR]["average_gap"], abs=1e-7)
    text = (d / "fairness_summary.md").read_text(encoding="utf-8")
    assert "Who decides the accuracy-fairness trade-off" in text and "Equal accuracy does not mean equal treatment" in text and "25" in text
    assert "job_stratified_group_gap.csv" in text


def test_job_stratified_group_analysis_is_descriptive_and_auditable(audit):
    res = audit["res"]
    table = res["job_stratified_group"]
    summary = res["summary"]["job_stratified_group_analysis"]
    assert len(table) > 0
    assert set(table["attribute"]) == {"inferred_gender", "college_tier"}
    assert (table["n_candidates"] > 0).all()
    assert table["within_job_score_gap"].notna().any()
    assert summary["inferred_gender"]["jobs_with_comparable_groups"] > 0
    assert summary["college_tier"]["jobs_with_comparable_groups"] > 0
    assert summary["inferred_gender"]["equal_job_weighted_mean_gap"] > 0
    assert summary["college_tier"]["equal_job_weighted_mean_gap"] > 0
    assert "resume_id" not in table.columns


# ------------------------------------------------------------------ metrics & helpers
def test_gap_summary_hand_computed():
    s = gap_summary([0.0, 0.002, 0.02, 0.0], tolerance=0.01)
    assert s["average_gap"] == pytest.approx(0.0055) and s["median_gap"] == pytest.approx(0.001) and s["maximum_gap"] == 0.02 and s["minimum_gap"] == 0.0
    assert s["non_zero_gap_pairs"] == 2 and s["pairs_exceeding_tolerance"] == 1


def test_reduction_is_safe_and_honest():
    assert reduction_pct(0.2, 0.05) == pytest.approx(75.0)
    assert np.isnan(reduction_pct(0.0, 0.0)) and np.isnan(reduction_pct(0.0, 0.1))
    assert reduction_pct(0.1, 0.2) == pytest.approx(-100.0)                 # a worse result is reported as negative, not hidden


def test_compare_table_shape():
    t = compare_summaries(gap_summary([0.1, 0.2]), gap_summary([0.0, 0.1]))
    assert list(t["metric"][:5]) == ["Average twin gap", "Median twin gap", "Maximum twin gap", "Minimum twin gap", "Non-zero gap pairs"]
    assert t.loc[0, "improvement_pct"] == pytest.approx(66.6667, abs=1e-3)


def test_relative_gap_and_empty_input():
    assert relative_gap(0.4, 0.6) == pytest.approx(0.4) and np.isnan(relative_gap(0, 0))
    with pytest.raises(ValueError):
        gap_summary([])


def test_group_gap_uses_course_helper():
    rep = group_gap_report([0.2, 0.4, 0.6, 0.8], ["a", "a", "b", "b"])
    assert rep["group_means"] == {"a": pytest.approx(0.3), "b": pytest.approx(0.7)} and rep["max_gap"] == pytest.approx(0.4)


def test_group_gap_handles_unequal_sizes_and_missing_values():
    means, gap = group_gap([0.2, 0.4, 0.6, 0.8, 1.0], ["a", "a", "b", "b", "b"])
    assert means == {"a": pytest.approx(0.3), "b": pytest.approx(0.8)}
    assert gap == pytest.approx(0.5)

    means, gap = group_gap([0.2, np.nan, 0.6, 0.8], ["a", "a", "b", None])
    assert means == {"a": pytest.approx(0.2), "b": pytest.approx(0.6)}
    assert gap == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("scores", "groups"),
    [([0.2, 0.4], ["a"]), ([np.nan], ["a"]), ([float("inf")], ["a"])],
)
def test_group_gap_rejects_mismatched_or_unusable_inputs(scores, groups):
    with pytest.raises(ValueError):
        group_gap(scores, groups)


def test_competition_rank_shares_ties():
    assert competition_rank(pd.Series([0.9, 0.5, 0.5, 0.1])).tolist() == [1, 2, 2, 4]


def test_score_pairs_records_signed_direction_and_absolute_gap():
    pairs = pd.DataFrame([{
        "pair_id": "P1", "resume_id_original": "A", "resume_id_twin": "B", "attribute_changed": "name",
    }])
    ranked = pd.DataFrame([
        {"job_id": "J1", "resume_id": "A", "final_score": 0.2, "tfidf_similarity": 0.2, "skill_overlap": 0.2},
        {"job_id": "J1", "resume_id": "B", "final_score": 0.5, "tfidf_similarity": 0.5, "skill_overlap": 0.5},
    ])
    result = score_pairs(ranked, pairs)
    assert result.loc[0, "signed_gap_b_minus_a"] == pytest.approx(0.3)
    assert result.loc[0, "twin_gap"] == pytest.approx(0.3)
    assert result.loc[0, "higher_scored"] == "B"
    assert result.loc[0, "non_zero_gap"]


def test_score_pairs_fails_instead_of_skipping_a_missing_twin():
    pairs = pd.DataFrame([{
        "pair_id": "P1", "resume_id_original": "A", "resume_id_twin": "B", "attribute_changed": "name",
    }])
    ranked = pd.DataFrame([
        {"job_id": "J1", "resume_id": "A", "final_score": 0.2, "tfidf_similarity": 0.2, "skill_overlap": 0.2},
    ])
    with pytest.raises(ValueError, match="lacks both twin scores"):
        score_pairs(ranked, pairs)


# ------------------------------------------------------------------ data loading
def test_pair_loader_validation(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("pair_id,resume_id_original,resume_id_twin,attribute_changed,skills_identical,experience_identical,expected_score_difference\nP1,A,B,x,Y,Y,0\nP2,B,C,x,Y,Y,0\n")
    with pytest.raises(AuditDataError):
        load_pairs(f)
    f.write_text("pair_id,resume_id_original\nP1,A\n")
    with pytest.raises(AuditDataError):
        load_pairs(f)
    f.write_text("pair_id,resume_id_original,resume_id_twin,attribute_changed,skills_identical,experience_identical,expected_score_difference\nP1,,B,x,Y,Y,0\n")
    with pytest.raises(AuditDataError, match="missing values"):
        load_pairs(f)


def test_real_pairs_are_25_and_link_existing_resumes():
    p = load_pairs(paths.PAIRS_CSV, RESUMES, expected_count=25)
    assert len(p) == 25 and (p["expected_score_difference"] == 0).all()
    assert len(primary_jobs(p, RESUMES)) == 25


def test_pair_loader_rejects_wrong_expected_count():
    with pytest.raises(AuditDataError, match="expected 24 pairs"):
        load_pairs(paths.PAIRS_CSV, RESUMES, expected_count=24)


def test_missing_protected_file_is_reported_not_invented(tmp_path):
    assert load_protected(tmp_path / "absent.csv") is None
    with pytest.raises(FileNotFoundError):
        load_protected(tmp_path / "absent.csv", required=True)


def test_fair_processing_with_missing_text():
    df = pd.DataFrame({"resume_id": ["R1", "R2"], "resume_text": [np.nan, "Python and SQL"]})
    out = process_resumes_fair(df, build_default_extractor(JOBS))
    assert len(out) == 2 and bool(out.loc[0, "empty_text"]) and out.loc[1, "n_skills"] >= 2
