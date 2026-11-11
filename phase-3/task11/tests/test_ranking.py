"""Pipeline-level tests: coverage, ordering, score validity, fairness restrictions, leakage and missing-value safety."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cortexhr_task11 import data_loading as dl
from cortexhr_task11 import job_processor, paths, ranking_engine, resume_processor
from cortexhr_task11.pipeline import run
from cortexhr_task11.ranking_engine import RankingConfig, rank_resumes, top_k_for_job
from cortexhr_task11.skill_extractor import build_default_extractor

SRC = Path(__file__).resolve().parents[1] / "src" / "cortexhr_task11"


# ---------------------------------------------------------------- coverage / integrity
def test_all_jobs_and_resumes_processed(result):
    c = result["summary"]["counts"]
    assert c["jobs"] == 10 and c["resumes"] == 185
    assert len(result["profiles"]) == 10 and len(result["processed"]) == 185
    assert all(len(p.required) > 0 for p in result["profiles"])


def test_every_valid_pair_scored_once_and_no_missing_ids(result):
    r, jobs, res = result["ranked"], result["jobs"], result["resumes"]
    assert len(r) == len(jobs) * len(res) == 1850
    assert not r.duplicated(["job_id", "resume_id"]).any()
    assert set(r["resume_id"]) == set(res["resume_id"]) and set(r["job_id"]) == set(jobs["job_id"])
    assert (r.groupby("job_id")["resume_id"].nunique() == len(res)).all()


def test_scores_valid_and_formula(result):
    r = result["ranked"]
    for c in ("tfidf_similarity", "skill_overlap", "final_score"):
        assert r[c].notna().all() and r[c].between(0, 1).all()
    np.testing.assert_allclose(r["final_score"], 0.6 * r["tfidf_similarity"] + 0.4 * r["skill_overlap"], atol=1e-12)


def test_ranking_order_is_correct(result):
    for _, g in result["ranked"].groupby("job_id"):
        g = g.sort_values("rank")
        assert g["rank"].tolist() == list(range(1, len(g) + 1))
        assert (g["final_score"].diff().dropna() <= 1e-15).all()          # non-increasing down the ranking
        ties = g[g.duplicated("final_score", keep=False)]
        for _, t in ties.groupby("final_score"):                           # exact ties: resume_id ascending
            assert t["resume_id"].tolist() == sorted(t["resume_id"])


def test_overlap_matches_explanation_fields(result):
    for r in result["ranked"].itertuples():
        m = [x for x in str(r.matched_skills).split("; ") if x and x != "nan"]
        n_missing = [x for x in str(r.missing_skills).split("; ") if x and x != "nan"]
        assert len(m) + len(n_missing) == r.n_required_skills
        assert r.skill_overlap == pytest.approx(len(m) / r.n_required_skills)


def test_required_skills_come_from_job_description_columns(result):
    jobs = result["jobs"].set_index("job_id")
    for p in result["profiles"]:
        assert p.required_skills == [s.strip() for s in jobs.loc[p.job_id, "required_skills"].split(";")]


# ---------------------------------------------------------------- fairness restrictions
IDENTITY_COLS = ["candidate_name", "email", "city", "highest_education", "college_tier"]


def test_scoring_text_contains_no_identity_or_proxy_strings(result):
    res = result["resumes"].set_index("resume_id")
    for row in result["processed"].itertuples():
        t = row.scoring_text
        r = res.loc[row.resume_id]
        for part in str(r["candidate_name"]).lower().split():
            assert not re.search(rf"\b{re.escape(part)}\b", t), (row.resume_id, part)
        assert "mail-demo" not in t and "@" not in t and "tier" not in t.split()
        assert str(r["city"]).lower() not in t
        assert not re.search(
            r"\b(?:male|female|non[- ]?binary|he|him|his|she|her|hers|they|them|their|theirs|"
            r"age|aged|born|dob|pronouns?|gender|stanford|harvard|university|college|institute|"
            r"ivy league|prestigious|elite)\b",
            t,
            flags=re.I,
        )


def _variant(resumes, rid, **repl):
    """Same resume text, different name / city / college tier / education (columns and text changed together)."""
    out = resumes.copy()
    i = out.index[out["resume_id"] == rid][0]
    text = out.at[i, "resume_text"]
    for col, new in repl.items():
        old = str(out.at[i, col])
        text = text.replace(old, new).replace(old.upper(), new.upper())
        out.at[i, col] = new
    out.at[i, "resume_text"] = text
    return out


def test_name_city_college_education_do_not_change_scores(result):
    jobs, res = result["jobs"], result["resumes"]
    ex = build_default_extractor(jobs, res)
    profiles = job_processor.process_jobs(jobs, ex)
    base = rank_resumes(resume_processor.process_resumes(res, ex), profiles)
    rid = "RES-0001"
    changed = _variant(res, rid, candidate_name="Zoya Qureshi", city="Shimla", college_tier="Tier 1", highest_education="MBA")
    alt = rank_resumes(resume_processor.process_resumes(changed, ex), profiles)
    a = base[base.resume_id == rid].set_index("job_id")[["final_score", "tfidf_similarity", "skill_overlap"]]
    b = alt[alt.resume_id == rid].set_index("job_id")[["final_score", "tfidf_similarity", "skill_overlap"]]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-6)


def test_demographic_and_university_mentions_do_not_change_scores(data_copy, result):
    df = pd.read_csv(data_copy / "resumes.csv")
    i = df.index[df.resume_id == "RES-0001"][0]
    original_text = df.at[i, "resume_text"]
    name = df.at[i, "candidate_name"]
    df["age"] = ""
    df["gender"] = ""
    df["date_of_birth"] = ""
    df["pronouns"] = ""
    df["university"] = ""
    df.at[i, "age"] = "24"
    df.at[i, "gender"] = "Female"
    df.at[i, "date_of_birth"] = "Born in 2002"
    df.at[i, "pronouns"] = "she/her"
    df.at[i, "university"] = "Stanford University"
    df.at[i, "college_tier"] = "Tier 1"
    df.at[i, "resume_text"] = (
        f"Candidate Name: {name} | Female | she/her | Age: 24 | Born in 2002 | Tier-1 college | "
        f"Stanford University | {original_text}"
    )
    df.to_csv(data_copy / "resumes.csv", index=False)
    altered = run(data_copy, data_copy / "out", write=False)
    base = result["ranked"].query("resume_id == 'RES-0001'").set_index("job_id")
    actual = altered["ranked"].query("resume_id == 'RES-0001'").set_index("job_id")
    pd.testing.assert_frame_equal(
        base[["final_score", "tfidf_similarity", "skill_overlap"]],
        actual[["final_score", "tfidf_similarity", "skill_overlap"]],
        check_exact=False,
        atol=1e-12,
    )
    processed_text = altered["processed"].set_index("resume_id").loc["RES-0001", "scoring_text"]
    for token in ("female", "she", "her", "age", "born", "stanford", "university", "tier"):
        assert token not in processed_text.lower()


def test_years_of_experience_do_not_determine_score(result):
    jobs, res = result["jobs"], result["resumes"]
    ex = build_default_extractor(jobs, res)
    profiles = job_processor.process_jobs(jobs, ex)
    base = rank_resumes(resume_processor.process_resumes(res, ex), profiles)
    changed = res.copy()
    i = changed.index[changed.resume_id == "RES-0001"][0]
    changed.at[i, "resume_text"] = changed.at[i, "resume_text"].replace("4.8 years", "29 years").replace("(2.2 yrs)", "(20 yrs)")
    alt = rank_resumes(resume_processor.process_resumes(changed, ex), profiles)
    a, b = (x[x.resume_id == "RES-0001"].set_index("job_id") for x in (base, alt))
    assert b["experience_years"].iloc[0] == 29.0 and a["experience_years"].iloc[0] == 4.8      # extracted value changed...
    pd.testing.assert_series_equal(a["final_score"], b["final_score"], check_exact=False, atol=1e-6)  # ...score did not


def test_ranking_inputs_exclude_identity_columns(result):
    cols = set(result["processed"].columns)
    assert not cols & set(IDENTITY_COLS + ["total_experience_years", "target_job_family", "certifications",
                                           "age", "gender", "date_of_birth", "pronouns", "university"])
    assert not {"human_suitability_score", "shortlist", "required_skill_coverage"} & cols


def test_structured_resume_columns_do_not_change_scores(data_copy, result):
    """Permuting structured columns across resumes (skills list, certifications, stated experience, target family, title, company,
    file format, layout) must not move a single score: only the resume TEXT and the job text are scored."""
    df = pd.read_csv(data_copy / "resumes.csv")
    rng = np.random.default_rng(1)
    for c in ["skills", "certifications", "total_experience_years", "target_job_family", "current_title", "current_company",
              "source_format", "layout_difficulty"]:
        df[c] = rng.permutation(df[c].to_numpy())
    df.to_csv(data_copy / "resumes.csv", index=False)
    permuted = run(data_copy, data_copy / "out", write=False)
    pd.testing.assert_frame_equal(permuted["ranked"], result["ranked"])


def test_experience_is_reported_but_weakly_related_to_score(result):
    assert abs(result["summary"]["experience_dependence"]["spearman_rho_score_vs_experience"]) < 0.3


# ---------------------------------------------------------------- leakage
FORBIDDEN = ["human_suitability", "shortlist", "resume_job_labels", "load_labels", "required_skill_coverage", "nice_skill_coverage",
             "reviewer_note", "candidate_name", "college_tier", "gender", "date_of_birth"]


@pytest.mark.parametrize("module", ["ranking_engine.py", "resume_processor.py", "job_processor.py", "skill_extractor.py", "preprocessing.py"])
def test_scoring_modules_never_reference_labels_or_protected_columns(module):
    src = (SRC / module).read_text(encoding="utf-8")
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    code = re.sub(r'""".*?"""', "", code, flags=re.S)                                # ignore documentation strings
    if module == "resume_processor.py":                                              # protected values appear only in the redaction allowlist
        code = re.sub(r"SCRUB_COLUMNS = \[.*?\]", "", code, flags=re.S)
    if module == "preprocessing.py":                                                 # protected terms appear only in scrubber patterns/logic
        code = re.sub(r"_(?:TIER|PRESTIGE|INSTITUTION_NAME|UNIVERSITY_OF|KNOWN_INSTITUTIONS|AGE|DEMOGRAPHIC_LABEL|GENDER)\s*=\s*re\.compile\(.*?\n\)", "", code, flags=re.S)
        code = re.sub(r"def scrub_identity\(.*?(?=\ndef tfidf_text)", "", code, flags=re.S)
    for word in FORBIDDEN:
        assert word not in code, f"{word} found in {module}"


def test_scores_do_not_depend_on_labels(data_copy, result):
    lab = pd.read_csv(data_copy / "resume_job_labels.csv")
    rng = np.random.default_rng(0)
    lab["human_suitability_score"] = rng.permutation(lab["human_suitability_score"].to_numpy())
    lab["shortlist"] = np.where(lab["human_suitability_score"] >= 4, "Y", "N")
    lab.to_csv(data_copy / "resume_job_labels.csv", index=False)
    shuffled = run(data_copy, data_copy / "out", write=False)
    pd.testing.assert_frame_equal(shuffled["ranked"], result["ranked"])                       # scores identical
    assert shuffled["quality"].iloc[-1]["ndcg_at_5"] != result["quality"].iloc[-1]["ndcg_at_5"]  # metrics respond to labels


# ---------------------------------------------------------------- robustness
def test_missing_values_do_not_crash():
    jobs = pd.DataFrame({"job_id": ["J1", "J2"], "job_title": ["Dev", None], "min_experience_years": [2, np.nan],
                         "required_skills": ["Python; SQL", np.nan], "nice_to_have_skills": [np.nan, "Git"], "description_text": [None, "text"]})
    res = pd.DataFrame({"resume_id": ["R1", "R2", "R3"], "resume_text": ["Python and SQL with 3 years of experience.", np.nan, ""],
                        "candidate_name": [np.nan, "Ann Lee", None]})
    ex = build_default_extractor(jobs)
    out = rank_resumes(resume_processor.process_resumes(res, ex), job_processor.process_jobs(jobs, ex))
    assert len(out) == 6 and out["final_score"].between(0, 1).all() and out["final_score"].notna().all()
    r1 = out[(out.job_id == "J1") & (out.resume_id == "R1")].iloc[0]
    assert r1["skill_overlap"] == 1.0 and r1["experience_years"] == 3.0
    r2 = out[(out.job_id == "J1") & (out.resume_id == "R2")].iloc[0]
    assert r2["skill_overlap"] == 0.0 and np.isnan(r2["experience_years"])
    assert out[out.job_id == "J2"]["n_required_skills"].eq(0).all()


def test_invalid_config_and_duplicates_rejected():
    with pytest.raises(ValueError):
        RankingConfig(tfidf_weight=0.7, skill_weight=0.7)
    ex = build_default_extractor()
    res = pd.DataFrame({"resume_id": ["R1", "R1"], "resume_text": ["Python", "SQL"]})
    jobs = pd.DataFrame({"job_id": ["J1"], "job_title": ["Dev"], "min_experience_years": [1], "required_skills": ["Python"],
                         "nice_to_have_skills": [""], "description_text": ["x"]})
    with pytest.raises(ValueError):
        rank_resumes(resume_processor.process_resumes(res, ex), job_processor.process_jobs(jobs, ex))


def test_empty_inputs():
    assert rank_resumes(pd.DataFrame(columns=["resume_id"]), []).empty


# ---------------------------------------------------------------- API-shaped output
def test_top_k_for_job_shape(result):
    out = top_k_for_job(result["ranked"], "J04", top_k=5)
    assert out["job_id"] == "J04" and len(out["ranked_candidates"]) == 5
    first = out["ranked_candidates"][0]
    assert set(first) == {"resume_id", "rank", "final_score", "tfidf_similarity", "skill_overlap", "matched_skills", "missing_skills", "experience_years"}
    assert [c["rank"] for c in out["ranked_candidates"]] == [1, 2, 3, 4, 5]
    assert isinstance(first["matched_skills"], list)
    with pytest.raises(KeyError):
        top_k_for_job(result["ranked"], "NOPE")
    with pytest.raises(ValueError):
        top_k_for_job(result["ranked"], "J04", top_k=0)


# ---------------------------------------------------------------- data loading
def test_loader_validation(tmp_path):
    (tmp_path / "r.csv").write_text("resume_id,resume_text\nR1,a\nR1,b\n")
    with pytest.raises(dl.DataError):
        dl.load_resumes(tmp_path / "r.csv")
    (tmp_path / "l.csv").write_text("resume_id,job_id,human_suitability_score,shortlist\nR1,J1,9,Y\n")
    with pytest.raises(dl.DataError):
        dl.load_labels(tmp_path / "l.csv")
    (tmp_path / "duplicate_labels.csv").write_text(
        "resume_id,job_id,human_suitability_score,shortlist\nR1,J1,4,Y\nR1,J1,3,N\n"
    )
    with pytest.raises(dl.DataError):
        dl.load_labels(tmp_path / "duplicate_labels.csv")
    with pytest.raises(dl.DataError):
        dl.load_jobs(tmp_path / "missing.csv")
