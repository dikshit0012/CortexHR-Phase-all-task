import math

import numpy as np
import pandas as pd
import pytest

from cortexhr_task11.evaluation import (dcg_at_k, evaluate_micro, evaluate_pool, ndcg_at_k, precision_at_k, recall_at_k, top_k_hits)
from cortexhr_task11.misranking_analysis import find_misrankings


def test_hits_precision_recall_hand_computed():
    rel = [1, 0, 1, 0, 0, 1, 0]                       # 3 relevant overall, 2 in top 5
    assert top_k_hits(rel, 5) == 2
    assert precision_at_k(rel, 5) == pytest.approx(2 / 5)
    assert recall_at_k(rel, 5) == pytest.approx(2 / 3)


def test_precision_uses_available_candidates_when_fewer_than_k():
    assert precision_at_k([1, 0, 1], 5) == pytest.approx(2 / 3)
    assert recall_at_k([1, 0, 1], 5) == 1.0


def test_recall_and_ndcg_undefined_without_relevant():
    assert math.isnan(recall_at_k([0, 0, 0], 5)) and math.isnan(ndcg_at_k([0, 0, 0], 5))


def test_ndcg_hand_computed_exponential_gain():
    grades = [3, 0, 2, 4, 1, 4]                       # rank order of a hypothetical list; k = 5
    dcg = (2**3 - 1) / math.log2(2) + 0 + (2**2 - 1) / math.log2(4) + (2**4 - 1) / math.log2(5) + (2**1 - 1) / math.log2(6)
    ideal = sorted(grades, reverse=True)[:5]          # 4, 4, 3, 2, 1
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    assert dcg_at_k(grades, 5) == pytest.approx(dcg)
    assert ndcg_at_k(grades, 5) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_and_worst_order():
    assert ndcg_at_k([4, 3, 2, 1, 0, 0], 5) == pytest.approx(1.0)
    dcg = 15 / math.log2(5) + 7 / math.log2(6)                    # grades 4 and 3 sit at ranks 4 and 5
    idcg = 15 / 1 + 7 / math.log2(3) + 3 / math.log2(4)           # ideal order 4, 3, 2
    assert ndcg_at_k([0, 0, 0, 4, 3, 2], 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([0, 0, 0, 4, 3, 2], 5) < ndcg_at_k([4, 0, 0, 3, 2, 0], 5)


def test_ndcg_linear_gain_differs_from_exponential():
    g = [1, 3, 0, 2, 0]
    assert ndcg_at_k(g, 5, exponential=False) != pytest.approx(ndcg_at_k(g, 5, exponential=True))


def _pool(rows):
    df = pd.DataFrame(rows, columns=["job_id", "resume_id", "pool_rank", "human_suitability_score", "final_score"])
    df["job_title"] = "T"
    df["relevant"] = (df["human_suitability_score"] >= 4).astype(int)
    df["rel_grade"] = df["human_suitability_score"] - 1
    df["n_pool"] = df.groupby("job_id")["resume_id"].transform("count")
    return df


def test_evaluate_pool_small_example():
    pool = _pool([("J", f"R{i}", i, s, 1 - i / 10) for i, s in enumerate([5, 2, 4, 1, 2, 4], start=1)])
    t = evaluate_pool(pool, 5).iloc[0]
    assert t["top5_hits"] == 2 and t["precision_at_5"] == pytest.approx(0.4) and t["recall_at_5"] == pytest.approx(2 / 3)
    assert t["n_relevant"] == 3 and t["max_possible_recall_at_5"] == 1.0


def test_misranking_definition_and_symmetry():
    # R = 2 shortlisted; ranks: 1 (score 5, ok), 2 (score 3, false shortlist), 3 (score 4, missed), 4 (score 1)
    pool = _pool([("J", "A", 1, 5, .9), ("J", "B", 2, 3, .8), ("J", "C", 3, 4, .7), ("J", "D", 4, 1, .6)])
    m = find_misrankings(pool)
    assert set(zip(m["resume_id"], m["category"])) == {("B", "false_shortlist"), ("C", "missed_shortlist")}


# ---------------------------------------------------------------- on the supplied data
def test_real_quality_table(result):
    q = result["quality"]
    assert len(q) == 12 and q.iloc[-1]["job_id"] == "ALL (macro avg; counts summed)"
    jobs = q[q["job_id"].str.startswith("J")]
    assert len(jobs) == 10
    assert (jobs["top5_hits"] <= np.minimum(5, jobs["n_relevant"])).all()
    for c in ("precision_at_5", "recall_at_5", "ndcg_at_5"):
        assert jobs[c].between(0, 1).all()
    assert jobs["n_relevant"].sum() == int((result["labels"]["shortlist"] == "Y").sum()) == 95


def test_real_macro_and_micro_metrics_are_recomputed_from_job_pools(result):
    pool = result["pool"]
    k = 5
    hits = slots = relevant = 0
    expected_hits = 0.0
    job_recalls = []
    job_ndcgs = []
    dcg_sum = idcg_sum = 0.0
    for _, g in pool.sort_values(["job_id", "pool_rank"]).groupby("job_id"):
        rel = g["relevant"].tolist()
        grades = g["rel_grade"].tolist()
        job_hits = top_k_hits(rel, k)
        job_slots = min(k, len(g))
        job_relevant = int(sum(rel))
        hits += job_hits
        slots += job_slots
        relevant += job_relevant
        expected_hits += job_slots * job_relevant / len(g)
        job_recalls.append(job_hits / job_relevant)
        job_ndcgs.append(ndcg_at_k(grades, k))
        dcg_sum += dcg_at_k(grades, k)
        idcg_sum += dcg_at_k(sorted(grades, reverse=True), k)
    micro = evaluate_micro(pool, k)
    macro = result["quality"].iloc[-1]
    assert result["summary"]["counts"]["labelled_pairs_evaluated"] == len(pool) == 555
    assert len(result["quality"]) - 2 == pool["job_id"].nunique() == 10
    assert micro["top_k_hits"] == hits
    assert micro["precision_at_k"] == pytest.approx(hits / slots)
    assert micro["recall_at_k"] == pytest.approx(hits / relevant)
    assert micro["random_expected_precision_at_k"] == pytest.approx(expected_hits / slots)
    assert micro["ndcg_at_k"] == pytest.approx(dcg_sum / idcg_sum)
    assert result["quality"].iloc[-2]["job_id"] == "ALL (micro avg; counts summed)"
    assert result["quality"].iloc[-2]["recall_at_5"] == pytest.approx(micro["recall_at_k"])
    assert result["quality"].iloc[-2]["ndcg_at_5"] == pytest.approx(micro["ndcg_at_k"])
    assert macro["recall_at_5"] == pytest.approx(np.mean(job_recalls))
    assert macro["ndcg_at_5"] == pytest.approx(np.mean(job_ndcgs))
    assert macro["recall_at_5"] != pytest.approx(micro["recall_at_k"])
    assert macro["ndcg_at_5"] != pytest.approx(micro["ndcg_at_k"])


def test_pool_contains_only_labelled_pairs(result):
    pool, labels = result["pool"], result["labels"]
    assert len(pool) == len(labels) == 555
    assert set(zip(pool.job_id, pool.resume_id)) == set(zip(labels.job_id, labels.resume_id))
    for _, g in pool.groupby("job_id"):
        assert sorted(g["pool_rank"]) == list(range(1, len(g) + 1))


def test_shortlist_equals_score_ge_4(result):
    assert result["summary"]["inspection"]["resume_job_labels"]["shortlist_equals_score_ge_4"] is True


def test_misrankings_independently_recomputed(result):
    pool = result["pool"].copy()
    R = pool.groupby("job_id")["relevant"].transform("sum")
    expected = int(((pool.relevant == 1) & (pool.pool_rank > R)).sum() + ((pool.relevant == 0) & (pool.pool_rank <= R)).sum())
    m = result["misrankings"]
    assert len(m) == expected
    assert (m[m.category == "missed_shortlist"].shape[0]) == (m[m.category == "false_shortlist"].shape[0])
    assert m["reason_for_misranking"].str.len().gt(20).all() and m["possible_improvement"].str.len().gt(20).all()


def test_at_least_five_genuine_misrankings_on_supplied_data(result):
    assert len(result["misrankings"]) >= 5


def test_five_documented_misrankings_have_source_support(result):
    examples = [
        ("J08", "RES-0103", "false_shortlist", "Tally Prime Certified", "Tally <-"),
        ("J09", "RES-0012", "missed_shortlist", "qualified leads", "Lead Generation"),
        ("J02", "RES-0119", "false_shortlist", "Partial match", "Overlap"),
        ("J09", "RES-0065", "false_shortlist", "Partial match", "Overlap"),
        ("J04", "RES-0130", "missed_shortlist", "structured skills", "Python; scikit-learn; SQL"),
    ]
    mis = result["misrankings"].set_index(["job_id", "resume_id"])
    labels = result["labels"].set_index(["job_id", "resume_id"])
    resumes = result["resumes"].set_index("resume_id")
    for job_id, resume_id, category, source_evidence, explanation_evidence in examples:
        row = mis.loc[(job_id, resume_id)]
        label = labels.loc[(job_id, resume_id)]
        text = str(resumes.loc[resume_id, "resume_text"])
        assert row["category"] == category
        if source_evidence == "structured skills":
            structured_skills = str(resumes.loc[resume_id, "skills"])
            assert all(skill in structured_skills for skill in ("Python", "scikit-learn", "SQL"))
        else:
            assert source_evidence.lower() in text.lower() or source_evidence.lower() in str(label["reviewer_note"]).lower()
        assert any(
            explanation_evidence.lower() in str(row[field]).lower()
            for field in ("reason_for_misranking", "extraction_gap_skills", "credited_beyond_profile")
        )
        assert int(row["human_suitability_score"]) == int(label["human_suitability_score"])
        if category == "missed_shortlist":
            assert label["shortlist"] == "Y" and row["predicted_rank"] > row["size_matched_cutoff_R"]
        else:
            assert label["shortlist"] == "N" and row["predicted_rank"] <= row["size_matched_cutoff_R"]


def test_tie_break_does_not_change_metrics(result):
    t = result["tie_sensitivity"]
    assert t["mean_ndcg_at_5"].nunique() == 1 and t["top5_hits_total"].nunique() == 1


def test_extraction_quality(result):
    s = result["extraction_summary"].set_index("layout_difficulty")
    assert s.loc["ALL", "experience_match_rate"] == 1.0
    assert s.loc["ALL", "skill_recall_micro"] > 0.95
    assert s.loc["Clean", "skill_recall_micro"] == 1.0


# ---------------------------------------------------------------- reports
EXPECTED = ["ranked_candidates.csv", "top5_per_job.csv", "ranking_quality.csv", "ranking_quality_ablation.csv", "tie_sensitivity.csv",
            "misranking_analysis.csv", "fairness_audit.csv", "extraction_validation.csv", "dataset_inspection.json", "run_summary.json"]


def test_reports_saved_and_readable(result, reports_dir):
    for name in EXPECTED:
        p = reports_dir / name
        assert p.exists() and p.stat().st_size > 0, name
    rc = pd.read_csv(reports_dir / "ranked_candidates.csv")
    assert len(rc) == 1850 and {"eval_only_human_suitability_score", "eval_only_shortlist"} <= set(rc.columns)
    assert rc["eval_only_shortlist"].notna().sum() == 555                      # unlabeled pairs stay blank, not 'N'
    quality = pd.read_csv(reports_dir / "ranking_quality.csv")
    assert len(quality) == 12
    assert set(quality["job_id"].tail(2)) == {
        "ALL (micro avg; counts summed)", "ALL (macro avg; counts summed)"
    }
    assert len(pd.read_csv(reports_dir / "misranking_analysis.csv")) == len(result["misrankings"])
    assert len(pd.read_csv(reports_dir / "top5_per_job.csv")) == 50
