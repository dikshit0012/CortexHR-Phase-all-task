import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from completeness_guard import MONTHLY_DERIVED, check_batch, check_record

FEATS = ["a", "b", "c", *MONTHLY_DERIVED]
FULL = {f: 1.0 for f in FEATS}


def test_complete_record_passes():
    r = check_record(FULL, FEATS)
    assert r["status"] == "complete" and r["missing"] == [] and r["missing_share"] == 0.0


def test_any_missing_monthly_feature_needs_review():
    for f in MONTHLY_DERIVED:
        rec = {**FULL}; rec.pop(f)
        r = check_record(rec, FEATS)
        assert r["status"] == "review" and r["missing_critical"] == [f]


def test_nan_none_and_blank_count_as_missing():
    rec = {**FULL, "a": float("nan"), "b": None, "c": "  "}
    assert set(check_record(rec, FEATS)["missing"]) == {"a", "b", "c"}


def test_too_many_missing_noncritical_needs_review():
    feats = [f"x{i}" for i in range(10)] + list(MONTHLY_DERIVED)
    rec = {f: 1 for f in feats}; rec["x0"] = None; rec["x1"] = None
    assert check_record(rec, feats)["status"] == "review"      # 2/13 = 15% > 10%
    rec["x1"] = 1
    assert check_record(rec, feats)["status"] == "complete"    # 1/13 = 7.7%


def test_batch_coverage_and_no_mutation():
    recs = [dict(FULL), {k: v for k, v in FULL.items() if k != MONTHLY_DERIVED[0]}]
    before = [dict(r) for r in recs]
    out = check_batch(recs, FEATS)
    assert out["n_complete"] == 1 and out["n_review"] == 1 and out["coverage"] == 0.5
    assert recs == before


def test_empty_batch():
    assert check_batch([], FEATS)["coverage"] == 0.0
