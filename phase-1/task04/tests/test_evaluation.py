"""Tests for src/cortexhr/evaluation.py (Day 4, Task 4).

Mostly SYNTHETIC data with answers worked out by hand. ``TestRealDataset`` re-checks the headline numbers on
the actual workbook and is skipped if it is missing.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score

from cortexhr.evaluation import (BAND_LABELS, band_summary, binary_report, bootstrap_ci, calibration_table, expected_calibration_error,
                                 gains_curve, lift_table, risk_bands, top_k_mask)
from cortexhr.paths import WORKBOOK


@pytest.fixture
def toy():
    """SYNTHETIC: 10 people, 3 leavers, scores already ranked (the PR-AUC worked example in the notebook)."""
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    s = np.linspace(0.95, 0.05, 10)
    return y, s


class TestTopK:
    def test_exact_size_and_picks_the_highest(self):
        s = np.array([0.1, 0.9, 0.5, 0.7, 0.3, 0.2, 0.8, 0.4, 0.6, 0.05])
        m = top_k_mask(s, 0.3)
        assert m.sum() == 3 and set(np.flatnonzero(m)) == {1, 6, 3}

    def test_ties_follow_input_order_or_a_random_seed(self):
        s = np.zeros(10)
        assert list(np.flatnonzero(top_k_mask(s, 0.2))) == [0, 1]                        # input order
        a, b = top_k_mask(s, 0.2, tie_seed=1), top_k_mask(s, 0.2, tie_seed=1)
        assert np.array_equal(a, b) and a.sum() == 2                                      # reproducible
        assert not all(np.array_equal(top_k_mask(s, 0.2, tie_seed=i), a) for i in range(2, 10))

    def test_bad_fraction(self):
        with pytest.raises(ValueError):
            top_k_mask(np.arange(5), 0)


class TestBinaryReport:
    def test_hand_checked_values(self, toy):
        y, s = toy
        r = binary_report(y, s, top_frac=0.3, threshold=0.5)
        assert r["n"] == 10 and r["leavers"] == 3 and r["prevalence"] == pytest.approx(0.3)
        assert r["top_k_people"] == 3 and r["precision_at_top"] == pytest.approx(2 / 3)     # ranks 1-3 hold 2 leavers
        assert r["recall_at_top"] == pytest.approx(2 / 3) and r["lift_at_top"] == pytest.approx((2 / 3) / 0.3)
        assert r["roc_auc"] == pytest.approx(roc_auc_score(y, s)) and r["pr_auc"] == pytest.approx(average_precision_score(y, s))
        assert r["flagged_share"] == pytest.approx(0.5)                                      # scores >= 0.5: first five
        assert r["precision"] == pytest.approx(2 / 5) and r["recall"] == pytest.approx(2 / 3)
        assert r["brier_baseline"] == pytest.approx(0.3 * 0.7)

    def test_precision_and_recall_at_top_are_different_things(self, toy):
        """Top 2 of 10 holds 1 of the 3 leavers: precision 1/2, recall 1/3 (they are only equal when k equals the number of leavers)."""
        y, s = toy
        r = binary_report(y, s, top_frac=0.2)
        assert r["top_k_people"] == 2
        assert r["precision_at_top"] == pytest.approx(1 / 2) and r["recall_at_top"] == pytest.approx(1 / 3)
        assert r["lift_at_top"] == pytest.approx(0.5 / 0.3)

    def test_random_scores_give_chance_level_numbers(self):
        rng = np.random.default_rng(0)
        y = (rng.random(4000) < 0.15).astype(int)
        r = binary_report(y, rng.random(4000))
        assert abs(r["roc_auc"] - 0.5) < 0.04 and abs(r["lift_at_top"] - 1) < 0.25 and abs(r["pr_auc"] - y.mean()) < 0.04

    def test_perfect_scores(self):
        y = np.array([0] * 90 + [1] * 10)
        r = binary_report(y, y.astype(float))
        assert r["roc_auc"] == 1 and r["precision_at_top"] == 1 and r["recall_at_top"] == 1 and r["lift_at_top"] == pytest.approx(10)

    def test_brier_is_nan_for_scores_that_are_not_probabilities(self, toy):
        y, s = toy
        assert np.isnan(binary_report(y, s * 10)["brier"])

    @pytest.mark.parametrize("y,s,msg", [([0, 0, 0], [0.1, 0.2, 0.3], "both classes"), ([0, 1], [0.1], "same length"),
                                          ([0, 1, 2], [0.1, 0.2, 0.3], "both classes"), ([0, 1], [0.1, np.nan], "NaN")])
    def test_bad_inputs_are_refused(self, y, s, msg):
        with pytest.raises(ValueError, match=msg):
            binary_report(y, s)


class TestRiskBands:
    def test_sizes_are_exact_for_the_real_row_count(self):
        b = risk_bands(np.random.default_rng(0).random(2379))
        assert b.value_counts().to_dict() == {"Low": 1665, "Medium": 476, "High": 238}

    def test_highest_scores_are_high_and_bands_are_ordered(self):
        s = np.random.default_rng(1).random(100)
        b = risk_bands(s)
        assert b.value_counts().to_dict() == {"Low": 70, "Medium": 20, "High": 10}
        assert s[b == "High"].min() > s[b == "Medium"].max() and s[b == "Medium"].min() > s[b == "Low"].max()
        assert list(b.cat.categories) == list(BAND_LABELS) and b.cat.ordered

    def test_input_order_and_index_do_not_matter_and_are_preserved(self):
        s = pd.Series(np.random.default_rng(2).random(50), index=[f"p{i}" for i in range(50)])
        b = risk_bands(s.values, index=s.index)
        shuffled = s.sample(frac=1, random_state=0)
        b2 = risk_bands(shuffled.values, index=shuffled.index)
        assert (b2.reindex(b.index) == b).all() and list(b.index) == list(s.index)

    def test_rank_based_so_any_increasing_transform_gives_the_same_bands(self):
        s = np.random.default_rng(3).random(200)
        assert (risk_bands(s) == risk_bands(np.log(s) * 5 + 2)).all() and (risk_bands(s) == risk_bands(s ** 3)).all()

    def test_custom_fractions_and_bad_fractions(self):
        assert risk_bands(np.arange(100.0), high=0.05, medium=0.15).value_counts().to_dict() == {"Low": 80, "Medium": 15, "High": 5}
        for kw in ({"high": 0}, {"high": 0.6, "medium": 0.5}, {"medium": -0.1}):
            with pytest.raises(ValueError):
                risk_bands(np.arange(10.0), **kw)

    def test_summary_counts_add_up_and_shares_of_leavers_sum_to_one(self):
        rng = np.random.default_rng(4)
        s = rng.random(1000)
        y = (rng.random(1000) < 0.1 + 0.3 * s).astype(int)                 # higher score -> more leavers
        bs = band_summary(y, risk_bands(s))
        assert bs.people.sum() == 1000 and bs.leavers.sum() == y.sum() and bs.share_of_all_leavers.sum() == pytest.approx(1)
        assert list(bs.band) == ["High", "Medium", "Low"] and bs.leaver_rate.is_monotonic_decreasing


class TestCalibration:
    def test_matches_scikit_learn_quantile_curve(self):
        rng = np.random.default_rng(5)
        p = rng.random(2000)
        y = (rng.random(2000) < p).astype(int)
        mine = calibration_table(y, p, n_bins=10, strategy="quantile")
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        np.testing.assert_allclose(mine.mean_predicted, mean_pred, atol=0.02)
        np.testing.assert_allclose(mine.observed_rate, frac_pos, atol=0.02)

    def test_well_calibrated_scores_have_small_gap_and_inflated_scores_a_big_one(self):
        rng = np.random.default_rng(6)
        p = rng.random(6000) * 0.6
        y = (rng.random(6000) < p).astype(int)
        assert expected_calibration_error(calibration_table(y, p)) < 0.03
        assert expected_calibration_error(calibration_table(y, np.clip(p + 0.3, 0, 1))) > 0.2          # over-stated risk

    def test_people_are_counted_once(self):
        rng = np.random.default_rng(7)
        p = rng.random(500)
        y = (rng.random(500) < 0.3).astype(int)
        for strategy in ("quantile", "uniform"):
            t = calibration_table(y, p, strategy=strategy)
            assert t.people.sum() == 500 and t.leavers.sum() == y.sum()

    def test_refuses_scores_that_are_not_probabilities(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            calibration_table(np.array([0, 1, 0, 1]), np.array([0.1, 1.5, 0.2, 0.3]))


class TestLiftAndGains:
    def test_random_scores_have_lift_one_everywhere(self):
        rng = np.random.default_rng(8)
        y = (rng.random(5000) < 0.15).astype(int)
        lt = lift_table(y, rng.random(5000))
        assert lt.lift.between(0.7, 1.3).all() and lt.cumulative_share_of_leavers.iloc[-1] == pytest.approx(1)

    def test_perfect_ranking_puts_every_leaver_in_the_first_decile(self):
        y = np.array([1] * 10 + [0] * 90)
        lt = lift_table(y, y.astype(float))
        assert lt.leavers.iloc[0] == 10 and lt.lift.iloc[0] == pytest.approx(10) and lt.leavers.iloc[1:].sum() == 0

    def test_gains_curve_endpoints_and_monotone(self, toy):
        y, s = toy
        x, g = gains_curve(y, s)
        assert (x[0], g[0]) == (0, 0) and (x[-1], g[-1]) == (1, 1) and np.all(np.diff(g) >= 0)
        assert g[3] == pytest.approx(2 / 3)                    # after 3 people: 2 of the 3 leavers found

    def test_bootstrap_interval_contains_the_estimate_and_is_reproducible(self):
        rng = np.random.default_rng(9)
        s = rng.random(600)
        y = (rng.random(600) < 0.1 + 0.3 * s).astype(int)
        est = roc_auc_score(y, s)
        lo, hi = bootstrap_ci(roc_auc_score, y, s, n_boot=200)
        assert lo < est < hi and (lo, hi) == bootstrap_ci(roc_auc_score, y, s, n_boot=200)


@pytest.fixture(scope="module")
def scores():
    if not WORKBOOK.exists():
        pytest.skip("real workbook not present in data/raw/")
    from cortexhr.baselines import load_model_table, make_folds, make_models, model_pipeline, out_of_fold_proba
    from cortexhr.evaluation import calibrated_logreg
    X, y, source = load_model_table()
    folds = make_folds(y)
    raw = out_of_fold_proba(model_pipeline(make_models()["Logistic Regression (balanced)"]), X, y, folds)
    cal = out_of_fold_proba(model_pipeline(calibrated_logreg()), X, y, folds)
    return y, raw, cal, source


@pytest.mark.skipif(not WORKBOOK.exists(), reason="real workbook not present in data/raw/")
class TestRealDataset:
    def test_as_trained_scores_are_inflated_and_calibration_fixes_it(self, scores):
        y, raw, cal, _ = scores
        t_raw, t_cal = calibration_table(y, raw), calibration_table(y, cal)
        assert raw.mean() > 0.3 and expected_calibration_error(t_raw) > 0.15                 # not probabilities
        assert abs(cal.mean() - y.mean()) < 0.02 and expected_calibration_error(t_cal) < 0.05
        assert binary_report(y, raw)["brier"] > binary_report(y, raw)["brier_baseline"] > binary_report(y, cal)["brier"]
        assert abs(roc_auc_score(y, raw) - roc_auc_score(y, cal)) < 0.01                    # ranking is unchanged

    def test_bands_on_the_real_rows(self, scores):
        y, _, cal, _ = scores
        bands = risk_bands(cal, index=y.index)
        bs = band_summary(y, bands).set_index("band")
        assert bs.people.to_dict() == {"High": 238, "Medium": 476, "Low": 1665}
        assert bs.leaver_rate["High"] > bs.leaver_rate["Medium"] > bs.leaver_rate["Low"]
        assert bs.lift["High"] > 1.8 and 0.15 < bs.share_of_all_leavers["High"] < 0.3
        assert bs.share_of_all_leavers["Low"] > 0.3                    # the Low band still holds many leavers: the list misses people

    def test_top10_numbers_match_task3_ranking(self, scores):
        y, raw, _, _ = scores
        r = binary_report(y, raw)
        assert r["roc_auc"] == pytest.approx(0.7102530805820995) and r["top_k_people"] == 238
        assert r["lift_at_top"] > 2
