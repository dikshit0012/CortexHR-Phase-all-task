"""Tests for src/cortexhr/baselines.py (Day 3, Task 3).

Most tests use SYNTHETIC numeric data, to check the evaluation logic itself.
``TestRealDataset`` runs the baselines on the actual workbook and is skipped if it is missing.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from cortexhr.baselines import (RANDOM_STATE, SuspiciouslyGoodScoreError, bootstrap_auc_interval, check_suspicious_scores,
                                load_benchmarks, load_model_table, make_folds, make_models, model_pipeline, out_of_fold_proba,
                                per_fold_scores, resubstitution_auc, score_predictions, summarise)
from cortexhr.paths import WORKBOOK


@pytest.fixture
def noise():
    """SYNTHETIC: 300 rows of pure noise with ~15% positives (no real signal exists)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=list("abcdef"))
    y = pd.Series((rng.random(300) < 0.15).astype(int))
    return X, y


@pytest.fixture
def signal():
    """SYNTHETIC: same shape but column 'a' really predicts y."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(scale=1.0, size=400) > 1.0).astype(int))
    return X, y


class TestFolds:
    def test_every_row_is_tested_exactly_once(self, noise):
        _, y = noise
        folds = make_folds(y)
        tested = np.concatenate([te for _, te in folds])
        assert len(folds) == 5 and sorted(tested) == list(range(len(y)))
        for tr, te in folds:
            assert not set(tr) & set(te)

    def test_folds_are_stratified(self, noise):
        _, y = noise
        shares = [y.iloc[te].mean() for _, te in make_folds(y)]
        assert max(shares) - min(shares) < 0.03          # each fold keeps ~the same leaver share
        assert abs(np.mean(shares) - y.mean()) < 0.01

    def test_same_random_state_gives_identical_folds_and_other_seed_differs(self, noise):
        _, y = noise
        a, b, c = make_folds(y), make_folds(y), make_folds(y, random_state=RANDOM_STATE + 1)
        assert all(np.array_equal(a[i][1], b[i][1]) for i in range(5))
        assert not all(np.array_equal(a[i][1], c[i][1]) for i in range(5))

    def test_two_models_scored_on_the_same_folds_see_the_same_test_rows(self, signal):
        """The same list object is handed to every model, so test rows cannot differ between models."""
        X, y = signal
        folds = make_folds(y)
        before = [te.copy() for _, te in folds]
        out_of_fold_proba(LogisticRegression(max_iter=500), X, y, folds)
        out_of_fold_proba(DecisionTreeClassifier(max_depth=3, random_state=0), X, y, folds)
        assert all(np.array_equal(before[i], folds[i][1]) for i in range(5))


class TestOutOfFoldPredictions:
    def test_one_probability_per_row_between_0_and_1(self, signal):
        X, y = signal
        p = out_of_fold_proba(LogisticRegression(max_iter=500), X, y, make_folds(y))
        assert p.shape == (len(y),) and p.min() >= 0 and p.max() <= 1

    def test_matches_a_manual_loop_that_fits_preprocessing_on_the_training_fold_only(self, signal):
        """Proves the scaler inside the estimator is re-fitted per fold (not on all rows)."""
        X, y = signal
        folds = make_folds(y)
        est = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
        got = out_of_fold_proba(est, X, y, folds)
        manual = np.zeros(len(y))
        for tr, te in folds:
            m = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]).fit(X.iloc[tr], y.iloc[tr])
            manual[te] = m.predict_proba(X.iloc[te])[:, 1]
        np.testing.assert_allclose(got, manual)

    def test_memorising_model_looks_perfect_on_training_rows_but_is_chance_out_of_fold(self, noise):
        X, y = noise
        knn = KNeighborsClassifier(n_neighbors=1)
        assert resubstitution_auc(KNeighborsClassifier(n_neighbors=1), X, y) > 0.99        # memorised
        oof = roc_auc_score(y, out_of_fold_proba(knn, X, y, make_folds(y)))
        assert 0.35 < oof < 0.65                                                           # noise stays noise

    def test_unlimited_tree_overfits_noise(self, noise):
        X, y = noise
        tree = DecisionTreeClassifier(random_state=0)
        assert resubstitution_auc(tree, X, y) > 0.99
        assert roc_auc_score(y, out_of_fold_proba(tree, X, y, make_folds(y))) < 0.65

    def test_shuffled_labels_give_chance_level(self, signal):
        X, y = signal
        ys = pd.Series(np.random.default_rng(5).permutation(y.values))
        auc = roc_auc_score(ys, out_of_fold_proba(LogisticRegression(max_iter=500), X, ys, make_folds(ys)))
        assert 0.40 < auc < 0.60

    def test_y_must_be_binary_zero_one(self, signal):
        X, y = signal
        with pytest.raises(ValueError, match="classes 0 and 1"):
            out_of_fold_proba(LogisticRegression(), X, y.replace({1: 2}), make_folds(y))


class TestDummy:
    def test_dummy_is_accurate_but_useless(self, noise):
        X, y = noise
        p = out_of_fold_proba(DummyClassifier(strategy="most_frequent"), X, y, make_folds(y))
        s = score_predictions(y, p)
        assert s["roc_auc"] == 0.5 and s["recall"] == 0 and s["precision"] == 0
        assert s["accuracy"] == pytest.approx(1 - y.mean())
        assert s["pr_auc"] == pytest.approx(y.mean())             # PR-AUC floor = share of leavers


class TestScoring:
    def test_score_predictions_hand_checked(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.6, 0.4, 0.9, 0.3])
        s = score_predictions(y, p)                  # yes/no at 0.5 -> [0,0,1,0,1,0]
        assert s["precision"] == pytest.approx(0.5) and s["recall"] == pytest.approx(0.5)
        assert s["accuracy"] == pytest.approx(4 / 6) and s["roc_auc"] == pytest.approx(6 / 8)   # 6 of the 8 leaver/stayer pairs are ranked correctly

    def test_threshold_changes_yes_no_scores_but_not_ranking(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.2, 0.4, 0.45, 0.9])
        low, high = score_predictions(y, p, threshold=0.3), score_predictions(y, p, threshold=0.8)
        assert low["recall"] == 1.0 and high["recall"] == 0.5
        assert low["roc_auc"] == high["roc_auc"]

    def test_per_fold_and_summary_shapes(self, signal):
        X, y = signal
        folds = make_folds(y)
        p = out_of_fold_proba(LogisticRegression(max_iter=500), X, y, folds)
        pf = per_fold_scores(y, p, folds)
        s = summarise(y, p, folds)
        assert pf.shape == (5, 6) and s["roc_auc_mean"] == pytest.approx(pf["roc_auc"].mean())
        assert s["roc_auc_sd"] == pytest.approx(pf["roc_auc"].std(ddof=1)) and "roc_auc_pooled" in s

    def test_bootstrap_interval_is_ordered_reproducible_and_contains_the_score(self, signal):
        X, y = signal
        p = out_of_fold_proba(LogisticRegression(max_iter=500), X, y, make_folds(y))
        lo, hi = bootstrap_auc_interval(y, p, n_boot=300)
        assert lo < roc_auc_score(y, p) < hi and (lo, hi) == bootstrap_auc_interval(y, p, n_boot=300)


class TestLeakageAlarm:
    def test_raises_for_suspiciously_high_scores_with_helpful_message(self):
        with pytest.raises(SuspiciouslyGoodScoreError, match="leakage") as err:
            check_suspicious_scores({"fine": 0.71, "LEAKY": 1.0})
        assert "LEAKY" in str(err.value) and "fine" not in str(err.value)

    def test_passes_for_honest_scores_and_respects_the_threshold(self):
        check_suspicious_scores({"a": 0.5, "b": 0.71, "c": 0.89})
        with pytest.raises(SuspiciouslyGoodScoreError):
            check_suspicious_scores({"a": 0.86}, threshold=0.85)


@pytest.fixture(scope="module")
def data():
    if not WORKBOOK.exists():
        pytest.skip("real workbook not present in data/raw/")
    return load_model_table()


@pytest.mark.skipif(not WORKBOOK.exists(), reason="real workbook not present in data/raw/")
class TestRealDataset:

    def test_inputs_and_target(self, data):
        X, y, source = data
        assert X.shape == (2379, 33) and len(y) == len(source) == 2379
        assert "left_within_12m" not in X.columns and set(y.unique()) == {0, 1}
        assert int(y.sum()) == 349

    def test_baselines_score_like_honest_models_not_like_leaks(self, data):
        X, y, _ = data
        folds = make_folds(y)
        models = make_models()
        auc = {}
        for name in ["Dummy (always stays)", "Logistic Regression (balanced)", "Decision Tree (depth 4, balanced)"]:
            auc[name] = roc_auc_score(y, out_of_fold_proba(model_pipeline(models[name]), X, y, folds))
        assert auc["Dummy (always stays)"] == 0.5
        assert 0.66 < auc["Logistic Regression (balanced)"] < 0.76          # near the benchmark's 0.713
        assert auc["Logistic Regression (balanced)"] > auc["Decision Tree (depth 4, balanced)"] > 0.55
        check_suspicious_scores(auc)                                        # nothing above 0.9

    def test_shuffled_labels_on_real_features_give_chance(self, data):
        X, y, _ = data
        ys = pd.Series(np.random.default_rng(0).permutation(y.values), index=y.index)
        auc = roc_auc_score(ys, out_of_fold_proba(model_pipeline(make_models()["Logistic Regression (balanced)"]), X, ys, make_folds(ys)))
        assert 0.42 < auc < 0.58

    def test_benchmark_sheet_loads_with_expected_rows(self):
        b = load_benchmarks()
        assert len(b) == 5 and {"model", "roc_auc_mean", "roc_auc_sd", "pr_auc_mean", "recall_mean"} <= set(b.columns)
        assert b.loc[b.model.str.startswith("Logistic Regression"), "roc_auc_mean"].iloc[0] == pytest.approx(0.713)
        assert b.loc[b.model.str.startswith("LEAKY"), "roc_auc_mean"].iloc[0] == 1.0
