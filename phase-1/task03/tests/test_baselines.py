import sys
from pathlib import Path

# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cortexhr import baselines as bl  # noqa: E402
from cortexhr import features as ft  # noqa: E402
from make_synthetic_data import build_employees  # noqa: E402


@pytest.fixture(scope="module")
def data():
    df = build_employees()
    X, y = ft.select_features(df)
    return df, X, y, bl.make_folds(X, y)


def lr():
    return ft.make_model(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))


def test_target_and_excluded_columns_not_in_X(data):
    _, X, _, _ = data
    assert ft.TARGET not in X.columns
    assert not set(ft.EXCLUDED_COLUMNS) & set(X.columns)


def test_folds_are_a_stratified_partition(data):
    _, X, y, folds = data
    assert len(folds) == 5
    tests = np.concatenate([te for _, te in folds])
    assert sorted(tests) == list(range(len(X)))
    shares = [y.iloc[te].mean() for _, te in folds]
    assert max(shares) - min(shares) < 0.01


def test_folds_are_reproducible(data):
    _, X, y, folds = data
    again = bl.make_folds(X, y)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again))


def test_positive_column_is_class_one():
    assert bl.positive_column(pd.Series([0, 1, 0, 1])) == 1
    with pytest.raises(ValueError):
        bl.positive_column(pd.Series([0, 0, 0]))


def test_dummy_has_chance_auc_and_zero_recall(data):
    _, X, y, folds = data
    ev = bl.evaluate("dummy", ft.make_model(DummyClassifier(strategy="most_frequent")), X, y, folds)
    assert ev.pooled["roc_auc"] == pytest.approx(0.5)
    assert ev.pooled["recall"] == 0 and ev.pooled["precision"] == 0
    assert ev.pooled["accuracy"] == pytest.approx(1 - y.mean(), abs=1e-3)


def test_logreg_beats_chance_and_proba_is_valid(data):
    _, X, y, folds = data
    ev = bl.evaluate("lr", lr(), X, y, folds)
    assert 0.6 < ev.pooled["roc_auc"] < bl.LEAKAGE_ALARM
    assert np.all((ev.proba >= 0) & (ev.proba <= 1))


def test_pipeline_matches_manual_per_fold_loop(data):
    _, X, y, folds = data
    proba = bl.out_of_fold_proba(lr(), X, y, folds)
    manual = np.zeros(len(X))
    for tr, te in folds:
        pre = ft.build_feature_pipeline().fit(X.iloc[tr])
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(pre.transform(X.iloc[tr]), y.iloc[tr])
        manual[te] = clf.predict_proba(pre.transform(X.iloc[te]))[:, 1]
    assert np.allclose(proba, manual)


def test_leakage_alarm_direction():
    bl.leakage_alarm(0.71)
    with pytest.raises(RuntimeError):
        bl.leakage_alarm(0.95)


def test_post_outcome_column_triggers_alarm(data):
    df, _, y, folds = data
    X_leaky, _ = ft.select_features(df, extra_columns=["exit_interview_completed"])
    model = ft.make_model(LogisticRegression(max_iter=1000), extra_categorical=["exit_interview_completed"])
    ev = bl.evaluate("leaky", model, X_leaky, y, folds)
    with pytest.raises(RuntimeError):
        bl.leakage_alarm(ev.pooled["roc_auc"])


def test_unlimited_tree_overfits(data):
    from sklearn.tree import DecisionTreeClassifier
    _, X, y, folds = data
    tv = bl.train_vs_validation(ft.make_model(DecisionTreeClassifier(random_state=42)), X, y, folds)
    assert tv["train_roc_auc"].mean() > 0.99
    assert tv["validation_roc_auc"].mean() < 0.65


def test_missing_target_explicitly_handled():
    df = build_employees().copy()
    df.loc[0:4, ft.TARGET] = np.nan
    with pytest.warns(UserWarning, match="Dropped 5 row"):
        X, y = ft.select_features(df)
    assert len(X) == len(df) - 5
    assert len(y) == len(df) - 5
    assert set(y.unique()) <= {0, 1}


def test_invalid_target_values_raise_error():
    df = build_employees().copy()
    df.loc[0, ft.TARGET] = 99
    with pytest.raises(ValueError, match="Target .* must contain valid binary"):
        ft.select_features(df)


def test_comparison_table_columns_and_rows():
    import pandas as pd
    from cortexhr.paths import REPORTS
    csv_path = REPORTS / "task3_baseline_comparison.csv"
    if csv_path.exists():
        df_comp = pd.read_csv(csv_path)
        required_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Average Precision"]
        for col in required_cols:
            assert col in df_comp.columns, f"Missing required column: {col}"
        assert len(df_comp) >= 2


def test_evaluation_results_json_schema():
    import json
    # pyrefly: ignore[missing-import]
    from cortexhr.paths import REPORTS
    json_path = REPORTS / "task3_evaluation_results.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        assert "dataset" in data
        assert "cross_validation" in data
        assert "models" in data
        assert "five_sentence_note" in data
        assert data["cross_validation"]["n_splits"] == 5

