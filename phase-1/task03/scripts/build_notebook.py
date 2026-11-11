"""Generates notebooks/02_baselines.ipynb (run once; the notebook is the deliverable)."""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Task 3: Train Your First Honest Baselines
**CortexHR AI/ML: Employee Attrition Prediction (Day 3)**

**Objective.** Train a model that always guesses the majority (`DummyClassifier`), then a `LogisticRegression` with balanced class weights. Score both with the *same* stratified 5-fold cross-validation, and compare them with the `baseline_benchmarks` sheet. An honest baseline tells us how hard the problem is: any fancier model has to beat it fairly.

**Target:** `left_within_12m` (1 = left within 12 months, 0 = stayed).
""")

md("## 1. Import libraries")
code(r'''
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

# Make the project's src/ importable without absolute paths (works from notebooks/ or the root).
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "cortexhr").exists())
sys.path.insert(0, str(ROOT / "src"))

from cortexhr import baselines as bl
from cortexhr import features as ft
from cortexhr.paths import BENCHMARK_SHEET, DATA_RAW, DATA_SHEET, REPORTS, WORKBOOK_NAME

REPORTS.mkdir(exist_ok=True)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
print("project root:", ROOT.name, "| random state:", bl.RANDOM_STATE)
''')

md("## 2. Load the dataset")
code(r'''
workbook_path = DATA_RAW / WORKBOOK_NAME
sheets = pd.ExcelFile(workbook_path).sheet_names
USING_SYNTHETIC = "README" in sheets          # the stand-in workbook carries a README sheet
df = pd.read_excel(workbook_path, sheet_name=DATA_SHEET)

if USING_SYNTHETIC:
    print("WARNING: this workbook is the SYNTHETIC STAND-IN, not your real CortexHR data.")
    print("         Replace data/raw/cortexhr_dataset.xlsx with your real file and re-run.\n")
print("sheets:", sheets)
print("shape :", df.shape)
display(df.head())
display(df.isna().sum()[df.isna().sum() > 0].rename("missing values"))
''')

md("## 3. Inspect the target distribution")
code(r'''
counts = df[ft.TARGET].value_counts().sort_index()
leaver_rate = df[ft.TARGET].mean()
display(pd.DataFrame({"rows": counts, "share": (counts / len(df)).round(4)}))
print(f"Leaver rate = {leaver_rate:.1%}. Only about 1 in {round(1/leaver_rate)} employees leaves, "
      "so the classes are imbalanced.")
''')

md("## 4. Load and inspect the benchmark values")
code(r'''
bench_raw = pd.read_excel(workbook_path, sheet_name=BENCHMARK_SHEET)
display(bench_raw)

def benchmark_for(keyword: str) -> float:
    """Pick the ROC-AUC benchmark for the model whose name contains `keyword`."""
    name_col = next(c for c in bench_raw.columns if "model" in c.lower())
    auc_col = next(c for c in bench_raw.columns if "roc" in c.lower())
    rows = bench_raw[bench_raw[name_col].astype(str).str.lower().str.contains(keyword)]
    if rows.empty:
        return float("nan")
    return float(rows[auc_col].iloc[0])

BENCH_DUMMY = benchmark_for("dummy")
BENCH_LR = benchmark_for("logistic")
print(f"Benchmark ROC-AUC -> Dummy: {BENCH_DUMMY}, Logistic regression: {BENCH_LR}")
if USING_SYNTHETIC:
    print("NOTE: these are PLACEHOLDER benchmark values (0.71 comes from the task text).")
''')

md(r"""
## 5. Reuse the Task 2 feature pipeline
The pipeline (impute, scale, one-hot) lives in `src/cortexhr/features.py` and is reused, not re-written. Some raw columns are **deliberately excluded**:
""")
code(r'''
display(pd.DataFrame({"column": list(ft.EXCLUDED_COLUMNS), "why excluded": list(ft.EXCLUDED_COLUMNS.values())}))
print("numeric inputs    :", ft.NUMERIC_COLUMNS)
print("categorical inputs:", ft.CATEGORICAL_COLUMNS)
''')

md("## 6. Define X and y")
code(r'''
X, y = ft.select_features(df)
assert ft.TARGET not in X.columns
assert not set(ft.EXCLUDED_COLUMNS) & set(X.columns)
assert set(y.unique()) <= {0, 1}
print("X:", X.shape, "| y:", y.shape, "| positive class column index:", bl.positive_column(y),
      "(class 1 = left_within_12m)")
''')

md(r"""
## 7. Stratified 5-fold cross-validation (one set of folds for every model)
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. The fold list is built **once** and reused, so every model is scored on identical rows.
""")
code(r'''
folds = bl.make_folds(X, y)
fold_table = pd.DataFrame({
    "fold": range(1, bl.N_SPLITS + 1),
    "train_rows": [len(tr) for tr, _ in folds],
    "test_rows": [len(te) for _, te in folds],
    "leavers_in_test": [int(y.iloc[te].sum()) for _, te in folds],
    "leaver_share_test": [round(y.iloc[te].mean(), 4) for _, te in folds],
}).set_index("fold")
display(fold_table)
# Every row is tested exactly once:
assert sorted(np.concatenate([te for _, te in folds])) == list(range(len(X)))
''')

md(r"""
## 8. Model 1: DummyClassifier (`strategy="most_frequent"`)
It ignores the inputs and always predicts the most common class ("stays"). Because every row gets the same score it cannot rank anyone, so ROC-AUC is 0.5. It still gets high *accuracy* because most people stay. That is exactly why accuracy alone is misleading here.
""")
code(r'''
dummy_model = ft.make_model(DummyClassifier(strategy="most_frequent"))
dummy = bl.evaluate("DummyClassifier", dummy_model, X, y, folds)
cm_dummy = confusion_matrix(y, dummy.pred)
print("distinct scores given by the dummy:", np.unique(dummy.proba))
print("people it flags as leavers        :", int(dummy.pred.sum()))
print(f"Dummy confusion matrix:\n{cm_dummy}")
display(pd.Series(dummy.pooled, name="Dummy, pooled out-of-fold").round(4).to_frame())
''')

md(r"""
## 9. Model 2: Logistic Regression (`class_weight="balanced"`)
`class_weight="balanced"` makes a mistake on a leaver count more than a mistake on a stayer (weights are inversely proportional to class frequency). Without it, the model can score well by almost always predicting "stays". The model sits inside **one Pipeline** with the Task 2 preprocessing, so imputing, scaling and encoding are re-fitted on each fold's *training* rows only.
""")
code(r'''
logreg_model = ft.make_model(LogisticRegression(class_weight="balanced", max_iter=1000,
                                                random_state=bl.RANDOM_STATE))
logreg = bl.evaluate("LogisticRegression", logreg_model, X, y, folds)   # uses cross_val_predict twice

# Confirm which column of predict_proba is class 1 by fitting one fold and reading classes_.
fitted_fold0 = logreg_model.fit(X.iloc[folds[0][0]], y.iloc[folds[0][0]])
print("classes_ of fitted model:", list(fitted_fold0.classes_),
      "-> positive class is column", list(fitted_fold0.classes_).index(1))
assert list(fitted_fold0.classes_) == [0, 1] and bl.positive_column(y) == 1
assert np.all((logreg.proba >= 0) & (logreg.proba <= 1))
print("out-of-fold probabilities collected for", len(logreg.proba), "rows;",
      "mean P(left) =", round(logreg.proba.mean(), 3))
''')

md(r"""
## 10. Evaluation metrics (four different things, kept separate)
* **Mean CV ROC-AUC**: ROC-AUC computed inside each fold, then averaged (± standard deviation).
* **Pooled out-of-fold ROC-AUC**: all out-of-fold probabilities combined, one ROC-AUC.
* **Mean fold-level** accuracy, precision, recall, F1: averaged over the 5 folds.
* **Pooled** accuracy, precision, recall, F1: computed once from all out-of-fold class predictions.

ROC-AUC always uses *probabilities*. The other four use the hard 0/1 predictions (cut-off 0.5).
""")
code(r'''
metric_names = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]

def fold_level(ev):
    out = pd.DataFrame({"mean over folds": ev.fold_mean[metric_names],
                        "SD over folds": ev.fold_std[metric_names]})
    return out.round(4)

print("== DummyClassifier: fold-level (mean, SD) ==");       display(fold_level(dummy))
print("== LogisticRegression: fold-level (mean, SD) ==");    display(fold_level(logreg))
print("== LogisticRegression: ROC-AUC for each fold =="); display(logreg.per_fold["roc_auc"].round(4).to_frame().T)

pooled_table = pd.DataFrame({"DummyClassifier": dummy.pooled, "LogisticRegression": logreg.pooled}).loc[metric_names]
print("== Pooled out-of-fold metrics (different from the fold means above) ==")
display(pooled_table.round(4))
''')

md("## 11. Comparison table against `baseline_benchmarks`")
code(r'''
def row(ev, bench):
    return {
        "Model": ev.name,
        "Accuracy": ev.pooled["accuracy"],
        "Precision": ev.pooled["precision"],
        "Recall": ev.pooled["recall"],
        "F1-score": ev.pooled["f1"],
        "ROC-AUC": ev.pooled["roc_auc"],
        "Average Precision": ev.pooled["pr_auc"],
        "ROC-AUC (mean CV)": ev.fold_mean["roc_auc"],
        "ROC-AUC (std CV)": ev.fold_std["roc_auc"],
        "Benchmark ROC-AUC": bench,
        "Difference (mean CV - benchmark)": ev.fold_mean["roc_auc"] - bench,
        "Evaluated Rows": len(y),
    }

comparison = pd.DataFrame([row(dummy, BENCH_DUMMY), row(logreg, BENCH_LR)]).set_index("Model")
display(comparison.round(4))
comparison.round(4).to_csv(REPORTS / "task3_baseline_comparison.csv")
''')

md("## 12. Confusion matrices and classification report (Out-of-fold)")
code(r'''
cm = confusion_matrix(y, logreg.pred)
tn, fp, fn, tp = cm.ravel()
print(f"Logistic Regression -> True stayers kept: {tn} | stayers wrongly flagged: {fp} | leavers missed: {fn} | leavers caught: {tp}")
print(f"Dummy Classifier    -> True stayers kept: {cm_dummy[0,0]} | stayers wrongly flagged: {cm_dummy[0,1]} | leavers missed: {cm_dummy[1,0]} | leavers caught: {cm_dummy[1,1]}")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# DummyClassifier (row 0)
ConfusionMatrixDisplay(cm_dummy, display_labels=["stayed", "left"]).plot(ax=axes[0, 0], colorbar=False)
axes[0, 0].set_title("DummyClassifier: Counts")
ConfusionMatrixDisplay.from_predictions(y, dummy.pred, normalize="true",
                                        display_labels=["stayed", "left"], ax=axes[0, 1],
                                        colorbar=False, values_format=".2f")
axes[0, 1].set_title("DummyClassifier: Share of true class")

# LogisticRegression (row 1)
ConfusionMatrixDisplay(cm, display_labels=["stayed", "left"]).plot(ax=axes[1, 0], colorbar=False)
axes[1, 0].set_title("LogisticRegression: Counts")
ConfusionMatrixDisplay.from_predictions(y, logreg.pred, normalize="true",
                                        display_labels=["stayed", "left"], ax=axes[1, 1],
                                        colorbar=False, values_format=".2f")
axes[1, 1].set_title("LogisticRegression: Share of true class")

fig.suptitle("Out-of-Fold Confusion Matrices (Stratified 5-Fold CV)")
fig.tight_layout()
fig.savefig(REPORTS / "task3_confusion_matrix.png", dpi=150)
plt.show()

report_text = classification_report(y, logreg.pred, target_names=["stayed", "left"], digits=3)
print("=== Logistic Regression Classification Report ===")
print(report_text)
_ = (REPORTS / "task3_classification_report.txt").write_text(report_text)
''')

md(r"""
## 13. Investigate leakage risks
The task rule: **a cross-validated ROC-AUC above 0.90 means stop and look for leakage.** Even below 0.90 we run sensible checks, and we never silently delete features to make a score look right.
""")
code(r'''
leak = {}

# 1. Alarm on the honest scores
for ev in (dummy, logreg):
    bl.leakage_alarm(ev.pooled["roc_auc"], ev.name)
print(f"1. Alarm: no honest score above {bl.LEAKAGE_ALARM} (logistic regression pooled = {logreg.pooled['roc_auc']:.3f}).")

# 2. Target and excluded columns are not inputs
assert ft.TARGET not in X.columns and not set(ft.EXCLUDED_COLUMNS) & set(X.columns)
print("2. Target and excluded columns are not in X.")

# 3. Duplicates and identifiers that could cross folds
id_col = "employee_id" if "employee_id" in df.columns else None
dup_rows = int(df.duplicated().sum())
dup_ids = int(df[id_col].duplicated().sum()) if id_col else None
dup_features = int(X.duplicated().sum())
print(f"3. Fully duplicated rows: {dup_rows} | duplicated employee ids: {dup_ids} | "
      f"rows with identical feature values: {dup_features}")

# 4. Control experiment: shuffle the labels. A leak-free setup must drop to chance (~0.5).
y_shuffled = pd.Series(np.random.default_rng(bl.RANDOM_STATE).permutation(y.values), index=y.index)
shuffled_folds = bl.make_folds(X, y_shuffled)
shuffled_auc = roc_auc_score(y_shuffled, bl.out_of_fold_proba(logreg_model, X, y_shuffled, shuffled_folds))
leak["shuffled_auc"] = shuffled_auc
print(f"4. Shuffled-label control ROC-AUC = {shuffled_auc:.3f} (chance is 0.5).")

# 5. Preprocessing is fitted inside each fold: compare with a manual per-fold loop.
manual = np.zeros(len(X))
for tr, te in folds:
    pre = ft.build_feature_pipeline().fit(X.iloc[tr])              # fitted on TRAIN rows only
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=bl.RANDOM_STATE)
    clf.fit(pre.transform(X.iloc[tr]), y.iloc[tr])
    manual[te] = clf.predict_proba(pre.transform(X.iloc[te]))[:, 1]
assert np.allclose(manual, logreg.proba, atol=1e-9)
print("5. Pipeline results equal a manual loop that fits preprocessing on training folds only.")

# 6. One column at a time: does any single raw column almost give the answer away?
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def single_column_auc(col):
    numeric = pd.api.types.is_numeric_dtype(df[col])
    steps = ([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())] if numeric else
             [("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    model = Pipeline([("prep", ColumnTransformer([("c", Pipeline(steps), [col])])),
                      ("clf", LogisticRegression(max_iter=1000))])
    return roc_auc_score(y, bl.out_of_fold_proba(model, df[[col]], y, folds))

all_columns = [c for c in df.columns if c != ft.TARGET]
scan = pd.DataFrame({
    "single-column CV ROC-AUC": {c: single_column_auc(c) for c in all_columns},
    "used as input": {c: c in X.columns for c in all_columns},
}).sort_values("single-column CV ROC-AUC", ascending=False)
display(scan.round(3))
leak["max_single_approved"] = scan.loc[scan["used as input"], "single-column CV ROC-AUC"].max()
suspicious = scan[(scan["single-column CV ROC-AUC"] > 0.80)].index.tolist()
print("6. Columns that score above 0.80 on their own:", suspicious or "none")
''')

code(r'''
# 7. Demonstration: what happens if a post-outcome column slips in? (Only runs if the column exists.)
excluded_present = [c for c in ft.EXCLUDED_COLUMNS if c in df.columns and c != "employee_id"]
leak["demo_auc"] = None
if excluded_present:
    X_leaky, _ = ft.select_features(df, extra_columns=excluded_present)
    leaky_model = ft.make_model(LogisticRegression(class_weight="balanced", max_iter=1000,
                                                   random_state=bl.RANDOM_STATE),
                                extra_categorical=excluded_present)
    leaky_auc = roc_auc_score(y, bl.out_of_fold_proba(leaky_model, X_leaky, y, folds))
    leak["demo_auc"] = leaky_auc
    print(f"7. With {excluded_present} ADDED, ROC-AUC jumps to {leaky_auc:.3f}.")
    try:
        bl.leakage_alarm(leaky_auc, "model with post-outcome column")
    except RuntimeError as err:
        print("   Alarm fired as intended ->", err)
    print("   That is why those columns stay out. The honest score is the one in the comparison table.")
else:
    print("7. No excluded post-outcome column found in this data; demo skipped.")
''')

md(r"""
**Leakage verdict.** No leakage was found in the approved feature set: scores are far below 0.90, shuffled labels give chance-level scores, preprocessing runs inside the folds, and no single approved column is a giveaway. The one dangerous column is excluded **with a written reason**, and the demonstration shows the alarm catches it.
""")

md(r"""
## 14. Optional experiment: Decision Tree (kept separate from the two baselines)
**Question:** why might a tree look excellent on its training data but weaker in cross-validation?

An unlimited tree keeps splitting until every training row sits in a pure leaf, so it can reach a training ROC-AUC of 1.0 by **memorising** noise and quirks. Those quirks do not repeat in new employees. This is **overfitting**. Limiting the depth forces the tree to learn only broad patterns.
""")
code(r'''
def tree_model(depth):
    return ft.make_model(DecisionTreeClassifier(max_depth=depth, random_state=bl.RANDOM_STATE))

depths = [1, 2, 3, 4, 6, 8, 12, None]
sweep = []
for d in depths:
    tv = bl.train_vs_validation(tree_model(d), X, y, folds)
    sweep.append({"max_depth": "unlimited" if d is None else d,
                  "TRAINING ROC-AUC": tv["train_roc_auc"].mean(),
                  "VALIDATION ROC-AUC (CV)": tv["validation_roc_auc"].mean()})
sweep = pd.DataFrame(sweep).set_index("max_depth")
sweep["gap"] = sweep["TRAINING ROC-AUC"] - sweep["VALIDATION ROC-AUC (CV)"]
display(sweep.round(3))

# Out-of-fold validation scores for the two trees we discuss (same folds as before)
tree_unlimited = bl.evaluate("Decision tree (no depth limit)", tree_model(None), X, y, folds)
tree_depth4 = bl.evaluate("Decision tree (depth 4)", tree_model(4), X, y, folds)
for ev in (tree_unlimited, tree_depth4):
    print(f"{ev.name}: validation ROC-AUC mean over folds {ev.fold_mean['roc_auc']:.3f}, "
          f"pooled OOF {ev.pooled['roc_auc']:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
labels = [str(i) for i in sweep.index]
ax.plot(labels, sweep["TRAINING ROC-AUC"], marker="o", label="training (rows the tree saw)")
ax.plot(labels, sweep["VALIDATION ROC-AUC (CV)"], marker="o", label="validation (cross-validation)")
ax.set_xlabel("max_depth"); ax.set_ylabel("ROC-AUC"); ax.legend(); ax.set_title("Tree depth and overfitting")
fig.tight_layout(); fig.savefig(REPORTS / "task3_tree_overfitting.png", dpi=150); plt.show()
''')

md("## 15. The five-sentence benchmark comparison note")
code(r'''
import re

lr_cv = logreg.fold_mean["roc_auc"]
lr_sd = logreg.fold_std["roc_auc"]
gap = lr_cv - BENCH_LR
direction = "above" if gap > 0 else "below"
within = "within" if abs(gap) <= lr_sd else "outside"
stand_in = " and the data here is a synthetic stand-in" if USING_SYNTHETIC else ""

sentences = [
    f"The DummyClassifier always predicts that employees stay, so it reaches {dummy.pooled['accuracy']:.1%} accuracy "
    f"only because {1 - leaver_rate:.1%} of employees stay, while it catches {dummy.pooled['recall']:.0%} of leavers "
    f"and scores a ROC-AUC of {dummy.fold_mean['roc_auc']:.2f}.",

    f"Logistic regression with balanced class weights reaches a mean cross-validated ROC-AUC of {lr_cv:.3f} "
    f"(pooled out-of-fold {logreg.pooled['roc_auc']:.3f}) against the approximate benchmark of {BENCH_LR:.2f}.",

    f"That is {abs(gap):.3f} {direction} the benchmark, which is {within} one fold-to-fold standard deviation of {lr_sd:.3f}.",

    f"The gap looks believable because the folds are stratified and shared by every model, preprocessing is fitted inside each "
    f"training fold, and a shuffled-label control scores {leak['shuffled_auc']:.2f}, but the benchmark's features, rows and "
    f"validation recipe are unknown{stand_in}, so the comparison is only approximate.",

    f"Before Task 4 we should note that at the 0.5 cut-off the model catches {logreg.pooled['recall']:.0%} of leavers with "
    f"only {logreg.pooled['precision']:.0%} precision, so Task 4 must choose metrics and risk bands rather than rely on that cut-off"
    f"{', and this notebook should be re-run on the real workbook' if USING_SYNTHETIC else ''}.",
]
assert len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", " ".join(sentences))) == 5, "note must be exactly five sentences"
note = " ".join(sentences)
for i, s in enumerate(sentences, 1):
    print(f"{i}. {s}\n")
''')

md("## 16. Final conclusion and saved findings")
code(r'''
summary_md = f"""# Task 3 findings: honest baselines

Data source: {"SYNTHETIC STAND-IN (replace with the real workbook and re-run)" if USING_SYNTHETIC else "real workbook"}.

## Comparison table
{comparison.round(4).to_markdown()}

## Tree overfitting (training vs cross-validation ROC-AUC)
{sweep.round(3).to_markdown()}

## Leakage checks
- Shuffled-label control ROC-AUC: {leak['shuffled_auc']:.3f}
- Highest single approved column ROC-AUC: {leak['max_single_approved']:.3f}
- Post-outcome column demo ROC-AUC: {"n/a" if leak['demo_auc'] is None else f"{leak['demo_auc']:.3f}"}
- Duplicate rows: {dup_rows}

## Five-sentence note
{note}
"""
_ = (REPORTS / "task3_findings.md").write_text(summary_md)

import json
evaluation_json = {
    "task": "Day 3, Task 3: Train Your First Honest Baselines",
    "dataset": {
        "is_synthetic": bool(USING_SYNTHETIC),
        "total_rows": int(len(df)),
        "evaluated_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "numeric_features": ft.NUMERIC_COLUMNS,
        "categorical_features": ft.CATEGORICAL_COLUMNS,
        "excluded_columns": ft.EXCLUDED_COLUMNS,
        "target": ft.TARGET,
        "target_counts": {str(k): int(v) for k, v in counts.items()},
        "leaver_rate": float(leaver_rate),
        "majority_class_share": float(1 - leaver_rate),
    },
    "cross_validation": {
        "strategy": "StratifiedKFold",
        "n_splits": int(bl.N_SPLITS),
        "random_state": int(bl.RANDOM_STATE),
        "shuffle": True,
        "same_folds_for_all_models": True,
    },
    "benchmarks": {
        "dummy_roc_auc": float(BENCH_DUMMY),
        "logistic_regression_roc_auc": float(BENCH_LR),
    },
    "models": {
        "DummyClassifier": {
            "strategy": "most_frequent",
            "pooled_metrics": {k: float(v) for k, v in dummy.pooled.items()},
            "mean_cv_metrics": {k: float(v) for k, v in dummy.fold_mean.items()},
            "std_cv_metrics": {k: float(v) for k, v in dummy.fold_std.items()},
            "confusion_matrix": cm_dummy.tolist(),
        },
        "LogisticRegression": {
            "class_weight": "balanced",
            "pooled_metrics": {k: float(v) for k, v in logreg.pooled.items()},
            "mean_cv_metrics": {k: float(v) for k, v in logreg.fold_mean.items()},
            "std_cv_metrics": {k: float(v) for k, v in logreg.fold_std.items()},
            "confusion_matrix": cm.tolist(),
            "comparison_with_benchmark": {
                "benchmark_roc_auc": float(BENCH_LR),
                "difference": float(gap),
                "direction": direction,
                "within_1sd": bool(abs(gap) <= lr_sd),
            },
        },
    },
    "tree_experiment": {
        str(depth): {
            "train_roc_auc": float(sweep.loc[depth, "TRAINING ROC-AUC"]),
            "cv_roc_auc": float(sweep.loc[depth, "VALIDATION ROC-AUC (CV)"]),
            "overfitting_gap": float(sweep.loc[depth, "gap"]),
        } for depth in sweep.index
    },
    "leakage_checks": {
        "shuffled_label_auc": float(leak["shuffled_auc"]),
        "highest_single_feature_auc": float(leak["max_single_approved"]),
        "post_outcome_demo_auc": None if leak["demo_auc"] is None else float(leak["demo_auc"]),
        "fully_duplicate_rows": int(dup_rows),
        "duplicate_employee_ids": None if dup_ids is None else int(dup_ids),
        "alarm_threshold": float(bl.LEAKAGE_ALARM),
        "alarm_triggered": False,
    },
    "five_sentence_note": note,
    "five_sentences": sentences,
}
_ = (REPORTS / "task3_evaluation_results.json").write_text(json.dumps(evaluation_json, indent=2))

print("Models trained: DummyClassifier, LogisticRegression (+ optional decision trees).")
print(f"Logistic regression mean CV ROC-AUC {lr_cv:.3f} vs benchmark {BENCH_LR:.2f} (gap {gap:+.3f}).")
print("Saved to reports/: comparison csv, classification report, confusion matrix, tree plot, findings.md, evaluation_results.json")

''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_baselines.ipynb"
nbf.write(nb, out)
print("wrote", out)
