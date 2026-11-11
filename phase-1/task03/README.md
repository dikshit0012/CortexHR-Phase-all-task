# CortexHR: Employee Attrition (Tasks 1-3)

## Task 3: honest baselines
Trains `DummyClassifier` and `LogisticRegression(class_weight="balanced")` with the same
stratified 5-fold CV (`random_state=42`), compares them with `baseline_benchmarks`, and runs leakage checks.

### Data
Put your real workbook at `data/raw/cortexhr_dataset.xlsx` with sheets `employees` (target `left_within_12m`)
and `baseline_benchmarks`. If you do not have it yet, `python scripts/make_synthetic_data.py` creates a
**synthetic stand-in** (it has a `README` sheet, and the notebook prints a warning when it is used).
If your Task 2 pipeline differs, edit `src/cortexhr/features.py` (column lists, `EXCLUDED_COLUMNS`).

### Run
```bash
pip install -r requirements.txt
python scripts/run_notebook.py notebooks/02_baselines.ipynb   # about 30 s
pytest -q                                                     # 14 tests
docker compose run --rm baselines                             # same thing in Docker
```
Outputs go to `reports/`: comparison CSV, evaluation results JSON, classification report, confusion matrix, tree plot, findings.md.


### How to read the results
- **ROC-AUC** uses probabilities and measures ranking (0.5 = chance). **Dummy = 0.5** by definition.
- **Accuracy** is misleading here: always saying "stays" is about 85% accurate and catches nobody.
- **Mean CV** (per-fold, averaged) and **pooled out-of-fold** numbers are different things; both are shown.
- **Any honest score above 0.90 stops the run** (leakage alarm); investigate before trusting it.
- Training scores (the tree experiment) are never used as validation performance.
