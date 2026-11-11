# CortexHR AI/ML — Employee Attrition (Task 1: Data Audit, Day 2: Feature Pipeline)

*Day 2 / Task 2 documentation starts at [Day 2 — Feature Pipeline](#day-2--feature-pipeline-task-2). Task 1 documentation is unchanged below.*

Profiles the master attrition table (`attrition_dataset`) and records, for every problem found, a written decision.
The decisions become the rules of the reusable pipeline in Task 2.

## Quick start (Docker)

```bash
# 1. put the master workbook in place (not committed to git)
cp 02_CortexHR_AI_ML_Dataset.xlsx data/raw/

# 2a. re-run the entire audit headlessly (writes figures, tables, findings.json, docs)
docker compose run --rm audit

# 2b. ...or explore interactively: http://localhost:8888
docker compose up jupyter

# 3. run the unit tests
docker compose run --rm test
```

Without Docker:

* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  $env:PYTHONPATH="src"
  python scripts/run_audit.py
  ```

* **Linux / macOS:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  PYTHONPATH=src python scripts/run_audit.py
  ```

## Layout

```
data/raw/            master workbook (read-only, never edited)
data/interim/        cleaned intermediate data   (Task 2)
data/processed/      model-ready feature tables  (Task 2)
notebooks/           01_data_audit.ipynb  <- the audit source (saved outputs cleared)
src/cortexhr/        reusable code: paths, audit helpers, data dictionary,
                     features.py (Day 2 pipeline), monthly.py (Day 2 attendance/payroll join)
scripts/             run_audit.py (Task 1 headless re-run), run_pipeline.py (Day 2)
tests/               unit tests: test_audit.py, test_features.py, test_monthly.py
reports/             generated locally by the audit and pipeline; not committed
docs/                data_dictionary.md, decisions.md (hand-off to Task 2), open_items.md (questions for the Python team + response log)
```

## Key results

| | |
|---|---|
| Rows / columns | 2,389 x 42 (383 master-snapshot employees + 2,000 simulated history rows + 6 duplicates) |
| Label | 14.6% left within 12 months (85.4% stayed) |
| Missing | Mostly *structural* (never promoted / no raise / no appraisal yet); only commute (55) and education (15) are random gaps |
| Impossible values | 4 negative tenures (not corrected, awaiting join dates), 2 attendance > 100%, 13 city spelling variants (+ 213 out-of-range WFH shares, 137 experience < tenure) |
| Outliers | IQR flags 1.6-7x more rows than z-score; 3 salaries are 10x typos |
| Leakage | 6 columns reveal the answer (model ROC-AUC 0.70 -> 1.00 when added) |
| Open items | 2 questions for the Python team (negative tenure join dates; `dept_attrition_rate_prior_12m` cut-off). Nothing assumed; record answers in `docs/open_items.md` |

## Notes

* **Wording:** the whole workbook is fictional (its README sheet says so). "Master-snapshot" rows (`EMP-…`, 383 people measured on 30 Sep 2025) means the rows from the master employee table, as opposed to the 2,000 simulated historical rows (`HIST-…`).
* Audit reports are computed by the notebook and written locally; generated outputs and workbook data are not included in this repository.
* `ydata-profiling` is used for the automatic comparison profile. Upstream has renamed it to `fg-data-profiling`; the old package still works but is no longer updated.
* The workbook includes an answer key sheet (`answer_key_planted_dirt`). Every check in the notebook is computed from the data; the key is used only to validate completeness in section 9. It is a validation, not a blind test, because the key was visible while exploring the file.

---

## Day 2 — Feature Pipeline (Task 2)

Full evidence (test output, numbers, open questions): [`reports/day2_feature_pipeline.md`](reports/day2_feature_pipeline.md).

### What the pipeline does

`build_pipeline()` in [`src/cortexhr/features.py`](src/cortexhr/features.py) turns raw employee rows into model-ready numbers.
It is a scikit-learn `Pipeline` with two steps: a `RawCleaner` that applies the Task 1 cleaning rules, then a `ColumnTransformer`
(numbers: median imputation + `StandardScaler`; text: imputation + `OneHotEncoder(handle_unknown="ignore")`).
The same fitted object is used for training, testing and scoring new employees, so there is no copy-pasted preprocessing.

```python
from cortexhr.features import build_pipeline, select_features

X = select_features(table)          # allowed columns only; the target is not in X
pipe = build_pipeline()             # include_unconfirmed=False, use_monthly=True, window_months=3
pipe.fit(X_train)                   # learn medians / scaling / categories from TRAIN only
X_train_p = pipe.transform(X_train)
X_test_p = pipe.transform(X_test)   # transform only, never fit again
```

Signature: `build_pipeline(include_unconfirmed=False, use_monthly=True, window_months=3, numeric_features=None, categorical_features=None) -> Pipeline`

### Safe-feature selection

The input columns are listed explicitly in `features.py` (33 by default) and checked against the data dictionary in a test, so a column cannot be added or forgotten silently.
`select_features()` and the pipeline itself raise `ForbiddenColumnError` (with the reason) if any of these appears as an input:

| Excluded | Why |
|---|---|
| `left_within_12m` | the target; kept in a separate `y` |
| `exit_date`, `resignation_date`, `exit_reason`, `exit_interview_done`, `fnf_settlement_amount`, `voluntary_left_within_12m` | only exist after the person has left (Task 1 leakage list, confirmed against the real columns) |
| `row_id`, `employee_id` | identifiers (`employee_id` is used for grouping/joining only) |
| `source` | says which simulation produced the row, not anything about the employee |
| `observation_date` | the prediction date. Used to build the monthly windows, not a feature (no other date column is left once the exit dates are removed) |
| `dept_attrition_rate_prior_12m` | **not confirmed safe**, so it is off by default (see limitations). `build_pipeline(include_unconfirmed=True)` switches it on for a with/without comparison |

Row-level Task 1 decisions are applied by `prepare_training_rows()` before splitting: the 6 duplicate rows are dropped and the 4 negative-tenure rows are left out of training (not corrected; open item O1).

### Attendance and payroll

`attendance_monthly_features` and `payroll_monthly_features` have one row per employee per month. [`src/cortexhr/monthly.py`](src/cortexhr/monthly.py) summarises them per attrition row and joins by `employee_id`:

* **Window:** the 3 most recent *completed* months on or before the row's `observation_date`. For the master snapshot (30 Sep 2025) that is July, August and September 2025. Later months are never used (the tables run to September 2026).
* **Duplicates:** the payroll table has 15 exact duplicate employee-month rows; they are removed. Rows that share an employee and month but disagree raise an error instead of being guessed.
* **No row multiplication:** the monthly rows are collapsed to one row per attrition row, joined with `validate="one_to_one"`, and the row count is checked (2,379 in, 2,379 out).
* **Missing months:** employees with fewer than 3 months use the months they have (the counts are in the diagnostics: 366 rows with 3, 10 with 2, 3 with 1). Employees with no monthly data (all 2,000 simulated `HIST-` rows) get NaN, **not zero**. The pipeline later fills those with the training median.
* Features used by the model: `attendance_rate_change_3m`, `avg_unpaid_absence_days_3m`, `avg_variable_pay_3m`.

### The new derived feature

`attendance_rate_change_3m` = attendance rate in the latest month of the window minus the rate in the earliest month. Negative means attendance is slipping. An HR manager would care about a downward trend even when the average still looks fine. It uses `attendance_rate` and `month` from `attendance_monthly_features`, only from completed months before the observation date, and is NaN when fewer than two months are available (then filled by the training median in the pipeline). On the real data it is a weak signal (single-column AUC 0.53 on 376 master-snapshot rows), so treat it as a candidate, not a proven winner.

### Missing values, scaling and encoding

* **Numbers:** `SimpleImputer(strategy="median")` (robust to the long tails found in Task 1), then `StandardScaler`. For numeric columns that had blanks in the training data, an extra 0/1 "was missing" column is added.
* **Structural blanks (Task 1 decisions D11/D12):** blank "months since promotion" means never promoted, so it is filled with tenure and flagged `never_promoted`; blank raise % means no raise, so it is filled with 0 and flagged `never_raised`.
* **Skewed columns:** pay, overtime and variable pay use `log1p` before scaling.
* **Text:** blanks become their own `"Unknown"` category, then one-hot encoding. Categories not seen in training are encoded as all zeros instead of raising an error.
* **Monthly features:** median imputation + scaling, without a "was missing" flag (it would be identical for all simulated rows and add nothing beyond the median fill). Note that the constant fill still makes those rows easy to recognise; see the limitations.
* Output is a dense float array with no NaN or infinite values (checked in the tests and in `run_pipeline.py`).

### Run it

```bash
# pipeline: build features, split 80/20 (stratified), fit on train, transform train and test
python scripts/run_pipeline.py                     # confirmed-safe features
python scripts/run_pipeline.py --include-unconfirmed   # also dept_attrition_rate_prior_12m
docker compose run --rm pipeline                   # same thing in Docker

# the three-row test (shows its output)
pytest tests/test_features.py::TestThreeRows -v -s

# all tests (Task 1 + Day 2)
pytest -q
docker compose run --rm test
```

`pytest.ini` sets `pythonpath = src`, so `PYTHONPATH` is not needed for the tests. The three-row test uses synthetic rows that follow the real schema; a separate `TestRealDataset` class runs the same checks on the actual workbook and is skipped if `data/raw/02_CortexHR_AI_ML_Dataset.xlsx` is missing.

### Why fit on training data only

I fit the preprocessing pipeline on training data only so that information from the test set cannot influence imputation, scaling, or category encoding. This helps prevent data leakage and provides a more realistic evaluation of how the model performs on unseen employees. The fitted pipeline is then used to transform test data and future records without fitting it again.

In this project that means the medians used for blanks, the means and standard deviations used for scaling, the list of known departments, roles and cities, and the typical pay mid-point per role used to spot salary typos are all learned from the training split. A new hire scored next year is transformed with those saved numbers, exactly like a test row. The tests check this directly (scaler statistics equal the training rows' statistics, and `transform` leaves the fitted values unchanged).

### Known limitations and open questions

* **`dept_attrition_rate_prior_12m` is unconfirmed (open item O2).** For the 383 master-snapshot rows it is constant per department and sits about 0.7 points from the prior-12-month exit rate in the headcount sheet (versus 10.8 points from the following 12 months). For the 2,000 simulated rows it varies row by row (e.g. 497 different values within one department) and does not track the headcount sheet, so it cannot be verified. Its link to the target is about nil in both sources (AUC 0.53 and 0.49). Until the Python team answers, it is excluded by default.
* **Negative tenure (O1):** the 4 rows are excluded from training, not corrected.
* **Monthly data covers only the 383 master-snapshot employees.** 2,000 of 2,379 rows (84%) have no attendance or payroll history, so these features are filled with the training median for them and can only help on about 379 rows (about 50 leavers). The three monthly features were weak in a quick single-column check (AUC 0.48–0.56).
* **Month availability:** the window assumes the September 2025 attendance and payroll figures were known on 30 Sep 2025. This cannot be confirmed from the data. `add_monthly_features(..., lag_months=1)` is available for a more conservative window; its effect has not been measured.
* The monthly attendance average is only moderately correlated (r = 0.83) with the existing `attendance_rate_3m` column, so I did not add it as a separate input; why they differ is not documented.
* Simplifications versus the Task 1 decision log: blanks are filled with the overall median (decisions D14/D15 mention department/city medians), and the "was missing" columns for engagement, performance rating and commute come from the imputer rather than hand-named flags.
* **The two row sources are easy to tell apart.** A random-forest check found that the master-snapshot rows and the simulated rows can be separated from the pipeline output with AUC 1.0 (0.998 even without the monthly features), mainly through `attendance_rate_3m`, `months_since_last_raise` and the monthly features. Their leaver rates (13.2% vs 15.0%) do not differ significantly (p = 0.42), so this is not target leakage, but a model trained on the mix may partly learn "which source is this". Report metrics per source in Task 3. In a quick logistic-regression check (5-fold CV, not a Task 3 result) the AUC was 0.68 on master-snapshot rows and 0.72 on simulated rows.
* A random 80/20 split mixes simulated rows from 2021 to 2025; a time-aware split may be needed in Task 3.
* Tested with pandas 2.3, NumPy 2.3, scikit-learn 1.8 and pytest 9; `requirements.txt` was not changed. pandas 3 has not been tried.
* The GitHub repository could not be opened from the environment used to write this, so the work was done on a local copy of the project and has not been pushed.
