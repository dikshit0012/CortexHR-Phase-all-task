# CortexHR AI/ML — Employee Attrition (Task 1: Data Audit)

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
src/cortexhr/        reusable code: paths, audit helpers, data dictionary
scripts/             run_audit.py  (headless re-run)
tests/               unit tests for the helpers
reports/             generated locally by the audit; not committed
docs/                data_dictionary.md, decisions.md (hand-off to Task 2), open_items.md (questions for the Python team + response log)
```

## Key results

| | |
|---|---|
| Rows / columns | 2,389 x 42 (383 real employees + 2,000 simulated history rows + 6 duplicates) |
| Label | 14.6% left within 12 months (85.4% stayed) |
| Missing | Mostly *structural* (never promoted / no raise / no appraisal yet); only commute (55) and education (15) are random gaps |
| Impossible values | 4 negative tenures (not corrected, awaiting join dates), 2 attendance > 100%, 13 city spelling variants (+ 213 out-of-range WFH shares, 137 experience < tenure) |
| Outliers | IQR flags 1.6-7x more rows than z-score; 3 salaries are 10x typos |
| Leakage | 6 columns reveal the answer (model ROC-AUC 0.70 -> 1.00 when added) |
| Open items | 2 questions for the Python team (negative tenure join dates; `dept_attrition_rate_prior_12m` cut-off). Nothing assumed; record answers in `docs/open_items.md` |

## Notes

* Audit reports are computed by the notebook and written locally; generated outputs and workbook data are not included in this repository.
* `ydata-profiling` is used for the automatic comparison profile. Upstream has renamed it to `fg-data-profiling`; the old package still works but is no longer updated.
* The workbook includes an answer key sheet (`answer_key_planted_dirt`). Every check in the notebook is computed from the data; the key is used only to validate completeness in section 9. It is a validation, not a blind test, because the key was visible while exploring the file.
