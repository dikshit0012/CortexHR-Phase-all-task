# Task 8 - The Anomaly Service (CortexHR AI/ML, Phase 2)

A FastAPI service that puts the **Task 5 attendance detector**, the **Task 6 payroll detector** and the **Task 7 explanations**
(reason code, plain-language text, money at risk, ranking) behind validated HTTP endpoints. The official Task 7 output format
(`sample_anomaly_results`, 12 columns) is returned unchanged. *All data is fictional (Zenith Nova Technologies / Aura Workforce).*

| Endpoint | Purpose | Success | Errors |
|---|---|---|---|
| `POST /v1/anomaly/attendance` | one attendance record -> one official result row | 200 | 422, 500, 503 |
| `POST /v1/anomaly/payroll` | one payroll record -> one official result row | 200 | 422, 500, 503 |
| `POST /v1/anomaly/attendance/batch` | 1..N attendance records -> `{detector, count, results[]}` | 200 | 422, 500, 503 |
| `POST /v1/anomaly/payroll/batch` | 1..N payroll records -> `{detector, count, results[]}` | 200 | 422, 500, 503 |
| `GET /v1/health` | liveness + "models loaded?" | 200 | 503 if not loaded |
| `GET /v1/models` | artifact versions, limits (no paths, no data) | 200 | 503 |
| `POST /v1/chat` | (Task 9) HR question -> intent, confidence, FAQ reply or fallback | 200 | 422, 503 |

Interactive docs: `http://127.0.0.1:8000/docs`.

Task 9 (`POST /v1/chat`) is loaded in the same FastAPI process. The intent model is loaded once at startup. If that artifact is missing, only `/v1/chat` returns 503; the anomaly endpoints keep working. Set `CORTEXHR_TASK9_ROOT` to the folder that contains `src/cortexhr_task9` when Task 9 is not next to this `task 8` folder.

## 1. How it works

```
build time (once)                          run time (every request)
scripts/build_artifacts.py                 app startup (lifespan, once per process)
  reference workbook                         load reference workbook  +  load 2 artifacts   (no fitting)
    -> Task 5 / Task 6 training pipelines  request -> Pydantic validation
    -> joblib artifacts + safe metadata      -> combined = reference history + request rows
                                             -> Task 5/6 features + rules, fitted forest .predict()
                                             -> Task 7 explain -> rank -> official output (request rows only)
```

* **Nothing is fitted at startup.** `scripts/build_artifacts.py` runs the existing training pipelines once and saves the fitted
  objects with `joblib`. The API only *loads* them (`app/bootstrap.py`, inside FastAPI's `lifespan`) - once per process.
  Fitting at startup is not required by the existing code, so it is not done. The service never calls `fit`
  (a test makes `fit` raise and then serves requests).
* **History (Option 1).** Both detectors need an employee's earlier months, the previous month's role median, other payslips of the
  same employee-month and a dataset-wide deduction share. The reference workbook is loaded once from the path in
  `CORTEXHR_WORKBOOK`. For each request: `combined = reference rows + request rows`; Task 5/6/7 code runs on `combined`; only the
  request rows are returned. A request whose id already exists in the reference *replaces* that row (re-scoring a correction).
  Requests are never written back into the reference.
* **Task 7 min-max scaling is preserved, not recomputed per request.** Task 7 scales the raw Isolation Forest score over *all*
  scored rows of a dataset. The build script stores that min and max (from scoring the reference) in the artifact
  (`score_scale` - visible in `/v1/models`) and every request uses the same two numbers. So a record gets the **same
  `anomaly_score` alone, in a batch, or in another request** (tested), and for the reference data it reproduces Task 7's values
  exactly. Rows outside the reference range are clipped to 0..1 (the official column must stay in 0..1).
* **Ranking.** Task 7's `rank_flags` (money at risk, reason priority, signal count, raw score, record id - deterministic) is applied to
  the flagged records of a response; `result_id` `AN-0001...` follows that rank. Un-flagged records get the next ids (by raw score, then
  record id). The `results` list itself is in **input order**; sort by `result_id` to get the ranking. `result_id` is unique within one
  response, not globally.
* **Reused, not copied:** Task 5 `build_feature_frame`, `apply_rules`, `combine_flags`; Task 6 `build_payroll_features`, `apply_rules`,
  `combine_flags`; Task 7 `explain_attendance`, `explain_payroll`, `rank_flags`, `to_official`, `validate_official_values`. Tasks 5-7 were not modified.

### Project structure and inputs

The service expects these existing source folders and reads only the two feature sheets in the local reference workbook:

```
CortexHR source checkout/
  phase-2/task05/src/                  attendance detector
  phase-2/task06/task 6/src/           payroll features, rules, detector
  phase-2/task07/task 7/src/           reasons, explanations, money at risk, ranking
task8_anomaly_service/
  task 8/app/                  FastAPI service, validation, artifacts, scoring
  task 8/scripts/              offline artifact builder
  task 8/tests/                synthetic and optional real-workbook tests
  task 8/examples/             fictional request JSON
  task 8/reports/              captured test, API, startup, benchmark, comparison evidence
  task 8/models/               locally generated joblib files; excluded from Git
```

In the current Windows layout the source checkout is separate from the Task 8 folder, so configure
`CORTEXHR_TASKS_ROOT` as shown below. `CORTEXHR_WORKBOOK` must point to a local `.xlsx` containing
`attendance_monthly_features` and `payroll_monthly_features`; the service does not read answer-key or label sheets.

## 2. Schema changes versus the starter skeleton (and why)

The skeleton (`05_Starter/api_skeleton/app/main.py`) only had a placeholder for `POST /v1/anomaly/attendance`. Its models could not
drive the real detectors or carry the official output, so they were replaced after reading the Task 5-7 code and the sample sheet:

| Skeleton | Problem | Now |
|---|---|---|
| `AttendanceIn`: 7 fields (`employee_id`, `scheduled_days`, `present_days`, `wfh_days`, `late_arrivals`, `avg_hours_worked`, `overtime_hours`) | Task 5 `REQUIRED_COLUMNS` also needs `attendance_feature_id`, `month`, `half_days_unpaid`, `paid_leave_days`, `unpaid_absence_days`, `longest_unpaid_streak_days`, `attendance_rate`; without them it raises "missing required columns" | `AttendanceRecord` = Task 5 `REQUIRED_COLUMNS` (+ optional `wfh_days`, `job_level`, `department_id`) |
| *(none)* | no payroll endpoint | `PayrollRecord` = Task 6 `REQUIRED_COLUMNS` (+ optional `department_id`, `variable_pay_type`) |
| `AnomalyOut`: 6 fields, `flagged: bool` | official sample has 12 columns and `flagged` is `"Y"` | `AnomalyResult` = the 12 `OFFICIAL_COLUMNS`, same order; `flagged` is `"Y"`/`"N"` |

The skeleton's `/v1/attrition/risk` and `/v1/chat` placeholders belong to other tasks and are not part of this service.
`GET /v1/health` was kept (extended with `models_loaded`). The service lives in `task 8/app/` (the skeleton was not in the repository).

**Two documented extensions of the official format** (the sample sheet only contains flagged rows):
1. Every submitted record gets a result. If it is **not** flagged: `flagged="N"`, `reason_code`, `reason_text`, `money_at_risk` = `null`.
2. `money_at_risk` is `null` whenever Task 7 has no defensible formula (all attendance reasons, statistical outliers, missing data) - same as the sample sheet. No amount is invented.

`model_version` keeps the Task 7 official strings (`iforest+rules-attendance-task5`, `iforest+rules-payroll-task6`) so the output format is unchanged;
the model artifact versions `attendance-anomaly-v1` / `payroll-anomaly-v1` are reported by `GET /v1/models`.
The serialized artifact container is format version 2 and records the exact Python, NumPy, pandas, SciPy,
scikit-learn, joblib, and openpyxl versions; the loader refuses mismatches.

## 3. Validation rules (Pydantic)

Taken from the data and the detector contracts, not guessed. Unknown fields -> 422. Blank ids -> 422. `NaN`/`Infinity` -> 422.
Numeric measures are *required keys* but may be `null` (Task 5 imputes them; Task 6 has an explicit `INVALID_PAYROLL_DATA` rule for a missing amount, so a null is scored, not rejected).
Money may be negative (negative net pay is what the detector must catch). Day/hour counts >= 0; `attendance_rate` in 0..1; `overtime_eligible` in `Y`/`N`;
payroll `month` must be the 1st of a month and `run_id` must be `PR-YYYY-MM` (Task 6 requirements). Error bodies contain only `loc`, `type`, `msg` - submitted values are never echoed.
Every one of the 7,339 attendance and 7,355 payroll rows of the real workbook passes these schemas (real-data test).

## 4. Batch endpoints

* **Limit:** `100` records per request by default; set `CORTEXHR_MAX_BATCH_SIZE` (1..1000, otherwise the service refuses to start). Shown in `GET /v1/models`.
  100 keeps a request under about a second here (measured: 100 attendance records ~0.5 s, 100 payroll records ~0.75 s; the cost is one feature/rule/explanation pass over the reference history per request, not per row) and bounds memory and request size for a training service.
* **One detector per endpoint** (`/attendance/batch`, `/payroll/batch`); an unknown detector name is a 404.
* **Empty batch** -> 422. **Too many records** -> 422 with the limit in the message. **Any invalid record** -> 422 for the whole batch (all-or-nothing, no partial results). **Duplicate record ids inside one batch** -> 422.
* Results are in input order; the detector, scaling and explanation code is the same loaded instance used by the single endpoints; nothing is loaded or fitted per row.

## 5. Install and run (Windows PowerShell)

The verified Windows environment is Python 3.14.6 with the exact dependency versions pinned in `requirements.txt`.
The Task 8 folder may be separate from the Task 5–7 checkout. In that layout, set `CORTEXHR_TASKS_ROOT` to
the repository root containing `phase-2/task05/src`, `phase-2/task06/task 6/src`, and `phase-2/task07/task 7/src` before importing the service.
When Task 8 is placed directly beside those folders in one repository, the default source root works without
that variable.

```powershell
Set-Location 'C:\path\to\CortexHR-Phase-2-3\phase-2\task09\cortexhr_task9\task 8'
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt

# Root of the existing Task 5–7 checkout (omit only in the integrated layout described above).
$env:CORTEXHR_TASKS_ROOT = 'C:\path\to\CortexHR-'

# Local reference workbook; keep this file outside Git.
$env:CORTEXHR_WORKBOOK = 'C:\path\to\02_CortexHR_AI_ML_Dataset.xlsx'

# Optional; default is .\models
# $env:CORTEXHR_MODEL_DIR = 'C:\path\to\local-models'

.\.venv\Scripts\python.exe .\scripts\build_artifacts.py
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`CORTEXHR_WORKBOOK` is required. `CORTEXHR_MODEL_DIR` defaults to `.\models`; `CORTEXHR_MAX_BATCH_SIZE`
defaults to 100 and accepts 1..1000. `CORTEXHR_TASKS_ROOT` is optional only when the Task 5–7 folders are
beside `task 8`.

The offline artifact builder fits the existing Task 5/6 detectors, stores `attendance-anomaly-v1` and
`payroll-anomaly-v1` joblib artifacts plus safe metadata under the model directory, and records the actual
Python, NumPy, pandas, and scikit-learn versions. Artifact format version 2 refuses any mismatch in those
four versions, so build and serve in the same pinned environment. The API never rebuilds artifacts.

If the workbook, a model artifact or a setting is missing or unusable the process **exits at startup with a clear message** - it never falls back to dummy predictions:

```
app.config.StartupError: CORTEXHR_WORKBOOK is not set. Point it at the reference workbook (02_CortexHR_AI_ML_Dataset.xlsx).
app.config.StartupError: Reference workbook not found: C:/nope/missing.xlsx (check CORTEXHR_WORKBOOK).
app.config.StartupError: Model artifact missing: payroll-anomaly-v1.joblib. Rebuild with: python scripts/build_artifacts.py
```
(verified startup failures are recorded in `reports/task8_startup_failures.txt`). Artifacts are also refused if
they are corrupted, incomplete, swapped, built with different Python/NumPy/pandas/scikit-learn versions, or
built from a different reference workbook.

## 6. curl examples

Request bodies are in `examples/` (fictional values). Run them from the `task 8` folder. On Windows use **`curl.exe`** (plain `curl` in PowerShell is an alias of `Invoke-WebRequest`).

```powershell
curl.exe -sS -X POST http://127.0.0.1:8000/v1/anomaly/attendance -H "Content-Type: application/json" --data-binary "@examples\attendance_request.json"
curl.exe -sS -X POST http://127.0.0.1:8000/v1/anomaly/payroll -H "Content-Type: application/json" --data-binary "@examples\payroll_request.json"
curl.exe -sS -X POST http://127.0.0.1:8000/v1/anomaly/attendance/batch -H "Content-Type: application/json" --data-binary "@examples\attendance_batch_request.json"
curl.exe -sS -X POST http://127.0.0.1:8000/v1/anomaly/payroll/batch -H "Content-Type: application/json" --data-binary "@examples\payroll_batch_request.json"
curl.exe -sS -o NUL -w "HTTP %{http_code}`n" -X POST http://127.0.0.1:8000/v1/anomaly/attendance -H "Content-Type: application/json" --data-binary "@examples\invalid_attendance_request.json"
curl.exe -sS -o NUL -w "HTTP %{http_code}`n" -X POST http://127.0.0.1:8000/v1/anomaly/payroll -H "Content-Type: application/json" --data-binary "@examples\invalid_payroll_request.json"
```
The `-o NUL -w` calls verify the HTTP status without writing a response body to a file. PowerShell alternative:
`Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/anomaly/attendance -ContentType 'application/json' -InFile .\examples\attendance_request.json`.
Use `Invoke-RestMethod` with the corresponding payroll or batch JSON file for the other valid endpoints.
`GET /v1/models` reports artifact/model versions, reference row counts, feature lists, score scale, and batch limit;
it does not expose file paths or employee data.

**Captured responses** (fictional sample requests; actual Windows command output is in `reports/task8_curl_evidence.md`):

Attendance (`examples/attendance_request.json`, HTTP 200):
```json
{
 "result_id": "AN-0001",
 "source_dataset": "attendance_monthly_features",
 "record_id": "AF-EX-0001",
 "employee_id": "EMP-0002",
 "period": "2026-10-01",
 "anomaly_score": 1.0,
 "flagged": "Y",
 "reason_code": "NO_ATTENDANCE",
 "reason_text": "No attendance was recorded for the whole month although 21 working days were scheduled and paid leave does not cover them. Please check whether the employee was absent, on leave that was not recorded, or missing from the attendance system.",
 "money_at_risk": null,
 "model_version": "iforest+rules-attendance-task5",
 "scored_at": "2026-10-28 10:17:24"
}
```
Payroll (`examples/payroll_request.json`, gross pay 10x too high, HTTP 200):
```json
{
 "result_id": "AN-0001",
 "source_dataset": "payroll_monthly_features",
 "record_id": "PI-EX-0001",
 "employee_id": "EMP-0002",
 "period": "2026-10-01",
 "anomaly_score": 0.667,
 "flagged": "Y",
 "reason_code": "GROSS_ABOVE_NORM",
 "reason_text": "The gross pay (1,600,000.00) is higher than the pay components on this payslip add up to (160,000.00). About 1,440,000.00 may be affected. Please check the fixed pay, overtime and bonus amounts.",
 "money_at_risk": 1440000.0,
 "model_version": "iforest+rules-payroll-task6",
 "scored_at": "2026-10-28 10:17:24"
}
```
Invalid request (`present_days: -3`, `attendance_rate: 1.5`) -> **HTTP 422**:
```json
{"detail": [{"loc": ["body", "present_days"], "type": "greater_than_equal", "msg": "Input should be greater than or equal to 0"},
            {"loc": ["body", "attendance_rate"], "type": "less_than_equal", "msg": "Input should be less than or equal to 1"}]}
```
Batch (3 attendance records; input order kept, `result_id` carries the rank: the flagged record is `AN-0001`): see the evidence file.
The attendance example uses a **fictional** record id; `EMP-0002` exists in the reference, so its earlier months are used as history.

## 7. Tests

```powershell
Set-Location 'C:\path\to\CortexHR-Phase-2-3\phase-2\task09\cortexhr_task9\task 8'
$env:CORTEXHR_TASKS_ROOT = 'C:\path\to\CortexHR-'
$env:CORTEXHR_WORKBOOK = 'C:\path\to\02_CortexHR_AI_ML_Dataset.xlsx'
.\.venv\Scripts\python.exe -m pytest -v
```
Verified on Windows with the reference workbook configured: **80 passed** (`reports/task8_test_output.txt`).
The run covered both real-data schema validation and all flagged Task 7 results. A separate clean virtual-environment
installation was attempted but the package-download process stalled; the tests and server were run using an existing
Windows Python 3.14.6 environment after installing the pinned requirements there. The exact dependency set is recorded
in `requirements.txt` and in the captured `/v1/models` output.

The tests build a small fictional company (40 employees x 6 months), build artifacts from it with the real build code and start the real app on those.
They cover the six required cases (valid attendance, valid payroll, invalid attendance 422, invalid payroll 422, load-once, output schema/explanation) plus: money-at-risk formulas,
duplicate payslip (no double count), several flags on one record (one reason, one amount), batch (valid, order, ranking, empty, oversized, invalid record, duplicate ids), score independence from batch
composition, unknown detector, 503 without models, generic 500 on prediction / preprocessing failure (no leaked internals), no echo of submitted values in 422s, and startup failures
(missing / corrupted / swapped / incomplete / incompatible Python, NumPy, pandas, SciPy, scikit-learn, joblib or openpyxl /
wrong-reference artifacts; missing / wrong-type / unreadable / sheet-less / column-less workbook; bad settings).
The real-data comparison reproduced all 254 attendance and 253 payroll flags, scores, reasons, texts, money-at-risk values,
and per-detector ranking order exactly against a fresh Task 7 run; detailed counts are in
`reports/task8_task7_comparison.txt`.

Task 5–7 regression suites were run separately in the existing Windows environment: Task 5 **14 passed**; Task 6
**96 passed, 5 warnings**; Task 7 **132 passed**. Running their three test directories in one pytest invocation
caused collection-name collisions between duplicate `test_pipeline_smoke.py` modules, so the verified commands were
run independently. Task 6 warnings are pandas deprecation warnings from its reporting module.

## 8. Versioning a future model

Versions are explicit (`attendance-anomaly-v1`, `payroll-anomaly-v1`) and part of the artifact file name, its metadata and `/v1/models`. To update: change the detector in its own task, bump the version constant in `app/artifacts.py` (e.g. `-v2`),
rebuild into a **new** model directory, run both test suites **and** the real-data comparison, compare flag counts / reasons against the previous version on the same workbook, then point `CORTEXHR_MODEL_DIR` at the new directory. Never copy a single new artifact into an old directory:
the loader rejects mixed, old-format, or incompatible artifacts. The current artifacts use format version 2;
the official Task 7 `model_version` strings are unchanged unless the format owner agrees.

## 9. Security and data

* No dataset, labels, ID mappings or credentials are stored in `task 8/`. `.gitignore` excludes `task 8/data/`, workbooks/CSVs under `task 8/`, `models/*.joblib` and `models/*.metadata.json`. Only fictional JSON examples are intended to be committed.
* The adjacent `CortexHR-` checkout previously tracked the Task 5 workbook, Task 5 label-derived missed-anomaly reports,
  and both Task 7 evaluation-only CSVs. Ignore rules were added and all five files were removed from the Git index
  without deleting their local copies. The local `origin/master` tracking ref still contains earlier commits with these
  files; no history rewrite or push was done. If the remote repository contains those commits, treat the dataset and
  evaluation information as exposed and coordinate history remediation before publication. The repository documentation
  identifies the Task 5 dataset as fictional; its answer-key and evaluation content should still remain out of future commits.
* A separate Task 3 workbook remains tracked in the adjacent checkout. Its workbook README and Task 3 findings identify
  it as a synthetic stand-in, not the official Task 3 dataset; its employee sheet still contains employee IDs and
  attrition-related fields. It was not included in the Task 5/7 evaluation-file removals. Confirm its intended public
  distribution before publishing or sharing that repository.
* Older `answer_key_comparison.csv` files, Task 2's `day5_missed_anomalies_by_type.csv`, and Task 6's results/review
  tables remain tracked as aggregate-only evaluation reports; their schemas contain counts and metrics, not row IDs.
  They were left unchanged as report deliverables, but should also be excluded if aggregate answer-key-derived metrics
  are restricted.
* A scan of tracked filenames and text files found no credential-named files or high-confidence secret patterns. Binary
  workbooks and documents were not included in the content scan. The live remote was not queried.
* Only the two feature sheets are read; label sheets are never opened (tested). No label is used for any prediction, reason or amount.
* `joblib` can execute code when loading: load only artifacts you built with `scripts/build_artifacts.py`.
* Error responses never contain stack traces, file paths or submitted values; failures are logged by exception type only.
* With several uvicorn workers every process loads its own copy of the models and reference (memory x workers).

## 10. Limitations

* **Latency:** each request recomputes features, rules, and explanations over the complete reference history (~7,300 rows); this is the main known inefficiency. On Windows, TestClient benchmark results (5 measured runs after one warm-up; startup excludes Python import and artifact building) were: lifespan startup 7.66 s; attendance single median 1.376 s; payroll single 2.306 s; batch of 100 attendance 1.615 s; batch of 100 payroll 2.756 s. These are local observations, not guarantees; feature engineering and explanation are included in request timings. No behavior-changing optimization was attempted. See `reports/task8_benchmark.txt`.
* **Not stateless in the data sense:** new months are not added to the reference; to include them, extend the workbook and rebuild the artifacts (the loader detects a stale artifact).
* **Historical Task 7 output drift:** the committed output and a fresh run use the same workbook and Task 5–7 source at `HEAD`, but the recorded prior Linux environment was scikit-learn 1.8.0/pandas 3.0.2 and this Windows verification used scikit-learn 1.9.1/pandas 3.1.0. Attendance matched exactly. Payroll had 253 flagged rows in each output but 9 IDs appeared only in each respective output; among 244 shared IDs, 229 scores, 9 reason texts, and 9 money-at-risk values differed, and 123 `result_id` values moved because ranks changed. Repeating Task 7 twice in the current environment produced identical output for all 507 rows. The fixed random seeds and repeat run make nondeterministic fitting unlikely; dependency-version drift is the explanation best supported by evidence, but because both scikit-learn and pandas changed, this run does not isolate which library caused which difference. Do not treat the committed payroll CSV as a cross-version oracle. The service and a fresh Task 7 run matched exactly in the same pinned Windows environment.
* `result_id` is the response-local ranking position, not a globally unique identifier. Attendance has no money formula, so its `money_at_risk` is always `null`.
* Windows Uvicorn, `curl.exe`, and `Invoke-RestMethod -InFile` requests were executed successfully; no `--reload` option was used. The clean isolated install itself was not completed; see the Windows installation note in section 7.
