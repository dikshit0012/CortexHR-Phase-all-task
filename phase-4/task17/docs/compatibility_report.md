# Dependency and artifact compatibility report (Task 17)

Produced before deployment. Every statement below was measured in this project's sandbox unless marked "not verified".

## 1. Artifact inventory

| Capability | Artifact(s) | Format | Trained / built with | Needs project code to unpickle | Inference without retraining |
|---|---|---|---|---|---|
| Attrition | `attrition-v2.joblib` (+ metadata) | `CalibratedClassifierCV` around a Task 2 pipeline | Python 3.12.3, scikit-learn **1.8.0**, numpy 2.4.4, pandas 3.0.2 | yes: `cortexhr` (phase-1/task04/src) | yes |
| Attendance anomaly | `attendance-anomaly-v1.joblib` (+ metadata) | Task 5 bundle | Python **3.14.6**, numpy 2.5.4, scikit-learn 1.9.1, pandas **3.1.0**, scipy 1.18.1, joblib 1.6.0, openpyxl 3.1.5 | yes: Tasks 5-7 sources | yes, **but needs the private reference workbook** |
| Payroll anomaly | `payroll-anomaly-v1.joblib` (+ metadata) | Task 6 bundle | same as above | yes: Tasks 5-7 sources | yes, **but needs the private reference workbook** |
| Pay-band | `paybands-v1_{gradient_boosting,linear_regression,average_by_role_baseline}.joblib` | sklearn pipelines + `RoleMeanBaseline` | scikit-learn **1.9.0** (from the loader's version warning) | baseline needs `cortexhr_task15` | yes; band ratios are **not** in the artifacts |
| HR chatbot | `hr_intent_classifier.joblib` (+ metadata) | TF-IDF + linear classifier | **retrained for this release** from the committed Task 9 CSV (original artifact unavailable); scikit-learn 1.9.1 | `cortexhr_task9` | yes |
| Resume ranking | none (stateless Task 11 pipeline) | TF-IDF fitted per request | n/a | `cortexhr_task11` | n/a |
| Headcount forecast | none (Task 16 fits statsmodels at startup) | live, or precomputed CSV | statsmodels 0.15.0 | `cortexhr_task16` | live fit once at startup; never per request |

## 2. Version conflicts found and how they were resolved

1. **Attrition was trained on scikit-learn 1.8.0; the anomaly artifacts on 1.9.1.** One environment cannot have both. Test: the same 200 varied rows were scored in two clean environments (scikit-learn 1.8.0 / numpy 2.4.4 / pandas 3.0.2, and 1.9.1 / 2.5.4 / 3.0.6). **Maximum absolute difference in probability: 0.0.** The release therefore pins 1.9.1 for everything. (Evidence is for this model and these inputs, not a general guarantee.)
2. **The Task 8 loader is deliberately strict**: it raises unless Python, numpy, scikit-learn, pandas, scipy, joblib and openpyxl equal the build versions. It was not weakened. `requirements.txt` pins the exact versions and the Dockerfile installs the exact interpreter 3.14.6 with `uv`.
3. **Pay-band models: 1.9.0 -> 1.9.1** is a patch difference; they load and predict. Calibration ratios for the bands were never saved, so bands are returned only when a calibration file is supplied.
4. **Folder layout**: the artifacts existed only on branch `phase1-tasks-1-5` (old `Task N/` layout) and are gitignored on `main`. They were extracted and committed under `phase-4/task17/artifacts/` (about 2 MB, weights only, no employee rows).

## 3. Clean-install verification

`requirements.txt` was installed into a fresh Python **3.14.6** virtual environment (uv) and the full test-suite was run there: see the Task 17 report for the counts. Versions imported: numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, openpyxl 3.1.5, statsmodels 0.15.0, FastAPI 0.163.1, pydantic 2.14.3.

## 4. What is NOT verified

* **Anomaly inference**: the reference workbook was not available, so the Task 8 service could not start and no real anomaly prediction was made. The endpoints answer 503 (never a guess) until the workbook is provided. The exact-version check of the Task 8 loader has therefore only been exercised up to the missing-workbook error.
* **Docker image build**: no Docker daemon was available. The Dockerfile follows the verified uv procedure but was not built here.
* **Hosted behaviour** (cold start, real HTTPS, secret files) is unverified until deployment.
