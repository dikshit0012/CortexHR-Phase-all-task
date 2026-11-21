# Final Evaluation Results

Run date: 2026-11-20 (UTC). Base commit: `ea20740` (branch `day-20-final-audit`; documents added afterwards). Environments: Python 3.12.3; attrition/pay-band/Tasks 13-15 in a venv pinned to scikit-learn 1.8.0, numpy 2.4.4, pandas 3.0.2; other tasks scikit-learn 1.9.1. Data: fictional workbook and CSVs supplied by the project owner (not committed). Logs are in `docs/evidence/`.

This is a historical audit snapshot. Phase 4 Tasks 16-19 and the Task 20 materials were consolidated into `main` on 2026-11-21; updated availability and test results are recorded in [CONSOLIDATION_UPDATE.md](CONSOLIDATION_UPDATE.md).

## 1. Evaluation comparison

| Component | Evaluation | Previous Result | Final Result | Change | Status |
|---|---|---|---|---|---|
| Attrition-v2 | `run_task13.py --version attrition-v2` (5-fold OOF) | shipped `metrics.json`: ROC-AUC 0.7114, PR-AUC 0.3097 | identical; **76 of 76 numeric metrics equal** | none | **Pass** |
| Attrition-v2 | row/fold audit | 2,379 rows, 349 leavers, fold fingerprint `646e474fae66cb0c` | same | none | Pass |
| Attrition-v2 artifact | load supplied `model.joblib` | - | file corrupted in upload (starts `EF BF BD`) | - | **Blocked** (regenerated locally, deterministic) |
| Attendance anomaly | `run_task5.py` with `unlock_CX-05` | committed reports | `git diff` of Task 5 reports: 0 lines | none | **Pass** |
| Payroll anomaly | `run_task6.py` with `unlock_CX-06` | committed results tables | metrics CSVs identical; IDs/timestamps differ (salted hashes) | none | **Pass** |
| Anomaly explanations | `run_task7.py` | 507 results rows | 507 rows, same reason-code counts; scores correlate 0.9987 on overlap | negligible | Pass (human review of 20 explanations still PENDING) |
| Intent classifier | `train_intent_model.py` + `evaluate_intent_model.py` | committed reports | CSV/JSON reports identical | none | Pass |
| Resume ranking | `run_task11.py` | P@5 0.92, NDCG@5 0.912 | same (tracked reports unchanged) | none | Pass |
| Resume fairness | `run_task12.py` with `unlock_CX-12` | twin gap 0.000 | same; 50 tests pass | none | Pass |
| Pay bands | `run_task15.py` | report: gradient boosting MAE 60,795.9 | MAE 60,801.4 | +5.5 (+0.009%) | Pass (library-version effect) |
| Pay-band gender / career-gap audit | official audit with `unlock_CX-15` | blocked earlier | ran; gaps documented in model card | - | **Warning** (gaps exist, mitigations ineffective) |
| Drift + fairness monitor | `monitor.py` on real data | none | status **WATCH** (2 PSI watches; fairness OK) | - | Warning |
| Headcount forecasting | backtest | - | Task 16 not present in the 2026-11-20 audit snapshot | - | **Not run in this snapshot** |
| API security + endpoints | local TestClient | - | see `SERVICE_VERIFICATION.md` | - | Pass (local only) |
| Live deployed API | any | - | no deployed URL supplied | - | **Not run** |
| GitHub Actions monitoring job | cron | - | cannot run here | - | **Not run** |
| Attrition fairness | group audit | - | not implemented in the project | - | **Not run** |
| Task 14 explanation tests | `pytest` | - | stalled in the SHAP/KernelExplainer fixtures; **not completed** | - | **Blocked** |

## 2. Detailed metrics

**Attrition-v2 (out-of-fold):** ROC-AUC 0.7114 (sd 0.0182), pooled 0.7080 (0.6766-0.7357); PR-AUC 0.3097; P@10% 0.332, R@10% 0.226, lift 2.26; threshold 0.5: precision 0.327, recall 0.023, F1 0.040; Brier 0.1163; ECE 0.0200.
**Attendance:** combined P 0.488 / R 0.827 / F1 0.614 (254 flags); Isolation Forest 0.412 / 0.607 / 0.491; rules 0.918 / 0.600 / 0.726; LATE_SPIKE recall 0.133.
**Payroll:** combined P 0.383 / R 0.97 / F1 0.550 (253 flags, 97 TP, 3 FN); rules 0.767 / 0.92 / 0.836; IF 0.304 / 0.56 / 0.394.
**Intent classifier:** accuracy 0.803, macro-F1 0.7997 (132 rows); at threshold 0.64: answers 44% of messages at 0.966 accuracy.
**Resume ranking:** hybrid P@5 0.92, NDCG@5 0.912 versus TF-IDF 0.90/0.854 and skill-only 0.90/0.870.
**Pay bands:** GB MAE 60,801 / R2 0.978; linear 82,826; role baseline 110,298; coverage of 80% band 0.802.
**Monitor (2026-11-20):** `tenure_months` PSI 0.120 WATCH; `attendance_rate_3m` PSI 0.039 vs matched reference (WATCH) and **1.467 vs the full training reference (training-serving skew)**; six other features OK; twin-resume gap 0.000; pay gaps gender 0.0170 and career 0.0348 (baselines 0.0170 / 0.0350).

## 3. Test suite (latest run per task)

| Test Type | Passed | Failed | Skipped | Status |
|---|---:|---:|---:|---|
| Task 1 data audit | 4 | 0 | 0 | Pass (needs `PYTHONPATH=src`) |
| Task 2 pipeline | 56 | 0 | 4 | Pass |
| Task 3 baselines | 14 | 0 | 0 | Pass (stale test fixed on this branch) |
| Task 4 evaluation | 110 | 0 | 0 | Pass |
| Task 5 attendance | 17 | 0 | 0 | Pass |
| Task 6 payroll | 96 | 0 | 0 | Pass |
| Task 7 explanations | 132 | 0 | 0 | Pass |
| Task 8 anomaly API | 80 | 0 | 0 | Pass |
| Task 9 intent model | 157 | 0 | 0 | Pass |
| Task 10 chatbot | 172 | 3 | 4 | **Fail** (patch not applied to Task 8 app) |
| Task 11 resume ranking | 72 | 0 | 0 | Pass |
| Task 12 fairness | 50 | 0 | 0 | Pass |
| Task 13 attrition v2 | 21 | 0 | 0 | Pass |
| Task 14 explanations | - | - | - | Blocked (did not finish) |
| Task 15 pay bands | 33 | 0 | 0 | Pass |
| Task 17 API | - | - | - | Blocked (test file cannot be collected; conftest missing) |
| Task 18 monitoring | 104 | 2 | 0 | **Fail** (2 tests assume private data is absent; 33/33 pass in `test_fairness_monitor.py` when local data is hidden) |
| Task 19 console | 105 | 0 | 0 | Pass |
| Final challenge guard | 6 | 0 | 0 | Pass (separate branch) |
| **Total (counted)** | **1,229** | **5** | **8** | |

Mixed-version note: Task 10 results are for scikit-learn 1.8.0 (the intent model was retrained with it); under 1.9.1 the same model produced load errors, which is a real artifact/version coupling.
