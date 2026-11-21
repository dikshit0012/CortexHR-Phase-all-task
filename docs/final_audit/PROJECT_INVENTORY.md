# CortexHR - Project Inventory (Day 20 audit)

Audited commit base: `main` (tag `phase-3`) plus the Phase 4 files supplied for this audit (Tasks 17 partial, 18, 19).
Legend - **Implementation**: ML = trained model, STAT = statistical/unsupervised + rules, RULE = rules only, ORCH = orchestration, API, UI.
All data in the project is fictional (Zenith Nova Technologies / Aura Workforce); no real-world production data was used.

## AI / ML and decision-support components

| Component | Task/Phase | Type | Implementation | Evaluation (re-run in this audit) | Deployment | Status |
|---|---|---|---|---|---|---|
| Attrition model `attrition-v2` | T13 / P3 | Classification (ML) | Calibrated Logistic Regression (balanced), 33 features | Reproduced exactly: 76/76 metrics identical to the shipped `metrics.json` | Loaded by local API (verified locally) | Verified |
| Attrition explanations | T14 / P3 | Explainability layer | SHAP KernelExplainer on `attrition-v2`; total_experience_years suppressed in reasons | 23+ tests passed; full-scoring test too slow to finish in this session | Offline report only | Partially verified |
| Resume ranking | T11 / P3 | Information retrieval (not trained) | TF-IDF cosine (0.6) + required-skill overlap (0.4), fixed weights | Re-run: P@5 0.92, NDCG@5 0.912 | Loaded by local API | Verified |
| Fair resume screener | T12 / P3 | Preprocessing/masking + fairness audit | Masks names, e-mail, location, tier; twin-pair audit | Re-run with real protected-attribute file: 50 tests passed | Not deployed separately | Verified |
| Attendance anomaly detector | T5 / P2 | Unsupervised ML + rules | IsolationForest (contamination 0.03, 9 features) OR rules | Re-run with hidden labels: reproduces committed metrics exactly | Loaded by local API | Verified |
| Payroll anomaly detector | T6 / P2 | Unsupervised ML + rules | IsolationForest + rules | Re-run with hidden labels: results tables identical to committed | Loaded by local API | Verified |
| Anomaly explanations | T7 / P2 | Rules (no model) | Reason codes, text, money-at-risk estimate | 132 tests passed; human review of explanations still PENDING | Part of Task 8 service | Verified (synthetic + data) |
| HR intent classifier `hr-intent-v1` | T9 / P2 | Text classification (ML) | TF-IDF word+char + Logistic Regression (balanced), threshold 0.64 | Re-trained: accuracy 0.803, macro-F1 0.7997 on 132 test rows | Loaded by local API (retrained locally) | Verified |
| Chatbot orchestration | T10 / P2 | Orchestration + mock HR service | Month/slot extraction, memory, role-based access, mock data | 172 passed, 3 failed (patch not applied to Task 8 app) | Not integrated in repo | Warning |
| Pay-band model `paybands-v1` | T15 / P3 | Regression (ML) | GradientBoostingRegressor (300 trees, depth 3) vs Linear and role-average baselines | Re-run: MAE 60,801, R2 0.978; gender and career-gap audit now run | Loaded by local API | Verified |
| Headcount forecasting | T16 / P4 | Forecasting | **Code not in the repository and not supplied** | Not run | API returns 503 for `/v1/forecast/headcount` | **Not verifiable** |

## Non-ML systems

| Component | Task | Type | Notes |
|---|---|---|---|
| Data audit and cleaning rules | T1-T2 | Pipeline | 4 tests (T1), 56 (T2) pass; planted-dirt answer key (`unlock_CX-01`) was supplied but not compared in this audit |
| Baselines and evaluation harness | T3-T4 | Evaluation | 14 (T3) and 99+ (T4) tests pass |
| Anomaly service (FastAPI) | T8 | API | 80 tests pass; artifacts must be rebuilt with `build_artifacts.py` |
| Unified API `cortexhr_api` | T17 | API | 11 endpoints; source supplied only as files touched by Task 18; its own test suite and Task 16 module are missing |
| Drift + fairness monitor | T18 | Monitoring | PSI on 8 features + twin-resume + pay-gap re-runs; ran on real data: status WATCH |
| GitHub Actions `model-monitoring.yml` | T18 | CI/cron | Present; **not executed** (no GitHub runner in this audit) |
| Streamlit console | T19 | UI | 105 tests pass; live integration tests not run |
| Render blueprint / Dockerfile | T17 | Deployment config | Present; **never deployed or built in this audit** |

## Corrections found during the inventory
* `MODEL_CARD_TEMPLATE.md` did not exist; created.
* Task 9 exists twice in the repo (`cortexhr_task9` and `cortexhr_task9_audit`); the former is used by the API.
* Task 10's `/v1/chat/converse` is only a patch file; the Task 8 app in the repo does not contain it.
* Datasets, model artifacts and the hidden-label files are (correctly) not committed; this audit used locally supplied copies.
