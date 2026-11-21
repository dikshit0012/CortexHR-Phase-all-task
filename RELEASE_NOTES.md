# CortexHR Release Notes

Format follows Keep a Changelog. **This is a demonstration release on fictional data. It is not approved for real HR decisions.**

## [1.0.0] - 2026-11-20

### Added
* Attrition prediction (`attrition-v2`) with calibrated probabilities and SHAP reason codes (offline report).
* Resume ranking (TF-IDF + skill overlap) and a masking "fair screener" with a counterfactual twin-pair audit.
* Attendance and payroll anomaly detectors (Isolation Forest + rules) with reason codes and money-at-risk estimates.
* HR intent classifier and a chatbot service layer (mock HR service).
* Pay-band recommendation model with a gender / career-gap audit.
* One versioned FastAPI app (`cortexhr_api`, Task 17), a drift + fairness monitor (Task 18), a GitHub Actions monitoring workflow, and a Streamlit console (Task 19).
* Day 20 audit set, model cards, architecture, final report, demo script.

### Models
| Model | Purpose | Recommendation |
|---|---|---|
| attrition-v2 | 12-month leave probability | Release with warnings |
| Resume ranker + fairness audit | shortlist support | Release with warnings |
| Attendance / payroll anomaly | review aids | Release with warnings |
| hr-intent-v1 / chatbot | HR questions | Intent model with warnings; chatbot demo only |
| paybands-v1 | salary range support | Demonstration only; **not for real pay decisions** |
| Headcount forecast (Task 16) | - | Not included: code not in this repository |

### APIs
Verified **locally only**: `/v1/health`, `/v1/ready` (503 while forecast is unavailable), `/v1/attrition/predict`, `/v1/anomaly/attendance`, `/v1/anomaly/payroll`, `/v1/chat`, `/v1/resume/rank`, `/v1/payband/analyze`, `/v1/monitoring`. `/v1/forecast/headcount` returns 503 (module missing).

### Monitoring
PSI on 8 input features, twin-resume and pay-gap re-runs, OK/WATCH/ALERT status with exit codes. First run on 2026-11-20: WATCH.

### Documentation
`docs/model_cards/` (7 cards), `docs/final_audit/` (inventory, verification, evaluation results, service verification, release readiness, manager decision), `docs/architecture/`, `docs/reports/CORTEXHR_FINAL_REPORT.md`, `docs/demo/DEMO_SCRIPT.md`, `MODEL_CARD_TEMPLATE.md`.

### Testing
Task 17 API: 43 passed, 3 skipped. Task 18 monitoring: 106 passed. Final challenge guard: 6 passed. Earlier consolidated audit counts: 1,229 passed, 5 failed, 8 skipped, with Task 14 incomplete and Task 10/Task 17 limitations documented. Details: `docs/final_audit/FINAL_EVALUATION_RESULTS.md`.

### Fixed in this release
Task 3 stale test updated to the current report schema; Task 12 / Task 15 loaders and tests adopted from the Task 18 patch (real unlock-file format). Task 18 missing-data drift reporting now emits eight NOT_EVALUATED features, unlock-file salary hire records are filtered from attrition snapshots, duplicate hires are rejected, and monitoring test fixtures are hermetic.

### Limitations
Fictional data; injected-anomaly labels; attrition AUC about 0.71 and untested fairness; pay-band model reproduces historical pay gaps; scoring CSV lacks three monthly features and the API silently imputes them; `attendance_rate_3m` shows training-serving skew; chatbot artifact retrained; `load_benchmarks()` ignores `CORTEXHR_WORKBOOK`; Task 6/7 scripts overwrite tracked reports; Task 10 patch not applied (3 failing tests).

### Not Included
Headcount forecasting code and backtest; any remote deployment, Docker build or CI run; live-endpoint verification; real data; per-user authorisation; outcome-based performance monitoring; attrition fairness audit; human review of the 20 Task 7 explanations.

### Security
Verified locally: API-key auth (401), strict schemas, 422 on malformed, oversized and out-of-range input; secret scan of committed files found only placeholders. Remaining: shared keys, no per-user access control or audit trail found, no penetration test, sensitive salary data behind one key.

### Responsible AI
Human review required for every output. Fairness tested: resume twins (25 pairs, gap 0.000, small and gender-coded only) and pay gaps (measured -4.0% career gap, -1.5% women, mitigations ineffective). Attrition fairness not tested. See each model card.

### Future Work
Restore Task 16 and add a backtest; wire the completeness guard (branch `day-20-final-challenge`) into the API; fairness audit for attrition; per-user auth and audit logs; deploy and health-check remotely; fix the hardcoded workbook path; review the 20 explanations with a payroll clerk.
