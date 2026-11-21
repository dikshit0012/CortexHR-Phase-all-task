# CortexHR Release Notes

Format follows Keep a Changelog. **This is a demonstration release on fictional data. It is not approved for real HR decisions.**

## [Unreleased] - Repository consolidation (2026-11-21)

### Added
* Consolidated the Task 20 audit, model cards, final report, and standalone completeness-guard experiment into the existing repository.
* Added Task 16 headcount forecasting to Phase 4, with a six-month forecast, 80% intervals, and a leakage-aware holdout backtest.
* Added a headcount-forecast model card and a dated update distinguishing current repository status from the historical audit snapshot below.

### Verification
* Task 16: 54 passed.
* Task 17: 43 passed, 3 skipped.
* Task 18: 106 passed.
* Task 19: 105 passed.
* Task 20 completeness guard: 6 passed.

### Privacy
* Local workbooks, trained model binaries, and raw service-verification responses containing per-record identifiers or outputs are not included.

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
| Headcount forecast (Task 16) | 6-month headcount and exits with 80% intervals | Demonstration only; validate inputs and review forecasts |

### APIs
Service verification was local only. The Task 17 API and Task 16 forecasting source are now present; forecast availability still depends on configuring the required local data and artifacts. No remote deployment is claimed.

### Monitoring
PSI on 8 input features, twin-resume and pay-gap re-runs, OK/WATCH/ALERT status with exit codes. First run on 2026-11-20: WATCH.

### Documentation
`docs/model_cards/` (7 cards), `docs/final_audit/` (inventory, verification, evaluation results, service verification, release readiness, manager decision), `docs/architecture/`, `docs/reports/CORTEXHR_FINAL_REPORT.md`, `docs/demo/DEMO_SCRIPT.md`, `MODEL_CARD_TEMPLATE.md`.

### Testing
Task 16: 54 passed. Task 17 API: 43 passed, 3 skipped. Task 18 monitoring: 106 passed. Task 19 console: 105 passed. Final challenge guard: 6 passed. Earlier consolidated audit counts are a historical snapshot: 1,229 passed, 5 failed, 8 skipped, with Task 14 incomplete and Task 10 limitations documented. Details: `docs/final_audit/FINAL_EVALUATION_RESULTS.md` and `docs/final_audit/CONSOLIDATION_UPDATE.md`.

### Fixed in this release
Task 3 stale test updated to the current report schema; Task 12 / Task 15 loaders and tests adopted from the Task 18 patch (real unlock-file format). Task 18 missing-data drift reporting now emits eight NOT_EVALUATED features, unlock-file salary hire records are filtered from attrition snapshots, duplicate hires are rejected, and monitoring test fixtures are hermetic.

### Limitations
Fictional data; injected-anomaly labels; attrition AUC about 0.71 and untested fairness; pay-band model reproduces historical pay gaps; scoring CSV lacks three monthly features and the API silently imputes them; `attendance_rate_3m` shows training-serving skew; chatbot artifact retrained; `load_benchmarks()` ignores `CORTEXHR_WORKBOOK`; Task 6/7 scripts overwrite tracked reports; Task 10 patch not applied (3 failing tests).

### Not Included
Any remote deployment, Docker build or CI run; live-endpoint verification; real data; per-user authorisation; outcome-based performance monitoring; attrition fairness audit; human review of the 20 Task 7 explanations.

### Security
Verified locally: API-key auth (401), strict schemas, 422 on malformed, oversized and out-of-range input; secret scan of committed files found only placeholders. Remaining: shared keys, no per-user access control or audit trail found, no penetration test, sensitive salary data behind one key.

### Responsible AI
Human review required for every output. Fairness tested: resume twins (25 pairs, gap 0.000, small and gender-coded only) and pay gaps (measured -4.0% career gap, -1.5% women, mitigations ineffective). Attrition fairness not tested. See each model card.

### Future Work
Wire the completeness guard into the API; fairness audit for attrition; per-user auth and audit logs; deploy and health-check remotely; fix the hardcoded workbook path; review the 20 explanations with a payroll clerk.
