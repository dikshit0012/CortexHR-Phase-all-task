# CortexHR - Final Report

## 1. Executive summary
CortexHR is an AI-powered HR analytics demonstration built over 20 days: attrition prediction, resume ranking, attendance and payroll anomaly detection, an HR chatbot, pay-band recommendation, one API, monitoring and a console. On Day 20 I audited it end to end. Six components reproduced their reported results (the attrition model matched 76 of 76 metrics). The audit also found real problems, listed below. **Verdict: a good portfolio project; not ready for real HR use.**

## 2. Introduction
Fictional company data (Zenith Nova Technologies). The goal was to learn the full ML engineering cycle, not to automate HR decisions.

## 3. Problem statement
HR teams face scattered signals: who may leave, which payslips or attendance records look wrong, which resumes fit a role, what a role should pay. Each needs support that is explainable, tested and reviewed by a person.

## 4. Objectives
Build and evaluate models with proper baselines; guard against leakage; test fairness; expose models safely through an API; monitor drift and fairness; document honestly.

## 5. System architecture
See `docs/architecture/CORTEXHR_ARCHITECTURE.md`: private data -> cleaning and leakage checks -> models -> API (`/v1`) -> human review, with a separate monitor.

## 6. Technology stack
Python 3.12, pandas, scikit-learn (1.8-1.9), LightGBM (challenger), SHAP, FastAPI, Streamlit, GitHub Actions, Docker/Render configs, pytest.

## 7. Data pipeline
Task 1 audits the master table; Task 2 builds the feature pipeline; Task 13 rejects leakage columns and rebuilds fixed folds. Data is gitignored and supplied locally. All data is fictional; anomaly labels are injected.

## 8-9. AI/ML components and model summary
| Component | Method | Key verified result |
|---|---|---|
| Attrition-v2 | calibrated Logistic Regression | ROC-AUC 0.711, P@top10% 0.33, lift 2.26 |
| Resume ranker | TF-IDF 0.6 + skills 0.4 (not trained) | P@5 0.92, NDCG@5 0.912 |
| Attendance anomaly | Isolation Forest OR rules | P 0.49, R 0.83 |
| Payroll anomaly | Isolation Forest OR rules | P 0.38, R 0.97 |
| Intent classifier | TF-IDF + Logistic Regression | accuracy 0.80, macro-F1 0.80 |
| Pay band | Gradient Boosting | MAE 60,801, R2 0.978 |
| Headcount forecast | Task 16 statistical models with 80% intervals | Six-month holdout WAPE 1.08%; results are limited by one fictional holdout window |

## 10. Evaluation results
All in `docs/final_audit/FINAL_EVALUATION_RESULTS.md`. Highlights: attrition reproduced exactly; Task 5 and Task 6 reports identical to committed; 1,229 tests passed, 5 failed, 8 skipped; Task 14 and 17 suites not completed.

## 11. Fairness and responsible AI
Resume twins: gap 0.000 over 25 pairs (but small, gender-coded, names inferred). Pay model: controlled historical gaps of -4.0% (12+ month career gap) and -1.5% (women); the model does not use those fields yet still differs by group, and two mitigations did not help. Attrition fairness was never tested. Every output needs human review.

## 12. API and deployment
11 endpoints in one FastAPI app; 6 of 7 models load locally; forecast returns 503. Only local verification was done; no remote deployment was tested.

## 13. Monitoring
`monitor.py` on 2026-11-20: WATCH. `attendance_rate_3m` PSI 0.039 vs matched reference but 1.467 vs full training data (training-serving skew); fairness checks OK. The GitHub Actions cron job was not executed here.

## 14. Security
Auth, schema validation and size limits behaved as designed locally. Open: shared keys, no per-user authorisation or audit trail found, salary data sensitivity, no penetration test.

## 15. Development journey
Phase 1 audited data and built baselines; Phase 2 built anomaly detection, explanations and the chatbot; Phase 3 built attrition v2, explanations, resume fairness and pay bands; Phase 4 added the API, monitoring and console; Day 20 audited everything.

## 16. Major challenges
Corrupted binary uploads; tests tied to the author's machine; synthetic labels; a modest attrition signal; fairness gaps that mitigation did not remove.

## 17. Solutions implemented
Deterministic regeneration of the artifact (verified identical metrics); test and loader fixes; a completeness-guard experiment; honest model cards and a manager decision.

## 18. Limitations
Fictional data; injected labels; no real deployment; forecast missing; attrition fairness untested; pay-model gaps; scoring CSV incomplete (3 features) and the API imputes silently (Final Challenge: aggregate AUC unchanged 0.7103 -> 0.7107, but 12% of the top-10% list changes).

## 19. Future scope
Restore Task 16; integrate the guard; audit attrition fairness; fix pay-model data; per-user auth; deploy and monitor remotely; hermetic tests.

## 20. Conclusion
CortexHR shows end-to-end ML engineering with unusually honest evaluation. Presented as a portfolio project: **yes**. Used for real HR decisions: **no**.

## 21. GitHub repository
https://github.com/dikshit0012/CortexHR-Phase-all-task. The existing `main` branch contains Phases 1-4 and the Task 20 release materials.

## 22. Release information
Version `v1.0.0` (demonstration release); see `RELEASE_NOTES.md`.

## Post-audit consolidation (2026-11-21)
The verified repository now includes Phase 4 Tasks 16-19 and the Task 20 release materials. Task 16 forecasting and its holdout backtest are present; its focused suite passed 54 tests. The Phase 4 suites passed: Task 17 (43 passed, 3 skipped), Task 18 (106 passed), and Task 19 (105 passed). The final-challenge guard passed 6 tests. The findings above describe the 2026-11-20 audit snapshot and should be read with the [dated consolidation update](../final_audit/CONSOLIDATION_UPDATE.md). No remote deployment or production readiness is claimed.
