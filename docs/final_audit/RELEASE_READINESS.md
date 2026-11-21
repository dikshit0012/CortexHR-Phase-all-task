# Release Readiness

## Verdict: **DEMONSTRATION RELEASE** (not production ready)
Suitable to present as an engineering portfolio project and as a decision-support demo on fictional data. Not suitable for real HR decisions.

| Category | Status | Evidence | Issues |
|---|---|---|---|
| Models | Warning | 6 models reproduced; attrition AUC 0.71; anomaly precision 0.38-0.49 | forecasting model unverifiable; chatbot artifact retrained |
| Data | Warning | all data fictional/synthetic; labels are injected anomalies | no real-world validation; scoring CSV lacks 3 features; `attendance_rate_3m` skew (PSI 1.47 vs training) |
| Evaluation | Pass with warnings | attrition 76/76 reproduced; Task 5/6/9 reports identical | Task 14 tests incomplete; some thresholds untuned |
| Fairness | **Fail for pay, gap for attrition** | pay model -4.0% career-gap and -1.5% gender controlled gaps, mitigations ineffective; resume twins 0.000 | attrition fairness untested; "Other" group n=11; inferred genders |
| Security | Warning | 401 without/bad key (local); strict schemas; 422 on malformed/oversized/out-of-range input | shared API keys, no per-user authorisation or audit trail found; secrets scan clean; no pen test |
| APIs | Warning | 11 endpoints; 6/7 models load locally | forecast 503; Task 17 tests missing; `/v1/chat/converse` not in repo app |
| Deployment | **Not verified** | `render.yaml`, Dockerfile exist | never deployed, built or health-checked remotely |
| Monitoring | Warning | monitor ran: WATCH; fairness OK | cron job never executed; no outcome-based performance monitoring |
| Documentation | Pass | 7 model cards, audit set, architecture, report, demo script, handover | forecast card absent by design |
| Testing | Warning | 1,229 passed, 5 failed, 8 skipped | Task 14 and Task 17 not run; 3 chatbot integration and 2 monitor tests fail |

## Why not higher
Production readiness would need real data validation, a fairness audit for attrition and pay, per-user access control, a deployed and health-checked service, executed monitoring, and the missing forecasting code.
