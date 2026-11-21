# Task 19, Phase 1: API contract audit

Source of truth: the **running** Task 17 API with the **Task 18 patch applied** (`cortexhr_api` 1.0.0, real artifacts), its `/openapi.json`, and the code in
`phase-4/task17/cortexhr_api/`. The console's forms are generated from that spec (`scripts/sync_contract.py` → `contract_snapshot.json`)
and `tests_integration/test_live_api.py::test_contract_snapshot_matches_live_openapi` fails if the API drifts.

| Feature | Endpoint | Request | Response (fields the console uses) | Auth | Status |
|---|---|---|---|---|---|
| Health | `GET /v1/health` | none | `application, api_version, uptime_seconds, environment, models_available, models_unavailable, models{status,version,mode}` | public | works |
| Readiness | `GET /v1/ready` | none | `ready, blocking, degraded` (503 when not ready, body still returned) | public | works |
| Attrition risk | `POST /v1/attrition/predict` | `{records:[{record_ref?, 33 feature fields}]}` (1–100) | `model_version, n_records, risk_band_note, results[{record_ref, attrition_probability_12m, risk_band}]` | X-API-Key | works (`risk_band` only for batches ≥ 10) |
| Attendance anomaly | `POST /v1/anomaly/attendance` | `{records:[AttendanceRecord ×16 fields]}` | `model_version, n_records, n_flagged, results[{result_id, employee_id, period, anomaly_score, flagged Y/N, reason_code, reason_text, money_at_risk}]` | X-API-Key | **503 unless the private reference workbook is configured** |
| Payroll anomaly | `POST /v1/anomaly/payroll` | `{records:[PayrollRecord ×24 fields]}` | same as above | X-API-Key | **same 503** |
| HR chatbot | `POST /v1/chat` | `{message ≤500, session_id?}` (`extra=forbid`) | `reply, intent, confidence, fallback, fallback_reason, faq_ids, session_id, model_version` | X-API-Key | works (FAQ only, no employee records) |
| Resume ranking | `POST /v1/resume/rank` | `{job:{job_title, required_skills, nice_to_have_skills, min_experience_years, description_text}, resumes:[{resume_id, resume_text ≤20k}] (1–50)}` | `method, weights, fairness_note, n_resumes, ranking[{rank, resume_id, final_score, tfidf_similarity, skill_overlap, matched/missing/preferred skills, experience_years, meets_min_experience}]` | X-API-Key | works (JSON text only, no file upload) |
| Pay band | `POST /v1/payband/analyze` | `{job_level 1–7, prior_experience_years, market_index, role_id, department_id, work_location, highest_education}` | `predicted_salary_mid, recommended_salary_low/high, band_status, comparison, features_used, fairness_note, decision_note` | X-API-Key | works (midpoint only; band needs calibration file) |
| Forecast | `POST /v1/forecast/headcount` | `{department_id (ALL, D01–D10), horizon_months 1–6}` | `mode live/precomputed, trained_through, method, interval, forecast[{month, expected_headcount, lower_80, upper_80}], event_adjustments, notes` | X-API-Key | works (currently `precomputed`) |
| Monitoring (Task 18) read | `GET /v1/monitoring` | none | `status OK/WATCH/ALERT, timestamp, execution_status, execution_ok, evidence_complete, drift{features_checked, ok, watch, alert, not_evaluated, error}, fairness{twin_resume{status}, pay_gap{status}}, reasons[{category, check, item, status}], last_run, report_age_hours, stale, request_id` | X-API-Key | works (404 until a report is published; 503 if unreadable) |
| Monitoring publish | `PUT /v1/monitoring/report` | the `latest_report.json` from `monitor.py` (≤1 MB) | `{stored, status, timestamp}`; 401/409 older/413/422 | **X-Monitoring-Publish-Key** (write-only, separate from read keys) | works; **used by the workflow, never by the console** |

Error format (all routes): `{"error": {"code", "message", "request_id"}}`; codes seen live: 400 `batch_too_large`/`invalid_records`/`no_usable_text`,
401 `unauthorized`, 404 `unknown_department`/`not_found`, 413 `payload_too_large`, 422 `validation_error`, 429 `rate_limited` (+`Retry-After`),
503 `model_unavailable`/`auth_not_configured`/`monitoring_report_invalid`/`monitoring_publish_not_configured`; 404 `monitoring_report_not_found`, 500 `internal_error`.

## Gaps and findings
1. **Monitoring is now served** (Task 18 patch). The API stores and returns **aggregates only**: drift *counts* and feature *names*, not per-feature PSI values or group data. The console cannot show PSI numbers because the API does not serve them. Reports older than 48 h are marked `stale`. On a free host the stored report is lost on spin-down until the next publish.
2. **Anomaly detection cannot run without the private reference workbook.** Verified only: validation, auth, honest 503. Real inference **not verified**.
3. **No historical headcount in the forecast response**, so the chart cannot draw an "actual" line (only forecast + 80% interval).
4. **No score explanations in the attrition response** (Task 14 has SHAP reason generation, but Task 17 does not expose it), so the console shows no reasons.
5. **No aggregate endpoints** (employees scored, high-risk count, anomaly totals): the Overview shows "Data unavailable" for those.
6. **The personalised assistant (Task 10 `/v1/chat/converse`) is not part of the API**: it exists only as `task8_main_py_task10.patch`. The console therefore offers FAQ answers only (no leave balance / payslip).
7. **The API has one shared key and no user identity.** There is no per-user authorization: whoever holds the key (the console) can use every capability. The console adds an access gate (`CORTEXHR_CONSOLE_PASSWORD`), but real per-user roles would need API-side identity.
8. **The "chat role-spoofing" flaw does not exist in Task 17**: `ChatRequest` has no role field and `extra=forbid` rejects it with 422 (tested live). It existed only in the kit skeleton `main.py`.
9. The chat reply for out-of-scope questions is a generic fallback; intent accuracy is modest (Task 17 report).
10. Rate limiting is per client IP: all console users share the console server's IP, so heavy use can hit 429 (handled with a friendly message).
