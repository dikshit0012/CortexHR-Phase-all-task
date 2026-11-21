# Task 19, Phase 1: API contract audit

Source of truth: the current Task 17 source (`phase-4/task17/cortexhr_api/`) and the deployed Vercel API. The Task 18 monitoring routes described in older drafts are not present in the current Task 17 router; the live `GET /v1/monitoring` returns 404.

| Feature | Endpoint | Request | Response (fields the console uses) | Auth | Status |
|---|---|---|---|---|---|
| Health | `GET /v1/health` | none | `application, api_version, uptime_seconds, environment, models_available, models_unavailable, models{status,version,mode}` | public | live 200; only `resume_ranking` is loaded |
| Readiness | `GET /v1/ready` | none | `ready, blocking, degraded` (503 when not ready, body still returned) | public | live 503; required model artifacts/resources are missing |
| Attrition risk | `POST /v1/attrition/predict` | `{records:[{record_ref?, 33 feature fields}]}` (1–100) | `model_version, n_records, risk_band_note, results[{record_ref, attrition_probability_12m, risk_band}]` | X-API-Key | route exists; live model unavailable (artifact not deployed) |
| Attendance anomaly | `POST /v1/anomaly/attendance` | `{records:[AttendanceRecord ×16 fields]}` | `model_version, n_records, n_flagged, results[{result_id, employee_id, period, anomaly_score, flagged Y/N, reason_code, reason_text, money_at_risk}]` | X-API-Key | live model unavailable; private reference workbook is not configured |
| Payroll anomaly | `POST /v1/anomaly/payroll` | `{records:[PayrollRecord ×24 fields]}` | same as above | X-API-Key | same 503 blocker |
| HR chatbot | `POST /v1/chat` | `{message ≤500, session_id?}` (`extra=forbid`) | `reply, intent, confidence, fallback, fallback_reason, faq_ids, session_id, model_version` | X-API-Key | route exists; live model unavailable (artifact not deployed) |
| Resume ranking | `POST /v1/resume/rank` | `{job:{job_title, required_skills, nice_to_have_skills, min_experience_years, description_text}, resumes:[{resume_id, resume_text ≤20k}] (1–50)}` | `method, weights, fairness_note, n_resumes, ranking[{rank, resume_id, final_score, tfidf_similarity, skill_overlap, matched/missing/preferred skills, experience_years, meets_min_experience}]` | X-API-Key | loaded in live health; protected inference not tested because no key was used |
| Pay band | `POST /v1/payband/analyze` | `{job_level 1–7, prior_experience_years, market_index, role_id, department_id, work_location, highest_education}` | `predicted_salary_mid, recommended_salary_low/high, band_status, comparison, features_used, fairness_note, decision_note` | X-API-Key | route exists; live model unavailable (joblibs not deployed) |
| Forecast | `POST /v1/forecast/headcount` | `{department_id (ALL, D01–D10), horizon_months 1–6}` | `mode live/precomputed, trained_through, method, interval, forecast[{month, expected_headcount, lower_80, upper_80}], event_adjustments, notes` | X-API-Key | precomputed file is outside the Vercel project root; model unavailable |
| Monitoring (Task 18) read | `GET /v1/monitoring` | none | Planned aggregate report contract | X-API-Key | **not implemented in current Task 17 source; live API returns 404** |
| Monitoring publish | `PUT /v1/monitoring/report` | `latest_report.json` from `monitor.py` | Planned whitelisted aggregate storage | **X-Monitoring-Publish-Key** (write-only) | **not implemented; workflow publication is blocked** |

Error format (all routes): `{"error": {"code", "message", "request_id"}}`; codes seen live: 400 `batch_too_large`/`invalid_records`/`no_usable_text`,
401 `unauthorized`, 404 `unknown_department`/`not_found`, 413 `payload_too_large`, 422 `validation_error`, 429 `rate_limited` (+`Retry-After`),
503 `model_unavailable`/`auth_not_configured`; 500 `internal_error`.

## Gaps and findings
1. **Monitoring is not integrated into Task 17.** Task 19's client expects Task 18 routes, but the current API source and live deployment do not provide them. A generic 404 is now shown as a missing endpoint, not as a successful "no report" state. Durable storage and a separate write-only publish key are still required before enabling publication.
2. **Anomaly detection cannot run without the private reference workbook.** Verified only: validation, auth, honest 503. Real inference **not verified**.
3. **No historical headcount in the forecast response**, so the chart cannot draw an "actual" line (only forecast + 80% interval).
4. **No score explanations in the attrition response** (Task 14 has SHAP reason generation, but Task 17 does not expose it), so the console shows no reasons.
5. **No aggregate endpoints** (employees scored, high-risk count, anomaly totals): the Overview shows "Data unavailable" for those.
6. **The personalised assistant (Task 10 `/v1/chat/converse`) is not part of the API**: it exists only as `task8_main_py_task10.patch`. The console therefore offers FAQ answers only (no leave balance / payslip).
7. **The API has one shared key and no user identity.** There is no per-user authorization: whoever holds the key (the console) can use every capability. The console adds an access gate (`CORTEXHR_CONSOLE_PASSWORD`), but real per-user roles would need API-side identity.
8. **The "chat role-spoofing" flaw does not exist in Task 17**: `ChatRequest` has no role field and `extra=forbid` rejects it with 422 (contract-tested; not checked against the live API in this audit). It existed only in the kit skeleton `main.py`.
9. The chat reply for out-of-scope questions is a generic fallback; intent accuracy is modest (Task 17 report).
10. Rate limiting is per client IP: all console users share the console server's IP, so heavy use can hit 429 (handled with a friendly message).
