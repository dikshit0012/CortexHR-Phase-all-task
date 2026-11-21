# Service Verification

**All tests below were LOCAL**: the FastAPI app was exercised in-process with FastAPI's `TestClient` on 2026-11-20, with artifacts mounted from a local directory (attrition-v2 and pay bands regenerated locally; intent classifier and anomaly detectors rebuilt locally). **No deployed (Render) endpoint was tested, no Docker image was built, and no GitHub Actions run was executed.** Raw evidence: `docs/evidence/service_verification.json` and `verify_service_local.py`.

| Service | Endpoint | Test Method | Result | Verified |
|---|---|---|---|---|
| platform | `GET /v1/health` | local TestClient | HTTP 200 | local only |
| platform | `GET /v1/ready` | local TestClient (expected 503: forecast not loadable) | HTTP 503 | local only |
| security | `POST /v1/chat` | no key | HTTP 401 | local only |
| security | `POST /v1/chat` | bad key | HTTP 401 | local only |
| attrition | `POST /v1/attrition/predict` | 5 real scoring rows | HTTP 422 | local only |
| resume | `POST /v1/resume/rank` | 1 job x 6 resumes | HTTP 200 | local only |
| payband | `POST /v1/payband/analyze` | 1 real hire row | HTTP 200 | local only |
| attendance_anomaly | `POST /v1/anomaly/attendance` | 3 injected + 3 unlabelled rows | HTTP 200 | local only |
| payroll_anomaly | `POST /v1/anomaly/payroll` | 3 injected + 3 unlabelled rows | HTTP 200 | local only |
| chatbot | `POST /v1/chat` | How many casual leaves do I get in a year? | HTTP 200 | local only |
| chatbot | `POST /v1/chat` | what is the notice period policy | HTTP 200 | local only |
| chatbot | `POST /v1/chat` | tell me a joke | HTTP 200 | local only |
| forecast | `POST /v1/forecast/headcount` | expected unavailable (Task 16 code missing) | HTTP 503 | blocked |
| security | `POST /v1/chat` | malformed JSON | HTTP 422 | local only |
| security | `POST /v1/chat` | 200 KB body | HTTP 422 | local only |
| security | `POST /v1/payband/analyze` | out-of-range job_level | HTTP 422 | local only |
| security | `POST /v1/attrition/predict` | empty batch | HTTP 422 | local only |
| monitoring | `GET /v1/monitoring` | no report published yet | HTTP 404 | local only |
| platform | `GET /docs` | docs enabled in development | HTTP 200 | local only |

## Notes on the results
* `GET /v1/ready` returned 503 because `headcount_forecast` cannot load (`No module named 'cortexhr_task16'`); `/v1/health` still returned 200 listing 6 of 7 models available. This is the designed behaviour.
* The first attrition call returned 422 because the scoring CSV contains `dept_attrition_rate_prior_12m`, an excluded column; the strict schema rejects unknown fields. With only the model's 33 features the call returned 200 (25 records), `model_version: attrition-v2`, probabilities 0.019-0.100 for the first five rows, and rank-based bands (needs at least 10 records).
* The API accepted those records **without** `attendance_rate_change_3m`, `avg_unpaid_absence_days_3m` and `avg_variable_pay_3m` (median-imputed silently). See the Final Challenge.
* `POST /v1/payband/analyze` returns the midpoint only: `band_status: unavailable: calibration file not configured` (low/high are null).
* The official sample `sample_attrition_risk` (model `attrition-logre...`, first row risk 0.035) is a different, earlier model version and cannot be compared one-to-one with `attrition-v2` (0.0997 for the same first record); equality was **not** claimed.
* Chat: 'How many casual leaves do I get in a year?' -> `hr_policy_leave` (0.953); notice-period question -> `notice_period_policy` (0.927); 'tell me a joke' -> `out_of_scope` fallback. A different question earlier fell back with confidence 0.545 (below the 0.64 threshold).
* The Task 17 test file could not be collected (its conftest and other Task 17 tests were not supplied), so Task 17 has no passing automated test run in this audit.
