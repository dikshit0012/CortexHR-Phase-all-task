# Service Verification

**Historical snapshot (2026-11-20).** All tests below were local: the FastAPI app was exercised in-process with FastAPI's `TestClient`, with artifacts mounted from a local directory. No deployed (Render) endpoint was tested, no Docker image was built, and no GitHub Actions run was executed. Raw request/response evidence is intentionally omitted because it contained per-record identifiers and model outputs; this summary retains only endpoint and status information.

| Service | Endpoint | Test Method | Result | Verified |
|---|---|---|---|---|
| platform | `GET /v1/health` | local TestClient | HTTP 200 | local only |
| platform | `GET /v1/ready` | local TestClient (expected 503: forecast not loadable) | HTTP 503 | local only |
| security | `POST /v1/chat` | no key | HTTP 401 | local only |
| security | `POST /v1/chat` | bad key | HTTP 401 | local only |
| attrition | `POST /v1/attrition/predict` | representative request | HTTP 422 | local only |
| resume | `POST /v1/resume/rank` | representative request | HTTP 200 | local only |
| payband | `POST /v1/payband/analyze` | representative request | HTTP 200 | local only |
| attendance_anomaly | `POST /v1/anomaly/attendance` | representative batch | HTTP 200 | local only |
| payroll_anomaly | `POST /v1/anomaly/payroll` | representative batch | HTTP 200 | local only |
| chatbot | `POST /v1/chat` | How many casual leaves do I get in a year? | HTTP 200 | local only |
| chatbot | `POST /v1/chat` | what is the notice period policy | HTTP 200 | local only |
| chatbot | `POST /v1/chat` | tell me a joke | HTTP 200 | local only |
| forecast | `POST /v1/forecast/headcount` | Task 16 code was not present during this audit snapshot | HTTP 503 | historical |
| security | `POST /v1/chat` | malformed JSON | HTTP 422 | local only |
| security | `POST /v1/chat` | 200 KB body | HTTP 422 | local only |
| security | `POST /v1/payband/analyze` | out-of-range job_level | HTTP 422 | local only |
| security | `POST /v1/attrition/predict` | empty batch | HTTP 422 | local only |
| monitoring | `GET /v1/monitoring` | no report published yet | HTTP 404 | local only |
| platform | `GET /docs` | docs enabled in development | HTTP 200 | local only |

## Notes on the results
* `GET /v1/ready` returned 503 because the Task 16 forecast module was not present during this audit; `/v1/health` still returned 200. This is historical and does not describe the current repository layout.
* A strict-schema validation test returned 422 for an excluded input field. Detailed predictions and per-record comparisons are intentionally not published.
* The API accepted records without three monthly-derived features, which are median-imputed; the Task 20 Final Challenge discusses this limitation and a review-flagging alternative.
* `POST /v1/payband/analyze` returned midpoint-only output because no calibration file was configured.
* Chat: 'How many casual leaves do I get in a year?' -> `hr_policy_leave` (0.953); notice-period question -> `notice_period_policy` (0.927); 'tell me a joke' -> `out_of_scope` fallback. A different question earlier fell back with confidence 0.545 (below the 0.64 threshold).
* Task 17 source and tests were added in the later consolidation; see `CONSOLIDATION_UPDATE.md` for current test results.
