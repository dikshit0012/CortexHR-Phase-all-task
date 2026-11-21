# Task 17 report – Release to production: one CortexHR API

Everything below is a measured result from this work session unless it says "not verified". **A partial Vercel deployment exists**: health returns 200, readiness returns 503, and only resume ranking is loaded. Historical local model tests below do not describe the current hosted availability.

## A. Introduction
A model that only runs in a notebook cannot help an HR team. Releasing it means a stable interface, validated inputs, controlled access, monitoring and a repeatable deployment. This task combines the seven CortexHR capabilities behind one versioned API and states clearly what works and what does not yet.

## B. System architecture
One FastAPI application (`cortexhr_api`) with `/v1` routers, a `ModelRegistry` filled once during startup (FastAPI lifespan), middleware for request ids, body-size limit, rate limiting and security headers, and a static dashboard at `/app/`. Upstream project code from Phases 1-3 and Task 16 is imported from the repository rather than copied, so each capability keeps its original preprocessing and feature engineering.

## C. Model integration
| Capability | Endpoint | Real artifact / code | Local result |
|---|---|---|---|
| Attrition | `POST /v1/attrition/predict` | `attrition-v2.joblib` | API output equals direct `predict_proba` (tested to 1e-12) |
| Attendance anomaly | `POST /v1/anomaly/attendance` | Task 5-7 + `attendance-anomaly-v1` | **not exercised**: needs private workbook; returns 503 |
| Payroll anomaly | `POST /v1/anomaly/payroll` | Task 6-7 + `payroll-anomaly-v1` | **not exercised**: same |
| HR chatbot | `POST /v1/chat` | Task 9 service, retrained intent model | FAQ answered; salary-of-others questions fall back |
| Resume ranking | `POST /v1/resume/rank` | Task 11 pipeline | relevant resume ranked first with matched/missing skills |
| Pay-band | `POST /v1/payband/analyze` | `paybands-v1` gradient boosting (+2 comparison models) | midpoint returned; band only with calibration file |
| Headcount forecast | `POST /v1/forecast/headcount` | Task 16 code, live fit at startup | reproduces the committed Task 16 forecast exactly (ALL, D01, D05 checked) |

So: **five of seven capabilities run real inference; two (anomaly) are integrated but unverified for inference.**

## D. API design
Everything is under `/v1`. Requests use Pydantic models with `extra=forbid`, numeric bounds (sanity guards, not tuned to training data), string length limits and date validation (via the Task 8 schemas). Missing optional values are allowed because the saved pipelines impute them. Responses carry a `request_id`; all errors share `{"error": {"code", "message", "request_id"}}`. Codes used: 200, 400 (batch too large/invalid records), 401, 404 (unknown department), 413, 422, 429, 500 (generic), 503 (model unavailable / auth not configured).

## E. Model loading
`ModelRegistry.load_all` runs once in the lifespan, one `try` per capability, recording status, version, mode, trained-with vs running-with library versions and a safe reason on failure. A test confirms object identities do not change across requests, and another that an empty artifact directory yields `unavailable` models, 503 responses and a still-healthy process. Startup on the exact production environment (Python 3.14.6) was ready in about 5 s including the live forecast fit (model selection by rolling-origin CV runs once, never per request).

## F. Security
API key (constant-time compare, fails closed), secrets only via environment/secret files, no values echoed in errors, whitelisted logs (a test asserts a record reference and a salary never reach the logs), protected attributes rejected by schema, CORS allow-list, 1 MB body limit, rate limit, security headers, no tracebacks or paths in responses (tested). Secret scan of files and full git history: nothing found. The chatbot answers only from an approved FAQ list and has no access to employee records.

## G. Hosting evaluation
See the README table (Render, Railway, Hugging Face Spaces, Google Cloud Run). **Selected: Render**, Free for demonstration and Starter ($7/month) for real use, because it deploys a Dockerfile from GitHub with a Blueprint, provides HTTPS, dashboard secrets and secret files. Two findings from the official docs drive the setup: Free has 0.1 CPU and sleeps after 15 minutes of inactivity; secret files are limited to 1 MB combined, so the master workbook needs a disk or the signed-URL download. Cloud Run figures could not be verified in this session and are marked as such.

## H. Deployment
**Not performed.** No hosting or GitHub credentials/network access were available. Prepared: `render.yaml`, `Dockerfile` (exact Python 3.14.6 via uv), `.dockerignore`, `.env.example`, branch `phase-4-task-17` (committed locally), and step-by-step instructions in the README. Public URL: none.

## I. Testing
Automated (pytest, FastAPI TestClient, real artifacts unless marked `mocked`):

| Environment | Private data | Result |
|---|---|---|
| Python 3.14.6, exact `requirements.txt` (clean venv) | headcount CSV | **45 passed, 1 skipped** |
| Python 3.14.6, exact `requirements.txt` | none | **43 passed, 3 skipped** |
| Python 3.12 sandbox | none | **43 passed, 3 skipped** |

Skipped tests: real anomaly inference (needs the workbook), and two live-forecast tests (need the history CSV; the precomputed mode is tested instead). One failure occurred during development (a wrong regex in a new test) and was fixed. During development two real defects were found by the tests and fixed: optional fields were mistakenly required in the schemas, and the body-size middleware design was replaced.

Smoke test against a **local** uvicorn server (not the live URL) on Python 3.14.6:

```
Endpoint                    | Status | Response Time | Result
GET  /v1/health             |    200 |         13 ms | PASS: 5/7 models loaded
GET  /v1/ready              |    200 |          2 ms | PASS: ready (degraded: attendance_anomaly,payroll_anomaly)
POST /v1/attrition/predict  |    200 |         88 ms | PASS: p=0.199 (attrition-v2)
POST /v1/anomaly/attendance |    503 |          6 ms | SKIP: model unavailable on this server (not counted as pass)
POST /v1/anomaly/payroll    |    503 |          2 ms | SKIP: model unavailable on this server (not counted as pass)
POST /v1/chat               |    200 |          6 ms | PASS: intent=notice_period_policy conf=0.9755
POST /v1/resume/rank        |    200 |         15 ms | PASS: top=R-1 score=0.438 matched=Python,SQL
POST /v1/payband/analyze    |    200 |         15 ms | PASS: mid=1469931 band=n/a
POST /v1/forecast/headcount |    200 |          5 ms | PASS: live 2026-10-01..2027-03-01 first=444.4
TOTAL 9  PASSED 7  FAILED 0  SKIPPED 2  -> PASS WITH SKIPS (not all endpoints verified)
```
Run without `--allow-unavailable`, the same script reports 2 FAILED and OVERALL RESULT: FAIL, which is the honest status of the anomaly endpoints.

## J. Performance
Local only: one uvicorn worker, client on the same machine, `1` CPU(s) and `3992` MB RAM in the sandbox, mixed endpoints (attrition, chat, pay-band, forecast, resume), 6 requests per user, Python 3.14.6 environment:

| Concurrent users | Requests | OK | Errors | Timeouts | Avg ms | P95 ms | P99 ms | Req/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 6 | 0 | 0 | 36 | 103 | 103 | 27.7 |
| 10 | 60 | 60 | 0 | 0 | 203 | 693 | 749 | 45.1 |
| 25 | 150 | 150 | 0 | 0 | 567 | 1573 | 1950 | 42.9 |
| 50 | 300 | 300 | 0 | 0 | 1147 | 2164 | 2299 | 40.9 |
| 100 | 600 | 600 | 0 | 0 | 2349 | 3973 | 4571 | 40.1 |

Memory: 222 MB idle, 236 MB peak. **Maximum tested concurrency: 100 users, locally only, short burst (600 requests).** This is not evidence that a hosted free instance supports 100 users: Render Free has 0.1 CPU versus this sandbox, so latencies there will be far higher.

## K. Failure analysis (what breaks first at 100 simultaneous callers)
1. **CPU.** Throughput plateaus at about 40 requests/s while latency grows linearly with users (queueing), so CPU is the first limit, not memory. Free hosting CPU (0.1) makes this much worse (not measured).
2. **Heavy requests.** Resume ranking (TF-IDF per request) and large batches cost the most; a 1 MB body limit, batch caps (100 records, 50 resumes) and the rate limit bound the damage.
3. **Rate limiter.** 120 requests/min per client by default: a load test from one IP receives 429s (disabled for the local test above). It is per instance, not shared.
4. **Memory.** 236 MB peak against 512 MB on Render Free/Starter leaves limited headroom; running more workers multiplies it. The deployment uses one worker.
5. **Cold start.** A sleeping Free instance needs to rebuild/start; here startup took about 5 s on a fast local CPU (estimate for 0.1 CPU: not measured).
6. **Thread safety.** Prediction code paths only read loaded models; no failures or 500s occurred in 600 concurrent requests. Concurrency behaviour of the Task 8 service is not verified (not loadable here).
7. **Chat sessions.** None are stored; `session_id` is only echoed back.

## L. Limitations
Not deployed; Docker image not built; anomaly inference unverified; dashboard not tested in a browser; chatbot retrained rather than the original artifact; pay-band ranges need a calibration file; forecast horizon is capped at 6 months and applies no future events (none known); precomputed forecast mode is fixed output when the history file is absent; attrition artifact was trained on scikit-learn 1.8.0 and is run on 1.9.1 (identical predictions on 200 test rows).

## M. Conclusion
The source is a tested, secure-by-default multi-model API, but the current Vercel deployment is not ready for predictions. It needs the missing model artifacts and approved runtime resources; the private workbook is required for anomaly detection.

## Final audit
| Item | Status | Evidence |
|---|---|---|
| All seven functionalities integrated | partly | five verified; anomaly wired, unverified |
| Existing trained models reused | yes | six recovered artifacts + retrained chatbot (disclosed) |
| Models load once | yes | identity test, lifespan |
| All endpoints under `/v1` | yes | OpenAPI test |
| Pydantic schemas | yes | tests |
| Health / readiness | yes | tests, smoke |
| Swagger | yes | `/docs`, `/openapi.json` tested |
| Frontend calls the real API | partly | same-origin calls to real endpoints; not run in a browser |
| Env vars / no secrets committed | yes | scan of files and history |
| Requirements pinned, clean install | yes | Python 3.14.6 venv, tests pass |
| Docker build works | **no evidence** | no Docker daemon |
| GitHub deployment configured | yes (config only) | `render.yaml` |
| Public HTTPS URL | **partial deployment** | Vercel responds, but `/v1/ready` is 503; see the current-status note above |
| Live smoke test | **no** | local only |
| Automated tests pass | yes | counts above |
| Load testing documented | yes (local) | table above |
| Hosting comparison | yes | Cloud Run unverified |
| README / report | yes | this document |
