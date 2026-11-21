# Task 19 report: CortexHR Insights Console (final, with Task 18 monitoring)

Everything here is a measured result from this work session unless it says "not verified". **The Task 19 Streamlit console is not deployed; the Task 17 Vercel API is deployed but only partially available.**

## Re-audit update (20 Nov 2026)

| Requirement | Result | Evidence |
|---|---|---|
| Five tabs exist | PASS | Risks, Anomalies, Assistant, Resume Screener, Forecast; Overview, Pay Bands and Monitoring also exist |
| API URL from environment variable | PASS | `CORTEXHR_API_URL` is loaded centrally in `config.py` |
| Central API client | PASS | All HTTP calls go through `api_client.py` |
| AI review labels | PASS | Mocked UI tests verify labels on risk, anomaly, assistant, resume and forecast outputs |
| Error handling | PASS | Mocked tests cover API and friendly-error states |
| Automated tests | PASS | `108 passed` in `python -m pytest tests -q` |
| Live API URL | PARTIAL | Task 17 Vercel URL responds; health is 200, readiness is 503, and protected inference was not tested with a key |
| Deployed console | BLOCKED | No public URL or deployment credentials are available |
| Physical/mobile device test | BLOCKED | Streamlit served locally, but no physical phone or visual browser session was available |

Changes made in this re-audit:
- Fixed Windows/Python source decoding of the assistant example text.
- Fixed the Streamlit forecast and resume prompts to use ASCII quotation marks.
- Made the source-safety test read Python files explicitly as UTF-8.
- Added stable IDs for the large-data upload test parameters.
- Made the live connection-refused test accept both connection and timeout classifications on Windows.

## A. Verdict: PASS WITH WARNINGS (not deployment-verified)
The console unit/UI suite passes, but this repository's current Task 17 source and live Vercel API do not contain the Task 18 monitoring routes.
The current deployment is also not ready to serve most model-backed endpoints. Earlier local checks against a patched API do not establish that
the current source or deployed service has those routes. Warnings: no Streamlit deployment, no screenshots, no phone/tablet testing, anomaly
inference requires the private workbook, the GitHub schedule has not run, and the API uses a shared key rather than per-user authorization.

## B. Initial audit
**Available:** the Task 17 API routes in the current OpenAPI spec, one error format, API-key auth, rate limit, and body-size limit.
**Missing:** the Task 18 monitoring read/publish routes; the current live endpoint returns 404.
**Incorrect in my first console draft** (built from the kit skeleton before the real API was available): attrition by id lookup; a `user_role` in chat;
a health check expecting `status`; a generic monitoring renderer; no pay-band page. Core pages were aligned to the Task 17 contract; monitoring remains unavailable until Task 17 implements it.
**Missing in the API:** historical headcount, score explanations, aggregate counts, per-feature PSI values (the planned monitoring contract serves aggregates only, but its routes are absent), the
personalised assistant (Task 10 `/v1/chat/converse` exists only as an unapplied patch). See `api_contract_audit.md`.

## C. Architecture
`phase-4/task19/`: `app.py` → `views/` (8 pages) → `api_client.py` (the only HTTP module) → API. `config.py`, `contracts.py` + `contract_snapshot.json`
(generated from the live OpenAPI), `components/` (alerts, charts, uploads validation, gate, cache, sidebar, theme), `tests/` (mocked transport with
responses captured from the real API), `tests_integration/` (real API process).

## D. Current feature availability
| Feature | Current evidence | Current status |
|---|---|---|
| Attrition | Route exists; live health reports the model artifact missing | Unavailable on Vercel |
| Attendance/payroll anomalies | Live health reports the reference workbook is not configured | Unavailable; no live inference |
| HR Assistant | Live health reports the chatbot artifact missing | Unavailable on Vercel |
| Resume Screener | Live health reports `resume_ranking` loaded; no protected request was made | Model loaded; live inference not verified |
| Pay Bands | Live health reports the joblibs missing | Unavailable on Vercel |
| Workforce Forecast | Live health reports neither history nor precomputed forecast is available | Unavailable on Vercel |
| Monitoring | Current source lacks read/publish routes; live `GET /v1/monitoring` returns 404 | Not implemented |

Historical local integration results against a temporary Task 18 API patch are not evidence for the current source or deployment.

## E. Endpoint mapping
Current Task 17 source routes: `GET /v1/health`, `GET /v1/ready`, `POST /v1/attrition/predict`,
`POST /v1/anomaly/attendance`, `POST /v1/anomaly/payroll`, `POST /v1/chat`,
`POST /v1/resume/rank`, `POST /v1/payband/analyze`, and `POST /v1/forecast/headcount`.
The Task 18 monitoring read/publish routes are absent.

## F. UI/UX
Navy sidebar, API status + last-checked, Overview with a "Start here: Attrition Risks" banner, spinners, empty states, friendly errors with a support
reference id, human-review label beside every AI result, responsive CSS. **Not tested:** real desktop/tablet/phone rendering and screenshots (no browser
could be installed; the download is blocked). The usability questions have **not been tested with a real user**.

## G. Security
- Key/password only in Streamlit secrets; the client does not render them or log request bodies. The console has no publish method or publish key.
- No role is ever sent. **Correction to my earlier report:** the "role-spoofing fix" and `auth.py` I first delivered were written against the kit skeleton, **not your Task 17 API, and are withdrawn.** The chat schema has no role field; local contract tests reject `user_role`. This was not checked against the live API in this audit.
- The current API uses one shared key; there is no per-user authorization. Protected live requests were not tested in this audit.
- Upload validation, identity-column refusal, size guards, double-submit guard, password gate with lockout.
- **Residual risk:** shared API key; no per-user authorization; session-level lockout only (use SSO for real protection).

## H. Deployment
The Task 19 console is not deployed. The Task 17 Vercel API is deployed but only partially available: health 200, readiness 503, and monitoring 404.

## I. Screenshots
**None.** A headless `streamlit run` served the app over HTTP against the real API, which is not visual evidence. Capture the three evidence screenshots after deployment (Overview, Attrition Risks, Monitoring or Forecast).

## J. Testing
| Suite | Result |
|---|---|
| Task 19 unit/page tests (`tests/`) | **108 passed** |
| Task 17 full suite | **46 passed, 3 skipped** (private workbook/history required for the skipped tests) |
| Task 8 monorepo path test | **1 passed** |
| Dashboard JavaScript syntax | **Passed** (`node --check`) |
| Live API probes | Health 200; readiness 503; monitoring 404 |
| Docker image build | **Not verified**: Docker Desktop Linux engine was unavailable |

The Task 19 integration suite that exercises monitoring writes targets a separate Task 18-patched API; it was not run as a validation of the current deployment.

## K. Remaining problems
1. Deploy the console only after deciding how its API key and access gate will be managed; the console itself is not deployed.
2. Restore the required model artifacts and approved runtime inputs securely; do not put HR data or model binaries in Git.
3. Implement and deploy Task 18 monitoring routes with durable storage before enabling the scheduled publish workflow.
4. Provide the private reference workbook and headcount history through approved storage to enable anomaly/forecast inference.
5. Add API-side identity and authorization before treating the shared-key service as a production HR system.
6. Verify protected live inference, visual/mobile behavior, and Docker deployment after those prerequisites are met.

## L. Evaluation
The console unit/page tests pass and its monitoring client now distinguishes a missing route from an unpublished report. Deployment and most model-backed capabilities remain unavailable; this is not ready for HR decisions.

## Final design question: which tab first?
**Attrition Risks.** The Overview opens with a "Start here: Attrition Risks" banner and button. Anomalies, Forecast and Monitoring remain one click away; the monitoring view is unavailable until the API routes are deployed.
