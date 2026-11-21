# Task 19 report: CortexHR Insights Console (final, with Task 18 monitoring)

Everything here is a measured result from this work session unless it says "not verified". **Nothing is deployed.**

## Re-audit update (20 Nov 2026)

| Requirement | Result | Evidence |
|---|---|---|
| Five tabs exist | PASS | Risks, Anomalies, Assistant, Resume Screener, Forecast; Overview, Pay Bands and Monitoring also exist |
| API URL from environment variable | PASS | `CORTEXHR_API_URL` is loaded centrally in `config.py` |
| Central API client | PASS | All HTTP calls go through `api_client.py` |
| AI review labels | PASS | Mocked UI tests verify labels on risk, anomaly, assistant, resume and forecast outputs |
| Error handling | PASS | Mocked tests cover API and friendly-error states |
| Automated tests | PASS | `105 passed` in `python -m pytest tests -q` |
| Live API URL | BLOCKED | No `CORTEXHR_API_URL`, API key, or test API URL is configured |
| Deployed console | BLOCKED | No public URL or deployment credentials are available |
| Physical/mobile device test | BLOCKED | Streamlit served locally, but no physical phone or visual browser session was available |

Changes made in this re-audit:
- Fixed Windows/Python source decoding of the assistant example text.
- Fixed the Streamlit forecast and resume prompts to use ASCII quotation marks.
- Made the source-safety test read Python files explicitly as UTF-8.
- Added stable IDs for the large-data upload test parameters.
- Made the live connection-refused test accept both connection and timeout classifications on Windows.

## A. Verdict: PASS WITH WARNINGS (not deployment-verified)
All eight pages are built against the real Task 17 API **with the Task 18 patch applied** and were verified locally against the running API with its real
model artifacts. Warnings: no deployment, no screenshots, no phone/tablet testing, anomaly inference not verified (needs the private workbook), the
GitHub schedule was never run (Task 18's own caveat), and no per-user authorization (shared API key).

## B. Initial audit
**Available:** a unified API (Task 17, 9 routes) plus the Task 18 patch (`GET /v1/monitoring`, `PUT /v1/monitoring/report`), one error format,
API-key auth, rate limit, body limit.
**Incorrect in my first console draft** (built from the kit skeleton before the real API was available): attrition by id lookup; a `user_role` in chat;
a health check expecting `status`; a generic monitoring renderer; no pay-band page. All rewritten against the live spec.
**Missing in the API:** historical headcount, score explanations, aggregate counts, per-feature PSI values (monitoring serves aggregates only), the
personalised assistant (Task 10 `/v1/chat/converse` exists only as an unapplied patch). See `api_contract_audit.md`.

## C. Architecture
`phase-4/task19/`: `app.py` → `views/` (8 pages) → `api_client.py` (the only HTTP module) → API. `config.py`, `contracts.py` + `contract_snapshot.json`
(generated from the live OpenAPI), `components/` (alerts, charts, uploads validation, gate, cache, sidebar, theme), `tests/` (mocked transport with
responses captured from the real API), `tests_integration/` (real API process).

## D. Feature verification (local, real API, real artifacts)
| Feature | API connected | Verified live | Status |
|---|---|---|---|
| Attrition Risks (single + CSV batch) | yes | Console output equals direct API output; bands 10%/20%/70% on a batch of 20; deterministic; validation, protected-field rejection, batch limit | **Verified locally** |
| Anomalies: attendance, payroll | yes | Validation + auth + honest 503. **Real inference not verified** (private workbook) | **Integrated, unverified** |
| HR Assistant | yes | FAQ answered; salary-of-another-person falls back with no figures; `user_role` rejected 422 | **Verified locally** |
| Resume Screener | yes | Relevant candidate first; identity text does not change ranking; bad input rejected; file names never sent | **Verified locally** |
| Pay Bands (extra) | yes | Console = direct API; higher level → higher midpoint | **Verified locally** |
| Workforce Forecast | yes | Console table = direct API for ALL and D03; 11 departments; interval ordered and widening | **Verified locally** (`precomputed` mode) |
| **Monitoring (Task 18)** | yes | Six real `monitor.py` reports (OK, WATCH, drift ALERT, fairness ALERT, missing-evidence ALERT, stale) published with the real write key and read back by client and pages; Task 18's own `verify_live.py` returned VERIFIED | **Verified locally** |

### Monitoring details
- Overall status OK / WATCH / ALERT exactly as the report states; drift counts + a chart; names of drifting features; twin-resume (Task 12) and pay-gap (Task 15) status; last completed run and report age.
- A **stale** report (older than the API's limit, default 48 h) is shown as "OK · outdated", never as plainly healthy.
- A **failed run** (`execution_ok=false`) and **missing evidence** (`evidence_complete=false`, checks `NOT_EVALUATED`) are called out: a check that did not run is not a pass.
- No report yet → clear message, no invented status. Unreadable report → fixed message plus support reference.
- The console is read-only: it never runs monitoring and never has the publish key (tested by scanning the app source and by the client having no write method).
- Limitation: the API serves **aggregates only**, so per-feature PSI numbers are not shown. They are in the report artifact / `latest_report.md`.

## E. Endpoint mapping
`GET /v1/health`, `GET /v1/ready`, `POST /v1/attrition/predict`, `POST /v1/anomaly/attendance`, `POST /v1/anomaly/payroll`, `POST /v1/chat`,
`POST /v1/resume/rank`, `POST /v1/payband/analyze`, `POST /v1/forecast/headcount`, `GET /v1/monitoring`. (`PUT /v1/monitoring/report` exists but is used only by the workflow.)

## F. UI/UX
Navy sidebar, API status + last-checked, Overview with a "Start here: Attrition Risks" banner, spinners, empty states, friendly errors with a support
reference id, human-review label beside every AI result, responsive CSS. **Not tested:** real desktop/tablet/phone rendering and screenshots (no browser
could be installed; the download is blocked). The usability questions have **not been tested with a real user**.

## G. Security
- Key/password only in secrets; never rendered (live test scans every page); not in logs (test); not in the repo (scan). The publish key is never used by the console.
- No role is ever sent. **Correction to my earlier report:** the "role-spoofing fix" and `auth.py` I first delivered were written against the kit skeleton, **not your Task 17 API, and are withdrawn.** The real chat has no role field and no employee data; `user_role` is rejected 422 (live test).
- Monitoring write path (checked live): no key / wrong key / the *read* key all get 401; bad JSON, bad status and future dates 422; an older report 409 and does not overwrite; >1 MB 413; the stored file contains only whitelisted aggregates.
- Upload validation, identity-column refusal, size guards, double-submit guard, password gate with lockout.
- **Residual risk:** shared API key; no per-user authorization; session-level lockout only (use SSO for real protection).

## H. Deployment
**Not deployed.** No GitHub/Streamlit/Render credentials. Steps are in the README and in `task18/docs/GITHUB_SETUP.md`. No API URL exists (Task 17: not deployed).

## I. Screenshots
**None.** A headless `streamlit run` served the app over HTTP against the real API, which is not visual evidence. Capture the three evidence screenshots after deployment (Overview, Attrition Risks, Monitoring or Forecast).

## J. Testing
| Suite | Result |
|---|---|
| Console unit + page tests (`tests/`; HTTP mocked with payloads captured from the real API) | **105 passed** |
| Console live tests (`tests_integration/`; real patched API with real artifacts, 3 server instances) | **73 passed** (50 core + 24 monitoring − 1 superseded) |
| Mutation checks (a wrong expectation must fail against the live API) | confirmed twice |
| Task 18 `verify_live.py` against the patched API | VERIFIED |
| Existing project suites on the patched repo, code unmodified by me | |
| Task 1 | 4 passed (needs `PYTHONPATH=src`) |
| Task 2 | 56 passed, 4 skipped |
| Task 3 | 13 passed, **1 failed** (pre-existing: test expects column `Model`, code returns `model`) |
| Task 4 | 99 passed, 11 skipped |
| Task 5 | 17 passed (run `pytest tests`; from the task root a duplicate test module in `files (1)/` collides) |
| Task 6 | 94 passed, 2 skipped |
| Task 7 | 130 passed, 2 skipped |
| Task 8 | 77 passed, 3 skipped |
| Task 9 | 157 passed |
| Task 10 | 5 errors as shipped (artifact `hr_intent_classifier.joblib` missing); after rebuilding it in my sandbox: 172 passed, 4 skipped, **3 failed** (the Task 10 patch for the Task 8 app is not applied) |
| Task 11 | 47 passed, 25 skipped |
| Tasks 12, 13, 14 | 1 skipped each: **effectively unverifiable here** (private data) |
| Task 15 | 9 passed, 24 skipped (was 8 before the Task 18 patch) |
| Task 16 | 40 passed, 14 skipped |
| Task 17 (with Task 18 patch) | **75 passed**, 3 skipped (was 43; +32 monitoring API tests) |
| Task 18 | **106 passed** |
The Task 18 patch applied cleanly except its `.gitignore` hunk (my copy of the repo differs there); I added those three lines by hand. I did not change, delete or weaken any existing test or code. Monitoring reports used in the tests come from running your real `monitor.py` on **synthetic** data (twin/pay-gap results for the healthy scenarios use Task 18's own test helpers because the real fairness data is private); their timestamps are rewritten to "now" by the tests.

## K. Remaining problems
1. **Deploy** the API and the console, verify with real URLs, take screenshots, test on a phone. (Needs your accounts.)
2. **Run the Task 18 workflow on GitHub** (secrets, merge to the default branch, `schedule` evidence). Never executed here.
3. **Render free plan wipes the stored report** on every spin-down: the console will then say "no report published" until the next publish. Use a persistent disk or `CORTEXHR_MONITORING_REPORT_URL`.
4. Provide the private reference workbook to verify real anomaly inference.
5. Per-user authorization (shared key today). If the Task 10 personalised assistant is ever exposed, it needs real server-side identity.
6. Optional API improvements: historical headcount in the forecast; score explanations (Task 14 SHAP); aggregate endpoints; the Task 15 calibration file so pay bands appear; serve per-feature PSI if it is safe to do so.
7. Pre-existing project issues: Task 3 test; Task 10 missing artifact + unapplied patch; stray `files (1)/` in Task 5; Tasks 12–14 untestable without data.
8. **Observation on your real WATCH report (`latest_report.md`)**: `attendance_rate_3m` is held at WATCH, but its PSI against the *full training reference* is 1.467 (far above the 0.25 ALERT level), and `tenure_months` is 0.209 against training. Task 18 explains this as a reference-composition effect, but a person should confirm it is not training-serving skew. The console shows only the WATCH status and feature name, since the API does not serve the PSI values.

## L. Manager evaluation: 8 / 10
Strong: faithful to the real API, all eight pages verified live, Monitoring handles stale / failed / missing-evidence honestly, thorough tests with mutation checks, secure handling. Held back by: nothing deployed, no visual/mobile evidence, unverified anomaly inference, the schedule never run, and no per-user authorization.

## Final design question: which tab first?
**Attrition Risks.** The Overview opens with a "Start here: Attrition Risks" banner and button; Anomalies, Forecast and Monitoring stay one click away, and the Overview's monitoring card shows the live status at a glance.
