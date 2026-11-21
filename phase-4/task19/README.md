# CortexHR – Task 19: Insights Console

A Streamlit front end for the **Task 17 CortexHR API**. An HR manager can estimate attrition probability, check attendance and payroll
records for anomalies, ask HR policy questions, rank resumes, get pay-band recommendations and view headcount forecasts, and see the
service's model status. Every AI result is labelled **"AI-generated, for human review"** next to the result.

> **Status.** The console is not deployed. The Task 17 Vercel API responds, but readiness is 503 and most model services are unavailable.
> The current Task 17 source and live API do not implement monitoring routes; the console reports this as a missing endpoint. The current unit/page
> suite passes (108 tests); protected live inference and visual/mobile behavior were not verified.

## Pages
| Page | API route | Notes |
|---|---|---|
| Overview | `/v1/health`, `/v1/ready`, `/v1/forecast/headcount` | Real figures only. Metrics without an endpoint show "Data unavailable". |
| Attrition Risks | `POST /v1/attrition/predict` | One employee (33-field form) or CSV batch (≤100 rows). High/Medium/Low bands appear for batches ≥ 10. |
| Anomalies | `POST /v1/anomaly/attendance`, `/payroll` | One record or CSV. Needs the API's private reference data; otherwise a clear "not available" message. |
| HR Assistant | `POST /v1/chat` | FAQ answers only; cannot see employee records. |
| Resume Screener | `POST /v1/resume/rank` | Pasted text or `.txt` files (no PDF/Word: the API does not accept them). |
| Pay Bands | `POST /v1/payband/analyze` | Role profile, not a named person. |
| Workforce Forecast | `POST /v1/forecast/headcount` | Expected headcount + 80% interval; events table. |
| Monitoring | `/v1/monitoring` (not implemented by current Task 17) | The page reports the missing API route; it does not claim that a report is merely unpublished. |

**First stop for an HR manager:** the Overview opens with a "Start here: Attrition Risks" banner and button.

## Monitoring (Task 18)
The console includes a read-only monitoring page, but the current Task 17 source and deployed Vercel API do **not** implement
`GET /v1/monitoring` or `PUT /v1/monitoring/report`. A generic 404 is shown as a missing endpoint; only the planned, explicit
`monitoring_report_not_found` response is treated as “no report published.” The Task 18 workflow cannot publish until the routes
and durable report storage are implemented and deployed. Do not configure a publish key for the current API.

## Architecture
```
Browser ─► Streamlit (this app, server-side) ──HTTPS + X-API-Key──► CortexHR API (Task 17) ─► saved models
            app.py → views/*.py → api_client.py   (the only module that makes HTTP calls)
```
The API key lives only in the Streamlit server's secrets; it is never sent to the browser or shown. CORS is therefore not needed.
`contract_snapshot.json` (generated from the live `/openapi.json`) drives the forms; `scripts/sync_contract.py` regenerates it.

## Configuration
| Variable | Required | Meaning |
|---|---|---|
| `CORTEXHR_API_URL` | yes | API root, HTTPS (HTTP only for localhost) |
| `CORTEXHR_API_KEY` | yes in production | One of the API's `CORTEXHR_API_KEYS` |
| `CORTEXHR_CONSOLE_PASSWORD` | **strongly recommended** | Gate in front of the console. Without it, anyone with the link can use the API key's power. |
| `CORTEXHR_TIMEOUT_SECONDS` | no | Default 30. Use ~90 against a free host that sleeps. |

Set as environment variables or in `.streamlit/secrets.toml` (see `secrets.toml.example`; the real file is git-ignored).

## Run locally
```bash
cd phase-4/task19
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export CORTEXHR_API_URL=http://localhost:8000 CORTEXHR_API_KEY=<a key the API accepts>
streamlit run app.py
```

## Tests
```bash
python -m pytest tests -q                      # 108 unit/page tests; HTTP mocked
# The monitoring-specific integration suite targets an API with the Task 18 routes.
# Current Task 17 source does not have those routes, so it is not a current deployment pass.
```
When configured, integration tests compare console output with direct API calls and check error handling and key redaction. The live
integration suite was not run in this audit because its API environment variables were not configured; monitoring publish cases require
the Task 18 routes, which are absent from current Task 17 source. See `tests_integration/README.md`.

## Deploy (Streamlit Community Cloud)
1. Push the repo to GitHub (no secrets in it).
2. share.streamlit.io → *New app* → repo + branch, **main file `phase-4/task19/app.py`**.
3. *Advanced settings → Secrets*: `CORTEXHR_API_URL`, `CORTEXHR_API_KEY`, `CORTEXHR_CONSOLE_PASSWORD`, optionally `CORTEXHR_TIMEOUT_SECONDS = "90"`.
4. Open the app, sign in, press **Check API connection** (sidebar), then try each page. The current Task 17 deployment is only partially available: `/v1/ready` returns 503 and most model services report unavailable.
5. Record the verified URLs and screenshots in `docs/TASK19_REPORT.md`.

## Security
- API key and console password only in secrets; never logged, shown or committed (`repr` of the config hides them).
- Logs contain route label, status and latency only: no ids, payloads, resume text or keys (tested).
- Backend error text is never shown verbatim; server configuration details (e.g. "set CORTEXHR_WORKBOOK") are replaced by fixed messages; the request id is shown for support.
- Uploads are validated before any request: `.csv` ≤ 700 KB / ≤ 100 rows, `.txt` resumes ≤ 20,000 chars, only documented columns, identity columns (name, e-mail, gender, age…) refused, numbers range-checked. Request bodies over 950 KB are refused client-side (server limit 1 MB).
- Resume file names stay in the browser; the API receives generic ids (`C-1`, `C-2`…).
- Double-submit protection on every action; password gate locks after 5 wrong attempts per session.
- **Limit:** the API has one shared key, so the console cannot enforce per-user permissions. Use the console password (or put the app behind SSO) and keep the URL private.

## Human-review limitations
Attrition scores are estimates, anomalies are investigation signals, resume ranking is decision support (no automatic rejection), pay bands are not compensation decisions, forecasts are uncertain, assistant answers can be wrong. See `contracts.py`.

## Verification status
See [docs/TASK19_REPORT.md](docs/TASK19_REPORT.md). Unit/page tests pass. **Not verified**: protected live inference, deployment of this console, anomaly inference (needs the private workbook), phone/tablet layout, screenshots, and the GitHub schedule.

## Known issues
- No historical series, score explanations or aggregate counts in the API (see `docs/api_contract_audit.md`). Per-feature PSI values are not served.
- The live API reports headcount forecasting unavailable because neither the history file nor the Task 16 precomputed forecast is present in its Vercel project.
- Free hosts sleep; the first request after idling can take about a minute.
