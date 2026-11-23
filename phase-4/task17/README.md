# CortexHR – Task 17: one versioned production API

CortexHR is an HR analytics system. Earlier phases built separate models. This task puts them behind **one FastAPI service** with a common `/v1` interface, a small web dashboard, security controls, tests and deployment files.

> **Live status (audited 21 Nov 2026).** The Vercel API responds, but it is **not ready for predictions**: `/v1/health` reports only `resume_ranking` loaded and `/v1/ready` returns 503. Model joblibs are intentionally git-ignored; anomaly detection also requires a private reference workbook. The Vercel project root omits sibling Task 16 resources. Task 18 monitoring routes now exist in this source, but that deployment has not been updated and still returns 404 for `/v1/monitoring`. Do not treat it as a working HR service.

## 1. What it does (for HR readers)

| Question | Endpoint | Output |
|---|---|---|
| Who may leave in the next 12 months? | `POST /v1/attrition/predict` | probability per employee; High/Medium/Low band for batches of 10+ |
| Is this attendance record unusual? | `POST /v1/anomaly/attendance` | score, flag, reason code, explanation |
| Is this payroll line unusual, and what is at risk? | `POST /v1/anomaly/payroll` | score, flag, reason code, money at risk |
| Quick HR policy answers | `POST /v1/chat` | answer from the approved FAQ, intent, confidence, fallback |
| Which resumes fit this job? | `POST /v1/resume/rank` | ranked list with matched/missing skills |
| What should this role pay? | `POST /v1/payband/analyze` | recommended midpoint (band when calibrated) |
| How many people will we have? | `POST /v1/forecast/headcount` | 1-6 month forecast with 80% interval |
| Is the service OK? | `GET /v1/health`, `GET /v1/ready` | model-by-model status |
| What did the latest monitoring run find? | `GET /v1/monitoring` | aggregate drift and fairness status |

All outputs are **decision support**, not automatic decisions.

## 2. Architecture (for ML engineers)

```
client / dashboard ──HTTPS──▶ FastAPI (cortexhr_api.main)
   middleware: request-id · body-size limit (413) · rate limit (429) · security headers · JSON logs
   /v1 routers ──▶ ModelRegistry (loaded ONCE in lifespan, read-only afterwards)
                     ├─ attrition-v2.joblib        (Task 13, needs `cortexhr` from Phase 1)
                     ├─ Task 8 anomaly service     (Tasks 5-7 code + 2 artifacts + private reference workbook)
                     ├─ hr_intent_classifier       (Task 9, retrained for this release)
                     ├─ Task 11 ranking pipeline   (stateless)
                     ├─ paybands-v1 models         (Task 15)
                     └─ Task 16 forecaster         (fitted once at startup from the private history, or precomputed file)
```

* Upstream code is **reused from the repository** (added to `sys.path` by the registry); nothing was re-implemented, and no mock model exists. A model that cannot load is reported `unavailable` and its endpoint returns **503**.
* `health` reports three separate things: the process is up, which models loaded, and readiness. Each model also has a `health_classification`: `healthy`, `degraded`, `unavailable`, or `not_configured`. `ready` is 200 only if the core five (attrition, chatbot, resume, pay-band, forecast) are loaded **and** authentication is configured; optional anomaly detectors remain visible in the degraded list.
* Monitoring reads require the API key. The scheduled job alone publishes through `PUT /v1/monitoring/report` with a separate `CORTEXHR_MONITORING_PUBLISH_KEYS` secret; the API stores aggregate fields only.
* `models_available` still lists only loaded capabilities. `health_classification` distinguishes limited-but-loaded modes (`degraded`), absent configuration/assets (`not_configured`), and load/compatibility failures (`unavailable`).
* Layout: `cortexhr_api/` (app), `artifacts/` (weights + metadata), `tests/`, `scripts/`, `docs/`; root `render.yaml`; Task 16 code is in `task16/`.

## 3. Model integration notes

| Model | Reuse | Honest caveats |
|---|---|---|
| Attrition | real `attrition-v2` (calibrated logistic regression incl. its Task 2 pipeline) | no hard yes/no threshold exists upstream, so none is invented; the official band is rank-based, so it appears only for batches ≥ 10 |
| Anomalies | Task 8 `load_service` unchanged (strict version/fingerprint checks) | needs `CORTEXHR_WORKBOOK`; see §8 |
| Chatbot | Task 9 service | artifact **retrained** from the committed CSV (the original was not in the repo); FAQ-only, no access to employee records; no server-side sessions |
| Resume | Task 11 | TF-IDF is fitted on each request's texts, so scores compare within a request. Identity labels must be on their own lines to be scrubbed (upstream behaviour) |
| Pay band | CV-selected gradient boosting + comparison models | the Task 15 band ratios were never saved: midpoint only unless `CORTEXHR_PAYBAND_CALIBRATION` is supplied. Group-gap fairness metrics are offline-only |
| Forecast | Task 16 code | live fit at startup if `CORTEXHR_HEADCOUNT_CSV` exists (reproduces Task 16's committed output exactly); otherwise a **labelled precomputed** result |

## 4. Security

* API key in `X-API-Key` for every prediction endpoint, compared in constant time; **fails closed** (503) if no key is configured. Keys, data paths and URLs come from environment variables / secret files; `.env` is git-ignored and `.env.example` has placeholders only.
* Monitoring reads require `X-API-Key`; publishing requires a separate write-only key. The GET response is schema-whitelisted and excludes employee-level data, dataset details and per-feature values.
* Validation errors name the field and rule but never echo submitted values; unhandled errors return a generic 500 with a request id (details only in server logs). Health output contains no filesystem paths.
* Logs are whitelisted JSON (method, path, status, duration, request id). Request bodies, employee ids, salaries and resume text are never logged (tested).
* Protected attributes (gender, age, career gap, exit data) are rejected by the schemas (`extra=forbid`).
* CORS only for configured origins; 1 MB body limit; in-memory rate limit (per instance). No file uploads are accepted (resumes are JSON text), so file-type validation is not applicable.
* Scan of tracked files and full git history found no keys or private keys. The Phase 1 Dockerfiles disable the Jupyter token for local use only; they are not part of this deployment.

## 5. Local installation and run

Python **3.14.6** is required for the anomaly artifacts (other versions only affect those two models).

```bash
cd phase-4/task17
uv venv --python 3.14.6 .venv && uv pip install -r requirements-dev.txt        # or any venv + pip
export CORTEXHR_API_KEYS="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
export CORTEXHR_HEADCOUNT_CSV=/path/to/monthly_headcount_history.csv           # optional (else precomputed forecast)
export CORTEXHR_WORKBOOK=/path/to/reference_workbook.xlsx                      # optional (else anomaly = 503)
python scripts/verify_artifacts.py
uvicorn cortexhr_api.main:app --port 8000     # dashboard: /app/   docs: /docs
```

Docker (from the **repository root**): `docker build -f phase-4/task17/Dockerfile -t cortexhr-api .` then `docker run -p 8000:8000 -e CORTEXHR_API_KEYS=change-me cortexhr-api`. The root `.dockerignore` excludes private datasets, secrets, and ignored model binaries; provide required files through approved mounted storage at runtime.

## 6. Tests

`python -m pytest` (markers: `real_model` = real artifacts/code, `mocked` = stand-in/unit, `needs_workbook`). Results and what is skipped are in [docs/TASK17_REPORT.md](docs/TASK17_REPORT.md).

```bash
python scripts/smoke_test.py --base-url https://YOUR-API --api-key "$CORTEXHR_API_KEY"      # 9 endpoints, PASS/FAIL/SKIP
python scripts/load_test.py  --base-url https://YOUR-API --api-key "$CORTEXHR_API_KEY" --levels 1,10,25
```

Keep hosted load tests small on a free plan; a single client IP also hits the 120/min rate limit (reported as 429, not error).

## 7. Hosting comparison and choice

Sources: official pricing/docs pages read during this task (Render pricing and docs, Railway docs/pricing, Hugging Face Hub docs). Prices change; re-check before committing money.

| Platform | Memory limit | Sleep behaviour | Build limits | Monthly price | Suitability |
|---|---|---|---|---|---|
| Render Free | 512 MB, 0.1 CPU | spins down after 15 min idle | not verified | $0 | demo only; very little CPU |
| Render Starter | 512 MB, 0.5 CPU | always on | not verified | $7 | **chosen for real use** |
| Render Standard | 2 GB, 1 CPU | always on | not verified | $25 | headroom for exact-pinned pandas/py3.14 and load |
| Railway Free / Hobby | Free: 0.5 GB, 1 vCPU (30-day $5 trial, then $1/mo); Hobby up to 48 GB | not verified | 4 GB image on Free | Hobby $5 minimum incl. $5 usage; usage ≈ $0.00000386/GB-s RAM, $0.00000772/vCPU-s | good, usage-billed; secret-file handling not verified |
| Hugging Face Spaces | CPU Basic 2 vCPU / 16 GB | free hardware sleeps after 48 h | disk 50 GB, not persistent | free CPU Basic; CPU Upgrade $0.03/h; **docs conflict on whether Docker Spaces need a paid plan** | generous CPU/RAM but public-by-default ecosystem, weaker fit for private HR data |
| Google Cloud Run | **not verified** (official pricing page could not be retrieved in this session) | scales to zero | not verified | usage-based, free tier exists (unverified) | technically strong; needs a billing account |

**Selection: Render** (Docker + GitHub deploy + `render.yaml` + HTTPS + dashboard secrets and secret files). Use Free for a demo, **Starter ($7/mo)** for anything real. Constraint found in Render's docs: **secret files are limited to 1 MB combined**: enough for `monthly_headcount_history.csv` (≈1000 rows), likely **not** for the master workbook. For the workbook use a persistent disk or the built-in HTTPS download (`CORTEXHR_WORKBOOK_URL` + mandatory `CORTEXHR_WORKBOOK_SHA256`, from private storage with a signed URL).

## 8. Deployment instructions (Render)

1. Push the current branch to GitHub.
2. Render dashboard → **New → Blueprint** → select the repo/branch (reads root `render.yaml`).
3. Set `CORTEXHR_API_KEYS` (dashboard prompt). Add Secret File `monthly_headcount_history.csv`. For anomaly detection provide the workbook (disk or signed-URL + SHA-256) and set `CORTEXHR_WORKBOOK`.
4. Set `CORTEXHR_MONITORING_PUBLISH_KEYS` to a new, distinct random secret. Set `CORTEXHR_MONITORING_DIR` to durable storage if the latest report must survive service replacement; the default container path is not a durable store.
5. Wait for the build (installs Python 3.14.6 through uv), check runtime logs for the startup lines `model <name>: loaded/unavailable`.
6. Verify: `python scripts/smoke_test.py --base-url https://<service>.onrender.com --api-key ...`. The first call after idle on Free can take ~1 minute.

**Vercel & Model Artifacts:**
* Production model binaries (`phase-4/task17/artifacts/*.joblib`) are now tracked in Git with selective unignore rules in `.gitignore`.
* Tracked artifacts include `hr_intent_classifier.joblib` (chatbot), `attrition-v2.joblib`, `paybands-v1_*.joblib`, and anomaly detectors.
* Deployments must build from the branch containing the committed artifacts (`day-20-final-audit` or when merged into `main`).
* Ensure `CORTEXHR_API_KEYS` is defined in Vercel project environment variables to authorize incoming requests.

**Live API:** responds at the current Vercel URL, but readiness is **503**. Do not use it for HR decisions. The Docker deployment path has not been built or deployed in this audit.

## Limitations

* The Vercel API responds but only resume ranking is loaded; `/v1/ready` returns 503. Monitoring source changes are not deployed there yet. The Docker build could not be verified because the Docker Desktop Linux engine was unavailable.
* Anomaly endpoints verified only for validation/auth/503 behaviour; real inference needs the private workbook and Python 3.14.6.
* Chatbot intent accuracy is modest (retrained model; see its evaluation); out-of-scope questions fall back.
* Rate limiting and (absent) sessions are per-instance. Single worker by design.
* Pay-band ranges and fairness group-gaps unavailable without Task 15 calibration/private data.

## Future improvements

Persisted Task 15 calibration, a shared rate-limit store, object-storage artifact loading, request-level audit log, browser tests for the dashboard, per-endpoint latency budgets.
