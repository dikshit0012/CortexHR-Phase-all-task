# GitHub setup: push, secrets, run, evidence (about 10 minutes)

Nothing here could be executed from the build sandbox (no GitHub credentials), so **scheduling and live-deployment evidence do not exist yet**.
Follow the steps; each ends with the evidence to capture.

## 0. Push
```powershell
git clone https://github.com/dikshit0012/CortexHR-Phase-all-task ; cd CortexHR-Phase-all-task
git fetch ..\phase-4-task-17.bundle phase-4-task-17:phase-4-task-17      # only if your clone lacks the Task 16/17 branch
git checkout -b phase-4-task-18 phase-4-task-17
git am path\to\task18.patch            # or: git apply task18.patch ; git add -A ; git commit -m "Task 18: drift and fairness monitoring"
git push -u origin phase-4-task-18
```
A workflow's `schedule:` only fires once the file is on the repository's **default branch**: open a pull request and merge it. `workflow_dispatch` ("Run workflow")
works from any branch that contains the file, but the workflow only appears in the Actions tab once it exists on the default branch, so merge first.

## 1. Secrets (Settings -> Secrets and variables -> Actions -> New repository secret, or `gh secret set NAME`)
| Secret | Value | Needed for |
|---|---|---|
| `CORTEXHR_MONITORING_DATA_URL` | private https URL of a zip with the data (below) | real data in CI |
| `CORTEXHR_MONITORING_DATA_SHA256` | SHA-256 of that zip (`Get-FileHash -Algorithm SHA256`) | integrity check |
| `CORTEXHR_API_URL` | `https://<your-service>.onrender.com` (no trailing slash) | publish + verify |
| `CORTEXHR_API_KEY` | a **read** key (one of the server's `CORTEXHR_API_KEYS`) | verification GET |
| `CORTEXHR_MONITORING_PUBLISH_KEY` | a new random value: `python -c "import secrets;print(secrets.token_urlsafe(32))"` | write-only publish key |

Data zip, files at the root: `attrition_dataset.csv`, `attrition_scoring_2026.csv`, `salary_training.csv`, `task11_data/{job_descriptions,resumes,resume_job_labels}.csv`,
`resume_bias_pairs.csv`, `private/unlock_CX-15_protected_attributes.csv`.

The workflow's data step exports each CSV above (and the Task 11/12/15 paths) automatically when present in the zip; workbooks are used if you supply `CortexHR_Dataset.xlsx` / `02_CortexHR_AI_ML_Dataset.xlsx` instead. Host the zip somewhere private (signed, expiring URL). Never commit it.

## 2. Server side (Render dashboard -> cortexhr-api -> Environment)
Set `CORTEXHR_MONITORING_PUBLISH_KEYS` to the same value as the `CORTEXHR_MONITORING_PUBLISH_KEY` secret. Keep `CORTEXHR_API_KEYS`. Redeploy from a branch that contains the Task 18 API changes.

**Free-plan caveat:** the container filesystem is wiped on every spin-down, so a stored report disappears until the next publish. Choose one:
a paid plan with a persistent disk and `CORTEXHR_MONITORING_DIR=/var/data/monitoring`, or set `CORTEXHR_MONITORING_REPORT_URL` (+ `CORTEXHR_MONITORING_REPORT_TOKEN`)
to a durable private https location of the report so the API repopulates itself (it re-validates and minimises whatever it downloads).

## 3. Run and capture scheduling evidence
Actions -> **model-monitoring** -> Run workflow. Capture:
* the run URL and result; the log lines of `Run monitoring` and of `Publish to the deployed API and verify it` (ends with `VERIFIED`);
* the **Job summary** (the Markdown report) and the `monitoring-results` artifact;
* **scheduling evidence** = a run whose event is `schedule`: `gh run list --workflow model-monitoring.yml --event schedule`. It appears after the next 02:00 UTC once the file is on the default branch
  (GitHub may delay scheduled runs by several minutes).

## 4. Verify the live endpoint by hand (any time)
```powershell
$env:CORTEXHR_API_URL='https://<service>.onrender.com'; $env:CORTEXHR_API_KEY='<read key>'
python phase-4\task18\verify_live.py --report phase-4\task18\outputs\monitoring\latest_report.json     # add --publish (with CORTEXHR_MONITORING_PUBLISH_KEY) to push first
```
Expected: `GET /v1/monitoring -> 200` and `VERIFIED`. A first request to a sleeping free-plan service can take about a minute (the script retries).
