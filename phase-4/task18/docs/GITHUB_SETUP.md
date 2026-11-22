# GitHub setup: required secrets and workflow run

The workflow is on `main`. The latest run, [45025122306](https://github.com/dikshit0012/CortexHR-Phase-all-task/actions/runs/45025122306), failed because the required data URL secret was missing. A new run after adding the two required data secrets is needed to verify the private inputs.
Follow the steps; each ends with the evidence to capture.

## 1. Add the required secrets (Settings -> Secrets and variables -> Actions -> New repository secret)
| Secret | Value | Needed for |
|---|---|---|
| `CORTEXHR_MONITORING_DATA_URL` | private, direct HTTPS download URL for the ZIP (not a Drive folder page) | **required** |
| `CORTEXHR_MONITORING_DATA_SHA256` | SHA-256 of that exact ZIP (`(Get-FileHash .\CortexHR-monitoring-data.zip -Algorithm SHA256).Hash.ToLower()` in PowerShell) | **required integrity check** |

Add these at the repository's **Settings -> Secrets and variables -> Actions -> New repository secret**. Use the exact names above. Do not paste the signed URL or checksum into workflow files, commits, chat, or logs. For `CORTEXHR_MONITORING_DATA_URL`, use a direct signed HTTPS link to the ZIP that GitHub Actions can download; a Google Drive folder URL is not a direct ZIP URL. The workflow masks the URL, refuses non-HTTPS redirects, verifies the checksum before extraction, and never commits the bundle. If either secret is absent, the monitor still writes a NOT_EVALUATED report and fails.

ZIP layout at root: `task11_data/job_descriptions.csv`, `task11_data/resumes.csv`, `resume_bias_pairs.csv`, and `private/unlock_CX-15_protected_attributes.csv` are required for fairness checks. Provide `attrition_dataset.csv` and `attrition_scoring_2026.csv` for drift, or use `02_CortexHR_AI_ML_Dataset.xlsx` for reference and `CortexHR_Dataset.xlsx` for scoring. Provide `salary_training.csv` or `CortexHR_Dataset.xlsx` for pay-gap monitoring. `task11_data/resume_job_labels.csv` is optional. CSV and workbook sources can be mixed. If required files are absent, the archive is not used to invent results; affected checks stay NOT_EVALUATED and the job fails.

## 2. Optional API publishing (only after the monitoring routes are deployed)
The currently deployed Task 17 API does not provide the monitoring routes (GET returns 404). Do not configure API publishing until both monitoring routes are implemented and deployed. Once available, configure `CORTEXHR_API_URL`, `CORTEXHR_API_KEY`, and `CORTEXHR_MONITORING_PUBLISH_KEY`; set the server-side `CORTEXHR_MONITORING_PUBLISH_KEYS` to the publish secret. Keep this optional integration separate from the required data secrets.

**Free-plan caveat:** the container filesystem is wiped on every spin-down, so a stored report disappears until the next publish. Choose one:
a paid plan with a persistent disk and `CORTEXHR_MONITORING_DIR=/var/data/monitoring`, or set `CORTEXHR_MONITORING_REPORT_URL` (+ `CORTEXHR_MONITORING_REPORT_TOKEN`)
to a durable private https location of the report so the API repopulates itself (it re-validates and minimises whatever it downloads).

## 3. Run and capture evidence
Actions -> **model-monitoring** -> Run workflow (choose `alert` for `fail_on`; `never` does not suppress missing evidence). Capture:
* the run URL and result; the log lines of `Fetch private monitoring data` and `Run monitoring`;
* the **Job summary** (the Markdown report) and the `monitoring-results` artifact;
* if API routes have not been deployed, the API-publish step is optional and should not be configured; do not expect `VERIFIED`;
* **scheduling evidence** = a run whose event is `schedule`: `gh run list --workflow model-monitoring.yml --event schedule`. It appears after the next 02:00 UTC once the file is on the default branch
  (GitHub may delay scheduled runs by several minutes).

## 4. Verify the live endpoint by hand (any time)
Only after the monitoring API routes are deployed (section 2):
```powershell
$env:CORTEXHR_API_URL='https://<service>.onrender.com'; $env:CORTEXHR_API_KEY='<read key>'
python phase-4\task18\verify_live.py --report phase-4\task18\outputs\monitoring\latest_report.json     # add --publish (with CORTEXHR_MONITORING_PUBLISH_KEY) to push first
```
Expected: `GET /v1/monitoring -> 200` and `VERIFIED`. A first request to a sleeping free-plan service can take about a minute (the script retries).
