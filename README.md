# CortexHR Workforce Analytics

> CortexHR is an AI-powered HR analytics demonstration and decision-support platform on **fictional data**. Its models are not designed to make autonomous hiring, firing, promotion, compensation, or other high-impact employment decisions. Human review is required.

**Release:** v1.0.0 demonstration release (see [RELEASE_NOTES.md](RELEASE_NOTES.md)). **Verdict:** a complete portfolio project, not production ready ([release readiness](docs/final_audit/RELEASE_READINESS.md)).

## What it does
Attrition risk, resume ranking with fairness audit, attendance and payroll anomaly detection with explanations, an HR intent classifier and chatbot layer, pay-band recommendation, a versioned API, drift/fairness monitoring and a console.

## Architecture
See [docs/architecture/CORTEXHR_ARCHITECTURE.md](docs/architecture/CORTEXHR_ARCHITECTURE.md) (Mermaid diagram, data/API/inference/monitoring flows).

## Models and verified highlights
| Model | Verified result (re-run 2026-11-20) | Card |
|---|---|---|
| Attrition-v2 | ROC-AUC 0.711; all 76 metrics reproduced | [card](docs/model_cards/attrition_model.md) |
| Resume ranker | P@5 0.92, NDCG@5 0.912 | [card](docs/model_cards/resume_ranking_model.md) |
| Resume fairness audit | twin gap 0.000 (25 pairs) | [card](docs/model_cards/resume_fairness_evaluation.md) |
| Attendance anomaly | P 0.49 / R 0.83 | [card](docs/model_cards/attendance_anomaly_model.md) |
| Payroll anomaly | P 0.38 / R 0.97 | [card](docs/model_cards/payroll_anomaly_model.md) |
| HR intent classifier | accuracy 0.80 | [card](docs/model_cards/hr_chatbot_model.md) |
| Pay band | MAE 60,801; career-gap pay gap -4.0% found | [card](docs/model_cards/payband_model.md) |
| Headcount forecast (Task 16) | code not in this repository | - |

## Known limitations
Fictional data and injected anomaly labels; attrition fairness untested; pay-band model reproduces historical gaps; API verified locally only; forecast module missing; 5 known failing tests. Full list in the release notes.

## Responsible AI
Human review for every output; no autonomous decisions; fairness results and their limits are in each model card and in the [manager's release decision](docs/final_audit/MANAGER_RELEASE_DECISION.md).

## Documentation
[Inventory](docs/final_audit/PROJECT_INVENTORY.md) · [Model card verification](docs/final_audit/MODEL_CARD_VERIFICATION.md) · [Final evaluation](docs/final_audit/FINAL_EVALUATION_RESULTS.md) · [Service verification](docs/final_audit/SERVICE_VERIFICATION.md) · [Final report](docs/reports/CORTEXHR_FINAL_REPORT.md) · [Demo script](docs/demo/DEMO_SCRIPT.md) · Final challenge on branch `day-20-final-challenge`.

Repository: https://github.com/dikshit0012/CortexHR-Phase-all-task

## Repository layout

| Phase | Tasks | Work |
|---|---|---|
| [Phase 1](phase-1/) | Tasks 1-4 | Attrition data audit, feature pipeline, baselines, and shared modeling package |
| [Phase 2](phase-2/) | Tasks 5-10 | Attendance/payroll anomaly detection, explanations, API, FAQ classifier, and HR integration |
| [Phase 3](phase-3/) | Tasks 11-15 | Resume ranking and fairness, attrition model and explanations, and fair-pay recommendations |
| [Phase 4](phase-4/) | Tasks 17-19 | Unified API (partial source), monitoring, console. Task 16 (forecasting) is not in this repository |

Each phase contains numbered task folders (`task01`, `task02`, etc.). The Task 2-4 shared `cortexhr` package is in `phase-1/task04/src`; Tasks 13-14 use it from there.

## Set up and run

Use a separate virtual environment for each task because dependency versions differ. For example, from the repository root in Windows PowerShell:

```powershell
python -m venv .venv-task05
.\.venv-task05\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r phase-2\task05\requirements.txt
python -m pytest phase-2\task05\tests
```

The Task 5 tests use synthetic fixtures. Running its real-data pipeline requires an authorized local workbook:

```powershell
$env:CORTEXHR_WORKBOOK = 'C:\approved-data\02_CortexHR_AI_ML_Dataset.xlsx'
python phase-2\task05\scripts\run_task5.py
```

Other tasks may require additional data or locally generated model artifacts. Read their task-specific documentation before running them.

## Data, artifacts, and privacy

Do not commit workbooks, candidate or employee data, protected-attribute files, credentials, `.env` files, local models, or person-level outputs. Obtain required data through approved channels, keep it in access-controlled local paths, and configure the task-specific environment variables documented by each project. Ignore rules are not a substitute for checking Git status before pushing.

The programs are for research and decision support. They are not production-ready HR systems and must not be used as the sole basis for employment, compensation, or other consequential decisions. Use only approved data and follow applicable privacy and retention requirements.

## Testing
```bash
# per task, in its own venv (see task README); Task 1 needs PYTHONPATH=src
python -m pytest phase-2/task05/tests
```
Latest counts per task: `docs/final_audit/FINAL_EVALUATION_RESULTS.md`.

## Verification

Run synthetic tests first. Data-backed tests and full pipelines require the approved inputs described in each task README. Existing reports may not be reproducible from a clean clone because some inputs and generated artifacts are intentionally omitted.

