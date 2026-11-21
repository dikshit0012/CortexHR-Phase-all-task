# CortexHR - 5-minute demo script

I have **not** recorded this video. Below is the script and checklist for you to record manually. Every number can be reproduced from `docs/final_audit/` and the model cards.
**Demo link status:** no recording or link is available in v1.0.0; do not mark the demo checklist complete until a recording is captured and its link is tested.

## Before recording (checklist)
1. Terminal 1: `cd phase-4/task17` then start the API with local artifacts:
   `CORTEXHR_API_KEYS=demo CORTEXHR_ENV=development CORTEXHR_ARTIFACT_DIR=<artifact dir> CORTEXHR_WORKBOOK=<workbook> uvicorn cortexhr_api.main:app --port 8000`
2. Browser tabs: `http://localhost:8000/docs` (Swagger), `docs/architecture/CORTEXHR_ARCHITECTURE.md` (rendered Mermaid), `docs/model_cards/attrition_model.md`, `docs/final_audit/MANAGER_RELEASE_DECISION.md`.
3. Terminal 2: `phase-4/task18` -> `python monitor.py --fail-on never` (have the WATCH report ready).
4. Optional: `streamlit run phase-4/task19/app.py` with the API URL and key in `.streamlit/secrets.toml`.
5. Copy these three `curl` bodies into a scratch file: a 10-record attrition batch, one payband request, one chat message.
6. Do not show any real people's data (all data is fictional), keys, or the `private/` folders.

## 0:00-0:30 Introduction
"This is CortexHR, an HR analytics demo I built over twenty days. It helps an HR team look at attrition risk, unusual attendance and payroll records, resume shortlisting, pay ranges and HR questions. It is decision support only: every output goes to a human."

## 0:30-1:00 Architecture
Show the Mermaid diagram. "Data goes through cleaning and a leakage check, into seven components, behind one API with API-key security. A monitoring job watches drift and fairness. The dashed boxes are things that are missing or not integrated, and I show them on purpose."

## 1:00-2:30 ML demonstration
1. **Attrition** - POST 10+ records to `/v1/attrition/predict`. "Probability and a rank-based band. The model's ROC-AUC is 0.71, so it ranks moderately well; at a 0.5 cut-off it would catch only 2% of leavers, so we use ranks, not a fixed cut-off."
2. **Anomalies** - POST 6 attendance rows. "Isolation Forest plus rules. On planted anomalies: precision 0.49 and recall 0.83 for attendance; for payroll recall is 0.97 but precision 0.38, meaning many false alarms."
3. **Chat** - ask about casual leave: intent `hr_policy_leave`, confidence 0.95. Then "tell me a joke": safe fallback.

## 2:30-3:30 Responsible AI
Open the pay-band model card, fairness section. "The model never sees gender, yet historical pay already differs: -4% for people with 12+ month career gaps and -1.5% for women, even after controlling for job factors. My two mitigations did not fix it, so I would not release this model for real pay decisions." Then show the resume twin test: gap 0.0 across 25 twin pairs, **but** say "zero on 25 twins is not proof of fairness." Mention that data is fictional and that every row requires human review.

## 3:30-4:30 API and application
Show `/docs`, call `/v1/health` (6 of 7 models loaded, forecast unavailable) and `/v1/ready` (503). Run `monitor.py`: status WATCH, `attendance_rate_3m` flagged as training-serving skew. Optionally show the Streamlit Monitoring page.

## 4:30-5:00 Conclusion
"I built and tested models, an API, monitoring and a fairness audit, and I audited my own work. I found real problems: a corrupted artifact in transit, an incomplete scoring file, and an integration patch never applied. What remains: the forecasting module, a live deployment test, and fairness checks for attrition. It is a good portfolio project and not ready for real HR decisions."
