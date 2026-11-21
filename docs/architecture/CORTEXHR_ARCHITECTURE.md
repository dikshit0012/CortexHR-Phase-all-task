# CortexHR Architecture

Built from the code in this repository. Solid boxes exist and were exercised locally in this audit; dashed boxes exist only as configuration or are missing/not integrated. All data is fictional.

```mermaid
flowchart TD
    subgraph SRC["Private data sources (never committed)"]
        WB["Master workbook / CSV exports<br/>attrition, attendance, payroll, salary, resumes, FAQ"]
        VAULT["Hidden-label vault (unlock_CX-*)<br/>evaluation only"]
    end

    subgraph PREP["Validation and preparation"]
        CLEAN["Task 1-2: audit + cleaning rules + feature pipeline"]
        LEAK["Task 13: leakage audit and fixed CV folds"]
        MASK["Task 12: masking of name, e-mail, location, tier"]
    end

    subgraph MODELS["Models and detectors"]
        ATTR["Attrition-v2<br/>Calibrated Logistic Regression (T13)"]
        SHAP["SHAP reason codes (T14)<br/>offline report"]
        RES["Resume ranker<br/>TF-IDF 0.6 + skill overlap 0.4 (T11)"]
        ATT["Attendance anomaly<br/>IsolationForest OR rules (T5)"]
        PAY["Payroll anomaly<br/>IsolationForest OR rules (T6)"]
        EXPL["Reason codes + money at risk (T7, rules)"]
        INT["HR intent classifier<br/>TF-IDF + Logistic Regression (T9)"]
        BAND["Pay-band model<br/>Gradient Boosting (T15)"]
        FC["Headcount forecast (T16)<br/>code not in repository"]:::missing
        ORCH["Chatbot orchestration + mock HR service (T10)<br/>patch not applied"]:::missing
    end

    subgraph API["API layer: cortexhr_api /v1 (FastAPI, T17)"]
        SEC["X-API-Key auth, rate limit, body-size and schema validation"]
        EP["/health /ready /attrition/predict /anomaly/attendance<br/>/anomaly/payroll /chat /resume/rank /payband/analyze<br/>/forecast/headcount /monitoring"]
    end

    subgraph MON["Monitoring (T18)"]
        MONPY["monitor.py: dataset validation, PSI on 8 features,<br/>twin-resume and pay-gap re-runs"]
        REPORT["latest_report.json / .md + history"]
        CRON["GitHub Actions cron 02:00 UTC<br/>(not executed in this audit)"]:::missing
    end

    UI["Streamlit console (T19)"]
    HUMAN["Human HR review and decision<br/>(required; no autonomous decisions)"]

    WB --> CLEAN --> LEAK --> ATTR --> SHAP
    WB --> ATT --> EXPL
    WB --> PAY --> EXPL
    WB --> BAND
    WB --> RES
    MASK --> RES
    WB --> INT
    WB -.-> FC
    INT -.-> ORCH
    VAULT -. evaluation only .-> ATT
    VAULT -. evaluation only .-> PAY
    VAULT -. audit only .-> BAND

    ATTR --> EP
    RES --> EP
    ATT --> EP
    PAY --> EP
    EXPL --> EP
    INT --> EP
    BAND --> EP
    FC -.-> EP
    SEC --> EP

    WB --> MONPY --> REPORT
    CRON -.-> MONPY
    REPORT -->|publish key| EP
    EP --> UI
    EP --> HUMAN
    UI --> HUMAN
    SHAP --> HUMAN

    classDef missing stroke-dasharray: 5 5,stroke:#b00,color:#900;
```

## Components
* **Data and preparation** - Task 1-2 clean rows (duplicates, negative tenure) and build the feature pipeline; Task 13 refuses leakage columns and rebuilds the Task 3 folds (fingerprint `646e474fae66cb0c`); Task 12 masks identity text before resume scoring.
* **Models** - one trained classifier for attrition, one text classifier for intents, one regressor for pay bands, two unsupervised anomaly detectors with rules, and a non-trained resume ranker. The forecast model (Task 16) and the Task 10 conversational layer are not available in the repository.
* **API** - one versioned FastAPI app; models load once at start-up and an unavailable model returns 503 for its own endpoint while others keep working (verified locally: forecast returned 503, the rest 200).
* **Monitoring** - `monitor.py` never retrains; it writes a report and exits 0/1/2/3. The API only stores and serves the latest report.

## Flows
* **Data flow**: private workbook/CSV -> cleaning -> features -> model artifact (built offline) -> artifact directory read by the API at start-up.
* **API flow**: client sends `X-API-Key` -> auth, rate limit, size and schema checks -> model call -> JSON with `model_version` and `request_id`; errors return a uniform error body (401, 422, 503).
* **Inference flow (attrition)**: records -> strict schema (unknown fields rejected) -> saved pipeline (impute/scale/encode) -> calibrated probability -> rank-based band (needs 10+ records) -> human review.
* **Monitoring flow**: monitor.py -> PSI (OK < 0.10, WATCH 0.10-0.25, ALERT > 0.25) + fairness re-runs -> `OK/WATCH/ALERT` -> artifact / optional publish to `/v1/monitoring/report` -> Streamlit Monitoring view.
