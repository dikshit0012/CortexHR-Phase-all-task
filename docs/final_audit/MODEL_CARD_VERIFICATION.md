# Model Card Verification

Each card was checked against code, rerun outputs or local API responses. "Verified" means I produced or diffed the number in this audit.

| Model | Card Created | Implementation Verified | Metrics Verified | Fairness Verified | Status |
|---|---|---|---|---|---|
| Attrition-v2 | Yes | Yes (metadata, code, local API load) | Yes - 76/76 reproduced | **No fairness test exists** (stated in card) | Verified with gap |
| Resume ranking | Yes | Yes (weights 0.6/0.4 in code; not trained) | Yes | Partly (25 twins + group means; inferred genders) | Verified with limits |
| Resume fairness / fair screener | Yes | Yes | Yes (50 tests, real protected file) | Yes, but small and inferred groups | Verified with limits |
| Attendance anomaly | Yes | Yes (IsolationForest 0.03, 9 features) | Yes - reports identical | Not evaluated (stated) | Verified |
| Payroll anomaly | Yes | Yes | Yes - tables identical | Not evaluated (stated) | Verified |
| HR intent classifier / chatbot | Yes | Yes - retrained, not the original artifact | Yes (retrained model) | Not evaluated | Verified (retrained binary) |
| Pay band | Yes | Yes (GB 300/0.05/3) | Yes (+5.5 MAE vs report, explained) | Yes - official audit run | Verified |
| Headcount forecasting | **No card** | **No** - code not in repo | No | No | Cannot verify |

## Discrepancies found and corrected
1. Task 7's README says labels were a substitute (`unlock_CX-06` absent); the real file is now used and Task 6 tables are identical.
2. Metadata `dataset.workbook` names `02_CortexHR_AI_ML_Dataset.xlsx`; the rerun used a differently named but identical-content workbook, so a naive metadata equality check shows a difference in the file name only; all numeric metrics are identical.
3. Pay-band report 60,795.9 vs rerun 60,801.4 MAE (library versions); card states both.
4. Chatbot artifact: the card says "retrained", because the original binary was not available.
5. Pay-band API returns midpoint only unless a calibration file is configured; the card says so.
6. Not verified anywhere: production performance, remote deployment, CI execution, real-user behaviour.
