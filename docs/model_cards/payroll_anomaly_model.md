# Model Card: Payroll anomaly detector (`payroll-anomaly-v1`)

## 1. Name / version
Task 6 detector: `IsolationForest` + a rule layer (e.g. duplicate payment, stale salary, deduction mismatches), flags if rules OR model fire. Artifact `payroll-anomaly-v1` (scikit-learn 1.8.0 in this audit). Task 7 adds reason codes and money-at-risk estimates (rules, no new model).

## 2-4. Overview, use, out of scope
Helps a payroll clerk find likely payslip errors. Not for accusing anyone of fraud or for automatic payment holds.

## 5. Dataset
`payroll_monthly_features`: 7,355 rows (fictional). Hidden labels `unlock_CX-06`: 100 injected errors, evaluation only. **Synthetic injections.** Task 7's README states the original evaluation used a substitute label sheet; this audit re-ran with the real `unlock_CX-06`, and the committed results tables were identical.

## 7. Evaluation (re-run)
| Detector | Flagged | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Published baseline | 184 | 40 | 144 | 60 | 0.217 | 0.40 | 0.281 |
| Isolation Forest only | 184 | 56 | 128 | 44 | 0.304 | 0.56 | 0.394 |
| Rules only | 120 | 92 | 28 | 8 | 0.767 | 0.92 | 0.836 |
| **Combined** | 253 | 97 | 156 | 3 | 0.383 | 0.97 | 0.550 |
The rule `rule_salary_drop` caught 0 of its 12 STALE_SALARY targets. The original experiment (file-order duplicate rule) reported combined F1 0.574; the current order-independent configuration gives 0.550 and is the one reported here.

## 8. Fairness
Not evaluated.

## 9. Limitations
Precision 0.38 means about 6 of 10 flags are false alarms; rules are tuned to injection patterns; money-at-risk is an estimate; human review of 20 sample explanations is still pending.

## 10. Ethical considerations
Payroll data is salary data (sensitive). Restrict access; never expose individual records in logs or UI beyond the reviewer role.

## 11. Deployment
`POST /v1/anomaly/payroll`; verified locally (6 rows, 2 flagged). Not tested remotely.

## 12. Monitoring
None specific beyond the shared Task 18 pipeline.

## 13. Release recommendation
**RELEASE WITH WARNINGS.**
