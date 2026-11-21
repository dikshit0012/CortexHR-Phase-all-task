# CortexHR Final Evaluation Summary

Run date: 2026-11-21. This is a local, synthetic/fictional-data validation unless stated otherwise.

| Component | Verified metric or result |
|---|---|
| Attrition-v2 | ROC-AUC 0.7114; PR-AUC 0.3097; pooled ROC-AUC 0.7080; Brier 0.1163 |
| Attendance anomaly | Combined precision 0.488; recall 0.827; F1 0.614 |
| Payroll anomaly | Combined precision 0.383; recall 0.970; F1 0.550 |
| HR intent classifier | Accuracy 0.803; macro-F1 0.7997 |
| Resume ranking | P@5 0.92; NDCG@5 0.912 |
| Resume fairness | 25 twin pairs; maximum gap 0.000 |
| Pay bands | Gradient-boosting MAE 60,801; R2 0.978 |
| Pay fairness | Career-gap controlled gap -4.0%; women gap -1.5% |
| Monitor | Real-data run status WATCH; 2 PSI watches |
| Final challenge guard | 6 tests passed |
| Task 17 API | 43 passed, 3 skipped |
| Task 18 monitoring | 106 passed |

## Testing fixes re-run
- Task 18 duplicate unlock-record handling: 3 focused tests passed.
- Task 18 full rerun: 106 passed.
