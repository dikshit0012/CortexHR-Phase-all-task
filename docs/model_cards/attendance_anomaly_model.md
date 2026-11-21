# Model Card: Attendance anomaly detector (`attendance-anomaly-v1`)

## 1. Name / version
Task 5 detector: `IsolationForest` (contamination 0.03, seed 42) on 9 median-imputed features, combined (OR) with a rule layer. Artifact `attendance-anomaly-v1` built by Task 8's `build_artifacts.py` (scikit-learn 1.8.0 in this audit).

## 2-4. Overview, use, out of scope
Flags unusual monthly attendance records for a human to review (absence streaks, impossible hours, late spikes, sudden drops, zero attendance). Not for discipline, pay deduction or performance rating; a flag is a question, not a finding.

## 5. Dataset
`attendance_monthly_features`: 7,339 rows, 528 employees, 18 months (fictional). Models are unsupervised; the hidden labels (`unlock_CX-05`, 150 injected anomalies: ABSENCE_STREAK 40, IMPOSSIBLE_HOURS 35, LATE_SPIKE 30, SUDDEN_DROP 30, ZERO_ATTENDANCE 15) are used only for evaluation. **Labels are synthetic injections, so the evaluation measures recovery of planted patterns, not real misconduct.**

## 7. Evaluation (re-run; identical to the committed reports, 0 differences)
| Detector | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|
| Published baseline | 0.403 | 0.593 | 0.480 | 221 |
| Isolation Forest (our features) | 0.412 | 0.607 | 0.491 | 221 |
| Rules only | 0.918 | 0.600 | 0.726 | 98 |
| **Combined (IF OR rules)** | 0.488 | 0.827 | 0.614 | 254 (3.46%) |
Per type recall: four types 1.000, **LATE_SPIKE 0.133** (26 of 30 missed). High-priority tier precision 0.894 (47 flagged), Medium 0.404, Low 0.375.

## 8. Fairness
Not evaluated. No protected attributes were examined; legitimate absences (illness, caregiving, disability accommodation) can look anomalous and may fall unevenly across groups.

## 9. Limitations
Planted anomalies are easier than real ones; many false positives (130 of 254 flags at the combined setting); fixed 3% contamination; LATE_SPIKE essentially undetected.

## 10. Ethical considerations
Attendance monitoring is sensitive workplace surveillance. Use for data-quality and payroll-error review; explain flags to the employee before any action.

## 11. Deployment
`POST /v1/anomaly/attendance`; verified locally (6 rows, 2 flagged). Artifact must be rebuilt from the reference workbook. Not tested remotely.

## 12. Monitoring
None specific (no drift check covers attendance features); rule/model flag-rate is not tracked over time.

## 13. Release recommendation
**RELEASE WITH WARNINGS** (review aid only).
