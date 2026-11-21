# CortexHR Final Submission Checklist

## Models
- [x] Every model and decision-support component is inventoried.
- [x] Model cards are present for the seven audited ML systems.
- [x] Reproducible attrition, ranking, anomaly, intent, and pay-band metrics were re-run or compared.
- [x] Fairness limitations and untested attrition fairness are documented.

## Architecture
- [x] Architecture documentation and Mermaid diagram match the repository's major components.
- [x] Data, API, inference, monitoring, and UI flows are documented.

## Evaluation
- [x] Synthetic and replayable evaluations are clearly separated from live or unavailable evidence.
- [x] Final metrics are recorded in `docs/final_audit/FINAL_EVALUATION_RESULTS.md`.
- [x] Missing forecasting code and incomplete Task 14 evidence are explicitly documented.

## Testing
- [x] Final challenge guard tests run: 6 passed.
- [x] Task 17 API tests run: 43 passed, 3 skipped.
- [x] Full Task 18 suite rerun: 106 passed.
- [x] Known Task 10 patch failures and Task 17 limitations are not presented as passed.

## Documentation
- [x] README reviewed.
- [x] Release notes completed.
- [x] Final report and model cards completed.
- [x] Final challenge documented.
- [x] Demo script prepared.

## Release
- [x] Git status and existing `v1.0.0` tag inspected.
- [ ] Release commit and final tag verification remain dependent on final validation.
- [x] Secret and private-data patterns were reviewed.
- [ ] Remote push and GitHub release status depend on permission.

## Demo
- [x] Demo script is present.
- [ ] No recording was provided, so recording and playable-link status are not verified.
