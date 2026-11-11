# Final Submission Checklist
Status for this Task 6 workspace, checked 2026-10-26. `[x]` = verified. `[ ]` = not done or not verifiable here.

- [ ] **Live URL works from a phone and for someone else** - Not applicable to Task 6 (a command-line pipeline and report, nothing hosted). If the programme needs a deployed URL for the whole project, that is a separate decision.
- [ ] **GitHub repository is clean, with a clear README and setup steps** - The README has setup steps, but this workspace has no `.git` metadata, so status, staging, tracked files, and history cannot be checked. Review these in the complete repository before submission.
- [ ] **All Core outcomes met; Level-ups listed** - Task 6 outcomes are implemented and tested, including official-label evaluation. Task 5 files are not present in this workspace, so Task 5 regression status is not verified here.
- [x] **Official checks/answer-key comparisons attached as evidence** - The current workbook includes `unlock_CX-06`; 100 labels match the feature records with no unmatched or duplicate keys. The report and per-type metrics were regenerated from this sheet.
- [x] **Architecture diagram and release notes** - `docs/ARCHITECTURE.md` (and the diagram in the README) and `docs/RELEASE_NOTES.md`.
- [ ] **5-minute demo video** - Not recorded. Script and timings: `docs/WEEKLY_DEMO_SCRIPT.md`.
- [ ] **No secrets in Git history** - Not verifiable without Git metadata. The local workbook, official-label CSV, row-level label-linked outputs, and private raw-ID mapping are git-ignored; confirm they are not already tracked in the complete repository.
- [x] **Fictional-data notice present** - Added to the Task 6 README and the root README (the workbook's own README sheet already says all data is fictional).

## Level-ups (beyond the brief)
- 95% Wilson confidence intervals for every per-type precision and recall (small samples).
- First experiment frozen and re-creatable with `--original-experiment`; before/after tables kept separate.
- Row-order independence found, fixed and covered by shuffle/reversal tests.
- Independent reconciliation script that recomputes the headline metrics without the project code.
- Tests for OR composition, duplicate-payment keys, zero-denominator metrics, row-order independence, and privacy masking; post-hoc sensitivity table (contamination, seeds, one-rule-at-a-time ablations).
- Experimental two-tier alert view (rule alerts vs model-only review queue), clearly labelled and not replacing any detector.
- Baseline-replica seed check showing the published baseline row is not reproducible from its documentation.

## Still needs you
1. Review staging in the complete Git repository and ensure `phase-2/task06/task 6/data/raw/`, `phase-2/task06/task 6/reports/private/`, row-level label-linked CSVs (`task6_flagged_records_with_labels.csv`, `task6_examples_masked.csv`, `task6_missed_errors.csv`), and any `*.xlsx` / `unlock_CX-06.csv` answer-key copies are not staged.
2. Record the demo video using `docs/WEEKLY_DEMO_SCRIPT.md`.
3. Run the Task 5 regression suite from the complete project checkout, then create the pull request. Do not commit or push until you have reviewed the diff.
