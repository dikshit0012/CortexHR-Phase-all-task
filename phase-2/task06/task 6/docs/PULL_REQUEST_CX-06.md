## Ticket
CX-06 Payroll Outlier Detector
_(ticket ID inferred from the `unlock_CX-06` sheet name; change it if your tracker uses a different ID)_

## What I changed
- Task 6 includes feature engineering, 10 business rules, Isolation Forest, combined detection, per-error-type evaluation, baseline comparisons, masked evidence, report reconciliation, 96 tests, README and project documents.
- `.gitignore` excludes the local workbook, official answer-key CSV, private raw-ID mapping, and row-level label-linked evidence.
- This workspace contains no Task 5 directory or tests; no Task 5 files were available to modify or run.
- The current workbook contains the official `unlock_CX-06` sheet. All 100 labels join cleanly; the pipeline labels the evaluation as final.
- The corrected default now calculates role-pay baselines from the previous calendar month so future months cannot affect an earlier row's role-reference feature. Legacy experiment mode preserves the full-window reference for historical comparison.
- Identifier masking now uses a process-random key by default; an environment secret is required for stable pseudonyms across separate runs.
- A first (order-dependent) experiment is frozen in `phase-2/task06/task 6/reports/original_experiment/`. A review then made the duplicate rule and the model fit independent of row order. Both sets of results are reported side by side.

## How I tested it
```
python -m pytest "phase-2\task06\task 6\tests"                     ->  96 passed   (Windows, Python 3.14.6)
python "phase-2\task06\task 6\scripts\run_task6.py"                ->  exit 0; official `unlock_CX-06` used
python "phase-2\task06\task 6\scripts\verify_task6_reports.py"     ->  OK (Windows; independent recomputation)
python -m pytest "phase-2\task05\tests"                            ->  NOT RUN at the time; Task 5 was absent from that workspace
```
- Tests cover OR composition, duplicate-payment keys, zero-denominator metrics, shuffled/reversed row order, and privacy masking.
- The real-data run processed 7,355 feature rows and 100 official labels; report reconciliation passed.
- Recreated the frozen original experiment in a temporary directory; all per-type metrics matched the preserved CSV.
- Git status, staging, tracked-file status, and history could not be checked because this workspace has no `.git` directory.

## Evidence (screenshot, output, link)
- Report: `phase-2/task06/task 6/reports/task6_comparison_report.md`
- Per-error-type table: `phase-2/task06/task 6/reports/task6_results_table.csv` (full metrics in `task6_metrics_per_type.csv`)
- Original vs corrected vs exploratory: `task6_original_vs_current_vs_exploratory.csv`
- Rules vs model: `task6_rules_vs_model_overlap.csv`, `task6_rule_hits.csv`
- Examples of rule catches (IDs masked): `task6_examples_masked.csv`
- Run summary with leakage checks: `task6_run_summary.json`
- Screenshot of the terminal run: _(add yours)_

| Detector (corrected default) | Flagged | Precision | Recall |
|---|--:|--:|--:|
| Rules only | 120 | 0.767 | 0.920 |
| Isolation Forest only | 184 | 0.304 | 0.560 |
| Combined (rules OR model) | 253 | 0.383 | 0.970 |

The combined detector finds more errors but also creates more false alarms; it is not better in every sense.

## Checklist
- [x] Tests pass  - [ ] Git history reviewed  - [x] README updated  - [ ] I can explain every line

_Left unticked on purpose: "I can explain every line" because only you can say that. Git history and staging were not available for review in this workspace; check the complete repository before committing._
