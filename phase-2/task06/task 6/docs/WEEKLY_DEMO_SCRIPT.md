# Weekly Demo (2 minutes, recorded)
Read it as a script (add any other tasks you finished this week). Timings add up to about 2 minutes.

- [ ] **Say your name, track and which tasks you finished this week** _(0:00-0:15)_
  "Hi, I'm Dikshit Kumar, AI/ML track. This week I finished Task 6, the Payroll Outlier Detector [and: any other tasks]."
- [ ] **Show the thing working (not slides)** _(0:15-1:00)_
  Terminal in the repo root: run `python "phase-2/task06/task 6/scripts/run_task6.py"`. Point out that the official `unlock_CX-06` sheet was used, then open `phase-2/task06/task 6/reports/task6_results_table.csv` and show the eight error types with rules, model and combined precision/recall.
- [ ] **Show one number or test that proves it is correct** _(1:00-1:30)_
  Run `python "phase-2/task06/task 6/scripts/verify_task6_reports.py"` and show "OK: reports ... reconcile with an independent recomputation". Say: "Rules alone find 92 of 100 errors, the model alone 56, together 97, but together they raise 253 alerts instead of 120."
- [ ] **Name one problem you hit and how you solved it** _(1:30-1:50)_
  "Shuffling the input rows changed my results, because my duplicate rule flagged the later row in the file and the model was fitted in file order. I made both order-independent, kept the original results side by side, and added shuffle tests."
- [ ] **Say what you will do next week** _(1:50-2:00)_
  "Run the Task 5 regression suite from the complete checkout, review the staging area, and discuss the `rule_salary_drop` rule with a payroll owner."

## If the final demo is 5 minutes
Add: the architecture diagram (README), `task6_original_vs_current_vs_exploratory.csv` (original vs corrected), the two hardest error types (double payment for the model, overtime spike for the combined detector), and the limitations (synthetic labels, 5-15 cases per type, and transductive evaluation).
