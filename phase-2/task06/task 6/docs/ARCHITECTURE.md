# Architecture - Task 6 Payroll Outlier Detector

```mermaid
flowchart LR
  A[("Workbook sheet: payroll_monthly_features")] --> B["payroll_features.py: validate + 9 ratio features"]
  B --> C["payroll_rules.py: 10 business rules"]
  B --> D["payroll_model.py: Isolation Forest"]
  C --> E{{"detection.py: rules OR model"}}
  D --> E
  E --> F["Label-free flagged records (IDs masked by privacy.py)"]
  L[("Official labels: unlock_CX-06")] -. "evaluation only" .-> G
  F --> G["payroll_evaluation.py: per-type metrics, join checks, baseline"]
  G --> X["experiments.py / sensitivity.py: before-after + post-hoc checks"]
  G --> H["reporting.py + report_writer.py: CSV, JSON, Markdown"]
  X --> H
  H --> V["scripts/verify_task6_reports.py: independent recomputation"]
```

| Module (`task 6/src/cortexhr_task6/`) | Job |
|---|---|
| `data_loading.py` | Finds and reads the workbook sheets (features, baseline). Never reads labels. |
| `payroll_features.py` | Validates rows, makes ratio features, and uses prior-calendar-month employee history and role references. |
| `payroll_rules.py` | Ten transparent rules with configurable thresholds. |
| `payroll_model.py` | Isolation Forest with a guard that rejects identifier or label columns. |
| `detection.py` | Runs features, rules and model, then combines them (rules OR model). |
| `payroll_evaluation.py` | The **only** module that reads labels; join validation, per-type metrics, intervals. |
| `experiments.py`, `sensitivity.py` | Original vs corrected vs exploratory comparisons; post-hoc checks (reporting only). |
| `reporting.py`, `report_writer.py` | Data dictionary, masked examples, the Markdown report (all numbers computed). |
| `privacy.py` | Keyed-hash masking of employee and payslip IDs. |

**Key rule:** predictions are produced and saved first; labels are loaded afterwards and never reach the model or the rules.
