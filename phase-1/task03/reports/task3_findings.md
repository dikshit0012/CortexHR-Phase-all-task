# Task 3 findings: honest baselines

**Correction:** the 2,400-row `cortexhr_dataset.xlsx` / `employees` sheet in this folder is a **synthetic stand-in**. It is not `attrition_dataset` (2,389 rows). Scores from that file must not be submitted.

Official Task 3 results use the real workbook, the Day 2 pipeline, 2,379 rows (6 duplicates dropped, 4 negative-tenure rows held out), and the same stratified 5-fold split for every model.

Source: `task 4/files/cortexhr-ml/reports/day3_baselines.md` and `notebooks/02_baselines.ipynb`.

## Comparison table (real data vs `baseline_benchmarks`)

| Model | ROC-AUC ours | ROC-AUC benchmark | diff | PR-AUC ours | PR-AUC bench | recall ours | recall bench | accuracy ours | accuracy bench |
|---|---|---|---|---|---|---|---|---|---|
| Dummy (always stays) | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.000 | 0.147 | 0.146 | 0.000 | 0.000 | 0.853 | 0.854 |
| Logistic Regression (balanced) | 0.712 ± 0.018 | 0.713 ± 0.010 | −0.001 | 0.309 | 0.332 | 0.633 | 0.630 | 0.686 | 0.683 |
| Decision Tree (depth 4, balanced) | 0.654 ± 0.038 | 0.650 ± 0.019 | +0.004 | 0.255 | 0.256 | 0.525 | 0.524 | 0.681 | 0.676 |
| Decision Tree (no depth limit) | 0.542 ± 0.028 | not in sheet |  | 0.164 |  | 0.226 |  | 0.765 |  |

Evaluated rows: **2,379** (349 leavers, 14.7%). Dummy accuracy is high only because most people stay.

## Five-sentence note

Our logistic regression reaches a ROC-AUC of 0.712 (sd 0.018 across the five folds) against the benchmark's 0.713, so it is level with the benchmark, and the depth-4 tree (0.654 vs 0.650) and the Dummy model (0.500 ROC-AUC, 0.853 accuracy) match it too.

The one number below the benchmark is the logistic regression's PR-AUC (0.309 vs 0.332), but that gap is smaller than twice the fold-to-fold sd of 0.023 and we also used 2,379 rows after removing duplicates and negative-tenure rows, so exact equality was never expected.

The gaps are believable because they are tiny compared with the noise: changing the fold split moves our ROC-AUC by about 0.005, and a bootstrap 95% interval (0.679 to 0.737) contains the benchmark.

Nothing looks like leakage: every honest score is far below 0.9, a control run with shuffled labels scores 0.49 (chance), and fitting the preprocessing outside the folds changed nothing measurable.

The Dummy result shows why accuracy misleads (it is 85% accurate and catches nobody), and the logistic regression shows the problem is genuinely hard: it ranks a leaver above a stayer about 71% of the time, but at a 0.5 cut-off it catches 63% of leavers with only 26% precision, which Task 4 judges with risk bands.

## Decision tree (why training scores lie)

With no depth limit the tree reaches AUC 1.00 on the rows it was fitted on and 0.54 in cross-validation. It memorises training employees. Depth 4 and logistic regression have much smaller train/CV gaps.

## Dummy result

A model that always says "will stay" is about 85% accurate because about 85% of people stay. It catches 0% of leavers. ROC-AUC is 0.5 (every row gets the same score). That is why accuracy is the wrong headline metric.
