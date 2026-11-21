# Model Card: Headcount Forecast (`headcount-forecast-1.0`)

## Overview and intended use

Forecasts aggregate month-end workforce headcount and expected exits for planning and analysis. Outputs are decision support only, require human review, and must not be used as the sole basis for employment decisions.

## Data

Task 16 uses fictional workforce history and event data from the local workbook described in `phase-4/task16/README.md`. The workbook is not committed. The example forecast CSV documents the output schema.

## Method

The pipeline compares a seasonal baseline with statistical candidates using rolling-origin cross-validation. The selected advanced model is drift with historical event indicators, fitted by OLS on monthly changes. Forecast horizon: six months. Uncertainty is represented with 80% prediction intervals. Historical events are descriptive, not causal; future events are not applied unless explicitly provided as assumptions.

## Evaluation

The holdout backtest hides April-September 2026. The advanced model reported MAE 4.52, RMSE 6.40, WAPE 1.08%, and interval coverage of 6/6 months (100%) on this single six-month window. The narrow test period and fictional dataset limit generalization. Task 16 tests: 54 passed.

## Limitations and oversight

The series is short, seasonality evidence is weak, and only one six-month holdout window was evaluated. Prediction intervals rely on model assumptions. Department forecasts are not reconciled to the company total, hires are not forecast independently, and event effects are not causal. Review input quality and assumptions before using any forecast.

## Release recommendation

**Internal demonstration only.** Do not use for real workforce or employment decisions without representative data, broader backtesting, and appropriate governance review.
