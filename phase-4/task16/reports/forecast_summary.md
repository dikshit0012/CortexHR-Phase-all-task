# Task 16 - Forecast the Future: Headcount with Uncertainty

_Final report. Every figure is computed by the pipeline from the CortexHR workbook (fictional company data); nothing was typed in by hand._

## A. Introduction

Hiring, budgets and office capacity are planned months ahead, but headcount only becomes known after the fact. This task forecasts the company's month-end headcount and attrition for the next six months and attaches an **80% prediction interval** to each number, so planners see both the expected value and how far it could reasonably be off. Following the project rules, a simple seasonal baseline is built first and the advanced model has to beat it on months it has never seen.

## B. Dataset overview

* `monthly_headcount_history`: 972 rows, 12 series (company total `ALL` plus D00-D10), 81 months (2020-01 to 2026-09). Columns: month, department_id, department_name, headcount_end, hires, exits, voluntary_exits, transfers_in, transfers_out, headcount_start, hiring_drive_flag, attrition_rate_monthly_pct, appraisal_cycle_flag, financial_year_end_flag. Missing values 0, duplicate keys 0, missing months 0, flow-identity violations 0; departments sum to the company total in every month: True.
* `workforce_events`: 9 events (event_id, start_date, end_date, department_id, event_name, description). All are in the past; the last ends 30 Apr 2026. The sheet has **no impact numbers**, only free-text descriptions.
* `sample_forecast`: 72 rows, columns `forecast_id`, `metric`, `department_id`, `forecast_month`, `forecast_value`, `lower_80`, `upper_80`, `method`, `trained_through`, `model_version`. It holds `headcount_end` for ALL and D01-D10 (6 months each) plus `attrition_rate_monthly_pct` for ALL (6 rows), trained through 2026-09. It is the output contract; D00 (one person) is not in it, so it is not forecast.

## C. Exploratory analysis

* **Trend:** headcount rose from 171 to 441; about 3.8 people a month on a straight-line fit. Year-over-year growth was 11.9% in the latest month.
* **Seasonality:** suggestive but weak (Kruskal-Wallis p = 0.024 on net change; hires p = 0.13, exits p = 0.62). Strongest months: Jun, Jan, Jul; weakest: Apr, May.
* **Unusual months:** Jul 2023 (+14), Sep 2026 (+14). The latest month (Sep 2026) is one of them, so the starting point of the forecast sits just after a jump.
* **Structure:** annual net change slowed after 2022-2023 and 2026 is already at +43 after 9 months; no single permanent level shift is statistically supported (p = 0.32).
* **Thin data:** only 6-7 observations per calendar month and 45 months usable as validation tests. Full write-up: `reports/forecast_analysis.md`.

## D. Forecasting methodology

**Seasonal baseline.** `forecast(t) = headcount(t - 12) + annual trend`, where the annual trend is the average year-over-year change over the last 12 months known at the forecast origin. Short histories fall back to the latest year-over-year change or a drift. Its 80% interval is the 10th-90th percentile of its own past errors at that horizon, measured only on data before the forecast origin.

**Advanced model.** Five statsmodels/numpy candidates were compared by rolling-origin cross-validation inside the training window only (origins every 3 months, 6-month horizon): drift + event dummies, damped-trend ETS, drift + two seasonal harmonics, Holt-Winters ETS, and SARIMA(0,1,1)(0,1,0)12. Prophet was not used because 81 monthly points do not justify it and it adds a heavy dependency. The simplest candidate within 3% of the best CV error is selected.

| model | complexity | CV MAE | CV 80% coverage |
|---|---:|---:|---:|
| drift_events | 1 | 5.98 | 82% |
| ets_damped | 2 | 6.22 | 85% |
| drift_events_fourier | 3 | 6.26 | 83% |
| ets_holt_winters | 4 | 6.73 | 74% |
| sarima | 5 | 8.25 | 81% |
| seasonal_baseline | 0 | 13.96 |  |

**Selected: Drift + event dummies (OLS on monthly change).** It models the *monthly net change* as a constant plus event dummies, then accumulates it. Its CV MAE is 5.98 people against 13.96 for the baseline. The more elaborate seasonal models did not beat it by enough to justify their extra parameters (Holt-Winters 6.73, SARIMA 8.25), consistent with the weak seasonality found in the EDA. Its intervals come from the regression variance including parameter uncertainty (normal errors, constant variance), so they widen with the horizon. Forecasts, lower and upper bounds are clipped at zero.

## E. Workforce events

| event_id | event_name | event_type | department_id | start_date | end_date | hires_multiplier | exits_multiplier | n_event_months_observed |
|---|---|---|---|---|---|---:|---:|---:|
| EV-01 | COVID-19 hiring freeze | hiring_freeze | ALL | 2020-04-01 | 2020-06-30 | 0.00 | 0.38 | 3 |
| EV-02 | Reduced increment cycle | policy_or_pay_cycle | ALL | 2021-04-01 | 2021-04-30 | 1.60 | 1.12 | 1 |
| EV-03 | Support scale-up for festive season | hiring_drive | D06 | 2022-09-01 | 2022-11-30 | 1.18 | 2.40 | 3 |
| EV-04 | Hybrid work policy introduced | policy_or_pay_cycle | ALL | 2023-07-01 | 2023-07-31 | 1.81 | 0.49 | 1 |
| EV-05 | Engineering hiring drive | hiring_drive | D01 | 2023-07-01 | 2023-08-31 | 2.57 | 1.80 | 2 |
| EV-06 | Sales expansion hiring drive | hiring_drive | D04 | 2024-01-01 | 2024-02-29 | 3.86 | 0.86 | 2 |
| EV-07 | Data & AI team build-out | hiring_drive | D03 | 2025-01-01 | 2025-06-30 | 2.57 | 0.67 | 6 |
| EV-08 | New 5-point appraisal framework | policy_or_pay_cycle | ALL | 2025-04-01 | 2025-04-30 | 1.26 | 1.31 | 1 |
| EV-09 | Performance bonus paid with April payroll | policy_or_pay_cycle | ALL | 2026-04-01 | 2026-04-30 | 0.75 | 1.33 | 1 |

* **Typing:** events are typed by transparent keyword rules (`event_handler.EVENT_RULES`): freezes, hiring drives, and policy/pay-cycle events. Only freezes and hiring drives have a documented headcount mechanism and are used as 0/1 dummies in the model's history; policy events (EV-02, 04, 08, 09) are listed but not modelled.
* **Multipliers** (event-month average divided by the department's average over up to 12 preceding event-free months) are **descriptive before/after ratios on 1-6 months, not causal estimates**.
* **Statistically estimated:** the dummy coefficients inside the model. **Event-based:** the windows themselves, taken from the table. **Assumed:** only multipliers a person supplies in `data/future_events.csv`.
* **Future events:** the sheet contains none and no `data/future_events.csv` exists, so **no future adjustment was applied**. Forecast months carry zero event dummies; nothing is invented. If a freeze or hiring drive is planned, add it to that CSV (hires/exits multipliers); the adjustment is computed from each department's normal hires and exits and added after the model, so it cannot be double counted with the dummies. This mechanism is unit-tested.

## F. Backtesting (hide Apr-Sep 2026, train on Jan 2020-Mar 2026)

The model was selected again using only the 75 training months (result: `drift_events`); the six hidden months were never used for fitting, tuning or selection.

| Model | MAE | RMSE | MAPE % | WAPE % | Bias | 80% coverage | Avg 80% width |
|---|---:|---:|---:|---:|---:|---|---:|
| Seasonal Baseline | 10.67 | 12.55 | 2.50 | 2.54 | -10.67 | 6/6 (100%) | 50.0 |
| Advanced Model | 4.52 | 6.40 | 1.06 | 1.08 | -1.11 | 6/6 (100%) | 20.0 |

Bias = mean(forecast - actual); negative means under-forecasting. Month by month:

| Month | Actual | Baseline | Advanced | Baseline abs. error | Advanced abs. error | Advanced 80% interval |
|---|---:|---:|---:|---:|---:|---|
| Apr 2026 | 407 | 402.3 | 411.2 | 4.7 | 4.2 | 405.8 - 416.6 |
| May 2026 | 409 | 403.3 | 414.4 | 5.7 | 5.4 | 406.7 - 422.1 |
| Jun 2026 | 417 | 408.3 | 417.6 | 8.7 | 0.6 | 408.1 - 427.2 |
| Jul 2026 | 421 | 411.3 | 420.8 | 9.7 | 0.2 | 409.7 - 431.9 |
| Aug 2026 | 427 | 416.3 | 424.0 | 10.7 | 3.0 | 411.5 - 436.6 |
| Sep 2026 | 441 | 416.3 | 427.2 | 24.7 | 13.8 | 413.4 - 441.1 |

The advanced model beat the baseline on MAE (4.52 vs 10.67). Both under-forecast on average. September 2026 (a +14 jump, 17 hires) accounts for 51% of the advanced model's total absolute error, and it is a month the models could not have anticipated. Only six hidden months were tested, so coverage of 6/6 is consistent with 80% but proves little; the cross-validated coverage over many origins was 82% for the selected model.

**Transparency note (not used for selection):** the other candidates on the same hidden months:

| model | MAE | RMSE | bias | coverage_80_pct |
|---|---:|---:|---:|---:|
| drift_events | 4.52 | 6.40 | -1.11 | 100% |
| ets_damped | 7.36 | 9.82 | -5.25 | 83% |
| drift_events_fourier | 3.18 | 5.07 | -2.15 | 100% |
| ets_holt_winters | 6.92 | 9.04 | -6.36 | 83% |
| sarima | 4.33 | 6.86 | -1.84 | 100% |

`drift_events_fourier` happened to score best on these six months (MAE 3.18), but cross-validation over many origins did not favour it, and picking it now would mean choosing the model by looking at the answer. The selection was left unchanged.

**Attrition and exits backtest.** Selected rate method `trailing_24m_mean`: MAE 0.39 percentage points, 80% interval contained 5/6 months; expected-exit counts MAE 1.65 exits/month (actual monthly rates are noisy: 0.0% to 3.1% in the history). The sample file's same-month method scored MAE 0.53 pp / 2.24 exits.

## G. Six-month forecast (trained through 2026-09)

| Month | Forecast headcount | Lower 80% | Upper 80% |
|---|---:|---:|---:|
| Oct 2026 | 444.4 | 438.9 | 450.0 |
| Nov 2026 | 447.8 | 440.0 | 455.7 |
| Dec 2026 | 451.3 | 441.5 | 461.0 |
| Jan 2027 | 454.7 | 443.4 | 466.0 |
| Feb 2027 | 458.1 | 445.4 | 470.8 |
| Mar 2027 | 461.5 | 447.5 | 475.6 |

Last actual headcount (Sep 2026): 441. **Reading the interval:** if the model's assumptions hold, about 8 in 10 actual month-end headcounts should land inside the band. It is a range for the actual future value, not a range for the average.

**Expected exits.** Headcount change = hires - exits + net transfers (net transfers are zero for the company total in the data's flow identity). Exits are forecast as the monthly attrition rate (method: mean of the last 24 months) times opening headcount. Hires are **not** forecast separately; *implied hires needed* is simply the forecast net change plus expected exits, i.e. what hiring would have to be for the headcount path to hold.

| Month | Expected attrition % | Expected exits | Exits lower 80% | Exits upper 80% | Expected net change | Implied hires needed |
|---|---:|---:|---:|---:|---:|---:|
| Oct 2026 | 1.22 | 5.4 | 1.9 | 8.6 | 3.4 | 8.8 |
| Nov 2026 | 1.22 | 5.4 | 1.9 | 8.7 | 3.4 | 8.8 |
| Dec 2026 | 1.22 | 5.5 | 1.9 | 8.8 | 3.4 | 8.9 |
| Jan 2027 | 1.22 | 5.5 | 1.9 | 8.8 | 3.4 | 8.9 |
| Feb 2027 | 1.22 | 5.5 | 1.9 | 8.9 | 3.4 | 9.0 |
| Mar 2027 | 1.22 | 5.6 | 2.0 | 9.0 | 3.4 | 9.0 |

Department forecasts for the final month (all six months are in `outputs/headcount_forecast.csv`):

| Department | Mar 2027 forecast | Lower 80% | Upper 80% |
|---|---:|---:|---:|
| ALL | 461.5 | 447.5 | 475.6 |
| D01 | 174.8 | 167.4 | 182.1 |
| D02 | 27.4 | 24.9 | 29.9 |
| D03 | 54.4 | 50.9 | 57.9 |
| D04 | 47.4 | 43.9 | 50.8 |
| D05 | 13.5 | 11.1 | 15.8 |
| D06 | 62.2 | 57.8 | 66.5 |
| D07 | 20.9 | 18.4 | 23.4 |
| D08 | 17.8 | 15.6 | 19.9 |
| D09 | 30.0 | 26.5 | 33.5 |
| D10 | 11.5 | 9.8 | 13.1 |

Department forecasts are fitted separately and are **not reconciled** to the company total: department sum minus company forecast ranges from -0.8 to -0.1 people (the one-person D00 is carried at its last value).

## H. Forecast reliability - which month would I trust least?

| month | step | forecast | 80% width | Width % of forecast | Hist. SD of that calendar month's change | CV MAE at this horizon |
|---|---:|---:|---:|---:|---:|---:|
| Oct 2026 | 1 | 444.4 | 11.1 | 2.49 | 2.14 | 3.00 |
| Nov 2026 | 2 | 447.8 | 15.8 | 3.52 | 3.72 | 5.30 |
| Dec 2026 | 3 | 451.3 | 19.4 | 4.31 | 2.88 | 6.10 |
| Jan 2027 | 4 | 454.7 | 22.6 | 4.97 | 1.90 | 5.93 |
| Feb 2027 | 5 | 458.1 | 25.5 | 5.56 | 5.91 | 7.15 |
| Mar 2027 | 6 | 461.5 | 28.1 | 6.09 | 4.75 | 8.38 |

The interval is widest for **Mar 2027** (step 6, width 28.1, 6.1% of the forecast), and cross-validated error also grows from 3.0 at one month ahead to 8.4 at six months ahead. Uncertainty accumulates because each month adds its own random change on top of the previous one. The calendar month with the most volatile history is **Feb 2027** (SD of net change 5.9, versus 4.8 for Mar 2027); the model uses one constant variance, so it does not widen the interval for that month. I would trust Mar 2027 least because both the model and the cross-validation say uncertainty is largest there, and treat Feb 2027 as a month where the stated band may be a little too narrow. No future workforce events touch any forecast month, so events do not explain the difference. This is an evidence-based reading of six rows, not a guarantee.

## I. Business interpretation

* **Recruitment planning:** to follow the central path the company needs roughly 9 hires a month (expected exits 5.5 a month plus net growth 3.4); actual hiring averaged 8.7 a month over the last 12 months, which is the capacity benchmark.
* **Workforce budgeting:** plan to the central value and test the budget at both ends of the band: by Mar 2027 headcount is expected at 462 (80% range 447-476). The width is about 28 people, so cost scenarios should span it.
* **Resource allocation:** use department forecasts for direction only; small departments have intervals that are wide relative to their size and are not reconciled to the total.
* **Hiring decisions:** the near months (Oct-Nov) are the most reliable basis for commitments; treat months 4-6 as scenarios and re-run the pipeline each month as new actuals arrive. Add known freezes or drives to `data/future_events.csv` before relying on the numbers.

## J. Limitations

* The data are fictional and short: 81 months, 6-7 observations per calendar month. Seasonality cannot be established with confidence.
* The backtest uses one six-month window; the September 2026 jump alone changes the picture. Cross-validation (14 origins) is a sturdier but still small guide.
* Intervals assume normally distributed, constant-variance monthly changes; real hiring is lumpy and drives could fatten the tails. Cross-validated coverage was near, not exactly, 80%.
* Event effects are descriptive ratios on a few months and are not causal. No future events are known, so none are applied; an unplanned drive or freeze would make the forecast wrong in a way the intervals do not cover.
* Hires are not forecast independently; exits are forecast as a flat rate (the data show no reliable seasonal exit pattern), and implied hires are derived.
* Department models share the company-level model choice, are not reconciled, and the smallest departments have very few people.
* The latest observation (Sep 2026, +14) is an outlier-level jump; level-based forecasts start from it, so if part of it is transient the forecast is too high, and if it signals a new hiring wave it is too low.

## K. Conclusion

A transparent seasonal baseline and a cross-validated statistical model were built, tested on six hidden months and used to forecast October 2026-March 2027. On the hidden months the advanced model had MAE 4.52 versus 10.67 for the baseline, and the final forecast is 444 in Oct 2026 rising to 462 in Mar 2027 (80% range 447-476). The main lesson is that, with this much history, a simple trend model with event handling was at least competitive with more elaborate seasonal models in cross-validation, and that honest interval width matters as much as the central number.
