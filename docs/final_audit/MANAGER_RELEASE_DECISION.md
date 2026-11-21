# Manager's Release Decision: the model I would NOT release

Historical decision from the 2026-11-20 audit. It remains applicable to real compensation use; current repository contents and verification are summarized in [CONSOLIDATION_UPDATE.md](CONSOLIDATION_UPDATE.md).

## Decision
**I would not release the pay-band model (`paybands-v1`, Task 15, served at `POST /v1/payband/analyze`) for any real compensation use.** It can remain in the portfolio as a documented demonstration. The attrition model is my second concern (see the end).

## Why this model
It is the only component whose output is a **number that turns directly into someone's salary**, and the audit measured, with the official protected-attribute file, that the data it learns from already contains unequal pay.

## Evidence from this audit
* Historical pay difference after controlling for job factors: **career gap 12+ months -4.0%** (95% CI -5.0 to -2.9, excludes zero); **women -1.5%** (CI -2.1 to -0.9, excludes zero). The model is trained to reproduce history, so it can launder these gaps as "market-based" recommendations.
* The model never sees gender or career gap, yet its predictions still differ by group: for the 12+ month career-gap group predicted pay is 661,112 versus 841,303 for no gap, and its signed error is +50,880 (it over-predicts versus the group's own history but still lands about 180,000 below the no-gap group).
* Both mitigations tried (label correction; dropping work_location and highest_education) left the career-gap error gap essentially unchanged (55,619 baseline; 58,911 and 55,463 after) - **no proven fix exists**.
* Test accuracy is good in aggregate (MAE 60,801, R2 0.978) which makes the output look trustworthy; the aggregate hides the group effects above.
* Groups: Female 246, Male 330, **Other 11** (below the 30-row reliability threshold), so non-binary and other employees cannot be assessed.

## Potential employee impact
Lower offers or raises for people with career breaks (disproportionately caregivers) and for women, anchored by a precise-looking figure; harder to challenge because the model appears objective.

## Dataset and evaluation limits
Synthetic historical offers; random 80/20 split by hire id with no stable employee id, so repeat-person leakage cannot be excluded; one snapshot; nothing about whether past offers were fair; no external validation; the 80% band calibration (0.802 coverage) says nothing about fairness of the midpoint.

## Security and misuse
Salary data is sensitive. The API is protected by shared `X-API-Key` values (verified: 401 without or with a wrong key). In the code I read I did not find per-user authorisation or an audit trail of who queried which role (not an exhaustive search), so any key holder can query any role. A manager could query many role/level/location combinations to infer pay structures, or use the midpoint as a negotiation ceiling. The API currently returns midpoint only (no calibrated low/high) unless a calibration file is configured.

## Missing evidence
Fairness audit on real, representative pay data; an equal-pay legal and compliance review; sensitivity to the "Other" and intersectional groups; employee-level splits; a counterfactual test showing recommendations do not worsen the measured gaps; a governance owner for overrides.

## Required before release
(1) Train on pay corrected for the controlled gaps, or recommend bands from role grades rather than learned history, and show the measured gap shrinks; (2) per-user authorisation and audit logging; (3) minimum group sizes and intersectional reporting; (4) human approval workflow with recorded rationale; (5) a pre-agreed gap tolerance that blocks release if exceeded; (6) a periodic re-audit via the Task 18 pay-gap monitor (which today only checks that the gap has not **worsened** versus the baseline, i.e. it would pass an unfair but stable model).

## Why not the others
* **Attrition-v2** is my second concern: modest AUC 0.71, recall 0.023 at 0.5, and **no fairness evaluation at all**. It is releasable only as a human-reviewed ranking after a group-level audit.
* **Chatbot** is demonstration-only because it uses mock services; **anomaly detectors** and **resume ranker** are review aids with documented false-positive rates and tested twin fairness.
* **Headcount forecasting** is an aggregate planning aid; its short fictional history and single holdout window limit external validity.
