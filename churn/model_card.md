# Model Card — cs-intel churn model

## Overview
- **Purpose:** predict whether a B2B account churns at contract renewal, to prioritize
  customer-success interventions. Decision *support* behind a human gate — never automated
  action.
- **Deployed model:** XGBoost (gradient-boosted trees). **Baseline:** logistic regression.
- **Data:** synthetic — 500 accounts × 24 months with a known churn-generating process
  (see `datagen/`). One row per account.

## Training & evaluation
- **Strict temporal split:** accounts ordered by renewal date, latest 30% held out (train
  373 / test 127). Training renewals all precede test renewals in calendar time.
- **Leakage controls:** features aggregated only over the 6 months strictly *before* each
  account's renewal; NPS/missing values imputed with **train-only** medians.

| Model | ROC-AUC | PR-AUC | Brier |
|-------|:------:|:------:|:-----:|
| XGBoost (deployed) | 0.806 | 0.624 | 0.169 |
| Logistic (baseline) | 0.823 | 0.669 | 0.206 |

_(Reference `docker compose up` run; ~±0.01 across machines from XGBoost/BLAS threading.)_

XGBoost is deployed for materially better calibration (Brier). See
`artifacts/{calibration_curve,pr_curve,shap_summary}.png`.

## Features
Static (size, MRR, tenure, segment/industry), usage (adoption level/trend, active-seat
ratio, logins, feature events), support (ticket volume, sentiment, NPS), and **LLM-derived
intent rates** (cancel-risk / expansion / confusion / bug-report / negative-sentiment
rates). Intent features add **+0.079 AUC** (repeated 5-fold CV); `cancel_risk_rate` is the
top SHAP driver.

## Explainability
Global and per-account SHAP. The dashboard shows each account's ranked risk drivers.

## Limitations & ethics
- Synthetic data: the pipeline, hygiene, and explanations are the deliverable, not the
  coefficients. Do not transfer numbers to production.
- No protected-attribute data is used. Scores must stay decision support with a human in
  the loop (the agent's approval gate enforces this pattern downstream).
- Calibration and temporal validity should be re-checked on any real dataset before use.
