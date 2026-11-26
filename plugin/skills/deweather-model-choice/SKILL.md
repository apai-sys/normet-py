---
name: deweather-model-choice
description: >-
  Choose the predictors and judge the model behind a normet de-weathering:
  meteorological covariates, random versus blocked splits, walk-forward
  cross-validation, R² and RMSE on held-out rows, feature importance, and the
  met-versus-emission decomposition. Use when the user asks which weather
  variables to include, whether a model is good enough to normalise with, or
  why two normalisations disagree.
---

# Choosing Predictors and Judging the Model

The normalised series inherits every error of the model, so the model must be
judged before the result is read.

## 1. Predictors

- Start from `normet_inspect_data` with the target: it names each column's
  role, missing values and correlation with the target.
- Include the meteorological variables that plausibly drive the pollutant:
  wind speed and direction, temperature, relative humidity, boundary-layer
  height, pressure, cloud, radiation, precipitation. Keep the calendar terms
  (`date_unix`, `day_julian`, `weekday`, `hour`) so that emissions, season and
  diurnal cycles survive in the normalised series.
- Drop a covariate that duplicates another (two temperature columns, two wind
  speeds) or is mostly missing; say which and why.
- Ask the user to confirm the list before `normet_start_do_all`.

## 2. Splits and validation

- A single random split flatters an autocorrelated series: neighbouring hours
  sit in the training rows. Prefer `split_method='blocked'`, or run
  `normet_start_cv` (walk-forward folds) and trust the folds when they sit
  below the random split.
- On panel data, never let one unit's rows appear in both sides of a split.

## 3. Reading the fit

- `normet_read_run` reports R², RMSE and FAC2 on the training and testing
  rows. Judge on testing: 0.7 or above good, 0.5–0.7 moderate, below 0.5 poor.
- Training far better than testing is expected of boosted trees; testing poor
  is the problem. Suggest more predictors, more search trials, a blocked split,
  or another backend.
- Feature importance is a split count (LightGBM) or gain: it shows what the
  model used, not what causes the pollutant.

## 4. When two normalisations disagree

- Check that both used the same target, covariates, resampled variables and
  seed.
- Compare their held-out fits first; a better-fitting model is not necessarily
  the right estimand (level versus paired de-weathering).
- `normet_start_decompose` with `method='meteorology'` says how much of the
  change each weather group explains; `method='emission'` says how much is
  left in the calendar terms.
- The `deweathered` SCM backend and the ICN path answer different questions;
  say which one the user needs before comparing numbers.

Quote numbers only as the tools give them.
