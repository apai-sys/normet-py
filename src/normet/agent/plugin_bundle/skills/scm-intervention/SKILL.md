---
name: scm-intervention
description: >-
  Estimate the effect of an intervention with a synthetic control in normet:
  specify the treated unit, cutoff date and donor pool, fit the control, check
  the pre-period fit, and read the placebo-in-space, placebo-in-time,
  pre-trend, conformal and leave-one-donor-out checks. Use when the user asks
  whether a clean-air zone, policy or other intervention changed a pollutant.
---

# Synthetic Control for an Intervention

A synthetic control compares the treated unit with a weighted combination of
untreated donors chosen to match its pre-intervention record. The effect is the
post-cutoff gap; its credibility rests on the pre-period fit and the placebo
tests.

## 1. Design

- The data must be a panel: one row per unit and date, with a unit column, an
  outcome column and a date column.
- Propose the treated unit, the cutoff date and the donor pool. Donors should
  share the treated unit's regime (same pollutant, comparable sites) and have a
  complete pre-period record.
- Ask the user to confirm all three before `normet_start_scm`.

## 2. Fit and read

- `normet_start_scm` runs the fit as a job; watch it with `normet_job_status`.
- `normet_read_result` reports the pre-period R² and RMSE, the donor weights,
  the post-cutoff effect and its verdict.
- A high pre-period R² is necessary but not sufficient: MY1's ULEZ fit reaches
  R² = 1.00 with a single donor, and shuffling the treated unit's pre-cutoff
  values still fits. Judge the design by the placebo checks below.

## 3. Checks

- **Placebo in space**: refit with each donor as the treated unit. The true
  effect should stand outside the placebo distribution; read the p value it
  reports.
- **Placebo in time**: refit at fake earlier cutoffs. A real intervention
  shows an effect only after the true cutoff.
- **Pre-trend and event study**: was the fit parallel before the intervention?
- **Conformal interval**: a finite-sample interval for the average post-cutoff
  effect.
- **Leave-one-donor-out**: how far the weights and the effect move when one
  donor drops.
- **RMSPE-ratio test**: Abadie's post/pre RMSPE ratio. A large pre-period
  RMSPE makes the ratio small even when the effect is real, so report it with
  the placebo-in-space result, not instead of it.

## 4. What to tell the user

- The effect with its interval and the number of post-cutoff dates that exclude
  zero.
- Which checks pass and which do not, plainly.
- The caveats the design carries: donor overlap, a short pre-period, a
  concurrent policy, and the difference between a level shift and a trend
  change.

Quote numbers only as the tools give them.
