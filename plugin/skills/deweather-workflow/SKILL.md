---
name: deweather-workflow
description: >-
  Run a standard meteorological normalisation (de-weathering) of an air-quality
  time series with normet: inspect the table, propose the meteorological
  covariates, train and normalise as a background job, judge the held-out fit,
  read the normalised series and its meteorological part, decompose by source,
  test a trend, and write the report. Use when the user asks to remove the
  effect of weather from pollution data, normalise a series, or quantify an
  emission change.
---

# Standard De-weathering Workflow

Follow this order with the `normet` MCP tools. Every job returns an id at once;
watch it with `normet_job_status` and never load a whole table into context.

## Steps

1. **Inspect the table first**
   - `normet_list_examples` if the user wants a bundled dataset (MY1, ULEZ,
     Beijing, ...).
   - `normet_inspect_data` with `data=` (or `example=`) and `target=`. Read the
     date span and step, gaps, each column's role (pollutant, met, other),
     missing values and the correlation with the target.
   - Propose the target and the meteorological covariates with reasons; ask the
     user to confirm before starting a job.

2. **Train and normalise**
   - `normet_start_do_all` with the confirmed `target` and `covariates`.
     Choose the backend (`flaml` default, `lightgbm`, or `chronos` when the
     foundation extra is installed) and the split (`random` or `blocked`).
   - `normet_job_status` until the job has ended. Report the job id and its
     `normet` command.

3. **Judge the model before using it**
   - `normet_read_run` gives the fit on the training and testing rows, the
     feature importance and the normalised series.
   - R² on the testing rows: 0.7 or above is good, 0.5–0.7 moderate, below 0.5
     poor. A poor fit is inherited by the normalised series — say so, and
     suggest more predictors, more search trials, or `normet_start_cv`.
   - `normet_start_cv` runs walk-forward cross-validation; trust its folds over
     a single random split on an autocorrelated record.

4. **Read the result**
   - The normalised series is the concentration expected under typical weather;
     emissions and policy survive in it. `normet_figure` draws the observed and
     normalised series and the quantile band.
   - Say what the normalised series shows by year or month as the reading gives
     it. Do not claim a trend the numbers do not show.

5. **Decompose** (optional)
   - `normet_start_decompose` with `method='emission'` splits the series into
     trend, season, week and hour; `method='meteorology'` splits the weather's
     contribution by group. Poll `normet_job_status`, then `normet_read_result`.

6. **Test a trend**
   - `normet_trend` on a result column (for example the emission total) gives
     the Mann-Kendall verdict and Theil–Sen slope. Where the record is
     autocorrelated, quote the block-bootstrap-corrected p it names, not the
     classical one.

7. **Report**
   - `normet_write_report` for the HTML report of a saved run; tell the user
     where it was written.

## Alternatives

- `normet_start_deweather` runs the de-weathering of one target directly.
- `normet_start_events` flags unusual episodes in the residual series.
- `normet_start_normalise_ox` and `normet_start_normalise_regime` use the
  conserved Ox–NOx and regime-conditioned estimators.

Quote numbers only as the tools give them.
