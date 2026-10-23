# Synthetic-control modelling

`normet.causal` ships six backends, all callable through `nm.run_scm`:

| Backend  | Estimator                                  | When to use                                         |
|----------|--------------------------------------------|-----------------------------------------------------|
| `scm`    | Ridge-augmented SCM (default)              | Most cases; robust to noisy donors                  |
| `mlscm`  | ML-SCM via FLAML                           | Many donors, non-linear donor-outcome relationships |
| `abadie` | Classic Abadie simplex SCM                 | Sparse donor pool; transparent weights              |
| `did`    | Difference-in-differences baseline         | Sanity check against parallel-trends assumption     |
| `mcnnm`  | Matrix Completion (Athey et al. 2021)      | Highly unbalanced panels; missing observations      |
| `robust` | Robust SCM — HSVT de-noising (Amjad 2018)  | Noisy/low-rank donor matrices; measurement error    |

```python
out = nm.run_scm(
    df=panel,
    date_col="date", unit_col="ID", outcome_col="value",
    treated_unit="Beijing", donors=["Tianjin", "Shanghai", "Guangzhou"],
    cutoff_date="2020-01-23",
    scm_backend="scm",
)
out.tail()  # observed, synthetic, effect
```

## Diagnostics

After fitting, get a quality report and donor-stability summary:

```python
diag = nm.scm_diagnostics(out, cutoff_date="2020-01-23")
# pre_rmse, pre_r2, post_n, att, hhi, effective_n_donors, top_donors, ...

stability = nm.loo_weight_stability(panel, ..., donors=donors)
```

## Pre-trend and event study

The parallel-trends check a reviewer asks for first: is the treated unit tracking
its synthetic control *before* the intervention?

```python
pre = nm.pre_trend_test(out, cutoff_date="2020-01-23")
print(pre)            # verdict: ok / caution / fail / insufficient
# slope, p_slope, rmpe_drift — the pre-period residual trend, and how much the
# fit quality deteriorates towards the cutoff
```

`ok` means the pre-period residuals show no slope at the 5% level and the
RMS-prediction-error of the last third of the pre-period is within 1.25× of the
first two thirds. `fail` is a significant slope at the 1% level — a trending
pre-period is not parallel whatever the drift says. `caution` is either. The
diagnostic is also part of the full report:

```python
nm.scm_diagnostics(out, cutoff_date="2020-01-23")["pre_trend"]
```

The companion figure is the event study — the effect by periods since the
cutoff, pre-period in grey and post-period in red:

```python
nm.plot_event_study(out, cutoff_date="2020-01-23", title="Effect by event time")
```

A placebo-in-time band can be laid over the post-period; its rows are already
aligned to event time 0, 1, 2, ... so it lines up row by row (a date-indexed
frame from `nm.effect_bands_space` works too, aligned by date):

```python
pt = nm.placebo_in_time(panel, ..., cutoff_date="2020-01-23", placebo_every=7)
nm.plot_event_study(out, cutoff_date="2020-01-23",
                    band=pt["ref_band_event_time"],
                    title="Effect by event time (placebo-in-time band)")
```

## Inference

Two flavours of significance test:

```python
# Conformal-style finite-sample CI for the post-period ATT
ci = nm.conformal_effect_interval(out, cutoff_date="2020-01-23",
                                  n_perm=1000, ci_level=0.95)

# Abadie's RMSPE-ratio placebo test
pl  = nm.placebo_in_space(panel, ..., scm_backend="scm")
rmt = nm.rmspe_ratio_test(pl, cutoff_date="2020-01-23")
# treated_ratio, placebo_ratios, p_value, rank
```
