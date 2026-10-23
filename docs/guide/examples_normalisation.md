# Weather normalisation

The core idea: hold every other feature fixed and *resample* the meteorology so
the model averages out weather variability, leaving a series that reflects
underlying emissions.

## One-shot pipeline

```python
import pandas as pd
import normet as nm

df = pd.read_csv("my_site.csv", parse_dates=["date"])

out, model, df_prep = nm.do_all(
    df=df,
    target="PM2.5",
    backend="flaml",
    covariates=["t2m", "blh", "u10", "v10", "date_unix", "day_julian", "weekday", "hour"],
    variables_resample=["t2m", "blh", "u10", "v10"],
    n_samples=300,
)
out.head()
```

`out` is a DataFrame indexed by date with columns `observed` and `normalised`.

## Quantile bands (resampling uncertainty)

For a single trained model, ask `normalise` for quantile columns:

```python
out = nm.normalise(
    df=df_prep, model=model,
    covariates=feats, variables_resample=met_vars,
    n_samples=300,
    return_quantiles=(0.025, 0.5, 0.975),
)
# columns: observed, normalised, q025, q500, q975
```

These bands reflect *resampling* uncertainty (one model, many shuffles). For
*model* uncertainty (many seeds, each with its own training run), use
`nm.do_all_unc` which trains several models and combines them.

## Counterfactual "what-if" conditions

The `conditional_on` argument restricts the resample pool:

```python
# What would PM2.5 have been with summer-like meteorology?
out_summer = nm.normalise(
    df=df_prep, model=model,
    covariates=feats, variables_resample=met_vars,
    conditional_on={"month": [6, 7, 8]},
)

# Holding wind direction in the south-westerly sector
out_sw = nm.normalise(
    df=df_prep, model=model,
    covariates=feats, variables_resample=met_vars,
    conditional_on={"wdir": lambda d: 180 <= d <= 270},
)
```

Values may be scalar (exact match), an iterable (`isin` semantics), or a
callable (any boolean mask).

## Log-scale fits (low-concentration, multi-decade pollutants)

A pollutant that ranges from sub-unit to hundreds of µg/m³ spends most of its
raw-scale variance on its high hours, so a linear fit buys accuracy where the
policy question is least — and the values a health assessment actually reads
are the low ones. Fitting on `log1p(value)` makes errors count in relative
rather than absolute terms — a fixed relative miss maps to a bounded error at
every level, while its absolute error grows with the level:

```python
out, model, df_prep = nm.do_all(
    df=df,
    target="PM2.5",
    backend="flaml",
    covariates=["t2m", "blh", "u10", "v10", "hour"],
    variables_resample=["t2m", "blh", "u10", "v10"],
    n_samples=300,
    log_transform=True,
)
```

What the flag changes, and what it doesn't:

- The model **trains on `log1p(value)`** and the de-weathering resamples and
  predicts on that scale; every column of `out` — `observed`, `normalised`,
  the quantile columns — is back-transformed through `expm1`, so the result
  is in the original units and non-negative by construction.
- The returned `df_prep` carries `log1p(value)` in its `value` column: it is
  the frame the model was trained on, which is what you want for
  `nm.modStats(df_prep, model)`.
- `nm.do_all_unc` accepts the same flag; every ensemble member trains on the
  log scale, the ensemble output is in original units, and the per-member
  statistics that drive the weights are log-space (the scale each member was
  fitted on).
- `nm.normalise` accepts it too, for a model you trained yourself: pass the
  frame and the model in the log1p space, get the original-scale result back.
- It needs a non-negative target: a concentration that is negative in the
  original units is rejected with an error that names the offending rows.
- The HTML report (via `make_run`'s `config`) notes the scale and the share
  of observations below 1 in the original units — the population the scale is
  for.

Judge the fit in the space it optimises:

```python
from normet.utils.metrics import Stats

Stats(out, mod="normalised", obs="observed",
      statistic=["RMSE", "logRMSE", "FAC2"])
```

FAC2 is scale-free and unchanged. Leave `log_transform=False` (the default)
for data that doesn't span decades; the linear path is bit-identical to
before.
