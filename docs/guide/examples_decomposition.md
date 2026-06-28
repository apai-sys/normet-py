# Time-series decomposition

`normet` exposes two decomposition strategies through a single entry point:

```python
nm.decompose(method="emission", df=df, value="PM2.5", model=model, feature_names=feats)
nm.decompose(method="meteorology", ...)
```

Both are leave-one-out routes built on `normalise`: they successively fix
variables in the resample pool and difference the resulting series to peel off
each component's contribution.

## Emission (temporal trends)

The `"emission"` path successively removes a time variable (`date_unix`,
`day_julian`, `weekday`, `hour`) from the resample pool to peel off temporal
trends, giving a hierarchical attribution (trend → seasonality → diurnal).

```python
df_emi = nm.decompose(method="emission", df=df_prep, model=model,
                      feature_names=feats)
# columns: observed, date_unix, day_julian, weekday, hour,
#          emi_total, emi_base, emi_noise
```

Each named time column holds that level's marginal contribution; `emi_total`
is the combined emission-driven signal, split into a constant `emi_base` and a
zero-mean `emi_noise`.

## Meteorology (weather effects)

The `"meteorology"` path does the same for the meteorological predictors,
ordered by feature importance, isolating each met variable's contribution on
top of the emission signal.

```python
df_met = nm.decompose(method="meteorology", df=df_prep, model=model,
                      feature_names=feats)
# columns: observed, emi_total, <each met feature>, met_total, met_base, met_noise
```

`met_total` (= `observed − emi_total`) is the weather-driven part, again split
into a constant `met_base` and a zero-mean `met_noise`.

If you omit `model`, `decompose` trains one for you — pass `backend=` (and
optionally `model_config=`) so it knows how.
