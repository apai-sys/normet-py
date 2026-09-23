# Time-series decomposition

`normet` exposes two decomposition strategies through a single entry point:

```python
nm.decompose(method="emission", df=df, target="PM2.5", model=model, covariates=feats)
nm.decompose(method="meteorology", ...)
```

Both are nested-normalisation routes built on `normalise`: they take variables
out of the resample pool (holding them at their observed values) and difference
the resulting series to peel off each component's contribution.

## Emission (temporal trends)

The `"emission"` path successively removes a time variable (`date_unix`,
`day_julian`, `weekday`, `hour`) from the resample pool to peel off temporal
trends, giving a hierarchical attribution (trend → seasonality → diurnal).

```python
df_emi = nm.decompose(method="emission", df=df_prep, model=model,
                      covariates=feats)
# columns: observed, date_unix, day_julian, weekday, hour,
#          emi_total, emi_base, emi_noise
```

Each named time column holds that level's marginal contribution; `emi_total`
is the combined emission-driven signal, split into a constant `emi_base` and a
zero-mean `emi_noise`.

## Meteorology (weather effects)

The `"meteorology"` path starts from `emi_total`, the normalised series with
every meteorological predictor resampled, and splits the model's prediction
minus `emi_total` into one contribution per predictor.

```python
df_met = nm.decompose(method="meteorology", df=df_prep, model=model,
                      covariates=feats)
# columns: observed, emi_total, <each met feature>, met_total, met_base, met_noise
```

`met_total` (= `observed − emi_total`) equals `met_base` (its mean) plus the
contributions plus `met_noise`, and `met_noise` is the model residual
(`observed − prediction`) shifted by `met_base` — what the model does not
explain, not a weather term.

By default the predictors are fixed one at a time in order of feature
importance (`attribution="sequential"`). That is cheap, but each contribution
is conditional on the predictors fixed before it, so the split changes with the
order — and importance can reorder when the model is refitted. Pin it with
`variable_order=`, or remove the order altogether with
`attribution="shapley"`, which averages each predictor's effect over every
order (exact for up to 10 predictors; pass `n_permutations=` for a sampled
estimate beyond that).

### Grouped attribution: transport vs local

To separate long-range transport from local meteorology, attribute the two
sets of predictors as groups. Groups default to Shapley attribution, and with
two groups that costs four normalisations:

```python
df_tr = nm.decompose(method="meteorology", df=df_prep, model=model,
                     covariates=feats,
                     groups={"local": met_cols, "transport": traj_cols})
# columns: observed, emi_total, local, transport, met_total, met_base, met_noise
```

Both columns are measured against `emi_total`, which averages over the air
masses in the resample pool, so over the record they are anomalies with a mean
near zero. To measure transport against a reference air mass instead, draw the
trajectory predictors from a pool of their own:

```python
clean = df_prep[df_prep["traj_resid_atlantic"] > 0.8]
df_ref = nm.decompose(method="meteorology", df=df_prep, model=model,
                      covariates=feats,
                      groups={"local": met_cols, "transport": traj_cols},
                      resample_pools={"transport": clean[traj_cols]})
```

`emi_total` is then the level under clean Atlantic air with average local
weather, and `transport` the change from that air mass to the one that
actually arrived. A pool's columns name the predictors drawn from it; the local
weather is still drawn from the whole record (`resample_df`, filtered by
`conditional_on` if given).

If you omit `model`, `decompose` trains one for you — pass `backend=` (and
optionally `model_config=`) so it knows how.
