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
    value="PM2.5",
    backend="flaml",
    feature_names=["t2m", "blh", "u10", "v10", "date_unix", "day_julian", "weekday", "hour"],
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
    feature_names=feats, variables_resample=met_vars,
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
    feature_names=feats, variables_resample=met_vars,
    conditional_on={"month": [6, 7, 8]},
)

# Holding wind direction in the south-westerly sector
out_sw = nm.normalise(
    df=df_prep, model=model,
    feature_names=feats, variables_resample=met_vars,
    conditional_on={"wdir": lambda d: 180 <= d <= 270},
)
```

Values may be scalar (exact match), an iterable (`isin` semantics), or a
callable (any boolean mask).
