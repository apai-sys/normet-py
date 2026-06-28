# Feature engineering

`normet` provides a few light helpers for building predictors from tidy,
long-format time-series (a `date` column plus one row per timestamp). They never
mutate the input and are happy with per-station panels via `group_col`.

```python
import normet as nm

df = nm.add_lag_features(df, cols=["ws", "blh"], lags=[1, 3, 24], group_col="station")
df = nm.add_rolling_features(df, cols=["pm25"], windows=[24], aggs=["mean"])
df = nm.cyclical_encode(df, "hour", period=24)
u, v = nm.wind_to_uv(df["ws"], df["wd"])
```

## Choosing meteorology–pollution lags

The lags you pass to `add_lag_features` are a modelling choice, not something the
helper guesses for you. To pick them in a data-driven way, use `analyze_lag`,
which returns a full lag-structure diagnostic:

- **ACF / PACF** of the target — how persistent the pollutant is, and which of
  its *own* past values to include as autoregressive lags.
- **Cross-correlation (CCF)** between a meteorological driver and the
  pollutant — how many steps the driver *leads* the pollutant.

```python
res = nm.analyze_lag(
    df,
    target="pm25",
    driver="ws",
    max_lag=48,        # rows (hours, if hourly)
    prewhiten=True,    # recommended — see below
)
print(res.summary())
res.plot()             # ACF / PACF / CCF stem plots with significance bands
```

`res` is a `LagDiagnostics` with:

| field | meaning |
|-------|---------|
| `acf`, `pacf` | tables (`lag`, `value`) for the target |
| `ccf` | table (`lag`, `value`); **lag `k > 0` means the driver leads the target by `k` rows** |
| `target_ar_lags` | PACF lags exceeding the band → suggested autoregressive lags of the target |
| `driver_lags` | significant driver-leading CCF lags → suggested lags for `add_lag_features` |
| `peak_lag` | driver-leading lag with the largest \|CCF\| |
| `band` | white-noise threshold `z / sqrt(n)`; a coefficient is "significant" when \|value\| > band |

The CCF sign convention matches `add_lag_features` exactly: a `peak_lag` of `8`
means `add_lag_features(df, ["ws"], lags=[8])` is the predictor to add.

### Why pre-whitening matters

Meteorology and pollution both carry strong diurnal and seasonal cycles and are
heavily autocorrelated. A naive CCF on the raw series is dominated by that
*shared seasonality* and produces spurious peaks at many lags. With
`prewhiten=True` (the default), `analyze_lag` performs Box–Jenkins
pre-whitening: it fits an AR(p) to the driver (order chosen by AIC up to
`max_ar`), takes its innovations as the whitened driver, and filters the target
with the same AR polynomial before computing the CCF. The peak that survives is
a credible lead–lag, not an artefact of the daily cycle. If pre-whitening fails
(e.g. too few points), it logs a warning and falls back to the raw CCF.

For a regularly spaced series this is essential; `analyze_lag` sorts by `date`
and assumes regular spacing, dropping non-finite values pairwise. For multi-site
panels, call it once per site (pass a single-site slice).

### From diagnostic to model

A typical workflow turns the diagnostic into predictors and lets the learner do
the final selection:

```python
res = nm.analyze_lag(df, target="pm25", driver="ws", max_lag=48)

df = nm.add_lag_features(df, cols=["ws"], lags=res.driver_lags or [res.peak_lag])
df = nm.add_lag_features(df, cols=["pm25"], lags=res.target_ar_lags or [1])

# ... then train as usual; LightGBM / FLAML will weight the lags via importance.
```

Run `analyze_lag` once per driver (`blh`, `tp`, `t2m`, …) to assemble the lag set
for each meteorological variable.
