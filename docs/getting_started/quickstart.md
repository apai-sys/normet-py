# Quick start

This page runs the whole meteorological-normalisation pipeline on data that ships with
the package. It needs a model backend: `pip install "normet[flaml]"` (see
{doc}`installation`).

## One-shot meteorological normalisation

`nm.do_all` runs the full pipeline — prepare → train → normalise — in a single
call. The example uses `normet`'s bundled real-world dataset: hourly NO2 and
ERA5 meteorology at London Marylebone Road (MY1), January–August 2020 — a
window spanning the UK COVID-19 lockdown:

```python
import normet as nm

df = nm.datasets.load_my1()

# Meteorological predictors + time features the model will see
met_vars = ["ws", "wd", "temp", "RH", "atmos_pres", "blh", "tcc", "tp", "ssrd"]
predictors = met_vars + ["date_unix", "day_julian", "weekday", "hour"]

out, model, df_prep = nm.do_all(
    df=df,
    target="NO2",
    backend="flaml",          # or "lightgbm"
    covariates=predictors,
    variables_resample=met_vars,  # resample met only — not time features
    n_samples=300,
    n_cores=4,                # parallelise resampling
)

print(out.head())
nm.modStats(df_prep, model)
```

`do_all` returns a 3-tuple: `(normalised_df, model, prepared_df)`. `normalised_df` is
indexed by `date` with the columns `observed` and `normalised`; `nm.normalise_plot(out)`
draws them.

## Bundled example data

Four real datasets from the normet model-description paper's case studies
ship with the package (`normet.datasets`):

| Loader | Contents |
|---|---|
| `load_my1()` | Hourly NO2 + ERA5 met at London Marylebone Road (MY1), Jan–Aug 2020 |
| `load_scm()` | Monthly deweathered NO2 panel, 104 UK sites, 2016–2021 (ULEZ SCM case) |
| `load_my1_pm25()` | Hourly PM2.5 + met at MY1, Jan–Aug 2020 (transport-aware case) |
| `load_traj_my1()` | 6-hourly HYSPLIT back-trajectory features arriving at MY1 |
| `example_traj_dir()` | Two-day sample of raw HYSPLIT `tdump` files for the trajectory readers |

Sources: UK AURN (Defra, Open Government Licence v3.0); ERA5 (Copernicus
Climate Change Service); HYSPLIT (NOAA ARL).

## Where to go next

- {doc}`../guide/workflow` — the same pipeline one step at a time, then
  decomposition, synthetic control and the advanced options.
- {doc}`../guide/examples_normalisation` — quantile bands, "what-if" weather and
  log-scale fits.
- {doc}`../guide/examples_foundation` — zero-shot de-weathering with Chronos-2 when
  there is too little history to train on.
- {doc}`../guide/gui` — the same workflow without writing code.
- {doc}`../guide/notebooks` — end-to-end notebooks for each case study.
