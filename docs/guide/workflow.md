# Step-by-step workflow

{doc}`../getting_started/quickstart` runs the pipeline in one call. This page takes it
apart: each step of `nm.do_all` on its own, then decomposition, synthetic control and
the options for larger studies. The snippets continue from the quick start's `df`,
`met_vars` and `predictors`.

## 1. Prepare data

```python
df_prep = nm.prepare_data(
    df=df,
    target="NO2",
    covariates=predictors,
    split_method="random",   # "random" | "ts" | "month_ts" | "season_ts"
    train_fraction=0.75,
)
```

The target column is renamed to `value`, which is what the later steps read.

## 2. Train model

```python
model = nm.train_model(
    df=df_prep,
    target="value",
    backend="flaml",
    covariates=predictors,
    model_config={
        "time_budget": 120,
        "metric": "r2",
        "estimator_list": ["lgbm"],
    },
    n_cores=4,
)

nm.modStats(df_prep, model)
```

## 3. Normalise

```python
df_norm = nm.normalise(
    df=df_prep,
    model=model,
    covariates=predictors,
    variables_resample=met_vars,
    n_samples=300,
    n_cores=4,
    return_quantiles=[0.025, 0.975],  # optional uncertainty bands
)
```

`df_norm` is indexed by `date` with columns `observed` and `normalised`.
Passing `return_quantiles` adds one column per quantile (e.g. `q025`,
`q975`); passing `aggregate=False` instead returns one `normalised` column
per resampling seed.

Supply a different reference period with `resample_df`, for example the weather of the
weeks before the lockdown:

```python
ref_weather = df_prep[df_prep["date"] < "2020-03-01"]

df_norm_ref = nm.normalise(
    df=df_prep,
    model=model,
    covariates=predictors,
    variables_resample=met_vars,
    resample_df=ref_weather,
    n_samples=300,
)
```

For a target that spans sub-unit to hundreds of µg/m³, fit on the log scale:
`nm.do_all(..., log_transform=True)` (also on `do_all_unc` and `normalise`)
trains on `log1p(value)` and back-transforms every result column through
`expm1`, so the output is in the original units and non-negative by
construction. See {ref}`log-scale-fits` in the normalisation guide.

More on this step: {doc}`examples_normalisation`.

## Decomposition

Split the time series into emission-driven and meteorology-driven components:

```python
df_emi = nm.decompose(df=df_prep, model=model,
                      covariates=predictors, method="emission")

df_met = nm.decompose(df=df_prep, model=model,
                      covariates=predictors, method="meteorology")
```

More on this step: {doc}`examples_decomposition`, and {doc}`examples_trend` for trend
tests on the emission component.

## Counterfactual modelling (Synthetic Control)

SCM asks: *what would have happened to the treated unit if the intervention
had not occurred?* It builds a synthetic counterfactual from a weighted
combination of untreated "donor" units and compares it to what was actually
observed after the intervention.

The bundled `load_scm()` panel reproduces the paper's ULEZ case study: the
London Ultra Low Emission Zone (launched 2019-04-08) is evaluated at the
kerbside site MY1, using deweathered monthly NO2 from traffic-type sites
elsewhere in the UK as the donor pool:

```python
scm_df = nm.datasets.load_scm()

treated_unit = "MY1"
cutoff_date = "2019-04-01"  # ULEZ launch month

# Paper's primary design: pre-COVID window, complete traffic-type donors
df = scm_df.query("'2017-01-01' <= date < '2020-03-01'")
df = df[df["type"].str.contains("Traffic")]
counts = df.dropna(subset=["NO2_dw"]).groupby("code").size()
donor_pool = [c for c in counts[counts == df["date"].nunique()].index if c != treated_unit]
df = df[df["code"].isin(donor_pool + [treated_unit])]
```

### Run SCM

```python
result = nm.run_scm(
    df=df,
    date_col="date",
    outcome_col="NO2_dw",
    unit_col="code",
    treated_unit=treated_unit,
    donors=donor_pool,
    cutoff_date=cutoff_date,
    scm_backend="scm",      # "scm" | "mlscm"
)
print(result.tail())
```

Available backends: `scm` (augmented ridge SCM), `abadie` (classic simplex), `did`
(DiD), `mcnnm` (matrix completion), `robust` (Robust SCM — HSVT de-noising, Amjad et
al. 2018), `mlscm` (**experimental** — emits `ExperimentalWarning`).

### Placebo tests

```python
placebo = nm.placebo_in_space(
    df=df, date_col="date", outcome_col="NO2_dw", unit_col="code",
    treated_unit=treated_unit, donors=donor_pool,
    cutoff_date=cutoff_date, scm_backend="scm",
)

bands = nm.effect_bands_space(placebo, level=0.95)
nm.plot_effect_with_bands(bands, cutoff_date=cutoff_date)
```

### Uncertainty quantification

```python
boot_bands = nm.uncertainty_bands(
    df=df, date_col="date", outcome_col="NO2_dw", unit_col="code",
    treated_unit=treated_unit, donors=donor_pool,
    cutoff_date=cutoff_date, scm_backend="scm",
    method="bootstrap", B=200,
)
nm.plot_uncertainty_bands(boot_bands, cutoff_date=cutoff_date)
```

More on this step: {doc}`examples_scm_guide` (diagnostics, pre-trend tests, staggered
adoption, inference) and {doc}`examples_scm` (Bayesian SCM).

## Zero-shot de-weathering (Chronos-2)

`normet[foundation]` wraps Amazon's Chronos-2 time-series foundation model.
Nothing is trained: the checkpoint conditions on the meteorology through its
covariate channel and the weather is marginalised out by resampling, as in
`normalise`. Useful when there is not enough history to train on, when a site
has just come online, or as a second estimate from a model that never saw the
target's own record.

```python
import pandas as pd
from normet.foundation import Chronos2Estimator, to_regular_index

df = pd.read_csv("my_site.csv", parse_dates=["date"]).set_index("date").sort_index()
df = to_regular_index(df)          # Chronos-2 reads position as time

met = ["t2m", "blh", "u10", "v10"]
est = Chronos2Estimator(met_covariates=met)

out = est.deweather(df, "PM2.5", met_features=met, n_samples=8, schema="normet")
nm.normalise_plot(out, ci_low="q100", ci_high="q900")
```

`schema="normet"` puts the result on the same `observed`/`normalised`/`qNNN`
shape `normalise` emits, so the existing plot and HTML-report paths apply
unchanged. `est.counterfactual(...)` projects a business-as-usual series across
an intervention and reports its own pre-intervention bias alongside the effect.

The pipelines take the same backend switch:

```python
out, estimator, df_prep = nm.do_all(df, target="PM2.5", backend="chronos-2",
                                    covariates=met, variables_resample=met)

# Meteorological decomposition works; method="emission" is refused, because the
# trend rides in the target's own history rather than the covariate channel.
parts = nm.decompose(df, target="PM2.5", method="meteorology",
                     backend="chronos-2", covariates=met)

# One 768-D vector per site, describing how it behaves rather than where it is.
vectors = nm.embed_multisite(df, site_col="site", target="PM2.5")
```

Three things decide whether this is the right tool for a given record:

- **The first `context_length` rows are not a result.** The model has no history
  to condition on until then, so `deweather` seeds them with the observed values
  and the curves coincide there by construction. Give it years, not months.
- **Every Monte-Carlo sample is a full forward pass**, so `n_samples` defaults to
  8 here rather than the AutoML path's 300. `device=None` picks CUDA, else Apple
  Silicon's Metal backend (`mps`), else the CPU; the CLI takes `--device` and the
  GUI has a **Device** selector.
- **`covariate_sensitivity` is the quality gate.** There is no parity plot and no
  feature importance — nothing was fitted. Shuffle the future weather and see how
  far the forecast moves; below ~1% the model is autoregressing and its
  "normalised" series means nothing.

`decompose`, `rolling` and `pdp` have no Chronos-2 equivalent — they are defined
in terms of a fitted model's response surface. Full guide: {doc}`examples_foundation`.

## Advanced features

### Ensemble uncertainty

`nm.do_all_unc` runs the full pipeline `n_models` times with different seeds and
returns meteorologically normalised contributions with confidence bands plus per-model
statistics:

```python
out, stats = nm.do_all_unc(
    df=df,
    target="NO2",
    backend="flaml",
    covariates=predictors,
    variables_resample=met_vars,
    n_samples=300,
    n_models=10,
    confidence_level=0.95,
    n_cores=4,
)
```

`do_all_unc` returns a 2-tuple: `(normalised_df_with_bands, model_stats_df)`.
{doc}`examples_uncertainty` attributes the spread to weather and model form.

### Multisite pipelines

`do_all_multisite` takes one long-format table with a site column and runs the
pipeline per site in parallel:

```python
results = nm.do_all_multisite(
    df_all_sites,
    site_col="site",
    target="PM2.5",
    backend="lightgbm",
    covariates=predictors,
    variables_resample=met_vars,
    n_cores=4,
)
```

More: {doc}`examples_multisite`.

### Feature engineering

```python
df = nm.add_lag_features(df, cols=["PM2.5"], lags=[1, 2, 24])
df = nm.add_rolling_features(df, cols=["PM2.5"], windows=[7, 30])
df = nm.cyclical_encode(df, col="hour", period=24)
u, v = nm.wind_to_uv(df["ws"], df["wd"])
```

More, including how to choose meteorology–pollution lags:
{doc}`examples_feature_engineering`.

### Cross-validation

`cv_score` trains and scores the backend across walk-forward (time-ordered) folds of
the prepared table:

```python
scores = nm.cv_score(df_prep, target="value", covariates=predictors, n_splits=5)
print(scores)
```

### Caching long-running computations

```python
memory = nm.make_memory(".normet_cache")

@memory.cache
def cached_normalise(df_hash, **cfg):
    df = pd.read_parquet("data.parquet")
    return nm.normalise(df, model, **cfg)

key = nm.dataframe_hash(df_prep)
result = cached_normalise(key, covariates=predictors, n_samples=300)
```

### Provenance tracking

```python
run = nm.make_run(
    df_norm,
    kind="normalise",
    config={"target": "NO2", "backend": "flaml"},
    df_prep=df_prep,
    model=model,
)
nm.save_run(run, "runs/my1_run")   # runs/my1_run.joblib + runs/my1_run.meta.json
```

`nm.generate_html_report(run, "report.html")` renders a run as a single-file HTML
report; `normet report` does the same from the command line, as HTML or Markdown.
More: {doc}`examples_caching`.

### Data adapters

```python
import normet.io as nio

# ERA5 reanalysis — single-point time-series straight from the CDS
era5 = nio.fetch_era5_timeseries(
    sites={"MY1": (51.52, -0.13)},
    variables=nio.ERA5_AQ_VARIABLES_DEFAULT,
    date_from="2018-01-01", date_to="2023-12-31",
    cache_dir=".era5_cache",
)

# UK air quality — 6 networks (AURN, AQE, SAQN, WAQN, NI, LMAM) from
# whole-year archives, or source="aurn_live" for a near-real-time window
aurn = nio.fetch_ukaq_measurements(
    "MY1", range(2018, 2024), source="aurn", pollutant="PM2.5",
)

# Open-Meteo — keyless ERA5-derived meteorology (no CDS account needed)
met = nio.fetch_openmeteo_timeseries(
    sites={"MY1": (51.52, -0.13)},
    date_from="2018-01-01", date_to="2023-12-31",
)

# OpenAQ
openaq = nio.fetch_openaq_measurements(
    location_id=12345, parameter="pm25",
    date_from="2024-01-01", date_to="2024-01-07",
)
```

More: {doc}`examples_data_adapters` and the worked {doc}`examples_openaq` example.
