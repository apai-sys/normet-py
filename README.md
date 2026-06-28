# normet

**Normalisation, Decomposition, and Counterfactual Modelling for Environmental Time-series**

[![PyPI version](https://badge.fury.io/py/normet.svg)](https://pypi.org/project/normet/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/normet-dev/normet-py/actions/workflows/ci.yml/badge.svg)](https://github.com/normet-dev/normet-py/actions)
[![Docs](https://readthedocs.org/projects/normet/badge/?version=latest)](https://normet.readthedocs.io)

`normet` is a Python package for **deweathering**, **causal inference**, and **policy evaluation** on environmental time-series data. It wraps AutoML model training, Monte Carlo weather normalisation, and Synthetic Control Methods behind a clean, high-level API — so you spend time on science, not boilerplate.

`normet` brings together three ideas behind one API:

1. **AutoML model training** (FLAML or LightGBM) learns how a pollutant
   concentration depends on meteorology, time, and other predictors.
2. **Monte Carlo weather normalisation ("deweathering")** re-runs that model
   thousands of times against resampled, randomised weather to cancel out
   day-to-day weather noise and reveal the underlying emission-driven trend.
3. **Synthetic Control Methods (SCM)** answer "what would have happened
   without this policy?" by building a counterfactual from untreated donor
   units, with placebo tests and uncertainty bands.

### Contents

- [Core capabilities](#core-capabilities)
- [Installation](#installation)
- [Quick start: one-shot weather normalisation](#quick-start-one-shot-weather-normalisation)
- [Step-by-step workflow](#step-by-step-workflow)
- [Decomposition](#decomposition)
- [Counterfactual modelling (Synthetic Control)](#counterfactual-modelling-synthetic-control)
- [Advanced features](#advanced-features)
- [CLI](#cli)
- [Dependencies](#dependencies)
- [Documentation](#documentation)
- [How to cite](#how-to-cite)
- [Contributing](#contributing)
- [License](#license)

---

## Core capabilities

| Module | What it does |
|---|---|
| `nm.prepare_data` | Imputation, date-feature engineering, train/test splitting |
| `nm.train_model` / `nm.build_model` | AutoML via FLAML or tuned LightGBM |
| `nm.normalise` | Monte Carlo weather normalisation (deweathering) |
| `nm.decompose` | Split time series into emission-driven and met-driven components |
| `nm.rolling` | Rolling-window normalisation for trend analysis |
| `nm.run_scm` / `nm.scm` | Synthetic Control Method (classic, augmented ridge) |
| `nm.mlscm` | ML-augmented SCM (**experimental** — see [Counterfactual modelling](#counterfactual-modelling-synthetic-control)) |
| `nm.placebo_in_space` / `nm.placebo_in_time` | Placebo significance tests |
| `nm.uncertainty_bands` | Bootstrap / Jackknife confidence intervals |
| `nm.do_all` | End-to-end pipeline in one call |
| `nm.do_all_unc` | End-to-end pipeline with model-ensemble uncertainty bands |
| `nm.do_all_multisite` | Parallel per-site pipelines |
| `nm.pdp` | Partial dependence plots |
| `nm.modStats` | Model performance metrics (R², RMSE, MAE, …) |
| `nm.detect_events` / `nm.anomaly_scores` | Event detection on normalised residuals |
| `nm.normalise_plot` / `nm.decomposition_stack` / `nm.scm_dashboard` | Plotting helpers for normalisation, decomposition, and SCM results |
| `nm.bayesian_scm` / `nm.plot_bayesian_scm` | Bayesian (PyMC) posterior SCM with credible bands — **optional**, needs `pymc` + `arviz` |
| `nm.scm_diagnostics` / `nm.loo_weight_stability` / `nm.conformal_effect_interval` | SCM fit diagnostics, donor-weight stability, conformal intervals |
| `nm.make_run` / `nm.save_run` / `nm.generate_html_report` | Provenance tracking and auto-generated HTML/Markdown run reports |
| `nm.make_memory` | On-disk caching for expensive pipelines |
| `normet.io` | ERA5, EEA, DEFRA, OpenAQ data adapters |

---

## Installation

Install the stable release from PyPI:

```bash
pip install normet
```

Install the development version from GitHub:

```bash
pip install git+https://github.com/normet-dev/normet-py.git
```

### Extras

`normet` requires at least one ML backend. Install extras as needed:

```bash
# AutoML backend (recommended)
pip install "normet[flaml]"

# Lightweight LightGBM tuner
pip install "normet[lgb]"

# Dask support for large datasets
pip install "normet[dask]"

# ERA5 / EEA / OpenAQ / AURN data adapters
pip install "normet[data]"

# CLI entry point
pip install "normet[cli]"

# Everything
pip install "normet[all]"
```

---

## Quick start: one-shot weather normalisation

`nm.do_all` runs the full pipeline — prepare → train → normalise — in a single call:

```python
import normet as nm
import pandas as pd

from notebooks._synth import make_my1_data
df = make_my1_data().set_index("date")

# All meteorological predictors the model will see
predictors = [
    "u10", "v10", "d2m", "t2m", "blh", "sp", "ssrd", "tcc", "tp", "rh2m",
    "date_unix", "day_julian", "weekday", "hour",
]

# Subset to resample (met variables only — not time features)
met_vars = ["u10", "v10", "d2m", "t2m", "blh", "sp", "ssrd", "tcc", "tp", "rh2m"]

out, model, df_prep = nm.do_all(
    df=df,
    value="PM2.5",
    backend="flaml",          # or "lightgbm"
    feature_names=predictors,
    variables_resample=met_vars,
    n_samples=300,
    n_cores=4,                # parallelise resampling
)

print(out.head())
nm.modStats(df_prep, model)
```

`do_all` returns a 3-tuple: `(normalised_df, model, prepared_df)`.

---

## Step-by-step workflow

### 1. Prepare data

```python
df_prep = nm.prepare_data(
    df=df,
    value="PM2.5",
    feature_names=predictors,
    split_method="random",   # "random" | "ts" | "season" | "month"
    fraction=0.75,
)
```

### 2. Train model

```python
model = nm.train_model(
    df=df_prep,
    value="value",
    backend="flaml",
    feature_names=predictors,
    model_config={
        "time_budget": 120,
        "metric": "r2",
        "estimator_list": ["lgbm"],
    },
    n_cores=4,
)

nm.modStats(df_prep, model)
```

### 3. Normalise

```python
df_norm = nm.normalise(
    df=df_prep,
    model=model,
    feature_names=predictors,
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

Supply a different reference period with `resample_df`:

```python
ref_weather = df_prep.loc["2019", met_vars]

df_norm_ref = nm.normalise(
    df=df_prep,
    model=model,
    feature_names=predictors,
    variables_resample=met_vars,
    resample_df=ref_weather,
    n_samples=300,
)
```

---

## Decomposition

Split the time series into emission-driven and meteorology-driven components:

```python
df_emi = nm.decompose(df=df_prep, model=model,
                      feature_names=predictors, method="emission")

df_met = nm.decompose(df=df_prep, model=model,
                      feature_names=predictors, method="meteorology")
```

---

## Counterfactual modelling (Synthetic Control)

SCM asks: *what would have happened to the treated unit if the intervention
had not occurred?* It builds a synthetic counterfactual from a weighted
combination of untreated "donor" units and compares it to what was actually
observed after the intervention.

```python
treated_unit = "2+26 cities"
donor_pool = [
    "Dongguan", "Zhongshan", "Foshan", "Beihai", "Nanning", "Nanchang",
    "Xiamen", "Taizhou", "Ningbo", "Guangzhou", "Huizhou", "Hangzhou",
    # ... full list in notebooks/4.Counterfactual Modelling.ipynb
]
cutoff_date = "2015-10-23"  # intervention start date

from notebooks._synth import make_aq_weekly
scm_df = make_aq_weekly()
df = scm_df.query("'2015-05-01' <= date < '2016-04-30'")
df = df[df["ID"].isin(donor_pool + [treated_unit])]
```

### Run SCM

```python
result = nm.run_scm(
    df=df,
    date_col="date",
    outcome_col="SO2wn",
    unit_col="ID",
    treated_unit=treated_unit,
    donors=donor_pool,
    cutoff_date=cutoff_date,
    scm_backend="scm",      # "scm" | "mlscm"
)
print(result.tail())
```

Available backends: `scm` (augmented ridge SCM), `abadie` (classic simplex), `did` (DiD), `mcnnm` (matrix completion), `robust` (Robust SCM — HSVT de-noising, Amjad et al. 2018), `mlscm` (**experimental** — emits `ExperimentalWarning`).

### Placebo tests

```python
placebo = nm.placebo_in_space(
    df=df, date_col="date", outcome_col="SO2wn", unit_col="ID",
    treated_unit=treated_unit, donors=donor_pool,
    cutoff_date=cutoff_date, scm_backend="scm",
)

bands = nm.effect_bands_space(placebo, level=0.95)
nm.plot_effect_with_bands(bands, cutoff_date=cutoff_date)
```

### Uncertainty quantification

```python
boot_bands = nm.uncertainty_bands(
    df=df, date_col="date", outcome_col="SO2wn", unit_col="ID",
    treated_unit=treated_unit, donors=donor_pool,
    cutoff_date=cutoff_date, scm_backend="scm",
    method="bootstrap", B=200,
)
nm.plot_uncertainty_bands(boot_bands, cutoff_date=cutoff_date)
```

---

## Advanced features

### Ensemble uncertainty

`nm.do_all_unc` runs the full pipeline `n_models` times with different seeds and
returns weather-normalised contributions with confidence bands plus per-model
statistics:

```python
out, stats = nm.do_all_unc(
    df=df,
    value="PM2.5",
    backend="flaml",
    feature_names=predictors,
    variables_resample=met_vars,
    n_samples=300,
    n_models=10,
    confidence_level=0.95,
    n_cores=4,
)
```

`do_all_unc` returns a 2-tuple: `(normalised_df_with_bands, model_stats_df)`.

### Multisite pipelines

```python
sites = {"Beijing": df_bj, "Shanghai": df_sh, "Guangzhou": df_gz}

results = nm.do_all_multisite(
    site_dfs=sites,
    value="PM2.5",
    backend="lightgbm",
    feature_names=predictors,
    variables_resample=met_vars,
    n_cores=4,
)
```

### Feature engineering

```python
df = nm.add_lag_features(df, cols=["PM2.5"], lags=[1, 2, 24])
df = nm.add_rolling_features(df, cols=["PM2.5"], windows=[7, 30])
df = nm.cyclical_encode(df, col="hour", period=24)
df = nm.wind_to_uv(df, speed_col="ws", dir_col="wd")
```

### Cross-validation

```python
scores = nm.cv_score(df_prep, model, feature_names=predictors, n_splits=5)
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
result = cached_normalise(key, feature_names=predictors, n_samples=300)
```

### Provenance tracking

```python
run = nm.make_run(
    model=model,
    df=df_prep,
    config={"value": "PM2.5", "backend": "flaml"},
    tags={"site": "MY1"},
)
nm.save_run(run, "runs/my1_run.joblib")
```

### I/O adapters

```python
import normet.io as nio

# ERA5 reanalysis — single-point time-series straight from the CDS
era5 = nio.fetch_era5_timeseries(
    sites={"MY1": (51.52, -0.13)},
    variables=nio.ERA5_AQ_VARIABLES_DEFAULT,
    date_from="2018-01-01", date_to="2023-12-31",
    cache_dir=".era5_cache",
)

# UK AURN network
aurn = nio.fetch_aurn_measurements(
    station="MY1", pollutant="PM2.5",
    date_from="2018-01-01", date_to="2023-12-31",
)

# OpenAQ
openaq = nio.fetch_openaq_measurements(
    location_id=12345, parameter="pm25",
    date_from="2024-01-01", date_to="2024-01-07",
)
```

---

## CLI

```bash
# One-shot normalisation from the command line
normet do-all --config config.yaml

# Decomposition
normet decompose --config config.yaml

# SCM
normet scm --config config.yaml

# Show installed version and available backends
normet info
```

---

## Dependencies

| Group | Packages |
|---|---|
| Core | `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `matplotlib` |
| AutoML | `flaml` (recommended) or `lightgbm` |
| I/O (data adapters) | `requests`, `cdsapi` |
| Parallel/large data | `dask` |
| CLI | `click`, `pyyaml` |
| GPU acceleration | `cupy` |

---

## Documentation

Full guides and the API reference are on [Read the Docs](https://normet.readthedocs.io):

- [Weather normalisation](https://normet.readthedocs.io/en/latest/guide/examples_normalisation.html)
- [Decomposition](https://normet.readthedocs.io/en/latest/guide/examples_decomposition.html)
- [Synthetic Control (SCM)](https://normet.readthedocs.io/en/latest/guide/examples_scm_guide.html)
- [Multisite pipelines](https://normet.readthedocs.io/en/latest/guide/examples_multisite.html)
- [Caching](https://normet.readthedocs.io/en/latest/guide/examples_caching.html)
- [Data adapters](https://normet.readthedocs.io/en/latest/guide/examples_data_adapters.html)
- [API reference](https://normet.readthedocs.io/en/latest/api.html)

Worked examples (Jupyter notebooks) live in [`notebooks/`](notebooks).

---

## How to cite

```bibtex
@Manual{normet-pkg,
  title        = {normet: Normalisation, Decomposition, and Counterfactual
                  Modelling for Environmental Time-series},
  author       = {Congbo Song and Contributors},
  year         = {2026},
  note         = {Python package version 0.4.0},
  organization = {University of Manchester},
  url          = {https://github.com/normet-dev/normet-py},
}
```

---

## Contributing

Contributions are welcome. Please open an issue or pull request on [GitHub](https://github.com/normet-dev/normet-py/issues), and see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT — see [LICENSE](LICENSE).
