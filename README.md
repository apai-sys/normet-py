# normet

**Normalisation, Decomposition, and Counterfactual Modelling for Environmental Time-series**

[![PyPI version](https://badge.fury.io/py/normet.svg)](https://pypi.org/project/normet/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/apai-sys/normet-py/actions/workflows/ci.yml/badge.svg)](https://github.com/apai-sys/normet-py/actions)
[![Build GUI](https://github.com/apai-sys/normet-py/actions/workflows/build-gui.yml/badge.svg)](https://github.com/apai-sys/normet-py/actions/workflows/build-gui.yml)
[![Docs](https://readthedocs.org/projects/normet/badge/?version=latest)](https://normet.readthedocs.io)

`normet` is a Python package for **deweathering**, **causal inference**, and **policy evaluation** on environmental time-series data. It wraps AutoML model training, Monte Carlo meteorological normalisation, and Synthetic Control Methods behind a clean, high-level API — so you spend time on science, not boilerplate.

`normet` brings together three ideas behind one API:

1. **AutoML model training** (FLAML or LightGBM) learns how a pollutant
   concentration depends on meteorology, time, and other predictors.
2. **Monte Carlo meteorological normalisation ("deweathering")** re-runs that model
   thousands of times against resampled, randomised meteorology to cancel out
   day-to-day meteorological noise and reveal the underlying emission-driven trend.
3. **Synthetic Control Methods (SCM)** answer "what would have happened
   without this policy?" by building a counterfactual from untreated donor
   units, with placebo tests and uncertainty bands.

### Contents

- [Core capabilities](#core-capabilities)
- [Installation](#installation)
- [Quick start: one-shot meteorological normalisation](#quick-start-one-shot-meteorological-normalisation)
- [Step-by-step workflow](#step-by-step-workflow)
- [Decomposition](#decomposition)
- [Counterfactual modelling (Synthetic Control)](#counterfactual-modelling-synthetic-control)
- [Advanced features](#advanced-features)
- [CLI](#cli)
- [Desktop GUI](#desktop-gui)
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
| `nm.normalise` | Monte Carlo meteorological normalisation (deweathering) |
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
pip install git+https://github.com/apai-sys/normet-py.git
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

## Quick start: one-shot meteorological normalisation

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
    value="NO2",
    backend="flaml",          # or "lightgbm"
    feature_names=predictors,
    variables_resample=met_vars,  # resample met only — not time features
    n_samples=300,
    n_cores=4,                # parallelise resampling
)

print(out.head())
nm.modStats(df_prep, model)
```

`do_all` returns a 3-tuple: `(normalised_df, model, prepared_df)`.

### Bundled example data

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

---

## Step-by-step workflow

### 1. Prepare data

```python
df_prep = nm.prepare_data(
    df=df,
    value="NO2",
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

Available backends: `scm` (augmented ridge SCM), `abadie` (classic simplex), `did` (DiD), `mcnnm` (matrix completion), `robust` (Robust SCM — HSVT de-noising, Amjad et al. 2018), `mlscm` (**experimental** — emits `ExperimentalWarning`).

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

---

## Advanced features

### Ensemble uncertainty

`nm.do_all_unc` runs the full pipeline `n_models` times with different seeds and
returns meteorologically normalised contributions with confidence bands plus per-model
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

## Desktop GUI

A native Qt desktop app covers the full workflow without writing code, and
runs on macOS, Windows, and Linux.

**Install from PyPI:**

```bash
pip install "normet[gui,flaml]"   # or normet[gui,lgb]
normet-gui                        # optionally: normet-gui mydata.csv
```

**Or download a ready-made installer** — `.dmg` (macOS), `.exe` (Windows,
Inno Setup), `.AppImage` (Linux) — built by
[`build-gui.yml`](.github/workflows/build-gui.yml)'s 3-OS matrix; grab the
latest from the workflow's **Actions → Artifacts**, or build one locally with
`packaging/macos/build_dmg.sh` (macOS) or `pyinstaller packaging/normet_gui.spec`
(any OS — see [`packaging/README.md`](packaging/README.md)).

**Main window** — the meteorological-normalisation workflow as numbered steps in a left-hand
panel (Data → Columns → Train → Normalise → Decompose → Rolling → PDP), with
results in tabs that activate as each step finishes. Model quality, and every
other result, is summarised by a traffic-light verdict banner. Step 1 exposes
the flaml estimator search list, plus "Time variables"/"Met only" one-click
toggles for the PDP variable list; Step 2 lets you choose exactly which
variables are resampled in the Monte-Carlo ("Met only" by default); Step 4's
Rolling plot shows the overall rolling mean (± spread across overlapping
windows), and an adjacent **Multi-scale decomposition** panel differences
rolling-deweathered series at increasing window widths (14/90/365 d by
default) against each other and against Step 2's full-record baseline —
isolating the meteorological residual specific to each timescale band,
analogous to wavelet detail coefficients. Includes drag-and-drop CSV
loading, recent files, config save/load, run history, CSV/HTML export, a
live log dock, and one-click synthetic example data.

**Data Studio** (🌐 toolbar button, or File → Get UK Data) — assemble a
model-ready dataset without leaving the app: browse/search every UK AURN
station by the pollutants it measures or its official site code (e.g.
"MAN3" for Manchester Piccadilly, via `fetch_aurn_site_codes`), pick a date
range, and fetch the hourly measurements together with ERA5-derived
meteorology from Open-Meteo (no API key; Copernicus CDS optional). The
merged hourly table can be saved as CSV or sent straight into Step 1.

**SCM Studio** (Analysis → Synthetic Control, or the 🧪 toolbar button) — the
counterfactual workflow on panel data: map date/unit/outcome columns, pick the
treated unit, cutoff and donor pool, choose an estimator
(`scm`, `mlscm`, `abadie`, `did`, `mcnnm`, `robust`, `bayesian`), then run
fit + diagnostics, placebo-in-space/time, jackknife/bootstrap uncertainty
bands and the all-units batch — each with a p-value / fit-quality verdict.

**Transport Studio** (🧭 toolbar button) — build transport-aware predictors
from HYSPLIT back-trajectories: parse existing `tdump` files (no external
binary needed) into per-receptor features (inflow direction, transport
distance/speed, residence time over named source regions, along-path
rainfall/boundary-layer height), preview them, then join straight onto the
loaded dataset by nearest hour — the new `traj_*` columns are auto-ticked as
predictors in Step 1. An advanced panel can also drive `hyts_std` itself for
a set of receptor times, downloading the matching GDAS1 meteorology first, if
HYSPLIT is installed locally.

Long computations run on a background thread; the window stays responsive and
tasks can be abandoned from the status bar.

---

## Dependencies

| Group | Packages |
|---|---|
| Core | `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `matplotlib` |
| AutoML | `flaml` (recommended) or `lightgbm` |
| I/O (data adapters) | `requests`, `cdsapi` |
| Parallel/large data | `dask` |
| CLI | `click`, `pyyaml` |
| GUI | `PySide6` |

---

## Documentation

Full guides and the API reference are on [Read the Docs](https://normet.readthedocs.io):

- [Meteorological normalisation](https://normet.readthedocs.io/en/latest/guide/examples_normalisation.html)
- [Decomposition](https://normet.readthedocs.io/en/latest/guide/examples_decomposition.html)
- [Synthetic Control (SCM)](https://normet.readthedocs.io/en/latest/guide/examples_scm_guide.html)
- [Multisite pipelines](https://normet.readthedocs.io/en/latest/guide/examples_multisite.html)
- [Caching](https://normet.readthedocs.io/en/latest/guide/examples_caching.html)
- [Data adapters](https://normet.readthedocs.io/en/latest/guide/examples_data_adapters.html)
- [API reference](https://normet.readthedocs.io/en/latest/api.html)

Paper-reproducibility notebooks live in [`notebooks/`](notebooks) — one per
case study of the normet model-description paper, each running end-to-end on
the bundled example data:

1. [Deweathering + adaptive convergence](notebooks/01_deweathering_and_adaptive_convergence.ipynb) — MY1 NO2, COVID lockdown signal, series-RSE stopping rule
2. [Decomposition + rolling normalisation](notebooks/02_decomposition_and_rolling.ipynb) — emission vs met components, moving-window trends
3. [ULEZ Synthetic Control](notebooks/03_scm_ulez.ipynb) — causal effect of the London ULEZ with placebo bands
4. [Transport-aware normalisation](notebooks/04_transport_aware_normalisation.ipynb) — HYSPLIT back-trajectory features and the transport contribution to PM2.5

---

## How to cite

```bibtex
@Manual{normet-pkg,
  title        = {normet: Normalisation, Decomposition, and Counterfactual
                  Modelling for Environmental Time-series},
  author       = {Congbo Song and Contributors},
  year         = {2026},
  note         = {Python package version 1.0.0},
  organization = {University of Manchester},
  url          = {https://github.com/apai-sys/normet-py},
}
```

---

## Contributing

Contributions are welcome. Please open an issue or pull request on [GitHub](https://github.com/apai-sys/normet-py/issues), and see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT — see [LICENSE](LICENSE).
