# normet

**Meteorological Normalisation, Decomposition, and Counterfactual Modelling for Air Quality Time Series**

[![PyPI version](https://badge.fury.io/py/normet.svg)](https://pypi.org/project/normet/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/apai-sys/normet-py/actions/workflows/ci.yml/badge.svg)](https://github.com/apai-sys/normet-py/actions)
[![Build GUI](https://github.com/apai-sys/normet-py/actions/workflows/build-gui.yml/badge.svg)](https://github.com/apai-sys/normet-py/actions/workflows/build-gui.yml)
[![Docs](https://img.shields.io/badge/docs-docs%2F-blue)](https://github.com/apai-sys/normet-py/tree/main/docs)

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

Zero-shot de-weathering with the Chronos-2 foundation model, data adapters, a CLI, an
MCP server for AI assistants and a desktop app build on the same API.

![The normet desktop app after a normalisation run](https://raw.githubusercontent.com/apai-sys/normet-py/main/docs/manual/figures/ui_normalise.png)

---

## Core capabilities

| Module | What it does |
|---|---|
| `nm.prepare_data` | Imputation, date-feature engineering, train/test splitting |
| `nm.train_model` / `nm.build_model` | AutoML via FLAML or tuned LightGBM |
| `nm.normalise` | Monte Carlo meteorological normalisation (deweathering, with optional `stratify_by` and `conditional_k` on-manifold resampling) |
| `nm.wpscf` / `nm.wcwt` / `nm.invert_source_footprints` / `nm.cluster_trajectories` / `nm.screen_donor_transport` | Physical BLH-coupled Lagrangian receptor mapping, regularized source-footprint inversion, TSV trajectory clustering, and SCM spillover screening |
| `nm.decompose` / `nm.decom_emi_shapley` / `nm.shapley_interactions` / `nm.cross_fit_decompose` / `nm.decom_met_treeshap` | Emission, meteorological, order-independent time-Shapley, 2nd-order Shapley interaction, multiplicative LMDI, temporally blocked cross-fitted, and TreeSHAP decompositions (`suggest_met_groups`, `compare_attributions`) |
| `nm.rolling` | Rolling-window normalisation for trend analysis |
| `nm.run_scm` / `nm.scm` | Synthetic Control Method with selectable backends: `scm` (default, ridge-augmented), `abadie` (classic simplex), `did`, `mcnnm` (nuclear-norm matrix completion), `robust` and `mlscm`; several treated units each with its own cutoff via `nm.run_scm_staggered` (staggered adoption: per-unit effects and placebo p-values, plus the pooled mean and median, with the cross-unit permutation test on the mean — rolling-out schemes such as the UK ULEZ expansion) |
| `nm.mlscm` | ML-augmented SCM (**experimental** — see [Synthetic control](https://github.com/apai-sys/normet-py/blob/main/docs/guide/examples_scm_guide.md)) |
| `nm.HarmonicCounterfactual` / `nm.HarmonicCounterfactualResult` | Business-as-usual harmonic/calendar counterfactual for policy evaluation — day-of-week × hour interaction, fleet-renewal trend and annual Fourier terms (ridge); `evaluate_intervention` reports the policy impact with a pre-intervention validation bias and an approximate p10/p90 band |
| `nm.placebo_in_space` / `nm.placebo_in_time` / `nm.rmspe_ratio_test` | Placebo significance tests and Abadie's RMSPE-ratio test |
| `nm.uncertainty_bands` | Bootstrap / Jackknife confidence intervals |
| `nm.do_all` | End-to-end pipeline in one call |
| `nm.do_all_unc` | End-to-end pipeline with model-ensemble uncertainty bands |
| `nm.do_all_multisite` | Parallel per-site pipelines |
| `nm.pdp` / `nm.pdp_grid` | Partial-dependence values (`nm.pdp`) and their plot grid (`nm.pdp_grid`) |
| `nm.modStats` | Model performance metrics (R², RMSE, MGE [mean absolute error], …) |
| `nm.trend` / `nm.detect_trend_breaks` | Trend analyses with the block-bootstrap correction for autocorrelated records, plus the segmented (piecewise) trend for records whose level or slope changes part-way through |
| `nm.mann_kendall` / `nm.theil_sen` | Classical Mann-Kendall test and Theil-Sen slope with its analytic interval — uncorrected for autocorrelation; the block-bootstrap versions are in `nm.trend` |
| `nm.crps_from_quantiles` / `nm.crps_from_samples` / `nm.pinball_loss` / `nm.crps_observed` | Probabilistic skill scores (Gneiting & Raftery 2005): CRPS for quantile and ensemble forecasts, the pinball loss, and a point forecast's CRPS reference |
| `nm.decompose_uncertainty` | Attribute a de-weathered estimate's total spread to its source — weather (met resampling) vs model form vs their interaction — via an additive per-date variance decomposition |
| `nm.detect_events` / `nm.anomaly_scores` | Event detection on normalised residuals |
| `nm.normalise_plot` / `nm.decomposition_stack` / `nm.plot_shapley_interactions` / `nm.scm_dashboard` / `nm.plot_receptor_grid` / `nm.plot_trajectory_clusters` / `nm.plot_footprint_inversion` | Plotting helpers for normalisation, diverging decomposition stacks, Shapley interactions, SCM, and back-trajectory receptor / inversion / cluster results |
| `nm.bayesian_scm` / `nm.plot_bayesian_scm` | Bayesian (PyMC) posterior SCM with credible bands — **optional**, needs `normet[bayes]` |
| `nm.scm_diagnostics` / `nm.pre_trend_test` / `nm.plot_event_study` / `nm.loo_weight_stability` / `nm.conformal_effect_interval` | SCM fit diagnostics, pre-period parallel-trends test and its event-study figure, donor-weight stability, conformal intervals |
| `nm.donor_spatial_summary` / `nm.spatial_summary` | Where the donor pool sits around the treated unit: per-donor distance, compass bearing and sector, the pool's centroid offset and sector coverage (a reading, not a fit input) |
| `nm.Chronos2Estimator` | Zero-shot de-weathering, counterfactuals and quantiles with the Chronos-2 foundation model — **optional**, needs `normet[foundation]`; fine-tuning (LoRA or full) needs `normet[finetune]` |
| `nm.normalise_in_context` / `Chronos2Estimator.normalise_in_context` | In-Context Normalisation (ICN): future-only intervention (D1) over a configurable meteorology pool (D2) that preserves short emission events `deweather` absorbs. Returns level + paired estimates with the anchor/Monte-Carlo spread split — **optional**, needs `normet[foundation]` |
| `nm.calibrate_quantiles` | Quantile calibration for the zero-shot band: per-level conformal half-width thresholds on a rolling hold-out of forecast blocks, with a cross-validated coverage table; `counterfactual(..., calibrated=True)` takes the band through it — duck-typed on any estimator; the Chronos-2 estimator it usually calibrates needs `normet[foundation]` |
| `nm.ChronosEmbedder` / `nm.embed_multisite` | 768-D station embeddings describing each site's dynamics; `embed_multisite` embeds every site of a long-format frame in one batch — **optional**, needs `normet[foundation]` |
| `nm.silhouette_curve` / `nm.embedding_projection` / `nm.plot_embedding_diagnostics` | Diagnostics for choosing how to use site embeddings — the silhouette against the number of groups, a 2-D projection, and the two-panel figure. No cluster labels, no recommended k — the UMAP projection needs `umap-learn`; the rest uses core dependencies |
| `nm.make_run` / `nm.save_run` / `nm.generate_html_report` / `nm.report_to_markdown` | Provenance tracking and auto-generated HTML/Markdown run reports |
| `nm.make_memory` | On-disk caching for expensive pipelines |
| `normet.io` | Data adapters for ERA5, EEA, OpenAQ, Open-Meteo and the UK air-quality networks; NOAA GDAS1 downloads; HYSPLIT back-trajectory endpoints, BLH-coupled physical footprint features, and parallel runs (`read_trajectory_tdump`, `read_trajectory_endpoints`, `trajectory_endpoint_weights`, `trajectory_features`, `build_trajectory_features`, `join_trajectory_features`, `run_back_trajectories`, `load_source_regions`); table-file readers |

---

## Installation

```bash
pip install "normet[flaml]"     # with the recommended AutoML backend
```

`normet` needs at least one model backend; the core package installs only `numpy`,
`pandas`, `scipy`, `scikit-learn`, `joblib` and `matplotlib`. Other optional features
come as extras, which combine (`pip install "normet[gui,flaml]"`):

| Extra | For |
|---|---|
| `flaml` / `lgb` | AutoML (FLAML, recommended) or the lightweight LightGBM tuner |
| `dask` | Dask-backed prediction for large datasets |
| `data` | ERA5 / EEA / OpenAQ / Open-Meteo / UK air quality (ukaq) data adapters |
| `geo` | Shapely/pyshp polygon and shapefile source regions for the trajectory features |
| `stats` | `statsmodels`-based features: STL event detection and the AR/ADF lag diagnostics |
| `foundation` | Chronos-2 foundation model (zero-shot de-weathering, station embeddings) |
| `finetune` | Fine-tuning the Chronos-2 checkpoint (`Chronos2Estimator.finetune`) — adds `accelerate` and `peft` to `foundation` |
| `bayes` | Bayesian SCM (`nm.bayesian_scm`) — `pymc` and `arviz` |
| `cli` | The `normet` command-line tool |
| `mcp` | MCP server for AI assistants (Claude Code, Claude Desktop, ...) |
| `gui` | The desktop app (`normet-gui`) |
| `all` | Everything |

The development version, which the documentation tracks:

```bash
pip install git+https://github.com/apai-sys/normet-py.git
```

**Desktop app.** `pip install "normet[gui,flaml]"`, then run `normet-gui`. Ready-made
installers — `.dmg` (macOS), `.exe` (Windows), `.AppImage` (Linux) — are built by the
[`build-gui.yml`](.github/workflows/build-gui.yml) workflow (**Actions → Artifacts**);
see [`packaging/README.md`](packaging/README.md) to build one locally.

The [installation guide](https://github.com/apai-sys/normet-py/blob/main/docs/getting_started/installation.md)
lists every extra and its dependencies.

---

## Quick start

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

`do_all` returns a 3-tuple: `(normalised_df, model, prepared_df)`. The
[step-by-step workflow](https://github.com/apai-sys/normet-py/blob/main/docs/guide/workflow.md)
takes it apart and continues into decomposition and synthetic control.

---

## Documentation

The full documentation lives in **[`docs/`](https://github.com/apai-sys/normet-py/tree/main/docs)**:

- **Getting started** — [installation](https://github.com/apai-sys/normet-py/blob/main/docs/getting_started/installation.md)
  and [quick start](https://github.com/apai-sys/normet-py/blob/main/docs/getting_started/quickstart.md),
  including the bundled example datasets.
- **User guide** — the [step-by-step workflow](https://github.com/apai-sys/normet-py/blob/main/docs/guide/workflow.md),
  then [meteorological normalisation](https://github.com/apai-sys/normet-py/blob/main/docs/guide/examples_normalisation.md),
  [zero-shot de-weathering with Chronos-2](https://github.com/apai-sys/normet-py/blob/main/docs/guide/examples_foundation.md),
  [decomposition](https://github.com/apai-sys/normet-py/blob/main/docs/guide/examples_decomposition.md),
  [trend analysis](https://github.com/apai-sys/normet-py/blob/main/docs/guide/examples_trend.md),
  [synthetic control](https://github.com/apai-sys/normet-py/blob/main/docs/guide/examples_scm_guide.md),
  [multisite pipelines](https://github.com/apai-sys/normet-py/blob/main/docs/guide/examples_multisite.md),
  [caching](https://github.com/apai-sys/normet-py/blob/main/docs/guide/examples_caching.md),
  [data adapters](https://github.com/apai-sys/normet-py/blob/main/docs/guide/examples_data_adapters.md)
  and more.
- **Interfaces** — the [desktop GUI](https://github.com/apai-sys/normet-py/blob/main/docs/guide/gui.md),
  the [command-line interface](https://github.com/apai-sys/normet-py/blob/main/docs/guide/cli.md)
  and the [MCP server for AI assistants](https://github.com/apai-sys/normet-py/blob/main/docs/guide/mcp.md).
- **[API reference](https://github.com/apai-sys/normet-py/blob/main/docs/api.rst)**, generated
  from the docstrings.
- **PDF user manuals** for the desktop app, in [English](docs/manual/normet_manual.pdf)
  and [Chinese](docs/manual/normet_manual_zh.pdf).
- **Notebooks** — [`notebooks/`](notebooks) reproduces each case study of the normet
  model-description paper end-to-end on the bundled data
  ([list](https://github.com/apai-sys/normet-py/blob/main/docs/guide/notebooks.md)).

What changed in each release is in [CHANGELOG.md](CHANGELOG.md), and planned work in
the [roadmap](https://github.com/apai-sys/normet-py/blob/main/docs/roadmap.md).

---

## How to cite

```bibtex
@Manual{normet-pkg,
  title        = {normet: Meteorological Normalisation, Decomposition, and
                  Counterfactual Modelling for Air Quality Time Series},
  author       = {Congbo Song},
  year         = {2026},
  note         = {Python package version 0.3.0},
  organization = {University of Manchester},
  url          = {https://github.com/apai-sys/normet-py},
}
```

The same metadata is in [CITATION.cff](CITATION.cff), which GitHub's **Cite this
repository** button reads.

---

## Contributing

Contributions are welcome. Please open an issue or pull request on [GitHub](https://github.com/apai-sys/normet-py/issues), and see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines. Report security problems privately, as described in [SECURITY.md](SECURITY.md). Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).
