# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Lag-structure diagnostics.** `analyze_lag` (and the `LagDiagnostics`
  result) computes a target's ACF/PACF and the pre-whitened cross-correlation
  (CCF) with a meteorological driver to suggest autoregressive and predictive
  lags for `add_lag_features`. Box–Jenkins pre-whitening keeps shared
  seasonality from producing spurious CCF peaks; the CCF sign convention
  (`lag k>0` = driver leads target by `k`) matches `add_lag_features(lags=[k])`.
  New user guide (`docs/guide/feature_engineering.md`) and a runnable tutorial
  (`notebooks/feature_engineering.ipynb`).
- **HYSPLIT back-trajectory adapter** (`normet.io.trajectory`):
  `read_trajectory_tdump`, `trajectory_features`, `build_trajectory_features`,
  and `run_back_trajectories` (drives `hyts_std` end-to-end) turn `tdump`
  output into transport-aware predictors — inflow direction, transport
  distance/speed, residence time over source regions, along-path rainfall/BLH.
- **GDAS1 met download** (`normet.io.gdas`): `fetch_gdas1` / `gdas1_filenames`
  stream and cache the weekly GDAS1 (1°) ARL files from NOAA ARL's archive so
  `run_back_trajectories` can run when no local meteorology is available.

### Changed (breaking)
- **Renamed the `value`/`feature_names`/`na_rm`/`fraction` parameters to
  `target`/`covariates`/`dropna`/`train_fraction` across the entire public
  API**, extending the rename already applied to `prepare_data`/`check_data`/
  `impute_values`/`split_into_sets`. Affected functions: `normalise`,
  `normalise_auto` (and `NormaliseConfig.feature_names` → `.covariates`),
  `decompose`, `decom_emi`, `decom_met` (and `DecomposeConfig`'s `value`/
  `feature_names`/`fraction` fields), `rolling` (and `RollingConfig`),
  `mlscm`, `build_model`, `train_model`, `do_all`, `do_all_unc` (and
  `SingleConfig`/`UncConfig`), `do_all_multisite`, `decompose_multisite`,
  `cv_score`, `polar_plot`, `time_series_plot`, and the `Backend.train`
  protocol (`flaml`/`lightgbm` backends). CLI flags follow suit:
  `--value` → `--target`, `--fraction` → `--train-fraction`, and the
  features flag is now `--covariates` (kebab-case, matching
  `--split-method`). Update any code, scripts, or saved YAML configs that
  call these functions or the CLI with the old keyword/flag names.
- **Removed the xarray/NetCDF gridded ERA5 path.** `fetch_era5_at_sites`,
  `download_era5`, and the generic xarray ingestion helpers
  (`prepare_from_xarray`, `sample_xarray_at_sites`) are gone, along with the
  `[xarray]` extra (and `xarray`/`netCDF4` from `[all]`). ERA5 meteorology is
  now fetched as pre-interpolated single-point time-series via
  `fetch_era5_timeseries`, which needs only `cdsapi` — no `xarray`/`netCDF4`.

### Internal
- Repaired the pre-commit `mypy` hook (pin `numpy<2.2` so its stubs parse under
  the Python 3.10 target; migrate the pytest hook to the `pre-push` stage) and
  cleared the type errors it then surfaced across `model/train`,
  `causal/variants`, `causal/run_scm`, and `analysis/{rolling,normalise}`.
- Brought the repository into `ruff` 0.5.6 compliance (UP038 `isinstance`
  unions; import ordering and formatting across the test suite and docs).

## [0.4.0] — 2026-06-20

### Changed (breaking)
- **Dropped Python 3.9 support** (EOL since 2025-10). Minimum is now Python 3.10,
  matching the modern scientific stack (numpy ≥ 2.1, scipy ≥ 1.14, pandas 3.x,
  scikit-learn ≥ 1.7). Update `ruff`/`black`/CI targets accordingly.

### Added
- `scm_robust` — Robust Synthetic Control (Amjad, Shah & Shen 2018): HSVT
  de-noising of the donor matrix followed by (optionally ridge) regression.
  Available via `run_scm(scm_backend="robust")`.
- `scm_mcnnm` gains cross-validated `lam` selection (`cv=`) and an optional
  randomized-SVD fast path (`max_rank=`) for large panels.
- `DEFAULT_SEED` constant in `normet.utils` centralising the default random seed.

### Performance
- `scm()` now solves all timestamps from a single SVD of the donor design with
  exact leave-one-out alpha selection, instead of refitting `RidgeCV` per
  timestamp (equivalent results, large speed-up on long pre-periods).
- `normalise` auto-convergence path builds its per-date accumulator by zipping
  columns instead of `DataFrame.iterrows()`.

### Fixed
- `pip install normet[all]` now pulls the `data` adapters' dependencies
  (`requests`, `cdsapi`); previously the I/O adapters were unusable under `[all]`.

### Internal
- Data adapters (OpenAQ, EEA, DEFRA) share a single HTTP helper
  (`io/_http.py`) with timeout, exponential backoff, and HTTP 429 handling;
  EEA gained retries it previously lacked.
- Shared synthetic-control primitives (`pivot_panel`, `solve_simplex_weights`)
  extracted to `causal/_common.py`, de-duplicating `scm` / `variants`.
- Repository hygiene: notebook data artifacts gitignored; broad `except` blocks
  given debug logging on silent fallbacks.

## [0.3.0] — 2026-06-10

### Added

#### Pipelines & analysis
- `do_all_multisite` / `decompose_multisite` for parallel per-site execution.
- `multisite_apply` generic per-site dispatcher.
- `decompose(method="shap")` — single-pass per-feature additive attribution
  (FLAML via `shap`).
- `decompose_shap` direct API.
- `normalise(return_quantiles=...)` — emit quantile columns of the per-date
  resample distribution (resampling uncertainty).
- `normalise(conditional_on={...})` — counterfactual scenarios by filtering
  the resample pool (scalar / iterable / callable values supported).

#### Causal
- New SCM backends: `scm_abadie` (classic simplex), `did_baseline` (DiD
  parallel-trends), `scm_mcnnm` (Matrix Completion Nuclear-Norm).
- `BACKENDS` registry now exposes `{scm, mlscm, abadie, did, mcnnm}`.
- `scm_diagnostics` — pre-period fit (RMSE/R²/MAE/MAPE), Herfindahl index,
  effective N donors, top-k donor weights.
- `loo_weight_stability` — leave-one-donor-out drift summary.
- `conformal_effect_interval` — finite-sample sub-sampling conformal CI for
  the post-period ATT.
- `rmspe_ratio_test` — Abadie's RMSPE-ratio placebo significance test.

#### Modelling
- `ml_predict(chunk_size=...)` — FLAML predictions are now batched by default
  to avoid blowing up memory.
- `ml_predict_dask` — lazy partition-wise predict for Dask DataFrames.

#### Utilities
- Feature engineering: `add_lag_features`, `add_rolling_features`,
  `cyclical_encode`, `wind_to_uv`.
- Walk-forward CV: `time_series_cv`, `cv_score`.
- `modStats(by=...)` — time-stratified metrics; built-in tokens
  (`season`, `month`, `hour`, `weekday`, `year`, `day_of_year`).
- Caching helpers: `make_memory`, `dataframe_hash`, `config_hash`.
- Provenance: `NormetRun`, `make_run`, `save_run`, `load_run` (joblib +
  JSON sidecar archive).

#### I/O
- `normet.io` package; `prepare_from_xarray` / `sample_xarray_at_sites` for
  ingesting gridded NetCDF/Zarr data.

#### CLI
- `normet` console entry point with subcommands `do-all`, `decompose`, `scm`,
  `cv`, `info`. Supports `--config foo.yaml` for any subcommand.

#### Docs
- Sphinx site under `docs/`; readthedocs config; user guides on
  normalisation, decomposition, SCM, multisite, caching.

#### Project
- Tests: 55 tests across 14 files; CI matrix on Py 3.9–3.12 with coverage
  (`--cov-fail-under=50`) and ruff lint + format checks.
- Pre-commit config (ruff + mypy).
- Optional dependency extras: `flaml`, `shap`, `xarray`, `dask`,
  `cli`, `docs`, `dev`, `all`.

#### Plotting (#26 / #29)
- `normalise_plot(result_df, ...)` — observed vs. deweathered time series with
  optional quantile uncertainty band; supports `resample=` for daily/weekly
  aggregation before display.
- `plot_bayesian_scm(result, cutoff_date=...)` — two-panel posterior credible
  band visualisation for :func:`bayesian_scm` output.
- `plotting.__all__` now exports all six public functions.

#### Causal (#29)
- `bayesian_scm` — `weights_summary` HDI column extraction is now robust to
  arviz version differences (dynamically searches for `hdi_*` columns instead
  of relying on a hardcoded format string).


### Fixed
- `pipeline.do_all` had a stray top-of-file token that made the package
  fail to import; duplicate `aggregate=` kwarg; dead `mod_stats` computation.
- `analysis.rolling` and `analysis.decomposition` were passing
  `weather_df=None` to `normalise`, which only accepts `resample_df`.
- `causal/__init__.py` `__all__` list had missing commas, silently joining
  symbol names.

### Changed
- Top-level `__init__.py` re-exports the new symbols (~35 additions).
- `pyproject.toml` declares `scikit-learn` as a core dependency (SCM uses
  `RidgeCV`).

---

## [0.2.4] — 2025-10-29
Last release before this changelog began. See git history for details.
