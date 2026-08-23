# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Fine-tuning: `Chronos2Estimator.finetune`.** Adapts the checkpoint to one
  site's own record and returns a *new* estimator, leaving the original on the
  pretrained weights so the two can be compared without reloading. LoRA is the
  default, because a single station's record is small next to 119M parameters,
  which is the setting full fine-tuning overfits; `mode="full"` is there for
  when it is not.

  Deliberately not `fit`. `fit` sits on the sklearn-shaped path that `do_all`
  walks, where a call costs nothing and is made freely -- silently turning that
  into a thousand optimiser steps on a GPU would be a trap, so it still trains
  nothing and a test pins that. Covariates are prepared through upstream's
  `from_list_of_dicts` with `known_covariates_names` set, because normet's
  covariates are meteorology, which is known across the forecast window; a run
  that trained them as past-only would adapt the model to a problem the package
  never poses. `mode="lora"` without `peft` installed raises rather than
  proceeding: upstream warns and silently falls back to full fine-tuning, which
  is a different and far more expensive thing than what was asked for. New
  `finetune` extra.
- **Multivariate targets: `predict_quantiles_multivariate`.** Chronos-2 takes a
  2-D target and attends across the variates, so the species measured at one
  site -- or a site and its neighbours, once they are columns of one frame --
  are forecast as one task rather than several. Returns one frame per target.
  The variates must share an index and a horizon; covariates are shared across
  them, being properties of the site rather than of the species. Upstream's
  target encoding of categoricals is only defined against a single target, so
  this path falls back to ordinal encoding -- documented rather than hidden.
- **Joint multi-site prediction: `predict_quantiles_multisite`.** Sites that do
  not share an index are separate tasks, not variates, and each keeps its own
  frame and history. `cross_learning=True` puts the batch in one group so the
  model may carry structure between them, which upstream reports helps most
  where an individual series has little history -- a newly commissioned station
  next to twenty established ones.

  Because the sharing is a *batch* property, results depend on `batch_size` and
  on which sites shared the call; both facts are in the docstring rather than
  left to be discovered. `cross_learning=False` reproduces the per-site loop to
  within float32 rounding, which the suite pins against `predict_quantiles`.
- **Categorical covariates, encoded natively.** Site type, wind sector, road
  class: anything pandas does not call numeric is now passed through as a
  category instead of being dropped by the numeric-dtype filter. Chronos-2
  target-encodes it against the observed target and maps the forecast window
  onto the categories seen in the past, so a level appearing only in the future
  is treated as unseen rather than silently renumbering the rest. One-hotting
  first would spend a covariate slot per level and discard the ordering the
  encoder recovers. Missing values are not imputed -- upstream gives NaN its own
  category, and a station with no recorded site type is a fact about the
  station.
- **Batched Chronos-2 forward passes.** `Chronos2Estimator` gained a
  `batch_size` (default 32): `deweather` now sends its Monte-Carlo draws to the
  model in batches instead of one per call, and `predict` batches its rolling
  blocks, which are independent because each conditions on observed history
  rather than on the previous block's output. `counterfactual` is unchanged --
  it rolls forward on its own median and cannot be batched.

  Measured on an L40S at a 512 h context and 48 h horizon this is worth ~3x from
  eight draws upward (0.10 s to 0.03 s at 8; 1.67 s to 0.55 s at 128). On a
  single-threaded CPU it is worth nothing (0.96-1.02x across 2-16 draws): the
  samples there are compute-bound, not dispatch-bound. Results are unchanged --
  `predict` is deterministic, and batching only reorders float32 accumulation
  (~1e-5).

  `n_samples` defaults are deliberately *not* device-dependent, so the same
  script and seed give the same answer on a laptop and on a cluster; the GPU
  headroom is documented instead, for callers to spend by hand.
- **Zero-shot meteorological decomposition.**
  `decompose(method="meteorology", backend="chronos-2")` fixes one meteorological
  feature at a time and takes successive differences, structurally identical to
  `decom_met` with `deweather` in place of `normalise`. Without fitted
  importances the order comes from each feature's individual covariate
  sensitivity; `variable_order` pins it.

  `method="emission"` is refused on this backend rather than served. That
  decomposition isolates trend, seasonal, weekly and diurnal components by
  resampling `date_unix` / `day_julian` / `weekday` / `hour`, which works because
  an AutoML model sees them as ordinary features. Chronos-2 conditions on the
  target's own history and `deweather` never resamples history, so a repeating
  calendar signal survives every draw. Measured with the calendar encoders
  supplied as ordinary covariates -- the most favourable setting -- covariate
  sensitivity came to 0.53% for a time index, 0.75% weekly, 1.50% diurnal and
  2.47% for all six together, against 4.78-10.72% for meteorology in the same
  runs. The diurnal signal was the largest injected component of all (amplitude
  15 against a meteorological scale of 10 and a series SD of 18.5), so
  attribution runs opposite to signal size: meteorology is irregular and must be
  read from the covariate channel, while calendar cycles repeat and can be read
  from history. Components would come back near zero and read as "no trend"
  rather than "not separable". `normet decompose --backend chronos-2` accepts the
  backend; `cv` still does not, having nothing to train.
- **Zero-shot `do_all`.** `do_all(..., backend="chronos-2")` skips the training
  step: nothing is fitted, the checkpoint is loaded and the de-weathering runs
  through `Chronos2Estimator.deweather`. The three-tuple return shape is
  unchanged, but the model slot holds the loaded estimator, and `model_config`
  is forwarded to its constructor (`device`, `context_length`,
  `prediction_length`) since the AutoML search settings have nothing to act on.
  `n_samples` now defaults to `None` and resolves per backend -- 300 for the
  AutoML backends, 8 for Chronos-2, where each sample is a full transformer
  forward pass rather than a tree-ensemble call. An explicit value always wins.
  `chronos-2` is deliberately *not* registered in `backend_registry`, whose
  contract is train/save/load; `normet do-all --backend chronos-2` accepts it,
  while `decompose` and `cv` do not advertise a zero-shot path they lack.
- **`embed_multisite`** (`normet.pipeline`) connects `ChronosEmbedder` to the
  multi-site drivers. `embed_stations` wants one column per site while
  multi-site frames here are long-format, so the two had no meeting point; this
  pivots the frame, embeds every site in one batched pass and returns
  `{site: 768-D vector}` keyed by the caller's own site values.

  What to do with those vectors is left to the caller. A `cluster_multisite`
  wrapper (KMeans plus a 2-D projection) was written and then cut before
  release: nothing in the package consumed its labels, and it fixed choices that
  belong to whoever is doing the analysis -- the metric (Euclidean KMeans on
  un-normalised 768-D embeddings is one opinion among several), the number of
  groups, and how to select it. Those are a few lines of scikit-learn to write
  and a poor thing to inherit as an API.
- **`to_indexed_frame`** (`normet.foundation`) puts a `prepare_data`-shaped
  frame on the gap-free DatetimeIndex Chronos-2 needs -- moved out of the GUI so
  `do_all` and the window share one implementation.
- **Chronos-2 foundation estimator** (`normet.foundation`): `Chronos2Estimator`
  wraps Amazon's Chronos-2 time-series foundation model behind a
  covariate-conditioned API — `predict_quantiles` (21 native quantile levels),
  `deweather` (meteorological normalisation by marginalising the covariate
  channel), `counterfactual` (business-as-usual projection that reports its own
  pre-intervention hold-out bias), and `covariate_sensitivity` (a diagnostic
  that shows whether the forecast responds to meteorology at all, rather than
  quietly autoregressing). Meteorology enters through Chronos-2's
  `past_covariates`/`future_covariates` channels; Chronos-1/Bolt checkpoints are
  rejected at construction because they silently drop covariates.
  Guard rails: `IrregularIndexError` on a non-uniform time index (Chronos-2
  reads position as time, so dropped hours shift the series against its own
  calendar covariates) and `InsufficientContextError` on a conditioning window
  that is mostly missing (Chronos-2 masks NaNs, so an empty context otherwise
  returns a forecast scaled to nothing). `to_regular_index` and
  `add_calendar_covariates` prepare a frame for both.
  Also `ChronosEmbedder` for zero-shot 768-D station embeddings, likewise on
  Chronos-2 (`Chronos2Pipeline.embed`) and likewise refusing
  Chronos-1/Bolt checkpoints — the two families return differently shaped
  embeddings from different encoders, so vectors from a mix of both cannot be
  compared. Every series is cut or NaN-padded to the same context
  length before batching: Chronos-2 left-pads a ragged batch, which changes the
  patch count and shifts a station's pooled embedding depending on which other
  stations shared the batch (0.30 max abs difference on a 300-point series next
  to a 512-point one). An all-missing station raises `InsufficientContextError`
  rather than embedding a masked-out series into a confident-looking vector.
  Requires the new `foundation` extra: `pip install normet[foundation]`.
- **Harmonic counterfactual** (`normet.counterfactual`): `HarmonicCounterfactual`
  estimates a business-as-usual baseline from calendar structure alone --- a
  168-cell day-of-week x hour-of-day interaction matrix, a linear fleet-renewal
  trend and 4th-order annual Fourier harmonics, fitted with ridge ---
  and `evaluate_intervention` reports the impact of a policy step against it
  together with a pre-intervention validation bias, so a baseline that cannot
  reproduce the months *before* the intervention says so. On a synthetic series
  carrying an injected 40% cut it recovers -40.6% with a 0.9% validation bias.
  Note the p10/p90 band is `bau +/- 1.645 * sigma` from the training residuals:
  it ignores parameter uncertainty and assumes homoscedastic, serially
  independent residuals, neither of which holds for hourly air quality, so read
  it as an indication rather than a calibrated interval.
- **Physics-informed graph models** (`normet.physics`): `PhysicsGraphBuilder`
  builds a static k-NN geospatial graph and a dynamic wind-guided directed
  advection graph over an irregular monitoring network; `build_pi_stgnn` and
  `build_adr_pde_loss` construct a dual-graph spatio-temporal GNN and an
  advection-diffusion-reaction mass-conservation loss. Torch is imported lazily,
  so `import normet.physics` and `PhysicsGraphBuilder` work without it; the
  neural pieces need the new `physics` extra: `pip install normet[physics]`.
  `ADR_PDE_Loss` takes `direction="upwind"` (default) or `"outflow"`.
  `A_wind[i, j]` weights `j` downwind of `i`, so summing over `j` untransposed
  charges the advection term to the *source* node and leaves a downwind receptor
  with exactly zero --- fine as an outflow formulation, wrong for the
  mass-conservation residual at a receptor, which is what `upwind` computes.
- **Chronos-2 in the GUI.** The Backend combo on the training tab gains a
  `chronos-2` entry, greyed out with an install hint when the `foundation` extra
  is absent. Picking it hides the training controls that a zero-shot model has
  nothing to act on --- time budget, estimator search space, train/test split,
  seed --- and retitles the button to "Load Chronos-2"; the frame is put on a
  regular hourly index automatically, and the model tab shows the
  covariate-sensitivity verdict in place of the parity plot and feature
  importances that do not exist here. De-weathering runs through
  `Chronos2Estimator.deweather(schema="normet")`, so the existing plot and
  report paths apply unchanged; decomposition, rolling windows and PDP stay
  disabled, as none of them has a Chronos-2 equivalent. The "Samples" count
  swaps between backends (300 for the AutoML backends, 8 for Chronos-2) because
  a Monte-Carlo sample is a tree ensemble call in one case and a full
  2048-context transformer forward pass in the other --- carrying 300 across
  would make a single normalisation run for days on CPU. The normalisation
  panel also names how many leading rows `deweather` seeded with the observed
  values (Chronos-2 has no history to condition on until `context_length` rows
  have gone by), shades them on the plot and takes the reported
  mean(normalised - observed) over the projected rows only --- otherwise a short
  record shows a curve that is mostly a copy of its input and an effect diluted
  toward zero by rows that could not have differed.
- **`resolve_device`** (`normet.foundation`) picks CUDA, else Apple Silicon's
  Metal backend (`mps`), else the CPU, and both `Chronos2Estimator` and
  `ChronosEmbedder` route through it. An accelerator chosen automatically falls
  back to the CPU with a warning if the pipeline will not load on it; a device
  named explicitly does not, so torch's own error surfaces instead of a silent
  downgrade. `.device` reports where the model actually landed. Exposed as
  `--device` on `normet deweather` and as a **Device** selector in the GUI.
- **`normet deweather` CLI command.** Zero-shot meteorological normalisation
  with Chronos-2 from the command line: rebuilds the time grid, reports the
  covariate-sensitivity diagnostic (and warns when the model turns out to be
  autoregressing) before committing, names how many leading rows were seeded
  rather than projected, and writes the result on `normalise`'s schema. Needs
  the `foundation` extra. `normet info` now reports `chronos-forecasting`,
  `torch` and `PySide6` under `optional`, plus a `backends` key.
- **User guide for the foundation models** (`docs/guide/examples_foundation.md`)
  and a README section covering when the zero-shot path is the right tool and
  when it is not.
- **Foundation results on normet's schema.** `to_normet_frame` renames a
  foundation-model frame onto the `date`-indexed `observed`/`normalised`/`qNNN`
  shape `normalise` emits, which is what `normalise_plot` and the HTML report
  dispatch on. `Chronos2Estimator.deweather(..., schema="normet")` and
  `CounterfactualResult.to_normet_frame()` apply it directly, so a de-weathered
  or counterfactual series plots and reports through the existing path.
- **UK air quality adapter** (`normet.io.ukaq`): `list_ukaq_stations` /
  `fetch_ukaq_measurements` cover all six UK networks (AURN, AQE, SAQN, WAQN,
  NI, LMAM — around 1500 stations) from the openair `.RData` archives via
  `source="aurn"`/`"aqe"`/`"saqn"`/`"waqn"`/`"ni"`/`"local"`, whole calendar
  years, or DEFRA's live SOS API (AURN only, near-real-time rolling window)
  via `source="aurn_live"` — both behind the same interface and schema
  (`aurn_live` rows leave `site_type`/`start_date`/`end_date` as `NaN`,
  which the SOS API does not carry).
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
- **`pandas>=2.0`** (was `>=1.5`), for `format="mixed"` in
  `normet.utils._time.to_datetime_coerced`. On 1.5 a format inferred from the
  first value was applied to every other one, so an ISO date followed by
  `"01/02/2024"` was coerced to `NaT` -- silently, with no warning at all. Date
  columns are now parsed value by value, which also removes the need for the
  warning suppression that entry previously carried.
- **Removed `normet.io.defra`** (`fetch_aurn_measurements`, `list_aurn_stations`,
  `fetch_aurn_site_codes`, `AURN_POLLUTANT_CODES`) — folded into
  `normet.io.ukaq` as `source="aurn_live"` instead (see Added, above); the two
  are complementary data sources, not one superseding the other.
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

### Fixed
- **`generate_html_report` retained every figure it drew.** The report builds
  its own plot, serialises it to an inline PNG and has no further use for it,
  but pyplot keeps each figure alive until closed -- so generating a report per
  site in a loop accumulated them all. `_auto_plot` also orphaned its figure
  when the plotting call raised, since the figure is created first and the
  handler returned `None`. Both are closed now. Figures passed in through
  `extra_plots` belong to the caller and are deliberately left open.
- **Warning noise in the test suite.** Nine call sites parsed user-supplied date
  columns with `pd.to_datetime(..., errors="coerce")` and no format. When the
  first value is unparseable pandas cannot infer one, falls back to per-element
  dateutil parsing and says so -- which is exactly the path these callers are
  built for, since each checks the resulting `NaT` values on the next line.
  They now share `normet.utils._time.to_datetime_coerced`, which silences that
  one message (and only that one). arviz's import-time `FutureWarning` about its
  own upcoming refactor is filtered in `pyproject.toml` alongside the existing
  pandas and joblib entries. Tests now close their figures after each case, so
  matplotlib's 20-figure alarm no longer fires against whichever test happens to
  cross the threshold -- a target that moved whenever tests were reordered.
- **Apple Silicon was never used.** Device auto-selection was
  `"cuda" if torch.cuda.is_available() else "cpu"`, so every Mac ran Chronos-2 on
  the CPU however capable its GPU. De-weathering spends one full forward pass per
  Monte-Carlo sample, which made that the difference between minutes and hours.
- **CLI `--backend` offered only `flaml`.** The choice list on `do-all`,
  `decompose` and `cv` was a hand-maintained literal that had gone stale long
  after the `lightgbm` backend was registered, so a backend the library
  supported was unreachable from the command line. It now comes from
  `backend_registry.available`, with a test asserting the two agree.
- **CLI config files could not supply list-valued options.** `covariates:
  [t2m, blh]` in a YAML config reached `_split_csv` as a list and raised
  `AttributeError: 'list' object has no attribute 'split'`; only the
  comma-separated string form worked. Both are accepted now.
- **FLAML backend accepts `custom_hp`** in `model_config`, to bound an
  estimator's search space (e.g. LGBM `num_leaves`, default range
  `[4, 32768]`) rather than only its search effort (`time_budget`/`max_iter`).
  Needed after an AutoML fit committed to `num_leaves` in the thousands —
  found at the same `best_iteration` as fits that landed on tiny models, so
  shrinking the budget alone would not have prevented it.

### Internal
- **The type-checking stack is pinned, and `warn_unused_ignores` is off.** The
  strict posture below went red on CI with thirteen errors that did not
  reproduce locally; the same tree under a second (numpy, pandas-stubs) pairing
  gave fourteen, sharing only four with CI's thirteen. The error set is a
  function of the stub and numpy versions rather than of the code, so
  `pandas-stubs` is now pinned exactly and that flag is off -- an ignore that is
  load-bearing under one pairing is "unused" under the next, and the package
  supports pandas>=2.0 and numpy across a major version boundary.

  Most of what the disagreement surfaced was worth fixing anyway:
  `DatetimeIndex.view(int64)` (deprecated in pandas 2.2+) became `astype`,
  several `to_numpy()` calls gained `dtype=float` so a datetime index could not
  leak into the inferred element type, `GroupBy.quantile` is passed an array
  rather than a list, `np.issubdtype` -- which cannot read an `ExtensionDtype`
  at all -- became `pd.api.types.is_datetime64_dtype`, and `pivot_table` is
  given a key instead of an `Index` object.
- **mypy runs in a strict posture.** The package ships `py.typed`, so a missing
  annotation is a missing promise; `pyproject.toml` now turns on the `--strict`
  set rather than the handful of flags it had before, and the 66 errors that
  surfaced are cleared. Most were mechanical: 25 unannotated `**kwargs`,
  inner helper functions (`_first_attr`, `_stem`, `_one_placebo`,
  `_one_jackknife`, `fit_ridge`), the `_import_lightgbm` / `_import_flaml_automl`
  shims, and seven `type: ignore` comments that had outlived the pandas and
  numpy stub versions they were written against.

  Two flags are deliberately off, for the same reason. `warn_return_any` has
  ~100 hits and `disallow_any_generics` ~94, almost all of them a pandas or
  numpy call that upstream itself types as `Any`, or an `np.ndarray` written
  without its dtype parameters. Satisfying them means a `cast()` around nearly
  every DataFrame operation and `npt.NDArray[np.float64]` spelled out across
  ~40 files -- casts that assert rather than check, silencing the checker
  without telling anyone whether the dtype is what we claim.
  `disallow_untyped_decorators` is off for `normet.cli` alone: `click` is an
  optional dependency reached through `require()`, so it is an `Any`-typed
  local and all 56 `@click.option` decorators read as untyped. The flag is
  asking for something that pattern cannot give; it stays on everywhere else.

  The rest of `--strict` earns its place. It is what flagged
  `cfg.get(k) if cfg.get(k) is not None else default` in the CLI, which reads
  the dict twice -- harmless at runtime, but the second read is the one a
  checker sees, so the `None` branch it appears to exclude is still in the
  type. Replaced by `_cfg_float` / `_cfg_int`, which read once and coerce, so
  a YAML config supplying `"0.75"` or `"7654321"` now works where it used to
  reach `do_all` as a string.
- **Tests for `prepare_panel` and `scm_all`**, which were public API with no
  coverage at all (11% and 29% of their statements). Both are about what they
  refuse rather than what they compute: `prepare_panel` screens a ragged panel
  before `scm()`'s ridge fit, which drops any date where *any* unit is missing,
  so a few sparse donors can collapse the sample to nothing without raising --
  and the two silent-empty-result traps its comments name (tz-aware input
  against tz-naive bounds, sub-daily input on a daily grid) now have
  regressions. `scm_all` must lose one failed unit and not the batch. Total coverage
  crosses the 80% mark it had been short of (78.7% to 80%, 493 tests).
- **The deprecation policy is in force**, no longer proposed wording. It covers
  the top-level public API: a symbol marked for removal warns for at least one
  minor release, says so in its docstring and here, names its replacement, and
  is only removed in a major release. A symbol that has never appeared in a
  release is explicitly *not* covered -- no downstream code can depend on it
  yet, so a warning cycle would protect nobody and would ship the mistake
  instead of fixing it. See `docs/roadmap.md`.
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
