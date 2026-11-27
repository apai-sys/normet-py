# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Optional Rust kernels for Theil–Sen, CRPS and trajectory angles**:
  `nm.theil_sen(..., backend="rust")` runs the same estimator through the
  `normet_rust` extension built from `rust/normet-python` on top of the pure
  `normet-core` crate (shared with the R wrapper via extendr), returning
  bit-identical slopes and Sen intervals without scipy's quadratic
  temporaries at any record length; `nm.crps_from_quantiles(...,
  backend="rust")` and `nm.crps_from_samples(..., backend="rust")` move the
  exact per-row CRPS integration (about twenty temporary arrays per row in
  Python) into the same kernel, with bit-identical scores;
  `nm.cluster_trajectories(..., metric="angle", backend="rust")` computes
  the Sirois–Bottenheim pairwise angle matrix in the kernel and returns
  bit-identical distances, cluster labels, centroids and TSV (numpy may
  dispatch SIMD `acos` on x86 builds, so parity is pinned per target). The
  default `backend="python"` paths are untouched, and the extension is
  imported only when a caller opts in.
- **Agent plugin and remote MCP transport**: `normet-plugin`
  (`normet.agent.plugin`) installs the normet MCP server, guardrails and four
  skills (`deweather-workflow`, `deweather-model-choice`, `scm-intervention`,
  `transport-receptor`) into Antigravity, Claude Code, DeepSeek Harness
  (`dsh`), OpenCode and OpenAI Codex, in user or project scope, with `--uvx`,
  `--roots`, `--env`, `--toolset core`, and remote `--url` / `--auth-token`
  options; `validate`, `status`, `doctor`, `print-config`, `prune` and
  `uninstall` manage the installations. A ready-made DeepSeek Harness bundle
  (`plugin/dsh/`, with its manifest, locales, icon and detail page) presents
  the server as one entry in DSH's **Plugins**. `normet-mcp` gains
  `--toolset core`
  (the 14 tools of the standard workflow, out of 26) and
  `--transport streamable-http` with the `--host`, `--port`, `--path`,
  `--auth-token` and `--stateless` options for a remote GPU or HPC server,
  backed by the new `JobStore.prune()` for retiring old job folders.
- **First-principles normalisation, chemical conservation, causal-transport coupling, and benchmarking**:
  - Orthogonal partially linear Double-ML meteorological normalisation (`nm.dml_normalise`, `nm.DMLNormaliseResult`) separating low-frequency emission trajectories from nonlinear meteorological effects while excluding `date_unix`/`year` from tree split features.
  - Variance-reduced Monte Carlo control variates (`NormaliseConfig(control_variates=True)`) and seam-free Markov-bridge trajectory block resampling (`NormaliseConfig(block_mode="markov")`).
  - Edge-preserving surprise-weighted anchor aggregation (`ICNDesign(surprise_tau=...)` keeping `ICNDesign.v1()` hash unchanged), offline `ICNDrawBank.save()` / `ICNDrawBank.load()`, and batched in-context group Shapley decomposition (`nm.icn_decompose`).
  - Photochemically conserved $\mathrm{O_x\text{–}NO_x}$ deweathering (`nm.normalise_ox_invariant`, `nm.ConservedOxResult`), precursor-regime-conditioned deweathering (`nm.normalise_regime_conditioned`, `nm.RegimeConditionedResult`), and causal ODE residence-time accumulation features (`add_met_physics_features(..., include_memory=True)`).
  - Multi-coalition fused resampling in `decom_met` / `shapley_interactions` eliminating repeated data prep and RNG sampling across $2^k$ coalitions with bit-identical results.
  - Two-stage uncertainty-propagated SCM (`nm.deweathered_scm`, `nm.DeweatheredSCMResult`), dynamic downwind plume spillover-adjusted SCM (`nm.transport_adjusted_scm`, `nm.TransportSCMResult`), and joint multi-receptor footprint inversion (`nm.invert_multisite_footprints`, `nm.MultisiteInversionResult`).
  - Built-in synthetic ground-truth benchmarks (`nm.make_synthetic_benchmark`, `nm.evaluate_benchmark`, `nm.SyntheticBenchmark`) covering `F1_additive`, `F2_multiplicative`, `F3_titration`, `F4_nonstationary`, and `F5_transport`, plus a macOS `KMP_DUPLICATE_LIB_OK` OpenMP coexistence guard between LightGBM and PyTorch.
- **Counterfactual separability audit and ICN exactness upgrades**:
  - `nm.counterfactual_separability_audit` reports the factual-only residual-correction score (Appendix I Eq. 66, out-of-fold), an interaction lower bound in raw and `log1p` coordinates, floor-saturation rate and a recommended coordinate; `nm.diagnose_normalisation` runs it whenever a factual series is available and warns on high RC-score, meteorology–emission interaction and boundary saturation.
  - `ICNDesign(paired_mode="auto|additive|ratio|log_ratio|rank")` and `point_estimate="median|quantile_mean"`: rank transport moves the observed value through each anchor's factual and counterfactual quantile fans, keeping multiplicative shocks without clipping; the quantile-mean point estimate removes the median's downward bias under skewed predictive fans. Both default to the pre-existing behaviour, and the `ICNDesign.v1()` hash is unchanged.
  - `ICNResult.factual` (aggregated factual baseline, also via `to_normet_frame(..., include_factual=True)`) and `nm.debias_icn_residuals`, an out-of-fold residual corrector (ridge or ExtraTrees, RC-score gated) that shifts `level`/`paired`/`factual` by the estimated action contrast without re-running the foundation model.
  - `past_only_covariates` on `normalise_in_context`, `normalise_in_context_multivariate`, `normalise_in_context_multisite` and `icn_decompose`, and `normalise_ox_invariant(..., method="icn")` with optional `icn_estimator` / `icn_design` / `icn_pipeline` to run the conserved Ox–NOx deweathering on Chronos-2 in-context predictions (the default ICN design shortens `min_context` for records under `context_length + block_h`).
  - Deepened counterfactual world-model diagnostics and evaluation:
    - Extended `nm.counterfactual_separability_audit` and `nm.diagnose_normalisation` with 1D Wasserstein-2 interaction bounds (`interaction_w2_lb_raw`, `interaction_w2_lb_log`), moving-block bootstrap 95% RC upper bound (`rc_score_linear_upper95`), chronological last-block RC score (`rc_score_last_block`), continuous shifted-power coordinate scan (`optimal_power_gamma`, `interaction_rho_lb_optimal_power`), residual non-persistence share (`residual_fresh_share`), ceiling saturation rate (`ceiling_saturation_rate`), and single-feature RC/interaction attributions (`rc_by_feature`, `rho_lb_by_feature`, `top_rc_feature`, `top_interaction_feature`).
    - Unified `nm.icn_decompose` to honor `ICNDesign.paired_mode` and `ICNDesign.point_estimate`, and added `ICNDrawBank.operating_window(...)` to score factual separability across horizon lead-time bins.
    - Added paired action-contrast variance decomposition (`estimate="paired"`, `paired_mode="auto"`) to `nm.decompose_uncertainty` / `UncDecomposition`.
    - Added `F6_interaction_dial` (`interaction_lambda`, `floor_clip_quantile`) and `nm.benchmark_interaction_diagnostics` to `normet.eval.benchmarks`, plus `cf_r2`, `factual_copy_r2`, and optional `shuffle_control` in `nm.evaluate_benchmark`.
    - Added hindsight episode counterfactuals (`nm.hindsight_episode_counterfactual`, `nm.HindsightEpisodeResult`) and persistence-gated AR(1) residual carryover (`residual_carryover=True`) plus pre-period separability checks in `HarmonicCounterfactual.evaluate_intervention`.
- **First-principles surfaces in the CLI, MCP server, and desktop GUI**:
  `normet do-all` accepts `--normalise-method dml` (with `--dml-cutoff-days`,
  `--dml-n-harmonics`), `--control-variates`, and `--block-mode markov`
  (`--block-hours`), and a YAML config now forwards every key that names a
  `SingleConfig` field instead of the previous fixed handful; `nm.do_all` /
  `nm.do_all_unc` forward the same settings. New `normet normalise-ox`,
  `normalise-regime`, `icn-decompose`, `shapley-interactions`, `benchmark`,
  `invert-footprints`, and `met-features` subcommands, mirrored by MCP tools
  (`normet_start_normalise_ox`, `normet_start_normalise_regime`,
  `normet_start_icn_decompose`, `normet_start_shapley_interactions`,
  `normet_make_benchmark`, `normet_invert_multisite`,
  `normet_make_met_features`). `normet deweather`
  gains `--surprise-tau`, `--save-bank`, and `--bank`, and `normet scm` routes
  `--backend deweathered|transport` with `--covariates`, `--deweather-method`,
  and `--spillover-exposure`. The GUI gains a Normalise engine selector
  (standard / DML / conserved Ox / regime-conditioned), control-variate and
  block-mode toggles, ICN surprise weighting and draw-bank Save/Load (the raw
  draws outlive the GPU run), an ICN decomposition engine with
  pairwise-interaction readout, synthetic-benchmark load/score menu actions,
  File → Add met physics features… (ventilation, VPD, wind, optional causal ODE
  memory per panel unit), the two staged SCM backends in SCM Studio, and
  multi-receptor joint inversion in Transport Studio.
- **Chronos-2 core primitives and first-principles operators** (`normet.foundation`):
  - `past_only_covariates` on `Chronos2Estimator` (constructor, `predict`,
    `predict_quantiles*`, `counterfactual`, `finetune`, `finetune_multisite`)
    so historical co-pollutants and activity series condition the context
    without appearing in `future_covariates` (and are masked post-intervention
    during counterfactual rollouts to prevent leakage).
  - `counterfactual(..., rollout="quantile")` using Chronos-2's native
    multi-patch forecast and 9-path quantile unrolling instead of median-only
    feedback, plus `CounterfactualResult.to_scm_frame()`.
  - `Chronos2Estimator.save()` / `load()` (pipeline weights or PEFT LoRA
    adapter + `normet_estimator.json` metadata sidecar) and multi-site domain
    adaptation via `Chronos2Estimator.finetune_multisite()`.
  - Batched `BatchICLPipeline` execution in `Chronos2ICLPipeline` and
    `normalise_in_context()`, `ICNDesign.v1()` frozen preset, external/reversed
    `MetPool.from_frame()` / `.reversed()` support via `pool=`, aggregated
    counterfactual `quantiles=` output, and `ICNResult.to_normet_frame()`.
  - Zero-shot covariate-conditioned Synthetic Control (`nm.chronos_scm` /
    `Chronos2Estimator.synthetic_control()`) using untreated donors and optional
    meteorology without pre-period parameter fitting.
  - Meteorology-conditioned probabilistic anomaly detection
    (`nm.chronos_anomaly_scores`, `nm.detect_chronos_events` /
    `Chronos2Estimator.anomaly_scores()`, `.detect_events()`, plus `method="chronos"`
    in `nm.anomaly_scores` / `nm.detect_events`) scored by PIT and tail
    surprise, supporting optional static split-conformal calibration and causal
    `AdaptiveConformalCalibrator` (ACI).
  - Log-space additive pairing (`ICNDesign(log_transform=True)`), empty-`met_cols`
    identity fast path, `ICNResult.paired_quantiles`, reusable zero-inference
    `nm.ICNDrawBank` / `nm.build_icn_draw_bank()`, joint multivariate ICN
    (`nm.normalise_in_context_multivariate`), and cross-learning multi-site ICN
    (`nm.normalise_in_context_multisite`).
  - Patch-level latent trajectory inspection (`ChronosEmbedder.embed_patches()`
    with exact context-patch special-token trimming, `.patch_regime_shifts()`,
    `pooling="mean"|"reg"`, and joint `embed_multivariate()`), plus
    pre-cutoff embedding donor ranking (`nm.select_donors_by_embedding()`).
  - White-box attention attribution (`nm.chronos_attention_attribution` /
    `Chronos2Estimator.attention_attribution()`), future-only in-context Partial
    Dependence (`nm.chronos_pdp` / `Chronos2Estimator.pdp()`, dispatched via
    `nm.pdp()`), and covariate-conditioned bidirectional gap imputation
    (`nm.chronos_impute` / `Chronos2Estimator.impute()` with optional
    conformal `calibration=`).
  - Rolling probabilistic verification and causal calibration closure
    (`nm.chronos_backtest` / `Chronos2Estimator.backtest()` scoring CRPS,
    pinball loss, PIT uniformity, and coverage by lead with optional prefix
    conformal calibration), zero-inference ICN variance partitioning
    (`nm.decompose_icn_uncertainty` / `ICNDrawBank.decompose_uncertainty()`),
    joint multivariate counterfactual BAU projections
    (`Chronos2Estimator.counterfactual_multivariate()`), and
    `nm.CovariateSensitivityCallback` to guard fine-tuning against covariate
    collapse.
  - Single-checkpoint in-process fast paths across `nm.do_all_multisite`,
    `nm.decompose_multisite`, `nm.do_all_unc(..., backend="chronos-2")`,
    `nm.placebo_in_space`, `nm.placebo_in_time`, `nm.scm_all`, and
    `nm.run_scm_staggered(..., scm_backend="chronos")` (eliminating duplicate
    per-worker checkpoint loads), explicit validated `min_past` / `callbacks` on
    `finetune()` / `finetune_multisite()`, and GUI helpers
    (`normet.gui._chronos.deweather` `method="icn"`, `save_estimator`,
    `load_saved_estimator`).
  - Full-stack Chronos-2 integration across `nm.run_scm(..., scm_backend="chronos")`,
    `nm.do_all(..., backend="chronos-2", model_config={"method": "icn", ...})`,
    `nm.decompose(..., backend="chronos-2", model_config={"method": "icn"}, resample_df=...)`,
    `normet deweather --log-transform`, `normet scm --backend chronos`, and
    MCP/agent tool `normet_start_deweather` (`Tools.start_deweather`).
  - CLI: `normet deweather --method icn --estimate level|paired` exposes ICN-v1,
    and `normet do-all` / `normet decompose` preserve `resolve_n_samples()`
    defaults (`8` for `chronos-2`, `300` for AutoML) when `--n-samples` is
    omitted.
  - Plotting, reporting, CLI, Agent/MCP, and GUI closure:
    `nm.pdp_grid()` renders native `[pdp_lower, pdp_upper]` quantile envelopes
    (with `pdp_mean ± pdp_std` fallback); added `nm.plot_attention_attribution()`
    and `nm.plot_imputation()`; `report._auto_plot()` automatically renders
    imputation, PDP, and backtest lead profiles; added `normet impute`,
    `normet backtest`, `normet events`, and `--model-path` on `deweather` /
    `scm`; added `Tools.start_impute`, `Tools.start_backtest`,
    `Tools.start_events`, their readings and `imputation` figure, plus MCP
    tools `normet_start_impute`, `normet_start_backtest`, `normet_start_events`;
    wired Chronos-2 PDP, Normalise ICN options (`deweather|icn`,
    `level|paired`, `log1p`), checkpoint Save/Load, and SCM Studio `chronos`
    backend in the desktop GUI.
- **Causal inference & counterfactual scientific upgrade (`normet.causal`, `normet.counterfactual`, `normet.plotting`)**:
  - Correctness & identification fixes: cutoff-split pre/post interpolation and
    `max_gap` in `prepare_panel`, pre-cutoff truncation (`< cutoff_date`) in
    `placebo_in_time`, optional `max_pre_mspe_ratio` screening in
    `placebo_in_space`, zero-LAPACK Newey–West HAC standard errors in
    `pre_trend_test`, pivotal test inversion in `effect_bands_space`, exact
    subsampling rescaling / donor-cloned replacement bootstrap and true
    moving-block pre-period resampling in `uncertainty_bands`, and $G$-tuple
    cross-unit permutation tests with not-yet-treated donor support and
    dynamic event-study tables in `run_scm_staggered`.
  - Econometric estimators & finite-sample inference: leak-free Ben-Michael et al.
    (2021) Ridge ASCM (`nm.scm_ascm`, `scm_backend="ascm"`), Synthetic
    Difference-in-Differences (`nm.sdid`, `scm_backend="sdid"`), demeaned and
    Elastic-Net options in `scm_abadie`, contiguous block-CV default in
    `scm_mcnnm`, circular shift option in `conformal_effect_interval`, full
    refitting Chernozhukov–Wüthrich–Zhu (2021) `nm.conformal_inference`, and
    Rambachan–Roth (2023) `nm.honest_did_bounds` + `nm.plot_honest_did`.
  - Atmospheric domain & probabilistic enhancements: donut-hole and
    Lagrangian transport spillover screening (`nm.screen_donors`,
    `nm.donut_hole_sensitivity`), intercept + AR(1) residuals + full posterior
    predictive intervals + ArviZ convergence diagnostics in `nm.bayesian_scm`,
    and held-out validation, time-blocked `alpha_ridge="auto"` CV, optional
    meteorological covariates, HAC SE, and moving-block bootstrap uncertainty
    in `nm.HarmonicCounterfactual`.
- **Scientific attribution & decomposition upgrade (`normet.analysis.decomposition`,
  `normet.analysis.normalise`, `normet.backends.lgb_backend`, `normet.plotting`)**:
  - Zero-resampling single-pass fast path (`variables_resample=[]`) in `normalise`
    and `Chronos2Estimator.deweather`; explicit residual separation
    (`include_diagnostics=True`: `predicted`, `met_explained`, `residual`);
    calendar-aware `stratify_by` + `exact_base=True` in `decom_emi`; forwarded
    `n_cores` in `decom_met`, Chronos-2 `stratify_by`/`conditional_k` rejection,
    multi-anchor sensitivity ranking, and diverging/double-count-free
    `decomposition_stack()`.
  - Second-order Shapley interaction analysis (`nm.shapley_interactions`,
    `ShapleyInteractionResult`, `nm.plot_shapley_interactions`), order-independent
    calendar Shapley (`nm.decom_emi_shapley`), atmospheric-physics group builder
    (`nm.suggest_met_groups`, `MET_PHYSICAL_GROUPS`), paired MC/permutation
    standard errors (`return_uncertainty=True`), and multiplicative LMDI log-scale
    closure (`scale="multiplicative"`).
  - Out-of-fold temporally blocked decomposition (`nm.cross_fit_decompose` with
    `embargo_hours`), kNN on-manifold conditional resampling (`conditional_k`),
    path-dependent TreeSHAP (`nm.decom_met_treeshap` + `LgbModel.predict_contrib`),
    and interventional-vs-conditional comparison (`nm.compare_attributions`).
- **Cross-regional transport physics & Lagrangian receptor attribution**
  (`normet.io.trajectory`, `normet.analysis.transport`, `normet.analysis.normalise`,
  `normet.plotting`):
  - BLH-coupled endpoint weights (`nm.trajectory_endpoint_weights`:
    boundary-layer gating, inverse mixing-depth sensitivity `1000 / max(BLH, min_blh_m)`,
    exponential chemical decay, and strictly causal downstream wet scavenging),
    continuous vector inflow (`inflow_sin/cos`, `mean_bearing_sin/cos`,
    `straightness`), `blh_contact_frac`, time-integrated `rain_mm`, physical
    per-region `traj_footprint_*` exposure, optional `horizons_h` bands, and
    multi-height / `n_workers` parallel HYSPLIT execution in
    `run_back_trajectories()`.
  - Quality-control column classification (`nm.is_trajectory_qc_column` /
    `TRAJ_QC_SUFFIXES`) so Transport Studio auto-ticks physical descriptors
    without ticking `traj_n_endpoints` / `traj_age_max_h` as predictors.
  - Stratified conditional resampling (`stratify_by=` in `NormaliseConfig` /
    `DecomposeConfig` / `normalise` / `decom_met`) to eliminate cross-season or
    cross-regime OOD extrapolation across local weather and transport pools.
  - Physical BLH-weighted gridded receptor models (`nm.wpscf`, `nm.wcwt`,
    `nm.gridded_receptor_model`), regularized non-negative source footprint
    inversion with moving-block bootstrap CIs and sliding-window intensity tracking
    (`nm.build_footprint_matrix`, `nm.invert_source_footprints`), Ward & angular
    trajectory clustering with TSV elbow selection (`nm.cluster_trajectories`),
    and SCM donor spillover / upwind coupling screening
    (`nm.screen_donor_transport`).
  - Transport visualisations (`nm.plot_receptor_grid`, `nm.plot_trajectory_clusters`,
    `nm.plot_footprint_inversion`) and Transport Studio `Analysis` workflow tabs.
- **Conda recipe.** `packaging/conda/meta.yaml` packages the 0.3.0 PyPI sdist
  as a `noarch: python` conda package (core dependencies; the CLI/MCP/GUI
  entry points), with local-build, personal-channel and conda-forge steps in
  `packaging/conda/README.md`. The root `packaging/README.md` points to it.
- **Data preparation, modelling and evaluation v2 (`normet.utils.prepare`,
  `normet.utils.featureeng`, `normet.utils.metrics`, `normet.utils.cv`,
  `normet.analysis.verification`, `normet.model`,
  `normet.backends.lgb_backend`)**:
  - `prepare_data` gains `deduplicate` (`"mean"`/`"first"`/`"last"`/`"raise"`),
    `strict_covariates` (a missing requested covariate raises instead of being
    dropped) and `local_time` (calendar features from the frame's civil time for
    tz-aware input; the default keeps the historical UTC encoding); its
    `time_interp` fills within each set for the time-ordered splits, so a
    value is not interpolated across the train/test boundary there, while a
    random split keeps the legacy whole-series interpolation.
    `impute_values` takes `within_split` to choose either behaviour, and
    `add_lag_features` / `add_rolling_features` accept `freq=` to align rows
    on a calendar grid (missing hours included) rather than on row counts.
  - `screen_aq_data` (flatline, Tukey-IQR and met-bound QC, now per station
    with `group_col=`), `add_met_physics_features` (ventilation index, BLH,
    Magnus dew point and related physics) and `analyze_lag` (prewhitened CCF
    with ACF/PACF diagnostics for choosing lags).
  - Model layer: `model_feature_names` freezes the training column order so
    prediction cannot silently reorder features, `permutation_importance`
    computes model-agnostic importance on a held-out split, `ml_predict` /
    `ml_predict_dask` take `strict_features`, LightGBM gains native
    categoricals, alternative objectives/metrics, `sample_weight` and a
    `deterministic` mode, and FLAML gains blocked/time CV. `save_model`
    accepts `metadata=` and writes a `*.meta.json` sidecar that
    `load_model_metadata` reads back; training-cache keys include the dates,
    covariate order and weights.
  - Evaluation: `modStats` gains extended skill scores and `effective_n`,
    `exceedance_stats`, `compare_models` (circular moving-block bootstrap
    interval and Newey–West Diebold–Mariano test for `score_b - score_a`, with
    an explicit `y_scale` for the truth), `cv_score` and `blocked_time_cv`
    (blocked/embargoed folds with out-of-fold predictions), and
    `fair_crps_from_samples`, `interval_score`, `interval_coverage`,
    `pit_diagnostics` and `calibration_curve` for probabilistic diagnostics.

### Performance
- **Flat-array endpoint weights across the transport family.**
  `wcwt`/`wpscf`, `build_footprint_matrix` and `screen_donor_transport`
  called `trajectory_endpoint_weights` once per trajectory, and about two
  thirds of that loop was pandas column extraction rather than arithmetic.
  They now convert each column once and run a flat numpy kernel over every
  row (`_endpoint_weights_flat` / `_endpoint_weights_flat_frame`),
  keeping per-group loops only where
  sequential order matters (decay `exp`, causal rainfall cumsum, and each
  region/hit `np.sum`). Results are bit-identical on the tested targets —
  the parity suite compares `uint64` views and replays every public
  function against its pre-flat scalar reference (numpy may dispatch SIMD
  transcendentals on x86 builds). Measured on a 147k-endpoint table
  (3,000 trajectories × 49 steps, macOS arm64, best-of-2): weight loop
  468 → 58 ms (8.0×), `wcwt` 534 → 127 ms (4.2×), `build_footprint_matrix`
  630 → 182 ms (3.5×), `screen_donor_transport` `donor_back` 605 → 170 ms
  (3.6×) and `treated_back` 1245 → 459 ms (2.7×).

### Changed
- **User manual rebuilt for 0.3.0** (`docs/manual/normet_manual.tex` and
  `normet_manual_zh.tex`, both PDFs): the books now document the v3 engines
  (DML, conserved Ox–NOx, regime-conditioned, ICN and its draw bank), the
  synthetic benchmarks, the meteorological physics features, the new SCM
  backends (ASCM, SDID, Chronos-2, de-weathered, transport), the Transport
  Studio's receptor maps, clustering and footprint inversion, the 17 CLI
  subcommands and 26 MCP tools; the screenshots and the example numbers are
  regenerated from the current package. A post-publication audit corrected the
  FLAML trial-bounded search text and defaults, the result-tab order, the model
  screenshot caption, the ICN design defaults, the SCM placebo RMSPE
  explanation and the Chinese 保形 term, in both books.
- **Package layout (`normet`)**: the loose top-level modules now live in
  subpackages — `normet.viz` (`plotting`, `style`, `figstyle`),
  `normet.reporting` (`report`), `normet.app` (`cli`, `mcp_server`),
  `normet.core` (`exceptions`, `i18n`), `normet.eval` (`benchmarks`) and
  `normet.data` (`datasets`, beside the CSV.GZ assets). The old module
  paths remain as aliases of the same module objects, so
  `from normet.plotting import ...`, the `normet.cli:main` /
  `normet.mcp_server:main` entry points and
  `normet.datasets.load_my1_pm25()` keep working unchanged.
- **De-weathering defaults restored (`normet.analysis.normalise`, `nm.pdp` /
  `nm.ale`, `nm.rolling`, `nm.decompose_uncertainty`, `nm.do_all`,
  `nm.do_all_multisite`, `nm.prepare_data`, `nm.train_lgb`)**:
  - `back_transform` defaults back to `"geometric"` (average the draws on the
    `log1p` scale, one `expm1` at the end) everywhere the option appears, so
    `log_transform=True` output from earlier releases stays bit-identical;
    `"arithmetic"` (per-draw `expm1` before averaging) and Duan's `"smearing"`
    remain opt-in.
  - `SingleConfig.embargo` is now `int | None = None`, meaning "not chosen":
    no purge for the ordinary split, and `crossfit_normalise`'s 24-row
    default when `cross_fit=True`; an explicit value (including 0) is used
    as given. `prepare_data` / `impute_values` default to the whole-frame
    `"median"` strategy, the pre-option behaviour; `"train_median"` stays
    available for a leak-free test block.
  - `do_all(chronos_mode="icn", log_transform=True)` rescales the ICN
    `anchor_spread` / `mc_spread` columns with the delta method (`1 + level`),
    so the uncertainty columns share `normalised`'s units instead of being
    log-scale SDs.
  - `train_lgb(..., native_nan=False)` (or `model_config={"native_nan": True}`)
    opts into LightGBM's native missing-value routing; the default keeps
    replacing every non-finite predictor by 0.0. `do_all_multisite` accepts a
    DataFrame `reference_pool` again (the `"pooled"` string test no longer
    evaluates a frame's truth value). `trajectory_features` /
    `build_trajectory_features` / `run_back_trajectories` emit the Layer-1
    physical descriptors (straightness, inflow/bearing components, BLH contact,
    integrated rain, physical footprints) only with the new `physics=True`, so
    the default `traj_*` column set is unchanged; Transport Studio passes it.

### Fixed
- **Packaging docs and the Windows installer fallback**: `packaging/README.md`
  names the macOS bundle `Normet.app`, as `build_dmg.sh` produces it, and
  `installer.iss` falls back to version 0.3.0 (not 1.0.0) when compiled by
  hand without `/DAppVersion`.
- **Surface defects found by the remaining-todo audit**:
  - SCM Studio's `deweathered` / `transport` backends now see their
    meteorological covariates and the spillover-exposure column in the design
    frame (`_design` selected only date/unit/outcome, so the staged fits
    failed on missing columns although the controls were filled in).
  - `normet invert-footprints` parses a named `date` column for CSV input;
    `read_csv(parse_dates=True)` only parses the index, so a wide table
    reached the estimator with a string index.
  - `normalise-ox` / MCP: `no_col=None` is sent as the explicit
    `--no-col none` sentinel instead of being dropped by `_options` (which
    made Click fall back to `"NO"`); the GUI's NO selector gains an explicit
    `"(none)"` entry instead of silently using the first numeric column.
  - `deweather --bank` and the GUI's draw-bank Load pass `surprise_tau`
    into `ICNDrawBank.aggregate`; the re-aggregation path had ignored it, and
    the GUI's `0` now keeps the bank's stored value.
  - `block_mode="markov"` requires `block_hours > 1` in `normalise`,
    `do_all`, the CLI, the MCP tool and the GUI; the previous fallback was
    silent independent resampling.
- **First-principles review of the v3 additions (`nm.dml_normalise`,
  `nm.normalise`, `nm.decom_met`, `nm.shapley_interactions`,
  `nm.add_met_physics_features`, `nm.normalise_ox_invariant`,
  `nm.deweathered_scm`, `nm.transport_adjusted_scm`,
  `nm.make_synthetic_benchmark`, `nm.ICNDrawBank`, `nm.icn_decompose`)**:
  - `add_met_physics_features` takes `group_col` and accumulates the causal
    memory features per group (a panel no longer feeds one site's air mass
    into the next), with decay applied per elapsed hour
    (`exp(-dt/tau)`) so hourly, daily and irregular records share the same
    physical kernel.
  - `dml_normalise` aggregates the fold draws on the requested
    `back_transform` scale (arithmetic and smearing had been no-ops that
    silently returned the geometric level), combines the fold with the slow
    trend in log space, and scales `mc_se`/quantiles from the same adjusted
    draws; `day_julian` joins `date_unix`/`year` outside the tree's split
    space.
  - Control-variate adjusted `normalise` quantiles (all aggregation paths)
    are built from the adjusted draws, so the bands share the central
    line's level.
  - `ICN` surprise weighting is computed in log space, so an anchor set
    whose Gaussian gates all underflow keeps its peaked weights instead of
    reverting to a uniform average; `ICNDrawBank.load` restores the saved
    timezone; `icn_decompose` rejects default player names that collide
    with result columns.
  - The `standard` branch of `normalise_ox_invariant` and
    `deweathered_scm` forwards `log_transform` (and `paired="log_ratio"`),
    matching the dml/cross-fit branches; `normalise_ox_invariant` coerces a
    string `date` column instead of grouping on raw labels and returning
    all-NaN `observed` columns.
  - `transport_adjusted_scm` centres outcomes and exposure over the same
    donor set and falls back to equal weights when a backend reports
    non-finite/all-zero weights (previously filled with 0, zeroing the
    spillover correction); it reports the weights actually used.
  - The fused coalition evaluator stores draw indices rather than a full
    panel tile per batch, keeping peak memory within the batch budget.
  - `make_synthetic_benchmark` honours `noise_sd` in `F3_titration` (the
    old branch hard-coded 0.5 ppb) and reports the pool expectation
    `E[O3 | emissions]` as ground truth, the quantity Monte Carlo
    normalisation actually estimates.
  - `load_lgb` applies the macOS OpenMP coexistence guard before
    unpickling a booster.
- **Cache, conditioning and aggregation defects (`nm.normalise`,
  `nm.decom_emi_shapley`, `nm.decom_met`, `nm.shapley_interactions`,
  `nm.cross_fit_decompose`, `nm.cluster_trajectories`,
  `nm.gridded_receptor_model`, `normet.io.trajectory`, `normet.causal`)**:
  - `normalise` cache keys include the `set` column when the Duan smearing
    factor is estimated from test rows, and the pool's `date` column when
    `month`/`season` strata are requested; the zero-resample fast path
    reproduces the resampling path's quantiles by replicating each row
    `n_samples` times before grouping; `conditional_on` filtering applies one
    positional mask to the pool and its strata source, so a non-unique index
    no longer raises `IndexError`.
  - `decom_emi_shapley` handles a model with no time feature itself (the
    delegation to `decom_emi` re-derived log space and double-transformed the
    observations under `scale="multiplicative"`), and its analytic base
    aggregates before `log1p`, matching the Monte Carlo estimand;
    `decom_met(return_uncertainty=True)` shifts each coalition's draws so
    their mean equals the aggregate-path point estimate (the per-seed
    back-transform added a Jensen gap of ~7 % on `emi_total`);
    `shapley_interactions` returns its standard summary columns for a
    calendar-only model; `cross_fit_decompose` imputes each fold from its
    training rows only.
  - `cluster_trajectories` reports the clusters `fcluster` actually returned
    instead of padding to the requested K with NaN centroids;
    `gridded_receptor_model` uses the published Polissar tiers
    (1.0/0.7/0.42/0.05, strict boundaries) and, like the WCWT/WPSCF entry
    points, prefers a de-weathered `normalised`/`emi_total` column over raw
    `observed`; `read_trajectory_endpoints` deduplicates on start location and
    `traj_id` as well, so trajectories from several start locations survive.
  - `uncertainty_bands` with the three-donor minimum falls back to
    moving-block time resampling (a donor bootstrap cannot vary a design that
    must contain all three) and applies the finite-population rescaling
    whenever it subsamples; `run_scm_staggered` measures event time and the
    requested horizon in panel periods rather than rows, and its placebo
    tasks refit the same donor pool as the real slot (the pool is now part of
    the deduplication key); `conformal_inference` treats an explicit
    `zeta_omega=None` as sdid's documented "auto" rule; `pre_trend_test`
    defaults back to OLS standard errors (`se_type="hac"` opts in);
    `cv_score` keeps missing validation targets as `NaN` and excludes them
    from metrics instead of scoring synthesized values; `compare_models`
    follows the explicit `y_scale` instead of guessing the truth's scale;
    `modStats(weekday_start=0)` keeps the historical 0–6 weekday labels by
    default; `ml_predict_dask` accepts the `strict_features` it documents.

## [0.3.0] — 2026-11-19

First PyPI upload since 0.2.4 (2025-10-29). This section covers everything
merged since then — the APIs the 0.3.0 wheel ships. The two June 2026
development snapshots, previously headed 0.3.0 and 0.4.0 in this file, were
never published; their entries are reproduced at the end of this section.

### Added
- **`nm.normalise_in_context` at the top level.** The ICN orchestrator was
  reachable only as `normet.foundation.normalise_in_context` although the
  README and every other foundation entry use `nm.*`; it is now exported from
  `normet` and pinned by `tests/test_imports.py`.
- **In-Context Normalisation (ICN) for Chronos-2** — `normet.foundation.in_context`
  with `ICNDesign` (D1–D9 axes), `MetPool` (record-pointwise / record-trajectory /
  climatology sampling with calendar alignment and the A2 support check), `ICNResult`
  (level, paired, weather effect, per-source spread, per-hour anchor contribution,
  diagnostics) and `Chronos2ICLPipeline`. `Chronos2Estimator.normalise_in_context`
  is the thin delegate; the heavy forward pass sits behind the `ICLPipeline` protocol,
  so the orchestration runs torch-free in CI via a `FakeICLPipeline`. ICN intervenes on
  the future only (D1) and draws the counterfactual weather from a pool (D2), so short
  emission events that `deweather` absorbs survive. `deweather` is untouched.
  Tests in `tests/test_in_context.py` assert the headline property: future-only
  intervention recovers the weather response, past+future intervention collapses it.
- **ICN: short records, honest aggregation names, separated uncertainty.** Four fixes
  found while setting up the ICN design scan (2026-11-11):
  - `ICNDesign.min_context` — an anchor used to need a full `context_length` of
    history, so a record shorter than that produced **no output at all** (under 85.3
    days at C=2048) and a three-month record yielded one partial anchor over its last
    136 hours (6 % of the record). With `min_context` anchors may start earlier and
    the first one uses whatever history exists. `None` keeps the old behaviour.
  - `ICNDesign.aggregate`: `"median"` was a (lead-weighted) **mean** and
    `"quantile_mean"` was the same code path as `"trimmed"`, so the three names were
    two estimators. `"mean"`, `"median"` (weighted median) and `"trimmed"` are now
    three, and `"quantile_mean"` is kept as a deprecated alias of `"trimmed"`. The
    default is unchanged in value (`"trimmed"`).
  - `ICNDesign.support_min_frac` was declared and never used; the A2 gate now drops
    anchors whose observed future lies outside the pool's range and records them in
    `diagnostics["low_support"]` (a no-op for record pools, which contain their own
    blocks; it matters for the external climatology pool of PR-2).
  - `ICNResult.spread_by_source["anchor"]` and `["mc"]` were both the std over every
    draw of an hour. `mc` is now the within-anchor resample spread (pooled over
    anchors) and `anchor` the across-anchor spread, i.e. two distinguishable
    sources.
  Tests in `tests/test_in_context.py` cover all four; `mean`/`median`/`trimmed` are
  asserted to differ on a skewed draw set.
- **Assistant: models forwarded by hpc-agent, and servers of your own.** A model that
  hpc-agent forwards from a cluster job is named `cluster:model`
  (`bede:hf.co/unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL`, or any unique part of the name,
  `bede:glm-5.3`). Its address is the forward's port, read from hpc-agent's registry
  (`~/.config/hpc-agent/forwards.json`, or `$NORMET_HPC_FORWARDS`) each time a message is
  sent, so a restarted job (new port) still answers; the model list offers those that
  accept a connection now and refreshes when opened. The cluster's llama.cpp key comes
  from **Key…** (keychain), `BEDE_API_KEY` / `CSF3_API_KEY`, or, failing both, the copy
  hpc-agent keeps in OpenCode's config for that port; it is read, never written or
  printed. Any other OpenAI-compatible server is added by typing `name:model` and
  clicking **Key…**, which asks for its address (kept in the settings; emptying the
  address removes it). In the library: `normet.agent.llm.hpc_forwards`, `hpc_models`,
  `hpc_forward`, `register_provider`, `forget_provider`, `provider_spec`,
  `known_providers`. The built-in providers, their names and addresses are unchanged.
- **Trend analysis (`normet.trend`)** — honest trends on autocorrelated
  environmental records. Three entry points:
  - `mann_kendall` : Mann-Kendall S with tie correction (Hirsch et al. 1982)
    and the seasonal variant of Sen (1968) for records that carry a repeating
    calendar structure (e.g. `seasonal_period=24` for hourly).
  - `theil_sen` : robust median pair-wise slope with the analytic Sen (1968)
    95 % interval, reported per day for timestamped series and with a
    per-year conversion.
  - `trend` : the combined entry point. Runs both, diagnoses AR(1) serial
    correlation of the OLS-residual ACF, and -- when it finds the
    autocorrelation that hourly normalised records almost always have --
    replaces the slope's error estimate with a moving-block bootstrap
    (deterministic under `seed`) plus a Wald p-value. The classical
    Mann-Kendall p-value is always reported *unadjusted* and labelled as
    such: the result carries an explicit note that it is the anti-conservative
    one under the detected autocorrelation, so a report can say which number
    was quoted and why.
  `res.summary()` returns a tidy two-column frame for reports. Nothing here
  resamples the weather; it is statistics on a series you already have
  (typically the emission component from `decompose` or a deweathered
  `normalised` column). User guide: `guide/examples_trend.md`.

- **Segmented trend: `detect_trend_breaks`** (`normet.analysis.trend`,
  exported at the top level). A single-slope record whose level or slope
  changes part-way through a policy year or a lockdown shows the change
  diluted into a muddier trend; the honest question is *where* the regime
  changes and how the slopes differ. `detect_trend_breaks` splits the record
  into at most `max_segments` (default 2) linear segments, each reported with
  its own Theil-Sen slope and Sen (1968) interval. The split comes from a
  full O(n) grid search (the two-segment SSE of every candidate position is
  O(1) from prefix sums) and is reported only under a two-part gate, because
  a grid searched over n-1 positions *always* finds something: the split must
  improve the BIC, and the SSE improvement of the final fit over the
  single-segment fit must beat the same search re-run on moved-block
  resamples of the series (the multiple-testing-corrected, autocorrelation-
  aware p-value). A record that fails either gate is reported as one regime,
  with the best candidate cut and its p kept in `best_break`,
  `best_split_pct` and `notes` so a report can say what was looked at.
  Deterministic under `seed`; `summary()` returns one row per segment
  (start, end, n, slope, CI, per-year when timestamps are present).

- **Uncertainty decomposition: `decompose_uncertainty`**
  (`normet.analysis.uncertainty`, exported at the top level, with the
  `UncDecomposition` result). A de-weathered estimate carries a Monte-Carlo
  spread and an ensemble of models (`do_all_unc`) a wider, model-form spread,
  but neither says *why* the band is wide. `decompose_uncertainty` attributes
  the total per-date variance to its source by an orthogonal (two-way)
  decomposition of the full model-by-weather prediction matrix, exploiting the
  package's common-random-number discipline rather than adding a new source of
  randomness: every ensemble member is trained on its own seed (so the members
  differ in *form*) but predicts on the *same* `n_samples` met resamples shared
  across the ensemble. At one timestamp the response is a
  `n_models x n_samples` matrix whose total variance splits orthogonally into
  the *model-form* term (variance across members, over the draws), the
  *weather* term (variance across met draws, over the members) and the
  interaction (what is not additive). The total per-date variance is exactly the
  three summands (pinned by a test), every term non-negative. The cost is one
  prediction per `(member, draw, timestamp)` -- the same total count as one
  `do_all_unc` run -- and the whole thing is reproducible under `seed`. The
  result's `frame()` is the per-date rows (`var_weather`, `var_model`,
  `var_interaction`, `var_total` and their `pct_*` shares), `summary()` the
  median share per source, and `plot()` the stacked-variance-band figure --
  the policy-evaluation deliverable that answers "of the band, which part is
  weather and which part is model form?" -- the question a policy effect
  smaller than its band reduces to: which error dominates decides whether it is
  separable.

- **GUI: the trend is a step of the window.** The Decompose page gains a
  *Trend test* box (the series -- the decomposition's emission part, its
  separated long-term trend, its weather part or the observed target; the
  seasonal period, `auto` or named; the number of segments; the bootstrap's
  replicates) and a *Run* button; a finished emission decomposition
  starts the test of its emission part under the box's settings (`Run after
  each emission decomposition`, on by default).

  **Segment 1 is the single-slope test** (`mann_kendall` + `theil_sen`, as
  `normet.trend`). **Segments 2 to 4 run the segmented fit**
  (`detect_trend_breaks`) on the chosen series: where the trend breaks, each
  segment with its own Theil-Sen slope and Sen interval, and the block-bootstrap
  p of the search. The separated *long-term trend* is a Series choice (and the
  default once Segments is raised above 1): the emission decomposition's
  `date_unix` contribution plus its base, the recurring calendar cycles already
  taken into their own component columns, so the segmentation runs on the drift
  plus one-off structural breaks rather than on the full, cycled `emi_total`.

  Its result is a Trend tab: the series with its single robust line (one
  segment) or its piecewise Theil-Sen lines with a marked break (two or more),
  a traffic-light verdict that says which p stands (the classical MK or
  Wald p for one segment, the block-bootstrap p of the search for the
  segmented fit), the test's notes and its tidy summary -- one metric row per
  segment (the tab's light, its CSV export and its command-log entry are the
  step's own; the Methods paragraph says of the trend what it ran). The
  window's Assistant gained the `trend` tool (series, seasonal; the single
  slope) and the `trend_breaks` tool (series, segments; the segmented fit);
  each runs the step as its button does and returns the reading
  (`readings.trend_reading`, `readings.trend_breaks_reading`). The headless MCP
  server gained `normet_trend` (a table's column, run in place) and
  `normet_trend_breaks` (a table's column; `'long-term'` is the separated
  component, the default).

- **GUI: the uncertainty decomposition is a step of the window.** The Decompose
  page gains an *Uncertainty decomposition* box under the Trend box (the number
  of ensemble members, the met draws they all share, and the members' backend)
  and a *Run* button; it needs an AutoML model -- it trains its own members
  from the table as loaded, the button greys for chronos-2, whose zero-shot
  forecasts carry no model form to vary -- and reuses the trained feature set
  and the variables ticked on the Normalise page. Its result is an
  Uncertainty tab: the per-date variance stacked by source (met resampling,
  model form, the interaction), a verdict that says which source carries the
  larger median share of the total spread, and the median share and median
  variance per source as the table, with the tab's light, CSV export,
  command-log entry (`decompose_uncertainty`) and Methods paragraph of its
  own; a saved session brings the run back whole, its figure re-drawn.

- **Skill scores: CRPS and the pinball loss** (`normet.analysis.verification`,
  exported at the top level). Point forecasts are verified by `modStats`, but a
  comparison with a baseline needs the spread's score. `pinball_loss` is the
  quantile loss at one level (`max(q r, (q-1) r)`, half the MAE at `q=0.5`);
  `crps_from_quantiles` scores a quantile forecast as the exact integral of the
  CDF's squared distance from the observation's -- the continuous form the
  pinball sum is a numerical rule for (Gneiting and Raftery 2005) -- and reads
  `normalise`'s `qNNN` columns (or the sample table) directly; `crps_from_samples`
  scores an ensemble, each sample at its rank; `crps_observed` is the point
  forecast's reference -- its MAE read as a normal's own MAE and returned as
  the sigma that would carry it, so the MAE times `sqrt(pi/2)` (a point mass's
  own CRPS is its MAE exactly: the factor is the normal-scale convention, not
  the degenerate integral) -- so model and baseline read on one scale.
  Non-finite rows are dropped and rows meaned, as the package scores them.
- **Notebook 06: the skill benchmark, on the bundled MY1 NO2 record.**
  The lightgbm model trained through `do_all` on a `month_ts` split keeps
  its quantiles (`return_quantiles`), so its spread is scored as the CRPS of
  those five levels; persistence and three climatologies are scored as point
  forecasts' references, and a Chronos-2 zero-shot row -- where the
  `foundation` extra is installed -- joins them. The zero-shot model has no
  forecast for its leading context (those rows carry the observed values),
  and a third of the held-out hours fall there, so the final table is
  re-scored on the hours the zero-shot model actually predicts and every
  method is read on that one set: how many hours that excludes, and where the
  window starts, is printed above the table rather than footnoted. The
  expensive calls are memoised with `nm.make_memory`.
- **GUI: window and figures refined.**
  - *Cycles*: `normet.cycle_profiles` (and `plotting.profile_axes`) draws a
    record's mean by hour of day, day of week and month with 95 % confidence
    intervals, as openair's `timeVariation`. The Normalise tab shows observed
    and normalised cycles under the series; the emission decomposition draws its
    diurnal and weekly components as cycles beside the stack.
  - *Readable time series*: a sub-daily record of more than 2,000 points opens
    as daily means; every series is drawn broken at a gap (`style.line`,
    `style.break_at_gaps`), time zones kept.
  - *Polar plot* (Analysis → Polar Plot…): a variable by wind direction and
    speed, wind columns found by name or made from u10/v10.
  - *For print* reads the figure against the journal's guide (width, height,
    smallest and largest text, overlapping tick labels, face), zooms to fit or
    to actual size, and keeps its choices for the next figure.
  - *PDP* panels mark each variable's deciles; whole-valued variables are steps.
    The parity plot becomes a hexbin density past 2,000 points, with n, slope
    and bias in its corner.
  - Hidden tabs draw when shown (`FullWidthCanvas`), covering the
    widget exactly.
  - *Notices* above the results replace "do this first" dialogs, with the step
    on a button; the status bar has a light per step.
  - *Save Results / Load Results / Reopen Last Session*: a whole session (table,
    settings, model, results, command log) in one `.normet` file, drawn again by
    each step's own handler; the last session is kept after every step.
  - *Load Session Script*: a script the command log wrote, run (after asking)
    and brought back into the window, the panel set from its calls.
  - *Methods Paragraph*: what the session ran, written as a paper's Methods in
    the window's language, from each step's call as it ran, with references.
  - *Report Language* (File menu) and `generate_html(..., lang="zh")` /
    `normet report --lang zh`: the HTML report and its figure in Chinese.
  - *Help → User Manual* (F1) opens the bundled manual in the window's
    language; *Keyboard Shortcuts* lists them from the menus.
  - The Assistant's conversation (`chatview`): colours mixed from
    the palette (readable in dark mode), Markdown with shaded code and ruled
    tables, tool calls as one line whose result opens with a click, Ctrl/Cmd ±
    to size the text.
- **GUI in Chinese** (View → Language: Follow System / English / 中文): every
  label, button, menu, tab, tooltip and dialog is translated
  as it is set, figures' texts as they are drawn, while the code still reads
  the English back. Data's own words (column names, backends, file names) and
  the Assistant conversation stay as written.
- **GUI: a Resolution selector** on the Data preview, Normalise, Decompose and
  Rolling tabs (as recorded / hourly / daily / weekly / monthly / yearly):
  the figure is redrawn from period means; the result and its export do not
  change. Steps finer than the record's own are disabled.
- **GUI opens ready to work**: the parameter panel folded (◀ Collapse /
  ▶ Expand Panel, Ctrl+B) and the Assistant open beside the results.
- **Figures in a journal style** (`normet.style`): 7–8.5 pt text, 0.6 pt axes with ticks
  outward, frameless legends, slim colour bars, Okabe–Ito colours with one
  meaning each (observed blue, normalised/synthetic vermilion, context grey),
  formulas set as chemistry writes them (`formula`: NO₂, PM₂.₅, SO₄²⁻), and
  concise date axes (the year once, then the months; panels stacked on one
  time axis dated on the bottom one only). The window applies it at start;
  every plotting function in `normet.plotting`, `normet.causal`'s band plots,
  the HTML report and the MCP figures draw in it whatever the session's
  rcParams. A canvas re-solves its margins at every draw (constrained layout),
  so titles no longer run into the panel above. The parity plot is square with
  a 1:1 line, PDP panels share one y label, and figure texts carry their
  Chinese translations (a formula in a Chinese title is written NO₂).
- **The user manual in Chinese** (`docs/manual/normet_manual_zh.pdf`), beside
  the English one: `normet_manual_zh.tex` is the English book
  translated label for label (tests/test_manual.py holds the two to the same
  chapters, sections, labels and screenshots), built by XeLaTeX with ctex, its
  screenshots of the window in Chinese in `figures_zh/`
  (`NORMET_UI_LANGUAGE=zh make_figures.py`). `build.sh` builds both (`en` or
  `zh` for one). The English book follows the window: the folded panel and the
  open Assistant, Ctrl+B, Resolution, the toolbar's For print…, View →
  Language, and the `data` extra Find stations needs.

- **`join_trajectory_features`** (`normet.io`, `normet`): joins a trajectory
  feature table onto a panel the way the module docstring recommends, so
  scripts and the GUI no longer each write their own `merge_asof`. A row takes
  the latest trajectory to have arrived, less than one release interval before
  it (inferred: the commonest step between one site's successive
  trajectories), so a window whose run is missing stays empty. With a `site`
  column in both frames each row takes only its own site's trajectories and
  other sites' rows are left alone, so sites can be joined one batch at a time;
  a column sent again is replaced. Rows without a time are kept, and times are
  compared as UTC, tz-aware or not.
- **GUI: the rest of Chronos-2.**
  - **Step 3 meteorological decomposition** on the chronos-2 backend, through
    the library's zero-shot path. The emission method stays unavailable there,
    with the reason in its tooltip. Step 3 also gains **Attribution**
    (auto/sequential/Shapley) and **Groups** (`local: t2m, blh; wind: u10, v10`)
    on either backend. The sample count follows the backend, as Step 2's
    already did.
  - **Fine-tuning**: LoRA when peft is installed, else full, with a step count.
    It runs on a worker; upstream's ~500 MB checkpoint copy goes to a
    temporary folder that is removed, not the current one.
  - **SCM Studio ranks donors by similarity**: the cosine of Chronos-2
    embeddings of each unit's pre-cutoff record only, so the effect cannot
    steer the choice. A Similarity tab shows the ranking, and the top N can be
    ticked as the donor pool.
  - **The Assistant reads Chronos-2's covariate-sensitivity check** as its
    model fit, and explains rather than runs an emission decomposition.
- **GUI: Antigravity agy and GitHub Copilot in the Assistant; finishing a
  figure for print.**
  - The Assistant now offers four command-line agents signed in to the
    user's own plan -- Claude Code, Codex, agy and Copilot -- each at the
    models it lists and with an effort to choose.
  - agy's entry for the window is added to its configuration for the
    conversation, enabled only while it answers, and removed when the
    conversation closes. Hooks in its empty workspace deny it any tool but the
    window's. A reply its stream broke off is asked once more on its own.
  - Copilot gets the window's tools alone, with its built-in servers off.
  - Export for Publication now shows the figure itself at its printed size,
    and it can be finished there: the legend placed in a corner or dragged,
    sized, put in columns, framed or not; data lines' width and marker size
    set; panels lettered; a title, label or tick's numbers edited by
    double-clicking. Every change is undone when the dialog closes.

- **Quantile calibration for zero-shot forecasters: `calibrate_quantiles`**
  (`normet.foundation`, exported at the top level, with the
  `CalibratedEstimator` wrapper and the `QuantileCalibration` record, and
  `CalibrationWarning` in `normet.exceptions`). A zero-shot Chronos-2 band is
  only as honest as the weather in its training data -- a record from a
  climate or regime the training set never saw earns a nominal coverage it
  did not buy, and the band is the number a policy question is compared
  against, so it should be checked, not assumed. `calibrate_quantiles`
  works for any estimator that scores with `predict_quantiles` (the
  duck-typed contract plus `prediction_length` / `context_length`, the
  chronos-2 estimator first among them): it lays whole forecast blocks of
  `prediction_length` steps back from `anchor` (up to seven, each with its
  context of rows behind it), so every forecast the model makes is one it
  could actually have made, and scores the held-out errors in the band's own
  5th–95th percentile half-width, so the correction is scale-free across
  heteroscedastic horizons. Each band level's threshold is the
  split-conformal order statistic of the held-out error multiples -- the
  `ceil((n+1) x nominal)`-th smallest, the extra step being the honesty
  margin for a fit made on hold-out data -- passed through a hand-rolled PAVA
  leg so corrected edges cannot cross, and the fit degrades to one constant
  threshold, named `method="constant"` in the result, where the record gives
  fewer than 30 error multiples per level and a per-level quantile is not a
  split. The wrapper's `predict_quantiles` rescales each requested band edge
  about the median to median x threshold x span and returns only the levels
  asked for; the median and every other method delegate to the base model
  unchanged, so nothing existing moves. `coverage` is the cross-validated
  scorecard -- per level the nominal, the uncalibrated and the calibrated
  empirical coverage, each scoring pass fitting its thresholds without the
  blocks it scores -- and `CalibrationWarning` names the level still off
  target by more than `tolerance` (10 points by default, either direction);
  the fit is returned, corrected and labelled, not suppressed, because a
  regime jump the hold-out cannot see is information, not failure.
  Deterministic; numpy/pandas only.

- **`counterfactual(..., calibrated=True)`** (the chronos-2 estimator). The
  counterfactual's p10/p90 are the model's own quantiles, whose nominal
  coverage is only as honest as the weather in the model's training data --
  and the band is what the effect is compared against, so it should be the
  checked band, not the claimed one. `calibrated=True` runs the record's own
  pre-intervention history through `calibrate_quantiles` -- the fit and its
  hold-out both end at the intervention date, the same conditioning the
  projection is made under -- and takes the band through the fitted
  thresholds block by block, the placebo run through the same fit. The
  default stays off and the default path bit-identical: the median is a
  centre quantity the correction never touches, so the roll-forward
  trajectory and `pre_intervention_bias_pct` are unchanged in either mode;
  what changes is the band (`counterfactual_p10` / `counterfactual_p90` and
  `fraction_outside_interval`), and the fit rides along on
  `result.calibration` with its cross-validated coverage table and the
  levels a `CalibrationWarning` names as still off target.
- **Pre-trend diagnostics for synthetic controls: `pre_trend_test`**
  (`normet.causal`, exported at the top level, with the `PreTrend` result).
  The first question a reviewer asks of an SCM is whether the pre-period was
  parallel, and nothing in the diagnostics answered it: `scm_diagnostics`
  reports how close the fit is on average before the cutoff, but not whether
  it is *getting* closer or further as the intervention approaches -- which
  is what a violation looks like when it is about to happen.
  `pre_trend_test(result, cutoff_date)` reads the pre-period residuals
  (`observed - synthetic`) two complementary ways: the OLS slope against the
  period index with its classical t-test (a slow linear drift shows up here
  even when nothing else is alarming), and the RMSPE of the final third of
  the pre-period over that of the first two thirds -- 1 is a stable fit,
  above 1 the residuals grow toward the cutoff (a late fit breakdown shows
  up here while the slope test, diluted over the whole span, stays quiet).
  The `verdict` is a traffic light that reads them together, with the
  numbers in the text: `fail` when the slope is significant at the 1 %
  level (a pre-period that is trending is not parallel), `caution` when the
  slope is borderline or the drift ratio exceeds 1.25, `ok` otherwise, and
  `insufficient` when the record has too few pre-period points for a test.
  `scm_diagnostics` now carries the fit's verdict under a new `pre_trend`
  key (everything it returned before is unchanged), so the assistant's
  SCM reading reports the pre-trend line as well.
- **Event-study plot: `plot_event_study`** (`normet.causal`, exported at the
  top level). The visual half of the pre-trend check: the treated effect drawn
  against *periods since the cutoff* — grey before the intervention, red after
  — so the pre-period points that a credible fit should scatter about zero sit
  in their own colour, and any slope in them is visible before the step that
  is read as the effect. Event time counts panel periods, so a monthly panel
  and a daily one read the same and the points line up row by row with
  `placebo_in_time(...)["ref_band_event_time"]`; pass that frame as `band=` to
  shade the post-period with the placebo band (`ci_lo`/`ci_hi`, or
  `low`/`high`, or `p10`/`p90`). No band is drawn over the pre-period, because
  placebo cutoffs have no pre-period of their own to estimate one from.
- **Spatial diagnostics for the donor pool: `donor_spatial_summary` /
  `spatial_summary`** (`normet.causal.spatial`, exported at the top level,
  with the pure-numpy `haversine` / `bearing` primitives the (since-removed)
  `normet.physics` graph builder used). The weights say how much each donor
  contributes to the synthetic control, not where the donors sit relative to
  the treated unit — and for a policy that acts through local or regional
  processes, a pool clustered on one side of the treated site is a different
  object from one that surrounds it, at the same weights.
  `donor_spatial_summary(panel, donors, treated_unit, *, unit_col, lat_col,
  lon_col, n_sectors=8, weights=None)` places each donor: great-circle
  distance to the treated unit, compass bearing (0 = north, forward azimuth)
  and compass sector, plus the fit's weight when given. A donor with no
  coordinate in the panel gets a row of NaNs and sector `"?"` rather than
  being dropped — a missing fix is itself information.
  `spatial_summary(table, *, n_donors=None, n_sectors=8)` reads the pool:
  the centroid offset and bearing (the donors' (east, north) offset vectors
  averaged — the honest way to average directions, since a mean of angles
  wraps), the per-sector donor counts, and the share of the compass the pool
  actually spans. `scm_diagnostics` now appends the section under a `spatial`
  key when called with `panel=`, `treated_unit=` and the coordinate columns
  present (donors default to the fit's weights' index); without them the
  result is exactly what it was before. It is a diagnostic, not a fitter:
  nothing here changes the weights, and no distance or bearing is fed back
  into the fit. The SCM Studio window gains the matching readout: Latitude /
  Longitude column pickers, a "▶ Show donor layout" button, and a Spatial tab
  with the donors scattered around the treated unit, compass spokes, the
  pool's centroid and sector coverage.
- **Staggered adoption: `run_scm_staggered`** (`normet.causal.batch`,
  exported at the top level). For an intervention that rolls out unit by unit
  (the UK ULEZ 2021 expansion, staggered driving restrictions across cities),
  several treated units each get their own cutoff and their own `run_scm` fit,
  with the *other treated units excluded from every run's donor pool* so one
  unit's intervention cannot leak into another unit's synthetic control. The
  runs (real and placebo) share the one joblib pool `scm_all` uses.
  The result is one row per treated unit — `cutoff_date`, `n_pre`, `n_post`,
  `pre_r2`, `effect` (post-cutoff mean or `sum` of the effect path) and
  `p_value` (that unit's placebo-in-space test at its own cutoff) — followed
  by `mean` / `median` summary rows of the units' effects. The `mean` row's
  `p_value` is the cross-unit permutation test: each donor in turn takes every
  treated unit's role (the full set of cutoffs) and its pseudo-effects are
  averaged the same way the real units' are. The real treated units are
  deliberately not permuted into the reference — their data already carries
  the intervention, and permuting treated identities in a staggered design is
  contaminated by it; the donor-only reference keeps the null honest.
  `cutoff_dates` takes a list aligned with `treated_units` or a dict keyed by
  unit. A unit whose cutoff falls after its last observation is an error, as
  is a treated unit inside `donors`.
- **User manual: a Staggered adoption section** in the synthetic-control
  chapter of both books (`normet_manual.tex` / `normet_manual_zh.tex`, PDFs
  rebuilt). It sets out the staggered reading — each treated unit fitted on its
  own pre- and post-period with the other treated units kept out of every
  donor pool — and, in full, why the cross-unit permutation test's reference is
  donor-only: a treated unit's data already carry the intervention, so a
  staggered design's permuted treated identities are contaminated by the true
  effect (and with two treated units the single non-identity permutation is
  contaminated), while a donor never intervened on draws from the null.
 - **Data Studio: the found stations on a map, picked and fetched together.**
  After *Find stations* the Stations tab draws the sites at their coordinates
  above the table, on a basemap: OpenStreetMap's standard style, Esri's muted
  Light Gray Canvas or Esri satellite imagery --- all key-free, chosen on the
  map's own row and credited under it as their terms ask. A click selects the
  station under the pointer, a drag selects every station inside the drawn box,
  and Command/Control-click adds or removes one; the table mirrors the
  selection and a table selection rings the points, while the search field
  narrows both views. The wheel and the +/− buttons zoom about the pointer,
  the hand toggle (or a middle-button drag) pans, and *Fit map* brings the
  stations back; *Deselect* or Esc empties the selection. A count above the
  map says how many stations the search found and how many are picked (or
  how many a box holds, while it is drawn), double-clicking a station zooms
  to its streets, and a right-click menu selects what the view holds, zooms
  to a station or copies its coordinates. Latitude ticks carry as many
  decimals as the zoom needs. Fetch & merge now takes *every*
  selected station, downloads each with its own meteorology and stacks them
  into one panel-shaped table (``date``, ``site``, the pollutants, the met
  columns), skipping a station that returns nothing and failing only if all
  do; the preview draws one line per station. Points and tiles are in Web
  Mercator, the tiles fetched after the points for the view shown and cached
  under ``~/.normet/osm_tiles``; offline (or without ``requests``) the map is
  the same points on a plain grid. Every network works: all three providers
  carry station coordinates.

- **Data Studio is no longer UK-only**: the Network picker now also offers
   the **EEA** reporting network (all of Europe) and **OpenAQ** (most of the
   world, free API key), and the entry point is renamed accordingly — the
   toolbar button is "🌐 Get data" and the menu is File → Get &Data
   (AQ networks + met)… The window gains a *Source settings* group, shown
   only for the non-UK networks: an ISO-2 country code (both) and an OpenAQ
   key field falling back to `OPENAQ_API_KEY`. EEA stations come from the
   agency's own keyless ArcGIS layer (new `normet.io.eea.list_eea_stations`),
   whose per-station popup names one public Parquet blob per pollutant with
   its year range — the picker prefills from/to from those years and the
   fetch downloads exactly that blob (`fetch_eea_station`), two GETs per
   station. The EEA layer serves the E1a validated dataset, whose UK record
   ends in 2019; the up-to-date country-level bulk path (`fetch_eea_data`)
   is unchanged. OpenAQ finds send the ticked pollutants to the server as one
   union filter (paginated past the 1000-per-call cap). Only the six mass
   pollutants OpenAQ registers (PM2.5/PM10/NO2/O3/SO2/CO) are offered; NO and
   NOx are ticked away for OpenAQ as the adapter's set does not include them.
   The met join is shared: every provider carries coordinates.
- **SCM Studio: the fit's own inference** — five library analyses now run from
  the window. The Fit tab runs the fit's pre-trend test (`pre_trend_test`)
  alongside `scm_diagnostics` and shows its slope, p-value and RMPE drift. The
  new *Fit Checks* tab hosts the checks that act on a finished fit:
  ▶ Pre-trend + event study (the pre-trend verdict with the effect path on the
  event-time axis, `plot_event_study`), ▶ Conformal interval (a finite-sample
  95 % interval for the post-cutoff ATT, `conformal_effect_interval`, with a
  sub-sample control) and ▶ Leave-one-donor-out (`loo_weight_stability`, how
  far the weights and the effect move when each donor is dropped). 'Placebo in
  space' also reports Abadie's RMSPE-ratio test (`rmspe_ratio_test`): the
  treated unit's post/pre RMSPE ratio, its rank and its p-value.
- **Model tab: the holdout on the probabilistic scale** — every trained model's tab now
  carries the point forecast's CRPS reference, `sqrt(pi/2) x MAE` on the held-out rows
  (`crps_observed`): the score a spread forecast must beat to be worth its spread. The
  Chronos-2 tab adds a quantile backtest — one anchored forecast of the last horizon at
  the 10/50/90 levels, scored with `crps_from_quantiles` against the observed tail, the
  median's point reference and the pinball loss at each level (`pinball_loss`), with a
  line saying whether the band beats its own median's reference.
- **Model tab: walk-forward CV** — a `Walk-forward CV` button (with a `CV folds` spin,
  5 by default) retrains the current backend on expanding windows and scores each
  future block with `cv_score`, the honest check on a random split's R²: every test
  hour's neighbours sit in the training rows. The tab gains a `View` selector between
  the split's figures and the folds — an R² bar per fold with the split's value drawn
  across them, a table of each fold's test window and its R²/RMSE/MB, and a verdict
  that says plainly when the split flatters the model. Unavailable for Chronos-2,
  whose check is the quantile backtest above.
- **Events & anomalies box, Events tab** — the library's event detection
  (`anomaly_scores`, `detect_events`) reaches the window: a box under the Trend
  test scores a series — the emission part, the separated long-term trend, the
  weather part or the observed target — with `iqr`, an IsolationForest on lags
  (`isolation`), or the STL residual (`stl_residual`), flags the consecutive
  stretches above `Threshold k` (in IQRs; the forest's own boundary when it
  runs) of at least `Min length` steps, and the new Events tab shows the
  score strip with each flagged window shaded beside a table of the windows:
  start, end, steps, max and mean score, ranked by the maximum.
- **Assistant: cached API prefixes and a reasoning effort for API models** — an
  Anthropic round now marks its tools, system prompt and history end with
  `cache_control`, so a later round of a tool loop reads the prefix at 0.1× the price
  instead of buying it again (measured 43,446 → 15,609 billed-equivalent input tokens
  over a 6-round loop), and the token count adds the cache's write and read back to
  input. The Effort selector, until now the command-line agents', is offered for API
  models too: `auto` (nothing sent) first, then the provider's values — DeepSeek's
  `low`/`high`/`xhigh`/`max`, the others' `low`/`medium`/`high` — sent as
  `reasoning_effort` or `output_config.effort`.
- **Assistant: Up/Down recalls what was sent from the box** — at the first line, Up
  walks back through the last 50 messages sent from it (a repeat after itself kept
  once), and past the newest Down puts back the draft as it was left. Inside a longer
  text the arrows move the caret first, as Qt's own. The history travels in the
  settings; New chat forgets the conversation, not it.
- **Assistant: "Add a server…" names a server of your own in one dialog** — the model
  list's last item asks for the name, the OpenAI-compatible address, the model it
  serves and a key if it wants one, checks them while the dialog is still open, and
  selects `name:model` once they are kept. It writes the same settings **Key…** does,
  so a server added here is edited, or removed, there; Cancel goes back to the model
  chosen before.

### Changed
- **FLAML's default search is trial-bounded, not wall-clock.** `train_flaml`
  (hence `build_model`/`train_model(backend="flaml")`) now defaults to
  `max_iter=30` with `time_budget=-1`, so model selection is reproducible
  for a given seed and data; the old 90 s wall-clock budget returned a
  different model on a loaded node, with nothing in the result saying the
  search was cut short. 30 trials of the default LightGBM search took 14 s
  on a 1,440-row synthetic frame and 64 s on 50,000 rows — at or under the
  budget it replaces on those sizes. Passing `time_budget` explicitly still
  asks for wall-clock search (the default trial cap is dropped); passing
  both applies both, whichever is reached first. The GUI's budget box now
  reads **Search trials** for both trained backends (it was seconds for
  flaml), and the SCM window's per-fit box likewise; saved sessions and
  scripts that carry `time_budget` are read as trials.
- **`multisite_apply` takes `inner_n_cores`.** The model-based multi-site
  drivers already pin each site's own pool single-threaded, but a caller's
  own `func` that resolves `n_cores=None` to the job allocation still opens
  one full pool per site (P × (C − 1) threads on C cores). The new keyword
  is forwarded to `func` as `n_cores`; it is opt-in rather than injected,
  because a generic `func` need not accept that keyword.
- **Theil–Sen on a long record no longer builds n × n matrices.**
  `scipy.stats.theilslopes` materialises two `n × n` difference arrays, their
  boolean mask and the `n(n-1)/2` pairwise slopes: ~11 GB of temporaries at
  n = 20,000 (28 s, measured) and ~19 GB at a three-year hourly record — an
  out-of-memory on a laptop — for a median it could select from the slopes
  alone. `theil_sen` above 4,000 points now builds the same slope multiset row
  by row and selects from it in place: **0.9 s at n = 20,000** and 6.8 s at
  n = 26,280, with the slopes vector (1.6 GB at 20,000) the only large array.
  The slope, the Sen (1968) interval and their rounding are **bit-identical**
  to scipy's, including tied x and an even-sized pairwise set (pinned by
  tests); below the threshold scipy's own path is kept unchanged. The
  residual block-bootstrap's full-record Theil–Sen fit -- run for its
  ``bootstrap_scheme="residual"`` base line -- takes the same path, and so do
  the per-segment reporting fits of ``detect_trend_breaks``, whose segments
  on a multi-year hourly record are far above the threshold. Neither scheme
  now allocates the matrix the per-replicate guard never covered.
- **Documentation reorganised.** The README is now a short overview (capabilities
  table, installation, quick start, links); the material it carried moved to the
  documentation site, which gains Getting started (installation with every extra,
  quick start), a step-by-step workflow page, pages for the desktop GUI, the CLI and
  the MCP server, a notebooks index, and an About section that includes the
  changelog, contributing guide, code of conduct, security policy, citing page and
  roadmap. The README's CLI examples (`normet do-all --config config.yaml`) omitted
  the required input file and `--out`; the CLI page gives the working form. Snippets
  for `wind_to_uv`, `cv_score`, `do_all_multisite`, `make_run` and the `resample_df`
  reference period now match the functions' signatures. New `CITATION.cff` and
  `SECURITY.md`; `CONTRIBUTING.md` documents the CI checks, the 70 % coverage gate,
  the macOS libomp hang, the docs build and the commit conventions; `pyproject.toml`
  lists the Documentation, Repository and Changelog URLs.
- **Opening a large table is ~4× faster.** `read_table`'s blank-row check
  converted every column to text and stripped each cell, so a 1M-row,
  six-column CSV spent 3.1 s of its 9.0 s read proving that five numeric
  columns held no blanks. The mask now asks `isna()` of numeric and datetime
  columns and strips only text columns (3.1 s → 0.36 s on the same frame,
  identical rows), and `parse_dates` skips its day-first probes for a column
  that is entirely year-first (a full-column match; ~0.7 s per million rows).
  Opening that file in the window: **10.9 s → 2.6 s** (the read 9.0 → 2.0 s).
- **`import normet` no longer imports matplotlib's plotting layer.** `normet.causal.bands`
  imported `matplotlib.pyplot` at module level, though only its three plotting helpers
  use it, so every import paid ~0.35 s for the figure stack. The import moved into
  those helpers: a cold `import normet` is **1.53 s → 1.12 s** (median of 7 starts;
  GUI import 1.80 s → 1.65 s), and `sys.modules` after the import no longer holds
  matplotlib. Results are unchanged; matplotlib arrives when a figure is drawn.
- **`import normet` no longer imports joblib.** The four modules that run work in
  parallel imported `Parallel`/`delayed` at module level, pulling joblib's loky and
  cloudpickle stack into every import (~0.28 s of the start-up path once matplotlib was
  gone too). The imports moved into the parallel branches; `import normet` is now
  **0.84 s** cold (from 1.53 s; GUI import 1.80 s → 1.16 s), and PDP/SCM calls import
  joblib when they first parallelise.
- **`normalise` uses `n_cores` on every in-process path.** The batch-reduce path
  (the default with `aggregate=True`) and the vectorised path (quantiles or the wide
  seed table) have no worker pool of their own, so `n_cores` — which sizes the
  per-seed joblib pool on the memory-saving path — was accepted and ignored. It now
  sizes LightGBM's prediction threads (other backends ignore it). 300 resamples of
  the bundled MY1 record take **23.6 s at one thread, 6.4 s at four, 4.5 s at the
  default**; 60 resamples with quantiles **4.9 s → 1.3 s** at four. Output is
  bit-identical (prediction is per-row). This matters on HPC harnesses that pin
  `OMP_NUM_THREADS=1`, where LightGBM would otherwise never use more than one.
- **`ml_predict`'s default chunk size scales with the frame's width.** The fixed
  200,000-row default split a 13-feature, 1.74M-row normalise batch into nine
  predict calls, where one call is 12% faster at eight threads (4.28 s vs
  4.85 s) — the per-call setup is paid nine times and the tail chunk is small.
  The default now derives the row count from the frame's width for a ~256 MB
  float64 payload: ~2.4M rows at 13 features (an ordinary batch predicts in
  one call), ~33k rows at 1,000 features, so a wide frame still chunks. An
  explicit `chunk_size` is unchanged; predictions are per-row and identical.
- **Importing the GUI no longer imports matplotlib.** The figure toolbar
  subclassed `NavigationToolbar2QT`, which pulled in
  `matplotlib.backends.backend_qtagg` — ~1.2 s of a cold start — at
  `import normet.gui.main_window`, before any figure exists. The class moved to
  `normet.gui._toolbar` (still importable from `_widgets`, which loads it when
  a figure widget is built): the window import drops
  **1.51 s → ~1.06 s** and `matplotlib` is absent from `sys.modules` until a
  figure is created. The GUI app itself still sets the Qt backend at launch.
- **The window's start no longer pays PySide6's source scan.** `import PySide6`
  replaces `builtins.__import__` with a hook that reads every later module's
  source to look for `from __feature__ import ...`; a name already in
  `pyside_feature_dict` is skipped. Normet is hundreds of files, pandas and
  numpy among them: importing it after PySide6 measured **6.2 s against 1.1 s**
  in the other order (warm), and the window's own modules added ~1 s more when
  first imported. The GUI entry now gives the hook a mapping that answers
  "known" for every non-Qt module — none of normet's uses the feature switch,
  and PySide6's own modules keep their handling — so the read is skipped. In
  the app's order (Qt → library → window) the start measured **7.4 s → 2.4 s**
  to the constructed window (offscreen, warm); `packaging/launcher.py` applies
  the same patch before its own `import normet`. These are PySide6 internals,
  so the patch is guarded and pinned by `tests/test_startup_imports.py`.
- **`pdp` scores each variable's whole grid in one stacked prediction.** The
  per-feature loop predicted once per grid point, so all 50 points re-paid the
  frame-to-matrix conversion and the backend dispatch. One copy of the frame is
  now stacked per point (blocks capped at ~1e6 rows, so a large training set is
  not inflated by the full grid) and each point's mean/SD is read off the
  result. Three features at one worker **2.6 s → 1.8 s**, all 13 features
  **11.9 s → 7.2 s**; values are bit-identical to the per-point loop. The
  dispatch also moved from loky processes to threads, which share the frame
  and let LightGBM score while holding no lock: three features at
  `n_cores=4` **6.9 s → 1.5 s** — the process pool was slower than serial —
  and the 13-feature panel takes 5.6 s on the default pool. The R package's
  `nm_pdp()` carries the same change (its NEWS).
- **Every export lives in the action bar's Export menu.** The bar carried two export
  buttons and the File menu repeated both, so a command had three homes. One
  **⬇ Export** menu now holds **Result table as CSV…** (Ctrl+E) and the **HTML
  report** submenu of languages, each entry on while the window has what it writes;
  File keeps none of them. The command palette indexes the Export menu too.
- **Ollama is no longer a shipped provider.** LM Studio remains the local one
  (`lmstudio:...`); a server on another port — llama.cpp, vLLM, or an Ollama left
  running — is named with **Add a server…** on the model list, which keeps its
  address in the settings. A model still named `ollama:...` says that no server is
  named `ollama` and points at **Key…** to give its address.
- **The packaged app's start-up card follows the window's theme and language.** It
  was drawn dark and in English whatever View → Appearance and View → Language said.
  The card now reads the stored `appearance` and `ui_language` with Qt alone, before
  `import normet`, and takes the dark or light palette and the Chinese or English
  line — the same two choices the window applies when it starts.
- **`rolling` names its windows `window_01`, `window_02`, …** (was
  `rolling_0`, `rolling_1`, …): counted from 1 and zero-padded, to three
  digits past 99 windows, so the names sort in the windows' order. Code that
  picks the windows with `startswith("rolling_")` needs `window_` now, or
  `normet.window_columns(result)`, new, which lists them in order. Its log
  lines count windows from 1 too. Sessions saved by the GUI with the old names
  are renamed as they are loaded.
- **GUI start-up**: the packaged app keeps matplotlib's font cache between
  launches (PyInstaller's runtime hook rebuilt it every time, ~20 s on
  macOS), shows a splash before loading normet, and `import normet` no longer
  loads scipy.stats or scikit-learn (~1 s); they and flaml are imported in the
  background once the window is up.
- **GUI: the figure toolbar** — 16 pt icons tinted for the theme,
  the bar in the window's colours, and *For print…* (was *Export for
  Publication…*) before the pointer's read-out.
- **GUI: the window can be narrowed** — the result tabs' minimum width no
  longer held it at ~1050 px (~1570 px with the Assistant open): the action
  buttons wrap, the welcome text and cards reflow, and the Assistant's top row
  shrinks. Tabs are named without ①②③ numbers.
- **GUI: the *Trend test* box is now *Trend analysis*** — the box title, its
  button, the status-bar line, the command-log entries, the Assistant's and
  the agent's wording, and the manual's tables and captions all follow. Its
  *Run after each emission decomposition* tick is unticked by default: the
  decomposition's figure answers the question it was run for, and the trend
  is the next question — it starts on purpose, or with the tick on.
- **Packaging: the installers carry the Events step's model methods** —
  IsolationForest (`sklearn.ensemble`) and STL (`statsmodels.tsa.seasonal`),
  plus the AR/ADF submodules the feature engineering imports by name. The
  PyInstaller scan cannot see the `require()` imports, so the frozen app said
  *"'statsmodels.tsa.seasonal' is required but not installed"* although the
  build env had statsmodels; the submodules are named in the spec now, and
  the CI installer builds install the `stats` extra.
- **GUI: the left panel's step bar follows the window's theme** — its local
  tab rule made Qt draw the tab faces with fixed light colours, so the bar
  sat white on a dark window; the rule now takes its colours from the
  palette (`palette(window)`, `palette(base)`), so the six step names stay
  compact and theme-correct in both appearances.
- **GUI: the Decompose page's boxes read shorter** — the description sentence
  under each box title (Multi-scale decomposition, Trend analysis, Events &
  anomalies, Uncertainty decomposition) moved into the title's hover tooltip.
  The Trend and Uncertainty descriptions gain Chinese translations (they had
  shown in English).
- **GUI: the result tabs follow the workflow** — the questions the
  decomposition invites (Multi-scale, Trend, Events, Uncertainty) now sit
  directly after Decompose, with the separate steps Rolling and PDP after
  them: Data, Model, Normalise, Decompose, Multi-scale, Trend, Events,
  Uncertainty, Rolling, PDP. The status bar's step lights follow the same
  order and gain the Events light they were missing (its result existed; the
  light did not).
- **GUI: the multi-scale "Energy by band (σ)" reads per band** — the three
  bars drew in one colour and read as a single block; each band now takes
  the cascade's light-to-dark colour (D_fast the lightest blue, D_slow the
  darkest), and the band panels above the ranking use the same colours.
- **The `gui` extra asks for `PySide6-Essentials`.** The window imports only
  `QtCore`, `QtGui` and `QtWidgets`, but the `PySide6` metapackage also
  installs the add-ons (3D, WebEngine, Multimedia, ...) — hundreds of
  megabytes it never loads. `normet[gui]` and `normet[all]` now depend on the
  Essentials subset; import paths and code are unchanged.

### Internal
- **Start-up guard tests** — `tests/test_startup_imports.py` runs the window's import
  and the layout it opens in (`MainWindow().start_layout()`) in a subprocess and
  asserts that torch, sklearn, flaml and chronos stay out of the wait before the
  window shows. The packaged app draws its splash before `import normet`, so a
  heavy import creeping back into either path would only lengthen that wait.
- **CI: two non-blocking probe jobs for the version upgrades.**
  `probe: numpy 2.x (py3.12)` installs the suite's dependencies and then forces
  `numpy>=2.0` on top of the deliberate `numpy<2.0` pin; `probe: python 3.13`
  runs the same suite on 3.13. Both have `continue-on-error`, so they can
  never block a merge — a failure is the breakage list for the upgrade, to be
  fixed before the matrix widens.
- **CI: Python 3.13 joins the blocking test matrix.** Its non-blocking probe
  job is retired: the probe's first run (the full suite with every extra
  installed, 2026-10-23) passed, so per the execution plan the version
  graduates from probe to a first-class matrix row. `pyproject.toml` gains the
  `Programming Language :: Python :: 3.13` classifier.
- **numpy 2.x support: the `numpy<2.0` pin is lifted** and the matrix now
  resolves numpy 2.x. Two things stood in the way. The probe's breakage list
  (issue #69) held exactly one failure: this test's own CRPS reference
  integral called `np.trapz`, removed in numpy 2.0; it now reads
  `getattr(np, "trapezoid", None) or np.trapz`, which works on both (the
  naively suggested `getattr(np, "trapezoid", np.trapz)` is itself broken on
  numpy 2 — the default is evaluated eagerly). Separately, strict mypy fails
  on three reassignments whose declared shape comes from the pandas stubs
  while the new value comes from numpy stubs that disagree on the shape
  parameter (different pairs fail under 2.2.x and 2.5.x; 2.4.x is clean); the
  three variables now carry an explicit `np.ndarray` annotation, stable
  across every numpy 2.x stub generation. With the pin gone, `pyarrow<26`
  goes with it (it existed only because pyarrow 26 needs NumPy 2, which the
  old pin ruled out), and the `probe: numpy 2.x` job retires. The "generic
  unit" timedelta DeprecationWarnings the probe showed under numpy 2.5 come
  from pandas' own Timestamp/Timedelta arithmetic, not from normet call sites
  — issue #69 carries the attribution.
- **The `[all]` extra kept a `pyarrow<26` cap the NumPy pin's lift had
  already made redundant**: the same workaround existed in both extras and
  only `[gui]` was cleared. `normet[all]` now resolves pyarrow 26+ (and,
  with it, whatever NumPy the rest of the tree wants).
- **The GUI's left panel is built one workflow step at a time.**
  `_build_param_panel` had grown to 420 lines, one straight run of layout from
  Data to PDP; each step is now its own `_build_*_box()` returning the group
  box, and the panel method only lists them in the order the workflow runs in.
  The widgets and their attribute names are unchanged, so nothing that reads
  them had to change: this is a move, not a rewrite.
- **The window's shared names and tick-list helpers moved out of it.**
  `TIME_VARS`, `MET_DEFAULTS` and `DOCS_URL` now live in `normet.gui._constants`,
  and `_checked_items`, `_fill_checklist`, `_set_all` and `parse_groups` in
  `normet.gui._lists`. Both are imported back into `main_window`, so code that
  reads them from there -- the tests do -- is unaffected. They move so a panel
  split out of the window can import them without importing the window itself.
- **The window is assembled from mixins.** The chrome -- menu, tabs, docks,
  status bar, command palette, appearance menu, About -- is
  `normet.gui._shell.ShellMixin`; the left panel is `_panel_inputs` (Data,
  Columns, the CSV loader, the series preview), `_panel_train` (Step 1 and
  fine-tuning) and `_panel_analyse` (Steps 2-5 and multi-scale). `main_window`
  keeps the assembly, the state, the run handlers and `closeEvent`:
  2329 -> 1238 lines. Every method moved verbatim, and every widget is still an
  attribute of the window, so the run handlers and the tests are untouched.
  A mixin inherits `QMainWindow`, so `self.statusBar()` still means something
  when one module is read on its own. `normet.gui` is excluded from mypy
  (`ignore_errors = true` in `pyproject.toml`), so none of this is type-checked,
  and the mixins do not carry a hand-maintained mirror of the window's names
  pretending otherwise.
- **`_wrap_row` became `_widgets.wrap_row`.** The normalise box and the
  trajectory window each had their own copy, identical down to the margins.

### Performance
- **Nested thread pools in the multi-site drivers.** `do_all_multisite` and
  `decompose_multisite` run a process pool across sites, and each worker
  resolved `n_cores=None` to the whole allocation: P concurrent sites opened
  P full LightGBM/resampling pools inside one allocation, i.e. P × (C − 1)
  threads on C cores, with nothing in the call to signal it. With more than
  one site each site now runs its model training, resampling and
  normalisation calls single-threaded, so the outer pool is the parallelism;
  a one-site call keeps the callee's own default.
- **One predictor thread per parallel worker.** `pdp` and
  `normalise(memory_save=True)` run one Python thread per variable/seed, and
  each prediction then opened its own LightGBM pool sized from the OpenMP
  defaults (three variables × eight cores = 24 threads). They now pass
  `num_threads=1`; LightGBM prediction is bit-identical across thread
  counts, so only the contention goes.
- **Batched process-pool dispatch in the causal bootstraps and the multi-site
  drivers.** joblib's process backend pickles the callable *and what it
  closes over* once per task; every bootstrap replicate, placebo, SCM unit and
  per-site run closed over the whole panel and shipped its own copy — on a
  48 MB frame over 40 tasks, ~1.9 GB of pickling. All eight `Parallel` call
  sites now pass a `batch_size` equal to the per-worker task count, so a
  batch's tasks share one pickled closure. Measured on that frame with four
  workers: 2.0 s → 0.20 s.

### Fixed
- **A dateline-crossing source box matched nothing.** The bbox test was
  `lon >= lon_min & lon <= lon_max`, so a region written the natural way for
  the Pacific -- `(170, -10, -170, 10)`, meaning 170°E to 170°W -- covered
  no longitude and every residence fraction over it was silently 0. A box
  whose `lon_min` exceeds `lon_max` now wraps (both sides match), and the
  overlapping-region warning treats its two sides as one region too.
- **Residence fractions counted endpoints, not trajectory time.** The
  docstring promised the fraction of time inside a source region, but a
  tdump written every 3 h gave a 1-h step the same weight as a 3-h one (a
  synthetic 1-h + 3-h path: 0.17 by endpoint count against 0.067 by time).
  Uneven spacing is now weighted by the time each endpoint represents
  (trapezoid); evenly spaced output -- including the hourly default that
  `hyts_std` writes -- keeps its numbers exactly. `traj_inflow_deg` is also
  documented as the bearing from the receptor toward the air's origin, not
  the local arrival direction.
- **`join_trajectory_features` used one release interval for every site.**
  The default tolerance was the commonest step across all sites' successive
  trajectories: with a 3-hourly and a 6-hourly site the mode is 3 h, so
  every second 6-hourly window stayed empty and its hours took no
  trajectory. When both frames carry a `site_col`, each site now stands on
  its own release interval (a site with a single trajectory still matches
  only its exact time).
- **`analyze_lag` kept ±inf and warned on tiny series.** `pd.to_numeric`
  passes ±inf through though the docstring promises non-finite values are
  dropped, and one inf turns every ACF/PACF/CCF value into NaN; inf is now
  read as missing. A target with fewer than 3 finite observations raised
  NumPy divide warnings from `acf`/`pacf` rather than saying what was wrong;
  it now raises `ValueError`. The `driver_lags` docstring warns that the
  bands are per-lag: with `2 * max_lag + 1` lags tested, a few exceed them
  by chance on unrelated series (about 3 of 50 at `alpha=0.05`).
- **`add_lag_features` silently truncated a fractional lag.** `int(1.9) ==
  1` turned `lags=[1.9]` into a one-row shift; non-whole lags now raise
  (integer-valued floats such as `2.0` are still accepted).
- **`make_memory(bytes_limit=...)` raised `TypeError` on joblib 1.5.** The
  size cap has never been a `Memory` constructor argument; the handle now
  applies it through `Memory.reduce_size`, which deletes the oldest entries
  until the store fits (a single item larger than the cap leaves the store
  slightly above it).
- **The on-disk caches served results written by an older release.**
  `normalise(cache=...)` and `train_model(cache=...)` keyed on data and
  config only, so after an upgrade the same script silently reused what the
  previous code wrote. The installed package version now joins both keys,
  together with the installed versions of the packages whose code can change
  a result (`dependency_versions`: LightGBM, FLAML, XGBoost, scikit-learn,
  numpy, pandas, scipy), so a backend or numeric-library upgrade also
  retrains rather than serving the old result. A source change that keeps
  the version (an editable checkout) is still not covered; the `cache`
  docstrings say to clear the directory for one.
- **A `conditional_on` callable joined the `normalise` cache key by its
  `repr`.** `repr` names the function object's address: a fresh run never
  hit the entry, and — the silent direction — a closure whose captured
  value moved kept the same key, so the cache served the draws of the old
  filter. Callables are now fingerprinted with `callable_hash`
  (cloudpickle bytes of the code and captured state, plus the source text;
  `repr` only when cloudpickle cannot serialise the object). The same
  defect covered long iterables (an ndarray or Series `repr` abbreviates
  its middle: two filters differing only there shared a key) and sets
  (their `repr` order follows the per-process string hash, so the same
  condition missed after a restart). Conditions are now hashed by content.
- **Two resample pools differing only in a served non-covariate column
  shared a cache entry.** The `normalise` cache key hashed only the pool
  columns that were also covariates, but a pool serves whichever of its
  columns are in `variables_resample`, and that list is not validated
  against `covariates`. Served columns now join the hashed set.
- **Overlapping autosaves corrupted the kept session.** Each step's autosave
  wrote through the same `<last_session>.part` path before replacing it. Two
  steps finishing inside one dump (a large session takes seconds) let the
  second write truncate and keep writing the file the first had already
  renamed into place, so the session the window opens on its next start was
  a mix of two dumps. Each write now owns a temporary file; the last
  completed one lands whole.
- **The multi-scale decomposition was not reproducible across processes.**
  Its three scales were seeded with `seed + hash(name) % 1000`, and Python
  salts str hashes per process: the same seed drew different weather for
  the same run after a restart, so the logged `compute_multiscale` command
  replayed to different numbers (and two scale names could share one
  offset). The offsets are positional now (`seed`, `seed + 1`, `seed + 2`).
- **A `conditional_on` filter column outside the resampled set was invisible
  to the `normalise` cache key.** The key hashed only the resampled columns
  of the pool, but the filter selects rows by its own column's values, and a
  caller's `resample_df` may carry that column ("season", a wind sector)
  without it being resampled. Two pools differing only there shared a cache
  entry, so the second call returned the first pool's draws. The filter
  columns now join the hashed set.
- **`resolve_config` handed the caller's own config back.** When no keyword
  overrode a field, the function returned the instance it was given, and
  callers resolve derived values onto that result: `normalise` writes the
  features auto-detected from the model into `_cfg.covariates`, and
  `do_all`/`decompose` write the backend-dependent `n_samples`. Those values
  leaked into a config the caller may reuse -- with another model the stale
  feature list was wrong, with another backend the stale sample count skipped
  the per-backend default. The result is now always a fresh dataclass.
- **A NaT index entry entered the Theil–Sen time grid as int64's minimum.**
  `trend`, `theil_sen` and `detect_trend_breaks` converted the datetime
  index to nanoseconds; NaT became about −2.5e14 days, tilting every
  pairwise slope. A row with no timestamp is now dropped and counted in
  `n_dropped`, like a missing value.
- **An undefined placebo statistic read as maximal significance.** In
  `rmspe_ratio_test` a treated ratio of NaN (zero pre-RMSPE, an empty
  period) made every `>=` comparison False and returned `p = 1/(n+1)`;
  `placebo_in_space`/`placebo_in_time` did the same for a NaN observed or
  placebo statistic and counted all-missing placebos in the denominator.
  All three now return NaN when the test cannot run and count only finite
  statistics otherwise.
- **`rolling` and `decompose` had the same config-override defect.** Both
  accepted real-valued keyword defaults (`n_samples=1000`, `window_days=30`,
  `rolling_every=1`, `seed`, `n_cores`, `method="emission"`, ...) and passed
  them to `resolve_config`, which treats every non-`None` keyword as an
  override -- so a `RollingConfig` or `DecomposeConfig` handed in by the
  caller was silently reset to those values. Every such parameter now defaults
  to `None`; `rolling` resolves it against an empty `RollingConfig` when none
  is given, and `decompose` keeps the old unfiltered construction in that
  case, so an explicit `target=None` still raises. Tests drive a config
  whose `n_samples`, seeds and `method` differ from the defaults and assert
  the config wins.
- **`do_all` / `do_all_unc`: a config was silently ignored.** Their keyword
  parameters defaulted to real values (`backend="flaml"`, `split_method="random"`,
  `train_fraction=0.75`, `seed=7654321`, `n_models=10`, `confidence_level=0.95`),
  and `resolve_config` treats any non-`None` keyword as an override. A caller who
  put `backend="lightgbm", split_method="ts", train_fraction=0.6, seed=1` in a
  `SingleConfig` actually ran flaml on a random 0.75 split; `do_all_unc`
  likewise reset `n_models`/`confidence_level`. The parameters now default to
  `None` ("not given"), which leaves the config -- or the dataclass default --
  in place (`do_all`/`do_all_unc` hand `resolve_config` an empty config when
  none was given, so a `None` keyword never reaches the dataclass). A call
  that names neither a target nor a config still raises, rather than silently
  fitting a column called `"value"`. The
  ensemble's weights also came from the **training** row of `modStats` (its
  first row) rather than the held-out test split, so members were weighted by
  their resubstitution R²; they now use `set == "testing"`, falling back to
  `"all"` when no test split exists.
- **`decom_emi` built the emission series from the observations.** `emi_total`
  was `result.get("hour", observed)`, so a supported model without an `hour`
  feature (a subset of the calendar variables) returned the raw concentrations
  as the emission part, and a trend on it described the observations. It is now
  the fit with the last *present* calendar variable held observed. The
  components are likewise differenced against the previous present variable:
  the hardcoded `hour->weekday->day_julian->date_unix->base` pairs skipped
  every step whose neighbour was absent, leaving `hour` at the whole nested
  prediction and the named parts unable to reconstruct `emi_total`.
  `decom_met` with a pre-trained model and no `covariates` now generates the
  model's calendar features before filtering its feature list, as `decom_emi`
  already did; otherwise they were dropped from the resampling list and an
  AutoML backend predicted them as zero. The components are true increments:
  each is the next nested fit minus the previous fit (kept aside rather than
  read back from the frame, which by then held the increment), so they sum to
  `emi_total` for any set of present calendar variables.
- **The jackknife band used a variance factor for fits that never ran.** The
  standard error divided by the number of nominal donors `n`, but a failed
  delete-one fit leaves `m < n` columns in the jackknife table; the
  `(n-1)/n` factor then inflated the band for replications that contributed
  nothing. The factor now counts the estimates actually computed, `(m-1)/m`.
- **`pre_mape` went NaN on one zero observation.** A pre-period observation of
  zero has no percentage error; it was replaced by NaN and averaged with
  `np.mean`, so a single zero made the whole diagnostic NaN. The mean now runs
  over the non-zero observations, and is NaN only when all are zero.
- **OpenAQ `locality` was stored as a dict.** v3 returns
  `"locality": {"id": ..., "name": ...}`; `openaq_locations` put the object
  itself in the `city` column, which downstream text filters and equality
  checks cannot read. The name is extracted (the older string form still
  passes through).
- **`normalise` and duplicate timestamps / missing model features.**
  `aggregate=False` called `DataFrame.pivot`, which raises "Index contains
  duplicate entries, cannot reshape" on repeated dates; the rows are now
  averaged per date first. The batch-reduce path built one result row per
  *input* row, so with duplicate dates its shape (and observed average)
  depended on the memory setting; it now groups by date like the other paths.
  A covariate the model needs but the frame lacks was dropped in silence by
  `check_data` and then filled with zeros by the backend; `normalise` now
  raises `ConfigError` naming the missing columns.
- **`ml_predict` swallowed its own "no features" error.** The
  `ValueError("No model features found in newdata")` was raised inside the
  `try` that catches everything, so a frame missing every feature the model
  names fell back to predicting on *all* its columns; for LightGBM that
  reindexed to the feature names, filled them with zeros and returned
  confident nonsense. Only a model whose feature names cannot be read at all
  falls back now.
- **CRPS with tied quantile values was understated.** When several levels
  share a forecast value the CDF jumps there from the *lowest* to the highest
  level; `_row_crps` kept only the highest, so the interval below the tie
  started from the wrong height and the score came back at 0.2117 where the
  exact value is 0.4417 (hand-integrated). Both limits are tracked now.
- **`detect_trend_breaks`: a NaN won the split search.** The vectorised SSE
  helpers divided by a candidate piece's centred `sxx` without the guard the
  scalar helper uses; duplicate timestamps made it zero, the piece scored
  NaN, `argmin` picked it, and NaN propagated through the BIC and bootstrap
  gates (`sse_segmented=nan`, `n_segments=2`, `significant=True`). The guard
  is now in both helpers, and a non-finite gain is skipped.
- **Harmonic counterfactual: wrong band and tz-aware records.** The fields
  named `counterfactual_p10`/`p90` used ±1.645 σ -- the 5th/95th percentiles
  -- so an "80 %" band actually carried 90 %; they now use ±1.2816 σ. The
  documented string dates parse naive and were compared with a tz-aware
  `full_series` index, raising `Invalid comparison between dtype=datetime64[ns,
  UTC] and Timestamp`; they are now read on the index's clock.
- **`modStats` / `Stats` on nullable or text columns.** A nullable
  `Int64`/`Float64` column with `pd.NA` (or numeric text) arrives from
  `to_numpy` as object, and `np.isfinite` raises `TypeError`; both are
  coerced to float with missing values as NaN before scoring.
- **`process_date` on mixed UTC offsets, and `impute_values` on nullable
  integers.** A column spanning a daylight-saving change has two offsets;
  pandas 2 left an object column (with a `FutureWarning`) and pandas 3 raises,
  so `process_date` could not use it -- it is now read on UTC, and a column
  named `date` outranks one named `time` when both parse. `fillna` of a
  fractional median on an `Int64` column raised "Invalid value '1.5' for dtype
  'Int64'" (integral medians happened to work); the column is converted to
  float when the fill needs it.
- **`time_series_cv` counted the gap once per split.** `n_splits *
  test_size + gap * n_splits` refuses runs whose oldest fold still has
  training rows (30 rows, 5 splits of 5, gap 2: 35 >= 30). Only the oldest
  fold's gap can empty its training set, so the check is
  `n - n_splits * test_size - gap < 1`.
- **`find_dates` read the wrong column and could lose the date.**
  - A `time,date` header picked the clock column by header order: its
    "13:00" values were read as *today*, and inserting the result over the
    real `date` raised "cannot insert date, already exists". Candidates are
    now tried in name order (`date` before `time`) and only accepted when
    their values parse -- but a named column keeps its place with a minority
    of unreadable rows left `NaT`, so one typo does not make the whole table
    lose its times (or pop a modal "No times found" in the window).
  - A full-date column sitting beside redundant `year`/`month` columns was
    treated as a day-of-month part and every date came back `NaT`. It is
    only a part when its values are numeric days.
  - A table holding only a clock ("13:00") silently carried the day it was
    read; it now raises in words instead of dating the rows today.
- **ERA5 cache keyed only by site and date.** Asking for one variable and
  then two reused the one-variable CSV (the second variable missing), and a
  moved site reused the old coordinates' meteorology. The filename now
  carries a fingerprint of the sorted variables and the coordinates.
- **`load_model` of a file path looked inside the file.** It always passed
  `filename="automl.joblib"`, so the path `save_model` had just returned was
  read as a directory -- and a directory holding both names loaded the wrong
  model. A file path is now passed to the backend with `filename=None`.
- **Quantile calibration: wrong multi-band fits, NaN errors, and a first
  block that could abort the fit.** Four defects in
  `normet.foundation.calibration`, found by auditing the fit end to end:
  - `_pava` read each level's regression *weight* (the pool size behind its
    threshold, ~169) as a run of output entries. Two bands' thresholds
    `[0.907, 1.948]` came back `[0.907, 0.907]`, and a merged block left the
    tail uninitialised — so a `quantiles=(0.1, 0.5, 0.9, 0.95)` fit corrected
    the 95 % band with the 90 % threshold. Blocks now carry their entry count.
  - A missing held-out target was scored as a NaN error. The conformal
    threshold is an upper order statistic, so enough NaNs made it — and every
    corrected edge — NaN; fewer NaNs silently understated both coverage
    columns. Missing observations are now excluded from the scored set.
  - A `ValueError` on the oldest block (a gap at the record's start, a record
    just longer than one context) aborted the whole fit although every younger
    block was valid; a block truncated by the record's end likewise discarded
    all younger ones. Both now skip that block and try the rest, raising only
    if none worked.
  - `empirical_uncalibrated` compared each step's error to the block's
    *median* band edge, scoring a lead-invariant band the model never made (a
    band whose width varies with lead — the normal case — read as more
    calibrated than it was, while the calibrated column beside it was scored
    per step). Hits are now accumulated at each lead's own edges.
- **Chronos-2 counterfactual and ICN: wrong units and rejected inputs.**
  - `counterfactual`'s `pre_intervention_bias_pct` divided by the record
    mean without `abs`: on a negative-valued target (temperature, an anomaly,
    a flux) a near-perfect placebo read as millions of per cent, while
    `net_impact_pct` beside it was already right.
  - `counterfactual(..., quantiles=(0.1, 0.9))` raised `KeyError: 0.5` from
    inside the roll-forward, because the projection always walks on the
    median. It now validates the levels (three or more, median included) and
    names the requirement; the parameter is documented.
  - `Chronos2Estimator._check_index` accepted a uniformly *descending*
    index (its steps are all equal) — every caller slices the past by
    position, so the counterfactual would have conditioned on the oldest
    rows — and rejected month-end/quarter/year indexes, whose nanosecond
    steps differ although they are uniform in the unit the horizon counts.
    Order is now required; a calendar frequency is accepted.
  - `normalise_in_context` inferred every non-calendar covariate as
    meteorological, categoricals included: the A2 support check then read a
    site label with `dtype=float` and raised on the first anchor. The pool is
    now the numeric columns; a categorical covariate stays a factual
    covariate, and `MetPool.support_check` skips non-numeric columns.
  Two docstrings claimed behaviour the code did not have: `to_regular_index`
  said it interpolates covariates (it leaves the restored rows NaN, which the
  input builder fills with the context mean), and `use_calendar` said it
  appends the six calendar covariates (it includes the ones the frame carries;
  add them with `add_calendar_covariates`).
- **`date_unix` and the cycle picker read any datetime resolution as time.**
  pandas 2.x preserves a non-nanosecond unit (`datetime64[s]`/`[ms]`/`[us]`,
  which `read_parquet` hands back), and integer conversion counts in that unit:
  a second-resolution column produced `date_unix` ≈ 1.7 instead of ≈ 1.7e9 —
  nine orders of magnitude lost from the trend feature, with no error — and
  `usable_cycles` read a daily record's step as 8.6e-5 s and offered it an
  "hour of day" profile it cannot show. Both convert to nanoseconds first.
- **A file that cannot be read is named in the dialog.** Every Load dialog showed
  the reader's own sentence as the whole message; pandas' errors often name no file
  ("No columns to parse from file"), and the window's phrase-by-phrase translation
  could bend a library's `"{s0}: {s1}"` sentence. The first line is now
  "Could not read <name>."; the reader's words keep a line of their own.
- **The HTML report build says it is running, and a failure is shown.** The status
  line and a wait cursor are painted before the figures are drawn (scipy's first
  import can be most of the wait), and a build that raises says "Could not build the
  report." in a box instead of leaving a traceback in the terminal.
- **A Chinese name in a figure gets the Chinese face** even when no catalogue entry
  translates it — a factor, site or column the analyst wrote in Chinese is data, and
  `localize_figure` now gives it the fallback font rather than drawing boxes.
- **A light/dark switch now reaches the parameter panel and the welcome cards.**
  Qt changes the application palette, but a plain container keeps the palette it
  resolved at its last polish, and a style sheet's `palette(role)` is resolved when
  the sheet is applied — the welcome cards' `palette(base)` stayed white on a dark
  window. A theme change now clears inherited palettes and re-applies the sheets that
  name a palette role before the `themeChanged` signal is sent.
- **Worker pools were sized from the machine, not the allocation.**
  Nine parallel sections (normalise's threaded path, PDP, rolling,
  decomposition, placebo-in-space, effect bands, staggered/batch SCM and
  multisite) sized themselves with `(os.cpu_count() or 2) - 1`.
  `os.cpu_count()` reports the host, not the allocation: inside a 4-core Slurm
  job on a 32-core node it returns 32, so those sections asked for 31 workers
  on 4 cores. normet-r has resolved this correctly for a while via
  `.nm_available_cores()`; the Python package had no affinity, scheduler or
  cgroup awareness at all.

  New `normet.utils.available_cores()` takes the minimum of `sched_getaffinity`,
  `SLURM_CPUS_PER_TASK` / `SLURM_CPUS_ON_NODE` / `NSLOTS` / `PBS_NUM_PPN` /
  `NCPUS` / `LSB_DJOB_NUMPROC`, the cgroup v2 `cpu.max` quota (v1
  `cfs_quota_us` fallback) and `os.cpu_count()`, so a 4-core batch gets 4.
  `default_workers()` is that minus one for the parent process, and
  `resolve_workers(n)` returns an explicit count unchanged — every former call
  site now goes through it. `OMP_NUM_THREADS` is deliberately *not* consulted:
  it caps threads per process, and users set it to 1 precisely to stop BLAS
  oversubscribing underneath these pools.

  No published result changes: the default `normalise()` path batches
  sequentially and only logged the worker count.

- **OpenAQ adapter: fetches came back empty for UK/EEA-sourced records.**
  Live verification (2026-10-23) against the `/parameters` endpoint showed
  each pollutant registered under several `parameters_id`s that differ by
  unit family — ids 1-6 µg/m3 (what EEA-sourced records, e.g. the UK AURN,
  use), ids 7-10 ppm, ids 15/32/101/102 ppb — and the adapter's single-id
  map held only the ppm ids for no2/o3/co/so2, so those sensors were never
  found. A slug now resolves to its full id family with the mass id first:
  sensor lookup takes µg/m3 when both exist, values arrive on the same scale
  as the UK/EEA adapters, and ppm-only locations still work (unit as
  reported). Same pass, same endpoint: a location that never reported data
  carries `datetimeLast: null` and crashed the locations parse; `limit`
  above 1000 is a hard 422 so it is now capped and `openaq_locations` walks
  the `page` cursor (`offset` is silently ignored); several slugs can go out
  as one union filter, which the Data Studio's station find now uses.
- **`do_all(backend="chronos-2")` refused the meteorology its own error
  message tells you to pass.** The shared `covariates must be provided` guard
  ran before the backend dispatch, yet the zero-shot branch documents its
  weather as passable through `variables_resample` instead of `covariates`
  (nothing is trained); a call in exactly that documented shape died at the
  guard before reaching the check that owns the error. The branch now
  dispatches first, and hands `prepare_data` the resampled columns (the union
  with any named covariates) so the frame carries what the de-weathering
  conditions on -- the covariates-only and both-provided calls prepare the
  same columns as before.
- **Packaged app: Data → Find stations failed with "'rdata' is required but
  not installed"**. The data downloaders (requests, cdsapi, rdata) are
  imported by name at run time, which PyInstaller cannot see, and the build
  did not install the `data` extra; both are fixed, and `build_dmg.sh` stops
  if they are missing.
- **Dates whose UTC offsets differ are read as UTC.** Local time across a
  daylight-saving change (`+00:00` in winter, `+01:00` in summer) has no one
  time zone to keep: pandas 3 refused such a CSV at load ("Mixed timezones
  detected"), and pandas 2 loaded the dates as objects -- with a FutureWarning
  -- that training then refused ("Tz-aware datetime.datetime cannot be
  converted to datetime64 unless utc=True"). `to_datetime_coerced`, which every
  date-taking entry point and the GUI's logged script use, now converts them to
  UTC, each value by its own offset; a value with no offset among them is read
  as UTC. A column with one offset throughout keeps it, and an explicit `utc`
  is left to pandas.
- **GUI: a CSV of zoned dates with one that did not parse would not load.**
  Dates in `...Z` parse tz-aware, an unparseable one becomes NaT, and with
  both matplotlib could not convert the data preview's axis: the error stopped
  the load half-way. The preview now leaves out rows without a date.
- **Transport Studio's join handed rows trajectories that had not arrived, and
  filled rows no trajectory covered.** "Send to main window" joined with
  `merge_asof(direction="nearest")` and no tolerance: 04:00 and 05:00 took
  06:00's trajectory, and two days of trajectories filled all 240 rows of a
  ten-day panel. It now uses `join_trajectory_features`; a row none matches
  stays empty, and the log says how many rows took one.
- **Transport Studio reported a join that had not happened.** With site names
  that differed from the data's (`Marylebone` vs `MY1`) no column was filled,
  yet the log said "Merged 9 transport feature column(s)". It now says which
  sites the data has and leaves the data as it was. Sending features a second
  time gave `traj_*_x` / `traj_*_y` pairs in place of the columns; they are now
  replaced. A date that did not parse (`Merge keys contain null values`) or a
  tz-aware one (a CSV in `...Z`) stopped the join; both now go through.
- **`build_trajectory_features` kept one start height of several.** A run with
  starts at 100, 500 and 1500 m came back as one row, the 1500 m trajectory's:
  rows were de-duplicated on the receptor time alone. Each height now has its
  own columns (`traj_dist_km_500m`, ...), and tdumps that start from different
  places are refused instead of interleaved.
- **`run_back_trajectories` could return an earlier run's trajectory.** In a
  reused `work_dir`, a run that failed left the last call's `tdump` of the
  same name in place, and it was read as this run's: (10N, 20E) came back with
  London's trajectory. The file is now removed before each run.
- **`run_back_trajectories` with tz-aware times** raised in the met-file
  filter ("Cannot compare tz-naive and tz-aware"), and would have started each
  run at local wall time; times are now converted to UTC, HYSPLIT's clock.
  A receptor time off the hour now passes its minutes to `CONTROL` and its
  tdump name, rather than running at the hour; repeated times run once.
- **`run_back_trajectories` ran times no met file covers**: each was handed
  every file and failed on its own (240 failed runs for a January panel given
  June's GDAS1). They are now skipped with one warning.
- **Transport Studio's Cancel did not stop HYSPLIT, and quitting could crash.**
  Cancel only detached the task, which went on starting runs (7 to 30 in
  3 s); quitting left that thread running ("QThread: Destroyed while thread is
  still running"). `run_back_trajectories` takes a `stop` event that kills the
  run in progress; Cancel and closing the window (or the main window) set it.
- **Transport Studio, smaller things.** It checks that the met files cover
  the receptor times' backward windows before running, names the missing
  GDAS1 weeks and sets the download range to them; the range follows the
  data's dates (it was the last week's), and a download of more than four
  weekly files is asked about first. Met files already on disk can be used.
  Both sections can null trajectories cut short (`min_hours`). A source region
  or receptor row that is not one (lon_min above lon_max, a missing name,
  text for a number) is named in a message instead of skipped silently. The
  feature count no longer counts the `site` tag.
- **GUI: figure toolbar icons after a light/dark switch.** matplotlib draws
  them once, in the palette of the moment, so after switching themes with a
  window open they were white on the light bar. They are now drawn again when
  the palette changes.
- **GUI: command log and figures for print**.
  View → Command Log lists every action that decides a result as the normet
  call that repeats it -- opening a table, training, normalising,
  decomposing, rolling, partial dependence. An entry is written from the
  arguments the step actually ran with, and only once it succeeded. File →
  Save Session as Script… writes the log as a Python file; the test suite
  runs that file and checks it reproduces the window's model data and
  normalised series exactly. Every figure's toolbar gains Export for
  Publication…, which lays the figure out at a journal's column width (Nature,
  Elsevier, ACS presets, or custom) with its lettering in points, previews it
  at printed size, and saves PNG, TIFF, PDF, SVG or EPS on a white page at
  exactly that width.
- **GUI: the Assistant** (View → Assistant, Ctrl+Shift+A). A language model works in the window with ten tools: it reads
  the table, the model's fit, the normalised series and the decomposition
  (`normet.agent.readings`), and runs train, normalise and decompose as their
  buttons do. A proposed target and predictors come as a card the analyst
  applies; nothing changes until they do. It runs on API models through
  `normet.agent.llm` (Anthropic's Messages API or any OpenAI-compatible one,
  Ollama included; standard library only), with keys from the macOS keychain or
  the environment. It also runs on Claude Code or Codex signed in to the user's
  own plan, served the window's tools over MCP on 127.0.0.1
  (`normet.agent.mcp_http`) with no shell or files of their own.
- **`normet-mcp` and `normet.agent`: normet for AI assistants**. `normet.agent` is a tool layer with three parts:
  - text readings of a table, a model's fit (`modStats` on the held-out rows,
    feature importance), a normalised series (means by year or month), a
    decomposition, a synthetic control (`scm_diagnostics`) and a
    cross-validation;
  - do-all, decompose, SCM and cv jobs, each a `normet` CLI command run in its
    own process and folder, its command line kept as its record;
  - figures and the HTML report of a run.

  `normet-mcp` (extra `[mcp]`) serves these tools over the Model Context
  Protocol to Claude Code, Claude Desktop and other clients. Its instructions
  ask the assistant to propose the predictors and the donor pool for the user
  to confirm, and to quote only numbers the tools return. Files are read only
  under `$NORMET_MCP_ROOTS`.
- **GUI: light and dark appearance, command palette, welcome page**. The window and its figures follow the system's
  light/dark mode, or View → Appearance pins one (Qt 6.8+). Figures are still
  drawn in matplotlib's light colours and recoloured for a dark window; what
  the figure toolbar saves is always on a white page. Ctrl+K (⌘K) finds any
  menu command or result tab by typing. Before a table is open, the Data tab
  shows the examples, the recently opened tables and the data studios.
- **`decom_met`: feature groups and Shapley attribution.** `groups=` attributes
  the meteorological features in named groups -- e.g. `{"local": met_cols,
  "transport": traj_cols}` -- with one result column per group, and
  `attribution="shapley"` averages each feature's or group's marginal effect
  over every order it could be fixed in, instead of fixing them one at a time.
  The sequential split depends on the order: on the bundled MY1 transport case
  the summed trajectory contribution had a standard deviation between 4.2 and
  5.6 ug/m3 depending only on the order, and the top-ranked feature -- which
  sets the default order -- changed between refits. Shapley values are exact
  (all `2**k` coalitions, up to 10 features or groups -- four normalisations
  for two groups) or, with `n_permutations=`, sampled in antithetic pairs;
  every coalition is normalised once and with the
  same seed, and the contributions add up exactly to `prediction - emi_total`
  whichever way they are split. Groups default to Shapley; without groups the
  default stays sequential, and those results are unchanged to the bit. The
  chronos-2 path (`decompose(backend="chronos-2")`) takes the same options.
- **`normalise`: `resample_pools=` draws chosen variables from their own pool.**
  A pool's columns name the variables drawn from it -- whole rows at a time,
  independently of `resample_df` and of the other pools. With a single pool
  every contribution is measured against the *average* conditions in it, so a
  transport term is an anomaly with a mean near zero; `{"transport":
  clean_hours[traj_cols]}` measures transport against a reference air mass
  instead. Each pool has its own random stream, derived from its name, so a
  variable's draws do not move when other variables are fixed -- `decom_met`'s
  differences stay paired -- nor when other pools are added or renamed.
  Without pools the draws are unchanged, and results are bit-identical on
  every execution path.
- **Notebook 05: transport vs local attribution.** On the bundled MY1 case:
  one transport-aware model scored on a blocked split, the grouped Shapley
  split into `local` and `transport`, the two fixed orders it averages, the
  two-model difference of notebook 04 for comparison (correlation 0.59 with
  the grouped transport term, 40% of its amplitude), and a clean-Atlantic
  reference pool that turns the transport anomaly (mean -0.03 ug/m3) into a
  contribution (mean +2.70 ug/m3). Notebook 04 stays as the paper's
  reproduction.
- **`decom_met` / `decom_emi` forward `resample_df`, `resample_pools` and
  `conditional_on`** to every `normalise` call. `decom_met` previously passed
  `resample_df=None` whatever it was given, unlike normet-R's
  `nm_decom_met`. The chronos-2 path refuses all three rather than ignore them.
- **Trajectory quality columns and `min_hours`.** `trajectory_features` /
  `build_trajectory_features` / `run_back_trajectories` now emit
  `traj_n_endpoints` and `traj_age_max_h`, and take `min_hours`. A trajectory
  that HYSPLIT ended early (met files ran out, or it left the domain) used to
  be indistinguishable from a legitimately short-range one: its `dist_km`
  shrank and its residence fractions were taken over fewer points, with no
  flag. `min_hours` sets every feature except the two quality columns to NaN for
  such rows; it is opt-in, so existing frames only gain two columns.
  `run_back_trajectories` warns about truncated runs either way.
- **Notebook 04, section 4: a leakage-safe check.** A 6-hourly trajectory
  carried onto an hourly panel is piecewise constant, so under a random split a
  tree model uses it as a time fingerprint: on the bundled MY1 case a
  trajectory-only model scores test R^2 0.84 (random) vs 0.46 (`month_ts`),
  against 0.79 / 0.46 for local meteorology. The new section crosses
  {nearest, backward} join with {random, `month_ts`} split. The transport-aware
  gain survives the blocked split (+0.07 with the backward join) but is smaller
  than the random-split, nearest-join figures of sections 2-3 suggest. Those
  sections are left as the paper's reproduction.
- **`run_back_trajectories` passes each run only the met files it can touch.**
  It used to list every file in `met_files` in every `CONTROL`, so a multi-year
  GDAS1 archive (hundreds of weekly files) was handed to, and opened by, each of
  thousands of runs. GDAS1 files (`gdas1.<mmm><yy>.w<N>`) are now selected from
  the dates in their names against the run's `[receptor - hours_back, receptor]`
  window; files with any other name are always kept, and if nothing overlaps all
  are passed so `hyts_std` reports the coverage problem itself (ported from
  `normet-r`). The window is widened by one 3-hourly record on each side: probed
  against `hyts_std` with two adjacent daily ARL files, a start time between one
  file's last record and the next file's first (23:30, 23:59) failed with only
  the earlier file and ran with both, so a strict overlap test would break
  hourly receptors in the last hours of every weekly file.
- **`build_trajectory_features` warns when source regions overlap.** An
  endpoint inside several regions counts towards each, so overlapping regions'
  residence fractions add up to more than 1 and are not shares of the
  trajectory -- in the bundled MY1 features they sum past 1 for 52% of
  trajectories, which nothing flagged. The warning names the overlapping pairs
  (boxes, and polygons via shapely); regions that only touch do not count. A
  bounding box given as numpy scalars (e.g. `np.float32`) used to fall through
  to the polygon path and crash; it is now read as a box.
- **Trajectory docs.** The `normet.io.trajectory` docstring now shows the
  backward-aligned join (`merge_asof(direction="backward")`) instead of
  `ffill(limit=8)`, which was longer than a 6-hourly release interval, and
  documents the random-split caveat.
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
- **The EEA adapter fetches again.** `fetch_eea_data` scraped the Discomap
  FME endpoint, which the agency has since closed behind HTTP Basic
  authentication: every call answered `401` (observed 2026-10-23, with or
  without query parameters). It now queries EEA's official Parquet download
  service — same country/pollutant/year interface, no key — verified against
  the live service. Reading the answer needs `pyarrow`, added to the
  `[data]` extra; the service caps one synchronous answer, so the adapter
  asks one calendar month at a time and concatenates. The response contract
  follows the service; see *Changed (breaking)* below.
- **`normalise` de-weathering depended on `memory_save`.** The threaded path
  (`memory_save=True`) resampled through `generate_resampled`, whose stream-0
  seed selected pandas' legacy `RandomState`; the batch and vectorised paths
  drew with `np.random.default_rng(seed)`. The same `seed` therefore produced
  a different normalised series whenever the memory setting changed. Every
  path now draws through `_draw_rows`, and `generate_resampled` matches them;
  the default path's values and the stream-k>0 pool draws are unchanged. The
  batch path also folds the observed values with a broadcast view instead of
  materialising a `b_size × n_rows` copy each batch.
- **`conformal_effect_interval` built its null from the raw effect series.**
  The sub-sampling blocks were drawn without imposing H0, so every window
  mixing pre- and post-period rows carried a share of the very effect under
  test: the null spread over the effect size and the interval came back wide
  around the estimate. The post-period residuals are now shifted down by the
  observed ATT before the blocks are drawn, as the old comment already
  claimed. A noiseless +10 step, whose true interval is [10, 10], used to
  return [0.4, 19.6].
- **`pre_trend_test` could SIGBUS on macOS while the GUI ran it.** The slope
  standard error came from `np.linalg.inv(x.T @ x)`; inside the SCM window's
  worker thread that LAPACK call died with `Fatal Python error: Bus error`
  (Accelerate/OpenMP), taking the fit-checks job down (`test_gui_scm_inference`
  and the SCM-studio test in `test_assistant` reproduced it at `origin/main`,
  and both files now pass with default BLAS threads). The 2×2 inverse is
  written out in closed form, which is the same arithmetic without LAPACK; a
  new test pins the resulting p-value to `scipy.stats.linregress`.

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
- **`fetch_eea_data`'s response follows the EEA Parquet service** (see
  Fixed). `site` is now the reporting sampling-point id
  (`"GB/GB_SamplingPoint_65449"`), not the legacy `GB0001A`-style code; the
  columns are `[date, site, country, pollutant, value, unit, aggregation,
  validity]` — latitude and longitude are gone, as the service does not
  report them; `station=` filters identifiers by case-insensitive substring
  (client side — the service has no station parameter); the service's `-99`
  missing sentinel becomes `NaN`; and the `output=` parameter is dropped,
  having configured nothing but the retired FME transport.

### Fixed
- **Lag analysis silently skipped pre-whitening on statsmodels 0.15.** The
  AR fit passed `old_names=False`, which statsmodels 0.15 removed, so the
  pre-whitening step raised, was caught, and fell back to the raw
  cross-correlation -- the spurious-peak problem pre-whitening exists to
  remove -- with only a log warning. It also turned CI red on every Python
  version (`test_prewhitened_ccf_recovers_true_lag`). The argument is dropped;
  its default had been False since before 0.14, so older statsmodels behave
  as before.
- **One long record gap aborted a whole Chronos-2 run, and the workaround
  corrupted it.** `Chronos2Estimator.deweather` and `.predict` refused the
  entire call as soon as a single block's context had less than
  `min_context_coverage` of its target observed, so a multi-year run with one
  outage could not complete. The only way through was
  `min_context_coverage=0`, which let every gap through as a forecast scaled to
  nothing: in a 16-year, three-site run, 71% of the hours inside week-long gaps
  at one site came back with a de-weathered level below 0.5 ug/m3, with no
  warning. The rolling paths now leave just those blocks NaN, log how many
  blocks and rows were skipped, and raise only if no block has enough context.
  The single-anchor methods (`predict_quantiles`, `covariate_sensitivity`,
  `counterfactual`) still refuse. Runs that used to succeed are unchanged to
  the bit. Also on `decompose(backend="chronos-2")`, where every coalition
  skips the same blocks, and whose default sequential order is now ranked from
  the latest context the model accepts rather than from the record's end,
  which a trailing outage made refuse. The `deweather` docstring now says what
  `dew_pNN` is: the model's predictive quantile averaged over the resampled
  weather, not a confidence band for `dew_p50`.
- **`decom_met(df, model=None)` crashed on a missing target** with "All arrays
  must be of the same length": the observed series was taken from the input,
  while the model trained on the fly -- and so the frame being decomposed --
  had dropped the rows with a missing target. It is now taken from the frame
  actually decomposed. Affected the CLI `decompose --method meteorology` on any
  table with gaps.
- **A feature listed twice in `decom_met`'s `variable_order` was credited with
  exactly zero.** The set-equality check let the duplicate through, the second
  pass overwrote the feature's column with `0`, and its effect moved into
  `met_noise` (measured: contribution sd 0.0 instead of 3.0, `met_noise` sd 3.2
  instead of 0.5). Duplicates are now refused.
- **Decomposition docs described a leave-one-out scheme.** Both decompositions
  fix variables cumulatively, so each component is conditional on those fixed
  before it; the docstrings, GUI tooltip and user guide now say so, and
  `met_noise` is documented as what it is -- the model residual shifted by
  `met_base` -- rather than a meteorological term.
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
- **Log-scale normalisation (`log_transform`)** for pollutants that span
  sub-unit to hundreds of µg/m³. `normalise`, `do_all` and `do_all_unc`
  accept `log_transform: bool = False`: with it on, the model trains on
  `log1p(value)` and the de-weathering runs on that scale, so errors count in
  relative rather than absolute terms (a fixed relative miss maps to a bounded
  error at every level); every column
  of the result (`observed`, `normalised`, the quantile columns, the
  ensemble's means and bounds) is back-transformed through `expm1` and is
  therefore in the original units and non-negative by construction. A
  negative target is rejected with an error that names the offending rows.
  The prepared frame returned by `do_all` carries `log1p(value)`, the frame
  the model was trained on; `do_all_unc`'s per-member statistics that drive
  the weights are computed on that same log scale. `Stats`/`modStats` gain
  `logMAE` and `logRMSE` — accuracy in log1p space, the space a log fit
  optimises (pairs with either side below -1 are dropped from those two
  metrics only). The HTML report notes the scale when a run's config carries
  the flag, with the share of observations below 1 in the original units.
  Default `False` is bit-identical to before. User guide:
  `guide/examples_normalisation.md` ("Log-scale fits").
- **Embedding diagnostics (`normet.foundation`)** for the choice the
  package deliberately does not make for you. `embed_multisite` returns one
  768-D vector per site and stops there (the removed `cluster_multisite`
  existed so the grouping is the user's decision); these functions supply
  the readings that decision starts from, and no more. `silhouette_curve`
  scores the mean silhouette for each `k` in `k_range` — infeasible `k`
  (`k >= n`, a collapsed cluster) get a NaN row rather than being dropped,
  so the curve is honest about what was not scored; the KMeans it uses as a
  yardstick is discarded before the score, so the result carries no
  membership and no recommended k. `embedding_projection` projects the
  vectors to two dimensions — `"pca"` is pure numpy (no new dependency),
  `"umap"` needs `umap-learn` — a view for the eye, not a statistic: PCA
  component signs are not stable across platforms. `plot_embedding_diagnostics`
  draws the two panels together, coloured by a grouping *you* supply through
  `groups` (or one colour otherwise), and annotates site names at 20 or
  fewer sites. Both optional imports (scikit-learn, umap-learn) are lazy.
  User guide: `guide/examples_foundation.md` ("Reading the embedding space
  before you group it").

### Changed
- **`Chronos2Estimator.deweather` docstring** now states plainly that each draw
  resamples the meteorological covariates in the *context* as well as the
  horizon while the target history stays observed, which breaks the
  concentration--weather correspondence the model needs to learn the
  meteorological response (a 2026-10 pilot measured daily NRMSE 61 % vs 28 %
  when only the future is intervened). Behaviour is unchanged. The
  `context_length` docstring is corrected: the checkpoint's native context is
  8192 h, not 2048 h (2048 h is the pipeline default this class uses).

### Development snapshot — 2026-06-20 (previously headed 0.4.0)

#### Changed (breaking)
- **Dropped Python 3.9 support** (EOL since 2025-10). Minimum is now Python 3.10,
  matching the modern scientific stack (numpy ≥ 2.1, scipy ≥ 1.14, pandas 3.x,
  scikit-learn ≥ 1.7). Update `ruff`/`black`/CI targets accordingly.

#### Added
- `scm_robust` — Robust Synthetic Control (Amjad, Shah & Shen 2018): HSVT
  de-noising of the donor matrix followed by (optionally ridge) regression.
  Available via `run_scm(scm_backend="robust")`.
- `scm_mcnnm` gains cross-validated `lam` selection (`cv=`) and an optional
  randomized-SVD fast path (`max_rank=`) for large panels.
- `DEFAULT_SEED` constant in `normet.utils` centralising the default random seed.

#### Performance
- `scm()` now solves all timestamps from a single SVD of the donor design with
  exact leave-one-out alpha selection, instead of refitting `RidgeCV` per
  timestamp (equivalent results, large speed-up on long pre-periods).
- `normalise` auto-convergence path builds its per-date accumulator by zipping
  columns instead of `DataFrame.iterrows()`.

#### Fixed
- `pip install normet[all]` now pulls the `data` adapters' dependencies
  (`requests`, `cdsapi`); previously the I/O adapters were unusable under `[all]`.

#### Internal
- Data adapters (OpenAQ, EEA, DEFRA) share a single HTTP helper
  (`io/_http.py`) with timeout, exponential backoff, and HTTP 429 handling;
  EEA gained retries it previously lacked.
- Shared synthetic-control primitives (`pivot_panel`, `solve_simplex_weights`)
  extracted to `causal/_common.py`, de-duplicating `scm` / `variants`.
- Repository hygiene: notebook data artifacts gitignored; broad `except` blocks
  given debug logging on silent fallbacks.

### Development snapshot — 2026-06-10 (previously headed 0.3.0)

#### Added

##### Pipelines & analysis
- `do_all_multisite` / `decompose_multisite` for parallel per-site execution.
- `multisite_apply` generic per-site dispatcher.
- `decompose(method="shap")` — single-pass per-feature additive attribution
  (FLAML via `shap`).
- `decompose_shap` direct API.
- `normalise(return_quantiles=...)` — emit quantile columns of the per-date
  resample distribution (resampling uncertainty).
- `normalise(conditional_on={...})` — counterfactual scenarios by filtering
  the resample pool (scalar / iterable / callable values supported).

##### Causal
- New SCM backends: `scm_abadie` (classic simplex), `did_baseline` (DiD
  parallel-trends), `scm_mcnnm` (Matrix Completion Nuclear-Norm).
- `BACKENDS` registry now exposes `{scm, mlscm, abadie, did, mcnnm}`.
- `scm_diagnostics` — pre-period fit (RMSE/R²/MAE/MAPE), Herfindahl index,
  effective N donors, top-k donor weights.
- `loo_weight_stability` — leave-one-donor-out drift summary.
- `conformal_effect_interval` — finite-sample sub-sampling conformal CI for
  the post-period ATT.
- `rmspe_ratio_test` — Abadie's RMSPE-ratio placebo significance test.

##### Modelling
- `ml_predict(chunk_size=...)` — FLAML predictions are now batched by default
  to avoid blowing up memory.
- `ml_predict_dask` — lazy partition-wise predict for Dask DataFrames.

##### Utilities
- Feature engineering: `add_lag_features`, `add_rolling_features`,
  `cyclical_encode`, `wind_to_uv`.
- Walk-forward CV: `time_series_cv`, `cv_score`.
- `modStats(by=...)` — time-stratified metrics; built-in tokens
  (`season`, `month`, `hour`, `weekday`, `year`, `day_of_year`).
- Caching helpers: `make_memory`, `dataframe_hash`, `config_hash`.
- Provenance: `NormetRun`, `make_run`, `save_run`, `load_run` (joblib +
  JSON sidecar archive).

##### I/O
- `normet.io` package; `prepare_from_xarray` / `sample_xarray_at_sites` for
  ingesting gridded NetCDF/Zarr data.

##### CLI
- `normet` console entry point with subcommands `do-all`, `decompose`, `scm`,
  `cv`, `info`. Supports `--config foo.yaml` for any subcommand.

##### Docs
- Sphinx site under `docs/`; readthedocs config; user guides on
  normalisation, decomposition, SCM, multisite, caching.

##### Project
- Tests: 55 tests across 14 files; CI matrix on Py 3.9–3.12 with coverage
  (`--cov-fail-under=50`) and ruff lint + format checks.
- Pre-commit config (ruff + mypy).
- Optional dependency extras: `flaml`, `shap`, `xarray`, `dask`,
  `cli`, `docs`, `dev`, `all`.

##### Plotting (#26 / #29)
- `normalise_plot(result_df, ...)` — observed vs. deweathered time series with
  optional quantile uncertainty band; supports `resample=` for daily/weekly
  aggregation before display.
- `plot_bayesian_scm(result, cutoff_date=...)` — two-panel posterior credible
  band visualisation for :func:`bayesian_scm` output.
- `plotting.__all__` now exports all six public functions.

##### Causal (#29)
- `bayesian_scm` — `weights_summary` HDI column extraction is now robust to
  arviz version differences (dynamically searches for `hdi_*` columns instead
  of relying on a hardcoded format string).


#### Fixed
- `pipeline.do_all` had a stray top-of-file token that made the package
  fail to import; duplicate `aggregate=` kwarg; dead `mod_stats` computation.
- `analysis.rolling` and `analysis.decomposition` were passing
  `weather_df=None` to `normalise`, which only accepts `resample_df`.
- `causal/__init__.py` `__all__` list had missing commas, silently joining
  symbol names.

#### Changed
- Top-level `__init__.py` re-exports the new symbols (~35 additions).
- `pyproject.toml` declares `scikit-learn` as a core dependency (SCM uses
  `RidgeCV`).

---

## [0.2.4] — 2025-10-29
Last release before this changelog began. See git history for details.
