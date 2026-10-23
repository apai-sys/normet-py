# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Changed
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

### Internal
- **CI: two non-blocking probe jobs for the version upgrades.**
  `probe: numpy 2.x (py3.12)` installs the suite's dependencies and then forces
  `numpy>=2.0` on top of the deliberate `numpy<2.0` pin; `probe: python 3.13`
  runs the same suite on 3.13. Both have `continue-on-error`, so they can
  never block a merge — a failure is the breakage list for the upgrade, to be
  fixed before the matrix widens.
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

### Fixed
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
