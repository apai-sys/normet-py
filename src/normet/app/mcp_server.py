"""normet as an MCP server: ``normet-mcp``, for Claude Desktop, Claude Code and
any other MCP client, over stdio on this machine.

Install and register it::

    pip install "normet[mcp]"
    claude mcp add normet -- normet-mcp                 # Claude Code
    # Claude Desktop, claude_desktop_config.json:
    #   {"mcpServers": {"normet": {"command": "normet-mcp"}}}

The tools are :mod:`normet.agent`'s: the examples and a table's readings;
do-all, decompose, SCM and cross-validation started as jobs (each a ``normet``
command run in a child process, its folder under ``$NORMET_MCP_HOME``,
``~/.normet/mcp`` by default) and watched, cancelled, read; a run's figures and
report.

The models are trained here and the tables stay here: a tool gives the model
normet's readings, small tables and figures -- never the data. Files are read
and written under ``$NORMET_MCP_ROOTS`` (folders separated as PATH separates
them; the home folder where unset).

Works with the MCP Python SDK 1.x (FastMCP) and 2.x (MCPServer).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import sys
from collections.abc import Callable
from typing import Any

INSTRUCTIONS = """\
normet removes the effect of weather from air-quality time series (meteorological
normalisation, or de-weathering) and estimates the effect of interventions with
synthetic control.

Work as an analyst's assistant, in this order: look at the data (normet_inspect_data,
with the target), train a model and normalise (normet_start_do_all), read how well the
model predicts the held-out rows and what the normalised series shows (normet_job_status
once the job is done, or normet_read_run), then decompose (normet_start_decompose) or,
for a panel of sites and an intervention date, run a synthetic control (normet_start_scm).

Jobs return an id at once; ask normet_job_status until the job has ended. A job's
command line runs again as it is.

Where you help most is choosing what to model and judging the result. Recommend, with
your reasons and how sure you are; the user decides.
- The predictors: propose the target and the meteorological covariates from the table's
  readings (roles, missing values, correlation with the target), say what you left out
  and why, and ask the user to confirm before starting a job.
- The model: judge it on the 'testing' rows (R2, RMSE, FAC2); say plainly where it is
  weak, and what would test it further (normet_start_cv).
- The result: describe what the normalised series and its meteorological part show,
  by year or month as the reading gives them; do not claim a trend or a cause the
  numbers do not show. A trend is what normet_trend says (Mann-Kendall, Theil-Sen
  slope, the block-bootstrap-corrected p where the record is autocorrelated):
  quote the corrected p it names, not the classical one.
- A synthetic control: propose the treated unit, the cutoff and the donor pool; judge
  the pre-period fit before reading the effect.
Quote normet's numbers as the tools give them; do not compute or invent others.
"""

WORKFLOW = """\
De-weathering {what} with normet:

1. normet_inspect_data with the target: the table's span and step, gaps, each column's
   role, missing values and correlation with the target. PROPOSE the covariates
   (meteorological columns) with your reasons; ask the user to confirm.
2. normet_start_do_all with the confirmed target and covariates. Watch it with
   normet_job_status.
3. Read the model's fit on the 'testing' rows and the normalised series' means by year or
   month. Say what they show and where the model is weak. normet_figure draws them.
4. If the user wants the change split by time scale or by meteorological variable,
   normet_start_decompose (method 'emission' or 'meteorology').
5. If the user asks whether the tested series is increasing or decreasing and how fast,
   normet_trend on the result table (its emi_total or met_total).
6. normet_write_report for the run.

Quote numbers only as the tools give them.
"""


def _sdk() -> tuple[Any, Any, Any]:
    """The SDK's server class, image and tool-error types: 2.x's MCPServer or
    1.x's FastMCP. Loaded by name: each major lacks the other's modules, so a
    static import of either is an error to a type checker under the other."""
    from importlib import import_module

    for server, name, image, error in (
        (
            "mcp.server.mcpserver",
            "MCPServer",
            "mcp.server.mcpserver.utilities.types",
            "mcp.server.mcpserver.exceptions",
        ),
        ("mcp.server.fastmcp", "FastMCP", "mcp.server.fastmcp", "mcp.server.fastmcp.exceptions"),
    ):
        try:
            return (
                getattr(import_module(server), name),
                import_module(image).Image,
                import_module(error).ToolError,
            )
        except ImportError:
            continue
    raise SystemExit(
        'normet-mcp needs the MCP Python SDK: pip install "normet[mcp]" (or pip install mcp)'
    )


#: What normet raises at what it was asked -- a wrong name, a path outside the
#: folders, a missing column -- and says why. Its words reach the model, which
#: can then ask again; anything else is a fault, kept on the server's side.
ASKED_WRONG = (ValueError, LookupError, OSError)


def _quiet(fn: Callable[..., Any], tool_error: Any) -> Callable[..., Any]:
    """A tool whose prints go to stderr -- stdout is the protocol's channel -- and
    whose answer to a wrong request reaches the model in normet's own words."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                return fn(*args, **kwargs)
            except ASKED_WRONG as exc:
                raise tool_error(str(exc) or type(exc).__name__) from exc

    return wrapper


def build_server(tools: Any = None) -> Any:
    """The MCP server over *tools* (a :class:`normet.agent.Tools`; one over the
    default job store and folders otherwise)."""
    from ..agent.tools import FIGURES, Tools

    Server, Image, ToolError = _sdk()
    t = tools or Tools()
    mcp = Server("normet", instructions=INSTRUCTIONS)

    def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
        mcp.tool(name=fn.__name__, description=fn.__doc__)(_quiet(fn, ToolError))
        return fn

    @tool
    def normet_list_examples() -> str:
        """The example tables normet ships, with what each is for. Give a tool
        example=<name> to use one."""
        return t.list_examples()

    @tool
    def normet_inspect_data(
        example: str | None = None, data: str | None = None, target: str | None = None
    ) -> str:
        """A table before anything is fitted: rows and columns, the dates' span, usual
        step and missing timestamps, whether it is a panel of units, and each numeric
        column's role (pollutant, met, other -- read from its name), missing values,
        range and, with target=<column>, its correlation with the target. Give
        example=<name> or data=<path of a CSV/Parquet/Feather table with a 'date'
        column>."""
        return t.inspect_data(data=data, example=example, target=target)

    @tool
    def normet_start_do_all(
        target: str,
        covariates: list[str],
        example: str | None = None,
        data: str | None = None,
        resample_vars: list[str] | None = None,
        backend: str | None = None,
        n_samples: int | None = None,
        split_method: str | None = None,
        train_fraction: float | None = None,
        normalise_method: str | None = None,
        control_variates: bool = False,
        block_mode: str | None = None,
        block_hours: int | None = None,
        dml_cutoff_days: float | None = None,
        seed: int | None = None,
        label: str = "",
    ) -> str:
        """Start a job: train a model of `target` on `covariates` (the user's choice;
        normet adds the time features itself) and normalise it by resampling the
        meteorology (`resample_vars`, all covariates by default) n_samples times
        (normet's default 300). backend: 'flaml' (AutoML, the default) or 'lightgbm'
        (faster); split_method: 'random', 'ts', 'month_ts', 'season_ts';
        train_fraction: share of rows trained on (0.75). normalise_method: 'standard'
        (Monte-Carlo resampling) or 'dml' (orthogonal Double-ML with blocked
        out-of-fold prediction); control_variates: reduce Monte-Carlo variance;
        block_mode: 'independent' or 'markov' block resampling (needs block_hours).
        Writes normalised.csv and the run (run.joblib); poll normet_job_status."""
        return t.start_do_all(
            target,
            covariates,
            data=data,
            example=example,
            resample_vars=resample_vars,
            backend=backend,
            n_samples=n_samples,
            split_method=split_method,
            train_fraction=train_fraction,
            normalise_method=normalise_method,
            control_variates=control_variates,
            block_mode=block_mode,
            block_hours=block_hours,
            dml_cutoff_days=dml_cutoff_days,
            seed=seed,
            label=label,
        )

    @tool
    def normet_start_decompose(
        target: str,
        covariates: list[str],
        method: str = "emission",
        example: str | None = None,
        data: str | None = None,
        backend: str | None = None,
        n_samples: int | None = None,
        seed: int | None = None,
        label: str = "",
    ) -> str:
        """Start a job: decompose `target`. method 'emission' splits the normalised
        series into time scales (trend, seasonal, weekly, diurnal); 'meteorology'
        splits observed − normalised into each meteorological variable's part. Trains
        its own model on `covariates`. Writes decomposition.csv; poll
        normet_job_status."""
        return t.start_decompose(
            target,
            covariates,
            method=method,
            data=data,
            example=example,
            backend=backend,
            n_samples=n_samples,
            seed=seed,
            label=label,
        )

    @tool
    def normet_start_deweather(
        target: str,
        met_vars: list[str],
        example: str | None = None,
        data: str | None = None,
        method: str = "icn",
        estimate: str = "level",
        n_samples: int | None = None,
        context_length: int | None = None,
        prediction_length: int | None = None,
        log_transform: bool = False,
        surprise_tau: float | None = None,
        save_bank: bool = False,
        bank_path: str | None = None,
        seed: int | None = None,
        model_path: str | None = None,
        label: str = "",
    ) -> str:
        """Start a zero-shot Chronos-2 de-weathering job (default future-only 'icn'
        or legacy 'deweather') conditioning on `met_vars`. surprise_tau gates
        anchor draws by exp(-(err/tau)^2); save_bank writes the raw ICN draw
        bank (bank.joblib) for offline re-aggregation, and bank_path re-aggregates
        a saved bank instead of running the model. Writes normalised.csv;
        poll normet_job_status."""
        return t.start_deweather(
            target,
            met_vars,
            data=data,
            example=example,
            method=method,
            estimate=estimate,
            n_samples=n_samples,
            context_length=context_length,
            prediction_length=prediction_length,
            log_transform=log_transform,
            surprise_tau=surprise_tau,
            save_bank=save_bank,
            bank_path=bank_path,
            seed=seed,
            model_path=model_path,
            label=label,
        )

    @tool
    def normet_start_scm(
        treated: str,
        cutoff: str,
        example: str | None = None,
        data: str | None = None,
        donors: list[str] | None = None,
        date_col: str | None = None,
        unit_col: str | None = None,
        outcome_col: str | None = None,
        backend: str | None = None,
        covariates: list[str] | None = None,
        resample_vars: list[str] | None = None,
        deweather_method: str | None = None,
        spillover_exposure: str | None = None,
        model_path: str | None = None,
        label: str = "",
    ) -> str:
        """Start a job: a synthetic control of the `treated` unit from the donor pool
        (`donors`, every other unit by default) in a panel (one row per unit and date:
        date_col 'date', unit_col 'ID', outcome_col 'value' by default; the scm example
        uses unit_col='code', outcome_col='NO2_dw'), and its effect after `cutoff`
        (YYYY-MM-DD). backend: 'scm', 'mlscm', 'abadie', 'did', 'mcnnm', 'robust',
        'ascm', 'sdid', 'deweathered', 'transport', 'chronos'. deweathered needs
        covariates (and optional resample_vars/deweather_method); transport needs
        spillover_exposure, a column carrying donor exposure F_j(t). Writes scm.csv;
        poll normet_job_status, which reads the pre-period fit and the effect."""
        return t.start_scm(
            treated,
            cutoff,
            data=data,
            example=example,
            donors=donors,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            backend=backend,
            covariates=covariates,
            resample_vars=resample_vars,
            deweather_method=deweather_method,
            spillover_exposure=spillover_exposure,
            model_path=model_path,
            label=label,
        )

    @tool
    def normet_start_normalise_ox(
        o3_col: str,
        no2_col: str,
        covariates: list[str],
        example: str | None = None,
        data: str | None = None,
        no_col: str | None = "NO",
        resample_vars: list[str] | None = None,
        unit: str | None = None,
        method: str | None = None,
        backend: str | None = None,
        n_samples: int | None = None,
        cutoff_days: float | None = None,
        seed: int | None = None,
        label: str = "",
    ) -> str:
        """Start a job: conserved Ox-NOx de-weathering. Deweathers Ox = O3 + NO2 and
        NOx = NO + NO2 in molar coordinates, then repartitions Ox with the
        meteorology-marginalised NO2 fraction, so titration is not double-counted.
        Writes ox_species.csv (observed/normalised per species); poll
        normet_job_status."""
        return t.start_normalise_ox(
            o3_col,
            no2_col,
            covariates,
            data=data,
            example=example,
            no_col=no_col,
            resample_vars=resample_vars,
            unit=unit,
            method=method,
            backend=backend,
            n_samples=n_samples,
            cutoff_days=cutoff_days,
            seed=seed,
            label=label,
        )

    @tool
    def normet_start_normalise_regime(
        target: str,
        precursors: list[str],
        covariates: list[str],
        example: str | None = None,
        data: str | None = None,
        resample_vars: list[str] | None = None,
        method: str | None = None,
        backend: str | None = None,
        n_samples: int | None = None,
        cutoff_days: float | None = None,
        seed: int | None = None,
        label: str = "",
    ) -> str:
        """Start a job: de-weather a secondary pollutant (e.g. O3) conditioned on the
        de-weathered trajectories of primary precursors (e.g. NOx, CO), holding the
        emission-driven regime fixed. Writes regime_normalised.csv; poll
        normet_job_status."""
        return t.start_normalise_regime(
            target,
            precursors,
            covariates,
            data=data,
            example=example,
            resample_vars=resample_vars,
            method=method,
            backend=backend,
            n_samples=n_samples,
            cutoff_days=cutoff_days,
            seed=seed,
            label=label,
        )

    @tool
    def normet_start_icn_decompose(
        target: str,
        met_vars: list[str],
        example: str | None = None,
        data: str | None = None,
        estimate: str = "level",
        groups: dict[str, list[str]] | None = None,
        attribution: str | None = None,
        n_permutations: int | None = None,
        interactions: bool = False,
        context_length: int | None = None,
        prediction_length: int | None = None,
        seed: int | None = None,
        model_path: str | None = None,
        label: str = "",
    ) -> str:
        """Start a job: in-context group-Shapley decomposition of observed minus
        counterfactual with Chronos-2, batched over common-random-number weather
        draws. groups maps a physical group name to its columns; interactions=True
        also writes pairwise Grabisch-Roubens indices. Writes icn_decomposition.csv
        (and icn_interactions.csv); poll normet_job_status."""
        return t.start_icn_decompose(
            target,
            met_vars,
            data=data,
            example=example,
            estimate=estimate,
            groups=groups,
            attribution=attribution,
            n_permutations=n_permutations,
            interactions=interactions,
            context_length=context_length,
            prediction_length=prediction_length,
            seed=seed,
            model_path=model_path,
            label=label,
        )

    @tool
    def normet_start_shapley_interactions(
        target: str,
        covariates: list[str],
        example: str | None = None,
        data: str | None = None,
        backend: str | None = None,
        n_samples: int | None = None,
        groups: dict[str, list[str]] | None = None,
        seed: int | None = None,
        label: str = "",
    ) -> str:
        """Start a job: exact second-order Shapley interaction indices for a trained
        AutoML model. Writes interactions.csv (main effects plus pairwise terms) and
        interactions_decomp.csv; poll normet_job_status."""
        return t.start_shapley_interactions(
            target,
            covariates,
            data=data,
            example=example,
            backend=backend,
            n_samples=n_samples,
            groups=groups,
            seed=seed,
            label=label,
        )

    @tool
    def normet_make_benchmark(
        family: str = "F1_additive",
        n_days: int | None = None,
        seed: int | None = None,
        evaluate: str | None = None,
        col: str | None = None,
        label: str = "",
    ) -> str:
        """Start a job: generate a synthetic benchmark (families F1-F5) with exact
        ground truth in benchmark/ (benchmark.csv, truth.csv, metadata.json); with
        evaluate=<normalised table> also scores it and writes benchmark/scores.json.
        Poll normet_job_status."""
        return t.make_benchmark(
            family=family, n_days=n_days, seed=seed, evaluate=evaluate, col=col, label=label
        )

    @tool
    def normet_invert_multisite(
        footprints: dict[str, str],
        series: str,
        date_col: str | None = None,
        unit_col: str | None = None,
        value_col: str | None = None,
        region_cols: list[str] | None = None,
        spatial_alpha: float | None = None,
        spatial_scale_km: float | None = None,
        n_bootstrap: int | None = None,
        seed: int | None = None,
        label: str = "",
    ) -> str:
        """Start a job: joint inversion of source footprints at several receptors.
        footprints maps site -> footprint CSV; series is a long table (with unit_col)
        or a wide table with one column per site. Writes inversion/ (regions.csv,
        site_metrics.csv, contributions.csv, summary.json); poll normet_job_status."""
        return t.invert_multisite(
            footprints,
            series,
            date_col=date_col,
            unit_col=unit_col,
            value_col=value_col,
            region_cols=region_cols,
            spatial_alpha=spatial_alpha,
            spatial_scale_km=spatial_scale_km,
            n_bootstrap=n_bootstrap,
            seed=seed,
            label=label,
        )

    @tool
    def normet_start_cv(
        target: str,
        covariates: list[str],
        example: str | None = None,
        data: str | None = None,
        n_splits: int | None = None,
        gap: int | None = None,
        backend: str | None = None,
        label: str = "",
    ) -> str:
        """Start a job: time-series cross-validation of a model of `target` on
        `covariates` -- trained on the past, scored on the next block, n_splits times
        (5), `gap` rows between them. Writes cv.csv; poll normet_job_status."""
        return t.start_cv(
            target,
            covariates,
            data=data,
            example=example,
            n_splits=n_splits,
            gap=gap,
            backend=backend,
            label=label,
        )

    @tool
    def normet_start_impute(
        target: str,
        met_vars: list[str] | None = None,
        example: str | None = None,
        data: str | None = None,
        direction: str = "bidirectional",
        max_gap: int | None = None,
        calibrate: bool = False,
        context_length: int | None = None,
        prediction_length: int | None = None,
        model_path: str | None = None,
        label: str = "",
    ) -> str:
        """Start a Chronos-2 covariate-conditioned gap imputation job. Writes
        imputed.csv; poll normet_job_status."""
        return t.start_impute(
            target,
            met_vars=met_vars,
            data=data,
            example=example,
            direction=direction,
            max_gap=max_gap,
            calibrate=calibrate,
            context_length=context_length,
            prediction_length=prediction_length,
            model_path=model_path,
            label=label,
        )

    @tool
    def normet_start_backtest(
        target: str,
        met_vars: list[str] | None = None,
        example: str | None = None,
        data: str | None = None,
        horizon: int | None = None,
        stride: int | None = None,
        calibrate: bool = False,
        calibration_frac: float | None = None,
        context_length: int | None = None,
        prediction_length: int | None = None,
        model_path: str | None = None,
        label: str = "",
    ) -> str:
        """Start a Chronos-2 rolling probabilistic backtest job (CRPS, pinball,
        PIT, empirical coverage by lead). Writes backtest.csv; poll
        normet_job_status."""
        return t.start_backtest(
            target,
            met_vars=met_vars,
            data=data,
            example=example,
            horizon=horizon,
            stride=stride,
            calibrate=calibrate,
            calibration_frac=calibration_frac,
            context_length=context_length,
            prediction_length=prediction_length,
            model_path=model_path,
            label=label,
        )

    @tool
    def normet_start_events(
        value_col: str,
        method: str = "iqr",
        covariates: list[str] | None = None,
        example: str | None = None,
        data: str | None = None,
        alpha: float | None = None,
        k: float | None = None,
        min_length: int | None = None,
        direction: str | None = None,
        adaptive: bool = False,
        context_length: int | None = None,
        prediction_length: int | None = None,
        model_path: str | None = None,
        label: str = "",
    ) -> str:
        """Start an anomaly/event detection job (method 'iqr', 'isolation',
        'stl_residual', or conditional Chronos-2 'chronos' with optional ACI
        `adaptive=True`). Writes events.csv; poll normet_job_status."""
        return t.start_events(
            value_col,
            method=method,
            covariates=covariates,
            data=data,
            example=example,
            alpha=alpha,
            k=k,
            min_length=min_length,
            direction=direction,
            adaptive=adaptive,
            context_length=context_length,
            prediction_length=prediction_length,
            model_path=model_path,
            label=label,
        )

    @tool
    def normet_job_status(job_id: str) -> str:
        """Where a job is (queued, running, done, failed, cancelled). Once done: the
        files it wrote and normet's readings of them -- for do-all, the model's fit on
        the training and held-out rows, its feature importance and the normalised
        series' means by year or month. Jobs run one at a time, in the order started."""
        return t.job_status(job_id)

    @tool
    def normet_cancel_job(job_id: str) -> str:
        """Stop a job: one waiting never starts, one running is stopped."""
        return t.cancel_job(job_id)

    @tool
    def normet_list_jobs(limit: int = 20) -> str:
        """The latest jobs, newest first: id, kind, state, label or command."""
        return t.list_jobs(limit)

    @tool
    def normet_read_run(run: str) -> str:
        """A saved do-all run (its run.joblib): what it was asked, the model's fit
        (normet.modStats on the training, held-out and all rows), its feature
        importance and the normalised series against the observed."""
        return t.read_run(run)

    @tool
    def normet_read_result(path: str, kind: str, cutoff: str | None = None) -> str:
        """A result table read as `kind` says: 'normalised' (observed and normalised
        columns), 'decomposition', 'scm' (with the `cutoff` it was run with),
        'cv', 'impute', 'backtest', or 'events'."""
        return t.read_result(path, kind, cutoff=cutoff)

    @tool
    def normet_trend(
        path: str,
        column: str | None = None,
        seasonal: int | None = None,
        n_bootstrap: int = 1000,
        seed: int = 7654321,
        name: str | None = None,
    ) -> str:
        """The trend analysis of one column of a table (a decomposition's emi_total or
        met_total, a normalised table's normalised, a raw table's target --
        `column` names it where it has to): Mann-Kendall, Theil-Sen slope with its
        interval, and the block-bootstrap-corrected interval and p where the record
        is autocorrelated. `seasonal` is the repeating calendar period in steps of
        the record (24 hourly, 7 daily, 12 monthly), else leave it out. Run once a
        table is made; it is not a job."""
        return t.trend(
            path,
            column=column,
            seasonal=seasonal,
            n_bootstrap=n_bootstrap,
            seed=seed,
            name=name,
        )

    @tool
    def normet_trend_breaks(
        path: str,
        column: str | None = None,
        segments: int = 2,
        n_bootstrap: int = 500,
        seed: int = 7654321,
        name: str | None = None,
    ) -> str:
        """Whether a column's trend breaks (normet.detect_trend_breaks): where, each
        segment's Theil-Sen slope with its interval, and the block-bootstrap p of
        the search over every candidate position -- the one to quote. `column`
        names the column; 'long-term' (the default) is the emission
        decomposition's separated long-term component (cycles already taken
        out, the cleaner input), 'emission' its emi_total. `segments` 2 to 4.
        Run once a table is made; it is not a job."""
        return t.trend_breaks(
            path,
            column=column,
            segments=segments,
            n_bootstrap=n_bootstrap,
            seed=seed,
            name=name,
        )

    def normet_figure(source: str, kind: str, cutoff: str | None = None, dpi: int = 100) -> Any:
        caption, png = t.figure(source, kind, cutoff=cutoff, dpi=dpi)
        return [caption, Image(data=png, format="png")]

    # Its kinds are FIGURES', which a literal docstring would copy.
    normet_figure.__doc__ = (
        "One of normet's figures, as an image, of a run (run.joblib) or a result table. "
        "kind: " + "; ".join(f"{k} ({v})" for k, v in FIGURES.items()) + "."
    )
    tool(normet_figure)

    @tool
    def normet_write_report(run: str, out: str | None = None) -> str:
        """The HTML report of a saved do-all run, with its provenance: written beside
        the run, or to `out`."""
        return t.write_report(run, out)

    @mcp.resource("normet://examples", name="examples", description="normet's example tables")
    def examples_resource() -> str:
        return t.list_examples()

    @mcp.resource(
        "normet://jobs/{job_id}/command",
        name="job-command",
        description="A job's normet command, which runs again as it is from its folder",
    )
    def job_command(job_id: str) -> str:
        return str(t.store.status(job_id)["command"])

    @mcp.prompt(
        name="deweather_workflow",
        description="De-weathering a pollutant's series, with the user choosing the "
        "predictors and judging the model",
    )
    def deweather_workflow(data: str = "the table given") -> str:
        return WORKFLOW.format(what=data)

    return mcp


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="normet-mcp",
        description="normet as an MCP server over stdio (Claude Desktop, Claude Code, ...). "
        "Jobs are kept in $NORMET_MCP_HOME (~/.normet/mcp); files are read and written "
        "under $NORMET_MCP_ROOTS (the home folder where unset).",
    )
    p.parse_args(argv)
    from ..agent.tools import Tools

    tools = Tools()
    tools.store.start_pumping()
    build_server(tools).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
