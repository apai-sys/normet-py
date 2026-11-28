# src/normet/cli.py
"""
Command-line interface for ``normet``.

Subcommands wrap the most common library entry points:

- ``normet do-all <input.csv> ...``
- ``normet decompose <input.csv> ...``
- ``normet deweather <input.csv> ...``
- ``normet impute <input.csv> ...``
- ``normet backtest <input.csv> ...``
- ``normet events <input.csv> ...``
- ``normet scm <panel.csv> ...``
- ``normet cv <input.csv> ...``
- ``normet report <run.joblib> ...``
- ``normet info``

The data subcommands (``do-all``, ``decompose``, ``deweather``, ``impute``,
``backtest``, ``events``, ``scm``, ``cv``) can read their options from a YAML
file via ``--config foo.yaml``; CLI flags take precedence over file values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils._lazy import require
from ..utils.logging import enable_default_logging, get_logger

log = get_logger(__name__)


# ---------- helpers ----------


def _load_table(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(p, parse_dates=True)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(p)
    raise ValueError(f"Unsupported input format: {suffix}")


def _save_table(df: pd.DataFrame, path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df.to_csv(p, index=isinstance(df.index, pd.DatetimeIndex))
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(p)
    elif suffix in {".feather", ".ft"}:
        df.reset_index().to_feather(p)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def _split_csv(s: Any) -> list[str] | None:
    """Normalise a comma-separated flag or an already-parsed YAML list.

    Both feed the same option: ``--covariates t2m,blh`` arrives as a string,
    while ``covariates: [t2m, blh]`` in a config file arrives as a list and used
    to hit ``AttributeError: 'list' object has no attribute 'split'``.
    """
    if not s:
        return None
    if isinstance(s, str):
        return [x.strip() for x in s.split(",") if x.strip()]
    return [str(x).strip() for x in s if str(x).strip()]


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    yaml = require("yaml", hint="pip install pyyaml")
    click = require("click", hint="pip install click")
    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise click.BadParameter(
                f"invalid YAML in {path}: {exc}", param_hint="--config"
            ) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise click.UsageError(f"--config file {path} must hold a YAML mapping")
    return data


def _parse_groups(spec: Any) -> dict[str, list[str]] | None:
    """Parse ``name=col1,col2;name2=col3`` into an ordered mapping of groups."""
    if not spec:
        return None
    groups: dict[str, list[str]] = {}
    for part in str(spec).split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"group must be name=col1,col2, got {part!r}")
        name, cols = part.split("=", 1)
        members = [c.strip() for c in cols.split(",") if c.strip()]
        if not name.strip() or not members:
            raise ValueError(f"group must name a group and at least one column, got {part!r}")
        groups[name.strip()] = members
    return groups or None


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    """Read a float option, falling back to ``default`` when absent or None.

    The obvious spelling, ``cfg.get(k) if cfg.get(k) is not None else d``, reads
    the dict twice; a type checker only sees the second read and so cannot tell
    the None branch is already excluded. Reading once also lets the value be
    coerced, which matters because a YAML config can hand over ``"0.75"``.
    """
    value = cfg.get(key)
    return default if value is None else float(value)


def _cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    """Read an integer option. See :func:`_cfg_float`."""
    value = cfg.get(key)
    return default if value is None else int(value)


def _merge_cli_over_yaml(yaml_cfg: Any, cli_args: dict[str, Any]) -> dict[str, Any]:
    """CLI args win; None values fall back to YAML."""
    if not isinstance(yaml_cfg, dict):
        click = require("click", hint="pip install click")
        raise click.UsageError("--config must hold a YAML mapping")
    out = dict(yaml_cfg)
    for k, v in cli_args.items():
        if v is not None:
            out[k] = v
    return out


def _require_option(cfg: dict[str, Any], key: str, flag: str) -> Any:
    """Return *cfg[key]* when set and non-empty, else raise ``click.UsageError``."""
    value = cfg.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        click = require("click", hint="pip install click")
        raise click.UsageError(f"{flag} is required")
    return value


# ---------- CLI ----------


def _build_cli() -> Any:
    click = require("click", hint="pip install click")

    from ..backends import backend_registry
    from ..pipeline import CHRONOS_BACKEND

    # The registry is the single source of truth for what --backend accepts;
    # the literal that used to sit here had gone stale and offered only flaml
    # long after the lightgbm backend was registered. Importing the registry is
    # cheap: both backend modules defer their heavy imports to `require`.
    backend_choice = click.Choice(backend_registry.available)
    # do-all and decompose also accept the zero-shot backend, which is
    # deliberately absent from the registry: that contract is train/save/load
    # and Chronos-2 does none of the three. cv has no zero-shot counterpart.
    zero_shot_backend_choice = click.Choice([*backend_registry.available, CHRONOS_BACKEND])

    @click.group()
    @click.version_option(package_name="normet")
    def cli() -> None:
        """Command-line interface for the normet modelling toolbox."""
        enable_default_logging("INFO")

    def _prepare_chronos_frame(
        df: pd.DataFrame, date_col: str, cols: list[str], cmd_name: str
    ) -> pd.DataFrame:
        from ..foundation import to_regular_index

        work = df.copy()
        if date_col in work.columns:
            work = work.set_index(pd.to_datetime(work[date_col])).drop(columns=[date_col])
        elif not isinstance(work.index, pd.DatetimeIndex):
            raise click.UsageError(
                f"no {date_col!r} column and the index is not datetime; pass --date-col"
            )
        work = work.sort_index()
        sel = list(dict.fromkeys(cols))
        missing = [c for c in sel if c not in work.columns]
        if missing:
            raise click.UsageError(f"missing required column(s): {missing}")
        before = len(work)
        work = to_regular_index(work[sel])
        if len(work) != before:
            click.echo(f"[{cmd_name}] re-inserted {len(work) - before} missing timestamps as NaN")
        return work

    def _load_chronos_estimator(cfg: dict[str, Any], met: list[str], cmd_name: str) -> Any:
        from ..foundation import Chronos2Estimator

        model_path = cfg.get("model_path")
        if model_path:
            click.echo(f"[{cmd_name}] loading saved Chronos-2 checkpoint {model_path}…")
            est = Chronos2Estimator.load(model_path, device=cfg.get("device"))
            if met:
                est.met_covariates = list(met)
            if cfg.get("context_length") is not None:
                est.context_length = int(cfg["context_length"])
            if cfg.get("prediction_length") is not None:
                est.prediction_length = int(cfg["prediction_length"])
        else:
            est = Chronos2Estimator(
                met_covariates=met,
                device=cfg.get("device"),
                context_length=cfg.get("context_length") or 2048,
                prediction_length=cfg.get("prediction_length") or 168,
            )
            click.echo(f"[{cmd_name}] loading {est.model_name} (first run downloads ~500 MB)…")
        est._load_pipeline()
        click.echo(f"[{cmd_name}] ready on {est.device}")
        return est

    # ---- do-all ----
    @cli.command("do-all")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--target", help="Target column name (e.g., PM2.5).")
    @click.option("--covariates", help="Comma-separated predictor columns.")
    @click.option(
        "--resample-vars", "resample_vars", help="Comma-separated subset of features to resample."
    )
    @click.option("--backend", type=zero_shot_backend_choice, default=None)
    @click.option("--n-samples", "n_samples", type=int, default=None)
    @click.option(
        "--split-method",
        "split_method",
        type=click.Choice(["random", "ts", "month_ts", "season_ts"]),
        default=None,
    )
    @click.option("--train-fraction", "train_fraction", type=float, default=None)
    @click.option("--seed", type=int, default=None)
    @click.option(
        "--normalise-method",
        "normalise_method",
        type=click.Choice(["standard", "dml"]),
        default=None,
        help="Deweathering engine: standard Monte-Carlo normalise, or the "
        "orthogonal Double-ML estimator (blocked out-of-fold).",
    )
    @click.option(
        "--control-variates",
        "control_variates",
        is_flag=True,
        default=None,
        help="Use Monte-Carlo control variates (variance reduction).",
    )
    @click.option(
        "--block-mode",
        "block_mode",
        type=click.Choice(["independent", "markov"]),
        default=None,
        help="Block resampling: independent blocks or a Markov bridge.",
    )
    @click.option("--block-hours", "block_hours", type=int, default=None)
    @click.option("--dml-cutoff-days", "dml_cutoff_days", type=float, default=None)
    @click.option("--dml-n-harmonics", "dml_n_harmonics", type=int, default=None)
    @click.option(
        "--out",
        "out_path",
        type=click.Path(path_type=Path),
        required=True,
        help="Output table for the normalised series.",
    )
    @click.option(
        "--run-path",
        "run_path",
        type=click.Path(path_type=Path),
        default=None,
        help="Optional. If set, also save a NormetRun archive (.joblib + .meta.json).",
    )
    @click.option(
        "--config",
        "config_path",
        type=click.Path(exists=True, path_type=Path),
        help="YAML config that supplies any of the above options.",
    )
    def do_all_cmd(input: Path, **opts: Any) -> None:
        from dataclasses import fields

        from .. import do_all, make_run, save_run
        from ..pipeline.do_all import SingleConfig

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        target = _require_option(cfg, "target", "--target")
        if cfg.get("block_mode") == "markov" and not (
            cfg.get("block_hours") is not None and int(cfg["block_hours"]) > 1
        ):
            # The library treats markov with no usable block length as
            # independent draws, which made --block-mode markov a silent no-op.
            raise click.UsageError("--block-mode markov needs --block-hours > 1")
        df = _load_table(input)

        # Every SingleConfig field the config file names reaches do_all, not
        # only the handful the CLI had flags for. The YAML route used to look
        # generic while the wrapper silently dropped sampling, block_hours,
        # paired, cross_fit and model_config on the floor.
        explicit = {
            "target",
            "backend",
            "covariates",
            "variables_resample",
            "split_method",
            "train_fraction",
            "n_samples",
            "seed",
            "verbose",
        }
        known = {f.name for f in fields(SingleConfig)}
        forwarded = {
            k: v for k, v in cfg.items() if k in known and k not in explicit and v is not None
        }

        out, model, df_prep = do_all(
            df=df,
            target=target,
            covariates=_split_csv(cfg.get("covariates")),
            variables_resample=_split_csv(cfg.get("resample_vars"))
            or _split_csv(cfg.get("variables_resample")),
            backend=cfg.get("backend") or "flaml",
            n_samples=int(cfg["n_samples"]) if cfg.get("n_samples") is not None else None,
            split_method=cfg.get("split_method") or "random",
            train_fraction=_cfg_float(cfg, "train_fraction", 0.75),
            seed=_cfg_int(cfg, "seed", 7_654_321),
            verbose=True,
            **forwarded,
        )
        _save_table(out.reset_index(), Path(cfg["out_path"]))
        click.echo(f"[do-all] wrote {cfg['out_path']}")

        if cfg.get("run_path"):
            run = make_run(
                result=out,
                model=model,
                df_prep=df_prep,
                df=df,
                kind="do_all",
                config=cfg,
                seed=cfg.get("seed"),
            )
            paths = save_run(run, cfg["run_path"])
            click.echo(f"[do-all] archived run → {paths['artifact']}")

    # ---- decompose ----
    @cli.command("decompose")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--target", required=False)
    @click.option("--covariates", help="Comma-separated predictor columns.")
    @click.option("--method", type=click.Choice(["emission", "meteorology"]), default=None)
    @click.option("--backend", type=zero_shot_backend_choice, default=None)
    @click.option("--n-samples", "n_samples", type=int, default=None)
    @click.option("--seed", type=int, default=None)
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def decompose_cmd(input: Path, **opts: Any) -> None:
        from .. import decompose

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        target = _require_option(cfg, "target", "--target")
        df = _load_table(input)

        out = decompose(
            method=cfg.get("method") or "emission",
            df=df,
            target=target,
            covariates=_split_csv(cfg.get("covariates")),
            backend=cfg.get("backend") or "flaml",
            n_samples=int(cfg["n_samples"]) if cfg.get("n_samples") is not None else None,
            seed=_cfg_int(cfg, "seed", 7_654_321),
            verbose=True,
        )
        _save_table(out.reset_index(), Path(cfg["out_path"]))
        click.echo(f"[decompose] wrote {cfg['out_path']}")

    # ---- deweather (zero-shot, Chronos-2) ----
    @cli.command("deweather")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--target", required=False, help="Target column name (e.g. PM2.5).")
    @click.option(
        "--met-vars",
        "met_vars",
        help="Comma-separated meteorological columns to condition on and marginalise over.",
    )
    @click.option("--date-col", "date_col", default=None, help="Timestamp column. Default: date.")
    @click.option(
        "--n-samples",
        "n_samples",
        type=int,
        default=None,
        help="Monte-Carlo weather resamples. Each one is a full forward pass. Default: 8.",
    )
    @click.option(
        "--quantiles",
        default=None,
        help="Comma-separated levels, e.g. 0.1,0.5,0.9. Default: 0.1,0.5,0.9.",
    )
    @click.option(
        "--device",
        default=None,
        help="torch device: cpu, cuda, mps. Default: cuda, else mps, else cpu.",
    )
    @click.option("--context-length", "context_length", type=int, default=None)
    @click.option("--prediction-length", "prediction_length", type=int, default=None)
    @click.option(
        "--model-path",
        "model_path",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Path to a saved Chronos-2 checkpoint or LoRA adapter directory.",
    )
    @click.option(
        "--method",
        type=click.Choice(["deweather", "icn"]),
        default=None,
        help="Normalisation mode: legacy past+future 'deweather' (default) or future-only 'icn'.",
    )
    @click.option(
        "--estimate",
        type=click.Choice(["level", "paired"]),
        default=None,
        help="For --method icn: output counterfactual 'level' (default) or event-preserving 'paired'.",
    )
    @click.option(
        "--log-transform",
        "log_transform",
        is_flag=True,
        default=False,
        help="Run --method icn in log1p space (additive pairing, non-negative).",
    )
    @click.option(
        "--surprise-tau",
        "surprise_tau",
        type=float,
        default=None,
        help="For --method icn: gate anchor draws by exp(-(err/tau)^2).",
    )
    @click.option(
        "--save-bank",
        "save_bank",
        type=click.Path(path_type=Path),
        default=None,
        help="For --method icn: persist the raw ICN draw bank for offline re-aggregation.",
    )
    @click.option(
        "--bank",
        "bank_path",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Re-aggregate a saved ICN draw bank instead of running the model.",
    )
    @click.option("--seed", type=int, default=None)
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def deweather_cmd(input: Path, **opts: Any) -> None:
        """Zero-shot meteorological normalisation with Chronos-2.

        Nothing is trained: the checkpoint conditions on the meteorology through
        its covariate channel and the weather is marginalised out by resampling,
        as in ``normalise`` (or via future-only ``--method icn``). Needs the
        ``foundation`` extra.
        """
        import numpy as np

        from ..foundation import ICNDesign, add_calendar_covariates

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        df = _load_table(input)

        method = cfg.get("method") or "deweather"
        estimate = cfg.get("estimate") or "level"
        log_transform = bool(cfg.get("log_transform"))
        if method not in ("deweather", "icn"):
            raise click.BadParameter(f"method must be 'deweather' or 'icn', got {method!r}")
        if estimate not in ("level", "paired"):
            raise click.BadParameter(f"estimate must be 'level' or 'paired', got {estimate!r}")
        if log_transform and method != "icn":
            raise click.BadParameter("--log-transform requires --method icn")
        surprise_tau = cfg.get("surprise_tau")
        save_bank = cfg.get("save_bank")
        bank_path = cfg.get("bank_path")
        if (surprise_tau, save_bank, bank_path) != (None, None, None) and method != "icn":
            raise click.BadParameter("--surprise-tau/--save-bank/--bank require --method icn")
        if save_bank and bank_path:
            raise click.BadParameter("--save-bank and --bank are mutually exclusive")
        if bank_path:
            from ..foundation import ICNDrawBank

            bank = ICNDrawBank.load(bank_path)
            # The bank stores draws, not an aggregate: a supplied tau must
            # re-weight them here, otherwise --surprise-tau was silently
            # dropped on the re-aggregation path.
            overrides: dict[str, Any] = {"schema": "normet"}
            if surprise_tau is not None:
                overrides["surprise_tau"] = float(surprise_tau)
            out = bank.aggregate(**overrides).to_normet_frame(estimate=estimate)
            _save_table(out.reset_index(), Path(cfg["out_path"]))
            click.echo(
                f"[deweather] re-aggregated bank {bank_path} ({estimate}); wrote {cfg['out_path']}"
            )
            return
        target = _require_option(cfg, "target", "--target")
        met = _split_csv(cfg.get("met_vars"))
        if not met:
            raise click.UsageError(
                "--met-vars is required: Chronos-2 conditions on meteorology through its "
                "covariate channel, and with none supplied the run is a plain forecast "
                "rather than a de-weathering."
            )
        date_col = cfg.get("date_col") or "date"
        df = _prepare_chronos_frame(df, date_col, [target, *met], "deweather")

        quantiles = (
            tuple(float(x) for x in _split_csv(cfg.get("quantiles")) or [])
            if cfg.get("quantiles")
            else (0.1, 0.5, 0.9)
        )
        est = _load_chronos_estimator(cfg, met, "deweather")

        seed = _cfg_int(cfg, "seed", 7_654_321)
        horizon = est.prediction_length
        if len(df) > est.context_length + horizon:
            shift = est.covariate_sensitivity(
                df,
                target,
                anchor=df.index[-horizon],
                horizon=horizon,
                met_features=met,
                random_state=seed,
            )
            # The honest quality gate: if shuffling the future weather barely
            # moves the forecast, the model is autoregressing and the
            # "normalised" series below means nothing.
            click.echo(
                f"[deweather] covariate sensitivity: shuffling the meteorology moves the "
                f"median by {shift['mean_abs_shift']:.3f} "
                f"({shift['pct_of_prediction']:.2f}% of the prediction)"
            )
            if shift["pct_of_prediction"] < 1.0:
                click.echo(
                    "[deweather] WARNING: no meaningful meteorological response — the model "
                    "is autoregressing. Check the --met-vars columns before using this output.",
                    err=True,
                )
        else:
            click.echo(
                f"[deweather] WARNING: only {len(df)} rows; need more than "
                f"{est.context_length + horizon} to run the sensitivity check",
                err=True,
            )

        if method == "icn":
            df_icn = add_calendar_covariates(df)
            design_overrides: dict[str, Any] = {
                "context_length": est.context_length,
                "seed": seed,
            }
            if cfg.get("n_samples") is not None:
                design_overrides["n_resamples"] = int(cfg["n_samples"])
            if cfg.get("prediction_length") is not None:
                bh = int(cfg["prediction_length"])
                design_overrides["block_h"] = bh
                design_overrides["stride"] = max(1, bh // 2)
            if log_transform:
                design_overrides["log_transform"] = True
            if surprise_tau is not None:
                design_overrides["surprise_tau"] = float(surprise_tau)
            design = ICNDesign.v1(**design_overrides)
            if save_bank:
                from ..foundation import build_icn_draw_bank

                bank = build_icn_draw_bank(
                    est,
                    df_icn,
                    target,
                    design=design,
                    met_features=met,
                    schema="normet",
                    quantiles=quantiles,
                )
                bank.save(Path(save_bank))
                click.echo(f"[deweather] saved ICN draw bank → {save_bank}")
                icn_res = bank.aggregate(schema="normet")
            else:
                icn_res = est.normalise_in_context(
                    df_icn,
                    target,
                    design=design,
                    met_features=met,
                    quantiles=quantiles,
                )
            out = icn_res.to_normet_frame(estimate=estimate)
            uncovered = int(icn_res.diagnostics["uncovered_hours"])
            click.echo(
                f"[deweather] ICN-v1 ({estimate}): {len(out):,} rows, {uncovered:,} uncovered hours"
            )
            tail = out.dropna(subset=["normalised", "observed"])
            if len(tail):
                delta = float(np.mean(tail["normalised"] - tail["observed"]))
                click.echo(f"[deweather] mean(normalised - observed) = {delta:+.3f}")
        else:
            out = est.deweather(
                df,
                target,
                met_features=met,
                n_samples=cfg.get("n_samples") or 8,
                quantiles=quantiles,
                random_state=seed,
                schema="normet",
            )
            out.index.name = "date"
            # The leading context_length rows are seeded with the observed values --
            # Chronos-2 has no history to condition on until then, so the two series
            # coincide there by construction rather than by result.
            seeded = min(est.context_length, len(out))
            click.echo(
                f"[deweather] first {seeded:,} of {len(out):,} rows are seeded with the observed "
                f"values ({100.0 * seeded / len(out):.0f}%); the normalisation starts after them"
            )
            tail = out.iloc[seeded:]
            if len(tail):
                delta = float(np.mean(tail["normalised"] - tail["observed"]))
                click.echo(f"[deweather] mean(normalised - observed) over the rest = {delta:+.3f}")
        _save_table(out.reset_index(), Path(cfg["out_path"]))
        click.echo(f"[deweather] wrote {cfg['out_path']}")

    # ---- icn-decompose (in-context group Shapley) ----
    @cli.command("icn-decompose")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--target", required=False, help="Target column name.")
    @click.option("--met-vars", "met_vars", help="Comma-separated meteorological columns.")
    @click.option("--date-col", "date_col", default=None)
    @click.option("--device", default=None)
    @click.option("--context-length", "context_length", type=int, default=None)
    @click.option("--prediction-length", "prediction_length", type=int, default=None)
    @click.option(
        "--model-path",
        "model_path",
        type=click.Path(exists=True, path_type=Path),
        default=None,
    )
    @click.option("--estimate", type=click.Choice(["level", "paired"]), default=None)
    @click.option(
        "--groups",
        default=None,
        help="Groups as name=col1,col2;name2=col3,... (default: one player per met variable).",
    )
    @click.option(
        "--attribution",
        type=click.Choice(["sequential", "shapley"]),
        default=None,
    )
    @click.option("--n-permutations", "n_permutations", type=int, default=None)
    @click.option("--interactions", is_flag=True, default=None)
    @click.option("--include-diagnostics", "include_diagnostics", is_flag=True, default=None)
    @click.option("--seed", type=int, default=None)
    @click.option(
        "--interactions-out", "interactions_out", type=click.Path(path_type=Path), default=None
    )
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def icn_decompose_cmd(input: Path, **opts: Any) -> None:
        """In-context meteorological attribution (``icn_decompose``)."""
        from ..foundation import ICNDesign, add_calendar_covariates, icn_decompose

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        target = _require_option(cfg, "target", "--target")
        met = _split_csv(cfg.get("met_vars"))
        if not met:
            raise click.UsageError("--met-vars is required.")
        df = _load_table(input)
        date_col = cfg.get("date_col") or "date"
        df = _prepare_chronos_frame(df, date_col, [target, *met], "icn-decompose")
        est = _load_chronos_estimator(cfg, met, "icn-decompose")
        design_kw: dict[str, Any] = {
            "context_length": est.context_length,
            "seed": _cfg_int(cfg, "seed", 0),
        }
        if cfg.get("prediction_length") is not None:
            bh = int(cfg["prediction_length"])
            design_kw["block_h"] = bh
            design_kw["stride"] = max(1, bh // 2)
        design = ICNDesign.v1(**design_kw)
        res = icn_decompose(
            est,
            add_calendar_covariates(df),
            target,
            design=design,
            met_features=met,
            groups=_parse_groups(cfg.get("groups")),
            attribution=cfg.get("attribution"),
            n_permutations=cfg.get("n_permutations"),
            estimate=cfg.get("estimate") or "level",
            include_diagnostics=bool(cfg.get("include_diagnostics")),
            return_interactions=bool(cfg.get("interactions")),
        )
        out_path = Path(cfg["out_path"])
        if cfg.get("interactions"):
            _save_table(res.decomposition.reset_index(), out_path)
            summary_path = (
                Path(cfg["interactions_out"])
                if cfg.get("interactions_out")
                else out_path.with_name(f"{out_path.stem}_interactions.csv")
            )
            _save_table(res.summary, summary_path)
            click.echo(f"[icn-decompose] wrote {out_path} and interactions {summary_path}")
        else:
            _save_table(res.reset_index(), out_path)
            click.echo(f"[icn-decompose] wrote {out_path}")

    # ---- shapley-interactions (second-order Shapley indices) ----
    @cli.command("shapley-interactions")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--target", required=False)
    @click.option("--covariates", help="Comma-separated meteorological/calendar covariates.")
    @click.option("--backend", default=None)
    @click.option("--n-samples", "n_samples", type=int, default=None)
    @click.option("--groups", default=None, help="Groups as name=col1,col2;name2=col3,...")
    @click.option("--seed", type=int, default=None)
    @click.option(
        "--decomposition-out", "decomposition_out", type=click.Path(path_type=Path), default=None
    )
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def shapley_interactions_cmd(input: Path, **opts: Any) -> None:
        """Main effects plus pairwise Grabisch-Roubens Shapley interactions."""
        from .. import shapley_interactions

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        target = _require_option(cfg, "target", "--target")
        df = _load_table(input)
        res = shapley_interactions(
            df,
            target=target,
            covariates=_split_csv(cfg.get("covariates")),
            backend=cfg.get("backend") or "flaml",
            n_samples=int(cfg["n_samples"]) if cfg.get("n_samples") is not None else None,
            groups=_parse_groups(cfg.get("groups")),
            seed=_cfg_int(cfg, "seed", 7_654_321),
            verbose=True,
        )
        _save_table(res.summary, Path(cfg["out_path"]))
        if cfg.get("decomposition_out"):
            _save_table(res.decomposition.reset_index(), Path(cfg["decomposition_out"]))
        click.echo(f"[shapley-interactions] wrote {cfg['out_path']}")

    # ---- benchmark (synthetic ground truth) ----
    @cli.command("benchmark")
    @click.option(
        "--family",
        type=click.Choice(
            ["F1_additive", "F2_multiplicative", "F3_titration", "F4_nonstationary", "F5_transport"]
        ),
        default=None,
    )
    @click.option("--n-days", "n_days", type=int, default=None)
    @click.option("--freq", default=None)
    @click.option("--start-date", "start_date", default=None)
    @click.option("--event-start-day", "event_start_day", type=int, default=None)
    @click.option("--event-duration-days", "event_duration_days", type=int, default=None)
    @click.option("--event-reduction-pct", "event_reduction_pct", type=float, default=None)
    @click.option("--trend-pct-per-year", "trend_pct_per_year", type=float, default=None)
    @click.option("--noise-sd", "noise_sd", type=float, default=None)
    @click.option("--seed", type=int, default=None)
    @click.option(
        "--evaluate",
        "evaluate_path",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Score this normalised table against the generated truth.",
    )
    @click.option("--col", default=None, help="Normalised column to score (default normalised).")
    @click.option(
        "--out",
        "out_dir",
        type=click.Path(path_type=Path),
        required=True,
        help="Output directory (benchmark.csv, truth.csv, metadata.json, scores.json).",
    )
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def benchmark_cmd(**opts: Any) -> None:
        """Generate a synthetic benchmark and optionally score a normalised series."""
        from ..eval.benchmarks import evaluate_benchmark, make_synthetic_benchmark

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        bench = make_synthetic_benchmark(
            cfg.get("family") or "F1_additive",
            n_days=_cfg_int(cfg, "n_days", 180),
            freq=cfg.get("freq") or "h",
            start_date=cfg.get("start_date") or "2023-01-01",
            event_start_day=cfg.get("event_start_day"),
            event_duration_days=cfg.get("event_duration_days"),
            event_reduction_pct=_cfg_float(cfg, "event_reduction_pct", -30.0),
            trend_pct_per_year=_cfg_float(cfg, "trend_pct_per_year", -5.0),
            noise_sd=_cfg_float(cfg, "noise_sd", 1.5),
            seed=_cfg_int(cfg, "seed", 42),
        )
        out_dir = Path(cfg["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_df = bench.df.reset_index()
        if "date" not in bench_df.columns:
            bench_df = bench_df.rename(columns={bench_df.columns[0]: "date"})
        _save_table(bench_df, out_dir / "benchmark.csv")
        truth_df = bench.truth.reset_index()
        if "date" not in truth_df.columns:
            truth_df = truth_df.rename(columns={truth_df.columns[0]: "date"})
        _save_table(truth_df, out_dir / "truth.csv")
        (out_dir / "metadata.json").write_text(json.dumps(bench.metadata, indent=2, default=str))
        click.echo(f"[benchmark] wrote {out_dir / 'benchmark.csv'} (family {bench.family})")
        if cfg.get("evaluate_path"):
            result = _load_table(Path(cfg["evaluate_path"]))
            scores = evaluate_benchmark(bench, result, col=cfg.get("col") or "normalised")
            (out_dir / "scores.json").write_text(json.dumps(scores, indent=2, default=str))
            click.echo(f"[benchmark] scores → {out_dir / 'scores.json'}")

    # ---- invert-footprints (multi-receptor inversion) ----
    @cli.command("invert-footprints")
    @click.option(
        "--footprint",
        "footprints",
        multiple=True,
        help="SITE=PATH footprint table (repeat once per receptor).",
    )
    @click.option(
        "--series",
        "series_path",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="Long table (date, unit, value) or wide table (date index, one column per site).",
    )
    @click.option("--date-col", "date_col", default=None)
    @click.option("--unit-col", "unit_col", default=None)
    @click.option("--value-col", "value_col", default=None)
    @click.option("--region-cols", "region_cols", default=None)
    @click.option("--spatial-alpha", "spatial_alpha", type=float, default=None)
    @click.option("--spatial-scale-km", "spatial_scale_km", type=float, default=None)
    @click.option("--n-bootstrap", "n_bootstrap", type=int, default=None)
    @click.option("--seed", type=int, default=None)
    @click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True)
    def invert_footprints_cmd(**opts: Any) -> None:
        """Joint multi-receptor footprint inversion (``invert_multisite_footprints``)."""
        from .. import invert_multisite_footprints

        cfg = _merge_cli_over_yaml({}, opts)
        if not cfg.get("footprints"):
            raise click.UsageError("give at least one --footprint SITE=PATH")
        footprints: dict[str, pd.DataFrame] = {}
        for spec in cfg["footprints"]:
            if "=" not in spec:
                raise click.BadParameter(f"--footprint must be SITE=PATH, got {spec!r}")
            site, path = spec.split("=", 1)
            footprints[site.strip()] = _load_table(Path(path.strip()))
        series = _load_table(Path(cfg["series_path"]))
        date_col = cfg.get("date_col") or "date"
        if date_col in series.columns:
            # ``_load_table`` reads CSVs with ``parse_dates=True``, which only
            # parses the index: a named date column stays a string, and the wide
            # path below indexes on it, so the estimator received a string index
            # and raised "must have a DatetimeIndex". Coerce it here.
            series[date_col] = pd.to_datetime(series[date_col])
        unit_col = cfg.get("unit_col")
        series_by_site: dict[str, pd.Series | pd.DataFrame] = {}
        if unit_col and unit_col in series.columns:
            for site, group in series.groupby(unit_col):
                series_by_site[str(site)] = group
        else:
            wide = series.set_index(date_col) if date_col in series.columns else series
            for site in footprints:
                if site in wide.columns:
                    series_by_site[site] = wide[site]
        if not series_by_site:
            raise click.UsageError(
                "no site matched the series table: pass --unit-col for a long table, "
                "or name the wide columns as the --footprint site keys"
            )
        res = invert_multisite_footprints(
            footprints,
            series_by_site,
            value_col=cfg.get("value_col"),
            date_col=date_col,
            region_cols=_split_csv(cfg.get("region_cols")),
            spatial_alpha=_cfg_float(cfg, "spatial_alpha", 0.0),
            spatial_scale_km=_cfg_float(cfg, "spatial_scale_km", 100.0),
            n_bootstrap=_cfg_int(cfg, "n_bootstrap", 200),
            seed=_cfg_int(cfg, "seed", 7_654_321),
        )
        out_dir = Path(cfg["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_table(res.regions, out_dir / "regions.csv")
        _save_table(res.site_metrics, out_dir / "site_metrics.csv")
        _save_table(res.contributions, out_dir / "contributions.csv")
        summary = {
            "alpha": res.alpha,
            "spatial_alpha": res.spatial_alpha,
            "r2": res.r2,
            "rmse": res.rmse,
            "n_obs": res.n_obs,
            "intercepts": res.intercepts,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        click.echo(f"[invert-footprints] wrote {out_dir}")

    # ---- normalise-ox (conserved Ox-NOx deweathering) ----
    @cli.command("normalise-ox")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--o3-col", "o3_col", default="O3", help="Ozone column.")
    @click.option("--no2-col", "no2_col", default="NO2", help="Nitrogen-dioxide column.")
    @click.option(
        "--no-col",
        "no_col",
        default="NO",
        help="Nitric-oxide column; pass 'none' when the table has no NO leg.",
    )
    @click.option("--covariates", help="Comma-separated meteorological/calendar covariates.")
    @click.option("--resample-vars", "resample_vars", help="Comma-separated variables to resample.")
    @click.option("--unit", type=click.Choice(["ug_m3", "ppb"]), default=None)
    @click.option(
        "--method",
        type=click.Choice(["dml", "crossfit", "standard"]),
        default=None,
        help="Deweathering engine (default dml).",
    )
    @click.option("--backend", default=None, help="Regression backend (default lightgbm).")
    @click.option("--cutoff-days", "cutoff_days", type=float, default=None)
    @click.option("--n-splits", "n_splits", type=int, default=None)
    @click.option("--n-samples", "n_samples", type=int, default=None)
    @click.option("--log-transform", "log_transform", is_flag=True, default=None)
    @click.option(
        "--unpaired", is_flag=True, default=None, help="Use the level series instead of paired."
    )
    @click.option("--seed", type=int, default=None)
    @click.option("--ox-out", "ox_out", type=click.Path(path_type=Path), default=None)
    @click.option("--nox-out", "nox_out", type=click.Path(path_type=Path), default=None)
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def normalise_ox_cmd(input: Path, **opts: Any) -> None:
        """Conserved Ox-NOx photochemical deweathering (``normalise_ox_invariant``)."""
        from .. import normalise_ox_invariant

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        df = _load_table(input)
        no_col_raw = cfg.get("no_col")
        # "none" (or an empty value) means the table has no NO leg; without
        # this the Click default "NO" was used and the run failed on a column
        # the user never had.
        no_col = None if str(no_col_raw or "").strip().lower() in ("", "none") else str(no_col_raw)
        res = normalise_ox_invariant(
            df,
            o3_col=cfg.get("o3_col") or "O3",
            no2_col=cfg.get("no2_col") or "NO2",
            no_col=no_col,
            covariates=_split_csv(cfg.get("covariates")) or [],
            variables_resample=_split_csv(cfg.get("resample_vars")),
            unit=cfg.get("unit") or "ug_m3",
            method=cfg.get("method") or "dml",
            backend=cfg.get("backend") or "lightgbm",
            cutoff_days=_cfg_float(cfg, "cutoff_days", 30.0),
            n_splits=_cfg_int(cfg, "n_splits", 5),
            n_samples=_cfg_int(cfg, "n_samples", 300),
            log_transform=bool(cfg.get("log_transform")),
            use_paired=not bool(cfg.get("unpaired")),
            seed=_cfg_int(cfg, "seed", 7_654_321),
            verbose=True,
        )
        _save_table(res.species.reset_index(), Path(cfg["out_path"]))
        if cfg.get("ox_out"):
            _save_table(res.ox_result.reset_index(), Path(cfg["ox_out"]))
        if cfg.get("nox_out"):
            _save_table(res.nox_result.reset_index(), Path(cfg["nox_out"]))
        click.echo(
            f"[normalise-ox] wrote {cfg['out_path']} "
            f"(max conservation error {res.conservation_max_err:.3g} mol m^-3)"
        )

    # ---- normalise-regime (precursor-conditioned deweathering) ----
    @cli.command("normalise-regime")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--target", required=False, help="Secondary pollutant column.")
    @click.option("--precursors", help="Comma-separated primary tracer columns.")
    @click.option("--covariates", help="Comma-separated meteorological/calendar covariates.")
    @click.option("--resample-vars", "resample_vars", help="Comma-separated variables to resample.")
    @click.option(
        "--method",
        type=click.Choice(["dml", "crossfit", "standard"]),
        default=None,
        help="Deweathering engine (default dml).",
    )
    @click.option("--backend", default=None, help="Regression backend (default lightgbm).")
    @click.option("--cutoff-days", "cutoff_days", type=float, default=None)
    @click.option("--n-splits", "n_splits", type=int, default=None)
    @click.option("--n-samples", "n_samples", type=int, default=None)
    @click.option("--log-transform", "log_transform", is_flag=True, default=None)
    @click.option("--unpaired", is_flag=True, default=None)
    @click.option("--seed", type=int, default=None)
    @click.option(
        "--precursors-out",
        "precursors_out",
        type=click.Path(path_type=Path),
        default=None,
    )
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def normalise_regime_cmd(input: Path, **opts: Any) -> None:
        """Deweather a secondary pollutant conditioned on deweathered precursors."""
        from .. import normalise_regime_conditioned

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        target = _require_option(cfg, "target", "--target")
        precursors = _split_csv(cfg.get("precursors"))
        if not precursors:
            raise click.UsageError(
                "--precursors is required (the primary tracers defining regime)."
            )
        df = _load_table(input)
        res = normalise_regime_conditioned(
            df,
            target,
            precursors,
            covariates=_split_csv(cfg.get("covariates")) or [],
            variables_resample=_split_csv(cfg.get("resample_vars")),
            method=cfg.get("method") or "dml",
            backend=cfg.get("backend") or "lightgbm",
            cutoff_days=_cfg_float(cfg, "cutoff_days", 30.0),
            n_splits=_cfg_int(cfg, "n_splits", 5),
            n_samples=_cfg_int(cfg, "n_samples", 300),
            log_transform=bool(cfg.get("log_transform")),
            use_paired=not bool(cfg.get("unpaired")),
            seed=_cfg_int(cfg, "seed", 7_654_321),
            verbose=True,
        )
        _save_table(res.normalised.reset_index(), Path(cfg["out_path"]))
        if cfg.get("precursors_out"):
            _save_table(res.precursors.reset_index(), Path(cfg["precursors_out"]))
        click.echo(f"[normalise-regime] wrote {cfg['out_path']}")

    # ---- impute (Chronos-2 gap filling) ----
    @cli.command("impute")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--target", required=False, help="Target column with gaps to impute.")
    @click.option(
        "--met-vars",
        "met_vars",
        default=None,
        help="Optional comma-separated covariate columns covering the gap.",
    )
    @click.option("--date-col", "date_col", default=None, help="Timestamp column. Default: date.")
    @click.option(
        "--direction",
        type=click.Choice(["bidirectional", "forward", "backward"]),
        default=None,
        help="Rollout direction across each gap. Default: bidirectional.",
    )
    @click.option(
        "--max-gap", "max_gap", type=int, default=None, help="Maximum gap length to fill."
    )
    @click.option(
        "--quantiles",
        default=None,
        help="Comma-separated lower,median,upper levels. Default: 0.1,0.5,0.9.",
    )
    @click.option(
        "--calibrate",
        is_flag=True,
        default=None,
        help="Apply split-conformal quantile calibration before emitting intervals.",
    )
    @click.option("--device", default=None, help="torch device: cpu, cuda, mps.")
    @click.option("--context-length", "context_length", type=int, default=None)
    @click.option("--prediction-length", "prediction_length", type=int, default=None)
    @click.option(
        "--model-path",
        "model_path",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Path to a saved Chronos-2 checkpoint or LoRA adapter directory.",
    )
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def impute_cmd(input: Path, **opts: Any) -> None:
        """Covariate-conditioned gap imputation with Chronos-2."""
        from ..foundation import calibrate_quantiles

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        target = _require_option(cfg, "target", "--target")
        df = _load_table(input)
        met = _split_csv(cfg.get("met_vars")) or []
        date_col = cfg.get("date_col") or "date"
        df_reg = _prepare_chronos_frame(df, date_col, [target, *met], "impute")
        direction = cfg.get("direction") or "bidirectional"
        if direction not in ("bidirectional", "forward", "backward"):
            raise click.BadParameter(
                f"direction must be 'bidirectional', 'forward', or 'backward', got {direction!r}"
            )
        quantiles = (
            tuple(float(x) for x in _split_csv(cfg.get("quantiles")) or [])
            if cfg.get("quantiles")
            else (0.1, 0.5, 0.9)
        )
        max_gap = int(cfg["max_gap"]) if cfg.get("max_gap") is not None else None
        est = _load_chronos_estimator(cfg, met, "impute")
        cal = None
        if cfg.get("calibrate"):
            cal = calibrate_quantiles(
                est,
                df_reg,
                target,
                df_reg.index[-1],
                quantiles=quantiles,
                covariates=met or None,
            )
        out = est.impute(
            df_reg,
            target,
            covariates=met,
            direction=direction,
            quantiles=quantiles,
            max_gap=max_gap,
            calibration=cal,
        )
        out.index.name = "date"
        _save_table(out.reset_index(), Path(cfg["out_path"]))
        n_filled = int(out["is_imputed"].sum())
        click.echo(f"[impute] filled {n_filled:,} rows → {cfg['out_path']}")

    # ---- backtest (Chronos-2 rolling probabilistic verification) ----
    @cli.command("backtest")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--target", required=False, help="Target column to evaluate.")
    @click.option(
        "--met-vars",
        "met_vars",
        default=None,
        help="Optional comma-separated covariate columns.",
    )
    @click.option("--date-col", "date_col", default=None, help="Timestamp column. Default: date.")
    @click.option("--horizon", type=int, default=None, help="Forecast horizon per anchor.")
    @click.option("--stride", type=int, default=None, help="Step between rolling anchors.")
    @click.option(
        "--calibrate",
        is_flag=True,
        default=None,
        help="Fit split-conformal calibration on the prefix and score the holdout.",
    )
    @click.option(
        "--calibration-frac",
        "calibration_frac",
        type=float,
        default=None,
        help="Prefix fraction reserved for calibration when --calibrate is set. Default: 0.5.",
    )
    @click.option(
        "--predictions-out",
        "predictions_out",
        type=click.Path(path_type=Path),
        default=None,
        help="Optional path to also write per-timestamp rolling predictions.",
    )
    @click.option("--device", default=None, help="torch device: cpu, cuda, mps.")
    @click.option("--context-length", "context_length", type=int, default=None)
    @click.option("--prediction-length", "prediction_length", type=int, default=None)
    @click.option(
        "--model-path",
        "model_path",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Path to a saved Chronos-2 checkpoint or LoRA adapter directory.",
    )
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def backtest_cmd(input: Path, **opts: Any) -> None:
        """Rolling multi-horizon probabilistic verification with Chronos-2."""
        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        target = _require_option(cfg, "target", "--target")
        df = _load_table(input)
        met = _split_csv(cfg.get("met_vars")) or []
        date_col = cfg.get("date_col") or "date"
        df_reg = _prepare_chronos_frame(df, date_col, [target, *met], "backtest")
        est = _load_chronos_estimator(cfg, met, "backtest")
        horizon = int(cfg["horizon"]) if cfg.get("horizon") is not None else None
        stride = int(cfg["stride"]) if cfg.get("stride") is not None else None
        cal_frac = _cfg_float(cfg, "calibration_frac", 0.5)
        res = est.backtest(
            df_reg,
            target,
            covariates=met,
            horizon=horizon,
            stride=stride,
            calibrate=bool(cfg.get("calibrate")),
            calibration_frac=cal_frac,
        )
        _save_table(res["by_lead"], Path(cfg["out_path"]))
        if cfg.get("predictions_out"):
            _save_table(res["predictions"], Path(cfg["predictions_out"]))
        s = res["summary"]
        msg = (
            f"[backtest] CRPS={s['crps']:.3f} (skill {s['crps_skill']:.3f}), "
            f"MAE={s['mae']:.3f}, PIT KS={s['pit_ks_stat']:.3f}"
        )
        if "coverage_80" in s:
            msg += f", cov80={s['coverage_80']:.2%}"
        if res.get("calibrated_summary") and "coverage_80" in res["calibrated_summary"]:
            msg += f" (calibrated cov80={res['calibrated_summary']['coverage_80']:.2%})"
        click.echo(msg)
        click.echo(f"[backtest] wrote {cfg['out_path']}")

    # ---- events (classical + Chronos-2 conditional episodes) ----
    @cli.command("events")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option(
        "--value-col", "--target", "value_col", required=False, help="Series column name."
    )
    @click.option("--date-col", "date_col", default=None, help="Timestamp column. Default: date.")
    @click.option(
        "--method",
        type=click.Choice(["iqr", "isolation", "stl_residual", "chronos"]),
        default=None,
        help="Detection method. Default: iqr.",
    )
    @click.option(
        "--covariates",
        default=None,
        help="Comma-separated covariate columns (used by --method chronos).",
    )
    @click.option(
        "--k", type=float, default=None, help="Threshold multiplier for classical methods."
    )
    @click.option("--alpha", type=float, default=None, help="Two-sided tail level for chronos.")
    @click.option("--min-length", "min_length", type=int, default=None, help="Minimum run length.")
    @click.option(
        "--direction",
        type=click.Choice(["both", "high", "low"]),
        default=None,
        help="Anomaly direction for --method chronos. Default: both.",
    )
    @click.option(
        "--adaptive",
        is_flag=True,
        default=None,
        help="Enable causal Adaptive Conformal Inference (only with --method chronos).",
    )
    @click.option("--seed", type=int, default=None)
    @click.option("--device", default=None, help="torch device for --method chronos.")
    @click.option("--context-length", "context_length", type=int, default=None)
    @click.option("--prediction-length", "prediction_length", type=int, default=None)
    @click.option(
        "--model-path",
        "model_path",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Saved Chronos-2 checkpoint/LoRA path (only with --method chronos).",
    )
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def events_cmd(input: Path, **opts: Any) -> None:
        """Detect anomalous episodes (univariate classical or conditional Chronos-2)."""
        from .. import detect_events

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        value_col = _require_option(cfg, "value_col", "--value-col")
        df = _load_table(input)
        date_col = cfg.get("date_col") or "date"
        method = cfg.get("method") or "iqr"
        min_length = _cfg_int(cfg, "min_length", 1)
        direction = cfg.get("direction") or "both"

        if method == "chronos":
            covs = _split_csv(cfg.get("covariates")) or []
            df_reg = _prepare_chronos_frame(df, date_col, [value_col, *covs], "events")
            est = _load_chronos_estimator(cfg, covs, "events")
            out = detect_events(
                df_reg,
                value_col=value_col,
                method="chronos",
                estimator=est,
                covariates=covs,
                alpha=_cfg_float(cfg, "alpha", 0.05),
                min_length=min_length,
                direction=direction,
                adaptive=bool(cfg.get("adaptive")),
            )
        else:
            if cfg.get("model_path"):
                raise click.BadParameter("--model-path is only supported with --method chronos")
            if cfg.get("adaptive"):
                raise click.BadParameter("--adaptive is only supported with --method chronos")
            work = df.copy()
            if date_col in work.columns:
                work = (
                    work.set_index(pd.to_datetime(work[date_col]))
                    .drop(columns=[date_col])
                    .sort_index()
                )
            elif not isinstance(work.index, pd.DatetimeIndex):
                raise click.UsageError(
                    f"no {date_col!r} column and the index is not datetime; pass --date-col"
                )
            if value_col not in work.columns:
                raise click.UsageError(f"missing required column: {value_col!r}")
            out = detect_events(
                work,
                value_col=value_col,
                method=method,
                k=_cfg_float(cfg, "k", 3.0),
                min_length=min_length,
                seed=_cfg_int(cfg, "seed", 7_654_321),
            )
        _save_table(out, Path(cfg["out_path"]))
        click.echo(f"[events] detected {len(out):,} event(s) ({method}) → {cfg['out_path']}")

    # ---- feature engineering ----
    @cli.command("met-features")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option(
        "--include-memory",
        is_flag=True,
        help="Add causal first-order ODE residence integrals (needs a dated record).",
    )
    @click.option(
        "--group-col",
        default=None,
        help="Panel unit column (e.g. site) so the memory state resets per unit.",
    )
    @click.option("--residence-tau-hours", type=float, default=12.0)
    @click.option("--temp-unit", type=click.Choice(["auto", "C", "K"]), default="auto")
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    def met_features_cmd(
        input: Path,
        include_memory: bool,
        group_col: str | None,
        residence_tau_hours: float,
        temp_unit: str,
        out_path: Path,
    ) -> None:
        """Add meteorological physics features (ventilation, VPD, wind, ODE memory).

        Whichever columns the table's meteorology supports are added without
        touching the inputs; with ``--include-memory`` the causal residence
        integrals are accumulated per ``--group-col`` unit when one is given.
        """
        from .. import add_met_physics_features

        df = _load_table(input)
        out = add_met_physics_features(
            df,
            include_memory=include_memory,
            group_col=group_col,
            residence_tau_hours=residence_tau_hours,
            temp_unit=temp_unit,
        )
        added = [c for c in out.columns if c not in df.columns]
        _save_table(out, out_path)
        click.echo(
            f"[met-features] added {len(added)} column(s)"
            + (f": {', '.join(added)}" if added else "")
            + f" → {out_path}"
        )

    # ---- scm ----
    @cli.command("scm")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--treated", required=False, help="Treated unit identifier.")
    @click.option("--donors", help="Comma-separated donor pool. If omitted, all non-treated units.")
    @click.option("--cutoff", required=False, help="Cutoff date (YYYY-MM-DD).")
    @click.option("--date-col", "date_col", default=None)
    @click.option("--unit-col", "unit_col", default=None)
    @click.option("--outcome-col", "outcome_col", default=None)
    @click.option(
        "--backend",
        "scm_backend",
        type=click.Choice(
            [
                "scm",
                "mlscm",
                "abadie",
                "did",
                "mcnnm",
                "robust",
                "ascm",
                "sdid",
                "deweathered",
                "transport",
                "chronos",
            ]
        ),
        default=None,
    )
    @click.option(
        "--covariates",
        help="Comma-separated covariates (deweathered/transport backends).",
    )
    @click.option(
        "--resample-vars",
        "resample_vars",
        help="Comma-separated meteorological variables to marginalise over "
        "(deweathered backend; defaults to covariates).",
    )
    @click.option(
        "--deweather-method",
        "deweather_method",
        type=click.Choice(["dml", "crossfit", "standard"]),
        default=None,
        help="Stage-1 deweathering engine for --backend deweathered.",
    )
    @click.option(
        "--spillover-exposure",
        "spillover_exposure",
        default=None,
        help="Column carrying the donor transport exposure F_j(t) for --backend transport.",
    )
    @click.option(
        "--model-path",
        "model_path",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Saved Chronos-2 checkpoint/LoRA path (only with --backend chronos).",
    )
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def scm_cmd(input: Path, **opts: Any) -> None:
        from .. import run_scm

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        treated = _require_option(cfg, "treated", "--treated")
        cutoff = _require_option(cfg, "cutoff", "--cutoff")
        df = _load_table(input)
        scm_backend = cfg.get("scm_backend") or "scm"
        scm_kwargs: dict[str, Any] = {}
        if scm_backend == "deweathered":
            scm_kwargs["covariates"] = _split_csv(cfg.get("covariates"))
            scm_kwargs["variables_resample"] = _split_csv(cfg.get("resample_vars"))
            if cfg.get("deweather_method"):
                scm_kwargs["deweather_method"] = cfg["deweather_method"]
        elif scm_backend == "transport":
            if not cfg.get("spillover_exposure"):
                raise click.BadParameter(
                    "--spillover-exposure is required with --backend transport"
                )
            scm_kwargs["spillover_exposure"] = cfg["spillover_exposure"]
        if cfg.get("model_path"):
            if scm_backend != "chronos":
                raise click.BadParameter("--model-path is only supported with --backend chronos")
            scm_kwargs["model_config"] = {"model_path": str(cfg["model_path"])}

        out = run_scm(
            df=df,
            date_col=cfg.get("date_col") or "date",
            unit_col=cfg.get("unit_col") or "ID",
            outcome_col=cfg.get("outcome_col") or "value",
            treated_unit=treated,
            cutoff_date=cutoff,
            donors=_split_csv(cfg.get("donors")),
            scm_backend=scm_backend,
            **scm_kwargs,
        )
        _save_table(out.reset_index(), Path(cfg["out_path"]))
        click.echo(f"[scm] wrote {cfg['out_path']}")

    # ---- cv ----
    @cli.command("cv")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--target", required=False)
    @click.option("--covariates", required=False, help="Comma-separated predictor columns.")
    @click.option("--n-splits", "n_splits", type=int, default=None)
    @click.option("--gap", type=int, default=None)
    @click.option("--backend", type=backend_choice, default=None)
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def cv_cmd(input: Path, **opts: Any) -> None:
        from .. import cv_score, prepare_data

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        target = _require_option(cfg, "target", "--target")
        df = _load_table(input)
        feats = _split_csv(cfg.get("covariates")) or []
        df_prep = prepare_data(
            df,
            target=target,
            covariates=feats,
            split_method="ts",
            train_fraction=0.999,  # we only need date sorting + value rename here
        )
        scores = cv_score(
            df_prep,
            target="value",
            covariates=feats,
            backend=cfg.get("backend") or "flaml",
            n_splits=cfg.get("n_splits") or 5,
            gap=cfg.get("gap") or 0,
            verbose=True,
        )
        _save_table(scores, Path(cfg["out_path"]))
        click.echo(f"[cv] wrote {cfg['out_path']}")

    # ---- report ----
    @cli.command("report")
    @click.argument("run_path", type=click.Path(exists=True, path_type=Path))
    @click.option(
        "--out",
        "out_path",
        type=click.Path(path_type=Path),
        required=True,
        help="Output HTML or Markdown file path (.html or .md).",
    )
    @click.option("--title", default=None, help="Custom report title.")
    @click.option(
        "--lang",
        type=click.Choice(["en", "zh"]),
        default="en",
        show_default=True,
        help="The HTML report's language: its headings, labels and figure (zh: Chinese).",
    )
    def report_cmd(run_path: Path, out_path: Path, title: str | None, lang: str) -> None:
        from .. import load_run
        from ..reporting.report import generate_html, report_to_markdown

        run = load_run(run_path)
        out_path = Path(out_path)
        if out_path.suffix.lower() == ".md":
            path = report_to_markdown(run, out_path, title=title)
        else:
            path = generate_html(run, out_path, title=title, lang=lang)
        click.echo(f"[report] wrote {path}")

    # ---- info ----
    @cli.command("info")
    def info_cmd() -> None:
        import importlib.metadata as md

        def _v(pkg: str) -> str:
            try:
                return md.version(pkg)
            except Exception:
                return "(not installed)"

        click.echo(
            json.dumps(
                {
                    "normet": _v("normet"),
                    "python": sys.version.split()[0],
                    "deps": {
                        "numpy": _v("numpy"),
                        "pandas": _v("pandas"),
                        "scipy": _v("scipy"),
                        "scikit-learn": _v("scikit-learn"),
                        "joblib": _v("joblib"),
                        "matplotlib": _v("matplotlib"),
                    },
                    "optional": {
                        "flaml": _v("flaml"),
                        "lightgbm": _v("lightgbm"),
                        "cdsapi": _v("cdsapi"),
                        "dask": _v("dask"),
                        "click": _v("click"),
                        # `normet deweather` and normet.foundation need these two.
                        "chronos-forecasting": _v("chronos-forecasting"),
                        "torch": _v("torch"),
                        "PySide6": _v("PySide6"),
                    },
                    "backends": backend_registry.available,
                },
                indent=2,
            )
        )

    return cli


def main() -> None:  # entry point
    """Run the ``normet`` console-script entry point."""
    cli = _build_cli()
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
