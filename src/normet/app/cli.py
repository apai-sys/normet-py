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
    with open(path) as f:
        return yaml.safe_load(f) or {}


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


def _merge_cli_over_yaml(yaml_cfg: dict[str, Any], cli_args: dict[str, Any]) -> dict[str, Any]:
    """CLI args win; None values fall back to YAML."""
    out = dict(yaml_cfg)
    for k, v in cli_args.items():
        if v is not None:
            out[k] = v
    return out


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
        from .. import do_all, make_run, save_run

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        df = _load_table(input)

        out, model, df_prep = do_all(
            df=df,
            target=cfg["target"],
            covariates=_split_csv(cfg.get("covariates")),
            variables_resample=_split_csv(cfg.get("resample_vars")),
            backend=cfg.get("backend") or "flaml",
            n_samples=int(cfg["n_samples"]) if cfg.get("n_samples") is not None else None,
            split_method=cfg.get("split_method") or "random",
            train_fraction=_cfg_float(cfg, "train_fraction", 0.75),
            seed=_cfg_int(cfg, "seed", 7_654_321),
            verbose=True,
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
        df = _load_table(input)

        out = decompose(
            method=cfg.get("method") or "emission",
            df=df,
            target=cfg["target"],
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
        target = cfg["target"]
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
            design = ICNDesign.v1(**design_overrides)
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
        df = _load_table(input)
        target = cfg["target"]
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
        df = _load_table(input)
        target = cfg["target"]
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
        df = _load_table(input)
        value_col = cfg["value_col"]
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
        type=click.Choice(["scm", "mlscm", "abadie", "did", "mcnnm", "robust", "chronos"]),
        default=None,
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
        df = _load_table(input)
        scm_backend = cfg.get("scm_backend") or "scm"
        scm_kwargs: dict[str, Any] = {}
        if cfg.get("model_path"):
            if scm_backend != "chronos":
                raise click.BadParameter("--model-path is only supported with --backend chronos")
            scm_kwargs["model_config"] = {"model_path": str(cfg["model_path"])}

        out = run_scm(
            df=df,
            date_col=cfg.get("date_col") or "date",
            unit_col=cfg.get("unit_col") or "ID",
            outcome_col=cfg.get("outcome_col") or "value",
            treated_unit=cfg["treated"],
            cutoff_date=cfg["cutoff"],
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
        df = _load_table(input)
        feats = _split_csv(cfg.get("covariates")) or []
        df_prep = prepare_data(
            df,
            target=cfg["target"],
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
