# src/normet/cli.py
"""
Command-line interface for ``normet``.

Subcommands wrap the most common library entry points:

- ``normet do-all <input.csv> ...``
- ``normet decompose <input.csv> ...``
- ``normet scm <panel.csv> ...``
- ``normet cv <input.csv> ...``
- ``normet info``

Any subcommand can read its options from a YAML file via ``--config foo.yaml``;
CLI flags take precedence over file values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .utils._lazy import require
from .utils.logging import enable_default_logging, get_logger

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


def _split_csv(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    yaml = require("yaml", hint="pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge_cli_over_yaml(yaml_cfg: dict[str, Any], cli_args: dict[str, Any]) -> dict[str, Any]:
    """CLI args win; None values fall back to YAML."""
    out = dict(yaml_cfg)
    for k, v in cli_args.items():
        if v is not None:
            out[k] = v
    return out


# ---------- CLI ----------


def _build_cli():
    click = require("click", hint="pip install click")

    @click.group()
    @click.version_option(package_name="normet")
    def cli():
        """Command-line interface for the normet modelling toolbox."""
        enable_default_logging("INFO")

    # ---- do-all ----
    @cli.command("do-all")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--value", help="Target column name (e.g., PM2.5).")
    @click.option("--features", help="Comma-separated predictor columns.")
    @click.option(
        "--resample-vars", "resample_vars", help="Comma-separated subset of features to resample."
    )
    @click.option("--backend", type=click.Choice(["flaml"]), default=None)
    @click.option("--n-samples", "n_samples", type=int, default=None)
    @click.option(
        "--split-method",
        "split_method",
        type=click.Choice(["random", "ts", "season", "month"]),
        default=None,
    )
    @click.option("--fraction", type=float, default=None)
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
    def do_all_cmd(input, **opts):
        from . import do_all, make_run, save_run

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        df = _load_table(input)

        out, model, df_prep = do_all(
            df=df,
            value=cfg["value"],
            feature_names=_split_csv(cfg.get("features")),
            variables_resample=_split_csv(cfg.get("resample_vars")),
            backend=cfg.get("backend") or "flaml",
            n_samples=cfg.get("n_samples") or 300,
            split_method=cfg.get("split_method") or "random",
            fraction=cfg.get("fraction") if cfg.get("fraction") is not None else 0.75,
            seed=cfg.get("seed") if cfg.get("seed") is not None else 7_654_321,
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
    @click.option("--value", required=False)
    @click.option("--features", help="Comma-separated predictor columns.")
    @click.option("--method", type=click.Choice(["emission", "meteorology"]), default=None)
    @click.option("--backend", type=click.Choice(["flaml"]), default=None)
    @click.option("--n-samples", "n_samples", type=int, default=None)
    @click.option("--seed", type=int, default=None)
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def decompose_cmd(input, **opts):
        from . import decompose

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        df = _load_table(input)

        out = decompose(
            method=cfg.get("method") or "emission",
            df=df,
            value=cfg["value"],
            feature_names=_split_csv(cfg.get("features")),
            backend=cfg.get("backend") or "flaml",
            n_samples=cfg.get("n_samples") or 300,
            seed=cfg.get("seed") if cfg.get("seed") is not None else 7_654_321,
            verbose=True,
        )
        _save_table(out.reset_index(), Path(cfg["out_path"]))
        click.echo(f"[decompose] wrote {cfg['out_path']}")

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
        type=click.Choice(["scm", "mlscm", "abadie", "did", "mcnnm"]),
        default=None,
    )
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def scm_cmd(input, **opts):
        from . import run_scm

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        df = _load_table(input)

        out = run_scm(
            df=df,
            date_col=cfg.get("date_col") or "date",
            unit_col=cfg.get("unit_col") or "ID",
            outcome_col=cfg.get("outcome_col") or "value",
            treated_unit=cfg["treated"],
            cutoff_date=cfg["cutoff"],
            donors=_split_csv(cfg.get("donors")),
            scm_backend=cfg.get("scm_backend") or "scm",
        )
        _save_table(out.reset_index(), Path(cfg["out_path"]))
        click.echo(f"[scm] wrote {cfg['out_path']}")

    # ---- cv ----
    @cli.command("cv")
    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("--value", required=False)
    @click.option("--features", required=False, help="Comma-separated predictor columns.")
    @click.option("--n-splits", "n_splits", type=int, default=None)
    @click.option("--gap", type=int, default=None)
    @click.option("--backend", type=click.Choice(["flaml"]), default=None)
    @click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
    @click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
    def cv_cmd(input, **opts):
        from . import cv_score, prepare_data

        cfg = _merge_cli_over_yaml(_load_yaml(opts.pop("config_path", None)), opts)
        df = _load_table(input)
        feats = _split_csv(cfg.get("features")) or []
        df_prep = prepare_data(
            df,
            value=cfg["value"],
            feature_names=feats,
            split_method="ts",
            fraction=0.999,  # we only need date sorting + value rename here
        )
        scores = cv_score(
            df_prep,
            value="value",
            feature_names=feats,
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
    def report_cmd(run_path, out_path, title):
        from . import load_run
        from .report import generate_html, report_to_markdown

        run = load_run(run_path)
        out_path = Path(out_path)
        if out_path.suffix.lower() == ".md":
            path = report_to_markdown(run, out_path, title=title)
        else:
            path = generate_html(run, out_path, title=title)
        click.echo(f"[report] wrote {path}")

    # ---- info ----
    @cli.command("info")
    def info_cmd():
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
                    },
                },
                indent=2,
            )
        )

    return cli


def main():  # entry point
    """Run the ``normet`` console-script entry point."""
    cli = _build_cli()
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
