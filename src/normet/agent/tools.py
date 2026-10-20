"""The tools an assistant drives normet with (:mod:`normet.agent`).

Each returns text -- normet's own readings (:mod:`normet.agent.readings`) --
or, for a figure, its caption and a PNG. A job's tool starts a ``normet``
command and returns at once (:mod:`normet.agent.jobs`); its status is asked
for after, and once it has ended its status carries the readings of what it
wrote.

Files are read and written only under the folders allowed:
``$NORMET_MCP_ROOTS`` (folders separated as PATH separates them), the user's
home folder where it is not set, and the jobs' own folder.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Callable
from typing import Any

import pandas as pd

from normet import style as S

from . import readings
from .jobs import JobStore

#: The examples normet ships, by the name a tool takes them by:
#: (loader in normet.datasets, what it is).
EXAMPLES: dict[str, tuple[str, str]] = {
    "my1": (
        "load_my1",
        "Hourly NO2 and meteorology at London Marylebone Road (MY1), Jan–Aug 2020: "
        "a single site, for do-all, decompose and cv (target NO2).",
    ),
    "my1_pm25": (
        "load_my1_pm25",
        "Hourly PM2.5 and meteorology at MY1, Jan–Aug 2020 (target PM2.5).",
    ),
    "scm": (
        "load_scm",
        "Monthly de-weathered NO2 at 104 UK sites, 2016–2021 (unit column 'code', "
        "outcomes NO2_obs / NO2_dw): a panel for synthetic control, e.g. the ULEZ at MY1.",
    ),
}

#: The figures a tool draws: kind -> what it shows and what it is drawn from.
FIGURES: dict[str, str] = {
    "normalised": "observed and normalised series, daily means (a run or a normalised CSV)",
    "model": "observed against predicted on the held-out rows (a run)",
    "importance": "the model's feature importance (a run)",
    "decomposition": "each component of a decomposition over time (a decomposition CSV)",
    "scm": "the treated unit against its synthetic control (an SCM CSV; cutoff marks the date)",
}

#: What each job writes, read back once it has ended.
_OUTPUTS = {
    "do_all": "normalised.csv",
    "decompose": "decomposition.csv",
    "scm": "scm.csv",
    "cv": "cv.csv",
}


def _cut(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else "…" + text[-limit:]


class Tools:
    """normet's tools over one job store and one set of allowed folders."""

    def __init__(self, store: JobStore | None = None, roots: list[str] | None = None) -> None:
        self.store = store or JobStore()
        if roots is None:
            env = os.environ.get("NORMET_MCP_ROOTS", "")
            roots = [p for p in env.split(os.pathsep) if p] or [os.path.expanduser("~")]
        self.roots = [os.path.realpath(os.path.expanduser(r)) for r in roots]

    # -- files ------------------------------------------------------------------
    def _allowed(self, path: str) -> bool:
        real = os.path.realpath(path)
        for root in [*self.roots, os.path.realpath(self.store.home)]:
            if real == root or real.startswith(root.rstrip(os.sep) + os.sep):
                return True
        return False

    def path(self, given: str, must_exist: bool = True) -> str:
        """*given* as an absolute path inside the allowed folders, or PermissionError."""
        p = os.path.abspath(os.path.expanduser(str(given)))
        if not self._allowed(p):
            raise PermissionError(
                f"{p} is outside the folders normet may use ({os.pathsep.join(self.roots)}); "
                "set NORMET_MCP_ROOTS to allow another"
            )
        if must_exist and not os.path.exists(p):
            raise FileNotFoundError(f"no file {p}")
        return p

    def list_examples(self) -> str:
        """The bundled examples, by the name a tool takes them by."""
        return "\n".join(f"{name}: {about}" for name, (_, about) in EXAMPLES.items())

    def example_path(self, name: str) -> str:
        """An example written as a CSV in the store's folder (once), its path."""
        if name not in EXAMPLES:
            raise ValueError(f"no example {name!r}; the examples are {', '.join(EXAMPLES)}")
        folder = os.path.join(self.store.home, "examples")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{name}.csv")
        if not os.path.exists(path):
            from normet import datasets

            getattr(datasets, EXAMPLES[name][0])().to_csv(path + ".part", index=False)
            os.replace(path + ".part", path)
        return path

    def _source(self, data: str | None, example: str | None) -> str:
        if (data is None) == (example is None):
            raise ValueError("give one of data=<path of a table> or example=<name>")
        return self.example_path(example) if example is not None else self.path(str(data))

    @staticmethod
    def _table(path: str) -> pd.DataFrame:
        from pathlib import Path

        from normet.cli import _load_table

        return _load_table(Path(path))

    # -- the data ---------------------------------------------------------------
    def inspect_data(
        self, data: str | None = None, example: str | None = None, target: str | None = None
    ) -> str:
        """A table before anything is fitted: its size and span, the time step and
        gaps, each numeric column's role, missing values and range, and -- with
        *target* -- each column's correlation with it."""
        path = self._source(data, example)
        return readings.data_reading(self._table(path), target, name=os.path.basename(path))

    # -- jobs -------------------------------------------------------------------
    @staticmethod
    def _options(**opts: Any) -> list[str]:
        argv: list[str] = []
        for key, value in opts.items():
            if value is None or value == [] or value == "":
                continue
            if isinstance(value, (list, tuple)):
                value = ",".join(str(v) for v in value)
            argv += [f"--{key.replace('_', '-')}", str(value)]
        return argv

    def _start(self, kind: str, argv: list[str], label: str) -> str:
        job = self.store.submit(kind, argv, label=label)
        ahead = job.get("ahead")
        return (
            f"Started job {job['id']} ({kind}): {job['state']}"
            + (f", {ahead} ahead of it" if ahead else "")
            + f".\nCommand: {job['command']}\nAsk normet_job_status for how it goes."
        )

    def start_do_all(
        self,
        target: str,
        covariates: list[str],
        data: str | None = None,
        example: str | None = None,
        resample_vars: list[str] | None = None,
        backend: str | None = None,
        n_samples: int | None = None,
        split_method: str | None = None,
        train_fraction: float | None = None,
        seed: int | None = None,
        label: str = "",
    ) -> str:
        """Train a model of *target* on *covariates* and normalise it (``normet do-all``)."""
        if not covariates:
            raise ValueError("give the covariates (predictor columns) to train on")
        argv = ["do-all", self._source(data, example)] + self._options(
            target=target,
            covariates=covariates,
            resample_vars=resample_vars,
            backend=backend,
            n_samples=n_samples,
            split_method=split_method,
            train_fraction=train_fraction,
            seed=seed,
        )
        argv += ["--out", _OUTPUTS["do_all"], "--run-path", "run"]
        return self._start("do_all", argv, label)

    def start_decompose(
        self,
        target: str,
        covariates: list[str],
        method: str = "emission",
        data: str | None = None,
        example: str | None = None,
        backend: str | None = None,
        n_samples: int | None = None,
        seed: int | None = None,
        label: str = "",
    ) -> str:
        """Decompose *target* by time scale (``emission``) or by meteorological
        variable (``meteorology``) (``normet decompose``)."""
        if method not in ("emission", "meteorology"):
            raise ValueError("method is 'emission' or 'meteorology'")
        argv = ["decompose", self._source(data, example)] + self._options(
            target=target,
            covariates=covariates,
            method=method,
            backend=backend,
            n_samples=n_samples,
            seed=seed,
        )
        argv += ["--out", _OUTPUTS["decompose"]]
        return self._start("decompose", argv, label)

    def start_scm(
        self,
        treated: str,
        cutoff: str,
        data: str | None = None,
        example: str | None = None,
        donors: list[str] | None = None,
        date_col: str | None = None,
        unit_col: str | None = None,
        outcome_col: str | None = None,
        backend: str | None = None,
        label: str = "",
    ) -> str:
        """A synthetic control of *treated* from the donors, its effect after
        *cutoff* (``normet scm``)."""
        argv = ["scm", self._source(data, example)] + self._options(
            treated=treated,
            cutoff=cutoff,
            donors=donors,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            backend=backend,
        )
        argv += ["--out", _OUTPUTS["scm"]]
        return self._start("scm", argv, label)

    def start_cv(
        self,
        target: str,
        covariates: list[str],
        data: str | None = None,
        example: str | None = None,
        n_splits: int | None = None,
        gap: int | None = None,
        backend: str | None = None,
        label: str = "",
    ) -> str:
        """Time-series cross-validation of a model of *target* (``normet cv``)."""
        argv = ["cv", self._source(data, example)] + self._options(
            target=target, covariates=covariates, n_splits=n_splits, gap=gap, backend=backend
        )
        argv += ["--out", _OUTPUTS["cv"]]
        return self._start("cv", argv, label)

    def _readings_of(self, job: dict[str, Any]) -> str:
        folder, kind = job["folder"], job["kind"]
        out = os.path.join(folder, _OUTPUTS.get(kind, ""))
        if kind == "do_all" and os.path.exists(os.path.join(folder, "run.joblib")):
            return self.read_run(os.path.join(folder, "run.joblib"))
        if not os.path.isfile(out):
            return ""
        argv = job.get("argv", [])
        cutoff = argv[argv.index("--cutoff") + 1] if "--cutoff" in argv else None
        return self.read_result(out, kind, cutoff=cutoff)

    def job_status(self, job_id: str) -> str:
        """Where a job is; once ended, its readings, the files it left, or its error."""
        job = self.store.status(job_id)
        lines = [
            f"Job {job['id']} ({job['kind']}{', ' + job['label'] if job.get('label') else ''}): "
            f"{job['state']}"
            + (f" after {job['elapsed_s']} s" if job.get("elapsed_s") is not None else "")
            + (f", {job['ahead']} ahead of it" if job.get("ahead") else "")
            + ".",
            f"Command (run from {job['folder']}): {job['command']}",
        ]
        if job["state"] == "failed":
            err = self.store.text(job_id, "stderr.txt", tail=3000).strip()
            out = self.store.text(job_id, "stdout.txt", tail=1500).strip()
            lines += ["", "It failed. Its last output:", _cut(err or out or "(nothing printed)")]
        elif job["state"] == "done":
            lines += ["", "Files:", *(f"  {f}" for f in job["files"])]
            try:
                said = self._readings_of(job)
            except Exception as exc:  # the files are there; a reading of them failed
                said = f"(could not read its output: {type(exc).__name__}: {exc})"
            if said:
                lines += ["", said]
        elif job["state"] == "running":
            tail = self.store.text(job_id, "stderr.txt", tail=600).strip()
            if tail:
                lines += ["", "Latest log lines:", tail]
        return "\n".join(lines)

    def cancel_job(self, job_id: str) -> str:
        job = self.store.cancel(job_id)
        return f"Job {job['id']}: {job['state']}."

    def list_jobs(self, limit: int = 20) -> str:
        ids = self.store.ids()[::-1][: max(1, int(limit))]
        if not ids:
            return "No jobs yet."
        rows = []
        for job_id in ids:
            job = self.store.status(job_id)
            rows.append(
                [job_id, job["kind"], job["state"], job.get("label") or job["command"][:60]]
            )
        return readings.table(["id", "kind", "state", "label / command"], rows)

    # -- results ----------------------------------------------------------------
    def _run(self, run: str) -> Any:
        from normet import load_run

        p = self.path(run if run.endswith(".joblib") else run + ".joblib")
        return load_run(p)

    def read_run(self, run: str) -> str:
        """A saved do-all run (``run.joblib``): the model's fit and the normalised series."""
        r = self._run(run)
        parts = []
        meta = r.metadata or {}
        cfg = meta.get("config") or {}
        if cfg:
            said = {
                k: cfg.get(k)
                for k in ("target", "covariates", "backend", "n_samples", "split_method")
                if cfg.get(k)
            }
            parts.append("Run: " + json.dumps(said, default=str))
        if r.model is not None and r.df_prep is not None:
            parts.append(readings.model_reading(r.df_prep, r.model))
        if isinstance(r.result, pd.DataFrame):
            parts.append(readings.normalised_reading(r.result))
        return "\n\n".join(parts)

    def read_result(self, path: str, kind: str, cutoff: str | None = None) -> str:
        """A result table read as its *kind* says: ``normalised``/``do_all``,
        ``decomposition``/``decompose``, ``scm`` (with *cutoff*) or ``cv``."""
        df = self._table(self.path(path))
        if kind in ("normalised", "do_all"):
            return readings.normalised_reading(df)
        if kind in ("decomposition", "decompose"):
            return readings.decomposition_reading(df)
        if kind == "scm":
            if not cutoff:
                raise ValueError("an SCM result is read against its cutoff date: give cutoff=")
            return readings.scm_reading(df, cutoff)
        if kind == "cv":
            return readings.cv_reading(df)
        raise ValueError("kind is normalised, decomposition, scm or cv")

    def figure(
        self, source: str, kind: str, cutoff: str | None = None, dpi: int = 100
    ) -> tuple[str, bytes]:
        """One of the figures in :data:`FIGURES`, as ``(caption, png)``."""
        if kind not in FIGURES:
            raise ValueError(f"kind is one of {', '.join(FIGURES)}")
        import matplotlib as mpl
        from matplotlib.figure import Figure

        with mpl.rc_context(S.rc()):  # the window's figure style
            fig = Figure(figsize=(6.5, 3.2), layout="constrained")
            ax = fig.add_subplot()
            draw: Callable[[], str]
            if kind in ("model", "importance") or source.endswith(".joblib"):
                run = self._run(source)

                def draw() -> str:
                    return _draw_run(ax, run, kind)

            else:
                df = self._table(self.path(source))

                def draw() -> str:
                    return _draw_table(fig, ax, df, kind, cutoff)

            caption = draw()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=max(dpi, 150))
        return caption, buf.getvalue()

    def write_report(self, run: str, out: str | None = None) -> str:
        """The HTML report of a saved run, written beside it or to *out*."""
        from normet.report import generate_html

        r = self._run(run)
        base = run[: -len(".joblib")] if run.endswith(".joblib") else run
        target = self.path(out or base + "_report.html", must_exist=False)
        path = generate_html(r, target)
        return f"Wrote {path}."


def _dated(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        return df.set_index(pd.to_datetime(df["date"])).drop(columns="date")
    return df


def _normalised_axes(ax: Any, d: pd.DataFrame) -> None:
    """Observed and normalised daily means, in the window's colours."""
    ax.plot(d.index, d["observed"], lw=0.8, color=S.OBSERVED, alpha=0.6, label="observed")
    ax.plot(d.index, d["normalised"], lw=1.2, color=S.MODELLED, label="normalised")
    if isinstance(d.index, pd.DatetimeIndex):
        S.format_time_axis(ax)
    ax.margins(x=0)
    S.grid(ax)
    S.legend(ax, loc="upper right", ncols=2)


def _draw_run(ax: Any, run: Any, kind: str) -> str:
    if kind == "normalised":
        df = run.result.resample("D").mean()
        _normalised_axes(ax, df)
        return "Observed and meteorologically normalised series (daily means)."
    if kind == "model":
        from normet import ml_predict

        test = run.df_prep[run.df_prep["set"] == "testing"]
        test = test if len(test) else run.df_prep
        pred = ml_predict(run.model, test)
        ax.scatter(test["value"], pred, s=5, alpha=0.35, color=S.OBSERVED, edgecolors="none")
        lo = float(min(test["value"].min(), pred.min()))
        hi = float(max(test["value"].max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], color=S.INK, lw=0.7, ls=(0, (4, 3)))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        S.grid(ax)
        return f"Observed against predicted on {len(test):,} held-out rows; dashed: 1:1."
    if kind == "importance":
        fi = readings.feature_importance(run.model)
        if fi is None:
            raise ValueError("this run's model does not report feature importance")
        names, vals = fi
        order = sorted(range(len(vals)), key=lambda i: vals[i])[-15:]
        ax.barh([names[i] for i in order], [vals[i] for i in order], color=S.COMPONENT, height=0.65)
        ax.tick_params(axis="y", length=0)
        S.grid(ax, axis="x")
        return "The model's feature importance (largest 15)."
    df = run.result
    return _draw_table(ax.figure, ax, df, kind, None)


def _draw_table(fig: Any, ax: Any, df: pd.DataFrame, kind: str, cutoff: str | None) -> str:
    df = _dated(df)
    if kind == "normalised":
        d = df[["observed", "normalised"]].resample("D").mean()
        _normalised_axes(ax, d)
        return "Observed and meteorologically normalised series (daily means)."
    if kind == "decomposition":
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        fig.clear()
        fig.set_size_inches(8, 1.4 * max(len(cols), 1) + 0.6)
        axes = fig.subplots(len(cols), 1, sharex=True, squeeze=False)[:, 0]
        for a, c in zip(axes, cols, strict=True):
            a.plot(df.index, df[c], lw=0.6, color=S.COMPONENT)
            a.set_ylabel(c, rotation=0, ha="right", va="center")
            a.margins(x=0)
            S.grid(a)
        if isinstance(df.index, pd.DatetimeIndex):
            S.format_time_axis(axes[-1])
        S.outer_ticks(axes)
        return f"The decomposition's {len(cols)} components."
    if kind == "scm":
        ax.plot(df.index, df["observed"], lw=1.2, color=S.OBSERVED, label="treated (observed)")
        ax.plot(
            df.index, df["synthetic"], lw=1.2, ls="--", color=S.MODELLED, label="synthetic control"
        )
        if cutoff:
            S.cutoff_line(ax, pd.Timestamp(cutoff))
        if isinstance(df.index, pd.DatetimeIndex):
            S.format_time_axis(ax)
        S.grid(ax)
        S.legend(ax)
        return "The treated unit and its synthetic control" + (
            f"; dotted: {cutoff}." if cutoff else "."
        )
    raise ValueError(f"a {kind!r} figure is drawn from a run (.joblib)")
