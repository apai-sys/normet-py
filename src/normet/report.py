# src/normet/report.py
"""
One-command HTML report generation from a :class:`NormetRun`.

Output is a single self-contained HTML file with embedded base64-encoded
PNG plots, full provenance metadata, model summary, and result preview.
No browser-side JS, no external CDN dependencies.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils.logging import get_logger
from .utils.provenance import NormetRun

log = get_logger(__name__)

__all__ = ["generate_html", "report_to_markdown"]


def _fig_to_b64(fig) -> str:
    """Serialise a matplotlib Figure to an inline data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _df_table(df: pd.DataFrame, *, max_rows: int = 12) -> str:
    """Render a small HTML table for the head of a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        return f"<pre>{html.escape(repr(df))}</pre>"  # type: ignore[unreachable]
    if df.empty:
        return "<p><em>(empty)</em></p>"
    head = df.head(max_rows)
    return head.to_html(classes="nm-tbl", border=0, na_rep="—", float_format=lambda x: f"{x:.4g}")


def _json_pre(obj: Any) -> str:
    """Pretty-print a metadata dict."""
    try:
        s = json.dumps(obj, indent=2, sort_keys=True, default=repr)
    except Exception:
        s = repr(obj)
    return f"<pre class='nm-meta'>{html.escape(s)}</pre>"


def _auto_plot(run: NormetRun) -> Any:
    """Pick a sensible default plot for the run kind. Returns the figure (or None)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kind = (run.metadata.get("kind") or "").lower()
    res = run.result

    if not isinstance(res, pd.DataFrame) or res.empty:
        return None

    fig = None
    try:
        # do_all / normalise → observed vs normalised
        if {"observed", "normalised"} <= set(res.columns):
            from .plotting import normalise_plot

            ci_low = None
            ci_high = None
            # Find any quantiles or uncertainty columns
            q_cols = [
                c for c in res.columns if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()
            ]
            if len(q_cols) >= 2:
                try:
                    q_cols_sorted = sorted(q_cols, key=lambda x: float(x[1:]))
                    ci_low = q_cols_sorted[0]
                    ci_high = q_cols_sorted[-1]
                except ValueError:
                    pass

            fig, ax = plt.subplots(figsize=(10, 4))
            normalise_plot(
                res,
                observed_col="observed",
                normalised_col="normalised",
                ci_low=ci_low,
                ci_high=ci_high,
                title=kind or "normalised series",
                ylabel="value",
                ax=ax,
            )

        # SCM-like
        elif {"observed", "synthetic", "effect"} <= set(res.columns):
            cutoff = run.metadata.get("config", {}).get("cutoff_date") if run.metadata else None
            if cutoff:
                if "synthetic_low" in res.columns:
                    from .plotting import plot_bayesian_scm

                    fig = plot_bayesian_scm(
                        res, cutoff_date=str(cutoff), title=kind or "Bayesian SCM"
                    )
                else:
                    from .plotting import scm_dashboard

                    fig = scm_dashboard(res, cutoff_date=str(cutoff), title=kind or "SCM")

        # decomposition (has many feature columns + observed)
        elif (
            "observed" in res.columns
            and len(
                [c for c in res.columns if c not in {"observed", "model_pred", "residual", "base"}]
            )
            >= 2
        ):
            from .plotting import decomposition_stack

            fig, ax = plt.subplots(figsize=(11, 4))
            decomposition_stack(res, ax=ax, title=kind or "decomposition")

        # CV scores
        elif {"fold", "RMSE"}.issubset(res.columns) or {"fold", "r"}.issubset(res.columns):
            metric = "RMSE" if "RMSE" in res.columns else "r"
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.bar(res["fold"], res[metric], color="tab:blue")
            ax.set_xlabel("fold")
            ax.set_ylabel(metric)
            ax.set_title("Walk-forward CV scores")
            ax.grid(axis="y", alpha=0.2)
    except Exception as e:
        log.debug("auto_plot failed: %s", e)
        return None

    return fig


_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1100px;
          margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
  h1 {{ font-size: 1.6rem; }}
  h2 {{ font-size: 1.2rem; margin-top: 2.2rem; }}
  .kv {{ display: grid; grid-template-columns: 12rem 1fr; gap: .25rem 1rem;
         font-size: .9rem; }}
  .kv dt {{ color: #888; }}
  .kv dd {{ margin: 0; font-family: ui-monospace, monospace; }}
  pre.nm-meta {{ background: #f7f7f8; padding: .8rem; border-radius: 4px;
                  overflow-x: auto; font-size: .8rem; line-height: 1.4; }}
  table.nm-tbl {{ border-collapse: collapse; font-size: .85rem; }}
  table.nm-tbl th, table.nm-tbl td {{ padding: .25rem .6rem; text-align: right; }}
  table.nm-tbl th {{ background: #f0f0f2; border-bottom: 1px solid #ccc; }}
  table.nm-tbl tr:nth-child(even) {{ background: #fafafa; }}
  img.nm-plot {{ max-width: 100%; border: 1px solid #eee; padding: 6px;
                  background: white; margin-top: .6rem; }}
  footer {{ margin-top: 3rem; color: #888; font-size: .8rem;
            border-top: 1px solid #eee; padding-top: .6rem; }}
</style>
</head><body>

<h1>{title}</h1>
<dl class="kv">
{kv_rows}
</dl>

<h2>Plot</h2>
{plot_block}

<h2>Result preview</h2>
{table_block}

<h2>Provenance</h2>
{meta_block}

{model_block}

<footer>
Generated by normet on {generated_at}.
</footer>

</body></html>
"""


def generate_html(
    run: NormetRun,
    out_path: str | Path,
    *,
    title: str | None = None,
    extra_plots: list | None = None,
) -> Path:
    """
    Render a single-file HTML report for a :class:`NormetRun`.

    Parameters
    ----------
    run : NormetRun
        Output of :func:`make_run` (or loaded via :func:`load_run`).
    out_path : str | Path
        Destination HTML file. Parent directory is created if missing.
    title : str, optional
        Report title. Defaults to ``"normet run report — {kind}"``.
    extra_plots : list of matplotlib.figure.Figure, optional
        Additional figures to embed below the auto-generated plot.

    Returns
    -------
    pathlib.Path
        Path to the written HTML file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(run.metadata or {})
    kind = meta.get("kind", "run")
    title = title or f"normet report — {kind}"

    # KV header
    kv_keys = [
        "kind",
        "normet_version",
        "python_version",
        "platform",
        "host",
        "user",
        "timestamp",
        "seed",
        "data_hash",
        "config_hash",
    ]
    kv_rows = "\n".join(
        f"<dt>{html.escape(str(k))}</dt><dd>{html.escape(str(meta.get(k, '—')))}</dd>"
        for k in kv_keys
    )

    # Plot
    fig = _auto_plot(run)
    plot_block_parts = []
    if fig is not None:
        plot_block_parts.append(f'<img class="nm-plot" src="{_fig_to_b64(fig)}" alt="auto plot">')
    for extra in extra_plots or []:
        try:
            plot_block_parts.append(
                f'<img class="nm-plot" src="{_fig_to_b64(extra)}" alt="user plot">'
            )
        except Exception as e:
            log.warning("extra plot serialisation failed: %s", e)
    plot_block = "\n".join(plot_block_parts) or "<p><em>No plot available.</em></p>"

    # Table preview
    table_block = _df_table(run.result)

    # Provenance
    meta_block = _json_pre(meta)

    # Model summary
    model_block = ""
    if run.model is not None:
        model_block = f"""
<h2>Model</h2>
<dl class="kv">
  <dt>type</dt><dd>{html.escape(type(run.model).__name__)}</dd>
  <dt>backend</dt><dd>{html.escape(str(getattr(run.model, "backend", "—")))}</dd>
</dl>
"""

    body = _TEMPLATE.format(
        title=html.escape(title),
        kv_rows=kv_rows,
        plot_block=plot_block,
        table_block=table_block,
        meta_block=meta_block,
        model_block=model_block,
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
    out_path.write_text(body, encoding="utf-8")
    log.info("HTML report → %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    return out_path


def report_to_markdown(
    run: NormetRun,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """
    Render a single-file plain-text Markdown report for a :class:`NormetRun`.

    Parameters
    ----------
    run : NormetRun
        Output of :func:`make_run` (or loaded via :func:`load_run`).
    out_path : str | Path
        Destination Markdown file. Parent directory is created if missing.
    title : str, optional
        Report title. Defaults to ``"normet report — {kind}"``.

    Returns
    -------
    pathlib.Path
        Path to the written Markdown file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(run.metadata or {})
    kind = meta.get("kind", "run")
    title = title or f"normet report — {kind}"

    lines = []
    lines.append(f"# {title}")
    lines.append("")

    # Metadata overview
    kv_keys = [
        "kind",
        "normet_version",
        "python_version",
        "platform",
        "host",
        "user",
        "timestamp",
        "seed",
        "data_hash",
        "config_hash",
    ]
    for k in kv_keys:
        lines.append(f"- **{k}**: `{meta.get(k, '—')}`")
    lines.append("")

    # Result preview
    lines.append("## Result Preview")
    lines.append("")
    if run.result is None or run.result.empty:
        lines.append("*(empty)*")
    else:
        try:
            table_str = run.result.head(12).to_markdown()
        except Exception as e:
            log.debug(
                "to_markdown() failed (is 'tabulate' installed?); building table manually: %s", e
            )
            df_head = run.result.head(12)
            cols = list(df_head.columns)
            if isinstance(df_head.index, pd.DatetimeIndex):
                header = ["date"] + cols
                align = ["---"] * len(header)
                tbl_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
                for idx, row in df_head.iterrows():
                    val_strs = [str(idx)] + [
                        f"{v:.4g}" if isinstance(v, int | float) else str(v) for v in row
                    ]
                    tbl_lines.append("| " + " | ".join(val_strs) + " |")
            else:
                header = cols
                align = ["---"] * len(header)
                tbl_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
                for idx, row in df_head.iterrows():
                    val_strs = [f"{v:.4g}" if isinstance(v, int | float) else str(v) for v in row]
                    tbl_lines.append("| " + " | ".join(val_strs) + " |")
            table_str = "\n".join(tbl_lines)
        lines.append(table_str)
    lines.append("")

    # Model Summary
    if run.model is not None:
        lines.append("## Model Summary")
        lines.append("")
        lines.append(f"- **type**: `{type(run.model).__name__}`")
        lines.append(f"- **backend**: `{getattr(run.model, 'backend', '—')}`")
        lines.append("")

    # Full Provenance Metadata JSON block
    lines.append("## Full Provenance Metadata")
    lines.append("")
    lines.append("```json")
    try:
        s = json.dumps(meta, indent=2, sort_keys=True, default=repr)
    except Exception:
        s = repr(meta)
    lines.append(s)
    lines.append("```")
    lines.append("")

    lines.append(
        f"Generated by normet on {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}."
    )

    body = "\n".join(lines)
    out_path.write_text(body, encoding="utf-8")
    log.info("Markdown report → %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    return out_path
