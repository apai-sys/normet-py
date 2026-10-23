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

from . import style as S
from .style import styled
from .utils.logging import get_logger
from .utils.provenance import NormetRun

log = get_logger(__name__)

__all__ = ["generate_html", "report_to_markdown"]


def _fig_to_b64(fig: Any) -> str:
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


@styled
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

            fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
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

            fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
            decomposition_stack(res, ax=ax, title=kind or "decomposition")

        # CV scores
        elif {"fold", "RMSE"}.issubset(res.columns) or {"fold", "r"}.issubset(res.columns):
            metric = "RMSE" if "RMSE" in res.columns else "r"
            fig, ax = plt.subplots(figsize=(6, 2.8), layout="constrained")
            ax.bar(res["fold"], res[metric], color=S.COMPONENT, width=0.6)
            ax.set_xlabel("fold")
            ax.set_ylabel(metric)
            ax.set_title("Walk-forward CV scores")
            S.grid(ax, axis="y")
    except Exception as e:
        log.debug("auto_plot failed: %s", e)
        # The figure was created before the call that raised, so nothing else
        # holds a reference to it and pyplot's registry would keep it alive for
        # the life of the process.
        if fig is not None:
            plt.close(fig)
        return None

    return fig


#: The report's own words in Chinese (lang="zh"); the figure's are the window's
#: catalogue's (normet.gui.ui_zh), read without Qt.
_REPORT_ZH = {
    "Plot": "图",
    "Result preview": "结果预览",
    "Provenance": "溯源信息",
    "Model": "模型",
    "No plot available.": "没有可用的图。",
    "Generated by normet on {at}.": "由 normet 于 {at} 生成。",
    "normet report — {kind}": "normet 报告——{kind}",
    "kind": "类型",
    "normet_version": "normet 版本",
    "python_version": "Python 版本",
    "platform": "平台",
    "host": "主机",
    "user": "用户",
    "timestamp": "时间",
    "seed": "随机种子",
    "data_hash": "数据哈希",
    "config_hash": "配置哈希",
    "type": "类型",
    "backend": "后端",
    "scale": "尺度",
    "Log-scale fit: the model was fitted on log1p(value) and every column of the result is back-transformed through expm1, so the numbers below are in the original units (and non-negative by construction).": "对数尺度拟合：模型在 log1p(value) 上训练，结果的每一列都经 expm1 反变换回原单位（按构造非负）。",
    " {share:.0%} of the observations sit below 1 in the original units — the low-concentration tail where a linear fit spends most of its error budget on the high values instead.": " 原单位下有 {share:.0%} 的观测低于 1——正是对数尺度所针对的低浓度尾部，线性拟合的误差预算大多花在了高值上。",
}


def _say(text: str, lang: str, **values: Any) -> str:
    """One of the report's own texts in ``lang``, its ``{name}`` filled from ``values``."""
    words = _REPORT_ZH.get(text, text) if lang == "zh" else text
    return words.format(**values) if values else words


def translate_figure(fig: Any, lang: str) -> None:
    """``fig``'s texts in ``lang`` -- titles, labels, legends and the date axis' words --
    as the window writes them (normet.gui.ui_zh, through the Qt-free catalogue engine),
    with a face that has the language's characters; a formula in a translated text is
    written out in sub- and superscripts, as mathtext has no Chinese. A text with no
    translation, and a figure where the machine has no such face, stay English."""
    if lang == "en" or fig is None:
        return
    import re

    from matplotlib.text import Text
    from matplotlib.ticker import Formatter

    from .gui import ui_zh
    from .gui._i18n import build_catalogue, figure_font, translate
    from .style import plain

    font = figure_font(lang)
    if font is None:
        return
    cat = build_catalogue(ui_zh.TEXTS, ui_zh.INNER, ui_zh.DATE_PARTS, sentences=False)

    def tr(text: str) -> str:
        if not re.search(r"[A-Za-z]{2}", re.sub(r"\$[^$]*\$", "", text)):
            return text
        out = translate(text, lang, cat)
        if out is None:
            return text
        return plain(out) if "$" in out else out

    class Words(Formatter):
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def __call__(self, x: Any, pos: Any = None) -> str:
            return tr(self.inner(x, pos))

        def format_ticks(self, values: Any) -> list[str]:
            return [tr(s) for s in self.inner.format_ticks(values)]

        def get_offset(self) -> str:
            return tr(self.inner.get_offset())

        def set_axis(self, axis: Any) -> None:
            super().set_axis(axis)
            self.inner.set_axis(axis)

        def set_locs(self, locs: Any) -> None:
            self.inner.set_locs(locs)

    for t in fig.findobj(Text):
        shown = t.get_text()
        new = tr(shown) if shown else shown
        if new != shown:
            t.set_text(new)
            families = list(t.get_fontfamily())
            if font not in families:
                t.set_fontfamily(families + [font])
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(Words(axis.get_major_formatter()))
            labels = axis.get_ticklabels(which="both")
            families = list(labels[0].get_fontfamily()) if labels else ["sans-serif"]
            if font not in families:
                axis.set_tick_params(which="both", labelfontfamily=families + [font])
            # The axis' offset (a date axis' "2024-Jan") is a text of its own.
            off = list(axis.offsetText.get_fontfamily())
            if font not in off:
                axis.offsetText.set_fontfamily(off + [font])


_TEMPLATE = """<!doctype html>
<html lang="{lang}"><head>
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

{scale_block}
<h2>{h_plot}</h2>
{plot_block}

<h2>{h_preview}</h2>
{table_block}

<h2>{h_provenance}</h2>
{meta_block}

{model_block}

<footer>
{footer}
</footer>

</body></html>
"""


def _low_value_share(result: Any) -> float | None:
    """The share of the run's observed series below 1 in the original units.

    That population is what the log scale is for: below 1, ``expm1`` is nearly
    linear, so a log fit's relative accuracy is exactly what a linear fit's
    absolute error budget would otherwise spend on the high values instead.
    None when the result carries no usable ``observed`` column.
    """
    if isinstance(result, pd.DataFrame) and "observed" in result.columns:
        obs = pd.to_numeric(result["observed"], errors="coerce")
        obs = obs[pd.notna(obs)]
        if len(obs):
            return float((obs < 1.0).mean())
    return None


def _log_scale_note(run: NormetRun, lang: str) -> str:
    """A paragraph saying the run was fitted on the log scale and back-transformed.

    The flag rides in the run's provenance -- ``metadata["config"]["log_transform"]``
    when the caller recorded its config, or ``metadata["log_transform"]`` -- and
    the note adds the share of low observations the scale exists for, when the
    result carries an observed column. Empty string when the flag is off, so a
    linear run's report is byte-for-byte what it was before.
    """
    meta = run.metadata or {}
    cfg = meta.get("config")
    flag = cfg.get("log_transform") if isinstance(cfg, dict) else meta.get("log_transform")
    if not flag:
        return ""
    text = _say(
        "Log-scale fit: the model was fitted on log1p(value) and every column of "
        "the result is back-transformed through expm1, so the numbers below are "
        "in the original units (and non-negative by construction).",
        lang,
    )
    share = _low_value_share(run.result)
    if share is not None:
        text += _say(
            " {share:.0%} of the observations sit below 1 in the original units — "
            "the low-concentration tail where a linear fit spends most of its "
            "error budget on the high values instead.",
            lang,
            share=share,
        )
    return f'<p class="nm-note">{html.escape(text)}</p>'


def generate_html(
    run: NormetRun,
    out_path: str | Path,
    *,
    title: str | None = None,
    extra_plots: list | None = None,
    lang: str = "en",
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
    lang : {"en", "zh"}, default "en"
        The report's language: its headings, labels and its figure's texts. Data's own
        words (column names, file names, the provenance record) stay as they are.

    Returns
    -------
    pathlib.Path
        Path to the written HTML file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(run.metadata or {})
    kind = meta.get("kind", "run")
    title = title or _say("normet report — {kind}", lang, kind=kind)

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
        f"<dt>{html.escape(_say(k, lang))}</dt><dd>{html.escape(str(meta.get(k, '—')))}</dd>"
        for k in kv_keys
    )

    # Plot
    fig = _auto_plot(run)
    plot_block_parts = []
    if fig is not None:
        translate_figure(fig, lang)
        try:
            plot_block_parts.append(
                f'<img class="nm-plot" src="{_fig_to_b64(fig)}" alt="auto plot">'
            )
        finally:
            # This figure was made here and has served its purpose as a PNG.
            # pyplot keeps every figure it creates alive until closed, so a
            # report generated per site in a loop otherwise accumulates them
            # all. Figures passed in through extra_plots belong to the caller
            # and are deliberately left alone.
            import matplotlib.pyplot as plt

            plt.close(fig)
    for extra in extra_plots or []:
        try:
            plot_block_parts.append(
                f'<img class="nm-plot" src="{_fig_to_b64(extra)}" alt="user plot">'
            )
        except Exception as e:
            log.warning("extra plot serialisation failed: %s", e)
    plot_block = "\n".join(plot_block_parts) or (
        f"<p><em>{html.escape(_say('No plot available.', lang))}</em></p>"
    )

    # Table preview
    table_block = _df_table(run.result)

    # Provenance
    meta_block = _json_pre(meta)

    # Model summary
    model_block = ""
    if run.model is not None:
        model_block = f"""
<h2>{_say("Model", lang)}</h2>
<dl class="kv">
  <dt>{_say("type", lang)}</dt><dd>{html.escape(type(run.model).__name__)}</dd>
  <dt>{_say("backend", lang)}</dt><dd>{html.escape(str(getattr(run.model, "backend", "—")))}</dd>
</dl>
"""

    body = _TEMPLATE.format(
        title=html.escape(title),
        kv_rows=kv_rows,
        scale_block=_log_scale_note(run, lang),
        plot_block=plot_block,
        table_block=table_block,
        meta_block=meta_block,
        model_block=model_block,
        lang="zh-CN" if lang == "zh" else "en",
        h_plot=_say("Plot", lang),
        h_preview=_say("Result preview", lang),
        h_provenance=_say("Provenance", lang),
        footer=_say(
            "Generated by normet on {at}.",
            lang,
            at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        ),
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
