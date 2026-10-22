"""The window's trend step: one series, tested, said.

What the Trend tab shows: :func:`trend_call` runs :func:`normet.trend` on the
series the window holds (it is the command log's stand-in for that call, as
:mod:`._multiscale` is for the multi-scale one); :func:`auto_seasonal` says
what repeating calendar cycle a record's step carries; :func:`trend_verdict`
is the traffic-light banner, with the one sentence on which p to quote;
:func:`trend_series` builds its series the way the step does; and
:func:`trend_fig` draws the series and its robust line.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: Fewer values than this, there is no trend to test.
MIN_ROWS = 15


def _datetime_index(frame: pd.DataFrame) -> pd.DatetimeIndex | None:
    """The frame's time, as an index: its own where it is one, else its ``date``."""
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.index
    if "date" in frame.columns:
        return pd.to_datetime(frame["date"], errors="coerce")
    return None


def trend_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """One column of a result table as a time-indexed series.

    The table's own time where it carries one (a Datetime index or a ``date``
    column); the table's position order where it does not -- the test then
    runs in steps of the record, and says so.
    """
    if column not in frame.columns:
        raise ValueError(f"the table has no {column!r} column")
    s = frame[column].copy()
    if (idx := _datetime_index(frame)) is not None:
        s.index = idx
    s.name = column
    return s


def auto_seasonal(index: pd.DatetimeIndex, n: int) -> tuple[int | None, str]:
    """The repeating calendar cycle a record's step carries, in steps of it.

    Sub-daily records carry the diurnal cycle (24 steps for the hourly, 48
    for the 30-minute step...), daily ones the weekly, monthly ones the
    annual. Where the record is not a whole number of those periods the
    seasonal test cannot run on it and the classical one must -- the note
    says so where the fallback happened.
    """
    if n < MIN_ROWS:
        return None, ""
    med = pd.Series(index).diff().median()
    if pd.isna(med) or med <= pd.Timedelta(0):
        return None, ""
    seconds = float(med.total_seconds())
    if seconds <= 86400 * 0.25:
        period = max(2, int(round(86400 / seconds)))
        cycle = f"diurnal, {period} per day"
    elif seconds <= 86400 * 1.5:
        period, cycle = 7, "weekly"
    elif seconds <= 45 * 86400:
        period, cycle = 12, "annual"
    else:
        return None, ""
    if n % period:
        return None, (
            f"The record carries a {cycle} cycle of {period} steps but holds "
            f"{n % period} extra ones, so the seasonal test cannot run; the "
            "classical test ran instead."
        )
    return period, ""


def trend_call(
    frame: pd.DataFrame,
    column: str,
    *,
    series_name: str | None = None,
    seasonal_period: int | None = None,
    n_bootstrap: int = 1000,
    seed: int = 7_654_321,
) -> Any:
    """normet.trend on one column of a table the window holds.

    ``column`` is a table column or ``"long-term"``, the separated long-term
    component (:func:`long_term_series`). The task the window submits is this
    call with the table as it stands (the command log names the table by the
    expression it is kept under); the series is built the way the step builds
    it, so a saved session repeats the test exactly as it ran.
    """
    from normet import trend

    series = long_term_series(frame) if column == "long-term" else trend_series(frame, column)
    return trend(
        series,
        name=series_name or (None if series.name is None else str(series.name)),
        seasonal_period=seasonal_period,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def long_term_series(frame: pd.DataFrame) -> pd.Series:
    """The separated long-term component, as a level: the emission decomposition's
    ``date_unix`` contribution plus its base (``date_unix + emi_base``) -- the series
    the Trend (long-term) panel draws.

    It is the clean input to a trend test: the recurring calendar cycles (seasonal,
    weekly, diurnal) sit in their own component columns, so what is left is the drift
    plus one-off structural breaks, without the cycles inflating the residuals.
    """
    for col in ("date_unix", "emi_base"):
        if col not in frame.columns:
            raise ValueError(f"the table has no {col!r} column")
    s = (frame["date_unix"] + frame["emi_base"]).copy()
    if (idx := _datetime_index(frame)) is not None:
        s.index = idx
    s.name = "long-term trend (separated)"
    return s


def trend_breaks_call(
    frame: pd.DataFrame,
    column: str,
    *,
    series_name: str | None = None,
    max_segments: int = 2,
    n_bootstrap: int = 500,
    seed: int = 7_654_321,
    alpha: float = 0.05,
) -> Any:
    """normet.detect_trend_breaks on one column of a table the window holds.

    ``column`` is a table column or ``"long-term"``, the separated long-term
    component (:func:`long_term_series`). As with :func:`trend_call`, the task
    is this call with the table as it stands, so the command log names the
    table and a saved session repeats the segment fit exactly as it ran.
    """
    from normet import detect_trend_breaks

    series = long_term_series(frame) if column == "long-term" else trend_series(frame, column)
    return detect_trend_breaks(
        series,
        name=series_name or (None if series.name is None else str(series.name)),
        max_segments=int(max_segments),
        n_bootstrap=int(n_bootstrap),
        seed=seed,
        alpha=alpha,
    )


def _quoted(res: Any) -> float:
    """The p the result says to quote: the bootstrap's where it ran, the
    classical one where it did not."""
    if res.ar_autocorrelated and res.p_wald_boot is not None:
        return res.p_wald_boot
    return res.test.p


def trend_verdict(res: Any, name: str = "series") -> tuple[str, str]:
    """The Trend tab's banner: the finding, and which p stands for it."""
    slope = res.slope
    quoted = _quoted(res)
    per = "per day" if slope.time_unit == "day" else "per step"
    if res.ar_autocorrelated and res.p_wald_boot is None:
        return (
            "warn",
            f"{name}: AR(1) residual autocorrelation (acf {res.ar1_acf:.2f}) "
            "was found but the block bootstrap gave no usable interval -- the "
            "slope stands without an honest error bar; do not quote a trend "
            "from the record yet.",
        )
    ci = f", 95% CI [{slope.ci_low:.4g}, {slope.ci_high:.4g}]"
    if res.ar_autocorrelated and res.test.p < 0.05 and quoted >= 0.05:
        return (
            "warn",
            f"{name}: the classical Mann-Kendall test is significant "
            f"(p {res.test.p:.3g}) but the record is autocorrelated (AR(1) acf "
            f"{res.ar1_acf:.2f}) and the block-bootstrap correction does not "
            f"confirm it (p {quoted:.3g}). Do not quote a trend from the record "
            "yet.",
        )
    if res.ar_autocorrelated and quoted < 0.05:
        also = (
            " and the block-bootstrap-corrected test"
            if res.test.p < 0.05
            else " that the classical test missed (MK p "
            f"{res.test.p:.3g}); the block bootstrap found it"
        )
        return (
            "ok",
            f"{name}: significant {res.test.direction} trend{also}, slope "
            f"{slope.slope:.4g} {per}{ci}; the corrected p is {quoted:.3g}, the "
            "one to quote.",
        )
    if quoted < 0.05:
        extra = (
            f", MK p {res.test.p:.3g}" if res.ar_autocorrelated else (f" (MK p {res.test.p:.3g})")
        )
        return (
            "ok",
            f"{name}: significant {res.test.direction} trend, slope "
            f"{slope.slope:.4g} {per}{ci}, p {quoted:.3g}{extra}.",
        )
    return (
        "warn",
        f"{name}: no significant trend at the 5% level (p {quoted:.3g}); "
        f"slope {slope.slope:.4g} {per}{ci}.",
    )


def _break_index_value(value: Any) -> Any:
    """A break point the figure can draw: its timestamp where the index holds one,
    the positional step where the record has no calendar time to place it on."""
    import numpy as _np

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if hasattr(value, "item"):
        return _np.atleast_1d(value).item()
    return value


def trend_breaks_verdict(res: Any, name: str = "series") -> tuple[str, str]:
    """The Trend tab's banner for a segment fit: whether the search found a break,
    where, and which p stands for it; with the gates' numbers where it did not,
    so the record can say why it was kept as one regime."""
    if res.n_segments > 1:
        breaks = ", ".join(f"{v} ({_fmt_break(v)})" for v in res.starts[1:])
        return (
            "ok",
            f"{name}: the trend breaks -- {res.n_segments} segments, at {breaks}. "
            f"The block bootstrap gives p={res.p_value:.3g} < {res.alpha} over "
            f"{res.n_bootstrap} resamples (block length {res.block_length}), the "
            "p to quote; the split cuts the single-slope SSE by "
            f"{res.improvement_pct:.1f}%. The segment table gives each Theil-Sen "
            "slope and its interval; compare adjacent segments on those.",
        )
    p = res.p_value if res.p_value is not None else float("nan")
    where = res.best_split_pct
    return (
        "warn",
        f"{name}: no break the gates accept for {res.max_segments} segments -- "
        "one regime reported. The search's best cut sits at "
        f"{_fmt_break(res.best_break)} and would cut the single-slope SSE by "
        f"{where:.1f}%, but the same search on {res.n_bootstrap} block-bootstrapped "
        f"resamples reaches at least that with p={p:.3g} >= {res.alpha}, so it is "
        "noise, not a regime change; the record trends as one slope (the table's).",
    )


def _fmt_break(value: Any) -> str:
    """A break point, spoken the way a report would: date to the minute, or step."""
    try:
        from normet.analysis.trend import _fmt_index as _fmt

        return _fmt(value)
    except Exception:  # noqa: BLE001 - a value the form does not know is still spoken
        return str(value)


def _segment_lines(series: pd.Series, res: Any) -> list[tuple[Any, np.ndarray, float]]:
    """Each segment as a line to draw: (its points on the series' own index, the
    Theil-Sen fit through them, its slope) -- each line through the segment's own
    mean at the middle of its span, so the pieces meet where the cut was made."""
    out: list[tuple[Any, np.ndarray, float]] = []
    data = series.dropna()
    if data.size < 2:
        return out
    time = (
        pd.Series((data.index - data.index[0]).total_seconds() / 86400.0, index=data.index)
        if isinstance(data.index, pd.DatetimeIndex)
        else pd.Series(data.index.to_numpy(dtype=float), index=data.index)
    )
    for start, end, seg in zip(res.starts, res.ends, res.segments, strict=True):
        try:
            mask = (data.index >= start) & (data.index <= end)
        except TypeError:
            continue  # the series' index and the result's cuts do not share a type
        part = data[mask]
        if len(part) < 2:
            continue
        d = time.loc[part.index].to_numpy(dtype=float)
        mid = float(part.to_numpy(dtype=float).mean())
        out.append((part.index, mid + float(seg.slope) * (d - d.mean()), float(seg.slope)))
    return out


def trend_breaks_fig(series: pd.Series, res: Any) -> Any:
    """The series with its segments: one piecewise Theil-Sen line per segment and a
    marked break where they meet -- the segment slopes and their intervals are in
    the table, as in :func:`trend_fig`.

    The lines are per-segment: a single line would dilute the level shift into its
    slope, and the markers place each break on the calendar the record is indexed by.
    """
    import matplotlib.pyplot as plt

    from normet import style as S

    fig, ax = plt.subplots(figsize=(8, 3))
    S.line(
        ax, series.index, series.to_numpy(), lw=0.6, color=S.OBSERVED, alpha=0.55, label="series"
    )
    lines = _segment_lines(series, res)
    per = "per day" if res.time_unit == "day" else "per step"
    labels = (
        [f"segment {i} ({s.slope:+.4g} {per})" for i, s in enumerate(res.segments, 1)]
        if res.n_segments > 1
        else [f"Theil-Sen slope ({res.segments[0].slope:.4g} {per})"]
    )
    for (idx, y, _slope), name in zip(lines, labels, strict=False):
        S.line(ax, idx, y, lw=1.4, color=S.MODELLED, label=name)
    if res.n_segments > 1:
        mid_level = float(np.nanmean(series.to_numpy()))
        for start in res.starts[1:]:
            at = _break_index_value(start)
            ax.axvline(at, color=S.REFERENCE, ls="--", lw=0.7)
            ax.annotate(
                f"break\n{_fmt_break(start)}",
                xy=(at, mid_level),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
            )
    if lines:
        S.legend(ax, loc="upper right", ncols=2)
    ax.margins(x=0)
    S.grid(ax)
    S.format_time_axis(ax)
    fig.tight_layout()
    return fig


def trend_fig(series: pd.Series, res: Any) -> Any:
    """The series with its robust line.

    The line is the Theil-Sen slope through the series' own mean at the
    middle of the record. Its interval is in the table, not drawn: it is on
    the slope, and a band on the figure could not say the difference.
    """
    import matplotlib.pyplot as plt

    from normet import style as S

    fig, ax = plt.subplots(figsize=(8, 3))
    S.line(
        ax, series.index, series.to_numpy(), lw=0.6, color=S.OBSERVED, alpha=0.55, label="series"
    )
    if len(series) >= 2 and series.dropna().size >= MIN_ROWS:
        if isinstance(series.index, pd.DatetimeIndex):
            d = (series.index - series.index[0]).total_seconds() / 86400.0
        else:
            d = series.index.to_numpy(dtype=float)
        d = np.asarray(d, dtype=float)
        mid = float(series.mean())
        slope = float(res.slope.slope)
        per = "per day" if res.slope.time_unit == "day" else "per step"
        S.line(
            ax,
            series.index,
            mid + slope * (d - d.mean()),
            lw=1.4,
            color=S.MODELLED,
            label=f"Theil–Sen slope ({slope:.4g} {per})",
        )
    ax.margins(x=0)
    S.grid(ax)
    S.legend(ax, loc="upper right", ncols=2)
    S.format_time_axis(ax)
    fig.tight_layout()
    return fig
