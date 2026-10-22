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

    The task the window submits is this call with the table as it stands (the
    command log names the table by the expression it is kept under); the
    series is built the way the step builds it, so a saved session repeats
    the test exactly as it ran.
    """
    from normet import trend

    return trend(
        trend_series(frame, column),
        name=series_name or column,
        seasonal_period=seasonal_period,
        n_bootstrap=n_bootstrap,
        seed=seed,
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
