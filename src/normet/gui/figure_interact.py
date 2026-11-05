"""The mouse on a result figure: what is under it said at the figure's lower left.

normet's data overview reads the sample under the cursor (_data_tab's `_hover`); this
does the same for the result figures -- the decomposition's component panels, the
meteorology influence panels and ranking, the trend's series and fit. The plot functions
note what a figure holds on it (`_figure_note` below); this module turns a mouse event
into the words for it.

The read-out is a label the result tab places at the figure's lower left (_widgets'
CanvasTab); the tab wires `motion_notify_event` and `figure_leave_event` to it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _figure_note(fig: Any, payload: Any) -> None:
    """The figure's mouse read-out note (`fig._normet`), set where a figure is drawn --
    matplotlib's Figure takes no type on its attributes."""
    fig._normet = payload          # noqa: SLF001


def _fmt(v: Any) -> str:
    """A value in the read-out's words: four significant figures, or a dash for none."""
    v = float(v)
    return "—" if not np.isfinite(v) else f"{v:.4g}"


def _at(values: Any, i: int) -> Any:
    """The i-th value of a series, by position (a pandas Series keys by label)."""
    if isinstance(values, pd.Series):
        return values.iloc[i]
    return values[i]


def _time(t: Any) -> str:
    """A record's index value as the read-out writes it."""
    if isinstance(t, (pd.Timestamp,)):
        return t.strftime("%Y-%m-%d %H:%M")
    if isinstance(t, (np.floating, float, int)):
        return _fmt(t)
    return str(t)


def _nearest(index: Any, xdata: Any) -> int | None:
    """The record's point nearest `xdata`, or None off the ends."""
    n = len(index)
    if n == 0:
        return None
    if isinstance(index, pd.DatetimeIndex):
        try:
            t = pd.Timestamp(xdata)
        except (ValueError, TypeError):
            return None
        i = int(np.argmin(np.abs((index - t).total_seconds())))
    elif isinstance(index, pd.RangeIndex):
        i = int(np.clip(round(float(xdata)), 0, n - 1))
    else:
        try:
            arr = index.to_numpy(dtype=float)
        except (ValueError, TypeError):
            return None
        i = int(np.argmin(np.abs(arr - float(xdata))))
    return i if 0 <= i < n else None


def readout(fig: Any, event: Any) -> str:
    """What is under the cursor on `fig`, in the window's words ("", nothing to say).

    The figure's note (`_figure_note`) names what it holds; the event's axes and data
    coordinates pick out the time and the series' values, or the bar under the pointer.
    """
    note = getattr(fig, "_normet", None)
    if not note or event.inaxes is None or event.xdata is None:
        return ""
    ax = event.inaxes

    for a, index, series in note.get("series", []):
        if ax is not a:
            continue
        i = _nearest(index, event.xdata)
        if i is None:
            return ""
        when = _time(index[i])
        parts = ", ".join(f"{label} {_fmt(_at(values, i))}" for label, values in series
                          if i < len(values))
        return f"{when} · {parts}" if parts else when

    for a, names, values, unit in note.get("bars", []):
        if ax is not a:
            continue
        j = int(round(event.ydata))
        if not (0 <= j < len(names)):
            return ""
        return f"{names[j]} · {unit} {_fmt(values[j])}"

    return ""
