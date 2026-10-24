"""The Stations tab's map: the found stations, clicked to pick one.

The station table answers "which sites are there?" by name and code; the map
answers "which of them is this one?" the way a scatter of coordinates does for
anyone who knows the area. Every provider carries coordinates -- the UK
archives' metadata, the EEA layer and OpenAQ locations alike -- so one view
serves all three networks.

There is no basemap. Tiles would mean an online service or a new dependency
for a picture whose points already sit where the stations are; the axes carry
the coordinates, and the aspect is set so a degree of longitude is drawn at
its true width for the map's mean latitude (a square-degree grid stretches
east-west by 1/cos(latitude) otherwise -- 66% too wide at the UK's ~53N).
"""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from normet import style as S

from ._widgets import ThemedToolbar, _canvas_class, adaptive_layout
from .theme import apply_figure_theme, theme_manager

#: How near the pointer, in screen pixels, a station must be to be picked or named.
PICK_PX = 10.0
#: Room left around the stations: this fraction of their span, and at least this many
#: degrees (one station alone has no span to take a fraction of).
PAD_FRAC = 0.04
PAD_MIN = 0.15


class StationMap(QWidget):
    """The stations of the current search, drawn by longitude and latitude.

    A left click on a station emits :attr:`stationClicked` with that row's index
    in the frame the map was given; the window selects it in the table, and the
    table's own selection rings the point here (:meth:`highlight`). Hovering
    names the station under the pointer. Zoom and pan are the figure toolbar's;
    clicks made while it is zooming or panning are left to it.

    The points are drawn in the house roles -- ``NEUTRAL`` for the list, the
    red ``MODELLED`` ring for the picked station -- which the dark theme leaves
    alone because they carry the data.
    """

    stationClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.figure = Figure(figsize=(5.0, 3.0), dpi=100)
        adaptive_layout(self.figure)
        self.ax = self.figure.add_subplot(111)

        self._frame: pd.DataFrame | None = None
        self._keys = np.zeros(0, dtype=np.int64)
        self._xy = np.zeros((0, 2), dtype=float)
        self._labels: list[str] = []
        self._selected: int | None = None
        self._press: tuple[float, float] | None = None
        self._hovered: int | None = None

        self._points = self.ax.scatter(
            np.zeros(0),
            np.zeros(0),
            s=18,
            c=S.NEUTRAL,
            alpha=0.85,
            edgecolors="none",
            zorder=2,
        )
        self._mark = self.ax.scatter(
            np.zeros(0),
            np.zeros(0),
            s=120,
            facecolors="none",
            edgecolors=S.MODELLED,
            linewidths=1.6,
            zorder=3,
        )
        self._tip = self.ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(9, 10),
            textcoords="offset points",
            fontsize=8,
            color=S.INK,
            zorder=5,
            visible=False,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#ffffff",
                "edgecolor": "#cccccc",
                "linewidth": 0.6,
            },
        )
        self._empty = self.ax.text(
            0.5,
            0.5,
            "Find stations to plot them here.",
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#333333",
        )
        self.ax.grid(True, color="#e5e5ea", linewidth=0.6, zorder=0)
        self.ax.margins(PAD_FRAC)
        self.ax.set_xlabel("Longitude", fontsize=8)
        self.ax.set_ylabel("Latitude", fontsize=8)
        self.ax.tick_params(labelsize=7)
        self._show_empty(True)

        self.canvas = _canvas_class()(self.figure)
        self.toolbar = ThemedToolbar(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("button_press_event", self._pressed)
        self.canvas.mpl_connect("button_release_event", self._released)
        self.canvas.mpl_connect("motion_notify_event", self._moved)
        self.canvas.mpl_connect("figure_leave_event", lambda _e: self._tip_off())
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)

        # In the window's language from the start (the English is kept), then the
        # theme that language's window is in.
        from .uilang import figure_texts

        figure_texts(self.figure)
        apply_figure_theme(self.figure, theme_manager().dark)
        theme_manager().themeChanged.connect(self._retheme)
        self._follow_language()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ content
    def set_stations(self, df: pd.DataFrame | None) -> None:
        """Draw *df* (one row per station, with ``lat``/``lon``); None empties it.

        The frame is the one the table is filled from, so a row's index here is
        the row's index in the window's station frame, filtered search included.
        """
        self._frame = df
        self._selected = None
        self._hovered = None
        self._mark.set_offsets(np.zeros((0, 2)))
        self._tip_off()
        if df is None or df.empty:
            self._keys = np.zeros(0, dtype=np.int64)
            self._xy = np.zeros((0, 2), dtype=float)
            self._labels = []
            self._points.set_offsets(np.zeros((0, 2)))
            self._show_empty(True)
            self.canvas.draw_idle()
            return

        coords = df[["lon", "lat"]].apply(pd.to_numeric, errors="coerce")
        ok = coords.notna().all(axis=1).to_numpy()
        self._xy = coords.to_numpy(dtype=float)[ok]
        self._keys = df.index.to_numpy()[ok].astype(np.int64)
        self._labels = [
            f"{site} ({code})" if code else str(site)
            for site, code in zip(df.loc[ok, "site"], df.loc[ok, "code"], strict=True)
        ]
        if not len(self._xy):
            self._points.set_offsets(np.zeros((0, 2)))
            self._show_empty(True)
            self.canvas.draw_idle()
            return

        self._points.set_offsets(self._xy)
        x, y = self._xy[:, 0], self._xy[:, 1]
        # A degree of longitude is shorter than a degree of latitude away from the
        # equator, so the aspect is 1/cos(latitude). Adjusting the data limits (not
        # the box) then widens the view to the panel's shape: a wide panel shows
        # more of the surrounding map rather than white bands beside a narrow one.
        mid = float(np.clip(np.mean(y), -80.0, 80.0))
        self.ax.set_aspect(1.0 / max(float(np.cos(np.deg2rad(mid))), 0.05), adjustable="datalim")
        if float(np.ptp(x)) == 0.0 or float(np.ptp(y)) == 0.0:
            # One station, or all on a line: no span to take a margin from.
            self.ax.set_xlim(float(x.mean()) - PAD_MIN, float(x.mean()) + PAD_MIN)
            self.ax.set_ylim(float(y.mean()) - PAD_MIN, float(y.mean()) + PAD_MIN)
        else:
            # The datalim adjustment reads ax.dataLim, which scatter's set_offsets
            # does not feed; hand it the points first, then let autoscale padding in.
            self.ax.update_datalim(self._xy)
            self.ax.autoscale_view()
        self._show_empty(False)
        self.canvas.draw_idle()

    def highlight(self, key: int | None) -> None:
        """Ring the station whose frame index is *key*, or none for None."""
        self._selected = key
        if key is None or not len(self._xy):
            self._mark.set_offsets(np.zeros((0, 2)))
        else:
            hit = np.flatnonzero(self._keys == key)
            self._mark.set_offsets(self._xy[hit] if hit.size else np.zeros((0, 2)))
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ internals
    def _show_empty(self, empty: bool) -> None:
        self._empty.set_visible(empty)
        self._mark.set_visible(not empty)
        if empty:
            self.ax.set_axis_off()
        else:
            self.ax.set_axis_on()

    def _interactive(self, event: Any) -> bool:
        """Whether *event* is over the map with no toolbar mode in the way."""
        toolbar = getattr(event.canvas, "toolbar", None)
        return (
            event.inaxes is self.ax
            and event.xdata is not None
            and not (toolbar is not None and getattr(toolbar, "mode", ""))
        )

    def _nearest(self, event: Any) -> int | None:
        """Index into the drawn points of the station within PICK_PX, or None."""
        if not len(self._xy) or event.x is None or event.y is None:
            return None
        here = self.ax.transData.transform(self._xy)
        d = np.hypot(here[:, 0] - event.x, here[:, 1] - event.y)
        i = int(np.argmin(d))
        return i if d[i] <= PICK_PX else None

    def _pressed(self, event: Any) -> None:
        self._press = (
            (event.x, event.y)
            if getattr(event, "button", None) == 1 and self._interactive(event)
            else None
        )

    def _released(self, event: Any) -> None:
        press, self._press = self._press, None
        if press is None or getattr(event, "button", None) != 1:
            return
        # A click, not the end of a drag: panning is the toolbar's.
        if abs(event.x - press[0]) > 3 or abs(event.y - press[1]) > 3:
            return
        i = self._nearest(event)
        if i is not None:
            self.stationClicked.emit(int(self._keys[i]))

    def _moved(self, event: Any) -> None:
        i = self._nearest(event) if self._interactive(event) else None
        if i == self._hovered:
            return
        self._hovered = i
        if i is None:
            self._tip_off()
            return
        self._tip.xy = (float(self._xy[i, 0]), float(self._xy[i, 1]))
        self._tip.set_text(self._labels[i])
        self._tip.set_visible(True)
        self.canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        self.canvas.draw_idle()

    def _tip_off(self) -> None:
        if self._hovered is not None or self._tip.get_visible():
            self._hovered = None
            self._tip.set_visible(False)
            self.canvas.setCursor(Qt.CursorShape.CrossCursor)
            self.canvas.draw_idle()

    def _retheme(self, dark: bool) -> None:
        apply_figure_theme(self.figure, dark)
        self.canvas.draw_idle()

    def _follow_language(self) -> None:
        """Redraw the axes' words on a language change; held weakly, like the tabs."""
        from . import uilang

        ref = weakref.ref(self)

        def changed(_code: str) -> None:
            w = ref()
            if w is None:
                return
            try:
                w._relanguage()
            except RuntimeError:  # its Qt side deleted already
                pass

        uilang.on_change(changed)

    def _relanguage(self) -> None:
        from .uilang import figure_texts

        figure_texts(self.figure)
        self.canvas.draw_idle()
