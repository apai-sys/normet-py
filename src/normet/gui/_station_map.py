"""The Stations tab's map: the found stations, clicked, boxed or picked.

The station table answers "which sites are there?" by name and code; the map
answers "which of them is this one?" the way a point on the ground does for
anyone who knows the area. Every provider carries coordinates -- the UK
archives' metadata, the EEA layer and OpenAQ locations alike -- so one view
serves all three networks.

The points are drawn in Web Mercator on a basemap (:mod:`._basemap`): a tile
is then a square of equal degrees in both directions, so the coastlines and
the stations line up exactly. Tiles are fetched after the points are on
screen, only for the view asked for, and again when the view is zoomed or
panned; when they cannot be had (offline, no ``requests``) the map falls back
to the same points on a plain grid, which is what it was before the basemap.

Picking
-------
A click selects the station under the pointer; a drag draws a box and selects
every station inside it; Command/Control-click adds a station to the
selection (or takes it out). The selection is a set of frame indices handed to
the window, which mirrors it in the table and fetches every station in it.
The hand button turns drags into pans, and the wheel zooms about the pointer.
"""

from __future__ import annotations

import math
import weakref
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from normet import style as S

from . import _basemap
from ._widgets import _canvas_class, adaptive_layout
from .theme import apply_figure_theme, theme_manager

#: How near the pointer, in screen pixels, a station must be to be picked.
PICK_PX = 10.0
#: Room left around the stations: this fraction of their span, and at least this many
#: degrees (one station alone has no span to take a fraction of).
PAD_FRAC = 0.04
PAD_MIN = 0.15
#: Each click of the wheel, and each press of + or -, changes the span by this factor.
ZOOM_STEP = 1.5
#: The closest the view may zoom in, in degrees across (a city block).
MIN_SPAN = 0.004
#: How long a view change waits for the next before its tiles are fetched.
TILE_DELAY_MS = 350


class StationMap(QWidget):
    """The stations of the current search, drawn on their streets.

    ``selectionChanged`` carries the whole selection as frame indices, whenever
    a click, a box or an additive click changes it; the window mirrors it in
    the table and the table's own selection comes back via :meth:`highlight`.
    Zoom, pan and the basemap are this widget's own controls.
    """

    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.figure = Figure(figsize=(5.0, 3.0), dpi=100)
        adaptive_layout(self.figure)
        self.ax = self.figure.add_subplot(111)

        self._frame: pd.DataFrame | None = None
        self._keys = np.zeros(0, dtype=np.int64)
        self._xy = np.zeros((0, 2), dtype=float)
        self._labels: list[str] = []
        self._selected: list[int] = []
        self._bbox_ll: tuple[float, float, float, float] | None = None
        self._fit_limits: tuple[float, float, float, float] | None = None
        self._bm_token = 0
        self._extent: tuple[float, float, float, float] | None = None
        self._press: tuple[float, float] | None = None
        self._pan_from: tuple[float, float] | None = None
        self._pan_mode = False
        self._hovered: int | None = None
        self._band: Any = None

        self._bg = self.ax.imshow(
            np.zeros((1, 1, 3), dtype=np.uint8),
            extent=(0.0, 1.0, 0.0, 1.0),
            interpolation="bilinear",
            zorder=0,
        )
        self._bg.set_visible(False)
        # Attribution is a condition of the tile providers' free use.
        self._credit = self.ax.text(
            0.995,
            0.004,
            "",
            transform=self.ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color="#333333",
            zorder=6,
            visible=False,
        )
        self._credit.set_url("https://www.openstreetmap.org/copyright")

        self._points = self.ax.scatter(
            np.zeros(0),
            np.zeros(0),
            s=20,
            c=S.NEUTRAL,
            alpha=0.9,
            # A white halo keeps a grey point readable on a photo basemap as well
            # as on the pale ones.
            edgecolors="#ffffff",
            linewidths=0.6,
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

        # The axes run in Web Mercator (a tile is then a square of equal degrees), so
        # the y ticks are drawn at their latitudes again.
        def lat_label(v: float, _pos: Any) -> str:
            # Enough decimals for the ticks to differ: the step between them is
            # about a fifth of the span.
            span = abs(self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
            step = span / 5.0
            decimals = 0 if step >= 1.0 else int(min(5, math.ceil(-math.log10(step))))
            return f"{float(_basemap.inverse_mercator(v)):.{decimals}f}"

        self.ax.yaxis.set_major_formatter(FuncFormatter(lat_label))
        self.ax.tick_params(labelsize=7)
        self._show_empty(True)

        self.canvas = _canvas_class()(self.figure)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self._controls())
        lay.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("button_press_event", self._pressed)
        self.canvas.mpl_connect("button_release_event", self._released)
        self.canvas.mpl_connect("motion_notify_event", self._moved)
        self.canvas.mpl_connect("scroll_event", self._scrolled)
        self.canvas.mpl_connect("key_press_event", self._key)
        self.canvas.mpl_connect("figure_leave_event", lambda _e: self._tip_off())
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        # The gestures are many and invisible; say them where the pointer rests.
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas.setToolTip(
            "Click a station to select it · drag a box to select several ·\n"
            "⌘/Ctrl-click to add or remove one · double-click to zoom to it ·\n"
            "wheel to zoom · middle-drag (or ✋) to pan · right-click for more."
        )

        # In the window's language from the start (the English is kept), then the
        # theme that language's window is in.
        from .uilang import figure_texts

        figure_texts(self.figure)
        apply_figure_theme(self.figure, theme_manager().dark)
        theme_manager().themeChanged.connect(self._retheme)
        self._follow_language()
        self._update_info()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ controls
    def _controls(self) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(QLabel("Basemap"))
        self.base_combo = QComboBox()
        self.base_combo.addItems(list(_basemap.PROVIDERS))
        self.base_combo.setToolTip(
            "The map under the stations: OpenStreetMap's standard detail, Esri's\n"
            "muted light canvas, or satellite imagery. All three need no key; the\n"
            "attribution under the map is theirs."
        )
        preferred = _basemap.DEFAULT_DARK if theme_manager().dark else _basemap.DEFAULT_LIGHT
        self.base_combo.setCurrentText(preferred)
        self.base_combo.currentTextChanged.connect(lambda _t: self._schedule_tiles(False))
        h.addWidget(self.base_combo)
        for text, tip, slot in (
            ("−", "Zoom out (−), about the middle of the map", lambda: self._zoom(1 / ZOOM_STEP)),
            ("+", "Zoom in (+)", lambda: self._zoom(ZOOM_STEP)),
            ("Fit map", "Fit the stations back in view (0)", self._fit_view),
        ):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        self.pan_btn = QToolButton()
        self.pan_btn.setText("✋")
        self.pan_btn.setCheckable(True)
        self.pan_btn.setToolTip(
            "Pan the map: drags move it instead of selecting.\n"
            "A middle-button drag pans without it."
        )
        self.pan_btn.toggled.connect(self._set_pan)
        h.addWidget(self.pan_btn)
        self.clear_btn = QToolButton()
        self.clear_btn.setText("Deselect")
        self.clear_btn.setToolTip("Empty the selection (Esc).")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(lambda: self.selectionChanged.emit([]))
        h.addWidget(self.clear_btn)
        h.addStretch(1)
        # What the map holds, and what is picked: the answer to "did my box take?"
        self.info = QLabel("")
        self.info.setStyleSheet("color: palette(placeholder-text);")
        h.addWidget(self.info)
        return row

    def _set_pan(self, on: bool) -> None:
        self._pan_mode = on
        self.canvas.setCursor(Qt.CursorShape.OpenHandCursor if on else Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------------ content
    def set_stations(self, df: pd.DataFrame | None) -> None:
        """Draw *df* (one row per station, with ``lat``/``lon``); None empties it.

        The frame is the one the table is filled from, so a row's index here is
        the row's index in the window's station frame, filtered search included.
        """
        self._frame = df
        self._selected = []
        self._hovered = None
        self._mark.set_offsets(np.zeros((0, 2)))
        self._tip_off()
        if df is None or df.empty:
            self._keys = np.zeros(0, dtype=np.int64)
            self._xy = np.zeros((0, 2), dtype=float)
            self._labels = []
            self._bbox_ll = None
            self._fit_limits = None
            self._points.set_offsets(np.zeros((0, 2)))
            self._update_info()
            self._show_empty(True)
            self.canvas.draw_idle()
            return

        coords = df[["lon", "lat"]].apply(pd.to_numeric, errors="coerce")
        ok = coords.notna().all(axis=1).to_numpy()
        ll = coords.to_numpy(dtype=float)[ok]
        self._ll = ll  # lon/lat as given, for the copy action
        self._xy = (
            np.column_stack([ll[:, 0], _basemap.mercator(ll[:, 1])])
            if len(ll)
            else np.zeros((0, 2))
        )
        self._keys = df.index.to_numpy()[ok].astype(np.int64)
        self._labels = [
            f"{site} ({code})" if code else str(site)
            for site, code in zip(df.loc[ok, "site"], df.loc[ok, "code"], strict=True)
        ]
        if not len(self._xy):
            self._points.set_offsets(np.zeros((0, 2)))
            self._update_info()
            self._show_empty(True)
            self.canvas.draw_idle()
            return

        self._points.set_offsets(self._xy)
        # The box the tiles must cover: the stations and a margin, in lon/lat.
        lo, la = ll[:, 0], ll[:, 1]
        pad_x = max(PAD_FRAC * float(np.ptp(lo)), PAD_MIN)
        pad_y = max(PAD_FRAC * float(np.ptp(la)), PAD_MIN)
        self._bbox_ll = (
            float(lo.min()) - pad_x,
            float(lo.max()) + pad_x,
            float(la.min()) - pad_y,
            float(la.max()) + pad_y,
        )
        self._fit()
        self._update_info()
        self._show_empty(False)
        self._bg.set_visible(False)  # the previous search's tiles are stale
        self._credit.set_visible(False)
        self.canvas.draw_idle()
        # Tiles need the network: fetched after this draw, so the points appear at once
        # and a slow (or absent) tile server never holds the window up.
        self._schedule_tiles(True)

    def highlight(self, keys: list[int] | None) -> None:
        """Ring the stations whose frame indices are in *keys*."""
        self._selected = [int(k) for k in (keys or [])]
        if not self._selected or not len(self._xy):
            self._mark.set_offsets(np.zeros((0, 2)))
        else:
            hit = np.isin(self._keys, self._selected)
            self._mark.set_offsets(self._xy[hit] if hit.any() else np.zeros((0, 2)))
        self._update_info()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ view
    def _fit(self) -> None:
        """Put the stations back in view at the scale the search chose."""
        if self._bbox_ll is None or not len(self._xy):
            return
        x0, x1, lat0, lat1 = self._bbox_ll
        if lat1 == lat0:  # one station, or all on a line: no span to pad from
            lat0, lat1 = lat0 - PAD_MIN, lat1 + PAD_MIN
        y0, y1 = float(_basemap.mercator(lat0)), float(_basemap.mercator(lat1))
        self._fit_limits = (x0, x1, y0, y1)
        # The map is conformal in Web Mercator: equal degrees each way is the true
        # shape, so the axes widen one side to the panel's shape (datalim). That
        # adjustment reads the axes' data limits, which scatter's set_offsets does
        # not feed -- hand it the box, and let autoscale's margins pad it.
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.dataLim.set_points(np.array([[x0, y0], [x1, y1]]))
        # Zooming set the limits explicitly, which turns autoscaling off for the
        # axis; fitting again has to turn it back on before it can act.
        self.ax.set_autoscalex_on(True)
        self.ax.set_autoscaley_on(True)
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _zoom(self, factor: float, at_px: tuple[float, float] | None = None) -> None:
        """Zoom by *factor* (>1 in) about the pointer, or the middle of the view."""
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        span_x, span_y = (x1 - x0) / factor, (y1 - y0) / factor
        if span_x < MIN_SPAN or span_x > 360.0:
            return
        if at_px is None:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            fx = fy = 0.5
        else:
            cx, cy = self.ax.transData.inverted().transform(at_px)
            fx = (cx - x0) / (x1 - x0)
            fy = (cy - y0) / (y1 - y0)
        self.ax.set_xlim(cx - fx * span_x, cx + (1.0 - fx) * span_x)
        self.ax.set_ylim(cy - fy * span_y, cy + (1.0 - fy) * span_y)
        self._view_changed()

    def _scrolled(self, event: Any) -> None:
        if not len(self._xy) or event.inaxes is not self.ax or not event.step:
            return
        self._zoom(ZOOM_STEP**event.step, (float(event.x), float(event.y)))

    def _view_changed(self) -> None:
        """After a zoom or pan: redraw, and fetch the tiles for the new view."""
        if self._extent is not None:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            ex0, ex1, ey0, ey1 = self._extent
            if x0 < ex0 or x1 > ex1 or y0 < ey0 or y1 > ey1:
                # The image no longer covers the view: a stretched corner of it
                # would be a blur, so show the plain map until its own tiles come.
                self._bg.set_visible(False)
                self._credit.set_visible(False)
        self.canvas.draw_idle()
        self._schedule_tiles(False)

    def _visible_bbox(self) -> tuple[float, float, float, float]:
        """The view in lon/lat, for the tiles that must cover it."""
        inv = self.ax.transData.inverted()
        # The axes' *display* box: get_position() is in figure (0..1) units and
        # would transform into a degenerate box near the origin.
        box = self.ax.get_window_extent()
        (x0, y0), (x1, y1) = inv.transform([(box.x0, box.y0), (box.x1, box.y1)])
        return (
            float(min(x0, x1)),
            float(max(x0, x1)),
            float(_basemap.inverse_mercator(min(y0, y1))),
            float(_basemap.inverse_mercator(max(y0, y1))),
        )

    # ------------------------------------------------------------------ basemap
    def _schedule_tiles(self, fit: bool) -> None:
        """Fetch tiles for the view after it has settled (a wheel is many events)."""
        self._bm_token += 1
        token = self._bm_token
        QTimer.singleShot(TILE_DELAY_MS, self, lambda: self._load_basemap(fit, token))

    def _load_basemap(self, fit: bool = True, token: int | None = None) -> None:
        """Fetch and show the tiles under the current view (or the search's box)."""
        if token is None:
            token = self._bm_token
        if not len(self._xy):
            return
        box = self.ax.get_window_extent()
        provider = self.base_combo.currentText()
        bbox = self._bbox_ll if fit or self._bbox_ll is None else self._visible_bbox()
        if bbox is None:
            return
        mosaic = _basemap.tile_mosaic(
            *bbox,
            width_px=max(box.width, 200),
            height_px=max(box.height, 120),
            provider=provider,
        )
        if token != self._bm_token:  # the view moved on while the tiles came
            return
        if mosaic is None:  # offline, or no requests/Pillow: the plain map stays
            self._bg.set_visible(False)
            self._credit.set_visible(False)
            self.canvas.draw_idle()
            return
        image, extent = mosaic
        self._extent = extent
        self._bg.set_data(image)
        self._bg.set_extent(extent)
        self._bg.set_visible(True)
        self._credit.set_text(_basemap.PROVIDERS[provider]["credit"])
        self._credit.set_visible(True)
        if fit:
            self._view_inside(extent)
        self.canvas.draw_idle()

    def _view_inside(self, extent: tuple[float, float, float, float]) -> None:
        """Show as much of the tile image as the panel's shape allows.

        The tiles come in whole squares, so their extent rarely matches the panel:
        cropping inside it (rather than leaving bands beside it) keeps the map
        filling the space, and the crop is clamped so no blank edge is exposed.
        """
        x0, x1, y0, y1 = extent
        width, height = x1 - x0, y1 - y0
        # The aspect ratio that matters is the axes' own box -- labels, ticks and the
        # toolbar take their share of the panel -- and it is known once drawn.
        box = self.ax.get_window_extent()
        panel = box.width / box.height if box.height > 0 else 1.0
        span_x, span_y = width, height
        if width / height < panel:
            span_y = width / panel
        else:
            span_x = height * panel
        cx = float(np.mean(self._xy[:, 0]))
        cy = float(np.mean(self._xy[:, 1]))
        cx = float(np.clip(cx, x0 + span_x / 2, x1 - span_x / 2))
        cy = float(np.clip(cy, y0 + span_y / 2, y1 - span_y / 2))
        self.ax.set_xlim(cx - span_x / 2, cx + span_x / 2)
        self.ax.set_ylim(cy - span_y / 2, cy + span_y / 2)

    # ------------------------------------------------------------------ internals
    def _update_info(self) -> None:
        """The count under the controls: how many stations, and how many picked."""
        total = len(self._keys) if hasattr(self, "_keys") else 0
        if not total:
            self.info.setText("")
        else:
            picked = len(self._selected)
            self.info.setText(
                f"{total} stations" + (f" · {picked} selected" if picked else " · none selected")
            )
        self.clear_btn.setEnabled(bool(self._selected))

    def _show_empty(self, empty: bool) -> None:
        self._empty.set_visible(empty)
        self._mark.set_visible(not empty)
        if empty:
            self._bg.set_visible(False)
            self._credit.set_visible(False)
            self.ax.set_axis_off()
        else:
            self.ax.set_axis_on()

    def _over_axes(self, event: Any) -> bool:
        return event.inaxes is self.ax and event.xdata is not None

    def _nearest(self, event: Any) -> int | None:
        """Index into the drawn points of the station within PICK_PX, or None."""
        if not len(self._xy) or event.x is None or event.y is None:
            return None
        here = self.ax.transData.transform(self._xy)
        d = np.hypot(here[:, 0] - event.x, here[:, 1] - event.y)
        i = int(np.argmin(d))
        return i if d[i] <= PICK_PX else None

    def _in_box(self, x0: float, x1: float, y0: float, y1: float) -> list[int]:
        inside = (
            (self._xy[:, 0] >= min(x0, x1))
            & (self._xy[:, 0] <= max(x0, x1))
            & (self._xy[:, 1] >= min(y0, y1))
            & (self._xy[:, 1] <= max(y0, y1))
        )
        return [int(k) for k in self._keys[inside]]

    def _pressed(self, event: Any) -> None:
        button = getattr(event, "button", None)
        if button == 3 and self._over_axes(event):  # right-click: the menu
            self._context(event)
            return
        if button == 2 and self._over_axes(event):  # middle-drag pans either way
            self._pan_from = (event.x, event.y)
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if button != 1 or not self._over_axes(event):
            self._press = None
            return
        if getattr(event, "dblclick", False):  # double-click: close in on it
            i = self._nearest(event)
            if i is not None:
                self.zoom_to(int(self._keys[i]))
            return
        self._press = (event.x, event.y)
        if self._pan_mode:
            self._pan_from = (event.x, event.y)
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        self._band = self.ax.add_artist(
            Rectangle(
                (event.xdata, event.ydata),
                0,
                0,
                facecolor=S.MODELLED,
                alpha=0.12,
                edgecolor=S.MODELLED,
                lw=1.0,
                ls="--",
                zorder=4,
            )
        )
        self._band.set_visible(False)

    def _moved(self, event: Any) -> None:
        if self._pan_from is not None and event.x is not None:
            self._pan(event)
            return
        if self._press is not None and self._band is not None:
            dx = abs(event.x - self._press[0])
            dy = abs(event.y - self._press[1])
            if dx > 3 or dy > 3:
                x0, y0 = self.ax.transData.inverted().transform(self._press)
                self._band.set_xy((x0, y0))
                self._band.set_width(event.xdata - x0)
                self._band.set_height(event.ydata - y0)
                self._band.set_visible(True)
                self.info.setText(
                    f"{len(self._in_box(x0, y0, event.xdata, event.ydata))} in the box"
                )
                self.canvas.draw_idle()
                return
        i = self._nearest(event) if self._over_axes(event) else None
        if i == self._hovered:
            return
        self._hovered = i
        if i is None:
            self._tip_off()
            return
        self._tip.xy = (float(self._xy[i, 0]), float(self._xy[i, 1]))
        self._tip.set_text(self._labels[i])
        # Sit the label on the side with the room: a name at the right edge would
        # otherwise run out of the map.
        box = self.ax.get_window_extent()
        near_right = event.x is not None and event.x > box.x0 + box.width * 0.65
        self._tip.set_horizontalalignment("right" if near_right else "left")
        self._tip.set_position((-9 if near_right else 9, 10))
        self._tip.set_visible(True)
        if not self._pan_mode:
            self.canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        self.canvas.draw_idle()

    def _pan(self, event: Any) -> None:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        box = self.ax.get_window_extent()
        dx = (event.x - self._pan_from[0]) * (x1 - x0) / max(box.width, 1)
        dy = (event.y - self._pan_from[1]) * (y1 - y0) / max(box.height, 1)
        self._pan_from = (event.x, event.y)
        self.ax.set_xlim(x0 - dx, x1 - dx)
        self.ax.set_ylim(y0 - dy, y1 - dy)
        self.canvas.draw_idle()

    def _released(self, event: Any) -> None:
        press, self._press = self._press, None
        if getattr(event, "button", None) == 2 and self._pan_from is not None:
            self._pan_from = None
            self.canvas.setCursor(
                Qt.CursorShape.OpenHandCursor if self._pan_mode else Qt.CursorShape.CrossCursor
            )
            self._view_changed()
            return
        if self._pan_from is not None:
            self._pan_from = None
            self.canvas.setCursor(Qt.CursorShape.OpenHandCursor)
            self._view_changed()
            return
        if press is None or getattr(event, "button", None) != 1:
            return
        band, self._band = self._band, None
        if band is None:
            return
        dragged = abs(event.x - press[0]) > 3 or abs(event.y - press[1]) > 3
        additive = self._additive()
        if dragged:
            x0, y0 = self.ax.transData.inverted().transform(press)
            picked = self._in_box(x0, event.xdata, y0, event.ydata)
            band.remove()
            keys = sorted(set(self._selected) | set(picked)) if additive else picked
            self.selectionChanged.emit(keys)
            return
        band.remove()
        i = self._nearest(event)
        if i is None:
            if not additive:
                self.selectionChanged.emit([])  # a click on empty space clears
            return
        key = int(self._keys[i])
        if additive:
            keys = [k for k in self._selected if k != key]
            if key not in self._selected:
                keys.append(key)
        else:
            keys = [key]
        self.selectionChanged.emit(sorted(keys))

    def _key(self, event: Any) -> None:
        """Esc empties the selection, +/- zoom, 0 (or F) fits."""
        key = str(getattr(event, "key", "") or "").lower()
        if key == "escape":
            self.selectionChanged.emit([])
        elif key in ("+", "="):
            self._zoom(ZOOM_STEP)
        elif key == "-":
            self._zoom(1 / ZOOM_STEP)
        elif key in ("0", "f", "home"):
            self._fit_view()

    def _fit_view(self) -> None:
        self._fit()
        self._schedule_tiles(True)

    def zoom_to(self, key: int, span: float = 0.05) -> None:
        """Centre the map on one station, close enough to see its streets."""
        hit = np.flatnonzero(self._keys == key)
        if not hit.size:
            return
        cx, cy = (float(v) for v in self._xy[hit[0]])
        box = self.ax.get_window_extent()
        span_x = max(float(span), MIN_SPAN)
        span_y = span_x * max(box.height, 1) / max(box.width, 1)
        self.ax.set_xlim(cx - span_x / 2, cx + span_x / 2)
        self.ax.set_ylim(cy - span_y / 2, cy + span_y / 2)
        self._view_changed()

    def _select_in_view(self) -> None:
        """Select every station in the current view (a right-click action)."""
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        keys = self._in_box(x0, x1, y0, y1)
        if keys:
            self.selectionChanged.emit(keys)

    def _copy(self, i: int) -> None:
        """Put the station's name and coordinates on the clipboard."""
        lat, lon = (float(v) for v in self._ll[i])
        QApplication.clipboard().setText(f"{self._labels[i]}\t{lat:.5f}\t{lon:.5f}")

    def _context(self, event: Any) -> None:
        """The right-click menu: what this station is, and what to do with it."""
        i = self._nearest(event)
        menu = QMenu(self)
        if i is not None:
            key = int(self._keys[i])
            menu.addAction("Select only this station", lambda: self.selectionChanged.emit([key]))
            menu.addAction(
                "Add it to the selection",
                lambda: self.selectionChanged.emit(sorted(set(self._selected) | {key})),
            )
            menu.addAction("Zoom to this station", lambda: self.zoom_to(key))
            menu.addAction("Copy name and coordinates", lambda: self._copy(i))
            menu.addSeparator()
        menu.addAction("Select the stations in view", self._select_in_view)
        if self._selected:
            menu.addAction("Clear the selection", lambda: self.selectionChanged.emit([]))
        menu.addAction("Fit the stations", self._fit_view)
        menu.exec(QCursor.pos())

    def _additive(self) -> bool:
        """Whether Command (macOS) or Control is held: add to the selection."""
        return bool(
            QApplication.keyboardModifiers()
            & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        )

    def _tip_off(self) -> None:
        if self._hovered is not None or self._tip.get_visible():
            self._hovered = None
            self._tip.set_visible(False)
            if not self._pan_mode:
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
