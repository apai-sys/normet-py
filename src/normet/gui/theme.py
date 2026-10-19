"""Light and dark appearance for the GUI, and the figures that follow it.

Adapted from the cume desktop app's theme manager. The window follows the
system's appearance (macOS / Windows / Linux dark mode, through Qt's colour
scheme) unless View → Appearance pins it to Light or Dark.

Figures are always *drawn* in matplotlib's default light colours — normet's
plotting helpers, the HTML report and scripts all share one look — and a
figure shown on a dark window is recoloured in place by
:func:`apply_figure_theme`: its faces, text, ticks, spines, grid and any mark
drawn in black ink. Colours that carry data (a series' own colour, a colour
map) are left alone. Every change is recorded on the figure, so
:func:`light_figure` can put it back for the time a figure is saved: an
exported PNG/PDF is on a white page with dark ink whatever the window shows.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

#: The dark theme's figure colours (as the cume app's).
DARK = {
    "figure": "#1c1c1e",
    "axes": "#242426",
    "ink": "#f2f2f7",
    "tick": "#aeaeb2",
    "edge": "#48484a",
    "grid": "#38383a",
}

#: matplotlib's default light colours, mapped to the dark theme's by role.
_TEXT = {"#000000": DARK["ink"]}
_LINE = {"#000000": DARK["ink"], "#b0b0b0": DARK["grid"]}
_FACE = {"#ffffff": DARK["axes"]}
_EDGE = {"#000000": DARK["edge"], "#cccccc": DARK["edge"]}
_COLLECTION = {"#000000": DARK["ink"], "#ffffff": DARK["axes"]}

#: View → Appearance choices, as stored in QSettings.
APPEARANCES = ("system", "light", "dark")


def _settings() -> QSettings:
    return QSettings("apai-sys", "normet-gui")


def is_dark_mode(app: QApplication | None = None) -> bool:
    """Whether the application is showing a dark palette now."""
    app = app or QApplication.instance()  # type: ignore[assignment]
    if app is None:
        return False
    return app.palette().color(QPalette.ColorRole.Window).lightnessF() < 0.5


def _hex(color: Any) -> str | None:
    from matplotlib.colors import to_hex

    try:
        return to_hex(color, keep_alpha=False).lower()
    except (ValueError, TypeError):
        return None


def apply_figure_theme(fig: Any, dark: bool) -> None:
    """Show *fig* in the dark theme (``dark=True``) or as it was drawn.

    Going dark records each colour it changes on the figure; going light
    replays that record, so the figure returns exactly to its own colours.
    Applying the theme a figure already has does nothing.
    """
    if fig is None or bool(getattr(fig, "_normet_dark", False)) == dark:
        return
    if not dark:
        for setter, was in reversed(getattr(fig, "_normet_undo", [])):
            setter(was)
        fig._normet_undo = []
        for ax in fig.get_axes():
            ax.tick_params(colors="black", which="both")
        fig._normet_dark = False
        return

    import numpy as np
    from matplotlib.collections import Collection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    undo: list[tuple[Callable[[Any], Any], Any]] = []

    def swap(get: Callable[[], Any], set_: Callable[[Any], Any], table: dict[str, str]) -> None:
        was = get()
        to = table.get(_hex(was) or "")
        if to is not None:
            set_(to)
            undo.append((set_, was))

    def swap_each(get: Callable[[], Any], set_: Callable[[Any], Any]) -> None:
        was = get()
        try:
            rows = np.asarray(was, dtype=float).reshape(-1, 4)
        except (ValueError, TypeError):  # "face", "none": not colours
            return
        to = [_COLLECTION.get(_hex(row) or "") for row in rows]
        if any(t is not None for t in to):
            set_(
                np.array(
                    [
                        to_rgba(t, alpha=row[3]) if t is not None else row
                        for t, row in zip(to, rows, strict=True)
                    ]
                )
            )
            undo.append((set_, was))

    swap(fig.get_facecolor, fig.set_facecolor, {"#ffffff": DARK["figure"]})
    for artist in fig.findobj():
        if artist is fig.patch:
            continue
        if isinstance(artist, Text):
            swap(artist.get_color, artist.set_color, _TEXT)
        elif isinstance(artist, Line2D):
            swap(artist.get_color, artist.set_color, _LINE)
            swap(artist.get_markeredgecolor, artist.set_markeredgecolor, _LINE)
            swap(artist.get_markerfacecolor, artist.set_markerfacecolor, _LINE)
        elif isinstance(artist, Patch):
            swap(artist.get_facecolor, artist.set_facecolor, _FACE)
            swap(artist.get_edgecolor, artist.set_edgecolor, _EDGE)
        elif isinstance(artist, Collection) and artist.get_array() is None:
            # One coloured by its values (a heat map, points by a variable) is in
            # the data's colours; setting them would end its mapping.
            swap_each(artist.get_edgecolor, artist.set_edgecolor)
            swap_each(artist.get_facecolor, artist.set_facecolor)
    for ax in fig.get_axes():
        # Through tick_params, so ticks made later (a zoom, a pan) are coloured too.
        ax.tick_params(colors=DARK["tick"], which="both")
    fig._normet_undo = undo
    fig._normet_dark = True


@contextlib.contextmanager
def light_figure(fig: Any) -> Iterator[None]:
    """For the time a figure is written to a file, the colours it was drawn in."""
    dark = bool(getattr(fig, "_normet_dark", False))
    apply_figure_theme(fig, False)
    try:
        yield
    finally:
        apply_figure_theme(fig, dark)


def stored_appearance() -> str:
    value = str(_settings().value("appearance", "system"))
    return value if value in APPEARANCES else "system"


def can_set_appearance() -> bool:
    """Whether this Qt lets the application choose its own colour scheme (6.8+)."""
    app = QApplication.instance()
    return app is not None and hasattr(app.styleHints(), "setColorScheme")


class ThemeManager(QObject):
    """Tells the windows when the application turns light or dark.

    ``themeChanged(dark)`` is emitted on a change of the system's appearance
    and on a View → Appearance choice; widgets that draw figures connect to it.
    """

    themeChanged = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent or QApplication.instance())
        self._dark = is_dark_mode()
        app = QApplication.instance()
        if app is not None:
            hints = app.styleHints()
            if hasattr(hints, "colorSchemeChanged"):
                hints.colorSchemeChanged.connect(lambda _scheme: self._check())
            app.installEventFilter(self)

    @property
    def dark(self) -> bool:
        return self._dark

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.ApplicationPaletteChange:
            self._check()
        return super().eventFilter(watched, event)

    def _check(self) -> None:
        dark = is_dark_mode()
        if dark != self._dark:
            self._dark = dark
            self.themeChanged.emit(dark)

    def set_appearance(self, appearance: str) -> None:
        """Pin the window to "light" or "dark", or follow the "system"; remembered."""
        if appearance not in APPEARANCES:
            raise ValueError(f"appearance must be one of {APPEARANCES}, not {appearance!r}")
        _settings().setValue("appearance", appearance)
        apply_stored_appearance()
        self._check()


def apply_stored_appearance() -> None:
    """Set the application's colour scheme from the stored View → Appearance choice."""
    if not can_set_appearance():
        return
    scheme = {
        "system": Qt.ColorScheme.Unknown,
        "light": Qt.ColorScheme.Light,
        "dark": Qt.ColorScheme.Dark,
    }[stored_appearance()]
    QApplication.instance().styleHints().setColorScheme(scheme)  # type: ignore[union-attr]


_MANAGER: ThemeManager | None = None


def theme_manager() -> ThemeManager:
    """The application's one ThemeManager, made on first use."""
    global _MANAGER
    if _MANAGER is not None:
        from shiboken6 import isValid

        if not isValid(_MANAGER):
            _MANAGER = None
    if _MANAGER is None:
        _MANAGER = ThemeManager()
    return _MANAGER
