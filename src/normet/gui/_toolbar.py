# src/normet/gui/_toolbar.py
"""The figure toolbar, in a module of its own so importing the window stays
matplotlib-free.

``NavigationToolbar2QT`` drags in ``matplotlib.backends.backend_qtagg``, which
drags in matplotlib -- ~1.2 s of the window's import, paid on every process
that imports the GUI and before any figure exists. The class moved here and
``_widgets`` imports it when a figure widget is built.
"""

from __future__ import annotations

from typing import Any

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QWidget

from .theme import light_figure, theme_manager


class ThemedToolbar(NavigationToolbar2QT):
    """The figure toolbar: icons
    16 pt and tinted for the theme, the bar in the window's own colours, Save
    writing the figure in its light colours, and *For print…* -- the figure at
    a journal's column width and lettering -- before the pointer's read-out."""

    #: The icons' size, in points: the size of the window's own small buttons.
    #: Matplotlib's 24, with its padding, made the bar the tallest control.
    ICON = 16

    def __init__(self, canvas: Any, parent: QWidget | None = None) -> None:
        super().__init__(canvas, parent)
        self.print_action = QAction("For print…", self)
        self.print_action.setToolTip(
            "Lay the figure out at a journal's column width and lettering, and save it."
        )
        # `triggered` hands a slot its checked flag: not a figure to print.
        self.print_action.triggered.connect(lambda: self.export_for_print())
        label = getattr(self, "locLabel", None)
        where = next(
            (a for a in self.actions() if label is not None and self.widgetForAction(a) is label),
            None,
        )
        if where is not None:
            self.insertAction(where, self.print_action)
        else:
            self.addAction(self.print_action)
        # The figure's right-click menu: copy it, save it, or lay it out for print --
        # the toolbar's three figure actions, reachable from the plot itself.
        canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        canvas.customContextMenuRequested.connect(self._canvas_menu)
        self._made_in: tuple[str, ...] | None = None
        theme_manager().themeChanged.connect(self._on_theme_changed)
        self.apply_theme(theme_manager().dark)

    def _render_icon(self, name: str, dark: bool) -> QIcon:
        """Matplotlib's icon *name*, tinted the theme's ink, at 1x and 2x."""
        from matplotlib import cbook

        try:
            regular = cbook._get_data_path("images", name)
        except Exception:
            return QIcon()
        large = regular.with_name(regular.name.replace(".png", "_large.png"))
        tint = QColor("#F2F2F7" if dark else "#1C1C1E")
        icon = QIcon()
        for path, dpr in ((regular, 1.0), (large, 2.0)):
            if not path.exists():
                continue
            src = QImage(str(path)).convertToFormat(QImage.Format.Format_ARGB32)
            if src.isNull():
                continue
            coloured = QImage(src.size(), QImage.Format.Format_ARGB32)
            coloured.fill(Qt.GlobalColor.transparent)
            p = QPainter(coloured)
            p.drawImage(0, 0, src)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            p.fillRect(coloured.rect(), tint)
            p.end()
            pm = QPixmap.fromImage(coloured)
            pm.setDevicePixelRatio(dpr)
            icon.addPixmap(pm)
        return icon

    def _icon(self, name: str) -> QIcon:  # matplotlib's hook for each tool's icon
        return self._render_icon(name, dark=theme_manager().dark)

    @staticmethod
    def _palette_key() -> tuple[str, ...]:
        pal = QApplication.palette()
        return tuple(
            pal.color(role).name()
            for role in (
                QPalette.ColorRole.Window,
                QPalette.ColorRole.WindowText,
                QPalette.ColorRole.Highlight,
            )
        )

    def apply_theme(self, dark: bool) -> None:
        """Icons tinted for the theme; the bar in the application palette's window
        colour, its hover, pressed and dividing shades mixed from that and the
        text's -- not a colour of its own, which stood out from the window."""
        for _text, _tip, image, callback in self.toolitems:
            if image and callback in self._actions:
                self._actions[callback].setIcon(self._render_icon(image + ".png", dark))
        self.setIconSize(QSize(self.ICON, self.ICON))
        pal = QApplication.palette()
        ground = pal.color(QPalette.ColorRole.Window)
        ink = pal.color(QPalette.ColorRole.WindowText)

        def mix(t: float) -> str:
            return QColor.fromRgbF(
                ground.redF() + (ink.redF() - ground.redF()) * t,
                ground.greenF() + (ink.greenF() - ground.greenF()) * t,
                ground.blueF() + (ink.blueF() - ground.blueF()) * t,
            ).name()

        accent = pal.color(QPalette.ColorRole.Highlight).name()
        self._made_in = self._palette_key()
        self.setStyleSheet(
            f"""
            QToolBar {{ background: {ground.name()}; border: none; padding: 0px 2px;
                        spacing: 0px; }}
            QToolButton {{ background: transparent; border: 1px solid transparent;
                           border-radius: 4px; padding: 1px; margin: 0px; }}
            QToolButton:hover {{ background: {mix(0.08)}; }}
            QToolButton:pressed, QToolButton:checked {{ background: {mix(0.15)};
                                                       border-color: {accent}; }}
            QToolBar::separator {{ width: 1px; background: {mix(0.18)}; margin: 3px 3px; }}
            QLabel {{ color: {mix(0.45)}; font-size: 11px; padding-right: 4px; }}
            """
        )

    def _on_theme_changed(self, dark: bool) -> None:
        try:
            self.apply_theme(dark)
        except (RuntimeError, AttributeError):  # the bar deleted underneath
            pass

    def changeEvent(self, event: Any) -> None:  # noqa: N802
        """Made again when the palette's colours change: a theme can be announced
        before the palette has changed, and the bar made then is in the old one."""
        super().changeEvent(event)
        if (
            event.type() == QEvent.Type.PaletteChange
            and self._made_in is not None
            and self._palette_key() != self._made_in
        ):
            self.apply_theme(theme_manager().dark)

    def export_for_print(self) -> None:
        from .printview import PrintDialog

        PrintDialog(self.canvas, self.window()).exec()

    def _canvas_menu(self, pos: Any) -> None:
        """The figure's right-click menu: the toolbar's figure actions, from the plot.
        Lambdas: `triggered` hands a slot its checked, which the slots would read as an
        argument (the print_action's click learned the same)."""
        menu = QMenu(self.canvas)
        menu.addAction("Copy image", lambda: self.copy_figure())
        menu.addAction("Save image…", lambda: self.save_figure())
        menu.addSeparator()
        menu.addAction("For print…", lambda: self.export_for_print())
        menu.exec(self.canvas.mapToGlobal(pos))

    def copy_figure(self) -> None:
        """The figure's bitmap to the clipboard, at the screen's resolution."""
        fig = self.canvas.figure
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = bytes(fig.canvas.buffer_rgba())
        img = QImage(buf, w, h, QImage.Format.Format_RGBA8888).copy()
        QApplication.clipboard().setImage(img)

    def save_figure(self, *args: Any) -> Any:
        with light_figure(self.canvas.figure):
            return super().save_figure(*args)
