"""A figure for print: a tab's figure laid out at the width a journal prints it,
in the lettering asked for, previewed at that size and saved at the resolution
chosen. Adapted, in much smaller form, from the cume app's print view.

A figure on screen is drawn at the tab's size; printed in a 89 mm column its
lettering shrinks with it. Here the figure is set to the printed size and its
text to a size in points, so what is saved is what the page shows. Nothing is
kept: the figure on the tab goes back to its own size, lettering, layout and
colours when the dialog closes (:func:`print_layout`).

The widths offered are the column widths the publishers' artwork guides give;
check the journal's own current guide before submitting.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterator
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .theme import light_figure

MM_PER_INCH = 25.4

#: Column widths in mm, as the publishers' artwork guides give them.
PRESETS: dict[str, float | None] = {
    "Nature: single column (89 mm)": 89.0,
    "Nature: double column (183 mm)": 183.0,
    "Elsevier: single column (90 mm)": 90.0,
    "Elsevier: 1.5 column (140 mm)": 140.0,
    "Elsevier: double column (190 mm)": 190.0,
    "ACS: single column (3.25 in)": 3.25 * MM_PER_INCH,
    "ACS: double column (7 in)": 7.0 * MM_PER_INCH,
    "Custom": None,
}

#: The resolutions offered, in dots per inch.
DPIS = (300, 600, 1200)

#: What a figure can be saved as, by extension.
FORMATS = "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif *.tiff);;EPS (*.eps)"


@contextlib.contextmanager
def print_layout(fig: Any, width_mm: float, height_mm: float, font_pt: float) -> Iterator[None]:
    """For the time being: *fig* at the printed size, its text at *font_pt*
    (titles a point larger), laid out again to fit, in its light colours.

    Everything changed is put back on leaving: the size, every text's size,
    the layout engine and the colours.
    """
    from matplotlib.text import Text

    size = tuple(fig.get_size_inches())
    engine = fig.get_layout_engine()
    titles = {id(ax.title) for ax in fig.get_axes()}
    if getattr(fig, "_suptitle", None) is not None:
        titles.add(id(fig._suptitle))
    texts = [t for t in fig.findobj(Text) if t.get_text()]
    was = [(t, t.get_fontsize()) for t in texts]
    with light_figure(fig):
        try:
            fig.set_size_inches(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH)
            for t in texts:
                t.set_fontsize(font_pt + 1 if id(t) in titles else font_pt)
            with contextlib.suppress(RuntimeError):  # a colour bar's engine stays as it is
                fig.set_layout_engine("constrained")
            yield
        finally:
            for t, pt in was:
                t.set_fontsize(pt)
            with contextlib.suppress(RuntimeError):
                fig.set_layout_engine(engine)
            fig.set_size_inches(*size)


def render(fig: Any, width_mm: float, height_mm: float, font_pt: float, dpi: float) -> bytes:
    """*fig* as a PNG at the printed size, as :func:`save` writes it."""
    buf = io.BytesIO()
    save(fig, buf, width_mm, height_mm, font_pt, dpi, fmt="png")
    return buf.getvalue()


def save(
    fig: Any,
    target: Any,
    width_mm: float,
    height_mm: float,
    font_pt: float,
    dpi: float,
    fmt: str | None = None,
) -> None:
    """Write *fig* at the printed size: the whole page, not cropped to its ink, so
    the file is exactly *width_mm* wide."""
    import matplotlib as mpl

    with (
        print_layout(fig, width_mm, height_mm, font_pt),
        mpl.rc_context({"savefig.bbox": "standard"}),
    ):
        fig.savefig(target, dpi=dpi, format=fmt)


class PrintDialog(QDialog):
    """The figure on *canvas* laid out for print while the dialog is open."""

    def __init__(self, canvas: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Figure for Publication")
        self.canvas, self.fig = canvas, canvas.figure
        w_in, h_in = self.fig.get_size_inches()
        # Its own proportions, within reason: a strip drawn to fill a wide tab
        # would print a column width wide and a couple of lines high.
        self._aspect = min(1.2, max(0.35, h_in / w_in if w_in else 0.6))

        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        self.width = QDoubleSpinBox()
        self.width.setRange(20, 500)
        self.width.setSuffix(" mm")
        self.height = QDoubleSpinBox()
        self.height.setRange(20, 500)
        self.height.setSuffix(" mm")
        self.font = QDoubleSpinBox()
        self.font.setRange(4, 14)
        self.font.setSingleStep(0.5)
        self.font.setValue(7.0)
        self.font.setSuffix(" pt")
        self.font.setToolTip("Tick labels, axis labels and legends; titles are a point larger.")
        self.dpi = QComboBox()
        self.dpi.addItems([str(d) for d in DPIS])
        self.dpi.setToolTip(
            "Dots per inch of a raster file (PNG, TIFF); PDF, SVG and EPS are vector."
        )

        form = QFormLayout()
        form.addRow("Journal", self.preset)
        form.addRow("Width", self.width)
        form.addRow("Height", self.height)
        form.addRow("Lettering", self.font)
        form.addRow("Resolution (dpi)", self.dpi)
        note = QLabel(
            "Widths are those the publishers' artwork guides give; check the journal's own "
            "guide before submitting. The file is on a white page, at exactly this width."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(placeholder-text); font-size: 11px;")
        save_btn = QPushButton("Save…")
        save_btn.setStyleSheet("font-weight: 600;")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)

        left = QVBoxLayout()
        left.addLayout(form)
        left.addWidget(note)
        left.addStretch(1)
        row = QHBoxLayout()
        row.addWidget(close_btn)
        row.addWidget(save_btn)
        left.addLayout(row)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(self.preview)
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(520, 420)
        self.size_label = QLabel()
        self.size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right = QVBoxLayout()
        right.addWidget(scroll, 1)
        right.addWidget(self.size_label)

        lay = QHBoxLayout(self)
        lay.addLayout(left)
        lay.addLayout(right, 1)

        self.preset.currentTextChanged.connect(self._preset_changed)
        for spin in (self.width, self.height, self.font):
            spin.valueChanged.connect(self._redraw)
        self.width.valueChanged.connect(self._keep_aspect)
        self._preset_changed(self.preset.currentText())

    def _preset_changed(self, name: str) -> None:
        width = PRESETS.get(name)
        self.width.setEnabled(width is None)
        if width is not None:
            self.width.setValue(round(width, 1))

    def _keep_aspect(self, width: float) -> None:
        self.height.blockSignals(True)
        self.height.setValue(round(width * self._aspect, 1))
        self.height.blockSignals(False)
        self._redraw()

    def settings(self) -> tuple[float, float, float, int]:
        return (
            self.width.value(),
            self.height.value(),
            self.font.value(),
            int(self.dpi.currentText()),
        )

    def _redraw(self) -> None:
        """The preview, at the printed size on this screen (96 px to the inch)."""
        w, h, pt, _ = self.settings()
        png = render(self.fig, w, h, pt, dpi=96 * self.devicePixelRatioF())
        pix = QPixmap()
        pix.loadFromData(png, "PNG")
        pix.setDevicePixelRatio(self.devicePixelRatioF())
        self.preview.setPixmap(pix)
        self.size_label.setText(f"{w:.0f} × {h:.0f} mm, shown at its printed size")
        self.canvas.draw_idle()

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", "figure.png", FORMATS)
        if not path:
            return
        w, h, pt, dpi = self.settings()
        save(self.fig, path, w, h, pt, dpi)
        self.canvas.draw_idle()
        self.accept()
