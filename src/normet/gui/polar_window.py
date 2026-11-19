"""Analysis -> Polar Plot: a variable by wind direction and speed (openair's polarPlot).

Where a pollutant comes from is read off the wind: the open table's variable averaged in
bins of wind direction and speed, drawn on a compass (:func:`normet.polar_plot`). The
wind columns are found by name -- ``ws`` and ``wd``, else made from ``u10`` and ``v10``
-- and can be chosen; the statistic is the mean, the median, the 95th percentile or the
maximum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ._widgets import CanvasTab, NoWheelComboBox

#: The statistics offered: (label, what polar_plot is passed).
STATISTICS = (("Mean", "mean"), ("Median", "median"), ("95th percentile", "p95"), ("Max", "max"))

#: Names a wind speed and a wind direction go by, first match wins (case ignored).
WS_NAMES = ("ws", "wind_speed", "windspeed", "ws10", "wspd")
WD_NAMES = ("wd", "wind_direction", "winddir", "wd10", "wdir")


def wind_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None]:
    """``df`` with a wind speed and a direction column, and their names: found by name,
    else made from ``u10``/``v10`` (as ``ws``/``wd``, the direction the wind blows
    from, degrees from north). (None, None) where there is no wind in it."""
    lower = {str(c).lower(): c for c in df.columns}
    ws = next((lower[n] for n in WS_NAMES if n in lower), None)
    wd = next((lower[n] for n in WD_NAMES if n in lower), None)
    if ws is not None and wd is not None:
        return df, str(ws), str(wd)
    if "u10" in lower and "v10" in lower:
        u, v = df[lower["u10"]].astype(float), df[lower["v10"]].astype(float)
        df = df.copy()
        df["ws"] = np.hypot(u, v)
        df["wd"] = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
        return df, "ws", "wd"
    return df, None, None


class PolarWindow(QDialog):
    """The polar plot of the open table, redrawn as its choices change."""

    def __init__(self, df: pd.DataFrame, target: str | None = None, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("Normet — Polar plot")
        self.resize(760, 640)
        self.df, ws, wd = wind_columns(df)
        numeric = [str(c) for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]

        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.var_combo = NoWheelComboBox()
        self.var_combo.addItems(numeric)
        if target and target in numeric:
            self.var_combo.setCurrentText(target)
        self.stat_combo = NoWheelComboBox()
        for label, _ in STATISTICS:
            self.stat_combo.addItem(label)
        self.ws_combo = NoWheelComboBox()
        self.wd_combo = NoWheelComboBox()
        for combo, guess in ((self.ws_combo, ws), (self.wd_combo, wd)):
            combo.addItems(numeric)
            if guess is not None:
                combo.setCurrentText(guess)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("Speed"))
        h.addWidget(self.ws_combo, 1)
        h.addWidget(QLabel("Direction"))
        h.addWidget(self.wd_combo, 1)
        form.addRow("Variable", self.var_combo)
        form.addRow("Statistic", self.stat_combo)
        form.addRow("Wind", row)
        lay.addLayout(form)

        self.tab = CanvasTab("Choose a variable and the wind columns.")
        lay.addWidget(self.tab, 1)
        self.has_wind = ws is not None and wd is not None
        for combo in (self.var_combo, self.stat_combo, self.ws_combo, self.wd_combo):
            combo.currentTextChanged.connect(self.redraw)
        self.redraw()

    def statistic(self) -> str:
        return dict(STATISTICS)[self.stat_combo.currentText()]

    def redraw(self, *_args: object) -> None:
        from normet import polar_plot

        var, ws, wd = (c.currentText() for c in (self.var_combo, self.ws_combo, self.wd_combo))
        if not (var and ws and wd):
            self.tab.show_message("This table has no wind speed and direction to plot against.")
            return
        try:
            ax = polar_plot(self.df, target=var, ws_col=ws, wd_col=wd, statistic=self.statistic())
        except Exception as exc:  # noqa: BLE001 - say why on the tab
            self.tab.show_message(f"No polar plot: {exc}")
            return
        self.tab.show_result(
            ax,
            lines=[
                f"{self.stat_combo.currentText()} of {var} in bins of wind direction (10°) "
                f"and speed ({ws}); rows missing any of the three left out."
            ],
        )
