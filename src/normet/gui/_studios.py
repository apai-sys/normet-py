"""The windows that open from this one, and the data they hand back.

SCM Studio, Data Studio and Transport Studio each have their own window; this
is the wiring -- construction, the signals back into the main table, and the
trajectory features merged onto it.
"""

from __future__ import annotations

import logging

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QMessageBox

from normet.utils._time import to_datetime_coerced

from . import commandlog

log = logging.getLogger(__name__)


class StudiosMixin(QMainWindow):
    def open_scm_studio(self) -> None:
        from .scm_window import SCMWindow

        if self._scm_window is None:
            self._scm_window = SCMWindow(parent=self)
        # Offer the loaded data if it looks like a panel (a repeated-unit column).
        if self.df_raw is not None:
            self._scm_window.offer_main_data(self.df_raw)
        self._scm_window.show()
        self._scm_window.raise_()
        self._scm_window.activateWindow()

    def open_scm_example(self) -> None:
        """The SCM Studio, its example panel loaded."""
        self.open_scm_studio()
        if self._scm_window is not None:
            self._scm_window.load_example()

    def open_polar_plot(self) -> None:
        """Analysis -> Polar Plot: the open table's target by wind direction and speed."""
        from .polar_window import PolarWindow

        if self.df_raw is None:
            self.notify("Open a table first: the polar plot is drawn from its wind columns.")
            return
        frame = self.df_raw.set_index("date") if "date" in self.df_raw.columns else self.df_raw
        self._polar_window = PolarWindow(frame, self.target_combo.currentText(), parent=self)
        self._polar_window.show()
        self._polar_window.raise_()

    def open_data_studio(self) -> None:
        from .data_window import DataWindow

        if self._data_window is None:
            self._data_window = DataWindow(parent=self)
        self._data_window.show()
        self._data_window.raise_()
        self._data_window.activateWindow()

    def ingest_dataframe(self, df: pd.DataFrame, label: str) -> None:
        """Load an in-memory table (e.g. from the Data Studio, on UTC) as the dataset, on
        the window's clock (the time zone chosen under Data)."""
        from normet.io.tables import utc_to_clock

        df = df.copy()
        df["date"] = utc_to_clock(to_datetime_coerced(df["date"]), self._zone())
        self._data_path = None  # not a file the time zone can read again
        self._show_dates(None)
        self._ingest(df, label)
        self._log_command(commandlog.from_memory(label))

    def open_transport_studio(self) -> None:
        from .trajectory_window import TrajectoryWindow

        if self._transport_window is None:
            self._transport_window = TrajectoryWindow(parent=self)
        if self.df_raw is not None:
            # HYSPLIT runs on UTC: the receptor times are the table's, on UTC.
            from normet.io.tables import clock_to_utc

            on_utc = self.df_raw.copy()
            on_utc["date"] = clock_to_utc(on_utc["date"], self._zone())
            self._transport_window.offer_main_data(on_utc)
        self._transport_window.show()
        self._transport_window.raise_()
        self._transport_window.activateWindow()

    def merge_trajectory_features(self, features: pd.DataFrame) -> int:
        """Join transport-aware ``traj_*`` columns (indexed by receptor time)
        onto the loaded dataset, then re-select them as features so they show
        up ticked in Step 1.

        Each row takes the latest trajectory to have arrived, within one
        release interval (:func:`normet.io.join_trajectory_features`); a row
        none matches stays empty. When both frames have a ``site`` column, a
        row takes only its own site's trajectories and other sites' rows are
        left as they were, so several sites can be sent one at a time.

        Returns the number of rows that took a trajectory: 0 when nothing was
        joined (the dataset is then left unchanged, and the user told why).
        """
        if self.df_raw is None:
            self.notify(
                "Open a table in the main window first: the trajectory features are joined onto it.",
                key="need-data",
                actions=(("Open Data…", self._open_csv),),
            )
            return 0
        from normet.io.tables import utc_to_clock
        from normet.io.trajectory import join_trajectory_features

        # The trajectories arrive on UTC; the table is on the window's clock.
        features = features.copy()
        features.index = pd.DatetimeIndex(utc_to_clock(features.index.to_series(), self._zone()))
        try:
            merged = join_trajectory_features(self.df_raw, features)
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "Transport features not joined", str(exc).strip("'\""))
            return 0
        traj_cols = [c for c in features.columns if c != "site"]
        by_site = "site" in features.columns and "site" in self.df_raw.columns
        feat_sites = sorted(features["site"].astype(str).unique()) if by_site else []
        in_scope = (
            self.df_raw["site"].astype(str).isin(feat_sites)
            if by_site
            else pd.Series(True, index=self.df_raw.index)
        )
        n_scope = int(in_scope.sum())
        n_hit = int((merged[traj_cols].notna().any(axis=1) & in_scope).sum())
        if not n_hit:
            if by_site and not n_scope:
                main_sites = sorted(self.df_raw["site"].astype(str).unique())
                why = (
                    f"The features are for site(s) {', '.join(feat_sites)}, but the data's "
                    f"'site' column holds {', '.join(main_sites[:8])}"
                    f"{', ...' if len(main_sites) > 8 else ''}.\n\nName the site as the data "
                    "does (Transport Studio, 'Site name' or the receptor's name) and send "
                    "it again."
                )
            else:
                dates = pd.to_datetime(features.index)
                why = (
                    "No row of the data falls within a release interval after a "
                    f"trajectory: the trajectories arrive {dates.min():%Y-%m-%d %H:%M} → "
                    f"{dates.max():%Y-%m-%d %H:%M}, the data spans "
                    f"{self.df_raw['date'].min():%Y-%m-%d} → {self.df_raw['date'].max():%Y-%m-%d}."
                )
            QMessageBox.warning(self, "Transport features not joined", why)
            log.warning("Transport features not joined: no row took a trajectory")
            return 0

        self._data_path = None  # not a file the time zone can read again
        self._show_dates(None)
        self._ingest(merged, f"{self.file_label.text().splitlines()[0]} + transport features")
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.text() in traj_cols:
                item.setCheckState(Qt.CheckState.Checked)
        log.info(
            "Joined %d transport feature column(s) onto the dataset: %d of %d row(s) took a "
            "trajectory%s",
            len(traj_cols),
            n_hit,
            n_scope,
            f" (site(s) {', '.join(feat_sites)})" if by_site else "",
        )
        return n_hit
