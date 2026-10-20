"""The task runner's lifecycle, and which controls a task leaves enabled.

``_sync_enabled`` is the one method that knows about every step's button; it
lives here, next to the runner, rather than being spread over the panels.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QMessageBox

from ._widgets import NoWheelSpinBox


class TaskMixin(QMainWindow):
    def _sync_enabled(self) -> None:
        busy = self.runner.busy
        has_data = self.df_raw is not None
        has_model = self.model is not None
        has_y_inf = "normalise" in self.results
        self.train_button.setEnabled(not busy and has_data)
        self.ft_button.setEnabled(not busy and has_model and self._is_chronos())
        for b in (self.norm_button, self.decom_button, self.roll_button, self.pdp_button):
            b.setEnabled(not busy and has_model)
        if not has_model:
            tip = "Train a model first."
            for b in (self.norm_button, self.decom_button, self.roll_button, self.pdp_button):
                b.setToolTip(tip)
        elif self._is_chronos():
            # Rolling and PDP call ml_predict() on a fitted scikit-learn-style
            # estimator. Chronos-2 forecasts autoregressively from a context
            # window instead, so there is nothing for them to call. The
            # meteorological decomposition has its own zero-shot path.
            tip = "Not available for chronos-2 — use flaml or lightgbm for this step."
            for b in (self.roll_button, self.pdp_button):
                b.setEnabled(False)
                b.setToolTip(tip)
        self.ms_button.setEnabled(not busy and has_model and has_y_inf)
        self.ms_button.setToolTip(
            self._ms_button_tip
            if has_y_inf
            else "Run the normalisation first — it provides Y_∞, the full-record baseline."
        )
        # A notice asking for a step that has now been taken has been answered.
        for key, done in (
            ("need-data", has_data),
            ("need-model", has_model),
            ("need-normalise", has_y_inf),
        ):
            if done:
                self.notices.clear(key)
        if hasattr(self, "_step_lights"):
            self._update_step_lights()
        has_result = bool(self.results)
        self.exp_csv.setEnabled(has_data)
        self.exp_html.setEnabled(has_result)
        self._export_action.setEnabled(has_data)
        self._report_action.setEnabled(has_result)
        self.cancel_btn.setEnabled(busy)

    def _task_started(self, name: str) -> None:
        self.progress.setRange(0, 0)
        self.statusBar().showMessage(f"Running: {name}…")
        for b in self._run_buttons:
            b.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def _task_finished(self) -> None:
        self.progress.setRange(0, 1)
        self._sync_enabled()
        self.autosave_session()  # File -> Reopen Last Session, should the window go

    def _abandon_task(self) -> None:
        if self.runner.busy:
            self.runner.abandon()
            self.statusBar().showMessage(
                "Task abandoned — it finishes in the background and its result is discarded."
            )

    def _check_data(self) -> bool:
        if self.df_raw is None:
            self.notify(
                "Open a table first: a CSV with a date column, the pollutant and its meteorology.",
                key="need-data",
                actions=(("Open Data…", self._open_csv), ("Load example", self.load_example)),
            )
            return False
        if not self._selected_features():
            self.notify(
                "Tick at least one feature (predictor) under Columns.",
                key="need-features",
                actions=(("Met only", self._check_met_only),),
            )
            return False
        return True

    def _check_model(self) -> bool:
        if self.model is None or self.df_prep is None:
            self.notify(
                "Train a model first: every later step reads it.",
                key="need-model",
                actions=(("Train model", self._run_train),),
            )
            return False
        return True

    def _show_error(self, tb: str) -> None:
        QMessageBox.critical(self, "Task failed", tb.splitlines()[-1] if tb else "Unknown error")
        self.log_view.appendPlainText(tb)

    def _n_cores(self, spin: NoWheelSpinBox) -> int | None:
        return None if spin.value() == 0 else spin.value()
