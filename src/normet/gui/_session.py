"""The session: config round-trip, run history, export, and the command log.

What the window remembers between runs, and what it writes out.  Mixed into
:class:`~.main_window.MainWindow`.
"""

from __future__ import annotations

import datetime
import json
import logging
import os

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from . import _chronos, _settings, commandlog
from ._lists import _checked_items

log = logging.getLogger(__name__)


class SessionMixin(QMainWindow):
    #: Which result each tab shows, for Export.
    _TAB_KEYS = {
        0: "data",
        1: "train",
        2: "normalise",
        3: "decompose",
        4: "rolling",
        5: "multiscale",
        6: "pdp",
    }

    def _record_task(self, fn, args, kwargs):  # noqa: ANN001, ANN202 - TaskRunner.record
        """The command a task makes, written now from what it was started with and
        added to the log once it has succeeded."""
        normalised = self.results.get("normalise")
        names = [(self.df_raw, "df"), (self.df_prep, "df_prep"), (self.model, "model")]
        if normalised is not None and "normalised" in normalised:
            names.append((normalised["normalised"], "normalised['normalised']"))
        entry = commandlog.task(fn, args, kwargs, [(o, n) for o, n in names if o is not None])
        if entry is None:
            return None
        return lambda: self._log_command(entry)

    def save_script(self) -> None:
        """The command log as a Python file that repeats the session."""
        if not len(self.command_log):
            QMessageBox.information(self, "Nothing to save", "Open a table and run a step first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session as script", "normet_session.py", "Python (*.py)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.command_log.script())
            log.info("Session script written to %s", path)

    def get_config(self) -> dict:
        return {
            "target": self.target_combo.currentText(),
            "features": _checked_items(self.feature_list),
            "backend": self.backend_combo.currentText(),
            "split_method": self.split_combo.currentText(),
            "train_fraction": self.train_fraction_spin.value(),
            "time_budget": (
                self.budget_spin.value()
                if self.backend_combo.currentText() == "flaml"
                else self._budget_values.get("flaml", 60)
            ),
            "lgb_trials": (
                self.budget_spin.value()
                if self.backend_combo.currentText() == "lightgbm"
                else self._budget_values.get("lightgbm", 20)
            ),
            "flaml_estimators": self._selected_estimators(),
            "seed": self.seed_spin.value(),
            "norm_vars": self._selected_norm_vars(),
            "norm_samples": self.norm_samples.value(),
            "norm_cores": self.norm_cores.value(),
            "norm_quantiles": self.norm_quantiles.isChecked(),
            "decom_method": self.decom_method.currentText(),
            "decom_samples": self.decom_samples.value(),
            "decom_attribution": self.decom_attribution.currentText(),
            "decom_groups": self.decom_groups.text(),
            "roll_window": self.roll_window.value(),
            "roll_step": self.roll_step.value(),
            "roll_samples": self.roll_samples.value(),
            "ms_fast": self.ms_fast.value(),
            "ms_meso": self.ms_meso.value(),
            "ms_slow": self.ms_slow.value(),
            "pdp_grid": self.pdp_grid.value(),
        }

    def apply_config(self, c: dict) -> None:
        if c.get("target"):
            self.target_combo.setCurrentText(c["target"])
        feats = set(c.get("features") or [])
        if feats:
            for i in range(self.feature_list.count()):
                item = self.feature_list.item(i)
                item.setCheckState(
                    Qt.CheckState.Checked if item.text() in feats else Qt.CheckState.Unchecked
                )
        self._budget_values["flaml"] = c.get("time_budget", 60)
        self._budget_values["lightgbm"] = c.get("lgb_trials", 20)
        self.backend_combo.setCurrentText(c.get("backend", "flaml"))
        self.split_combo.setCurrentText(c.get("split_method", "random"))
        self.train_fraction_spin.setValue(c.get("train_fraction", 0.75))
        self.budget_spin.setValue(self._budget_values.get(self.backend_combo.currentText(), 60))
        ests = set(c.get("flaml_estimators") or ["lgbm"])
        for i in range(self.estimator_list.count()):
            item = self.estimator_list.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.text() in ests else Qt.CheckState.Unchecked
            )
        self.seed_spin.setValue(c.get("seed", 7_654_321))
        nvars = set(c.get("norm_vars") or [])
        if nvars:
            for i in range(self.norm_vars.count()):
                item = self.norm_vars.item(i)
                item.setCheckState(
                    Qt.CheckState.Checked if item.text() in nvars else Qt.CheckState.Unchecked
                )
        # An explicit saved count wins; the fallback follows the restored backend,
        # so a config written before chronos-2 existed does not push 300 forward
        # passes onto it.
        self.norm_samples.setValue(
            c.get("norm_samples", _chronos.DEFAULT_SAMPLES if self._is_chronos() else 300)
        )
        self.norm_cores.setValue(c.get("norm_cores", 0))
        self.norm_quantiles.setChecked(c.get("norm_quantiles", False))
        self.decom_method.setCurrentText(c.get("decom_method", "emission"))
        self.decom_samples.setValue(c.get("decom_samples", 300))
        self.decom_attribution.setCurrentText(c.get("decom_attribution", "auto"))
        self.decom_groups.setText(c.get("decom_groups", ""))
        self.roll_window.setValue(c.get("roll_window", 14))
        self.roll_step.setValue(c.get("roll_step", 7))
        self.roll_samples.setValue(c.get("roll_samples", 100))
        self.ms_fast.setValue(c.get("ms_fast", 14))
        self.ms_meso.setValue(c.get("ms_meso", 90))
        self.ms_slow.setValue(c.get("ms_slow", 365))
        self.pdp_grid.setValue(c.get("pdp_grid", 50))

    def save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", "normet_config.json", "JSON (*.json)"
        )
        if path:
            with open(path, "w") as f:
                json.dump(self.get_config(), f, indent=2)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")

    def load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load config", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                self.apply_config(json.load(f))
            self.statusBar().showMessage(f"Loaded {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    def _log_history(self, task: str, detail: str) -> None:
        _settings.append_history(
            {
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "task": task,
                "target": self.target_combo.currentText(),
                "backend": self.backend_combo.currentText(),
                "rows": 0 if self.df_raw is None else len(self.df_raw),
                "detail": detail,
            }
        )

    def show_history(self) -> None:
        recs = _settings.read_history()
        dlg = QDialog(self)
        dlg.setWindowTitle("Run history")
        dlg.resize(760, 420)
        lay = QVBoxLayout(dlg)
        if not recs:
            lay.addWidget(QLabel("No runs logged yet."))
        else:
            cols = ["time", "task", "target", "backend", "rows", "detail"]
            tbl = QTableWidget(len(recs), len(cols))
            tbl.setHorizontalHeaderLabels(cols)
            tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            for i, rec in enumerate(reversed(recs)):  # newest first
                for j, col in enumerate(cols):
                    tbl.setItem(i, j, QTableWidgetItem(str(rec.get(col, ""))))
            tbl.resizeColumnsToContents()
            lay.addWidget(tbl)
        clr = QPushButton("Clear history")
        clr.clicked.connect(lambda: (_settings.clear_history(), dlg.accept()))
        lay.addWidget(clr)
        dlg.exec()

    def _current_result(self) -> tuple[str, pd.DataFrame | None]:
        key = self._TAB_KEYS.get(self.tabs.currentIndex(), "")
        if key == "data":
            return key, self.df_raw
        return key, self.results.get(key)

    def _export_current(self) -> None:
        key, result = self._current_result()
        if result is None:
            QMessageBox.information(
                self, "Nothing to export", "Run the current tab's analysis first."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export result", f"normet_{key}.csv", "CSV files (*.csv)"
        )
        if path:
            result.to_csv(path)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
            log.info("Exported %s result to %s", key, path)

    def _export_html(self) -> None:
        key, result = self._current_result()
        if result is None or key == "data":
            QMessageBox.information(
                self,
                "Nothing to report",
                "Switch to a result tab (Normalise, Decompose, …) that has been run.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HTML report", f"normet_{key}_report.html", "HTML (*.html)"
        )
        if not path:
            return
        try:
            from normet import generate_html_report, make_run

            run = make_run(
                result,
                kind=key,
                config=self.get_config(),
                df=self.df_raw,
                model=self.model,
                seed=self.seed_spin.value(),
            )
            generate_html_report(run, path)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
            log.info("HTML report written to %s", path)
        except Exception as exc:
            QMessageBox.critical(self, "Report failed", str(exc))
