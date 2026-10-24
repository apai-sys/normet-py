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


def _plain(value):  # noqa: ANN001, ANN202
    """A value as the Methods paragraph keeps it: numbers, text and their containers;
    a table or a model is not kept."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return None


def _plain_call(args, kwargs) -> dict:  # noqa: ANN001
    """A call's arguments, the plain ones only, with its positional ones as ``args``."""
    out = {k: _plain(v) for k, v in kwargs.items()}
    out["args"] = [_plain(a) for a in args]
    return out


#: The run history's steps (``_log_history``), as its table names them.
_TASK_WORDS = {
    "load": "Load",
    "train": "Train",
    "normalise": "Normalise",
    "decompose": "Decompose",
    "rolling": "Rolling",
    "multiscale": "Multi-scale",
    "pdp": "PDP",
    "trend": "Trend",
    "events": "Events",
    "uncertainty": "Uncertainty",
}


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
        7: "trend",
        8: "events",
        9: "uncertainty",
    }

    def _record_task(self, fn, args, kwargs):  # noqa: ANN001, ANN202 - TaskRunner.record
        """The command a task makes, written now from what it was started with and
        added to the log once it has succeeded -- and its arguments kept by the step's
        name, for the Methods paragraph (.methods), which says what ran."""
        normalised = self.results.get("normalise")
        # The rows kept where some are left out (the log's df_kept, ._data_tab), else df.
        names = [
            (self.df_raw, "df_kept" if self.exclusions else "df"),
            (self.df_loaded, "df"),
            (self.df_prep, "df_prep"),
            (self.model, "model"),
            (self.results.get("decompose"), "decomposition"),
        ]
        if normalised is not None and "normalised" in normalised:
            names.append((normalised["normalised"], "normalised['normalised']"))
        entry = commandlog.task(fn, args, kwargs, [(o, n) for o, n in names if o is not None])
        call = _plain_call(args, kwargs)
        step = getattr(fn, "__name__", "")

        def commit() -> None:
            if step == "build_model" or step == "load_estimator":
                self._ran.clear()  # a new model: what ran before was on the old one
            self._ran[step] = call
            if entry is not None:
                self._log_command(entry)

        return commit

    def methods_info(self) -> dict:
        """What the Methods paragraph is written from: the record, what ran, the fit."""
        import numpy as np

        import normet

        info: dict = {"version": normet.__version__, "ran": dict(self._ran)}
        if self.df_raw is not None and "date" in self.df_raw.columns:
            dates = pd.to_datetime(self.df_raw["date"]).dropna().sort_values()
            info.update(
                rows=len(self.df_raw),
                start=f"{dates.iloc[0]:%Y-%m-%d}",
                end=f"{dates.iloc[-1]:%Y-%m-%d}",
                target=self.target_combo.currentText(),
            )
            step = float(
                np.median(np.diff(dates.to_numpy()).astype("timedelta64[s]").astype(float))
            )
            for limit, en, zh in ((3600 * 1.5, "hourly", "逐时"), (86400 * 1.5, "daily", "逐日")):
                if step <= limit:
                    info.update(step=en, step_zh=zh)
                    break
        stats = self.results.get("train")
        if stats is not None and "set" in stats.columns:
            row = stats[stats["set"] == "testing"]
            if len(row):
                r = row.iloc[0]
                info["test"] = {"r2": float(r["R2"]), "rmse": float(r["RMSE"]), "n": int(r["n"])}
        if self.chronos_check is not None:
            info["chronos_pct"] = float(self.chronos_check[0].get("pct_of_prediction", 0.0))
        return info

    def show_methods_paragraph(self) -> None:
        """File -> Methods Paragraph: what the session ran, for a paper, ready to copy."""
        from PySide6.QtWidgets import QApplication, QDialogButtonBox, QPlainTextEdit

        from . import uilang
        from .methods import methods_paragraph

        text = methods_paragraph(self.methods_info(), uilang.language())
        if text is None:
            self.notify(
                "Train a model first: the Methods paragraph says what the session ran.",
                key="need-model",
                actions=(("Train model", self._run_train),),
            )
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Methods paragraph")
        dlg.resize(680, 520)
        lay = QVBoxLayout(dlg)
        note = QLabel(
            "What this session ran, in its own numbers, for a paper's Methods: check it "
            "against what you did, and cite normet as its release says."
        )
        note.setWordWrap(True)
        lay.addWidget(note)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setProperty(
            "normetKeepText", True
        )  # the paragraph is in the language it was written in
        lay.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy = buttons.addButton("Copy", QDialogButtonBox.ButtonRole.ActionRole)
        copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        self._methods_dialog = dlg
        dlg.open()

    def save_script(self) -> None:
        """The command log as a Python file that repeats the session."""
        if not len(self.command_log):
            self.notify("Nothing to save yet: open a table and run a step first.", key="nothing")
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
            "exclude": list(self.exclusions),
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
            "trend_series": self.trend_series.currentText(),
            "tre_seasonal": self.tre_seasonal.currentText(),
            "tre_segments": self.tre_segments.value(),
            "tre_n_bootstrap": self.tre_n_bootstrap.value(),
            "trend_after": self.trend_after.isChecked(),
            "pdp_grid": self.pdp_grid.value(),
            "unc_samples": self.unc_samples.value(),
            "unc_models": self.unc_models.value(),
            "unc_backend": self.unc_backend.currentText(),
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
        if "exclude" in c:  # the rows left out (the Data tab's Exclude box)
            self._set_exclusions(list(c.get("exclude") or []))
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
        # Segments before the series: raising the row moves a still-default
        # emission-part choice to the long-term one, and the restored choice
        # must stand after it.
        self.tre_segments.setValue(int(c.get("tre_segments", 1)))
        self.trend_series.setCurrentText(c.get("trend_series", "emission part (emi_total)"))
        self.tre_seasonal.setCurrentText(c.get("tre_seasonal", "auto"))
        self.tre_n_bootstrap.setValue(c.get("tre_n_bootstrap", 1000))
        self.trend_after.setChecked(c.get("trend_after", True))
        self.pdp_grid.setValue(c.get("pdp_grid", 50))
        self.unc_samples.setValue(c.get("unc_samples", 300))
        self.unc_models.setValue(c.get("unc_models", 4))
        self.unc_backend.setCurrentText(c.get("unc_backend", "lightgbm"))

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
        if getattr(self, "_restoring", False):
            return  # a saved session read back: its runs are not run again
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
            tbl.setHorizontalHeaderLabels(["Time", "Task", "Target", "Backend", "Rows", "Detail"])
            tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            from .uilang import tr

            for i, rec in enumerate(reversed(recs)):  # newest first
                for j, col in enumerate(cols):
                    value = str(rec.get(col, ""))
                    if col == "task":  # the step, in the window's words
                        value = tr(_TASK_WORDS.get(value, value))
                    tbl.setItem(i, j, QTableWidgetItem(value))
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
            self.notify("Nothing to export: run the current tab's analysis first.", key="nothing")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export result", f"normet_{key}.csv", "CSV files (*.csv)"
        )
        if path:
            result.to_csv(path)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
            log.info("Exported %s result to %s", key, path)

    def _export_html(self, lang: str | None = None) -> None:
        """The current tab's result as an HTML report in *lang* (the one last chosen when
        None), and *lang* kept for the next one. A cancelled save changes nothing."""
        key, result = self._current_result()
        if result is None or key == "data":
            self.notify(
                "Nothing to report: switch to a result tab (Normalise, Decompose, …) that has "
                "been run.",
                key="nothing",
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
            from ._shell import report_language, set_report_language

            if lang is None:
                lang = report_language()
            else:
                set_report_language(lang)
            generate_html_report(run, path, lang=lang)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
            log.info("HTML report written to %s", path)
        except Exception as exc:
            QMessageBox.critical(self, "Report failed", str(exc))
