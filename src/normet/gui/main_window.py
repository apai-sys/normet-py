"""Main window of the normet Qt GUI.

Design language: a scrollable parameter panel on the left walks the user
through the workflow top-to-bottom (Data → Columns → Train → Normalise →
Decompose → Rolling → PDP), each step with its own prominent "▶ Run"
button; results land in tabs on the right, which auto-activate when a run
finishes.  Verdict banners (green / amber / red) summarise quality at a
glance; every long computation runs on a background thread; the bottom dock
streams normet's log. Before a table is open the Data tab shows the welcome
(:mod:`.welcome`); Ctrl+K (⌘K) opens the command palette (:mod:`.palette`);
the window and its figures follow the system's light/dark appearance unless
View → Appearance pins one (:mod:`.theme`).

The synthetic-control workflow (panel data, treated unit vs. donors) has
genuinely different inputs, so it lives in its own window — see
:mod:`.scm_window` — opened from the toolbar button or the Analysis menu.

The chrome around all of this — menu, tabs, docks, status bar, command palette
— is in :mod:`.shell`; the left panel is built by :mod:`.panel_inputs`,
:mod:`.panel_train` and :mod:`.panel_analyse`. Each is a mixin, so every widget
is still an attribute of the window. What is left here is the window's state,
the run handlers and the workflow itself.
"""

from __future__ import annotations

import datetime
import json
import logging
import os

import pandas as pd
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from normet.utils._time import to_datetime_coerced

from . import _chronos, _settings, commandlog
from ._constants import MET_DEFAULTS, TIME_VARS
from ._lists import _checked_items, _fill_checklist, parse_groups
from ._panel_analyse import AnalysePanelMixin
from ._panel_inputs import InputsMixin
from ._panel_train import TrainPanelMixin
from ._shell import ShellMixin
from ._widgets import (
    NoWheelSpinBox,
)
from .workers import TaskRunner

log = logging.getLogger(__name__)


class MainWindow(InputsMixin, TrainPanelMixin, AnalysePanelMixin, ShellMixin):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Normet — meteorological normalisation & counterfactual toolkit")
        self.resize(1380, 900)
        self.setAcceptDrops(True)  # drag a CSV onto the window to load it

        self.df_raw: pd.DataFrame | None = None
        self.df_prep: pd.DataFrame | None = None
        self.model: object | None = None
        self.trained_features: list[str] = []
        self.results: dict[str, pd.DataFrame] = {}
        #: Chronos-2's covariate-sensitivity check of the model loaded: (numbers, verdict).
        self.chronos_check: tuple[dict, tuple[str, str]] | None = None
        self._scm_window = None
        self._data_window = None
        self._transport_window = None

        self.runner = TaskRunner(self)
        self.runner.started.connect(self._task_started)
        self.runner.finished.connect(self._task_finished)
        self.command_log = commandlog.CommandLog()
        self.runner.record = self._record_task

        self._build_central()
        self._build_log_dock()
        self._build_command_log_dock()
        self._build_assistant()
        self._build_status_bar()
        self._build_menu()
        self._sync_enabled()

        settings = QSettings("apai-sys", "normet-gui")
        geo = settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self.statusBar().showMessage("Open a CSV or load the example data to begin.")

    # --------------------------------------------------------- left panel
    def _build_param_panel(self) -> QWidget:
        """The left column: one box per workflow step, top to bottom.

        Each box is built by its own method below, so a step's controls stay
        findable by name rather than by scrolling through four hundred lines
        of layout.  The order here is the order the workflow runs in, and
        ``_run_buttons`` collects the buttons a running task greys out.
        """
        panel = QWidget()
        v = QVBoxLayout(panel)
        for box in (
            self._build_data_box(),
            self._build_columns_box(),
            self._build_train_box(),
            self._build_normalise_box(),
            self._build_decompose_box(),
            self._build_rolling_box(),
            self._build_multiscale_box(),
            self._build_pdp_box(),
        ):
            v.addWidget(box)
        v.addLayout(self._build_config_row())
        v.addStretch(1)

        self._run_buttons = [
            self.train_button,
            self.ft_button,
            self.norm_button,
            self.decom_button,
            self.roll_button,
            self.ms_button,
            self.pdp_button,
        ]

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(390)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _build_config_row(self) -> QHBoxLayout:
        """Save/load a config and the run history, under the last step."""
        row = QHBoxLayout()
        b_save = QPushButton("Save config")
        b_save.clicked.connect(self.save_config)
        b_load = QPushButton("Load config")
        b_load.clicked.connect(self.load_config)
        b_hist = QPushButton("History")
        b_hist.clicked.connect(self.show_history)
        row.addWidget(b_save)
        row.addWidget(b_load)
        row.addWidget(b_hist)
        return row

    # ----------------------------------------------------------- command log
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

    # ------------------------------------------------------------- config
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

    # ------------------------------------------------------------- history
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

    # -------------------------------------------------------------- export
    _TAB_KEYS = {
        0: "data",
        1: "train",
        2: "normalise",
        3: "decompose",
        4: "rolling",
        5: "multiscale",
        6: "pdp",
    }

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

    # ----------------------------------------------------------- helpers
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
            tip = "Train a model first (Step 1)."
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
            else "Run Step 2 (Normalise) first — it provides Y_∞, the full-record baseline."
        )
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

    def _abandon_task(self) -> None:
        if self.runner.busy:
            self.runner.abandon()
            self.statusBar().showMessage(
                "Task abandoned — it finishes in the background and its result is discarded."
            )

    def _check_data(self) -> bool:
        if self.df_raw is None:
            QMessageBox.information(self, "No data", "Open a CSV file first.")
            return False
        if not self._selected_features():
            QMessageBox.information(self, "No features", "Tick at least one feature.")
            return False
        return True

    def _check_model(self) -> bool:
        if self.model is None or self.df_prep is None:
            QMessageBox.information(self, "No model", "Train a model first (Step 1).")
            return False
        return True

    def _show_error(self, tb: str) -> None:
        QMessageBox.critical(self, "Task failed", tb.splitlines()[-1] if tb else "Unknown error")
        self.log_view.appendPlainText(tb)

    def _n_cores(self, spin: NoWheelSpinBox) -> int | None:
        return None if spin.value() == 0 else spin.value()

    # -------------------------------------------------------------- tasks
    def _run_train(self) -> None:
        if not self._check_data():
            return
        from normet import build_model

        target = self.target_combo.currentText()
        features = self._selected_features()
        backend = self.backend_combo.currentText()
        if backend == _chronos.BACKEND:
            self._run_load_chronos(target, features)
            return
        if backend == "flaml":
            model_config = {
                "time_budget": self.budget_spin.value(),
                "estimator_list": self._selected_estimators() or ["lgbm"],
            }
        else:
            model_config = {"n_trials": self.budget_spin.value()}

        # The model must see the time features alongside the met predictors —
        # they carry the trend/seasonality that survives normalisation.
        # prepare_data() creates the columns; build_model() only trains on
        # what is listed here.
        self._pending_features = features
        self.runner.submit(
            f"train ({backend})",
            build_model,
            self._train_done,
            self._show_error,
            self.df_raw,
            target,
            backend=backend,
            covariates=features + list(TIME_VARS),
            split_method=self.split_combo.currentText(),
            train_fraction=self.train_fraction_spin.value(),
            model_config=model_config,
            seed=self.seed_spin.value(),
            verbose=True,
        )

    def _run_load_chronos(self, target: str, features: list[str]) -> None:
        """Bring up the zero-shot estimator instead of training one.

        The checkpoint load pulls ~500 MB on a cold cache, so it goes through
        the same worker thread every other long task uses.
        """
        met = [f for f in features if f not in TIME_VARS]
        if not met:
            QMessageBox.information(
                self,
                "No covariates",
                "Chronos-2 conditions on meteorology through its covariate channel. "
                "Tick at least one meteorological feature — without any, the run is a "
                "plain forecast, not a de-weathering.",
            )
            return
        self._pending_features = features
        self.statusBar().showMessage("Loading Chronos-2 — first run downloads the checkpoint…")
        self.runner.submit(
            f"load ({_chronos.BACKEND})",
            _chronos.load_estimator,
            self._train_done,
            self._show_error,
            self.df_raw,
            target,
            features,
            met,
            self._selected_device(),
        )

    def _render_chronos_model_tab(self) -> None:
        """Report what the checkpoint is and whether it actually reacts to weather.

        There is no parity plot or feature importance to show: nothing was
        fitted and there is no held-out set. The covariate-sensitivity
        diagnostic answers the question those plots exist to answer — is this
        model's output driven by meteorology, or is it autoregressing?
        """
        import matplotlib.pyplot as plt

        est = self.model
        met = [f for f in self.trained_features if f not in TIME_VARS]
        try:
            shift = _chronos.sensitivity(est, self.df_prep, met)
            verdict = _chronos.verdict_for(shift)
            self.chronos_check = (shift, verdict)
            header = ["Diagnostic", "Value"]
            rows = [
                ["Mean absolute median shift", f"{shift['mean_abs_shift']:.3f}"],
                ["Max absolute median shift", f"{shift['max_abs_shift']:.3f}"],
                ["As % of prediction", f"{shift['pct_of_prediction']:.2f}%"],
            ]
        except Exception as exc:
            self.chronos_check = None
            log.exception("Chronos-2 covariate sensitivity failed")
            verdict = ("warn", f"Could not run the covariate-sensitivity check: {exc}")
            header, rows = None, None

        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.text(
            0.5,
            0.5,
            "Chronos-2 is zero-shot — no training curve,\n"
            "no held-out parity plot, no feature importance.\n\n"
            "The table reports how far the forecast moves when the\n"
            "future meteorology is shuffled.",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.axis("off")
        fig.tight_layout()

        self.tab_model.show_result(
            fig,
            verdict=verdict,
            lines=[
                f"Checkpoint: {est.model_name}   |   device: {est.device}   |   "
                f"context: {est.context_length} h   |   horizon: {est.prediction_length} h",
                f"Covariates: {len(met)} met + 6 calendar   |   rows: {len(self.df_prep):,}",
            ],
            header=header,
            rows=rows,
        )
        self._log_history("load", f"chronos-2 on {est.device}")

    def _train_done(self, result: tuple[pd.DataFrame, object]) -> None:
        self.df_prep, self.model = result
        self.trained_features = list(self._pending_features)
        _fill_checklist(
            self.pdp_vars,
            self.trained_features + list(TIME_VARS),
            set(self.trained_features[:4]),
        )
        self._reset_pdp_toggle_buttons()
        # Offer every trained variable (incl. the auto-added time features)
        # for resampling; default to Met only — the standard deweathering
        # choice, which keeps trend/seasonality in the normalised series.
        met = {f for f in self.trained_features if f.lower() in MET_DEFAULTS or f in MET_DEFAULTS}
        _fill_checklist(
            self.norm_vars,
            self.trained_features + list(TIME_VARS),
            met or set(self.trained_features),
        )
        if self._is_chronos():
            self._render_chronos_model_tab()
        else:
            self._render_model_tab()
        self._sync_enabled()
        self.tabs.setCurrentWidget(self.tab_model)
        self.statusBar().showMessage(
            "Model trained — check its quality, then run Step 2 (normalisation)."
        )
        log.info("Training finished")

    def _render_model_tab(self) -> None:
        """Stats table + parity plot + feature importances, with an R² verdict."""
        import matplotlib.pyplot as plt
        import numpy as np

        from normet import ml_predict, modStats

        header: list[str] | None = None
        rows: list[list] | None = None
        stats = None
        try:
            stats = modStats(self.df_prep, self.model).round(3)
            self.results["train"] = stats
            tbl = stats.reset_index() if stats.index.name or stats.index.nlevels > 1 else stats
            header = [str(c) for c in tbl.columns]
            rows = [[v for v in row] for row in tbl.itertuples(index=False)]
        except Exception:
            log.exception("modStats failed")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

        test = self.df_prep
        if "set" in self.df_prep.columns:
            test = self.df_prep[self.df_prep["set"] == "testing"]
        if len(test) == 0:
            test = self.df_prep
        obs = test["value"].to_numpy()
        pred = np.asarray(ml_predict(self.model, test))
        ax1.scatter(obs, pred, s=6, alpha=0.35, color="#2c7bb6", edgecolors="none")
        lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
        ax1.plot(lims, lims, color="k", lw=0.8, ls="--")
        ss_res = float(((obs - pred) ** 2).sum())
        ss_tot = float(((obs - obs.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(((obs - pred) ** 2).mean()))
        ax1.set_xlabel("Observed (test set)")
        ax1.set_ylabel("Predicted")
        ax1.set_title(f"Parity — R²={r2:.3f}, RMSE={rmse:.2f}", fontsize=10, loc="left")
        ax1.grid(alpha=0.2)

        imp = self._feature_importance()
        if imp is not None:
            names, vals = imp
            order = np.argsort(vals)[-15:]
            ax2.barh([names[i] for i in order], [vals[i] for i in order], color="#2c7bb6")
            ax2.set_title("Feature importance", fontsize=10, loc="left")
        else:
            ax2.text(
                0.5,
                0.5,
                "Feature importance not available\nfor this backend",
                ha="center",
                va="center",
            )
            ax2.axis("off")
        fig.tight_layout()

        if r2 >= 0.7:
            verdict = (
                "ok",
                f"Good fit — test R² = {r2:.2f}. The normalisation results should be reliable.",
            )
        elif r2 >= 0.5:
            verdict = (
                "warn",
                f"Moderate fit — test R² = {r2:.2f}. Consider more features or a longer time budget.",
            )
        else:
            verdict = (
                "error",
                f"Poor fit — test R² = {r2:.2f}. Normalised results will be noisy; revisit features/target.",
            )

        backend = getattr(self.model, "backend", "?")
        self.tab_model.show_result(
            fig,
            verdict=verdict,
            lines=[
                f"Backend: {backend}   |   features: {len(self.trained_features)}   |   rows: {len(self.df_prep):,}"
            ],
            header=header,
            rows=rows,
        )
        self._log_history("train", f"R2={r2:.3f}")

    def _feature_importance(self):
        """Best-effort (names, values) from the trained model, else None."""
        model = self.model
        names = self.trained_features + list(TIME_VARS)
        try:
            if hasattr(model, "feature_importance"):  # LgbModel
                vals = list(model.feature_importance())
                names = list(getattr(model, "feature_names", names))
                if len(vals) == len(names):
                    return names, vals
            automl = getattr(model, "model", model)
            est = getattr(automl, "model", automl)  # flaml AutoML.model → sklearn wrapper
            fi = getattr(est, "feature_importances_", None)
            fn = getattr(automl, "feature_names_in_", None)
            if fn is None:
                fn = getattr(est, "feature_names_in_", None)
            if fi is not None:
                fn = list(fn) if fn is not None else names
                if len(fn) == len(fi):
                    return fn, list(fi)
        except Exception:
            pass
        return None

    def _run_normalise(self) -> None:
        if not self._check_model():
            return
        resample_vars = self._selected_norm_vars()
        if not resample_vars:
            QMessageBox.information(
                self,
                "No resample variables",
                "Tick at least one variable to resample in Step 2 "
                "(usually the meteorological ones).",
            )
            return
        if self._is_chronos():
            self.runner.submit(
                "de-weather (chronos-2)",
                _chronos.deweather,
                self._normalise_done,
                self._show_error,
                self.model,
                self.df_prep,
                [v for v in resample_vars if v not in TIME_VARS],
                self.norm_quantiles.isChecked(),
                self.norm_samples.value(),
            )
            return

        from normet import normalise

        quantiles = (0.05, 0.25, 0.75, 0.95) if self.norm_quantiles.isChecked() else None
        self.runner.submit(
            "normalise",
            normalise,
            self._normalise_done,
            self._show_error,
            self.df_prep,
            self.model,
            covariates=self.trained_features + list(TIME_VARS),
            variables_resample=resample_vars,
            n_samples=self.norm_samples.value(),
            n_cores=self._n_cores(self.norm_cores),
            return_quantiles=quantiles,
            verbose=True,
        )

    def _normalise_done(self, result: pd.DataFrame) -> None:
        self.results["normalise"] = result
        from normet import normalise_plot

        qcols = sorted(c for c in result.columns if c.startswith("q") and c[1:].isdigit())
        fig = normalise_plot(
            result,
            ci_low=qcols[0] if len(qcols) >= 2 else None,
            ci_high=qcols[-1] if len(qcols) >= 2 else None,
            ylabel=self.target_combo.currentText(),
            title=f"Meteorologically normalised {self.target_combo.currentText()}",
        )
        # Chronos-2 has no history to condition on for the first context_length
        # rows, so deweather() seeds them with the observed values: normalised
        # equals observed there by construction, not by result. Averaging across
        # them drags the reported effect toward zero, and on a record not much
        # longer than the context that is most of the series -- so say how many
        # rows are seeded and take the mean over the rest.
        seeded = min(int(self.model.context_length), len(result)) if self._is_chronos() else 0
        projected = result.iloc[seeded:]
        if seeded:
            fig.axvspan(
                result.index[0],
                result.index[seeded - 1],
                color="0.85",
                zorder=0,
                label="seeded (= observed)",
            )
            fig.legend(loc="best", fontsize=8)
        n_rv = len(self._selected_norm_vars())
        lines = []
        if len(projected):
            delta = projected["normalised"].mean() - projected["observed"].mean()
            lines.append(
                f"{len(result):,} timestamps, {self.norm_samples.value()} Monte-Carlo samples, "
                f"{n_rv} resampled variables   |   mean(normalised − observed) = {delta:+.2f}"
            )
        else:
            lines.append(
                f"{len(result):,} timestamps, {self.norm_samples.value()} Monte-Carlo samples, "
                f"{n_rv} resampled variables"
            )
        if seeded:
            pct = 100.0 * seeded / len(result)
            lines.append(
                f"⚠ The first {seeded:,} rows ({pct:.0f}% of the record) are seeded with the "
                f"observed values — Chronos-2 needs that much history before it can project, "
                f"so the curves coincide there by construction."
                + (
                    f" The mean above is over the remaining {len(projected):,} rows."
                    if len(projected)
                    else " Nothing was projected: load a record longer than "
                    f"{self.model.context_length:,} rows."
                )
            )
        self.tab_norm.show_result(fig, lines=lines)
        self._sync_enabled()
        self.tabs.setCurrentWidget(self.tab_norm)
        self.statusBar().showMessage(
            "Normalisation done — the meteorologically normalised series is on tab ③."
        )
        self._log_history("normalise", f"n_samples={self.norm_samples.value()}")
        log.info("Normalisation finished (%d rows)", len(result))

    def _run_decompose(self) -> None:
        if not self._check_model():
            return
        from normet import decompose

        method = self.decom_method.currentText()
        options: dict[str, object] = {}
        if method == "meteorology":
            met = [f for f in self.trained_features if f not in TIME_VARS]
            try:
                groups = parse_groups(self.decom_groups.text(), met)
            except ValueError as exc:
                QMessageBox.warning(self, "Groups", str(exc))
                return
            if groups:
                options["groups"] = groups
            if self.decom_attribution.currentText() != "auto":
                options["attribution"] = self.decom_attribution.currentText()
        if self._is_chronos():
            # The zero-shot path: each coalition is a de-weathering of the
            # estimator's own, over the meteorology it was loaded with.
            options.update(target="value", backend=_chronos.BACKEND)
            covariates = [f for f in self.trained_features if f not in TIME_VARS]
        else:
            covariates = self.trained_features + list(TIME_VARS)
        self.runner.submit(
            f"decompose ({method})",
            decompose,
            self._decompose_done,
            self._show_error,
            self.df_prep,
            self.model,
            method=method,
            covariates=covariates,
            n_samples=self.decom_samples.value(),
            **options,
        )

    def _run_finetune(self) -> None:
        """Adapt the loaded Chronos-2 checkpoint to this site's record (on a worker)."""
        if not self._check_model() or not self._is_chronos():
            return
        self.runner.submit(
            f"fine-tune ({self.ft_mode.currentText()}, {self.ft_steps.value()} steps)",
            _chronos.finetune,
            self._finetune_done,
            self._show_error,
            self.model,
            self.df_prep,
            self.ft_mode.currentText(),
            self.ft_steps.value(),
        )

    def _finetune_done(self, estimator: object) -> None:
        self.model = estimator
        self.results.pop("normalise", None)
        self.results.pop("decompose", None)
        self._render_chronos_model_tab()
        self._sync_enabled()
        self.tabs.setCurrentWidget(self.tab_model)
        self.statusBar().showMessage("Fine-tuned — normalise again to use the tuned weights.")
        log.info("Fine-tuning finished")

    def _decompose_done(self, result: pd.DataFrame) -> None:
        self.results["decompose"] = result
        method = self.decom_method.currentText()
        target = self.target_combo.currentText()

        from . import _decomp_plots as DP

        try:
            if method == "emission":
                fig = DP.emission_figure(result, target=target)
                d = DP.indexed_result(result)
                comps = [(c, lbl) for c, lbl in DP.EMI_COMPONENTS if c in d.columns]
                dominant = max(comps, key=lambda cl: d[cl[0]].std())[1] if comps else "n/a"
                lines = [
                    f"emi_total = emi_base + {' + '.join(c for c, _ in comps)} — "
                    f"largest time-scale: {dominant}."
                ]
            else:
                fig = DP.meteorology_figure(result, target=target)
                d = DP.indexed_result(result)
                contribs = DP.met_contribution_columns(d)
                dominant = max(contribs, key=lambda c: d[c].std()) if contribs else "n/a"
                what = "groups" if self.decom_groups.text().strip() else "meteorological variables"
                how = self.decom_attribution.currentText()
                if how == "auto":
                    how = "shapley" if self.decom_groups.text().strip() else "sequential"
                lines = [
                    f"met_total = observed − normalised, split across {len(contribs)} "
                    f"{what} ({how} attribution) — strongest: {dominant}."
                ]
            self.tab_decom.show_result(fig, lines=lines)
        except Exception:
            log.exception("Decomposition figure failed; falling back to a line plot")
            import matplotlib.pyplot as plt

            plot_df = result.set_index("date") if "date" in result.columns else result
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for column in plot_df.columns:
                if pd.api.types.is_numeric_dtype(plot_df[column]):
                    ax.plot(plot_df.index, plot_df[column], lw=0.9, label=column)
            ax.legend(loc="upper right", fontsize=8, ncols=2)
            self.tab_decom.show_result(fig)

        self.tabs.setCurrentWidget(self.tab_decom)
        self.statusBar().showMessage("Decomposition done.")
        self._log_history("decompose", method)
        log.info("Decomposition finished")

    def _run_rolling(self) -> None:
        if not self._check_model():
            return
        from normet import rolling

        self.runner.submit(
            "rolling",
            rolling,
            self._rolling_done,
            self._show_error,
            self.df_prep,
            self.model,
            covariates=self.trained_features + list(TIME_VARS),
            variables_resample=self._selected_norm_vars() or None,
            window_days=self.roll_window.value(),
            rolling_every=self.roll_step.value(),
            n_samples=self.roll_samples.value(),
        )

    def _rolling_done(self, result: pd.DataFrame) -> None:
        self.results["rolling"] = result
        import matplotlib.pyplot as plt

        rolling_cols = [c for c in result.columns if c.startswith("rolling_")]
        mean = result[rolling_cols].mean(axis=1, skipna=True)
        overlap = result[rolling_cols].notna().sum(axis=1)
        std = result[rolling_cols].std(axis=1, skipna=True).where(overlap >= 2)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        if "observed" in result.columns:
            ax.plot(
                result.index,
                result["observed"],
                lw=0.6,
                alpha=0.35,
                color="#9a9a9a",
                label="observed",
            )
        ax.plot(mean.index, mean.values, lw=1.3, color="#2c7bb6", label="rolling mean")
        if std.notna().any():
            ax.fill_between(
                mean.index,
                mean - std,
                mean + std,
                color="#2c7bb6",
                alpha=0.2,
                label="±1 SD across overlapping windows",
            )
        ax.set_title(
            f"Rolling normalisation — mean of {len(rolling_cols)} windows "
            f"({self.roll_window.value()} d window, {self.roll_step.value()} d step)",
            fontsize=10,
            loc="left",
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        self.tab_roll.show_result(fig)
        self.tabs.setCurrentWidget(self.tab_roll)
        self.statusBar().showMessage("Rolling normalisation done.")
        self._log_history("rolling", f"{len(rolling_cols)} windows")
        log.info("Rolling finished (%d windows)", len(rolling_cols))

    def _run_multiscale(self) -> None:
        if not self._check_model():
            return
        if "normalise" not in self.results:
            QMessageBox.information(
                self,
                "Run Step 2 first",
                "Multi-scale decomposition needs Y_∞ from Step 2 (Normalise) as its "
                "full-record baseline — run Normalise, then come back here.",
            )
            return
        from normet.gui._multiscale import compute_multiscale

        y_inf = self.results["normalise"]["normalised"]
        self.runner.submit(
            "multi-scale decomposition",
            compute_multiscale,
            self._multiscale_done,
            self._show_error,
            df_prep=self.df_prep,
            model=self.model,
            covariates=self.trained_features + list(TIME_VARS),
            variables_resample=self._selected_norm_vars() or None,
            y_inf=y_inf,
            fast_days=self.ms_fast.value(),
            meso_days=self.ms_meso.value(),
            slow_days=self.ms_slow.value(),
            n_samples=self.roll_samples.value(),
            n_cores=self._n_cores(self.norm_cores),
        )

    def _multiscale_done(self, result: dict) -> None:
        from ._multiscale_plots import available_bands, multiscale_figure

        fig = multiscale_figure(result, target=self.target_combo.currentText())
        bands = available_bands(result)
        lines = list(result.get("notes") or [])

        if bands:
            sigmas = {key: float(result[key].std()) for key, _ in bands}
            dominant = max(sigmas, key=sigmas.get)
            verdict = (
                "ok",
                f"Dominant residual timescale: {dominant} (σ={sigmas[dominant]:.2f}) — "
                f"{dict(bands)[dominant]}",
            )
        else:
            verdict = (
                "warn",
                "No band was computable — see the notes for why each scale was skipped.",
            )

        self.tab_multiscale.show_result(fig, verdict=verdict, lines=lines)
        self.tabs.setCurrentWidget(self.tab_multiscale)

        export = {k: v for k, v in result.items() if isinstance(v, pd.Series)}
        if export:
            self.results["multiscale"] = pd.DataFrame(export)

        self.statusBar().showMessage("Multi-scale decomposition done.")
        self._log_history("multiscale", f"{len(bands)}/3 bands computed")
        log.info("Multi-scale decomposition finished (%d/3 bands)", len(bands))

    def _run_pdp(self) -> None:
        if not self._check_model():
            return
        variables = _checked_items(self.pdp_vars)
        if not variables:
            QMessageBox.information(self, "No variables", "Tick at least one variable in Step 5.")
            return
        from normet import pdp

        self.runner.submit(
            "pdp",
            pdp,
            self._pdp_done,
            self._show_error,
            self.df_prep,
            self.model,
            var_list=variables,
            grid_points=self.pdp_grid.value(),
        )

    def _pdp_done(self, result: pd.DataFrame) -> None:
        self.results["pdp"] = result
        from normet import pdp_grid as _pdp_grid

        fig = _pdp_grid(result, title="Partial dependence")
        self.tab_pdp.show_result(fig)
        self.tabs.setCurrentWidget(self.tab_pdp)
        self.statusBar().showMessage("PDP done.")
        self._log_history("pdp", f"{result['variable'].nunique()} variables")
        log.info("PDP finished")

    # ---------------------------------------------------------------- SCM
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

    # ---------------------------------------------------------- Data Studio
    def open_data_studio(self) -> None:
        from .data_window import DataWindow

        if self._data_window is None:
            self._data_window = DataWindow(parent=self)
        self._data_window.show()
        self._data_window.raise_()
        self._data_window.activateWindow()

    def ingest_dataframe(self, df: pd.DataFrame, label: str) -> None:
        """Load an in-memory table (e.g. from the Data Studio) as the dataset."""
        df = df.copy()
        df["date"] = to_datetime_coerced(df["date"])
        self._ingest(df, label)
        self._log_command(commandlog.from_memory(label))

    # ---------------------------------------------------------- Transport Studio
    def open_transport_studio(self) -> None:
        from .trajectory_window import TrajectoryWindow

        if self._transport_window is None:
            self._transport_window = TrajectoryWindow(parent=self)
        if self.df_raw is not None:
            self._transport_window.offer_main_data(self.df_raw)
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
            QMessageBox.information(self, "No data", "Load a dataset in the main window first.")
            return 0
        from normet.io.trajectory import join_trajectory_features

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

    # ------------------------------------------------------------- misc

    def closeEvent(self, event) -> None:  # noqa: N802
        QSettings("apai-sys", "normet-gui").setValue("geometry", self.saveGeometry())
        logging.getLogger("normet").removeHandler(self._log_handler)
        self.assistant.shutdown()
        self.runner.shutdown()
        if self._scm_window is not None:
            self._scm_window.close()
        if self._data_window is not None:
            self._data_window.close()
        if self._transport_window is not None:
            self._transport_window.close()  # stops a HYSPLIT run before the app goes
        super().closeEvent(event)
