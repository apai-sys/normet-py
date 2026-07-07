"""Synthetic Control (SCM) Studio — counterfactual analysis on panel data.

The synthetic-control workflow has genuinely different inputs from the
normalisation workflow (a long panel of many units instead of one site's
time series), so it lives in its own window, in the same design language
as the main one: parameter panel on the left (Data → Columns → Design →
Estimator → Inference), result tabs on the right with verdict banners.

Tabs
----
① Panel        — outcome by unit, treated highlighted; balance check.
Fit            — observed vs. synthetic + effect path + donor weights
                 (``scm_dashboard``) with ``scm_diagnostics`` verdict.
Placebo Space  — each donor treated as a pseudo-treated unit; permutation
                 p-value and placebo bands (``placebo_in_space``).
Placebo Time   — fake earlier cutoffs; event-time bands (``placebo_in_time``).
Uncertainty    — jackknife / bootstrap effect bands (``uncertainty_bands``).
All Units      — ``scm_all`` batch: post-cutoff ATT of every unit ranked.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDateEdit,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._widgets import (
    CanvasTab,
    NoWheelComboBox,
    NoWheelSpinBox,
    hint_label,
    run_button,
)
from .workers import TaskRunner

log = logging.getLogger(__name__)

TREATED_COLOR = "#d7191c"
DONOR_COLOR = "#9a9a9a"

BACKEND_TIPS = {
    "scm": "Augmented SCM (ridge) — the recommended default.",
    "mlscm": "ML-augmented SCM (AutoML) — experimental, slower.",
    "abadie": "Classic Abadie SCM (simplex weights).",
    "did": "Difference-in-differences baseline.",
    "mcnnm": "Matrix-completion (MC-NNM) estimator.",
    "robust": "Robust SCM (de-noised donor matrix).",
    "bayesian": "Bayesian simplex SCM — needs pymc, gives credible bands.",
}


def _checked(widget: QListWidget) -> list[str]:
    return [
        widget.item(i).text()
        for i in range(widget.count())
        if widget.item(i).checkState() == Qt.CheckState.Checked
    ]


class SCMWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Normet — Synthetic Control Studio")
        self.resize(1340, 880)

        self.df: pd.DataFrame | None = None
        self._main_df: pd.DataFrame | None = None
        self.result: dict | None = None  # {"synthetic": df, "weights": ...}
        self.diagnostics: dict | None = None
        self._tab_data: dict[int, pd.DataFrame] = {}
        self._updating = False

        self.runner = TaskRunner(self)
        self.runner.started.connect(self._task_started)
        self.runner.finished.connect(self._task_finished)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_panel())
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 4, 4, 0)
        rv.addWidget(self._build_action_bar())
        rv.addWidget(self._build_tabs(), 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([370, 970])
        self.setCentralWidget(splitter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setMaximumWidth(220)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip(
            "Abandon the running task: the UI unlocks now and the result is\n"
            "discarded when the background computation eventually finishes."
        )
        self.cancel_btn.clicked.connect(self._abandon)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Load a long panel CSV (date, unit, outcome) or the example.")

        self._sync_enabled()

    # ------------------------------------------------------------- left panel
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)

        # ---- Data
        data_box = QGroupBox("Panel data")
        dv = QVBoxLayout(data_box)
        row = QHBoxLayout()
        b_open = QPushButton("Open CSV…")
        b_open.setToolTip("A long panel: one row per (date, unit) with the outcome in a column.")
        b_open.clicked.connect(self._open_csv)
        b_example = QPushButton("Load example")
        b_example.setToolTip(
            "Synthetic weekly SO₂ panel with a ~35 % step-down in the treated\n"
            "unit ('2+26 cities') from 2015-10-23 — reproduces the tutorial."
        )
        b_example.clicked.connect(self.load_example)
        row.addWidget(b_open)
        row.addWidget(b_example)
        dv.addLayout(row)
        self.use_main_btn = QPushButton("Use main-window data")
        self.use_main_btn.setEnabled(False)
        self.use_main_btn.setToolTip(
            "Bring over the dataset loaded in the main window\n(needs a unit column with several distinct values)."
        )
        self.use_main_btn.clicked.connect(self._use_main_data)
        dv.addWidget(self.use_main_btn)
        self.file_label = hint_label("No panel loaded", small=False)
        dv.addWidget(self.file_label)
        v.addWidget(data_box)

        # ---- Columns
        col_box = QGroupBox("Columns")
        cf = QFormLayout(col_box)
        cf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.date_combo = NoWheelComboBox()
        self.unit_combo = NoWheelComboBox()
        self.outcome_combo = NoWheelComboBox()
        self.outcome_combo.setToolTip(
            "Tip: use a meteorologically normalised outcome (e.g. SO2wn from\n"
            "the Normalise step) so meteorological differences don't\n"
            "masquerade as effects."
        )
        cf.addRow("Date", self.date_combo)
        cf.addRow("Unit", self.unit_combo)
        cf.addRow("Outcome", self.outcome_combo)
        self.date_combo.currentTextChanged.connect(self._columns_changed)
        self.unit_combo.currentTextChanged.connect(self._columns_changed)
        self.outcome_combo.currentTextChanged.connect(self._redraw_panel_tab)
        v.addWidget(col_box)

        # ---- Design
        design_box = QGroupBox("Design")
        gf = QFormLayout(design_box)
        gf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.treated_combo = NoWheelComboBox()
        self.treated_combo.setToolTip("The unit that received the intervention.")
        self.treated_combo.currentTextChanged.connect(self._treated_changed)
        gf.addRow("Treated unit", self.treated_combo)
        self.cutoff_edit = QDateEdit()
        self.cutoff_edit.setCalendarPopup(True)
        self.cutoff_edit.setDisplayFormat("yyyy-MM-dd")
        self.cutoff_edit.setToolTip(
            "Intervention date: the synthetic control is fitted on data before it."
        )
        self.cutoff_edit.dateChanged.connect(lambda _d: self._redraw_panel_tab())
        gf.addRow("Cutoff date", self.cutoff_edit)
        gf.addRow(QLabel("Donor pool"))
        self.donor_list = QListWidget()
        self.donor_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.donor_list.setMinimumHeight(120)
        self.donor_list.setMaximumHeight(170)
        self.donor_list.setToolTip("Untreated units used to build the synthetic counterfactual.")
        gf.addRow(self.donor_list)
        dbrow = QHBoxLayout()
        b_all = QPushButton("All")
        b_none = QPushButton("None")
        b_all.clicked.connect(lambda: self._set_donors(True))
        b_none.clicked.connect(lambda: self._set_donors(False))
        dbrow.addWidget(b_all)
        dbrow.addWidget(b_none)
        gf.addRow(dbrow)
        v.addWidget(design_box)

        # ---- Estimator
        est_box = QGroupBox("Estimator")
        ef = QFormLayout(est_box)
        ef.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.backend_combo = NoWheelComboBox()
        for name, tip in BACKEND_TIPS.items():
            self.backend_combo.addItem(name)
            self.backend_combo.setItemData(
                self.backend_combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
            )
        self.backend_combo.setToolTip("Which synthetic-control estimator to use.")
        self.backend_combo.currentTextChanged.connect(self._backend_changed)
        ef.addRow("Backend", self.backend_combo)
        self.ml_backend_combo = NoWheelComboBox()
        self.ml_backend_combo.addItems(["flaml", "lightgbm"])
        ef.addRow("ML backend", self.ml_backend_combo)
        self.ml_budget = NoWheelSpinBox()
        self.ml_budget.setRange(5, 3600)
        self.ml_budget.setValue(15)
        self.ml_budget.setSuffix(" s")
        self.ml_budget.setToolTip(
            "AutoML time budget per fit (flaml). Placebo tests refit many\ntimes — keep this small."
        )
        ef.addRow("Time budget", self.ml_budget)
        self.bayes_draws = NoWheelSpinBox()
        self.bayes_draws.setRange(100, 10_000)
        self.bayes_draws.setValue(1000)
        self.bayes_draws.setToolTip("Posterior draws per chain (bayesian backend).")
        ef.addRow("Draws", self.bayes_draws)
        self.fit_btn = run_button(
            "▶  Run synthetic control",
            "Fit the counterfactual for the treated unit and show\nobserved vs. synthetic, the effect path and donor weights.",
        )
        self.fit_btn.clicked.connect(self._run_fit)
        ef.addRow(self.fit_btn)
        v.addWidget(est_box)

        # ---- Inference
        inf_box = QGroupBox("Inference")
        inf = QFormLayout(inf_box)
        inf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.placebo_space_btn = run_button(
            "▶  Placebo in space",
            "Treat each donor as if it were treated; the real effect should\nstand out from the placebo distribution (permutation p-value).",
        )
        self.placebo_space_btn.clicked.connect(self._run_placebo_space)
        inf.addRow(self.placebo_space_btn)

        self.pt_min_pre = NoWheelSpinBox()
        self.pt_min_pre.setRange(5, 100_000)
        self.pt_min_pre.setValue(30)
        self.pt_min_pre.setToolTip("Minimum pre-period length (rows per unit) for a fake cutoff.")
        inf.addRow("Min pre-period", self.pt_min_pre)
        self.pt_every = NoWheelSpinBox()
        self.pt_every.setRange(1, 10_000)
        self.pt_every.setValue(7)
        self.pt_every.setToolTip("Place a fake cutoff every N time steps.")
        inf.addRow("Placebo every", self.pt_every)
        self.placebo_time_btn = run_button(
            "▶  Placebo in time",
            "Re-run the fit at fake earlier cutoffs; a real intervention shows\nan effect only after the true cutoff.",
        )
        self.placebo_time_btn.clicked.connect(self._run_placebo_time)
        inf.addRow(self.placebo_time_btn)

        self.unc_method = NoWheelComboBox()
        self.unc_method.addItems(["jackknife", "bootstrap"])
        self.unc_method.setToolTip(
            "jackknife: leave one donor out.\nbootstrap: resample donors B times."
        )
        inf.addRow("Band method", self.unc_method)
        self.unc_b = NoWheelSpinBox()
        self.unc_b.setRange(10, 2000)
        self.unc_b.setValue(100)
        self.unc_b.setToolTip("Bootstrap replicates (ignored for jackknife).")
        inf.addRow("B", self.unc_b)
        self.unc_btn = run_button(
            "▶  Uncertainty bands",
            "Donor-resampling uncertainty bands around the effect path.",
        )
        self.unc_btn.clicked.connect(self._run_uncertainty)
        inf.addRow(self.unc_btn)

        self.all_btn = run_button(
            "▶  All units (batch)",
            "Fit a synthetic control for every unit and rank the post-cutoff\neffects — the treated unit should be an outlier.",
        )
        self.all_btn.clicked.connect(self._run_all_units)
        inf.addRow(self.all_btn)
        v.addWidget(inf_box)

        v.addStretch(1)
        self._run_buttons = [
            self.fit_btn,
            self.placebo_space_btn,
            self.placebo_time_btn,
            self.unc_btn,
            self.all_btn,
        ]

        self._backend_changed(self.backend_combo.currentText())

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(390)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(4, 0, 4, 0)
        self.exp_btn = QPushButton("⬇  Export current tab data")
        self.exp_btn.setToolTip("Export the data behind the current tab as CSV.")
        self.exp_btn.clicked.connect(self._export_current)
        self.exp_btn.setEnabled(False)
        h.addWidget(self.exp_btn)
        h.addStretch(1)
        return bar

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tab_panel = CanvasTab("Load a panel CSV to preview the outcome by unit.")
        self.tabs.addTab(self.tab_panel, "① Panel")
        self.tab_fit = CanvasTab("Run the synthetic control to see observed vs. synthetic here.")
        self.tabs.addTab(self.tab_fit, "Fit")
        self.tab_pspace = CanvasTab(
            "Run 'Placebo in space' to test the effect against the donor distribution."
        )
        self.tabs.addTab(self.tab_pspace, "Placebo Space")
        self.tab_ptime = CanvasTab("Run 'Placebo in time' to test against fake earlier cutoffs.")
        self.tabs.addTab(self.tab_ptime, "Placebo Time")
        self.tab_unc = CanvasTab(
            "Run 'Uncertainty bands' for donor-resampling bands around the effect."
        )
        self.tabs.addTab(self.tab_unc, "Uncertainty")
        self.tab_all = CanvasTab("Run 'All units' to rank every unit's pseudo-effect.")
        self.tabs.addTab(self.tab_all, "All Units")
        self.tabs.currentChanged.connect(lambda i: self.exp_btn.setEnabled(i in self._tab_data))
        return self.tabs

    # ---------------------------------------------------------------- data
    def offer_main_data(self, df: pd.DataFrame) -> None:
        """Called by the main window so its dataset can be reused here."""
        self._main_df = df
        self.use_main_btn.setEnabled(self._panel_candidate(df) is not None)

    @staticmethod
    def _panel_candidate(df: pd.DataFrame) -> str | None:
        """Return a plausible unit column (repeated categorical), else None."""
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) or c.lower() == "date":
                continue
            nun = df[c].nunique(dropna=True)
            if 3 <= nun <= max(3, len(df) // 4):
                return c
        return None

    def _use_main_data(self) -> None:
        if self._main_df is not None:
            self._ingest(self._main_df.copy(), "main-window data")

    def _open_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open panel CSV", "", "CSV files (*.csv *.csv.gz)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to read CSV", str(exc))
            return
        self._ingest(df, os.path.basename(path))

    def load_example(self) -> None:
        from ._examples import (
            SCM_EXAMPLE_CUTOFF,
            SCM_EXAMPLE_OUTCOME,
            SCM_EXAMPLE_TREATED,
            make_scm_example,
        )

        self._ingest(make_scm_example(), "Example panel (synthetic weekly SO₂)")
        # Reproduce the tutorial design out of the box.
        self.outcome_combo.setCurrentText(SCM_EXAMPLE_OUTCOME)
        self.treated_combo.setCurrentText(SCM_EXAMPLE_TREATED)
        self.cutoff_edit.setDate(QDate.fromString(SCM_EXAMPLE_CUTOFF, "yyyy-MM-dd"))
        self.statusBar().showMessage(
            "Example loaded — '2+26 cities' steps down ~35 % after 2015-10-23. Click ▶ Run synthetic control."
        )

    def _ingest(self, df: pd.DataFrame, label: str) -> None:
        self.df = df
        self.result = None
        self.diagnostics = None
        self._tab_data.clear()

        self._updating = True
        date_cols = [
            c for c in df.columns if c.lower() in ("date", "datetime", "time", "timestamp")
        ]
        obj_cols = [
            c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c not in date_cols
        ]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        self.date_combo.clear()
        self.date_combo.addItems(date_cols + [c for c in df.columns if c not in date_cols])
        self.unit_combo.clear()
        self.unit_combo.addItems(obj_cols + [c for c in df.columns if c not in obj_cols])
        cand = self._panel_candidate(df)
        if cand:
            self.unit_combo.setCurrentText(cand)
        self.outcome_combo.clear()
        self.outcome_combo.addItems(num_cols)
        wn = next((c for c in num_cols if c.lower().endswith("wn")), None)
        if wn:
            self.outcome_combo.setCurrentText(wn)
        self._updating = False

        self.file_label.setText(f"{label}\n{len(df):,} rows × {df.shape[1]} columns")
        self.file_label.setStyleSheet("")
        self._columns_changed()
        self.statusBar().showMessage(
            "Panel loaded — check the columns, pick treated unit and cutoff, then ▶ Run synthetic control."
        )
        # macOS returns focus to the parent (main) window after the file
        # dialog closes; keep the Studio in front.
        self.raise_()
        self.activateWindow()

    # ------------------------------------------------------------- selectors
    def _columns_changed(self, _text: str = "") -> None:
        if self._updating or self.df is None:
            return
        self._updating = True
        try:
            unit_col = self.unit_combo.currentText()
            date_col = self.date_combo.currentText()
            units: list[str] = []
            if unit_col in self.df.columns:
                units = [str(u) for u in pd.unique(self.df[unit_col].dropna())]
            self.treated_combo.clear()
            self.treated_combo.addItems(units)
            self._rebuild_donors()

            if date_col in self.df.columns:
                dates = pd.to_datetime(self.df[date_col], errors="coerce").dropna()
                if len(dates):
                    lo, hi = dates.min(), dates.max()
                    self.cutoff_edit.setDateRange(
                        QDate(lo.year, lo.month, lo.day), QDate(hi.year, hi.month, hi.day)
                    )
                    mid = lo + (hi - lo) * 0.6
                    self.cutoff_edit.setDate(QDate(mid.year, mid.month, mid.day))
        finally:
            self._updating = False
        self._sync_enabled()
        self._redraw_panel_tab()

    def _treated_changed(self, _text: str = "") -> None:
        if not self._updating:
            self._rebuild_donors()
            self._redraw_panel_tab()

    def _rebuild_donors(self) -> None:
        treated = self.treated_combo.currentText()
        unit_col = self.unit_combo.currentText()
        self.donor_list.clear()
        if self.df is None or unit_col not in self.df.columns:
            return
        for u in pd.unique(self.df[unit_col].dropna()):
            u = str(u)
            if u == treated:
                continue
            item = QListWidgetItem(u)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.donor_list.addItem(item)

    def _set_donors(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.donor_list.count()):
            self.donor_list.item(i).setCheckState(state)

    def _backend_changed(self, backend: str) -> None:
        is_ml = backend == "mlscm"
        is_bayes = backend == "bayesian"
        self.ml_backend_combo.setEnabled(is_ml)
        self.ml_budget.setEnabled(is_ml)
        self.bayes_draws.setEnabled(is_bayes)
        for b in (self.placebo_space_btn, self.placebo_time_btn, self.unc_btn, self.all_btn):
            b.setEnabled(not is_bayes and self.df is not None and not self.runner.busy)
            if is_bayes:
                b.setToolTip(
                    "Placebo / batch inference is not available for the bayesian backend\n— it provides its own posterior credible bands."
                )

    # ------------------------------------------------------------ panel tab
    def _redraw_panel_tab(self, *_args) -> None:
        if self._updating or self.df is None:
            return
        date_col = self.date_combo.currentText()
        unit_col = self.unit_combo.currentText()
        outcome = self.outcome_combo.currentText()
        if not all(c in self.df.columns for c in (date_col, unit_col, outcome)):
            return
        import matplotlib.pyplot as plt

        d = self.df[[date_col, unit_col, outcome]].copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col])
        try:
            wide = d.pivot_table(index=date_col, columns=unit_col, values=outcome, aggfunc="mean")
        except Exception:
            return
        treated = self.treated_combo.currentText()

        fig, ax = plt.subplots(figsize=(10, 4.6))
        cols = list(wide.columns)
        for c in cols:
            if str(c) == treated:
                continue
            ax.plot(wide.index, wide[c], color=DONOR_COLOR, lw=0.7, alpha=0.35)
        if treated in map(str, cols):
            tc = next(c for c in cols if str(c) == treated)
            ax.plot(wide.index, wide[tc], color=TREATED_COLOR, lw=1.8, label=f"treated: {treated}")
            ax.legend(loc="upper right", frameon=False)
        cutoff = pd.Timestamp(self.cutoff_edit.date().toPython())
        ax.axvline(cutoff, color="k", ls=":", lw=1)
        ax.set_ylabel(outcome)
        ax.set_title(f"{outcome} by unit ({wide.shape[1]} units)", fontsize=10, loc="left")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        n_cells = wide.shape[0] * wide.shape[1]
        missing = float(wide.isna().sum().sum()) / n_cells * 100 if n_cells else 0.0
        pre_rows = int((wide.index < cutoff).sum())
        if missing > 10:
            verdict = (
                "warn",
                f"Unbalanced panel — {missing:.1f} % of (date × unit) cells are missing.",
            )
        elif pre_rows < 10:
            verdict = (
                "warn",
                f"Only {pre_rows} pre-cutoff time steps — move the cutoff later for a better fit.",
            )
        else:
            verdict = (
                "ok",
                f"Balanced panel: {wide.shape[1]} units × {wide.shape[0]} time steps, "
                f"{pre_rows} pre-cutoff ({missing:.1f} % missing).",
            )
        self.tab_panel.show_result(fig, verdict=verdict)
        self._tab_data[self.tabs.indexOf(self.tab_panel)] = wide
        self.exp_btn.setEnabled(self.tabs.currentIndex() in self._tab_data)

    # --------------------------------------------------------------- helpers
    def _design(self) -> dict | None:
        """Collect and validate the current design; None (with a dialog) if invalid."""
        if self.df is None:
            QMessageBox.information(self, "No data", "Load a panel CSV first.")
            return None
        date_col = self.date_combo.currentText()
        unit_col = self.unit_combo.currentText()
        outcome = self.outcome_combo.currentText()
        treated = self.treated_combo.currentText()
        donors = _checked(self.donor_list)
        if not treated:
            QMessageBox.information(self, "No treated unit", "Pick the treated unit.")
            return None
        if len(donors) < 2:
            QMessageBox.information(self, "Donor pool too small", "Tick at least two donor units.")
            return None
        cutoff = self.cutoff_edit.date().toString("yyyy-MM-dd")
        d = self.df[[date_col, unit_col, outcome]].copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col])
        d[unit_col] = d[unit_col].astype(str)
        return {
            "df": d,
            "date_col": date_col,
            "unit_col": unit_col,
            "outcome_col": outcome,
            "treated_unit": treated,
            "donors": donors,
            "cutoff_date": cutoff,
        }

    def _ml_kwargs(self) -> dict:
        ml_backend = self.ml_backend_combo.currentText()
        if ml_backend == "flaml":
            model_config = {
                "time_budget": self.ml_budget.value(),
                "metric": "r2",
                "estimator_list": ["lgbm"],
                "task": "regression",
            }
        else:
            model_config = None
        return {"backend": ml_backend, "model_config": model_config}

    def _infer_kwargs(self) -> dict:
        """kwargs forwarded to run_scm-based inference helpers."""
        backend = self.backend_combo.currentText()
        kw: dict = {"scm_backend": backend}
        if backend == "mlscm":
            kw.update(self._ml_kwargs())
        return kw

    def _sync_enabled(self) -> None:
        busy = self.runner.busy
        has_df = self.df is not None
        self.fit_btn.setEnabled(not busy and has_df)
        is_bayes = self.backend_combo.currentText() == "bayesian"
        for b in (self.placebo_space_btn, self.placebo_time_btn, self.unc_btn, self.all_btn):
            b.setEnabled(not busy and has_df and not is_bayes)
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

    def _abandon(self) -> None:
        if self.runner.busy:
            self.runner.abandon()
            self.statusBar().showMessage(
                "Task abandoned — it finishes in the background and its result is discarded."
            )

    def _show_error(self, tb: str) -> None:
        QMessageBox.critical(self, "Task failed", tb.splitlines()[-1] if tb else "Unknown error")
        log.error("%s", tb)

    def _show_tab(self, tab: CanvasTab, data: pd.DataFrame | None = None) -> None:
        idx = self.tabs.indexOf(tab)
        if data is not None:
            self._tab_data[idx] = data
        self.tabs.setCurrentWidget(tab)
        self.exp_btn.setEnabled(idx in self._tab_data)

    def _export_current(self) -> None:
        data = self._tab_data.get(self.tabs.currentIndex())
        if data is None:
            QMessageBox.information(self, "Nothing to export", "Run this tab's analysis first.")
            return
        name = self.tabs.tabText(self.tabs.currentIndex()).strip("① ").lower().replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export data", f"normet_scm_{name}.csv", "CSV (*.csv)"
        )
        if path:
            data.to_csv(path)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
        self.raise_()
        self.activateWindow()

    # ------------------------------------------------------------------ fit
    def _run_fit(self) -> None:
        design = self._design()
        if design is None:
            return
        backend = self.backend_combo.currentText()
        ml_kwargs = self._ml_kwargs()
        draws = self.bayes_draws.value()

        def job():
            if backend == "mlscm":
                from normet.causal import mlscm

                synth = mlscm(
                    design["df"],
                    date_col=design["date_col"],
                    unit_col=design["unit_col"],
                    outcome_col=design["outcome_col"],
                    treated_unit=design["treated_unit"],
                    donors=design["donors"],
                    cutoff_date=design["cutoff_date"],
                    **ml_kwargs,
                )
                return {"synthetic": synth, "weights": None}
            if backend == "bayesian":
                from normet.causal import bayesian_scm

                return bayesian_scm(
                    design["df"],
                    date_col=design["date_col"],
                    unit_col=design["unit_col"],
                    outcome_col=design["outcome_col"],
                    treated_unit=design["treated_unit"],
                    donors=design["donors"],
                    cutoff_date=design["cutoff_date"],
                    draws=draws,
                )
            from normet.causal import BACKENDS

            return BACKENDS[backend](
                df=design["df"],
                date_col=design["date_col"],
                unit_col=design["unit_col"],
                outcome_col=design["outcome_col"],
                treated_unit=design["treated_unit"],
                donors=design["donors"],
                cutoff_date=design["cutoff_date"],
            )

        self._last_design = design
        self.runner.submit(f"synthetic control ({backend})", job, self._fit_done, self._show_error)

    def _fit_done(self, result: dict) -> None:
        self.result = result
        design = self._last_design
        cutoff = design["cutoff_date"]
        backend = self.backend_combo.currentText()

        from normet import scm_dashboard
        from normet.causal import scm_diagnostics

        try:
            self.diagnostics = scm_diagnostics(result, cutoff_date=cutoff)
        except Exception:
            log.exception("scm_diagnostics failed")
            self.diagnostics = None

        if backend == "bayesian":
            from normet import plot_bayesian_scm

            fig = plot_bayesian_scm(
                result, cutoff_date=cutoff, title=f"Bayesian SCM — {design['treated_unit']}"
            )
        else:
            fig = scm_dashboard(
                result,
                cutoff_date=cutoff,
                diagnostics=self.diagnostics,
                title=f"{design['treated_unit']} vs. synthetic ({backend})",
            )

        verdict = None
        lines = []
        header = rows = None
        if self.diagnostics:
            d = self.diagnostics
            pre_r2 = d.get("pre_r2", float("nan"))
            att = d.get("att", float("nan"))
            if pre_r2 >= 0.8:
                verdict = ("ok", f"Good pre-treatment fit (R² = {pre_r2:.2f}) — ATT = {att:+.2f}.")
            elif pre_r2 >= 0.5:
                verdict = (
                    "warn",
                    f"Moderate pre-treatment fit (R² = {pre_r2:.2f}) — interpret ATT = {att:+.2f} with caution.",
                )
            else:
                verdict = (
                    "error",
                    f"Poor pre-treatment fit (R² = {pre_r2:.2f}) — the counterfactual is unreliable.",
                )
            keys = [
                "pre_n",
                "pre_rmse",
                "pre_mae",
                "pre_r2",
                "post_n",
                "att",
                "att_cum",
                "post_rmse",
                "hhi",
                "effective_n_donors",
            ]
            header = ["metric", "value"]
            rows = [
                [k, f"{d[k]:.3f}" if isinstance(d.get(k), float) else d.get(k)]
                for k in keys
                if k in d
            ]
            lines = ["Next: run the Inference tools (left) to test significance."]

        self.tab_fit.show_result(fig, verdict=verdict, lines=lines, header=header, rows=rows)
        self._show_tab(self.tab_fit, result.get("synthetic"))
        self.statusBar().showMessage(
            "Fit done — check the pre-cutoff overlap, then run Placebo / Uncertainty for significance."
        )

    # ------------------------------------------------------------- inference
    def _run_placebo_space(self) -> None:
        design = self._design()
        if design is None:
            return
        kwargs = self._infer_kwargs()

        def job():
            from normet.causal import effect_bands_space, placebo_in_space

            out = placebo_in_space(
                design["df"],
                date_col=design["date_col"],
                unit_col=design["unit_col"],
                outcome_col=design["outcome_col"],
                treated_unit=design["treated_unit"],
                donors=design["donors"],
                cutoff_date=design["cutoff_date"],
                **kwargs,
            )
            bands = effect_bands_space(out, level=0.95, method="quantile")
            return out, bands

        self._last_design = design
        self.runner.submit("placebo in space", job, self._placebo_space_done, self._show_error)

    def _placebo_space_done(self, payload) -> None:
        out, bands = payload
        design = self._last_design
        cutoff = pd.Timestamp(design["cutoff_date"])
        import matplotlib.pyplot as plt

        from normet.causal import plot_effect_with_bands

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        placebos = out.get("placebos") or {}
        for name, dfp in placebos.items():
            eff = dfp["effect"] if "effect" in dfp.columns else dfp.iloc[:, 0]
            ax1.plot(dfp.index, eff, color=DONOR_COLOR, lw=0.7, alpha=0.4)
        tr = out["treated"]
        ax1.plot(tr.index, tr["effect"], color=TREATED_COLOR, lw=1.8, label="treated")
        ax1.axvline(cutoff, color="k", ls=":", lw=1)
        ax1.axhline(0, color="k", lw=0.5)
        ax1.set_title(f"Placebo effects ({len(placebos)} donors)", fontsize=10, loc="left")
        ax1.legend(loc="upper left", frameon=False)
        ax1.grid(alpha=0.2)
        plot_effect_with_bands(
            bands, cutoff_date=cutoff, title="Effect with 95 % placebo bands", ax=ax2
        )
        fig.tight_layout()

        p = out.get("p_value", float("nan"))
        if p == p and p <= 0.05:
            verdict = (
                "ok",
                f"Effect is significant against the placebo distribution (p = {p:.3f}).",
            )
        elif p == p and p <= 0.15:
            verdict = (
                "warn",
                f"Borderline significance (p = {p:.3f}) — more donors or a longer post-period would help.",
            )
        else:
            verdict = (
                "warn",
                f"Effect is not distinguishable from placebo variation (p = {p:.3f}).",
            )

        self.tab_pspace.show_result(fig, verdict=verdict)
        self._show_tab(self.tab_pspace, bands)
        self.statusBar().showMessage("Placebo-in-space done.")

    def _run_placebo_time(self) -> None:
        design = self._design()
        if design is None:
            return
        kwargs = self._infer_kwargs()
        min_pre = self.pt_min_pre.value()
        every = self.pt_every.value()

        def job():
            from normet.causal import effect_bands_time, placebo_in_time

            out = placebo_in_time(
                design["df"],
                date_col=design["date_col"],
                unit_col=design["unit_col"],
                outcome_col=design["outcome_col"],
                treated_unit=design["treated_unit"],
                donors=design["donors"],
                cutoff_date=design["cutoff_date"],
                min_pre_period=min_pre,
                placebo_every=every,
                **kwargs,
            )
            if not out.get("placebos"):
                return out, None
            bands = effect_bands_time(out, level=0.95, method="quantile", return_segments=True)
            return out, bands

        self._last_design = design
        self.runner.submit("placebo in time", job, self._placebo_time_done, self._show_error)

    def _placebo_time_done(self, payload) -> None:
        out, bands_dict = payload
        design = self._last_design
        cutoff = pd.Timestamp(design["cutoff_date"])
        if bands_dict is None:
            # Each fake cutoff needs the full post-period to fit before the
            # true cutoff, so it is only feasible when the cutoff sits late
            # in the sample.
            tr = out["treated"]
            n_post = int((tr.index >= cutoff).sum())
            self.tab_ptime.show_result(
                None,
                verdict=(
                    "warn",
                    f"No valid fake cutoffs: the {n_post}-step post-period must fit entirely "
                    f"before the true cutoff (with ≥ {self.pt_min_pre.value()} pre steps).",
                ),
                lines=[
                    "Reduce 'Min pre-period', or use placebo-in-time only when the cutoff "
                    "lies in the later part of the sample. Placebo-in-space (above) does "
                    "not have this constraint."
                ],
            )
            self._show_tab(self.tab_ptime)
            self.statusBar().showMessage("Placebo-in-time: no valid fake cutoffs for this design.")
            return
        import matplotlib.pyplot as plt

        from normet.causal import plot_effect_with_bands

        bands = bands_dict["bands"].copy()
        tr = out["treated"]
        post_eff = tr.loc[tr.index >= cutoff, "effect"].to_numpy()
        k = min(len(bands), len(post_eff))
        bands = bands.iloc[:k]
        bands["effect"] = post_eff[:k]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
        seg = bands_dict.get("segments")
        if seg is not None:
            for c in seg.columns:
                ax1.plot(seg.index, seg[c], color=DONOR_COLOR, lw=0.7, alpha=0.4)
        ax1.plot(bands.index, bands["effect"], color=TREATED_COLOR, lw=1.8, label="true cutoff")
        ax1.axhline(0, color="k", lw=0.5)
        ax1.set_title(
            f"Event-time effects: true cutoff vs. {len(bands_dict.get('cutoffs', []))} fake cutoffs",
            fontsize=10,
            loc="left",
        )
        ax1.set_xlabel("Event time (steps after cutoff)")
        ax1.legend(loc="upper left", frameon=False)
        ax1.grid(alpha=0.2)
        plot_effect_with_bands(
            bands, cutoff_date=None, title="Effect with 95 % placebo-in-time bands", ax=ax2
        )
        fig.tight_layout()

        p = out.get("p_value", float("nan"))
        n_plc = len(out.get("placebos") or {})
        min_p = 1.0 / (n_plc + 1) if n_plc else float("nan")
        if p == p and p <= 0.05:
            verdict = ("ok", f"Post-cutoff effect is significant vs. fake cutoffs (p = {p:.3f}).")
        elif p == p and abs(p - min_p) < 1e-9:
            verdict = (
                "warn",
                f"Effect is the largest among all {n_plc} fake cutoffs (p = {p:.3f} — the smallest "
                f"achievable with {n_plc} placebos; lower 'Placebo every' for more resolution).",
            )
        elif p == p:
            verdict = ("warn", f"Effect is not clearly larger than at fake cutoffs (p = {p:.3f}).")
        else:
            verdict = (
                "warn",
                "Too few valid fake cutoffs for a p-value — lower 'Min pre-period' or 'Placebo every'.",
            )

        self.tab_ptime.show_result(fig, verdict=verdict)
        self._show_tab(self.tab_ptime, bands)
        self.statusBar().showMessage("Placebo-in-time done.")

    def _run_uncertainty(self) -> None:
        design = self._design()
        if design is None:
            return
        kwargs = self._infer_kwargs()
        method = self.unc_method.currentText()
        n_b = self.unc_b.value()

        def job():
            from normet.causal import uncertainty_bands

            return uncertainty_bands(
                design["df"],
                date_col=design["date_col"],
                unit_col=design["unit_col"],
                outcome_col=design["outcome_col"],
                treated_unit=design["treated_unit"],
                donors=design["donors"],
                cutoff_date=design["cutoff_date"],
                method=method,
                B=n_b,
                **kwargs,
            )

        self._last_design = design
        self.runner.submit(
            f"uncertainty bands ({method})", job, self._uncertainty_done, self._show_error
        )

    def _uncertainty_done(self, out: dict) -> None:
        design = self._last_design
        method = self.unc_method.currentText()
        if out.get("low") is None:
            self.tab_unc.show_result(
                None,
                verdict=(
                    "warn",
                    "Not enough successful donor-resampled fits to form bands — "
                    "try the other method or more donors.",
                ),
            )
            self._show_tab(self.tab_unc)
            return
        from normet.causal import plot_uncertainty_bands

        from ._widgets import as_figure

        ret = plot_uncertainty_bands(
            out, cutoff_date=design["cutoff_date"], title=f"Effect with {method} bands"
        )
        fig = as_figure(ret)
        try:
            fig.tight_layout()
        except Exception:
            pass

        tr, low, high = out["treated"], out["low"], out["high"]
        post = tr.index >= pd.Timestamp(design["cutoff_date"])
        eff = tr.loc[post, "effect"]
        sig = ((low[post] > 0) | (high[post] < 0)).mean() * 100
        verdict = (
            "ok" if sig >= 50 else "warn",
            f"Mean post-cutoff effect {eff.mean():+.2f}; bands exclude zero on "
            f"{sig:.0f} % of post-cutoff days ({method}).",
        )
        export = pd.DataFrame({"effect": tr["effect"], "low": low, "high": high})
        self.tab_unc.show_result(fig, verdict=verdict)
        self._show_tab(self.tab_unc, export)
        self.statusBar().showMessage("Uncertainty bands done.")

    def _run_all_units(self) -> None:
        design = self._design()
        if design is None:
            return
        kwargs = self._infer_kwargs()

        def job():
            from normet.causal import scm_all

            return scm_all(
                design["df"],
                date_col=design["date_col"],
                unit_col=design["unit_col"],
                outcome_col=design["outcome_col"],
                donors=None,  # every unit uses all others as donors
                cutoff_date=design["cutoff_date"],
                **kwargs,
            )

        self._last_design = design
        self.runner.submit("all units (batch)", job, self._all_units_done, self._show_error)

    def _all_units_done(self, df_all: pd.DataFrame) -> None:
        design = self._last_design
        cutoff = pd.Timestamp(design["cutoff_date"])
        unit_col, date_col = design["unit_col"], design["date_col"]
        treated = design["treated_unit"]
        import matplotlib.pyplot as plt

        post = df_all[pd.to_datetime(df_all[date_col]) >= cutoff]
        att = post.groupby(unit_col)["effect"].mean().sort_values()

        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.28 * len(att))))
        colors = [TREATED_COLOR if str(u) == treated else DONOR_COLOR for u in att.index]
        ax.barh([str(u) for u in att.index], att.values, color=colors)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Post-cutoff mean effect (observed − synthetic)")
        ax.set_title("Pseudo-effects: every unit fitted as if treated", fontsize=10, loc="left")
        fig.tight_layout()

        rank = int((att.abs() >= abs(att.get(treated, float("nan")))).sum())
        n = len(att)
        pseudo_p = rank / n if n else float("nan")
        if treated in att.index and pseudo_p <= 0.1:
            verdict = (
                "ok",
                f"'{treated}' has the #{rank} largest |effect| of {n} units (pseudo-p ≈ {pseudo_p:.2f}).",
            )
        elif treated in att.index:
            verdict = (
                "warn",
                f"'{treated}' ranks #{rank} of {n} by |effect| (pseudo-p ≈ {pseudo_p:.2f}) — weak evidence.",
            )
        else:
            verdict = ("warn", "Treated unit missing from the batch results.")

        table = att.rename("ATT").reset_index()
        self.tab_all.show_result(
            fig,
            verdict=verdict,
            header=[unit_col, "ATT"],
            rows=[
                [str(u), f"{v:+.3f}"] for u, v in att.sort_values(key=abs, ascending=False).items()
            ][:12],
        )
        self._show_tab(self.tab_all, table)
        self.statusBar().showMessage("Batch run done — the treated unit should be an outlier.")

    # ------------------------------------------------------------------ misc
    def closeEvent(self, event) -> None:  # noqa: N802
        self.runner.shutdown()
        super().closeEvent(event)
