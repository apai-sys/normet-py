"""Steps 2-5 and the multi-scale box: what to run, and with how many samples.

The boxes are built here and so are the handlers that switch parts of them on
and off -- the attribution box under a meteorology decomposition, the
resample list's Met only button, the PDP toggles.  A mixin of
:class:`~.main_window.MainWindow`: the boxes are attributes of the window, so
the run handlers in ``main_window`` find them by name.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from ._constants import MET_DEFAULTS, TIME_VARS
from ._lists import ALL_TIP, _checked_items, _toggle_all
from ._widgets import (
    NoWheelComboBox,
    NoWheelSpinBox,
    hint_label,
    run_button,
    toggle_button,
    wrap_row,
)


class AnalysePanelMixin(QMainWindow):
    def _build_normalise_box(self) -> QGroupBox:
        """Step 2: which variables to resample, and how many samples."""
        box = QGroupBox("Normalise")
        nf = QFormLayout(box)
        nf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.norm_vars = QListWidget()
        self.norm_vars.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.norm_vars.setMaximumHeight(110)
        self.norm_vars.setToolTip(
            "Which variables are resampled in the Monte-Carlo (their influence is\n"
            "averaged out of the target). Ticked = normalised away; unticked stay\n"
            "at their observed values. Lists every trained variable, including the\n"
            "auto-added time features — resampling those too removes trend and\n"
            "seasonality as well. Default (Met only) is the usual deweathering choice."
        )
        nf.addRow("Resample vars", self.norm_vars)
        nv_row = QHBoxLayout()
        b_nv_met = QPushButton("Met only")
        b_nv_met.setToolTip("Tick only the recognised meteorological variables.")
        b_nv_met.clicked.connect(lambda: self._set_norm_vars_met_only())
        b_nv_all = QPushButton("All")
        b_nv_all.setToolTip(ALL_TIP)
        b_nv_all.clicked.connect(lambda: _toggle_all(self.norm_vars))
        nv_row.addWidget(b_nv_met)
        nv_row.addWidget(b_nv_all)
        nf.addRow(wrap_row(nv_row))  # the whole row: under an empty label it widened the panel
        self.norm_samples = NoWheelSpinBox()
        self.norm_samples.setRange(1, 10_000)
        self.norm_samples.setValue(300)
        self.norm_samples.setToolTip(
            "Monte-Carlo resamples of the meteorological conditions.\nMore = smoother but slower."
        )
        nf.addRow("Samples", self.norm_samples)
        self.norm_cores = NoWheelSpinBox()
        self.norm_cores.setRange(0, 64)
        self.norm_cores.setValue(0)
        self.norm_cores.setSpecialValueText("auto")
        nf.addRow("CPU cores", self.norm_cores)
        self.norm_quantiles = QCheckBox("5–95 % quantile band")
        self.norm_quantiles.setToolTip("Also return the Monte-Carlo quantiles and shade the band.")
        nf.addRow(self.norm_quantiles)
        self.norm_button = run_button(
            "▶  Run normalisation",
            "Remove the meteorological signal from the target\n(counterfactual under average meteorology).",
        )
        self.norm_button.clicked.connect(self._run_normalise)
        nf.addRow(self.norm_button)
        return box

    def _build_decompose_box(self) -> QGroupBox:
        """Step 3: emission or meteorology, attribution and groups."""
        box = QGroupBox("Decompose")
        df_ = QFormLayout(box)
        df_.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.decom_method = NoWheelComboBox()
        self.decom_method.addItems(["emission", "meteorology"])
        self.decom_method.setToolTip(
            "emission: split the normalised series into time scales\n"
            "(trend / seasonal / weekly / diurnal / noise).\n"
            "meteorology: split observed − normalised into the contribution\n"
            "of each individual meteorological variable."
        )
        df_.addRow("Method", self.decom_method)
        self.decom_attribution = NoWheelComboBox()
        self.decom_attribution.addItems(["auto", "sequential", "shapley"])
        self.decom_attribution.setToolTip(
            "How observed − normalised is split among the variables (meteorology only).\n"
            "sequential: fix them one at a time; the split depends on the order.\n"
            "shapley: average each one's effect over every order (2^k normalisations\n"
            "for k variables or groups, up to 10).\n"
            "auto: shapley when groups are given, else sequential."
        )
        df_.addRow("Attribution", self.decom_attribution)
        from ._groups import GroupEditor

        # The variables in named groups (meteorology only): a contribution per group.
        # Shown only for meteorology; the emission decomposition has no use for it.
        self._decom_form = df_
        self.decom_groups = GroupEditor()
        self._decom_groups_label = QLabel("Groups")
        df_.addRow(self._decom_groups_label)
        df_.addRow(self.decom_groups)
        self.decom_method.currentTextChanged.connect(self._decom_method_changed)
        self.decom_samples = NoWheelSpinBox()
        self.decom_samples.setRange(1, 10_000)
        self.decom_samples.setValue(300)
        df_.addRow("Samples", self.decom_samples)
        self._decom_method_changed(self.decom_method.currentText())
        self.decom_button = run_button(
            "▶  Run decomposition",
            "Split the observed series into additive contributions\n(nested normalisation).",
        )
        self.decom_button.clicked.connect(self._run_decompose)
        df_.addRow(self.decom_button)
        return box

    def _build_rolling_box(self) -> QGroupBox:
        """Step 4: window, step and samples for the rolling normalisation."""
        box = QGroupBox("Rolling normalisation")
        rf = QFormLayout(box)
        rf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.roll_window = NoWheelSpinBox()
        self.roll_window.setRange(1, 365)
        self.roll_window.setValue(14)
        self.roll_window.setSuffix(" d")
        rf.addRow("Window", self.roll_window)
        self.roll_step = NoWheelSpinBox()
        self.roll_step.setRange(1, 365)
        self.roll_step.setValue(7)
        self.roll_step.setSuffix(" d")
        rf.addRow("Step", self.roll_step)
        self.roll_samples = NoWheelSpinBox()
        self.roll_samples.setRange(1, 10_000)
        self.roll_samples.setValue(100)
        rf.addRow("Samples", self.roll_samples)
        self.roll_button = run_button(
            "▶  Run rolling normalisation",
            "Windowed re-normalisation — reveals slow changes in the\n"
            "emission signal. Resamples the variables ticked under Normalise.",
        )
        self.roll_button.clicked.connect(self._run_rolling)
        rf.addRow(self.roll_button)
        return box

    def _build_multiscale_box(self) -> QGroupBox:
        """Multi-scale: three rolling widths, differenced against Step 2."""
        box = QGroupBox("Multi-scale decomposition")
        mf = QFormLayout(box)
        mf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        mf.addRow(
            hint_label(
                "Differences Y_fast − Y_meso − Y_slow − Y_∞ isolate the "
                "meteorological residual specific to each timescale — like wavelet "
                "detail coefficients or MSTL's trend/seasonal/remainder."
            )
        )
        self.ms_fast = NoWheelSpinBox()
        self.ms_fast.setRange(1, 3650)
        self.ms_fast.setValue(14)
        self.ms_fast.setSuffix(" d")
        mf.addRow("Fast window", self.ms_fast)
        self.ms_meso = NoWheelSpinBox()
        self.ms_meso.setRange(1, 3650)
        self.ms_meso.setValue(90)
        self.ms_meso.setSuffix(" d")
        mf.addRow("Meso window", self.ms_meso)
        self.ms_slow = NoWheelSpinBox()
        self.ms_slow.setRange(1, 3650)
        self.ms_slow.setValue(365)
        self.ms_slow.setSuffix(" d")
        mf.addRow("Slow window", self.ms_slow)
        self._ms_button_tip = (
            "Runs rolling normalisation at three window widths and differences\n"
            "them against each other and against the normalisation's Y_∞ — needs\n"
            "the normalisation to have been run first, and is roughly 3× the cost\n"
            "of a single Rolling run."
        )
        self.ms_button = run_button("▶  Multi-scale decomposition", self._ms_button_tip)
        self.ms_button.clicked.connect(self._run_multiscale)
        mf.addRow(self.ms_button)
        return box

    def _build_pdp_box(self) -> QGroupBox:
        """Step 5: partial-dependence curves for the ticked variables."""
        box = QGroupBox("Partial dependence")
        pv = QVBoxLayout(box)
        pv.addWidget(QLabel("Variables"))
        self.pdp_vars = QListWidget()
        self.pdp_vars.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.pdp_vars.setMaximumHeight(120)
        self.pdp_vars.setToolTip("Filled with the trained features after training.")
        pv.addWidget(self.pdp_vars)
        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Grid points"))
        self.pdp_grid = NoWheelSpinBox()
        self.pdp_grid.setRange(5, 500)
        self.pdp_grid.setValue(50)
        grid_row.addWidget(self.pdp_grid)
        self.pdp_time_btn = toggle_button(
            "Time variables",
            "Toggle date_unix / day_julian / weekday / hour:\n"
            "click to tick all of them, click again to untick.",
        )
        self.pdp_time_btn.toggled.connect(lambda checked: self._toggle_pdp_vars(checked, TIME_VARS))
        self.pdp_met_btn = toggle_button(
            "Met only",
            "Toggle the recognised meteorological variables:\n"
            "click to tick all of them, click again to untick.",
        )
        self.pdp_met_btn.toggled.connect(lambda checked: self._toggle_pdp_vars(checked, None))
        grid_row.addStretch(1)
        pv.addLayout(grid_row)
        # A row of their own, as wide as the run button under them.
        pick_row = QHBoxLayout()
        pick_row.setContentsMargins(0, 0, 0, 0)
        for btn in (self.pdp_time_btn, self.pdp_met_btn):
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            pick_row.addWidget(btn, 1)
        pv.addLayout(pick_row)
        self.pdp_button = run_button(
            "▶  Compute PDP",
            "Partial-dependence curves: how the model's prediction\nresponds to each variable.",
        )
        self.pdp_button.clicked.connect(self._run_pdp)
        pv.addWidget(self.pdp_button)
        return box

    def _set_norm_vars_met_only(self) -> None:
        for i in range(self.norm_vars.count()):
            item = self.norm_vars.item(i)
            is_met = item.text().lower() in MET_DEFAULTS or item.text() in MET_DEFAULTS
            item.setCheckState(Qt.CheckState.Checked if is_met else Qt.CheckState.Unchecked)

    def _selected_norm_vars(self) -> list[str]:
        return _checked_items(self.norm_vars)

    def _toggle_pdp_vars(self, checked: bool, names: tuple[str, ...] | None) -> None:
        """Tick/untick a subset of Step 5's variable list.

        *names* selects an explicit subset (time variables); pass None to
        target the recognised meteorological variables instead.
        """
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.pdp_vars.count()):
            item = self.pdp_vars.item(i)
            text = item.text()
            matches = (
                text in names
                if names is not None
                else (text.lower() in MET_DEFAULTS or text in MET_DEFAULTS)
            )
            if matches:
                item.setCheckState(state)

    def _reset_pdp_toggle_buttons(self) -> None:
        """Un-press the Time-variables/Met-only toggles after the variable
        list is (re)filled, so their pressed state doesn't lie about what's
        actually ticked."""
        for btn in (self.pdp_time_btn, self.pdp_met_btn):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

    def _decom_method_changed(self, method: str) -> None:
        on = method == "meteorology"
        self.decom_attribution.setEnabled(on)
        self._decom_form.setRowVisible(self._decom_groups_label, on)
        self._decom_form.setRowVisible(self.decom_groups, on)
