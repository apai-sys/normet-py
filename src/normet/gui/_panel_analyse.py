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
from ._lists import ALL_TIP, _checked_items, _fill_checklist, _toggle_all
from ._widgets import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    run_button,
    toggle_button,
    wrap_row,
)


class AnalysePanelMixin(QMainWindow):
    #: The eight meteorological predictors of the formal Shapley protocol
    #: (2**8 = 256 coalitions, 10 draws per coalition).
    FORMAL_SHAPLEY_VARS = ("ws", "blh", "t2m", "rh2m", "sp", "tcc", "tp", "ssrd")

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
        self.norm_button.setShortcut("Ctrl+1")
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
        self.formal_shapley_btn = QPushButton("Formal Shapley preset")
        self.formal_shapley_btn.setToolTip(
            "Set the decomposition to the project's formal eight-variable Shapley run:\n"
            "meteorology, exact Shapley attribution over all 2^k coalitions (256 for\n"
            "eight variables), 10 Monte Carlo draws per coalition, and each variable\n"
            "counted alone (no groups). Ticks the eight formal predictors\n"
            "(ws, blh, t2m, rh2m, sp, tcc, tp, ssrd) when no model is trained yet."
        )
        self.formal_shapley_btn.clicked.connect(self._formal_shapley_preset)
        df_.addRow(self.formal_shapley_btn)
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
        self.decom_button.setShortcut("Ctrl+2")
        df_.addRow(self.decom_button)
        return box

    def _formal_shapley_preset(self) -> None:
        """The project's formal Shapley protocol: meteorology, exact Shapley values
        (all 2**k coalitions, 256 for the eight formal predictors), 10 Monte Carlo
        draws per coalition, each variable alone.

        With no model trained yet the formal eight predictors are ticked, so the next
        training is the one the protocol was written for; with a model, its own
        meteorological variables are what the decomposition will use, and the note
        says whether they are the eight.
        """
        self.decom_method.setCurrentText("meteorology")
        self.decom_attribution.setCurrentText("shapley")
        self.decom_samples.setValue(10)
        self.decom_groups.clear()

        available = [self.feature_list.item(i).text() for i in range(self.feature_list.count())]
        missing = [v for v in self.FORMAL_SHAPLEY_VARS if v not in available]
        if getattr(self, "model", None) is not None:
            met = [f for f in self.trained_features if f not in TIME_VARS]
            source = "the trained model"
            if set(met) == set(self.FORMAL_SHAPLEY_VARS):
                selection = "its eight meteorological predictors are the formal set"
            else:
                selection = (
                    "the formal protocol has eight predictors (ws, blh, t2m, rh2m, sp, "
                    "tcc, tp, ssrd); tick those and train again to match it"
                )
        else:
            if missing:
                met = [v for v in self.FORMAL_SHAPLEY_VARS if v in available]
                selection = (
                    f"the table has no {', '.join(missing)}; tick the eight formal "
                    "predictors once they are loaded, then train"
                )
            else:
                met = list(self.FORMAL_SHAPLEY_VARS)
                _fill_checklist(self.feature_list, available, set(met))
                selection = "the eight formal predictors are ticked for the next training"
            source = "the features ticked for training"

        n = len(met)
        if 0 < n <= 10:
            coalitions = f"{2**n:,} coalition{'s' if n != 1 else ''} (exact)"
        else:
            coalitions = "too many for exact Shapley values — group or reduce the predictors"
        self.statusBar().showMessage(
            f"Formal Shapley preset: meteorology, 10 draws per variable, no groups; "
            f"{n} variable{'s' if n != 1 else ''} in {source} → {coalitions}. "
            f"{selection[:1].upper()}{selection[1:]}."
        )

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
        self.roll_button.setShortcut("Ctrl+3")
        rf.addRow(self.roll_button)
        return box

    def _build_multiscale_box(self) -> QGroupBox:
        """Multi-scale: three rolling widths, differenced against Step 2."""
        box = QGroupBox("Multi-scale decomposition")
        # The box's description lives in the title's hover: a sentence under the
        # name lengthened the page for what is read once and then forgotten.
        box.setToolTip(
            "Differences Y_fast − Y_meso − Y_slow − Y_∞ isolate the "
            "meteorological residual specific to each timescale — like wavelet "
            "detail coefficients or MSTL's trend/seasonal/remainder."
        )
        mf = QFormLayout(box)
        mf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
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
        self.ms_button.setShortcut("Ctrl+4")
        mf.addRow(self.ms_button)
        return box

    def _build_trend_box(self) -> QGroupBox:
        """The trend analysis of the decomposition's series, the observed target, or both:
        runs after each emission decomposition where its tick is on."""
        box = QGroupBox("Trend analysis")
        box.setToolTip(
            "Mann–Kendall and Theil–Sen on the series, with a block-bootstrap "
            "correction where its residuals are autocorrelated -- which is which "
            "p to quote said in the Trend tab. With more than one segment, a "
            "segmented fit instead: where the trend breaks, if it does."
        )
        tf = QFormLayout(box)
        tf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.trend_series = NoWheelComboBox()
        self.trend_series.addItems(
            [
                "emission part (emi_total)",
                "long-term trend (separated)",
                "meteorology part (met_total)",
                "observed target",
            ]
        )
        self.trend_series.setToolTip(
            "Which series to test. The decomposition's parts need a decomposition "
            "first; the observed target only the model.\n"
            "The emission part is the series the whole toolkit exists to trend: what "
            "weather does not explain. The long-term trend is the separated part of "
            "it: the recurring calendar cycles already taken out, so a cleaner fit -- "
            "the series the decomposition's Trend (long-term) panel draws."
        )
        tf.addRow("Series", self.trend_series)
        self.tre_seasonal = NoWheelComboBox()
        self.tre_seasonal.addItems(["auto", "none", "24", "7", "12"])
        self.tre_seasonal.setToolTip(
            "Sen's seasonal test pairs by calendar slot (hour 14 with hour 14, ...), "
            "which is the right test of a record that carries a repeating cycle.\n"
            "auto: from the record's step (hourly → 24, 30-minute → 48, daily → 7, "
            "monthly → 12), falling back to the classical test where the record is "
            "not a whole number of periods.\n"
            "24 hourly / 7 daily / 12 monthly where auto would guess wrong."
        )
        tf.addRow("Seasonal period", self.tre_seasonal)
        self.tre_segments = NoWheelSpinBox()
        self.tre_segments.setRange(1, 4)
        self.tre_segments.setValue(1)
        self.tre_segments.setToolTip(
            "One segment: the single-slope test (Mann–Kendall, Theil-Sen).\n"
            "Two or more: a segmented fit -- the record is split where the trend "
            "breaks, if it breaks, each segment reported with its own slope and "
            "interval; the break's p is the block bootstrap's, over the search.\n"
            "The fit runs on the Series chosen above: the long-term trend is the "
            "cleaner input (the cycles are in their own components), the emission "
            "part it for the whole de-weathered signal. Raising this moves a "
            "still-default emission-part choice to the long-term trend."
        )
        tf.addRow("Segments", self.tre_segments)
        self.tre_n_bootstrap = NoWheelSpinBox()
        self.tre_n_bootstrap.setRange(100, 20_000)
        self.tre_n_bootstrap.setValue(1000)
        tf.addRow("Bootstrap replicates", self.tre_n_bootstrap)
        # The protocol a published trend interval states: a monthly series, a
        # residual block bootstrap of twelve-month blocks, and a robust slope in
        # each draw.  Defaults keep the moving-block test on the series as given.
        self.tre_aggregate = NoWheelComboBox()
        self.tre_aggregate.addItems(["none", "monthly"])
        self.tre_aggregate.setToolTip(
            "Aggregate the series before testing. 'monthly' takes monthly means -- "
            "the scale a 12-point block bootstrap and the project's trend intervals "
            "are written for."
        )
        tf.addRow("Aggregation", self.tre_aggregate)
        self.tre_bootstrap = NoWheelComboBox()
        self.tre_bootstrap.addItems(["moving", "residual"])
        self.tre_bootstrap.setToolTip(
            "moving: resample contiguous blocks of the series (the default, repairs "
            "serial correlation in the values).\n"
            "residual: resample blocks of the fitted trend's residuals and add them "
            "back -- the trend is preserved (the published protocol, with 12-month "
            "blocks and 2,000 replicates)."
        )
        tf.addRow("Bootstrap scheme", self.tre_bootstrap)
        self.tre_estimator = NoWheelComboBox()
        self.tre_estimator.addItems(["ols", "theil-sen"])
        self.tre_estimator.setToolTip(
            "Slope recomputed in each bootstrap draw. OLS is O(n) and suits a long "
            "record; Theil-Sen is robust but O(n^2) per draw, so use it on a monthly "
            "or otherwise short series (the published protocol uses Theil-Sen)."
        )
        tf.addRow("Replicate slope", self.tre_estimator)
        self.tre_block = NoWheelSpinBox()
        self.tre_block.setRange(0, 1000)
        self.tre_block.setSpecialValueText("auto")
        self.tre_block.setValue(0)
        self.tre_block.setToolTip(
            "Bootstrap block length in points. 12 on a monthly series is twelve "
            "months (the published protocol); auto uses the Politis-Romano/week "
            "rule."
        )
        tf.addRow("Block length", self.tre_block)
        self.trend_after = QCheckBox("Run after each emission decomposition")
        # Unticked by default: the decomposition's figure is the answer to the
        # question it was run for, and the trend is a separate question the
        # analyst starts on purpose (or ticks this to make it follow).
        self.trend_after.setChecked(False)
        self.trend_after.setToolTip(
            "Each finished emission decomposition starts the test of its emission "
            "part, in the background, under this box's settings; the Segments row "
            "says whether it is the single-slope test or the segmented fit."
        )
        tf.addRow(self.trend_after)
        self.trend_button = run_button(
            "▶  Run trend analysis",
            "One segment: Mann–Kendall, Theil–Sen slope and its interval, the "
            "block-bootstrap correction where the record is autocorrelated.\n"
            "Two or more: the segmented fit -- where the trend breaks, each segment "
            "with its own slope and interval. Seconds, not minutes.",
        )
        self.trend_button.clicked.connect(self._run_trend)
        self.trend_button.setShortcut("Ctrl+5")
        self.tre_segments.valueChanged.connect(self._trend_segments_changed)
        tf.addRow(self.trend_button)
        return box

    def _trend_segments_changed(self, value: int) -> None:
        """Going segmented moves the default emission-part choice to its separated
        long-term trend: with the cycles in their own component columns, the
        segmentation is cleaner there (the record can go back; the long-term one
        stays until the analyst chooses otherwise). It fires on a change only,
        never at the box's build."""
        if value > 1 and self.trend_series.currentText().startswith("emission"):
            self.trend_series.setCurrentText("long-term trend (separated)")

    def _build_events_box(self) -> QGroupBox:
        """The event windows of a series: per-timestamp anomaly scores, and
        the consecutive stretches above a threshold flagged as events."""
        box = QGroupBox("Events & anomalies")
        box.setToolTip(
            "Scores every timestamp for how far it strays from the record's "
            "normal behaviour, then flags the consecutive stretches above a "
            "threshold -- the smoke episodes, the plume events. The scores "
            "and the flagged windows land on the Events tab."
        )
        ef = QFormLayout(box)
        ef.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.events_series = NoWheelComboBox()
        self.events_series.addItems(
            [
                "emission part (emi_total)",
                "long-term trend (separated)",
                "meteorology part (met_total)",
                "observed target",
            ]
        )
        self.events_series.setToolTip(
            "Which series to score. The decomposition's parts need a "
            "decomposition first; the observed target only the model.\n"
            "Anomalies of the emission part are departures from the "
            "meteorologically normal -- events the weather did not cause."
        )
        ef.addRow("Series", self.events_series)
        self.events_method = NoWheelComboBox()
        self.events_method.addItems(["iqr", "isolation", "stl_residual"])
        self.events_method.setToolTip(
            "iqr: |value - median| / IQR, the robust univariate score; the "
            "threshold below is in multiples of it.\n"
            "isolation: an IsolationForest over the value and its lags, which "
            "catches shapes a univariate score cannot -- a spike train, a "
            "level shift. It scores against its own boundary: the threshold "
            "row does not apply.\n"
            "stl_residual: the score of the seasonal-decomposition residual -- "
            "what stands out against the record's own cycle, not against its "
            "whole distribution."
        )
        ef.addRow("Method", self.events_method)
        self.events_k = NoWheelDoubleSpinBox()
        self.events_k.setRange(1.0, 10.0)
        self.events_k.setSingleStep(0.5)
        self.events_k.setValue(3.0)
        self.events_k.setToolTip(
            "The threshold, in multiples of the IQR (of the residual IQR under "
            "stl_residual): 3 flags what sits beyond median + 3*IQR.\n"
            "Lower flags more; on an autocorrelated hourly record even k=3 "
            "flags the tails of ordinary weather -- raise it, or raise the "
            "minimum length, until only the episodes remain."
        )
        ef.addRow("Threshold k", self.events_k)
        self.events_min_len = NoWheelSpinBox()
        self.events_min_len.setRange(1, 240)
        self.events_min_len.setValue(3)
        self.events_min_len.setToolTip(
            "Shortest flagged stretch, in steps of the record (hours for an "
            "hourly one). 1 keeps single-step spikes; 3 reads as episodes "
            "rather than instrument flickers."
        )
        ef.addRow("Min length", self.events_min_len)
        self.events_button = run_button(
            "▶  Find events",
            "Score every timestamp and flag the consecutive stretches above "
            "the threshold. iqr is instant; isolation and stl_residual fit a "
            "model over the record first and take a few seconds more.",
        )
        self.events_button.clicked.connect(self._run_events)
        self.events_button.setShortcut("Ctrl+6")
        self.events_method.currentTextChanged.connect(self._events_method_changed)
        ef.addRow(self.events_button)
        return box

    def _events_method_changed(self, method: str) -> None:
        """The forest scores against its own decision boundary: the k row has
        nothing to say under isolation, so it goes grey rather than sit
        there looking applied."""
        self.events_k.setEnabled(method != "isolation")

    def _build_unc_box(self) -> QGroupBox:
        """The spread's sources: weather, model form, their interaction.

        A model behind the box is not what it runs on -- the decomposition trains
        its own members from the table as loaded. What the model gates is the
        choice: an AutoML model at all (chronos-2's zero-shot forecasts carry
        no model form to vary), and the feature set and the ticked resample
        variables it was trained with.
        """
        box = QGroupBox("Uncertainty decomposition")
        box.setToolTip(
            "Of the band a model's estimate carries, how much is the met "
            "resampling and how much the model form? An ensemble of members, "
            "each trained on its own seed, predicts the same shared met "
            "draws; the per-date spread splits orthogonally into weather, "
            "model and their interaction. Minutes, not seconds: every "
            "member is a fresh fit."
        )
        uf = QFormLayout(box)
        uf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.unc_samples = NoWheelSpinBox()
        self.unc_samples.setRange(2, 10_000)
        self.unc_samples.setValue(300)
        self.unc_samples.setToolTip(
            "Met resamples shared by every member (the weather factor).\n"
            "The Normalise page's count, kept here so the two agree."
        )
        uf.addRow("Samples", self.unc_samples)
        self.unc_models = NoWheelSpinBox()
        self.unc_models.setRange(2, 30)
        self.unc_models.setValue(4)
        self.unc_models.setToolTip(
            "Ensemble members, each trained on its own seed (the model-form "
            "factor). Two is the floor: with one member there is no model-form "
            "variance to attribute. The cost is this many fits, so the default "
            "stays small."
        )
        uf.addRow("Models", self.unc_models)
        self.unc_backend = NoWheelComboBox()
        self.unc_backend.addItems(["lightgbm", "flaml"])
        self.unc_backend.setToolTip(
            "How each member is trained. lightgbm: a fast random search on "
            "every member.\nflaml: an AutoML search per member, at the cost of "
            "its budget each."
        )
        uf.addRow("Member backend", self.unc_backend)
        self.unc_button = run_button(
            "▶  Run uncertainty decomposition",
            "Split the total per-date spread into weather, model form and "
            "their\ninteraction, each member on the same met draws.",
        )
        self.unc_button.clicked.connect(self._run_unc)
        self.unc_button.setShortcut("Ctrl+7")
        uf.addRow(self.unc_button)
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
        self.pdp_button.setShortcut("Ctrl+8")
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
        panel = getattr(self, "param_panel", None)  # not yet, while the panel is built
        if on and panel is not None:
            panel.refit()  # the groups' table, shown, may need the panel wider
