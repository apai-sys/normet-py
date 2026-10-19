"""Step 1's box: backend, split, budget, seed, device -- and fine-tuning.

Also the three questions the rest of the window asks about that box: which
estimators are ticked, where Chronos-2 runs, and whether it is the backend in
use.  They read the box this mixin built, which is an attribute of the window.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QWidget,
)

from . import _chronos
from ._lists import _checked_items
from ._widgets import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    hint_label,
    run_button,
)


class TrainPanelMixin(QMainWindow):
    def _build_train_box(self) -> QGroupBox:
        """Step 1: backend, split, budget, seed, device — and fine-tuning."""
        box = QGroupBox("Step 1 · Train model")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.backend_combo = NoWheelComboBox()
        self.backend_combo.addItems(["flaml", "lightgbm", _chronos.BACKEND])
        self.backend_combo.setToolTip(
            "flaml = AutoML search (uses the time budget below).\n"
            "lightgbm = LightGBM with random hyperparameter search.\n"
            "chronos-2 = zero-shot foundation model; nothing is trained."
        )
        chronos_ok, chronos_why = _chronos.chronos_availability()
        if not chronos_ok:
            # Grey the option out rather than letting the pick raise ImportError.
            item = self.backend_combo.model().item(2)
            item.setEnabled(False)
            item.setToolTip(chronos_why)
        self.backend_combo.currentTextChanged.connect(self._train_backend_changed)
        form.addRow("Backend", self.backend_combo)
        self.split_combo = NoWheelComboBox()
        self.split_combo.addItems(["random", "ts", "month_ts", "season_ts"])
        self.split_combo.setToolTip("How the train/test split is drawn.")
        self._split_label = QLabel("Split method")
        form.addRow(self._split_label, self.split_combo)
        self.train_fraction_spin = NoWheelDoubleSpinBox()
        self.train_fraction_spin.setRange(0.1, 0.95)
        self.train_fraction_spin.setSingleStep(0.05)
        self.train_fraction_spin.setValue(0.75)
        self._train_fraction_label = QLabel("Training fraction")
        form.addRow(self._train_fraction_label, self.train_fraction_spin)
        self.budget_spin = NoWheelSpinBox()
        self.budget_spin.setRange(5, 36_000)
        self.budget_spin.setValue(60)
        self.budget_spin.setSuffix(" s")
        self.budget_spin.setToolTip("AutoML search time (flaml backend).")
        self._budget_label = QLabel("Time budget")
        self._budget_values = {"flaml": 60, "lightgbm": 20}
        self._prev_backend = "flaml"
        # Parked value for the "Samples" spin box on the other side of the
        # chronos-2 divide; see _train_backend_changed.
        self._norm_samples_parked = _chronos.DEFAULT_SAMPLES
        self._decom_samples_parked = _chronos.DEFAULT_SAMPLES
        self._norm_samples_is_chronos = False
        form.addRow(self._budget_label, self.budget_spin)
        # flaml only: which AutoML estimators to search over.
        self.estimator_list = QListWidget()
        self.estimator_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.estimator_list.setMaximumHeight(96)
        self.estimator_list.setToolTip(
            "AutoML estimators flaml may try (searched within the time budget).\n"
            "lgbm is the fastest and usually sufficient; tick more to widen the search."
        )
        for name, tip in [
            ("lgbm", "LightGBM (default, fast)"),
            ("xgboost", "XGBoost"),
            ("xgb_limitdepth", "XGBoost, depth-limited"),
            ("rf", "Random forest"),
            ("extra_tree", "Extra trees"),
            ("histgb", "Histogram gradient boosting (sklearn)"),
        ]:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name == "lgbm" else Qt.CheckState.Unchecked)
            item.setToolTip(tip)
            self.estimator_list.addItem(item)
        self._estimator_label = QLabel("Estimators")
        form.addRow(self._estimator_label, self.estimator_list)
        self._train_form = form
        self.seed_spin = NoWheelSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(7_654_321)
        self._seed_label = QLabel("Seed")
        form.addRow(self._seed_label, self.seed_spin)
        self._chronos_hint = hint_label(
            "Chronos-2 is zero-shot — nothing is trained, so there is no budget, "
            "search space or train/test split. The first run downloads about "
            "500 MB of weights. Meteorology enters through the model's covariate "
            "channel; the ticked features below are what it sees."
        )
        self._chronos_hint.setVisible(False)
        form.addRow(self._chronos_hint)
        self._device_label = QLabel("Device")
        self.device_combo = NoWheelComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self.device_combo.setToolTip(
            "Where Chronos-2 runs.\n"
            "auto: CUDA, else Apple Silicon (mps), else CPU.\n"
            "Every Monte-Carlo sample is a full forward pass, so this is the "
            "difference between minutes and hours."
        )
        self._device_label.setVisible(False)
        self.device_combo.setVisible(False)
        form.addRow(self._device_label, self.device_combo)
        self.train_button = run_button(
            "▶  Train model",
            "Prepare the data and train the machine-learning model.\nEverything below needs a trained model.",
        )
        self.train_button.clicked.connect(self._run_train)
        form.addRow(self.train_button)
        # Chronos-2 only: adapt the loaded checkpoint to this site's own record.
        self._ft_box = QWidget()
        ft = QFormLayout(self._ft_box)
        ft.setContentsMargins(0, 0, 0, 0)
        self.ft_mode = NoWheelComboBox()
        self.ft_mode.addItems(["lora", "full"])
        if not _chronos.can_lora():
            self.ft_mode.model().item(0).setEnabled(False)
            self.ft_mode.setCurrentText("full")
        self.ft_mode.setToolTip(
            "lora: train small adapters, leave the base weights alone (needs peft).\n"
            "full: update all 119M parameters; overfits a single site's record more easily."
        )
        ft.addRow("Fine-tune", self.ft_mode)
        self.ft_steps = NoWheelSpinBox()
        self.ft_steps.setRange(1, 100_000)
        self.ft_steps.setValue(300)
        self.ft_steps.setToolTip(
            "Optimiser steps. On a CPU this takes a while; a GPU or Apple Silicon helps."
        )
        ft.addRow("Steps", self.ft_steps)
        self.ft_button = run_button(
            "▶  Fine-tune Chronos-2",
            "Adapt the loaded checkpoint to this site's own record. The tuned weights\n"
            "replace the model for this session; Load Chronos-2 again to go back.",
        )
        self.ft_button.clicked.connect(self._run_finetune)
        ft.addRow(self.ft_button)
        self._ft_box.setVisible(False)
        form.addRow(self._ft_box)
        return box

    def _train_backend_changed(self, backend: str) -> None:
        """The budget row means seconds for flaml, search trials for LightGBM;
        the estimator checklist only applies to flaml's AutoML search.

        Chronos-2 is zero-shot, so the whole training-control block — budget,
        estimator search space, train/test split, seed — has nothing to act on
        and is hidden rather than left to imply it does something.
        """
        is_chronos = backend == _chronos.BACKEND
        for label, widget in (
            (self._budget_label, self.budget_spin),
            (self._estimator_label, self.estimator_list),
            (self._split_label, self.split_combo),
            (self._train_fraction_label, self.train_fraction_spin),
            (self._seed_label, self.seed_spin),
        ):
            label.setVisible(not is_chronos)
            widget.setVisible(not is_chronos)
        self._chronos_hint.setVisible(is_chronos)
        self._ft_box.setVisible(is_chronos)
        self._device_label.setVisible(is_chronos)
        self.device_combo.setVisible(is_chronos)
        self.train_button.setText("▶  Load Chronos-2" if is_chronos else "▶  Train model")
        # A Monte-Carlo sample is a tree ensemble call under flaml/LightGBM and a
        # 2048-context transformer forward pass under Chronos-2, so the sensible
        # count differs by nearly two orders of magnitude. Swap the value when the
        # backend crosses that divide, parking the other side's so a user who set
        # it deliberately gets it back on the way over.
        if is_chronos != self._norm_samples_is_chronos:
            self._norm_samples_parked, restored = (
                self.norm_samples.value(),
                self._norm_samples_parked,
            )
            self.norm_samples.setValue(restored)
            self._decom_samples_parked, restored = (
                self.decom_samples.value(),
                self._decom_samples_parked,
            )
            self.decom_samples.setValue(restored)
            self._norm_samples_is_chronos = is_chronos
        # Chronos-2 decomposes by meteorological variable only: the calendar
        # components ride in the target's own history, which it never resamples.
        item = self.decom_method.model().item(0)
        item.setEnabled(not is_chronos)
        item.setToolTip(
            "Not on chronos-2: it reads the calendar off the target's own history."
            if is_chronos
            else ""
        )
        if is_chronos:
            self.decom_method.setCurrentText("meteorology")
        if is_chronos:
            return

        self._budget_values[self._prev_backend] = self.budget_spin.value()
        self._prev_backend = backend
        if backend == "flaml":
            self._budget_label.setText("Time budget")
            self.budget_spin.setRange(5, 36_000)
            self.budget_spin.setSuffix(" s")
            self.budget_spin.setToolTip("AutoML search time (flaml backend).")
        else:
            self._budget_label.setText("Search trials")
            self.budget_spin.setRange(1, 500)
            self.budget_spin.setSuffix("")
            self.budget_spin.setToolTip("Random hyperparameter-search trials (LightGBM backend).")
        self.budget_spin.setValue(self._budget_values.get(backend, 20))
        is_flaml = backend == "flaml"
        self._estimator_label.setVisible(is_flaml)
        self.estimator_list.setVisible(is_flaml)

    def _selected_estimators(self) -> list[str]:
        return _checked_items(self.estimator_list)

    def _selected_device(self) -> str | None:
        """``None`` means let :func:`normet.foundation.resolve_device` choose."""
        choice = self.device_combo.currentText()
        return None if choice == "auto" else choice

    def _is_chronos(self) -> bool:
        return self.backend_combo.currentText() == _chronos.BACKEND
