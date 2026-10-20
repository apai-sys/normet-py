"""Training, Chronos-2 loading and the Model tab.

Step 1's run handler, what it does when the task comes back, and the two
renderers for the Model tab (one for a trained estimator, one for Chronos-2's
zero-shot check).  Mixed into :class:`~.main_window.MainWindow`.
"""

from __future__ import annotations

import logging

import pandas as pd
from PySide6.QtWidgets import QMainWindow

from normet import style as S

from . import _chronos
from ._constants import MET_DEFAULTS, TIME_VARS
from ._lists import _fill_checklist

log = logging.getLogger(__name__)


#: modStats' columns and subsets, as the Model tab's table names them.
_STATS_HEADER = {"set": "Subset", "p_level": "Significance"}
_STATS_SET = {"training": "Training", "testing": "Testing", "all": "All"}


class TrainRunMixin(QMainWindow):
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
        if backend == "flaml":
            # FLAML searches for as long as it is given: its share done is the time.
            self.task_clock.expect(self.budget_spin.value())

    def _run_load_chronos(self, target: str, features: list[str]) -> None:
        """Bring up the zero-shot estimator instead of training one.

        The checkpoint load pulls ~500 MB on a cold cache, so it goes through
        the same worker thread every other long task uses.
        """
        met = [f for f in features if f not in TIME_VARS]
        if not met:
            self.notify(
                "Chronos-2 conditions on meteorology through its covariate channel. "
                "Tick at least one meteorological feature — without any, the run is a "
                "plain forecast, not a de-weathering.",
                key="chronos-covariates",
                kind="warn",
                actions=(("Met only", self._check_met_only),),
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

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(
            0.5,
            0.5,
            "Chronos-2 is zero-shot — no training curve,\n"
            "no held-out parity plot, no feature importance.\n\n"
            "The table reports how far the forecast moves when the\n"
            "future meteorology is shuffled.",
            ha="center",
            va="center",
            color=S.NEUTRAL,
        )
        ax.axis("off")

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
        # The meteorological decomposition's groups: the model's variables, each in the
        # group it had (a config read before training applies now).
        self.decom_groups.set_variables([f for f in self.trained_features if f not in TIME_VARS])
        self.param_panel.refit()  # the table's names may need the panel wider
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
        self.statusBar().showMessage("Model trained — check its quality, then normalise.")
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
            # The table in words, not modStats' column names: the subset a row is about
            # and the stars' column (the metrics' own abbreviations stay).
            header = [_STATS_HEADER.get(str(c), str(c)) for c in tbl.columns]
            rows = [
                [_STATS_SET.get(v, v) if isinstance(v, str) else v for v in row]
                for row in tbl.itertuples(index=False)
            ]
        except Exception:
            log.exception("modStats failed")

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1, 1.15]}
        )
        name = S.formula(self.target_combo.currentText())

        test = self.df_prep
        if "set" in self.df_prep.columns:
            test = self.df_prep[self.df_prep["set"] == "testing"]
        if len(test) == 0:
            test = self.df_prep
        obs = test["value"].to_numpy()
        pred = np.asarray(ml_predict(self.model, test))
        lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
        pad = 0.03 * (hi - lo)
        lims = [lo - pad, hi + pad]
        if len(obs) > 2000:
            # Thousands of points are a blot as dots: their density, on a log scale.
            hb = ax1.hexbin(
                obs,
                pred,
                gridsize=45,
                bins="log",
                mincnt=1,
                cmap="Blues",
                extent=(*lims, *lims),
                linewidths=0,
                rasterized=True,
            )
            S.colorbar(fig, hb, ax1, label="Count")
        else:
            ax1.scatter(
                obs, pred, s=5, alpha=0.35, color=S.OBSERVED, edgecolors="none", rasterized=True
            )
        ax1.plot(lims, lims, color=S.INK, lw=0.7, ls=(0, (4, 3)), label="1:1")
        # The fit beside the R² and RMSE in the title: how many points, the slope of
        # predicted on observed (below 1: the extremes are pulled in) and the mean bias.
        slope = float(np.polyfit(obs, pred, 1)[0]) if len(obs) > 1 else float("nan")
        ax1.text(
            0.97,
            0.04,
            f"n = {len(obs):,}\nslope = {slope:.2f}\nbias = {float((pred - obs).mean()):+.2f}",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=S.SMALL_SIZE + 0.5,
            color=S.INK,
            linespacing=1.3,
        )
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_aspect("equal", adjustable="box")
        ss_res = float(((obs - pred) ** 2).sum())
        ss_tot = float(((obs - obs.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(((obs - pred) ** 2).mean()))
        ax1.set_xlabel(f"Observed {name} (test set)")
        ax1.set_ylabel(f"Predicted {name}")
        ax1.set_title(f"Parity — R²={r2:.3f}, RMSE={rmse:.2f}")
        S.grid(ax1)
        S.legend(ax1, loc="upper left")

        imp = self._feature_importance()
        if imp is not None:
            names, vals = imp
            order = np.argsort(vals)[-15:]
            ax2.barh(
                [names[i] for i in order], [vals[i] for i in order], color=S.COMPONENT, height=0.65
            )
            ax2.set_title("Feature importance")
            ax2.set_xlabel("Importance")
            S.grid(ax2, axis="x")
            ax2.tick_params(axis="y", length=0)
            for side in ("top", "right"):
                ax2.spines[side].set_visible(False)
        else:
            ax2.text(
                0.5,
                0.5,
                "Feature importance not available\nfor this backend",
                ha="center",
                va="center",
                color=S.NEUTRAL,
            )
            ax2.axis("off")

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
            own_words=True,
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
