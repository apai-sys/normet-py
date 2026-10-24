"""Training, Chronos-2 loading and the Model tab.

Step 1's run handler, what it does when the task comes back, and the two
renderers for the Model tab (one for a trained estimator, one for Chronos-2's
zero-shot check).  Mixed into :class:`~.main_window.MainWindow`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

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

#: The levels the Chronos-2 Model-tab backtest forecasts and scores. The middle
#: one doubles as the point forecast whose CRPS reference the band must beat.
_BACKTEST_Q = (0.1, 0.5, 0.9)


def crps_reference(obs: Any, pred: Any) -> float | None:
    """The point forecast's CRPS reference (``sqrt(pi/2)`` x its MAE), or None
    where it cannot be read. A spread forecast earns its spread only by scoring
    below this: the number puts a point model on the probabilistic scale the
    CRPS measures on."""
    try:
        from normet.analysis import crps_observed

        return float(crps_observed(pred, obs))
    except Exception:
        log.exception("crps_observed failed")
        return None


def quantile_backtest(estimator: Any, df_prep: pd.DataFrame) -> tuple[list[list[str]], str] | None:
    """Score the estimator's quantile forecast of the last horizon against the
    observed tail, as the Chronos-2 Model tab shows it: the band's CRPS, the
    median's point reference, and the pinball loss at each level.

    The zero-shot model has no held-out parity plot, so this stands in for one:
    one anchored forecast over the last ``prediction_length`` steps, scored as
    a distribution rather than a point. A single window is thin evidence, and
    the line says so by naming its length; it is the Model tab's quick check,
    not the backtest a paper would quote. None where the record is too short or
    the forecast or scoring fails (logged).
    """
    try:
        from normet.analysis import crps_from_quantiles, crps_observed, pinball_loss

        indexed = _chronos.to_indexed_frame(df_prep)
        horizon = int(estimator.prediction_length)
        if len(indexed) <= int(estimator.context_length) + horizon:
            return None
        qf = estimator.predict_quantiles(
            indexed,
            "value",
            anchor=indexed.index[-horizon],
            horizon=horizon,
            quantiles=list(_BACKTEST_Q),
        )
        obs = indexed.loc[qf.index, "value"]
        crps = crps_from_quantiles(qf.to_numpy(dtype=float), obs, quantiles=list(_BACKTEST_Q))
        med_col = f"q{_BACKTEST_Q[1]}"
        ref = crps_observed(qf[med_col], obs)
        rows = [
            ["Quantile backtest — CRPS", f"{crps:.3f}"],
            ["Quantile backtest — point reference", f"{ref:.3f}"],
        ] + [
            [f"Pinball loss, q{q}", f"{pinball_loss(qf[f'q{q}'], obs, q):.3f}"] for q in _BACKTEST_Q
        ]
        n = len(qf)
        if crps <= ref:
            line = (
                f"Over the last {n} steps the band's CRPS {crps:.3f} beats the median's "
                f"point reference {ref:.3f} — the spread earns its keep."
            )
        else:
            line = (
                f"Over the last {n} steps the band's CRPS {crps:.3f} does not beat the "
                f"median's point reference {ref:.3f} — the spread costs more than it tells."
            )
        return rows, line
    except Exception:
        log.exception("quantile backtest failed")
        return None


#: The fold table's columns, under the names the CV view shows them by. The
#: scores keep modStats' abbreviations, as the holdout's own table does.
_CV_COLUMNS = [
    ("fold", "Fold"),
    ("test_start", "Test starts"),
    ("test_end", "Test ends"),
    ("n_test", "Rows"),
    ("R2", "R2"),
    ("RMSE", "RMSE"),
    ("MB", "MB"),
]


def _fold_day(value: Any) -> str:
    """A fold boundary to the day: a timestamp from Python, a string from R."""
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return str(value)[:10]
    return str(value)[:10] if pd.isna(stamp) else stamp.strftime("%Y-%m-%d")


def cv_table(folds: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    """The walk-forward folds as the Model tab's table: the window each fold
    forecast and the three scores its holdout table is read by. A backend
    that scores differently leaves an em dash in the column it did not fill."""
    header = [label for _key, label in _CV_COLUMNS]
    rows: list[list[str]] = []
    for rec in folds.to_dict("records"):
        row: list[str] = []
        for key, _label in _CV_COLUMNS:
            v = rec.get(key)
            if key in ("test_start", "test_end"):
                row.append(_fold_day(v))
            elif key in ("fold", "n_test"):
                try:
                    row.append(str(int(float(v))))
                except (TypeError, ValueError):
                    row.append("—")
            else:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    row.append("—")
                else:
                    row.append("—" if pd.isna(f) else f"{f:.3f}")
        rows.append(row)
    return header, rows


def cv_verdict(folds: pd.DataFrame, holdout_r2: float | None) -> tuple[tuple[str, str], list[str]]:
    """The CV view's verdict and its note line: the folds' median R², with the
    random split's read against it.

    Walk-forward CV is the honest check: a random split lets each test hour's
    neighbours into training, so the split's R² is an optimist's number. The
    interesting quantity is therefore the gap — over 0.1 R² the split flatters
    the model and the folds are what to trust. The fold thresholds match the
    holdout verdict's (0.7 good, 0.5 moderate).
    """
    import numpy as np

    r2 = (
        pd.to_numeric(folds["R2"], errors="coerce").dropna()
        if "R2" in folds
        else pd.Series(dtype=float)
    )
    if r2.empty:
        return (
            ("warn", "Walk-forward CV produced no scorable folds — try fewer folds."),
            [],
        )
    med = float(r2.median())
    lines = [f"Across {len(r2)} folds R² ranged {float(r2.min()):.2f} to {float(r2.max()):.2f}."]
    split = None if holdout_r2 is None else float(holdout_r2)
    if split is not None and np.isfinite(split) and split - med > 0.1:
        return (
            (
                "warn",
                f"Walk-forward CV — median fold R² = {med:.2f} against the random "
                f"split's {split:.2f}: the split flatters the model, trust the folds.",
            ),
            lines,
        )
    if med >= 0.7:
        return (
            (
                "ok",
                f"Walk-forward CV — median fold R² = {med:.2f}: the fit holds on future-ordered folds.",
            ),
            lines,
        )
    if med >= 0.5:
        return (
            (
                "warn",
                f"Walk-forward CV — median fold R² = {med:.2f}: consider more features or a longer time budget.",
            ),
            lines,
        )
    return (
        (
            "error",
            f"Walk-forward CV — median fold R² = {med:.2f}: on future-ordered folds the fit "
            "is poor, revisit features/target.",
        ),
        lines,
    )


def cv_figure(folds: pd.DataFrame, holdout_r2: float | None) -> Any:
    """R² per fold, with the random split's value drawn across the bars: the
    eye checks whether the split's number sits above the folds it should match."""
    import matplotlib.pyplot as plt
    import numpy as np

    r2 = pd.to_numeric(folds["R2"], errors="coerce") if "R2" in folds else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    x = np.arange(len(folds))
    starts = folds["test_start"] if "test_start" in folds else pd.Series(list(range(len(folds))))
    ax.bar(x, np.nan_to_num(r2.to_numpy(dtype=float), nan=0.0), width=0.6, color=S.COMPONENT)
    if len(r2.dropna()):
        ax.axhline(
            float(r2.dropna().median()), color=S.INK, lw=0.8, ls=(0, (4, 3)), label="CV median"
        )
    if holdout_r2 is not None and np.isfinite(float(holdout_r2)):
        ax.axhline(float(holdout_r2), color=S.TERTIARY, lw=1.2, label="Random split")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_fold_day(v) for v in starts], fontsize=S.SMALL_SIZE, rotation=0 if len(folds) <= 6 else 30
    )
    ax.set_ylabel("R²")
    ax.set_title(f"Walk-forward CV — {len(folds)} folds")
    S.grid(ax, axis="y")
    S.legend(ax, loc="upper right")
    return fig


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
        model's output driven by meteorology, or is it autoregressing? — and
        the quantile backtest scores its forecast of the last horizon as a
        distribution (CRPS against the median's point reference, pinball per
        level), the closest thing to a holdout a zero-shot model has here.
        """
        import matplotlib.pyplot as plt

        est = self.model
        # Chronos-2 has no holdout and its own check (the quantile backtest):
        # a CV view left by a previous backend must not ride along here.
        self._model_views = {}
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

        lines = [
            f"Checkpoint: {est.model_name}   |   device: {est.device}   |   "
            f"context: {est.context_length} h   |   horizon: {est.prediction_length} h",
            f"Covariates: {len(met)} met + 6 calendar   |   rows: {len(self.df_prep):,}",
        ]
        # The zero-shot model has no held-out parity plot: its stand-in is a
        # quantile forecast of the last horizon, scored as a distribution.
        bt = quantile_backtest(est, self.df_prep)
        if bt is not None:
            if header is None:
                header, rows = ["Diagnostic", "Value"], []
            rows = rows + bt[0]
            lines.append(bt[1])
        self.tab_model.show_result(
            fig,
            verdict=verdict,
            lines=lines,
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
        """Stats table + parity plot + feature importances, with an R² verdict.

        The figures are drawn by a callable rather than once: a walk-forward CV
        run registers the folds beside them as a tab view (see ``_cv_done``),
        and the tab redraws whichever view is picked.
        """
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

        name = S.formula(self.target_combo.currentText())
        test = self.df_prep
        if "set" in self.df_prep.columns:
            test = self.df_prep[self.df_prep["set"] == "testing"]
        if len(test) == 0:
            test = self.df_prep
        obs = test["value"].to_numpy()
        pred = np.asarray(ml_predict(self.model, test))
        # The verdict's numbers, computed here: the figure waits until the tab
        # draws this view, the verdict is read whether or not it is drawn.
        ss_res = float(((obs - pred) ** 2).sum())
        ss_tot = float(((obs - obs.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(((obs - pred) ** 2).mean()))

        def _holdout_figure(_df: pd.DataFrame | None = None) -> Any:
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1, 1.15]}
            )
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
                    [names[i] for i in order],
                    [vals[i] for i in order],
                    color=S.COMPONENT,
                    height=0.65,
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
            return fig

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
        lines = [
            f"Backend: {backend}   |   features: {len(self.trained_features)}   |   rows: {len(self.df_prep):,}"
        ]
        # The holdout's point forecast on the probabilistic scale: this is the
        # CRPS any spread forecast here would later have to beat.
        ref = crps_reference(obs, pred)
        if ref is not None:
            lines.append(
                f"The point forecast's CRPS reference: {ref:.2f} — a spread forecast "
                "must beat this to be worth its spread."
            )
        # The split's own R²: the walk-forward CV verdict reads the folds against it.
        self._holdout_r2 = float(r2)
        views: dict[str, tuple[Callable[[pd.DataFrame], Any], pd.DataFrame, dict[str, Any]]] = {
            "Holdout": (
                _holdout_figure,
                pd.DataFrame(),
                {
                    "verdict": verdict,
                    "lines": lines,
                    "header": header,
                    "rows": rows,
                    "own_words": True,
                },
            )
        }
        self._model_views = views
        self._show_model_view("Holdout")
        self._log_history("train", f"R2={r2:.3f}")

    def _show_model_view(self, key: str) -> None:
        """The Model tab's current view: the holdout's figures alone, or — once
        a walk-forward CV has run — a selector between them and the folds.

        The CV machinery passes frames through the tab's resolution selector;
        neither view is a time series (a parity plot, a fold table), so the
        selector stays out of the way (:meth:`CanvasTab._draw_series` shows it
        only for data that has a time axis).
        """
        views: dict = getattr(self, "_model_views", {})
        if len(views) > 1:
            self.tab_model.show_views(views, key, label="View")
        else:
            render, data, kwargs = views[key]
            self.tab_model.show_result(render(data), **kwargs)

    def _run_cv(self) -> None:
        """Score the current model form on walk-forward folds: the same backend
        retrained on each expanding window, the future block after it scored as
        a test would be. The random split's R² flatters — each test hour's
        neighbours sit in training — and this is the check of that number
        (``normet.cv_score``); it retrains as many models as there are folds,
        so it takes its budget times over."""
        if self.model is None or self._is_chronos() or not self._check_data():
            return
        from normet import cv_score

        backend = self.backend_combo.currentText()
        if backend == "flaml":
            model_config = {
                "time_budget": self.budget_spin.value(),
                "estimator_list": self._selected_estimators() or ["lgbm"],
            }
        else:
            model_config = {"n_trials": self.budget_spin.value()}
        self.runner.submit(
            f"walk-forward CV ({backend})",
            cv_score,
            self._cv_done,
            self._show_error,
            self.df_prep,
            "value",
            backend=backend,
            covariates=list(self.trained_features),
            n_splits=self.cv_folds_spin.value(),
            model_config=model_config,
            seed=self.seed_spin.value(),
            verbose=True,
        )
        if backend == "flaml":
            # Every fold searches for the time budget: the clock expects the run's total.
            self.task_clock.expect(self.budget_spin.value() * self.cv_folds_spin.value())

    def _cv_done(self, folds: pd.DataFrame | None) -> None:
        """Register the folds as the Model tab's second view, and show it: the
        tab now switches between what the split said and what the folds say."""
        if folds is None or folds.empty:
            self.notify(
                "Walk-forward CV produced no folds — the record is too short for the "
                "number of folds asked for.",
                key="cv-no-folds",
                kind="warn",
            )
            return
        self.cv_folds = folds
        holdout = getattr(self, "_holdout_r2", None)
        header, rows = cv_table(folds)
        verdict, lines = cv_verdict(folds, holdout)

        def _cv_fig(_df: pd.DataFrame | None = None) -> Any:
            return cv_figure(folds, holdout)

        views: dict = getattr(self, "_model_views", {})
        views["Walk-forward CV"] = (
            _cv_fig,
            folds,
            {"verdict": verdict, "lines": lines, "header": header, "rows": rows, "own_words": True},
        )
        self._model_views = views
        self._show_model_view("Walk-forward CV")
        self._log_history("cv", f"{len(folds)} folds")
        self.statusBar().showMessage(
            "Walk-forward CV done — the Model tab compares split and folds."
        )

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
