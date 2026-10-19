"""Steps 2-5: the run handlers and what each result tab shows.

Each pair is one step -- submit to the runner, then render the result and log
the command.  Mixed into :class:`~.main_window.MainWindow`.
"""

from __future__ import annotations

import logging

import pandas as pd
from PySide6.QtWidgets import QMainWindow, QMessageBox

from . import _chronos
from ._constants import TIME_VARS
from ._lists import _checked_items, parse_groups

log = logging.getLogger(__name__)


class AnalyseRunMixin(QMainWindow):
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
