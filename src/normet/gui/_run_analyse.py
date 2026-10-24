"""Steps 2-5: the run handlers and what each result tab shows.

Each pair is one step -- submit to the runner, then render the result and log
the command.  Mixed into :class:`~.main_window.MainWindow`.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import pandas as pd
from PySide6.QtWidgets import QMainWindow, QMessageBox

from normet import style as S

from . import _chronos
from ._constants import TIME_VARS
from ._lists import _checked_items

log = logging.getLogger(__name__)


def _long_term_columns(res: pd.DataFrame | None) -> pd.DataFrame | None:
    """Where the separated long-term component can be read from: the emission
    decomposition's table, which is the only method that separates its
    ``date_unix`` contribution next to its base (the met method splits by
    meteorology, and a chronos-2 run makes no emission part at all)."""
    if res is None:
        return None
    if "date_unix" in res.columns and "emi_base" in res.columns:
        return res
    return None


# -- events & anomalies --------------------------------------------------------------------


def events_call(
    frame: pd.DataFrame,
    column: str,
    *,
    method: str = "iqr",
    k: float = 3.0,
    min_length: int = 1,
    seed: int = 7_654_321,
) -> tuple[pd.Series, pd.DataFrame]:
    """The run as one worker task: per-timestamp scores for the strip, and the
    flagged windows for the table.

    ``column`` is a table column or ``"long-term"``, the separated long-term
    component; the task takes the table as it stands (the command log names it
    by the expression it is kept under), so a saved session repeats the search
    exactly as it ran -- the series is built the way the trend builds it. Both
    library calls score the same series with the same seed, and every method
    here is deterministic under it, so the shaded windows on the strip and the
    rows in the table are the same stretches, hour for hour. Scoring runs
    twice (``detect_events`` scores for itself); on a long record under
    ``stl_residual`` that is a second STL, which is why this pair lives on a
    worker thread rather than in the handler.
    """
    from normet import anomaly_scores, detect_events

    from . import _trend

    series = (
        _trend.long_term_series(frame)
        if column == "long-term"
        else _trend.trend_series(frame, column)
    )
    scores = anomaly_scores(series, method=method, seed=seed)
    events = detect_events(series, method=method, k=k, min_length=min_length, seed=seed)
    return scores, events


def _event_stamp(value: Any, fmt: str) -> str:
    """An event boundary to the stamp the table shows: a timestamp from
    Python, whatever string form the bridge handed back."""
    try:
        return pd.Timestamp(value).strftime(fmt)
    except (ValueError, TypeError):
        return str(value)


def events_table(events: pd.DataFrame, *, with_clock: bool) -> tuple[list[str], list[list[str]]]:
    """The event windows as the Events tab's table. *with_clock* puts the
    hour on the stamps: an hourly record's windows are hours, and day-only
    stamps would show two events on one day as if they were one."""
    fmt = "%Y-%m-%d %H:%M" if with_clock else "%Y-%m-%d"
    header = ["Start", "End", "Steps", "Max score", "Mean score"]
    rows: list[list[str]] = []
    for ev in events.to_dict("records"):
        try:
            max_s, mean_s = f"{float(ev['max_score']):.2f}", f"{float(ev['mean_score']):.2f}"
        except (TypeError, ValueError):
            max_s = mean_s = "—"
        rows.append(
            [
                _event_stamp(ev["start"], fmt),
                _event_stamp(ev["end"], fmt),
                str(int(ev["n"])),
                max_s,
                mean_s,
            ]
        )
    return header, rows


def _events_has_clock(scores: pd.Series) -> bool:
    """True when the record steps are finer than a day: its event windows
    are hours, and day-only stamps would draw two events on one day as
    though they were one."""
    if not isinstance(scores.index, pd.DatetimeIndex) or len(scores.index) < 3:
        return False
    med = pd.Series(scores.index).diff().median()
    return bool(pd.notna(med) and med < pd.Timedelta(days=1))


def events_verdict(
    events: pd.DataFrame,
    scores: pd.Series,
    *,
    method: str,
    k: float,
    min_length: int,
    with_clock: bool,
) -> tuple[tuple[str, str], list[str]]:
    """The Events tab's verdict and its note lines.

    An event is not an error — a record with nothing anomalous at k=3 is a
    fine answer, and one with events says no more than *look here* — so the
    verdict counts what was flagged rather than scolding it. The note lines
    carry the run's settings and the largest window, the two things a reader
    of the table asks first.
    """
    t = 0.0 if method == "isolation" else float(k)
    fmt = "%Y-%m-%d %H:%M" if with_clock else "%Y-%m-%d"
    steps = int(events["n"].sum()) if len(events) else 0
    pct = 100.0 * steps / max(len(scores), 1)
    # One sentence, plain counts: the zh catalogue keys these strings by their
    # shape with the numbers lifted out, so no plural branches and no
    # thousands separators.
    lines = [f"Method {method}, threshold {t:g}, windows of at least {min_length} steps."]
    if len(events) == 0:
        return (
            (
                "ok",
                f"No events: no stretch of the {len(scores)}-step record crosses "
                f"the threshold {t:g} for {min_length} running steps.",
            ),
            lines,
        )
    top = events.iloc[0]
    lines.append(
        f"The largest window: {_event_stamp(top['start'], fmt)} to "
        f"{_event_stamp(top['end'], fmt)}, {int(top['n'])} steps, "
        f"max score {float(top['max_score']):.2f}."
    )
    return (
        (
            "warn",
            f"Event windows flagged: {len(events)} ({pct:.1f}% of the record) — "
            "check the table against the shaded strip.",
        ),
        lines,
    )


def events_figure(
    scores: pd.Series, events: pd.DataFrame, *, threshold: float | None, name: str
) -> Any:
    """The score strip: every timestamp's anomaly score with each flagged
    window shaded, so a table row is visible as the bump that made it. The
    threshold is drawn where one applies — isolation scores against the
    forest's own boundary and says so by leaving the line out — and the
    shading is then simply where the curve crosses it."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(scores.index, scores.to_numpy(dtype=float), color=S.COMPONENT, lw=0.8)
    if threshold is not None:
        ax.axhline(threshold, color=S.INK, lw=0.8, ls=(0, (4, 3)), label=f"threshold {threshold:g}")
    for ev in events.to_dict("records"):
        ax.axvspan(ev["start"], ev["end"], color=S.MODELLED, alpha=0.15, lw=0)
    ax.set_ylabel("Anomaly score")
    ax.set_title(f"Anomaly scores — {name}")
    S.grid(ax, axis="y")
    if threshold is not None:
        S.legend(ax, loc="upper right")
    return fig


class AnalyseRunMixin(QMainWindow):
    def _run_normalise(self) -> None:
        if not self._check_model():
            return
        resample_vars = self._selected_norm_vars()
        if not resample_vars:
            self.notify(
                "Tick at least one variable to resample under Normalise (usually the meteorological ones).",
                key="need-resample-vars",
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
        target = self.target_combo.currentText()
        # Chronos-2 has no history to condition on for the first context_length
        # rows, so deweather() seeds them with the observed values: normalised
        # equals observed there by construction, not by result. Averaging across
        # them drags the reported effect toward zero, and on a record not much
        # longer than the context that is most of the series -- so say how many
        # rows are seeded and take the mean over the rest.
        seeded = min(int(self.model.context_length), len(result)) if self._is_chronos() else 0
        projected = result.iloc[seeded:]

        from normet.plotting import profile_axes, usable_cycles

        # The cycles beneath the series: observed and normalised averaged by the hour of
        # the day, the day of the week and the month -- what the weather moved and what
        # it did not. From the record as it is, whatever resolution the series is shown at.
        cycles = usable_cycles(projected.index) if len(projected) > 2 else []

        def render(d: pd.DataFrame):
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(8, 4.8 if cycles else 3), layout="constrained")
            gs = fig.add_gridspec(
                2 if cycles else 1,
                max(1, len(cycles)),
                height_ratios=[1.5, 1] if cycles else None,
            )
            ax = fig.add_subplot(gs[0, :])
            normalise_plot(
                d,
                ci_low=qcols[0] if len(qcols) >= 2 else None,
                ci_high=qcols[-1] if len(qcols) >= 2 else None,
                ylabel=S.formula(target),
                title=f"Meteorologically normalised {S.formula(target)}",
                ax=ax,
            )
            for j, cycle in enumerate(cycles):
                axc = fig.add_subplot(gs[1, j])
                profile_axes(axc, projected["observed"], cycle, color=S.OBSERVED, label="Observed")
                profile_axes(
                    axc,
                    projected["normalised"],
                    cycle,
                    color=S.MODELLED,
                    label="Normalised",
                    emphasis=True,
                )
                if j == 0:
                    axc.set_ylabel(S.formula(target))
            if seeded:
                ax.axvspan(
                    result.index[0],
                    result.index[seeded - 1],
                    color="#E5E5EA",
                    lw=0,
                    zorder=0,
                    label="seeded (= observed)",
                )
                S.legend(ax, loc="upper right", ncols=4)
            return fig

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
        self.tab_norm.show_timeseries(render, result, lines=lines)
        self._sync_enabled()
        self.tabs.setCurrentWidget(self.tab_norm)
        self.statusBar().showMessage(
            "Normalisation done — the meteorologically normalised series is on the Normalise tab."
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
                groups = self.decom_groups.groups(met)
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
        method = "meteorology" if "met_total" in result.columns else "emission"
        target = self.target_combo.currentText()

        from . import _decomp_plots as DP

        try:
            if method == "emission":
                # Its cycles from the record as it is, whatever resolution is shown.
                render = functools.partial(DP.emission_figure, target=target, cycles_from=result)
                d = DP.indexed_result(result)
                comps = [(c, lbl) for c, lbl in DP.EMI_COMPONENTS if c in d.columns]
                dominant = max(comps, key=lambda cl: d[cl[0]].std())[1] if comps else "n/a"
                lines = [
                    f"emi_total = emi_base + {' + '.join(c for c, _ in comps)} — "
                    f"largest time-scale: {dominant}."
                ]
            else:
                render = functools.partial(DP.meteorology_figure, target=target, cycles_from=result)
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
            # Each method's last result from this model, a Method selector to switch.
            if self._decom_model is not self.model:
                self._decom_views, self._decom_model = {}, self.model
            self._decom_views[method] = (render, result, {"lines": lines})
            self.tab_decom.show_views(self._decom_views, method)
        except Exception:
            log.exception("Decomposition figure failed; falling back to a line plot")
            import matplotlib.pyplot as plt

            plot_df = result.set_index("date") if "date" in result.columns else result
            fig, ax = plt.subplots(figsize=(8, 3))
            for column in plot_df.columns:
                if pd.api.types.is_numeric_dtype(plot_df[column]):
                    S.line(ax, plot_df.index, plot_df[column], lw=0.7, label=column)
            if isinstance(plot_df.index, pd.DatetimeIndex):
                S.format_time_axis(ax)
            ax.margins(x=0)
            S.grid(ax)
            S.legend(ax, loc="upper right", ncols=3)
            self.tab_decom.show_result(fig)

        self.tabs.setCurrentWidget(self.tab_decom)
        self.statusBar().showMessage("Decomposition done.")
        self._log_history("decompose", method)
        log.info("Decomposition finished")
        # The question a decomposition invites next: does its series trend? The box's
        # tick says so; the runner's own finished slot starts it, because this one
        # ends before the runner knows it is free.
        self._trend_wanted = (
            self.trend_after.isChecked() and method == "emission" and "emi_total" in result.columns
        )
        if getattr(self, "_restoring", False):
            self._trend_wanted = False

    def _trend_resolve(self) -> tuple[pd.DataFrame, str, str, int | None, str] | None:
        """The box's choice as a testable series: its table, the column, a name and
        the seasonal period (the auto's, spoken when it fell back to the
        classical). The column is ``"long-term"`` where the box points at the
        separated long-term component (a sum of the decomposition's columns,
        not one of them). Nothing the window can hold gives None; its why is
        said on screen."""
        target = self.target_combo.currentText()
        choice = self.trend_series.currentText()
        frame, column = self.df_prep, target
        if choice.startswith("long-term"):
            res = self.results.get("decompose")
            long = _long_term_columns(res)
            if res is None or long is None:
                self.notify(
                    "The long-term trend needs the emission decomposition that "
                    "separated it (its Trend (long-term) panel) -- run the "
                    "emission decomposition first, or choose another series.",
                    key="need-decompose",
                )
                return None
            frame, column = res, "long-term"
        elif not choice.startswith("observed"):
            res = self.results.get("decompose")
            if res is None:
                self.notify(
                    "Run the decomposition first: the trend is tested on its series.",
                    key="need-decompose",
                )
                return None
            column = "emi_total" if choice.startswith("emission") else "met_total"
            if column not in res.columns:
                self.notify(
                    f"The decomposition has no {column!r} in it: the emission "
                    "part needs the emission method (on chronos-2, a flaml or "
                    "lightgbm model), the weather part the meteorology one.",
                    key="need-decompose",
                )
                return None
            frame = res
        if not isinstance(frame, pd.DataFrame) or (
            column != "long-term" and column not in frame.columns
        ):
            self.notify(
                f"The table the window holds has no {column!r}: train again, or "
                "test the decomposition's series.",
                key="need-model",
            )
            return None
        if column == "long-term":
            if frame["date_unix"].notna().sum() < 15:
                self.notify(
                    "Fewer than 15 values in the series: the trend test needs more.",
                    key="need-trend",
                )
                return None
        elif frame.dropna(subset=[column]).shape[0] < 15:
            self.notify(
                "Fewer than 15 values in the series: the trend test needs more.",
                key="need-trend",
            )
            return None
        name = (
            f"{target} (observed)"
            if choice.startswith("observed")
            else f"{target} — long-term trend (separated)"
            if column == "long-term"
            else f"{target} — emission part"
            if column == "emi_total"
            else f"{target} — weather part"
        )
        segment = self.tre_segments.value()
        if segment > 1:
            name += f", {segment} segments"
        period, note = None, ""
        choice_seasonal = self.tre_seasonal.currentText()
        if choice_seasonal == "auto":
            from . import _trend

            index = _trend._datetime_index(frame)
            if index is not None:
                n = int(
                    frame["date_unix"].notna().sum()
                    if column == "long-term"
                    else frame[column].notna().sum()
                )
                period, note = _trend.auto_seasonal(index, n)
        elif choice_seasonal != "none":
            period = int(choice_seasonal)
        return frame, column, name, period, note

    def _run_trend(self) -> None:
        if not self._check_model():
            return
        got = self._trend_resolve()
        if got is None:
            return
        frame, column, name, period, note = got
        from . import _trend

        self._trend_note = note
        self._trend_name = name
        self._trend_table = (frame, column)
        self._trend_segmented = self.tre_segments.value() > 1
        if self._trend_segmented:
            ok = self.runner.submit(
                "trend (segmented)",
                _trend.trend_breaks_call,
                self._trend_breaks_done,
                self._show_error,
                frame,
                column,
                series_name=name,
                max_segments=self.tre_segments.value(),
                n_bootstrap=self.tre_n_bootstrap.value(),
                seed=self.seed_spin.value(),
            )
        else:
            ok = self.runner.submit(
                "trend",
                _trend.trend_call,
                self._trend_done,
                self._show_error,
                frame,
                column,
                series_name=name,
                seasonal_period=period,
                n_bootstrap=self.tre_n_bootstrap.value(),
                seed=self.seed_spin.value(),
            )
        if not ok:
            self._trend_note = ""

    def _trend_done(self, result: pd.DataFrame) -> None:
        from . import _trend

        self._trend_wanted = False
        self.results["trend"] = result.summary()  # the Trend tab's light, export and session
        summary = result.summary()
        frame, column = self._trend_table or (None, "")
        try:
            series = _trend.trend_series(frame, column) if frame is not None else None
        except ValueError:
            series = None
        self._trend_series = series
        self._trend_raw = result
        verdict = _trend.trend_verdict(result, self._trend_name or column)
        lines = ([self._trend_note] if self._trend_note else []) + list(result.notes)
        rows = [list(r) for r in summary.itertuples(index=False, name=None)]
        self.tab_trend.show_result(
            _trend.trend_fig(series, result) if series is not None else None,
            verdict=verdict,
            lines=lines,
            header=["metric", "value"],
            rows=rows,
        )
        self.tabs.setCurrentWidget(self.tab_trend)
        self.statusBar().showMessage("Trend test done.")
        self._log_history("trend", self._trend_name)
        log.info("Trend test finished (%s)", self._trend_name)

    def _trend_breaks_done(self, result: Any) -> None:
        from . import _trend

        self._trend_wanted = False
        self.results["trend"] = result.summary()  # the Trend tab's light, export and session
        summary = result.summary()
        frame, column = self._trend_table or (None, "")
        if frame is None:
            series = None
        else:
            try:
                series = (
                    _trend.long_term_series(frame)
                    if column == "long-term"
                    else _trend.trend_series(frame, column)
                )
            except ValueError:
                series = None
        self._trend_series = series
        self._trend_raw = result
        verdict = _trend.trend_breaks_verdict(result, self._trend_name or column)
        lines = ([self._trend_note] if self._trend_note else []) + list(result.notes)
        rows = [list(r) for r in summary.itertuples(index=False, name=None)]
        self.tab_trend.show_result(
            _trend.trend_breaks_fig(series, result) if series is not None else None,
            verdict=verdict,
            lines=lines,
            header=list(summary.columns),
            rows=rows,
        )
        self.tabs.setCurrentWidget(self.tab_trend)
        self.statusBar().showMessage("Segmented trend done.")
        self._log_history("trend", self._trend_name)
        log.info("Segmented trend finished (%s)", self._trend_name)

    def _events_resolve(self) -> tuple[pd.DataFrame, str] | None:
        """The events box's series choice as the table and column to score:
        the trend box's four choices read against what the window holds.
        None, said on screen, when the choice points at nothing the window
        has; the score itself needs what the trend test needs. The task is
        given the table and column, not the built series, so the command log
        can name the table and a saved session repeats the search."""
        from . import _trend

        choice = self.events_series.currentText()
        if choice.startswith("observed"):
            frame, column = self.df_prep, self.target_combo.currentText()
        elif choice.startswith("long-term"):
            res = self.results.get("decompose")
            if res is None or _long_term_columns(res) is None:
                self.notify(
                    "The long-term trend needs the emission decomposition that "
                    "separated it -- run the emission decomposition first, or "
                    "choose another series.",
                    key="need-decompose",
                )
                return None
            frame, column = res, "long-term"
        else:
            res = self.results.get("decompose")
            if res is None:
                self.notify(
                    "Run the decomposition first: events are flagged on its series.",
                    key="need-decompose",
                )
                return None
            column = "emi_total" if choice.startswith("emission") else "met_total"
            if column not in res.columns:
                self.notify(
                    f"The decomposition has no {column!r} in it: the emission "
                    "part needs the emission method, the weather part the "
                    "meteorology one.",
                    key="need-decompose",
                )
                return None
            frame = res
        try:
            series = (
                _trend.long_term_series(frame)
                if column == "long-term"
                else _trend.trend_series(frame, column)
            )
        except ValueError:
            self.notify(
                f"The table the window holds has no {column!r}: train again, or "
                "score the observed target.",
                key="need-model",
            )
            return None
        if int(series.notna().sum()) < 15:
            self.notify(
                "Fewer than 15 values in the series: the event search needs more.",
                key="need-events",
            )
            return None
        return frame, column

    def _run_events(self) -> None:
        """Flag the record's event windows on a worker: score every
        timestamp, then the consecutive stretches above the threshold
        (``normet.detect_events``). IsolationForest and STL each cost a
        fit over the record, which is not the GUI thread's to spend."""
        if not self._check_model():
            return
        got = self._events_resolve()
        if got is None:
            return
        frame, column = got
        method = self.events_method.currentText()
        name = S.formula(self.target_combo.currentText())
        choice = self.events_series.currentText()
        if choice.startswith("long-term"):
            name = "long-term trend"
        elif choice.startswith("emission"):
            name = "emi_total"
        elif choice.startswith("meteorology"):
            name = "met_total"
        # Spoken at submit, like the trend's note: the panel's combos stay
        # live while the task runs, and the result must say what was asked.
        self._events_spec = (method, self.events_k.value(), self.events_min_len.value(), name)
        self.runner.submit(
            f"events ({method})",
            events_call,
            self._events_done,
            self._show_error,
            frame,
            column,
            method=method,
            k=self.events_k.value(),
            min_length=self.events_min_len.value(),
            seed=self.seed_spin.value(),
        )

    def _events_done(self, got: tuple[pd.Series, pd.DataFrame]) -> None:
        """The Events tab: the score strip with the flagged windows shaded,
        the event table above it, and a verdict that counts what was
        flagged."""
        scores, events = got
        method, k, min_length, name = self._events_spec
        self.results["events"] = events
        with_clock = _events_has_clock(scores)
        header, rows = events_table(events, with_clock=with_clock)
        verdict, lines = events_verdict(
            events, scores, method=method, k=k, min_length=min_length, with_clock=with_clock
        )
        self.tab_events.show_result(
            events_figure(
                scores, events, threshold=None if method == "isolation" else k, name=name
            ),
            verdict=verdict,
            lines=lines,
            header=header,
            rows=rows,
        )
        self.tabs.setCurrentWidget(self.tab_events)
        self.statusBar().showMessage("Event search done.")
        self._log_history("events", method)
        log.info("Event search finished (%s, %d windows)", method, len(events))

    def _run_unc(self) -> None:
        if not self._check_model():
            return
        if self._is_chronos():
            self.notify(
                "Not available for chronos-2: its zero-shot forecasts carry no "
                "model form to vary, and the attribution needs several members "
                "each fitted to the record. Train with flaml or lightgbm.",
                key="need-model",
            )
            return
        from normet import decompose_uncertainty

        variables_resample = self._selected_norm_vars() or None
        cfg = (
            {"time_budget": self.budget_spin.value()}
            if self.unc_backend.currentText() == "flaml"
            else {"n_trials": self.budget_spin.value()}
        )
        self.runner.submit(
            "uncertainty decomposition",
            decompose_uncertainty,
            self._unc_done,
            self._show_error,
            self.df_raw,
            self.target_combo.currentText(),
            backend=self.unc_backend.currentText(),
            covariates=self.trained_features + list(TIME_VARS),
            variables_resample=variables_resample,
            model_config=cfg,
            n_samples=self.unc_samples.value(),
            n_models=self.unc_models.value(),
            seed=self.seed_spin.value(),
            n_cores=self._n_cores(self.norm_cores),
            verbose=True,
        )

    def _unc_done(self, result: Any) -> None:
        from . import _uncertainty

        self._unc_raw = result
        self.results["uncertainty"] = result.frame()  # the tab's light, export and session
        summary = result.summary()
        # The run's own numbers, read off the result: a restored session re-shows
        # the saved run under its own settings, whatever the panel stands at now.
        name = f"{result.target} ({result.n_models} members x {result.n_weather_draws} met draws)"
        self.tab_unc.show_result(
            result.plot(),
            verdict=_uncertainty.verdict(result, name),
            lines=list(result.notes),
            header=list(summary.columns),
            rows=[list(r) for r in summary.itertuples(index=False, name=None)],
        )
        self.tabs.setCurrentWidget(self.tab_unc)
        self.statusBar().showMessage("Uncertainty decomposition done.")
        self._log_history("uncertainty", f"n_models={self.unc_models.value()}")
        log.info("Uncertainty decomposition finished (%s)", result)

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
        from normet import window_columns

        rolling_cols = window_columns(result)
        window, step = self.roll_window.value(), self.roll_step.value()

        def render(d: pd.DataFrame):
            return self._rolling_figure(d, rolling_cols, window, step)

        self.tab_roll.show_timeseries(render, result)
        self.tabs.setCurrentWidget(self.tab_roll)
        self.statusBar().showMessage("Rolling normalisation done.")
        self._log_history("rolling", f"{len(rolling_cols)} windows")
        log.info("Rolling finished (%d windows)", len(rolling_cols))

    @staticmethod
    def _rolling_figure(result: pd.DataFrame, rolling_cols: list[str], window: int, step: int):
        """Mean of the rolling windows' series, ±1 SD where windows overlap."""
        import matplotlib.pyplot as plt

        mean = result[rolling_cols].mean(axis=1, skipna=True)
        overlap = result[rolling_cols].notna().sum(axis=1)
        std = result[rolling_cols].std(axis=1, skipna=True).where(overlap >= 2)

        fig, ax = plt.subplots(figsize=(8, 3))
        dense = len(result) > 1500
        if "observed" in result.columns:
            S.line(
                ax,
                result.index,
                result["observed"],
                lw=0.5 if dense else 0.9,
                alpha=0.4,
                color=S.NEUTRAL,
                label="observed",
            )
        if std.notna().any():
            S.band(
                ax,
                mean.index,
                mean - std,
                mean + std,
                color=S.MODELLED,
                label="±1 SD across overlapping windows",
            )
        S.line(
            ax,
            mean.index,
            mean.values,
            lw=0.8 if dense else 1.3,
            color=S.MODELLED,
            label="rolling mean",
        )
        ax.set_title(
            f"Rolling normalisation — mean of {len(rolling_cols)} windows "
            f"({window} d window, {step} d step)"
        )
        if isinstance(result.index, pd.DatetimeIndex):
            S.format_time_axis(ax)
        ax.margins(x=0)
        S.grid(ax)
        S.legend(ax, loc="upper right", ncols=3)
        return fig

    def _run_multiscale(self) -> None:
        if not self._check_model():
            return
        if "normalise" not in self.results:
            self.notify(
                "Multi-scale decomposition needs Y_∞ from the normalisation as its "
                "full-record baseline — run Normalise, then come back here.",
                key="need-normalise",
                actions=(("Run normalisation", self._run_normalise),),
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

        self._multiscale_raw = result
        target = self.target_combo.currentText()
        series = {k: v for k, v in result.items() if isinstance(v, pd.Series)}

        def render(d: pd.DataFrame):
            # The series at the resolution chosen; the bands' σ from the record as it is.
            drawn = {**result, **{k: d[k] for k in series if k in d.columns}}
            return multiscale_figure(drawn, target=target, sigma_from=result)

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

        if series:
            self.tab_multiscale.show_timeseries(
                render, pd.DataFrame(series), verdict=verdict, lines=lines
            )
        else:
            self.tab_multiscale.show_result(
                multiscale_figure(result, target=target), verdict=verdict, lines=lines
            )
        self.tabs.setCurrentWidget(self.tab_multiscale)

        if series:
            self.results["multiscale"] = pd.DataFrame(series)

        self.statusBar().showMessage("Multi-scale decomposition done.")
        self._log_history("multiscale", f"{len(bands)}/3 bands computed")
        log.info("Multi-scale decomposition finished (%d/3 bands)", len(bands))

    def _run_pdp(self) -> None:
        if not self._check_model():
            return
        variables = _checked_items(self.pdp_vars)
        if not variables:
            self.notify("Tick at least one variable for the PDP.", key="need-pdp-vars")
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

        # The tab names the figure; the shared y label says what is predicted.
        target = S.formula(self.target_combo.currentText())
        fig = _pdp_grid(
            result,
            ylabel=f"Partial dependence of {target}" if target else "Partial dependence",
            data=self.df_prep,  # each variable's deciles along its axis
        )
        self.tab_pdp.show_result(fig)
        self.tabs.setCurrentWidget(self.tab_pdp)
        self.statusBar().showMessage("PDP done.")
        self._log_history("pdp", f"{result['variable'].nunique()} variables")
        log.info("PDP finished")
