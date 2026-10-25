"""
Generate the screenshots embedded in the normet manual.

Runs the real window (PySide6) off-screen on normet's bundled MY1 record -- hourly
NO2 and ERA5 meteorology at London Marylebone Road, January to August 2020 -- and
walks the workflow as the panel's buttons do: train, normalise, decompose both ways,
rolling, multi-scale, partial dependence. The SCM Studio is shown on the bundled ULEZ
panel, set up as the README sets it up, and the Transport Studio on the two days of
HYSPLIT tdump files that ship with normet. Each tab is grabbed to a PNG, and the
numbers the manual quotes are printed as they are computed. Run from the build venv:

    QT_QPA_PLATFORM=offscreen python docs/manual/make_figures.py

With NORMET_UI_LANGUAGE=zh the window is drawn in Chinese, into figures_zh/ -- the
Chinese manual's (normet_manual_zh.tex); without it, in English into figures/.

The tasks run on the calling thread instead of the window's worker thread, so a
grab happens after the result is on screen; everything else is the window's own code.
"""

import os
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# The window keeps its recent tables and run history under ~/.normet, and the
# screenshots have no business showing, or adding to, those of whoever builds the
# manual. So it gets a home of its own, with matplotlib pointed back at the real font
# cache first: a fresh home would build that from scratch.
_REAL_HOME = os.path.expanduser("~")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_REAL_HOME, ".matplotlib"))
# The Data Studio figure shows its OpenStreetMap basemap: keep those tiles with
# the real cache, so a rebuild does not fetch them again into a temp home.
os.environ.setdefault("NORMET_TILE_CACHE", os.path.join(_REAL_HOME, ".normet", "osm_tiles"))
_HOME = tempfile.mkdtemp(prefix="normet-figures-home-")
os.environ["HOME"] = _HOME
os.environ["USERPROFILE"] = _HOME

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from normet import window_columns  # noqa: E402

# The window's language, set here and not left to the system's: an English build on a
# machine whose system is Chinese draws the English manual's figures in English still.
LANG = os.environ.setdefault("NORMET_UI_LANGUAGE", "en")
OUT = os.path.join(HERE, "figures" if LANG == "en" else f"figures_{LANG}")
os.makedirs(OUT, exist_ok=True)
SIZE = (1320, 860)

#: MY1's meteorology: every column but the target. atmos_pres is not among the
#: names the window recognises as meteorological, so it is ticked by hand.
MY1_MET = ["ws", "wd", "temp", "RH", "atmos_pres", "blh", "tcc", "tp", "ssrd"]


def run_inline(runner):
    """Run the window's tasks on this thread, through the same record/commit and
    result handlers its worker thread would call."""

    def submit(name, fn, on_result, on_error, *args, **kwargs):
        commit = runner.record(fn, args, kwargs) if runner.record is not None else None
        runner.task_name = name
        try:
            result = fn(*args, **kwargs)
        except Exception:
            on_error(traceback.format_exc(limit=8))
            raise
        if commit is not None:
            commit()
        on_result(result)
        runner.finished.emit()
        return True

    runner.submit = submit


def _settle(win):
    """Drain Qt's queue and redraw every canvas at its settled geometry.

    matplotlib sizes its Agg buffer from the widget's size at draw time, so a canvas
    that last drew before the layout finished leaves its edge unpainted. Pump first
    (layout requests post further events), redraw at the final size, pump again.
    """
    for _ in range(6):
        QApplication.sendPostedEvents()
        QApplication.processEvents()
    for canvas in win.findChildren(FigureCanvasQTAgg):
        if canvas.isVisible():
            canvas.draw()
    for _ in range(6):
        QApplication.sendPostedEvents()
        QApplication.processEvents()
    win.repaint()


def grab(win, name):
    _settle(win)
    win.grab().save(os.path.join(OUT, name))
    print("  wrote", name)


def said(tab):
    """Print what a result tab says above its figure: the verdict and the notes."""
    from PySide6.QtWidgets import QLabel

    for label in tab.findChildren(QLabel):
        if label.isVisible() and label.text().strip():
            print("  |", label.text().strip())


def tick(widget, names):
    """Tick exactly *names* in a checklist."""
    for i in range(widget.count()):
        item = widget.item(i)
        item.setCheckState(
            Qt.CheckState.Checked if item.text() in names else Qt.CheckState.Unchecked
        )


def main_window():
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    win.resize(*SIZE)
    win.show()  # lay the window out for real; a never-shown window grabs unpolished
    run_inline(win.runner)
    return win


def main_figures(app):

    win = main_window()
    print("welcome")
    # As the app opens: the panel folded, the Assistant beside the results. The steps
    # below are shown with the panel out and the Assistant closed, so their settings
    # are in view.
    win.start_layout()
    grab(win, "ui_welcome.png")
    win.toggle_panel(True)
    win.assistant.hide()

    # MY1 as File > Load Example opens it: the paper's target and predictors ticked, and
    # the command log starting from the loader a reader can repeat.
    win.load_example("my1")
    tick(win.feature_list, MY1_MET)
    print("data:", win.data_summary.text())
    grab(win, "ui_data.png")

    print("train")
    win.show_panel_page("Model")  # the panel's page for the step
    win.backend_combo.setCurrentText("flaml")
    win.budget_spin.setValue(30)
    tick(win.estimator_list, ["lgbm"])
    win.split_combo.setCurrentText("random")
    win._run_train()
    print(win.results["train"].to_string())
    said(win.tab_model)
    grab(win, "ui_model.png")

    print("normalise")
    win.show_panel_page("Normalise")  # the panel's page for the step
    tick(win.norm_vars, MY1_MET)  # Met only leaves atmos_pres out; it is meteorology
    win.norm_quantiles.setChecked(True)
    win.norm_samples.setValue(300)
    win._run_normalise()
    res = win.results["normalise"]
    print("  columns:", list(res.columns))
    monthly = res[["observed", "normalised"]].resample("MS").mean().round(1)
    print(monthly.to_string())
    said(win.tab_norm)
    grab(win, "ui_normalise.png")

    print("decompose: emission")
    win.show_panel_page("Decompose")  # the panel's page for the step
    win.decom_method.setCurrentText("emission")
    win.decom_samples.setValue(100)
    win._run_decompose()
    print("  columns:", list(win.results["decompose"].columns))
    emi = win.results["decompose"]
    print(
        "  sd:",
        emi[["date_unix", "day_julian", "weekday", "hour", "emi_noise"]].std().round(2).to_dict(),
    )
    # The box's "Run after each emission decomposition" tick is unticked by
    # default, so the decomposition's end did not start the trend and the
    # window's tabs are still where the decomposition put them: this grab
    # shows what it is named.
    win.tabs.setCurrentWidget(win.tab_decom)
    said(win.tab_decom)
    grab(win, "ui_decompose_emission.png")

    print("trend on the emission part")
    # Run it here, under the box's default settings (the emission part,
    # seasonal auto, one segment, 1000 replicates) -- the numbers the manual
    # quotes are this run's own. (It must happen before the meteorology
    # decomposition overwrites the table in results["decompose"].)
    print(
        "  box:",
        win.trend_series.currentText(),
        "/",
        win.tre_seasonal.currentText(),
        "/",
        win.tre_segments.value(),
        "/",
        win.tre_n_bootstrap.value(),
    )
    win.show_panel_page("Decompose")  # the Trend analysis box sits on the panel page
    win._run_trend()
    t = win.results["trend"]
    print(t.to_string())
    said(win.tab_trend)
    grab(win, "ui_trend.png")

    print("segmented trend (two segments)")
    # Raising the Segments row moves the still-default emission part to the
    # separated long-term trend (the box's own behaviour), so the segmented
    # fit runs on the cleaner input, under the box's settings.
    win.tre_segments.setValue(2)
    print(
        "  box:",
        win.trend_series.currentText(),
        "/",
        win.tre_segments.value(),
        "/",
        win.tre_n_bootstrap.value(),
    )
    win._run_trend()
    t = win.results["trend"]
    print(t.to_string())
    said(win.tab_trend)
    grab(win, "ui_trend_breaks.png")
    win.tre_segments.setValue(1)  # the rest of the run stands on the default
    win.trend_series.setCurrentText("emission part (emi_total)")

    print("events (emission part)")
    # The Events & anomalies box under Trend, on the emission part the Trend
    # test just ran (the box's default choice), the iqr score at threshold 2
    # and minimum length 3. The score strip with its flagged windows shaded and
    # the event table land on the Events tab; the numbers the manual quotes are
    # this run's own. The threshold is 2, not the box's default 3: on MY1's
    # de-weathered emission part nothing clears 3, so 2 is the first value that
    # shows the mechanism (the strip's shading is exactly where the score
    # crosses the line). It must run before the meteorology decomposition
    # overwrites results["decompose"] with its met columns.
    win.events_series.setCurrentText("emission part (emi_total)")
    win.events_method.setCurrentText("iqr")
    win.events_k.setValue(2.0)
    win.events_min_len.setValue(3)
    print(
        "  box:",
        win.events_series.currentText(),
        "/",
        win.events_method.currentText(),
        "/",
        win.events_k.value(),
        "/",
        win.events_min_len.value(),
    )
    win._run_events()
    ev = win.results["events"]
    top = float(ev["max_score"].max()) if len(ev) else float("nan")
    longest = int(ev["n"].max()) if len(ev) else 0
    print("  windows:", len(ev), "max score:", round(top, 3), "longest stretch (steps):", longest)
    if len(ev):
        for _, r in ev.head(3).iterrows():
            print(
                "  window:",
                str(r["start"]),
                "->",
                str(r["end"]),
                "steps",
                int(r["n"]),
                "max",
                round(float(r["max_score"]), 3),
            )
    said(win.tab_events)
    grab(win, "ui_events.png")

    print("decompose: meteorology, four groups")
    win.decom_method.setCurrentText("meteorology")
    win.decom_attribution.setCurrentText("auto")
    win.decom_groups.suggest()  # wind, temperature and humidity, boundary layer, sky
    win._run_decompose()
    met = win.results["decompose"]
    print("  columns:", list(met.columns))
    groups = ["Wind", "Temperature and humidity", "Boundary layer and pressure", "Sky and rain"]
    print("  sd:", met[groups].std().round(2).to_dict())
    said(win.tab_decom)
    grab(win, "ui_decompose_met.png")

    print("rolling")
    win.show_panel_page("Rolling")  # the panel's page for the step
    win.roll_window.setValue(14)
    win.roll_step.setValue(7)
    win.roll_samples.setValue(50)
    win._run_rolling()
    rolled = win.results["rolling"]
    print("  windows:", len(window_columns(rolled)))
    grab(win, "ui_rolling.png")

    print("multi-scale")
    win.show_panel_page("Decompose")  # the panel's page for the step
    win.ms_fast.setValue(14)
    win.ms_meso.setValue(60)
    win.ms_slow.setValue(120)
    win._run_multiscale()
    said(win.tab_multiscale)
    grab(win, "ui_multiscale.png")

    print("pdp")
    win.show_panel_page("PDP")  # the panel's page for the step
    tick(win.pdp_vars, ["ws", "temp", "blh", "hour"])
    win.pdp_grid.setValue(40)
    win._run_pdp()
    grab(win, "ui_pdp.png")

    print("uncertainty decomposition")
    # The box on the panel page; its members train fresh from the table as
    # loaded, so a lightgbm backend and a small search keep the figure run a
    # minute or two, not an hour.
    win.show_panel_page("Decompose")  # the panel page the box sits on
    win.unc_backend.setCurrentText("lightgbm")
    win.budget_spin.setValue(5)
    win.unc_models.setValue(3)
    win.unc_samples.setValue(50)
    win._run_unc()
    u = win.results["uncertainty"]
    print(
        "  median share:",
        {c: float(np.median(u[c])) for c in ("pct_weather", "pct_model", "pct_interaction")},
    )
    print("  median spread:", u["var_total"].median())
    said(win.tab_unc)
    grab(win, "ui_uncertainty.png")

    print("command log")
    win.show_log_page(1)  # View → Command Log…: a window of its own
    grab(win._log_windows[1], "ui_command_log.png")
    win._log_windows[1].close()
    # The script names the temporary CSV, so it is printed, not kept beside the figures.
    print(win.command_log.script())

    print("print dialog")
    from normet.gui.printview import PrintDialog

    canvas = win.tab_norm.findChild(FigureCanvasQTAgg)
    dlg = PrintDialog(canvas, win, settings={"preset": "nature"}, on_settings=lambda s: None)
    dlg.resize(1280, 860)
    dlg.show()
    grab(dlg, "ui_print.png")
    dlg.reject()
    _settle(win)

    print("polar plot")
    win.open_polar_plot()
    polar = win._polar_window
    polar.resize(760, 640)
    polar.show()
    grab(polar, "ui_polar.png")
    polar.close()

    print("methods paragraph")
    win.show_methods_paragraph()
    methods = win._methods_dialog
    methods.resize(760, 560)
    grab(methods, "ui_methods.png")
    methods.close()
    _settle(win)
    return win


def scm_figures(parent):
    from normet.gui.scm_window import SCMWindow

    print("scm")
    w = SCMWindow(parent=None)
    w.resize(*SIZE)
    w.show()
    run_inline(w.runner)
    # The README's design, as the studio's File > Load Example > London ULEZ sets it up:
    # pre-COVID window, traffic-type donors with a complete record.
    w.load_example("ulez")
    print(f"  {w.donor_list.count()} donors, {w.df['date'].nunique()} months")
    w.panel.show_page("Design")  # the treated unit, cutoff and donors in view
    said(w.tab_panel)
    grab(w, "ui_scm_panel.png")

    w.panel.show_page("Estimator")
    w.backend_combo.setCurrentText("scm")
    w._run_fit()
    said(w.tab_fit)
    d = w.diagnostics
    print(
        "  diagnostics:",
        {
            k: d[k]
            for k in (
                "pre_r2",
                "pre_rmse",
                "att",
                "hhi",
                "effective_n_donors",
                "n_donors",
                "top_donors",
            )
        },
    )
    grab(w, "ui_scm_fit.png")
    w.panel.show_page("Inference")
    w._run_placebo_space()
    said(w.tab_pspace)
    grab(w, "ui_scm_placebo.png")
    w.unc_method.setCurrentText("jackknife")
    w._run_uncertainty()
    said(w.tab_unc)
    grab(w, "ui_scm_bands.png")
    return w


def transport_figures():
    import normet as nm
    from normet.gui.trajectory_window import TrajectoryWindow

    print("transport")
    w = TrajectoryWindow(parent=None)
    w.resize(*SIZE)
    w.show()
    run_inline(w.runner)
    folder = nm.datasets.example_traj_dir()
    for name in sorted(os.listdir(folder)):
        if name.startswith("tdump"):
            p = os.path.join(str(folder), name)
            w._tdump_paths.append(p)
            w.tdump_list.addItem(name)
    w.build_site_edit.setText("MY1")
    w._sync_enabled()
    w._run_build()
    grab(w, "ui_transport.png")
    return w


def data_studio_figure():
    import json

    import pandas as pd

    from normet.gui.data_window import EEA_NETWORK, DataWindow

    print("data studio")
    w = DataWindow(parent=None)
    w.resize(*SIZE)
    # Show the non-UK provider: the Source settings group (country code) is the
    # new face of the window, and DE has enough stations to matter.
    w.net_combo.setCurrentText(EEA_NETWORK)
    w.country_edit.setText("DE")
    # The stations of a real EEA search for DE (captured 2026-10-24, 55 of 787,
    # spread over the country): the map and table show what a search fills them
    # with. Their OpenStreetMap tiles are fetched when the figure is built (and
    # cached under ~/.normet/osm_tiles); offline the map falls back to its grid.
    with open(os.path.join(HERE, "data_studio_stations.json"), encoding="utf-8") as fh:
        stations = pd.DataFrame(json.load(fh))
    w.stations = stations
    w._fill_station_table(stations)
    # Three of them selected, as a box drawn on the map would: the rings, the
    # table's highlighted rows and the hint all show the multi-station fetch.
    # Rows 1, 3 and 5 sit far apart (Aachen, Aalen, Altenburg) and stay in the
    # table's first screen, so the rings and the highlighted rows are both seen.
    picked = [int(i) for i in stations.index[:5:2]]
    w.station_map.selectionChanged.emit(picked)
    w.show()
    # The tiles are fetched on a timer once the search is shown; the figure is
    # grabbed now, so fetch them here (cached, a handful) instead of waiting.
    w.station_map._load_basemap(True)
    grab(w, "ui_data_studio.png")
    return w


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")  # as normet-gui starts it
    app.setApplicationName("Normet")
    app.setOrganizationName("apai-sys")
    from normet.gui import uilang

    uilang.install(app)  # the window's texts in LANG, before any window is built
    from normet.style import apply_style

    apply_style()  # the figures' style, as normet-gui sets it
    keep = [main_figures(app), scm_figures(None), transport_figures(), data_studio_figure()]
    print(f"done: {len(keep)} windows -> {OUT}")


if __name__ == "__main__":
    main()
