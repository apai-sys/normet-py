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

The tasks run on the calling thread instead of the window's worker thread, so a
grab happens after the result is on screen; everything else is the window's own code.
"""

import os
import sys
import tempfile
import traceback

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
_HOME = tempfile.mkdtemp(prefix="normet-figures-home-")
os.environ["HOME"] = _HOME
os.environ["USERPROFILE"] = _HOME

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from PySide6.QtCore import QDate, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = os.path.join(HERE, "figures")
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
    import normet as nm

    win = main_window()
    print("welcome")
    grab(win, "ui_welcome.png")

    # MY1 as a CSV, opened as File > Open CSV opens it, so the command log starts
    # from a read_csv a reader can repeat.
    data_dir = tempfile.mkdtemp(prefix="normet-figures-data-")
    path = os.path.join(data_dir, "my1.csv")
    nm.datasets.load_my1().to_csv(path, index=False)
    win.load_csv(path)
    tick(win.feature_list, MY1_MET)
    print("data:", win.data_summary.text())
    grab(win, "ui_data.png")

    print("train")
    win.backend_combo.setCurrentText("flaml")
    win.budget_spin.setValue(30)
    tick(win.estimator_list, ["lgbm"])
    win.split_combo.setCurrentText("random")
    win._run_train()
    print(win.results["train"].to_string())
    said(win.tab_model)
    grab(win, "ui_model.png")

    print("normalise")
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
    win.decom_method.setCurrentText("emission")
    win.decom_samples.setValue(100)
    win._run_decompose()
    print("  columns:", list(win.results["decompose"].columns))
    emi = win.results["decompose"]
    print(
        "  sd:",
        emi[["date_unix", "day_julian", "weekday", "hour", "emi_noise"]].std().round(2).to_dict(),
    )
    said(win.tab_decom)
    grab(win, "ui_decompose_emission.png")

    print("decompose: meteorology, four groups")
    win.decom_method.setCurrentText("meteorology")
    win.decom_attribution.setCurrentText("auto")
    win.decom_groups.setText(
        "wind: ws, wd; thermo: temp, RH; mixing: blh, atmos_pres; sky: tcc, tp, ssrd"
    )
    win._run_decompose()
    met = win.results["decompose"]
    print("  columns:", list(met.columns))
    print("  sd:", met[["wind", "thermo", "mixing", "sky"]].std().round(2).to_dict())
    said(win.tab_decom)
    grab(win, "ui_decompose_met.png")

    print("rolling")
    win.roll_window.setValue(14)
    win.roll_step.setValue(7)
    win.roll_samples.setValue(50)
    win._run_rolling()
    rolled = win.results["rolling"]
    print("  windows:", sum(c.startswith("rolling_") for c in rolled.columns))
    grab(win, "ui_rolling.png")

    print("multi-scale")
    win.ms_fast.setValue(14)
    win.ms_meso.setValue(60)
    win.ms_slow.setValue(120)
    win._run_multiscale()
    said(win.tab_multiscale)
    grab(win, "ui_multiscale.png")

    print("pdp")
    tick(win.pdp_vars, ["ws", "temp", "blh", "hour"])
    win.pdp_grid.setValue(40)
    win._run_pdp()
    grab(win, "ui_pdp.png")

    print("command log")
    win._command_dock.show()
    win._command_dock.raise_()
    win.tabs.setCurrentWidget(win.tab_norm)
    grab(win, "ui_command_log.png")
    # The script names the temporary CSV, so it is printed, not kept beside the figures.
    print(win.command_log.script())
    win._log_dock.raise_()

    print("print dialog")
    from normet.gui.printview import PrintDialog

    canvas = win.tab_norm.findChild(FigureCanvasQTAgg)
    dlg = PrintDialog(canvas, win)
    dlg.resize(1180, 820)
    dlg.show()
    grab(dlg, "ui_print.png")
    dlg.reject()
    _settle(win)
    return win


def scm_figures(parent):
    import normet as nm
    from normet.gui.scm_window import SCMWindow

    print("scm")
    # The README's design: pre-COVID window, traffic-type donors with a complete record.
    scm_df = nm.datasets.load_scm()
    df = scm_df.query("'2017-01-01' <= date < '2020-03-01'")
    df = df[df["type"].str.contains("Traffic")]
    counts = df.dropna(subset=["NO2_dw"]).groupby("code").size()
    donors = [c for c in counts[counts == df["date"].nunique()].index if c != "MY1"]
    df = df[df["code"].isin(donors + ["MY1"])]
    print(f"  {len(donors)} donors, {df['date'].nunique()} months")

    w = SCMWindow(parent=None)
    w.resize(*SIZE)
    w.show()
    run_inline(w.runner)
    w._ingest(df.reset_index(drop=True), "load_scm(): ULEZ design")
    w.unit_combo.setCurrentText("code")
    w.outcome_combo.setCurrentText("NO2_dw")
    w._columns_changed()
    w.treated_combo.setCurrentText("MY1")
    w.cutoff_edit.setDate(QDate(2019, 4, 1))
    w._redraw_panel_tab()
    said(w.tab_panel)
    grab(w, "ui_scm_panel.png")

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
    from normet.gui.data_window import DataWindow

    print("data studio")
    w = DataWindow(parent=None)
    w.resize(*SIZE)
    w.show()
    grab(w, "ui_data_studio.png")
    return w


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")  # as normet-gui starts it
    app.setApplicationName("Normet")
    app.setOrganizationName("apai-sys")
    keep = [main_figures(app), scm_figures(None), transport_figures(), data_studio_figure()]
    print(f"done: {len(keep)} windows")


if __name__ == "__main__":
    main()
