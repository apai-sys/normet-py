"""Application entry point for the normet GUI."""

from __future__ import annotations

import sys


def _install_excepthook() -> None:
    """Show a dialog instead of letting an unhandled slot exception abort the app.

    By default Qt terminates the process on an uncaught exception in a slot,
    which looks like the app 'just quitting'.  This keeps the window alive and
    surfaces the error.
    """
    import traceback

    from PySide6.QtWidgets import QMessageBox

    def hook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(msg)
        try:
            QMessageBox.critical(
                None,
                "Unexpected error",
                f"{exc_type.__name__}: {exc}\n\nThe app stayed open; your results "
                "are unaffected. Details were logged to the console.",
            )
        except Exception:
            pass

    sys.excepthook = hook


def _configure_frozen_runtime() -> None:
    """Make joblib safe inside a PyInstaller bundle.

    joblib's default loky backend launches workers by re-executing
    ``sys.executable`` — in a frozen app that is the GUI binary itself, so
    the workers die immediately (``TerminatedWorkerError``, seen in pdp and
    the SCM inference tools). Fall back to the threading backend:
    LightGBM/FLAML predictions release the GIL, so threaded workers still
    parallelise the heavy parts.
    """
    if not getattr(sys, "frozen", False):
        return
    try:
        import joblib.parallel

        joblib.parallel.DEFAULT_BACKEND = "threading"
    except Exception:  # pragma: no cover - joblib always present via normet
        pass


def _warm_up() -> None:
    """Import the modelling stack once the window is up, off the interface thread.

    sklearn, scipy.stats and flaml are not imported by ``import normet``; that
    keeps them out of the wait for the window, but on its own would only move
    the wait (flaml alone is ~4 s) to the first Train. A thread started after
    the window shows pays it while the user is still reading the screen, and
    Python's import locks make a later import on the main thread wait for this
    one rather than repeat it.
    """
    import threading

    def load() -> None:
        for name in ("scipy.stats", "sklearn.linear_model", "flaml.automl"):
            try:
                __import__(name)
            except Exception:  # noqa: BLE001 - a head start, not a step
                pass

    threading.Thread(target=load, name="normet-warm-up", daemon=True).start()


def main(argv: list[str] | None = None, *, splash: object | None = None) -> int:
    """Start the GUI.

    ``splash`` is a ``QSplashScreen`` the frozen launcher put up before
    ``import normet`` (packaging/launcher.py); it is closed when the window
    shows. The launcher also made the ``QApplication``, which is reused.
    """
    argv = list(sys.argv if argv is None else argv)
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise SystemExit(
            "PySide6 is required for the GUI. Install it with:\n    pip install 'normet[gui]'"
        ) from exc

    _configure_frozen_runtime()

    import matplotlib

    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"savefig.bbox": "tight"})  # toolbar exports cropped tight

    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(argv)
    # Cross-platform Fusion style so the layout is tidy and identical on every
    # OS (native macOS style sizes/aligns controls differently).
    app.setStyle("Fusion")
    app.setApplicationName("Normet")
    app.setApplicationDisplayName("Normet")
    app.setOrganizationName("apai-sys")
    _install_excepthook()

    # The window's language, before anything is built: its texts are translated as
    # they are set (.uilang) -- the system's language unless View → Language says.
    from . import uilang

    uilang.install(app)

    from .theme import apply_stored_appearance, theme_manager

    apply_stored_appearance()  # View → Appearance, as last chosen
    theme_manager()

    from .main_window import MainWindow

    window = MainWindow()
    window.start_layout()  # panel folded, Assistant open -- as cume opens
    window.show()
    if splash is not None:
        splash.finish(window)  # type: ignore[attr-defined]

    from PySide6.QtCore import QTimer

    QTimer.singleShot(0, _warm_up)

    # `normet-gui data.csv` opens the file straight away.
    for arg in argv[1:]:
        if arg.lower().endswith((".csv", ".csv.gz")):
            window.load_csv(arg, remember=True)
            break

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
