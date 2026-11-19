"""Application entry point for the normet GUI."""

from __future__ import annotations

import sys


def _install_excepthook() -> None:
    """Show a dialog instead of letting an unhandled slot exception abort the app.

    By default Qt terminates the process on an uncaught exception in a slot,
    which looks like the app 'just quitting'.  This keeps the window alive and
    surfaces the error.
    """
    import os
    import threading
    import time
    import traceback

    from PySide6.QtWidgets import QMessageBox

    from .uilang import tr

    def _log_to_file(msg):
        path = os.path.join(os.path.expanduser("~"), ".normet", "error.log")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{msg}\n")
            return path
        except OSError:
            return None

    def hook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        path = _log_to_file(msg)
        if sys.stderr is not None:
            try:
                sys.stderr.write(msg)
            except Exception:
                pass
        if threading.current_thread() is not threading.main_thread():
            return  # widgets only on the GUI thread; the file has it
        where = tr(f"Details were saved to {path}.") if path else tr("Details could not be saved.")
        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("Unexpected error")
            box.setText(f"{exc_type.__name__}: {exc}")
            box.setInformativeText(
                tr("The app stayed open; your results are unaffected.") + " " + where
            )
            box.setDetailedText(msg)
            box.exec()
        except Exception:
            pass

    sys.excepthook = hook


def _skip_qt_import_scan() -> None:
    """Stop PySide6's import hook from reading the source of every later module.

    ``import PySide6`` replaces ``builtins.__import__`` with a hook that, for
    each module imported afterwards, reads the module's source to decide whether
    it uses ``from __feature__ import ...``
    (``shibokensupport.feature.feature_imported``). A module already named in
    ``pyside_feature_dict`` is not read. For a library this size that read is
    the startup: importing normet after PySide6 measured **6.2 s against 1.1 s**
    the other way round (warm, both orders; pandas and numpy alone are hundreds
    of files), and the window's own modules paid ~1 s more when first imported.

    The mapping below answers "already known" for every module that is not Qt's.
    None of normet's modules uses the feature switch (no ``from __feature__
    import`` in the tree), so the scan has nothing to find; PySide6's own
    modules still get their feature defaults. Guarded because this is a PySide6
    internal: a version that renames either object starts the app unpatched,
    never broken.
    """
    try:
        import importlib

        feature = importlib.import_module("shibokensupport.feature")
    except Exception:  # pragma: no cover - PySide6 always provides it
        return

    class _NotQtModules(dict):
        def __contains__(self, name: object) -> bool:
            if isinstance(name, str) and not name.startswith(("PySide6", "shiboken6")):
                return True
            return dict.__contains__(self, name)

    try:
        feature.pyside_feature_dict = _NotQtModules(  # type: ignore[attr-defined]
            feature.pyside_feature_dict
        )
    except Exception:  # pragma: no cover - likewise
        return


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

    # First, before any module import of ours: PySide6 is loaded now, and its
    # hook would read each later module's file (see the docstring above).
    _skip_qt_import_scan()
    _configure_frozen_runtime()

    import matplotlib

    matplotlib.use("QtAgg")
    from normet.style import apply_style

    apply_style()  # every figure in the journal style; exports cropped tight

    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(argv)
    # Cross-platform Fusion style so the layout is tidy and identical on every
    # OS (native macOS style sizes/aligns controls differently).
    app.setStyle("Fusion")
    app.setApplicationName("Normet")
    app.setApplicationDisplayName("Normet")
    app.setOrganizationName("apai-sys")
    from PySide6.QtGui import QIcon

    from ._widgets import app_icon_path

    # Every window's icon on Windows and Linux; on macOS the Dock's while run from source.
    app.setWindowIcon(QIcon(app_icon_path()))
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
    window.start_layout()  # panel folded, Assistant open
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
