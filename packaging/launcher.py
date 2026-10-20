"""PyInstaller entry point for the normet GUI.

``freeze_support`` must run before anything else so that joblib/loky worker
processes re-spawned from the frozen binary bootstrap correctly instead of
opening another GUI window.

Then, before ``import normet`` (which takes seconds: pandas, matplotlib, the
whole library), the font cache is pointed somewhere that survives the run and
a splash screen goes up, so a double-click is answered at once.
"""

import multiprocessing

multiprocessing.freeze_support()

import os  # noqa: E402
import sys  # noqa: E402


def _keep_font_cache() -> None:
    """Give matplotlib a font cache that survives the run. Before importing it.

    PyInstaller's runtime hook points MPLCONFIGDIR at a fresh temporary
    directory and deletes it at exit, so the frozen app rebuilt matplotlib's
    font list on every launch: about 20 s on a Mac with the system's fonts.
    The hook exists for onefile apps, whose bundled font paths change with
    each unpacking; this app is a directory bundle, where they do not.

    Keyed by where the bundle is, so moving the app rebuilds the list once
    instead of reading paths that no longer exist.
    """
    bundle = getattr(sys, "_MEIPASS", None)
    if not bundle:
        return
    import hashlib

    here = hashlib.sha1(os.path.abspath(bundle).encode()).hexdigest()[:12]
    cache = os.path.join(os.path.expanduser("~"), ".normet", "matplotlib", here)
    try:
        os.makedirs(cache, exist_ok=True)
    except OSError:
        return  # keep the throwaway one rather than fail to start
    os.environ["MPLCONFIGDIR"] = cache


def _splash(app):
    """A plain splash card: drawn with Qt alone, so it shows before normet loads."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import QSplashScreen

    try:
        from importlib.metadata import version

        ver = version("normet")
    except Exception:
        ver = ""

    w, h = 440, 250
    pix = QPixmap(w, h)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QBrush(QColor("#1c1c1e")))
    p.setPen(QPen(QColor("#3a3a3c"), 1.5))
    p.drawRoundedRect(1, 1, w - 2, h - 2, 14, 14)

    logo = QPixmap(os.path.join(getattr(sys, "_MEIPASS", ""), "packaging", "assets", "normet.png"))
    top = 40
    if not logo.isNull():
        logo = logo.scaled(
            64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        p.drawPixmap((w - logo.width()) // 2, top, logo)
        top += 76
    else:
        top += 30

    p.setPen(QColor("#ffffff"))
    p.setFont(QFont("Helvetica Neue", 22, QFont.Weight.Bold))
    p.drawText(0, top, w, 32, Qt.AlignmentFlag.AlignCenter, "Normet")
    p.setPen(QColor("#98989d"))
    p.setFont(QFont("Helvetica Neue", 11))
    sub = "Weather normalisation & counterfactual modelling"
    p.drawText(
        0, top + 34, w, 20, Qt.AlignmentFlag.AlignCenter, f"{sub}  •  v{ver}" if ver else sub
    )
    p.end()

    splash = QSplashScreen(
        pix, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
    )
    splash.show()
    splash.showMessage(
        "Loading…",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        QColor("#8e8e93"),
    )
    app.processEvents()
    return splash


_keep_font_cache()

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication(sys.argv)
try:
    _splash_screen = _splash(_app)
except Exception:  # a splash is a courtesy, never a reason not to start
    _splash_screen = None

from normet.gui.app import main  # noqa: E402

raise SystemExit(main(splash=_splash_screen))
