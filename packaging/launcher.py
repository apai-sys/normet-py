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


def _splash_look(app):
    """The card's theme and language, from the settings the window will use.

    Read with Qt alone, before ``import normet``: the stored ``appearance``
    (View → Appearance) and ``ui_language`` (View → Language), the system's when
    neither is pinned. The window applies the same two choices when it starts, so the
    card and the window agree.
    """
    from PySide6.QtCore import QLocale, QSettings, Qt
    from PySide6.QtGui import QPalette

    s = QSettings("apai-sys", "normet-gui")
    appearance = str(s.value("appearance", "system"))
    if appearance == "dark":
        dark = True
    elif appearance == "light":
        dark = False
    else:
        hints = app.styleHints()
        scheme = hints.colorScheme() if hasattr(hints, "colorScheme") else None
        schemes = getattr(Qt, "ColorScheme", None)
        if schemes is not None and scheme == schemes.Dark:
            dark = True
        elif schemes is not None and scheme == schemes.Light:
            dark = False
        else:
            dark = app.palette().color(QPalette.ColorRole.Window).lightnessF() < 0.5

    forced = os.environ.get("NORMET_UI_LANGUAGE")
    choice = forced if forced in ("en", "zh") else str(s.value("ui_language", "system"))
    lang = (
        choice
        if choice in ("en", "zh")
        else ("zh" if QLocale.system().name().startswith("zh") else "en")
    )
    return dark, lang


def _splash(app):
    """A plain splash card: drawn with Qt alone, so it shows before normet loads.

    In the window's theme and language -- the start-up wait is the first thing the
    app shows, and a dark card in front of a light window, or an English line in a
    Chinese one, made the two look like different programs.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import QSplashScreen

    try:
        from importlib.metadata import version

        ver = version("normet")
    except Exception:
        ver = ""

    dark, lang = _splash_look(app)
    if dark:
        top, bottom, edge, ink, sub, line = (
            "#1c1c1e",
            "#121214",
            "#3a3a3c",
            "#ffffff",
            "#98989d",
            "#2c2c2e",
        )
    else:
        top, bottom, edge, ink, sub, line = (
            "#ffffff",
            "#f2f2f7",
            "#d1d1d6",
            "#1c1c1e",
            "#6e6e73",
            "#e5e5ea",
        )

    w, h = 440, 250
    pix = QPixmap(w, h)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    grad = QLinearGradient(0, 0, 0, h)
    grad.setColorAt(0.0, QColor(top))
    grad.setColorAt(1.0, QColor(bottom))
    p.setBrush(QBrush(grad))
    p.setPen(QPen(QColor(edge), 1.5))
    p.drawRoundedRect(1, 1, w - 2, h - 2, 14, 14)

    logo = QPixmap(os.path.join(getattr(sys, "_MEIPASS", ""), "packaging", "assets", "normet.png"))
    top_y = 40
    if not logo.isNull():
        logo = logo.scaled(
            64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        p.drawPixmap((w - logo.width()) // 2, top_y, logo)
        top_y += 76
    else:
        top_y += 30

    p.setPen(QColor(ink))
    p.setFont(QFont("Helvetica Neue", 22, QFont.Weight.Bold))
    p.drawText(0, top_y, w, 32, Qt.AlignmentFlag.AlignCenter, "Normet")
    p.setPen(QColor(sub))
    p.setFont(QFont("Helvetica Neue", 11))
    words = (
        "气象归一化与反事实建模"
        if lang == "zh"
        else "Weather normalisation & counterfactual modelling"
    )
    p.drawText(
        0, top_y + 34, w, 20, Qt.AlignmentFlag.AlignCenter, f"{words}  •  v{ver}" if ver else words
    )
    p.setPen(QPen(QColor(line), 1))
    p.drawLine(30, h - 25, w - 30, h - 25)
    p.end()

    splash = QSplashScreen(
        pix, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
    )
    splash.show()
    splash.showMessage(
        "载入中…" if lang == "zh" else "Loading…",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        QColor(sub),
    )
    app.processEvents()
    return splash


def _skip_qt_import_scan() -> None:
    """Stop PySide6's import hook from reading every later module's source.

    Same patch as ``normet.gui.app._skip_qt_import_scan`` (read its docstring
    for the measurements). It is repeated here because the launcher imports Qt
    before normet, and importing the helper would be the very import it saves:
    this script deliberately touches nothing but Qt until after the splash.
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


_keep_font_cache()

from PySide6.QtWidgets import QApplication  # noqa: E402

# Before `import normet` below: PySide6's hook would otherwise read the source
# of pandas, numpy and the whole library (6.2 s against 1.1 s, warm).
_skip_qt_import_scan()

_app = QApplication(sys.argv)
try:
    _splash_screen = _splash(_app)
except Exception:  # a splash is a courtesy, never a reason not to start
    _splash_screen = None

from normet.gui.app import main  # noqa: E402

raise SystemExit(main(splash=_splash_screen))
