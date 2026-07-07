# Cross-platform PyInstaller spec for the normet Qt GUI.
#
#   macOS  : packaging/macos/build_dmg.sh (wraps this same launcher via CLI
#            flags) is the primary path and also produces the .dmg; this spec
#            works there too if invoked directly.
#   Windows: pyinstaller --noconfirm packaging/normet_gui.spec
#            -> dist/normet/normet.exe (then packaging/windows/installer.iss)
#   Linux  : pyinstaller --noconfirm packaging/normet_gui.spec
#            -> dist/normet/normet (then packaging/linux/build_appimage.sh)
#
# All paths are anchored on SPECPATH (this file's own directory, injected by
# PyInstaller) rather than the current working directory, so the build works
# the same whether invoked from the repo root or from packaging/ itself.
import os
import re
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

REPO = os.path.dirname(SPECPATH)  # noqa: F821 — SPECPATH is injected by PyInstaller


def _read_version():
    """Single source of truth: pyproject.toml's [project] version."""
    with open(os.path.join(REPO, "pyproject.toml"), encoding="utf-8") as fh:
        m = re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.M)
    return m.group(1) if m else "0.0.0"


APP_VERSION = _read_version()
_ICON = {
    "darwin": os.path.join(SPECPATH, "assets", "normet.icns"),  # noqa: F821
    "win32": os.path.join(SPECPATH, "assets", "normet.ico"),  # noqa: F821
}.get(sys.platform)

hiddenimports = [
    "matplotlib.backends.backend_qtagg",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]
hiddenimports += collect_submodules("normet")
hiddenimports += collect_submodules("matplotlib.backends")

datas = [(os.path.join(SPECPATH, "assets", "normet.png"), "packaging/assets")]  # noqa: F821
binaries = []
# flaml/lightgbm are optional AutoML backends but normal enough to ship by
# default (xgboost is one of flaml's estimator_list choices); shapely/pyshp
# back Transport Studio's GeoJSON/Shapefile source-region loading.
for pkg in ("flaml", "lightgbm", "shapely"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += ["shapefile"]

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],  # noqa: F821
    pathex=[os.path.join(REPO, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        "tkinter", "IPython", "pytest",
        "PyQt5", "PyQt6", "PySide2",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="Normet", debug=False, strip=False, upx=False, console=False,
    target_arch=None, icon=_ICON,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Normet")

# A macOS .app bundle is Apple-specific; Windows/Linux ship the COLLECT
# onedir directly (see packaging/windows and packaging/linux).
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Normet.app",
        icon=_ICON,
        bundle_identifier="org.apai-sys.normet",
        version=APP_VERSION,
        info_plist={
            "CFBundleName": "Normet",
            "CFBundleDisplayName": "Normet",
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSHumanReadableCopyright": (
                "normet — weather normalisation & counterfactual modelling"
            ),
        },
    )
