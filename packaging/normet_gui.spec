# Cross-platform PyInstaller spec for the normet Qt GUI.
#
#   macOS  : packaging/macos/build_dmg.sh builds from this spec
#            -> dist/macos/Normet.app, then packs it into the .dmg
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

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

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
# The example datasets (File → Examples) ship inside the package.
datas += collect_data_files("normet")
# Help → User Manual opens the manual in the window's language, from the bundle.
for _book in ("normet_manual.pdf", "normet_manual_zh.pdf"):
    _path = os.path.join(REPO, "docs", "manual", _book)
    if os.path.exists(_path):
        datas.append((_path, "docs/manual"))
binaries = []
# flaml/lightgbm are optional AutoML backends but normal enough to ship by
# default. xgboost is not optional once flaml is: flaml.automl imports it at
# module level and silently drops AutoML if that fails. shapely/pyshp back
# Transport Studio's GeoJSON/Shapefile source-region loading.
def _not_tests(name):
    return not {"tests", "testing"} & set(name.split("."))


# Their own test suites stay out: importing them can need what the app does not
# ship (shapely's tests and xgboost.testing need hypothesis), which stops the build.
for pkg in ("flaml", "lightgbm", "xgboost", "shapely"):
    d, b, h = collect_all(pkg, filter_submodules=_not_tests)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += ["shapefile"]
# The data downloaders (Data → Download …, e.g. UK-AIR "Find stations") import
# requests / cdsapi / rdata through normet.utils.require(), i.e. by name at run
# time, which PyInstaller's import scan cannot see. Name them, or the frozen app
# says "'rdata' is required but not installed" although the build env had it.
hiddenimports += ["requests", "cdsapi", "rdata"]
# pandas reaches its table readers the same way (Open Data's Parquet, Feather, Excel).
hiddenimports += ["pyarrow", "openpyxl", "xlrd"]

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],  # noqa: F821
    pathex=[os.path.join(REPO, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        "tkinter", "IPython", "pytest",
        "PyQt5", "PyQt6", "PySide2",
        # NVIDIA's multi-GPU library (452 MB on Linux), which xgboost's Linux wheel
        # pulls in. xgboost opens it only for distributed GPU training and goes on
        # without it otherwise.
        "nvidia",
        # Chronos-2 stays out of the installers: torch alone is ~300 MB. The app says
        # it is not installed; the pip install (normet[foundation]) has it.
        "torch", "chronos",
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
            # The languages the app speaks: without them macOS takes the
            # app for English-only and draws its own panels and menu items in English
            # on a Chinese system -- they follow the system only where it is one of these.
            "CFBundleDevelopmentRegion": "en",
            "CFBundleLocalizations": ["en", "zh-Hans"],
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSHumanReadableCopyright": (
                "normet — weather normalisation & counterfactual modelling"
            ),
        },
    )
