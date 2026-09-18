# Building normet GUI installers

The GUI (`normet.gui`) is PySide6 + matplotlib, so the frozen app is
cross-platform. PyInstaller, however, **cannot cross-compile** — each OS's
installer must be built on that OS. The supported path is GitHub Actions
(`.github/workflows/build-gui.yml`), one spec, three OSes.

## Local build (macOS only)

```bash
packaging/macos/build_dmg.sh /path/to/python   # -> dist/macos/normet.app + .dmg
```

## Local build (Windows only)

```powershell
packaging\windows\build_installer.ps1               # -> dist\normet-setup-<version>.exe
packaging\windows\build_installer.ps1 -SkipFreeze   # recompile the installer only
```

Needs PyInstaller and the GUI extras on the interpreter it picks up (`-Python`
selects another one), plus Inno Setup 6:
`winget install --id JRSoftware.InnoSetup -e`.

## Local build (any OS, onedir — no installer wrapper)

```bash
pip install -e ".[gui,flaml,lgb,geo]" pyinstaller
pyinstaller --noconfirm packaging/normet_gui.spec   # -> dist/Normet/
```

## All three platforms via GitHub Actions

The workflow runs a matrix on `macos-latest`, `windows-latest`, `ubuntu-latest`.
Each runner installs the deps, runs PyInstaller (`packaging/normet_gui.spec`
or, on macOS, `packaging/macos/build_dmg.sh`), then packages its native
installer and uploads it as an artifact:

| OS | output |
|----|--------|
| macOS | `normet-<version>-macos-<arch>.dmg` (drag-to-Applications) |
| Windows | `normet-setup-<version>.exe` (Inno Setup installer) |
| Linux | `normet-<version>-x86_64.AppImage` (chmod +x, double-click) |

Trigger it by pushing to `main` (paths: `src/normet/gui/**`, `packaging/**`),
or manually from the Actions tab (`workflow_dispatch`).

## Packaging files

- `normet_gui.spec` — cross-platform PyInstaller spec (macOS `.app` BUNDLE;
  Windows/Linux ship the COLLECT onedir `dist/Normet/`). Version is read from
  `pyproject.toml` at build time — no separate version to keep in sync.
- `assets/normet.icns` / `normet.ico` / `normet.png` — per-OS icons, generated
  by `assets/make_icon.py` (Pillow). Re-run it if the design should change;
  the outputs are committed so CI doesn't need to regenerate them.
- `packaging/windows/installer.iss` — Inno Setup script.
- `packaging/windows/build_installer.ps1` — freeze + Inno Setup in one step,
  the Windows counterpart to the macOS/Linux shell scripts.
- `packaging/linux/build_appimage.sh` — AppDir + appimagetool.
- `launcher.py` — the frozen entry point (`multiprocessing.freeze_support()`
  then `normet.gui.main()`), shared by every OS's build.

## Notes / caveats

- flaml + lightgbm + xgboost (AutoML backends) are bundled by default — this
  makes the installer sizeable but keeps "Train model" working out of the box.
  xgboost is not optional once flaml is: `flaml.automl.model` imports it at
  module level and `flaml.automl/__init__.py` swallows the ImportError, so
  without it the app fails with *Module 'flaml.automl:AutoML' does not provide
  the requested attribute* for every estimator, not just the XGBoost ones.
  That is why the `flaml` extra is `flaml[automl]`.
- shapely + pyshp are bundled too, so Transport Studio's GeoJSON/Shapefile
  source-region loading works without a separate install.
- Linux GUI apps occasionally miss system `xcb` libs at runtime on minimal
  distros; the CI smoke-test job installs `libegl1 libgl1 libxkbcommon0
  libdbus-1-3 libxcb-cursor0` — mirror that list if the AppImage fails to
  start on a target distro.
- **Do not build on Windows with an Anaconda interpreter.** Anaconda keeps its
  own `VCRUNTIME140.dll` / `MSVCP140_1.dll` next to `python.exe`, older than
  the pair PySide6 ships. The loader mixes the two sets and Qt dies at import
  with `DLL load failed while importing QtCore: the specified procedure could
  not be found` (ERROR_PROC_NOT_FOUND). PyInstaller also copies whatever it
  found, so a build that limps along locally can still ship a broken bundle.
  Use a python.org install — which is what `actions/setup-python` gives CI.
  `build_installer.ps1` refuses a conda interpreter and drops conda off `PATH`
  for the freeze.
- The Linux leg has not been run end-to-end — the first GitHub Actions run on
  a pushed branch is the real test for it. The Windows leg is verified: built
  on Windows 10 22H2 with python.org 3.11.9 + Inno Setup 6.7.3, producing
  `dist/Normet/` at 485 MB and `normet-setup-1.0.0.exe` at 157 MB, which
  installs, launches and uninstalls cleanly. Launching proves little about
  training (the missing-xgboost bug only surfaced there), so it was also
  checked by freezing a console probe with the same collect flags that trains
  through `normet.backends.flaml_backend` with every estimator the GUI offers.
- The installer is unsigned, so SmartScreen shows "Windows protected your PC"
  on first run; users have to click through *More info → Run anyway*. Signing
  it needs a code-signing certificate and an `[Setup] SignTool` entry.
