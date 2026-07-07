# Building normet GUI installers

The GUI (`normet.gui`) is PySide6 + matplotlib, so the frozen app is
cross-platform. PyInstaller, however, **cannot cross-compile** — each OS's
installer must be built on that OS. The supported path is GitHub Actions
(`.github/workflows/build-gui.yml`), one spec, three OSes.

## Local build (macOS only)

```bash
packaging/macos/build_dmg.sh /path/to/python   # -> dist/macos/normet.app + .dmg
```

## Local build (any OS, onedir — no installer wrapper)

```bash
pip install -e ".[gui,flaml,lgb,geo]" pyinstaller
pyinstaller --noconfirm packaging/normet_gui.spec   # -> dist/normet/
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
  Windows/Linux ship the COLLECT onedir `dist/normet/`). Version is read from
  `pyproject.toml` at build time — no separate version to keep in sync.
- `assets/normet.icns` / `normet.ico` / `normet.png` — per-OS icons, generated
  by `assets/make_icon.py` (Pillow). Re-run it if the design should change;
  the outputs are committed so CI doesn't need to regenerate them.
- `packaging/windows/installer.iss` — Inno Setup script.
- `packaging/linux/build_appimage.sh` — AppDir + appimagetool.
- `launcher.py` — the frozen entry point (`multiprocessing.freeze_support()`
  then `normet.gui.main()`), shared by every OS's build.

## Notes / caveats

- flaml + lightgbm (AutoML backends) are bundled by default — this makes the
  installer sizeable (~300 MB) but keeps "Train model" working out of the box.
- shapely + pyshp are bundled too, so Transport Studio's GeoJSON/Shapefile
  source-region loading works without a separate install.
- Linux GUI apps occasionally miss system `xcb` libs at runtime on minimal
  distros; the CI smoke-test job installs `libegl1 libgl1 libxkbcommon0
  libdbus-1-3 libxcb-cursor0` — mirror that list if the AppImage fails to
  start on a target distro.
- The Windows/Linux legs have not been run end-to-end (only the macOS spec
  path is verified locally) — the first GitHub Actions run on a pushed branch
  is the real test for those two.
