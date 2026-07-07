#!/usr/bin/env bash
# Build normet.app with PyInstaller and package it into a compressed DMG.
#
# Usage:  packaging/macos/build_dmg.sh [python]
# The optional argument selects the interpreter whose environment holds
# normet + PySide6 + pyinstaller (default: "python3").

set -euo pipefail

PY="${1:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DIST="$REPO/dist/macos"
APP_NAME="Normet"
VERSION="$("$PY" -c 'import importlib.metadata as m; print(m.version("normet"))')"

echo "==> Building $APP_NAME $VERSION with $("$PY" --version 2>&1)"
cd "$REPO"

"$PY" -m PyInstaller "$REPO/packaging/launcher.py" \
    --name "$APP_NAME" \
    --windowed \
    --noconfirm \
    --clean \
    --distpath "$DIST" \
    --workpath "$REPO/build/pyinstaller" \
    --specpath "$REPO/build/pyinstaller" \
    --icon "$REPO/packaging/assets/normet.icns" \
    --osx-bundle-identifier "org.apai-sys.normet" \
    --collect-submodules normet \
    --collect-all flaml \
    --collect-all lightgbm \
    --collect-binaries xgboost \
    --collect-data xgboost \
    --collect-all shapely \
    --hidden-import shapefile \
    --exclude-module tkinter \
    --exclude-module IPython \
    --exclude-module pytest

APP="$DIST/$APP_NAME.app"
[ -d "$APP" ] || { echo "ERROR: $APP not produced"; exit 1; }

echo "==> Staging DMG contents"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

DMG="$DIST/${APP_NAME}-${VERSION}-macos-$(uname -m).dmg"
rm -f "$DMG"

echo "==> Creating $DMG"
hdiutil create -volname "$APP_NAME $VERSION" \
    -srcfolder "$STAGE" \
    -ov -format UDZO "$DMG"

echo "==> Done: $DMG"
du -sh "$DMG"
