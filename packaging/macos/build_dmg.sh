#!/usr/bin/env bash
# Build Normet.app from packaging/normet_gui.spec and package it into a compressed DMG.
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
# The data downloaders are imported by name at run time (normet.utils.require),
# so they are only bundled if the build env has them: fail here, not in the app.
"$PY" -c 'import requests, cdsapi, rdata' 2>/dev/null || {
    echo "ERROR: install the data extra first:  $PY -m pip install -e \".[data]\""; exit 1; }
VERSION="$("$PY" -c 'import importlib.metadata as m; print(m.version("normet"))')"

echo "==> Building $APP_NAME $VERSION with $("$PY" --version 2>&1)"
cd "$REPO"

# The shared spec, so the app carries what the other platforms do (the manuals) and
# what only a Mac bundle needs (its Info.plist, with the languages it speaks).
"$PY" -m PyInstaller "$REPO/packaging/normet_gui.spec" \
    --noconfirm \
    --clean \
    --distpath "$DIST" \
    --workpath "$REPO/build/pyinstaller"

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
