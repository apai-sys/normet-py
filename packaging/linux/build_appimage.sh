#!/usr/bin/env bash
# Package the PyInstaller onedir build (dist/normet) into a Linux AppImage.
# Run from the repository root, after:
#   pyinstaller --noconfirm packaging/normet_gui.spec
set -euo pipefail

APP=Normet
VERSION="$(sed -nE 's/^version\s*=\s*"([^"]+)".*/\1/p' pyproject.toml | head -1)"
DIST="dist/${APP}"
APPDIR="dist/${APP}.AppDir"

[ -d "$DIST" ] || { echo "ERROR: $DIST not found — run pyinstaller first"; exit 1; }

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -r "$DIST"/* "$APPDIR/usr/bin/"

# Icon
cp packaging/assets/normet.png "$APPDIR/normet.png"
cp packaging/assets/normet.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/normet.png"

# Desktop entry
cat > "$APPDIR/normet.desktop" <<'EOF'
[Desktop Entry]
Name=Normet
Comment=Weather normalisation & counterfactual modelling for air quality
Exec=Normet
Icon=normet
Type=Application
Categories=Science;Education;
Terminal=false
EOF
cp "$APPDIR/normet.desktop" "$APPDIR/usr/share/applications/normet.desktop"

# AppRun launcher
cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="$HERE/usr/bin:$PATH"
exec "$HERE/usr/bin/Normet" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# Fetch appimagetool and build
TOOL=/tmp/appimagetool
if [ ! -x "$TOOL" ]; then
  wget -qO "$TOOL" \
    "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
  chmod +x "$TOOL"
fi

ARCH=x86_64 "$TOOL" --appimage-extract-and-run "$APPDIR" "dist/Normet-${VERSION}-x86_64.AppImage"
echo "✓ dist/Normet-${VERSION}-x86_64.AppImage"
