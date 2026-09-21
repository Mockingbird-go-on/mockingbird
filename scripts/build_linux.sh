#!/usr/bin/env bash
# Build Mockingbird for Linux: PyInstaller onedir -> AppImage + .deb.
#
# Usage:  bash scripts/build_linux.sh [--cpu] [--no-deb] [--no-appimage]
#
# Requirements on the build machine:
#   - Python 3.11+ venv with the project installed (pip install -e .)
#   - patchelf (AppImage), appimagetool (downloaded on demand to build/)
#   - fpm + dpkg-deb for the .deb (skipped gracefully if absent)
#
# The script is idempotent: every stage can be re-run after a failure.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CPU_ONLY=0
MAKE_DEB=1
MAKE_APPIMAGE=1
for arg in "$@"; do
    case "$arg" in
        --cpu) CPU_ONLY=1 ;;
        --no-deb) MAKE_DEB=0 ;;
        --no-appimage) MAKE_APPIMAGE=0 ;;
        *) echo "unknown flag: $arg"; exit 2 ;;
    esac
done

VERSION="$(PYTHONPATH=src python -c 'from mockingbird import __version__; print(__version__)')"
echo "== Mockingbird Linux build v$VERSION =="

# --- 1. PyInstaller ----------------------------------------------------------
python -m PyInstaller --clean --noconfirm scripts/mockingbird_linux.spec
test -x dist/mockingbird/mockingbird || { echo "PyInstaller produced no executable"; exit 1; }

# --- 2. AppImage -------------------------------------------------------------
if [[ $MAKE_APPIMAGE -eq 1 ]]; then
    APPDIR=build/AppDir
    rm -rf "$APPDIR"
    mkdir -p "$APPDIR/usr" "$APPDIR/usr/share/metainfo" "$APPDIR/usr/share/icons/hicolor/scalable/apps"
    cp -r dist/mockingbird/. "$APPDIR/usr/"

    cat > "$APPDIR/mockingbird.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Mockingbird
GenericName=Live Speech-to-Text
Comment=Low-latency live speech-to-text for IT interviews
Exec=mockingbird
Icon=mockingbird
Terminal=false
Categories=Utility;AudioVideo;
StartupWMClass=mockingbird
DESKTOP

    ICON_SRC="src/mockingbird/assets/icons"
    # Prefer a dedicated app icon SVG; fall back to any bundled SVG.
    for cand in mockingbird.svg app.svg logo.svg; do
        if [[ -f "$ICON_SRC/$cand" ]]; then
            cp "$ICON_SRC/$cand" "$APPDIR/usr/share/icons/hicolor/scalable/apps/mockingbird.svg"
            break
        fi
    done

    cat > "$APPDIR/AppRun" <<'APPRUN'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/mockingbird" "$@"
APPRUN
    chmod +x "$APPDIR/AppRun"

    APPIMAGETOOL=build/appimagetool
    if [[ ! -x "$APPIMAGETOOL" ]]; then
        mkdir -p build
        echo "== downloading appimagetool =="
        ARCH="$(uname -m)"
        [[ "$ARCH" == "x86_64" ]] || { echo "appimagetool: unsupported arch $ARCH"; exit 1; }
        curl -fL -o "$APPIMAGETOOL" \
            "https://github.com/AppImage/appimagetool/releases/latest/download/appimagetool-x86_64.AppImage"
        chmod +x "$APPIMAGETOOL"
    fi

    mkdir -p dist
    ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "dist/Mockingbird-$VERSION-x86_64.AppImage"
    echo "== AppImage: dist/Mockingbird-$VERSION-x86_64.AppImage =="
fi

# --- 3. .deb (optional) -------------------------------------------------------
if [[ $MAKE_DEB -eq 1 ]]; then
    if ! command -v fpm >/dev/null 2>&1; then
        echo "== fpm not found, skipping .deb (install with: gem install fpm) =="
    else
        DEBDIR=build/deb
        rm -rf "$DEBDIR"
        mkdir -p "$DEBDIR/usr/bin" "$DEBDIR/usr/lib/mockingbird" \
                 "$DEBDIR/usr/share/applications" \
                 "$DEBDIR/usr/share/icons/hicolor/scalable/apps" "$DEBDIR/DEBIAN"
        cp -r dist/mockingbird/. "$DEBDIR/usr/lib/mockingbird/"
        ln -s /usr/lib/mockingbird/mockingbird "$DEBDIR/usr/bin/mockingbird"
        ln -s /usr/lib/mockingbird/mockingbird-cli "$DEBDIR/usr/bin/mockingbird-cli"

        cat > "$DEBDIR/usr/share/applications/mockingbird.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Mockingbird
Comment=Low-latency live speech-to-text for IT interviews
Exec=/usr/lib/mockingbird/mockingbird
Icon=mockingbird
Terminal=false
Categories=Utility;AudioVideo;
DESKTOP
        if [[ -f src/mockingbird/assets/icons/mockingbird.svg ]]; then
            cp src/mockingbird/assets/icons/mockingbird.svg "$DEBDIR/usr/share/icons/hicolor/scalable/apps/mockingbird.svg"
        fi

        fpm -s dir -t deb -n mockingbird -v "$VERSION" \
            --description "Low-latency live speech-to-text desktop app for IT interviews" \
            --maintainer "Mockingbird <dev@mockingbird.app>" \
            --url "https://github.com/Mockingbird-go-on/mockingbird" \
            --depends "libpulse0" \
            -C "$DEBDIR" usr
        mv -f mockingbird_"$VERSION"_*.deb dist/ 2>/dev/null || true
        echo "== .deb: dist/ =="
    fi
fi

echo "== Linux build complete =="
