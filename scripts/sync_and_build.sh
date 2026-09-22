#!/usr/bin/env bash
# Sync the WSL project to a Windows-native path and build the .exe there.
#
# Why: PyInstaller must run from a Windows-native path (not \\wsl.localhost\...).
# This script rsyncs the project to E:\mockingbird and launches the Windows
# build from WSL, streaming PowerShell output back to the terminal.
#
# Usage (from anywhere in WSL):
#   bash scripts/sync_and_build.sh            # default build
#   bash scripts/sync_and_build.sh --no-build # sync only, skip the build
#   bash scripts/sync_and_build.sh --clean    # full rebuild (drop PyInstaller cache)
#   bash scripts/sync_and_build.sh --cpu      # CPU-only build (no CUDA stack)
#
# Requires: rsync, /mnt/e mounted, Windows Python 3.11+ on the target machine.

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="/mnt/e/mockingbird"
WIN_DST="E:\\mockingbird"

if [ ! -d "$SRC" ]; then
    echo "ERROR: source not found: $SRC" >&2
    exit 1
fi
if [ ! -d /mnt/e ]; then
    echo "ERROR: /mnt/e is not mounted. Make sure drive E: is accessible from WSL." >&2
    exit 1
fi

# Forward our args to the PowerShell build script, unless --no-build is given.
BUILD_ARGS=()
DO_BUILD=1
for arg in "$@"; do
    case "$arg" in
        --no-build) DO_BUILD=0 ;;
        # translate to the PowerShell switch spelling
        --clean) BUILD_ARGS+=("-Clean") ;;
        --cpu) BUILD_ARGS+=("-Cpu") ;;
        *) BUILD_ARGS+=("$arg") ;;
    esac
done

echo ">>> Syncing $SRC -> $DST"
mkdir -p "$DST"
rsync -a --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='installer' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='onnxenv' \
    --exclude='docs' \
    --exclude='*:Zone.Identifier' \
    "$SRC/" "$DST/"
echo ">>> Sync complete."

if [ "$DO_BUILD" -ne 1 ]; then
    echo ">>> --no-build: skipping PyInstaller run."
    exit 0
fi

if ! command -v powershell.exe >/dev/null 2>&1; then
    echo "ERROR: powershell.exe not found on PATH." >&2
    exit 1
fi

PS_ARGS=(-ExecutionPolicy Bypass -File "${WIN_DST}\\scripts\\build_windows.ps1")
if [ "${#BUILD_ARGS[@]}" -gt 0 ]; then
    PS_ARGS+=("${BUILD_ARGS[@]}")
fi

echo ">>> Running: powershell.exe ${PS_ARGS[*]}"
powershell.exe "${PS_ARGS[@]}"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo ">>> Build failed with exit code $RC." >&2
    exit "$RC"
fi
echo ">>> Build complete. Output: ${DST}/dist/mockingbird/"

# Pull the freshly built installer(s) back into the WSL project's installer/
# so scripts/release.sh (which runs entirely in WSL) can find them. The
# Windows build writes them to E:\mockingbird\installer (RootDir-anchored in
# installer.iss); without this copy they never leave the Windows side.
if [ -d "$DST/installer" ]; then
    shopt -s nullglob
    pulled=("$DST"/installer/Mockingbird-*-windows-x64-*-setup.exe)
    shopt -u nullglob
    if [ "${#pulled[@]}" -gt 0 ]; then
        mkdir -p "$SRC/installer"
        cp -f "${pulled[@]}" "$SRC/installer/"
        echo ">>> Pulled ${#pulled[@]} installer(s) into $SRC/installer/"
    fi
fi
