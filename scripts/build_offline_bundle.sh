#!/usr/bin/env bash
# Build the Mockingbird offline bundle:
#   Mockingbird-OfflineBundle-<ver>.zip
#     ├── Mockingbird-Setup-<ver>.exe   (the Windows installer)
#     ├── cache/                        (HuggingFace snapshot of the default
#     │                                  whisper model, ready to drop into
#     │                                  %USERPROFILE%\.mockingbird\models\)
#     └── README.txt                    (copy-paste instructions)
#
# Designed for air-gapped / corporate-proxy / slow-link installs where
# pulling ~1.6 GB from huggingface.co at first launch is impossible.
#
# Re-uses the model already present in the HF cache (default location or
# the project's build/offline_model_cache/) — NEVER re-downloads what is
# already on disk. Pass --clean to force a fresh download from scratch.
#
# Usage:
#   bash scripts/build_offline_bundle.sh                 # default: large-v3-turbo
#   bash scripts/build_offline_bundle.sh small           # smaller model
#   bash scripts/build_offline_bundle.sh --clean         # purge any cached model
#
# Requires:
#   - python3 with `huggingface_hub` installed
#   - the Windows installer already built at installer/Mockingbird-Setup-<ver>.exe
#     (run scripts/sync_and_build.sh -Installer first)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL=""
CLEAN=0
for a in "$@"; do
  case "$a" in
    --clean) CLEAN=1 ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *) MODEL="${a}" ;;
  esac
done
MODEL="${MODEL:-large-v3-turbo}"

VERSION="$(python3 -c "import sys; sys.path.insert(0, '$ROOT_DIR/src'); from mockingbird import __version__; print(__version__)")"
echo "=== Mockingbird offline bundle — version $VERSION, model $MODEL ==="

# Locate the installer (project root or /mnt/e/mockingbird/installer).
INSTALLER_CANDIDATES=(
  "$ROOT_DIR/installer/Mockingbird-Setup-$VERSION.exe"
  "$ROOT_DIR/installer/Mockingbird-Setup-$VERSION.0.exe"
)
if [[ -d /mnt/e/mockingbird/installer ]]; then
  while IFS= read -r f; do
    INSTALLER_CANDIDATES+=("$f")
  done < <(ls -1t /mnt/e/mockingbird/installer/Mockingbird-Setup-$VERSION*.exe 2>/dev/null)
fi
INSTALLER=""
for c in "${INSTALLER_CANDIDATES[@]}"; do
  if [[ -f "$c" ]]; then
    INSTALLER="$c"
    break
  fi
done
if [[ -z "$INSTALLER" || ! -f "$INSTALLER" ]]; then
  echo "ERROR: installer not found (looked under $ROOT_DIR/installer and /mnt/e/mockingbird/installer)"
  echo "       run: bash scripts/sync_and_build.sh -Installer"
  exit 1
fi
echo "  installer: $INSTALLER ($(du -h "$INSTALLER" | cut -f1))"

# Resolve repo id the same way the engine does.
REPO_ID="$(python3 -c "
import sys; sys.path.insert(0, '$ROOT_DIR/src')
from mockingbird.stt.whisper_engine import _normalize_repo_id, _model_repo_id
size = '$MODEL'
repo = size if '/' in size else _model_repo_id(size)
print(_normalize_repo_id(repo))
")"
echo "  hf repo: $REPO_ID"

# Project-local cache (skipped on --clean; survives between runs so the
# second bundle build does NOT re-download the 1.6 GB).
CACHE_DIR="$ROOT_DIR/build/offline_model_cache"
if [[ "$CLEAN" == "1" ]]; then
  echo "  --clean: removing $CACHE_DIR"
  rm -rf "$CACHE_DIR"
fi
mkdir -p "$CACHE_DIR"

# Step 1: try to use the model already on disk (HF default cache OR
# project cache). No network call when the snapshot is present.
SNAPSHOT_DIR="$(python3 - <<PY 2>/dev/null || true
import os, sys
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("$REPO_ID", cache_dir="$CACHE_DIR", local_files_only=True)
    print(p)
except Exception:
    pass
PY
)"
if [[ -n "$SNAPSHOT_DIR" && -d "$SNAPSHOT_DIR" ]]; then
  echo "  cache hit: $SNAPSHOT_DIR (no download)"
else
  echo "  cache miss: downloading $REPO_ID (resume_download=True)…"
  SNAPSHOT_DIR="$(python3 - <<PY
import os, sys
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import snapshot_download
p = snapshot_download("$REPO_ID", cache_dir="$CACHE_DIR", resume_download=True)
print(p)
PY
)"
fi
if [[ -z "$SNAPSHOT_DIR" || ! -d "$SNAPSHOT_DIR" ]]; then
  echo "ERROR: snapshot not resolved (download failed?)"
  exit 1
fi
echo "  snapshot: $SNAPSHOT_DIR ($(du -sh "$SNAPSHOT_DIR" | cut -f1))"

# Step 2: assemble bundle. Only cache/ — the HF format the app already
# reads via its own local_files_only resolution. README documents the
# xcopy path.
BUNDLE_DIR="$ROOT_DIR/build/offline_bundle_stage"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"
# Copy the entire HF cache dir (blobs + snapshots + refs + trees). The
# snapshot resolution walks this layout via snapshot_download(...) so the
# bundled cache is a drop-in for %USERPROFILE%\.mockingbird\models.
cp -RL "$CACHE_DIR/." "$BUNDLE_DIR/cache/"
cp "$INSTALLER" "$BUNDLE_DIR/"

cat > "$BUNDLE_DIR/README.txt" <<EOF
Mockingbird $VERSION — Offline Bundle
====================================

Contents
--------
  Mockingbird-Setup-$VERSION.exe      Windows installer (1.9 GB)
  cache/                              HuggingFace model cache (large-v3-turbo)
  README.txt                          This file

When to use
-----------
First launch of Mockingbird tries to download the whisper model from
huggingface.co. If your machine is offline / behind a corporate proxy that
blocks SSL to huggingface.co, that download fails with:

  SSL: CERTIFICATE_VERIFY_FAILED  certificate verify failed:
      self-signed certificate in certificate chain

This bundle ships the model files so the install can skip the download.

Install instructions
--------------------
1. Run Mockingbird-Setup-$VERSION.exe and install Mockingbird normally.
2. Quit Mockingbird if it auto-started.
3. Copy the cache/ directory over your user model dir:

     xcopy /E /I cache  "%USERPROFILE%\.mockingbird\models"

   The app will pick up the snapshot on next launch and the warm-start
   download will be skipped (no network needed).

Verify
------
After step 3, the next Mockingbird launch should NOT show
"Downloading whisper model…" in its log file. If you see it, the cache
copy landed in the wrong place (check %USERPROFILE%\.mockingbird\models).

Support
-------
Logs: %USERPROFILE%\.mockingbird\logs\mockingbird.log
Issues: https://github.com/Mockingbird-go-on/mockingbird/issues
EOF

OUT="$ROOT_DIR/installer/Mockingbird-OfflineBundle-$VERSION.zip"
rm -f "$OUT"
echo "  zip: $OUT"
( cd "$BUNDLE_DIR" && zip -qr "$OUT" . )
echo "=== done: $(du -h "$OUT" | cut -f1) ==="
echo ""
echo "Bundle location: $OUT"
