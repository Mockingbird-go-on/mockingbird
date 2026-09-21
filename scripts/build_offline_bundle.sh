#!/usr/bin/env bash
# Build the Mockingbird offline bundle:
#   Mockingbird-OfflineBundle-<ver>.zip
#     ├── Mockingbird-Setup-<ver>.exe   (the Windows installer)
#     ├── model/                          (HuggingFace snapshot of the default
#     │                                    whisper model, ready to drop into
#     │                                    %USERPROFILE%\.mockingbird\models\)
#     └── README.txt                      (copy-paste instructions)
#
# Designed for air-gapped / corporate-proxy / slow-link installs where
# pulling ~1.6 GB from huggingface.co at first launch is impossible.
#
# Usage:
#   bash scripts/build_offline_bundle.sh                 # default: large-v3-turbo
#   bash scripts/build_offline_bundle.sh small           # smaller model
#   bash scripts/build_offline_bundle.sh --clean         # purge any cached model
#
# Requires:
#   - python3 with `huggingface_hub` and `huggingface_hub[snapshot]` installed
#   - the Windows installer already built at installer/Mockingbird-Setup-<ver>.exe
#     (run scripts/sync_and_build.sh -Installer first)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${1:-large-v3-turbo}"
CLEAN=0
if [[ "${1:-}" == "--clean" ]]; then
  CLEAN=1
  MODEL="large-v3-turbo"
fi
if [[ "${2:-}" == "--clean" ]] || [[ "${3:-}" == "--clean" ]]; then
  CLEAN=1
fi

VERSION="$(python3 -c "import sys; sys.path.insert(0, '$ROOT_DIR/src'); from mockingbird import __version__; print(__version__)")"
echo "=== Mockingbird offline bundle — version $VERSION, model $MODEL ==="

# Find the installer. The script also looks under /mnt/e (the synced WSL
# copy of E:\mockingbird) so users running this from WSL don't have to
# mirror the installer back to the repo root.
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

CACHE_DIR="$ROOT_DIR/build/offline_model_cache"
if [[ "$CLEAN" == "1" ]]; then
  echo "  --clean: removing $CACHE_DIR"
  rm -rf "$CACHE_DIR"
fi
mkdir -p "$CACHE_DIR"

# Resolve repo id the same way the engine does.
REPO_ID="$(python3 -c "
import sys; sys.path.insert(0, '$ROOT_DIR/src')
from mockingbird.stt.whisper_engine import _normalize_repo_id, _model_repo_id
size = '$MODEL'
repo = size if '/' in size else _model_repo_id(size)
print(_normalize_repo_id(repo))
")"
echo "  hf repo: $REPO_ID"

python3 - "$REPO_ID" "$CACHE_DIR" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
repo = sys.argv[1]
cache = sys.argv[2]
os.makedirs(cache, exist_ok=True)
# We want the snapshot path so we can copy it verbatim into the bundle; the
# engine's loader resolves HF cache symlinks transparently.
path = snapshot_download(repo_id=repo, cache_dir=cache)
print(f"  snapshot at: {path}")
PY

SNAPSHOT_DIR="$(ls -1d "$CACHE_DIR"/models--*/snapshots/* 2>/dev/null | head -1 || true)"
if [[ -z "$SNAPSHOT_DIR" || ! -d "$SNAPSHOT_DIR" ]]; then
  echo "ERROR: snapshot directory not found under $CACHE_DIR"
  exit 1
fi
echo "  bundle model: $SNAPSHOT_DIR ($(du -sh "$SNAPSHOT_DIR" | cut -f1))"

BUNDLE_DIR="$ROOT_DIR/build/offline_bundle_stage"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/model"
# Copy snapshot contents (no symlinks — Windows-side target dir is NTFS).
cp -RL "$SNAPSHOT_DIR/." "$BUNDLE_DIR/model/"
cp "$INSTALLER" "$BUNDLE_DIR/"

# Also include the repo's blob tree so the offline copy is a fully-functional
# HF cache: app's resolve_model_path calls snapshot_download(..., local_files_only=True),
# which looks up the snapshot path inside the cache_dir.
mkdir -p "$BUNDLE_DIR/cache"
cp -RL "$CACHE_DIR/." "$BUNDLE_DIR/cache/"

cat > "$BUNDLE_DIR/README.txt" <<EOF
Mockingbird $VERSION — Offline Bundle
====================================

Contents
--------
  Mockingbird-Setup-$VERSION.exe      Windows installer (1.9 GB)
  cache/                              HuggingFace model cache (large-v3-turbo)
  model/                              Standalone snapshot of the same files
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
3. Copy the cache/ directory over your user model dir, e.g.

     xcopy /E /I cache  "%USERPROFILE%\.mockingbird\models"

   The app will pick up the snapshot on next launch and the warm-start
   download will be skipped (no network needed).

Alternatively, point Mockingbird at the model/ folder directly via the
Settings dialog: "Каталог модели" → browse to model/.

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
