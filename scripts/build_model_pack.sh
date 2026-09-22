#!/usr/bin/env bash
# Build the Mockingbird whisper MODEL PACK:
#   Mockingbird-whisper-<model>-model.zip
#     ├── cache/       (HuggingFace snapshot of the whisper model, ready to
#     │                 drop into %USERPROFILE%\.mockingbird\models\)
#     └── README.txt   (copy-paste instructions)
#
# The model pack is DELIBERATELY separate from the installer:
#   - GitHub Releases cap each asset at 2 GiB; installer (~1.5 GB) + model
#     (~1.6 GB) in one zip (~3.0 GB) cannot be uploaded as a single file.
#   - the model is platform/variant independent, so the SAME pack works with
#     both the CUDA and the CPU installer.
#
# How it is consumed: the installer (installer.iss) has a [Files] entry that
# copies a `cache/` directory sitting NEXT TO setup.exe into
# %USERPROFILE%\.mockingbird\models. So the user just extracts this pack and
# the installer next to each other and runs setup.
#
# Designed for air-gapped / corporate-proxy / slow-link installs where pulling
# ~1.6 GB from huggingface.co at first launch is impossible.
#
# Re-uses the model already present in the HF cache (default location or the
# project's build/offline_model_cache/) — NEVER re-downloads what is already on
# disk. Pass --clean to force a fresh download from scratch.
#
# Usage:
#   bash scripts/build_model_pack.sh                 # default: large-v3-turbo
#   bash scripts/build_model_pack.sh small           # smaller model
#   bash scripts/build_model_pack.sh --clean         # purge any cached model
#
# Requires: python3 with `huggingface_hub` installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL=""
CLEAN=0
for a in "$@"; do
  case "$a" in
    --clean) CLEAN=1 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *) MODEL="${a}" ;;
  esac
done
MODEL="${MODEL:-large-v3-turbo}"

VERSION="$(python3 -c "import sys; sys.path.insert(0, '$ROOT_DIR/src'); from mockingbird import __version__; print(__version__)")"
echo "=== Mockingbird whisper model pack — model $MODEL (app v$VERSION) ==="

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
# second pack build does NOT re-download the 1.6 GB).
CACHE_DIR="$ROOT_DIR/build/offline_model_cache"
if [[ "$CLEAN" == "1" ]]; then
  echo "  --clean: removing $CACHE_DIR"
  rm -rf "$CACHE_DIR"
fi
mkdir -p "$CACHE_DIR"

# Step 1: try to use the model already on disk (HF default cache OR project
# cache). No network call when the snapshot is present.
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

# Step 2: assemble the pack. Only the SELECTED repo and only the parts the
# app's local_files_only resolution actually reads:
#   refs/      (REQUIRED: maps branch/tag -> commit hash)
#   snapshots/ (REQUIRED: the model files; symlinks dereferenced)
#   trees/     (optional metadata, tiny — kept)
# blobs/ is NOT shipped: in the HF cache, snapshots/* are symlinks into
# blobs/*; copying with symlink-dereference already yields real files in
# snapshots/, so shipping blobs/ too would double every weight (~1.6 GB).
# Verified against the shipped huggingface_hub (0.36.2): snapshot_download
# (local_files_only=True) resolves and WhisperModel loads with refs/ +
# snapshots/ only — no blobs/. Other cached repos (e.g. a leftover tiny
# model) and .locks/ are intentionally excluded.
PACK_DIR="$ROOT_DIR/build/model_pack_stage"
rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR"
# Derive the repo cache dir from the resolved snapshot path:
#   .../models--<slug>/snapshots/<hash>  ->  .../models--<slug>
REPO_CACHE="$(cd "$SNAPSHOT_DIR/../.." && pwd)"
echo "  repo cache: $REPO_CACHE"
if [[ ! -d "$REPO_CACHE/snapshots" || ! -d "$REPO_CACHE/refs" ]]; then
  echo "ERROR: $REPO_CACHE is missing snapshots/ or refs/ (cannot assemble pack)"
  exit 1
fi
mkdir -p "$PACK_DIR/cache"
# Keep the HF layout: cache_dir/models--<slug>/{refs,snapshots,trees}. The
# installer copies cache/* into %USERPROFILE%\.mockingbird\models, which the
# engine uses as cache_dir, so models--<slug> MUST sit directly under cache/.
REPO_LEAF="$(basename "$REPO_CACHE")"
if [[ "$REPO_LEAF" != models--* ]]; then
  echo "ERROR: unexpected repo cache dir '$REPO_LEAF' (expected models--<slug>)"
  exit 1
fi
DEST_REPO="$PACK_DIR/cache/$REPO_LEAF"
mkdir -p "$DEST_REPO"
# -L dereferences the snapshot symlinks into real files (no blobs/ needed).
cp -RL "$REPO_CACHE/refs" "$DEST_REPO/"
cp -RL "$REPO_CACHE/snapshots" "$DEST_REPO/"
if [[ -d "$REPO_CACHE/trees" ]]; then
  cp -RL "$REPO_CACHE/trees" "$DEST_REPO/"
fi

# Step 2b: VERIFY the assembled cache actually resolves the way the app will
# resolve it (local_files_only). Fail the build if it does not, so a broken
# pack never reaches the user. Mirrors app_dir()/models lookup.
echo "  verifying assembled cache resolves offline…"
HF_HUB_DISABLE_PROGRESS_BARS=1 python3 - <<PY
import os, sys
sys.path.insert(0, "$ROOT_DIR/src")
from huggingface_hub import snapshot_download
cache = os.path.join("$PACK_DIR", "cache")
try:
    p = snapshot_download("$REPO_ID", cache_dir=cache, local_files_only=True)
except Exception as exc:
    print("  ERROR: assembled cache does not resolve: %r" % (exc,))
    raise SystemExit(1)
required = ("config.json", "model.bin")
missing = [f for f in required if not os.path.isfile(os.path.join(p, f))]
if missing:
    print("  ERROR: resolved snapshot is missing: %s" % ", ".join(missing))
    raise SystemExit(1)
print("  verified: %s" % p)
PY

cat > "$PACK_DIR/README.txt" <<EOF
Mockingbird $VERSION — whisper model pack ($MODEL)
==================================================

Contents
--------
  cache/       HuggingFace model cache for $MODEL
  README.txt   This file

When to use
-----------
First launch of Mockingbird downloads the whisper model from huggingface.co.
If your machine is offline / behind a corporate proxy that blocks SSL to
huggingface.co, that download fails with:

  SSL: CERTIFICATE_VERIFY_FAILED  certificate verify failed:
      self-signed certificate in certificate chain

This pack ships the model files so the install can skip the download. It works
with BOTH the CUDA and the CPU installer (the model is platform-independent).

Install instructions
--------------------
1. EXTRACT this pack.
2. Put its cache/ directory NEXT TO the Mockingbird installer .exe, so both
   sit in the SAME folder:
       some-folder\
         Mockingbird-$VERSION-windows-x64-cuda-setup.exe
         cache\
3. Run the installer. It detects cache/ next to it and copies the model into
   %USERPROFILE%\.mockingbird\models automatically — no manual step needed.
4. Launch Mockingbird. The model is already on disk, so the warm-start
   download is skipped (no network needed).

Manual fallback (only if you moved the installer away from cache/)
------------------------------------------------------------------
If the installer could not find cache/, copy it yourself:

     xcopy /E /I cache  "%USERPROFILE%\.mockingbird\models"

Verify
------
After install, the first Mockingbird launch should NOT show
"Downloading whisper model…" in its log file. If you see it, the cache
copy landed in the wrong place (check %USERPROFILE%\.mockingbird\models).

Support
-------
Logs: %USERPROFILE%\\.mockingbird\\logs\\mockingbird.log
Community chat (Telegram): https://t.me/MOCKINGBird_release
Issues: https://github.com/Mockingbird-go-on/mockingbird/issues
EOF

mkdir -p "$ROOT_DIR/installer"
OUT="$ROOT_DIR/installer/Mockingbird-whisper-$MODEL-model.zip"
rm -f "$OUT"
echo "  zip: $OUT"
( cd "$PACK_DIR" && zip -qr "$OUT" . )
echo "=== done: $(du -h "$OUT" | cut -f1) ==="
echo ""
echo "Model pack location: $OUT"
