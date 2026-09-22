#!/usr/bin/env bash
# Publish a Mockingbird GitHub release from locally built artifacts.
#
# This script does NOT build anything — run the build scripts first:
#   bash scripts/sync_and_build.sh --clean -Installer         # CUDA installer
#   bash scripts/sync_and_build.sh --clean -Cpu -Installer    # CPU installer
#   bash scripts/build_model_pack.sh                          # whisper model pack
#   bash scripts/build_linux.sh                               # AppImage + .deb
#
# It then:
#   1. collects the expected assets (warns about anything missing);
#   2. ensures the PERMANENT "models" release holds the whisper model pack
#      (uploaded once; app releases just link to it);
#   3. renders release notes from scripts/release_notes.md.in;
#   4. creates the app release v<version> with the installers + Linux builds
#      + SHA256SUMS.txt.
#
# Why a separate "models" release: GitHub caps each asset at 2 GiB. The
# offline bundle (installer ~1.5 GB + model ~1.6 GB) cannot be a single file,
# and the model is version-independent, so it lives in its own permanent
# release and is referenced from every app release.
#
# Usage:
#   bash scripts/release.sh                 # version from mockingbird.__version__
#   bash scripts/release.sh 1.0.0.2         # explicit version
#   bash scripts/release.sh --draft         # create the app release as a draft
#   bash scripts/release.sh --force-models  # re-upload the model pack
#   bash scripts/release.sh --clobber       # replace an existing app release
#
# Requires: gh (authenticated), zip, sha256sum.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

VERSION=""
DRAFT=0
FORCE_MODELS=0
CLOBBER=0
for a in "$@"; do
  case "$a" in
    --draft) DRAFT=1 ;;
    --force-models) FORCE_MODELS=1 ;;
    --clobber) CLOBBER=1 ;;
    -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) VERSION="$a" ;;
  esac
done
VERSION="${VERSION:-$(python3 -c "import sys; sys.path.insert(0, 'src'); from mockingbird import __version__; print(__version__)")}"

MODEL="large-v3-turbo"
TAG="v$VERSION"
MODELS_TAG="models"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh (GitHub CLI) not found. Install it and run 'gh auth login'." >&2
  exit 1
fi
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)"
if [[ -z "$REPO" ]]; then
  echo "ERROR: could not determine the GitHub repo (run from a clone with a remote)." >&2
  exit 1
fi
MODELS_URL="https://github.com/$REPO/releases/download/$MODELS_TAG/Mockingbird-whisper-$MODEL-model.zip"

echo "=== Mockingbird release $TAG (repo $REPO) ==="
echo "  model pack url: $MODELS_URL"

# --- 1. collect assets -------------------------------------------------------
CUDA_EXE="installer/Mockingbird-$VERSION-windows-x64-cuda-setup.exe"
CPU_EXE="installer/Mockingbird-$VERSION-windows-x64-cpu-setup.exe"
MODEL_ZIP="installer/Mockingbird-whisper-$MODEL-model.zip"
APPIMAGE="dist/Mockingbird-$VERSION-linux-x86_64.AppImage"
# fpm emits mockingbird_<ver>_amd64.deb; normalise to the canonical asset name.
DEB_SRC="dist/mockingbird_${VERSION}_amd64.deb"
DEB="build/Mockingbird-$VERSION-linux-amd64.deb"
mkdir -p build
if [[ -f "$DEB_SRC" ]]; then
  cp -f "$DEB_SRC" "$DEB"
fi

APP_ASSETS=()
MISSING=0
for f in "$CUDA_EXE" "$CPU_EXE" "$APPIMAGE" "$DEB"; do
  if [[ -f "$f" ]]; then
    APP_ASSETS+=("$f")
    echo "  ok:   $f ($(du -h "$f" | cut -f1))"
  else
    echo "  MISS: $f"
    MISSING=1
  fi
done
if [[ ! -f "$MODEL_ZIP" ]]; then
  echo "  MISS: $MODEL_ZIP (offline model pack)"
  MISSING=1
else
  echo "  ok:   $MODEL_ZIP ($(du -h "$MODEL_ZIP" | cut -f1))"
fi
if [[ "${#APP_ASSETS[@]}" -eq 0 ]]; then
  echo "ERROR: no app assets found — build them first (see header)." >&2
  exit 1
fi
if [[ "$MISSING" -eq 1 ]]; then
  echo ""
  echo "WARNING: some assets are missing (see MISS above)."
  echo "         Release will proceed with what is present."
  echo "         Press Ctrl-C within 5s to abort."
  sleep 5
fi

# --- 2. permanent "models" release -------------------------------------------
if [[ -f "$MODEL_ZIP" ]]; then
  if gh release view "$MODELS_TAG" >/dev/null 2>&1; then
    HAS_ASSET="$(gh release view "$MODELS_TAG" --json assets \
      -q ".assets[].name" 2>/dev/null | grep -Fx "$(basename "$MODEL_ZIP")" || true)"
    if [[ -n "$HAS_ASSET" && "$FORCE_MODELS" -eq 0 ]]; then
      echo "  models release '$MODELS_TAG' already has the pack (use --force-models to replace)"
    else
      echo "  uploading model pack to '$MODELS_TAG' release…"
      if [[ -z "$HAS_ASSET" ]]; then
        gh release upload "$MODELS_TAG" "$MODEL_ZIP"
      else
        gh release upload "$MODELS_TAG" "$MODEL_ZIP" --clobber
      fi
    fi
  else
    echo "  creating permanent '$MODELS_TAG' release with the model pack…"
    gh release create "$MODELS_TAG" "$MODEL_ZIP" \
      --title "Whisper models (permanent)" \
      --notes "Platform-independent whisper model packs. Referenced by every app release for offline installs. Do not delete." \
      --latest=false
  fi
else
  echo "  (no model pack — skipping '$MODELS_TAG' release)"
fi

# --- 3. render release notes -------------------------------------------------
NOTES="build/release_notes-$VERSION.md"
mkdir -p build
python3 - "$VERSION" "$MODELS_URL" "$NOTES" <<'PY'
import sys
version, models_url, out = sys.argv[1], sys.argv[2], sys.argv[3]
tpl = open("scripts/release_notes.md.in", encoding="utf-8").read()
tpl = tpl.replace("__VERSION__", version).replace("__MODELS_URL__", models_url)
open(out, "w", encoding="utf-8").write(tpl)
print("  notes: %s" % out)
PY

# --- 4. app release ----------------------------------------------------------
# SHA256SUMS is generated over the app assets only (installers + linux).
SUMS="build/SHA256SUMS.txt"
: > "$SUMS"
for f in "${APP_ASSETS[@]}"; do
  ( cd "$(dirname "$f")" && sha256sum "$(basename "$f")" ) >> "$SUMS"
done
echo "  checksums: $SUMS"
cat "$SUMS" | sed 's/^/    /'

CREATE_ARGS=()
if [[ "$DRAFT" -eq 1 ]]; then
  CREATE_ARGS+=(--draft)          # a draft cannot be "latest" yet
else
  CREATE_ARGS+=(--latest)
fi
[[ "$CLOBBER" -eq 1 ]] && CREATE_ARGS+=(--clobber)

if gh release view "$TAG" >/dev/null 2>&1 && [[ "$CLOBBER" -eq 0 ]]; then
  echo "ERROR: release '$TAG' already exists. Re-run with --clobber to replace." >&2
  exit 1
fi

echo "  creating release $TAG…"
gh release create "$TAG" "${APP_ASSETS[@]}" "$SUMS" \
  --title "Mockingbird $VERSION" \
  --notes-file "$NOTES" \
  "${CREATE_ARGS[@]}"

echo ""
echo "=== done: $(gh release view "$TAG" --json url -q .url) ==="
