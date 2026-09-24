#!/usr/bin/env bash
# Mockingbird: единый вход для сборки релизных артефактов.
#
# Один скрипт с целями — можно собрать всё вместе или перечислить нужное:
#
#   bash scripts/build.sh all                          # всё: win-gpu + win-cpu + linux + model-pack
#   bash scripts/build.sh win-gpu win-cpu              # только Windows (CUDA + CPU)
#   bash scripts/build.sh linux model-pack             # только Linux + пак модели
#   bash scripts/build.sh win-gpu --clean              # CUDA с полной пересборкой
#   bash scripts/build.sh publish --draft              # опубликовать релиз (черновик)
#
# Цели:
#   win-gpu      Windows-инсталлятор с CUDA (GPU + CPU-fallback)  [нужен Windows]
#   win-cpu      Windows-инсталлятор CPU-only                      [нужен Windows]
#   linux        Linux AppImage (+ .deb, если установлен fpm)      [запускается в WSL]
#   model-pack   пак модели whisper (offline-install)              [запускается в WSL]
#   all          = win-gpu win-cpu linux model-pack
#   publish      публикация GitHub-релиза (тонкая обёртка над release.sh)
#
# Флаги:
#   --clean      полная пересборка без кэша (PyInstaller --clean); для
#                model-pack НЕ применяется (иначе повторно тянет 1.6 ГБ)
#   --draft      для цели publish: создать релиз черновиком
#
# Внутренние скрипты (их можно звать и напрямую, если нужен тонкий контроль):
#   scripts/sync_and_build.sh   rsync WSL -> Windows-каталог + build_windows.ps1
#   scripts/build_linux.sh      PyInstaller Linux spec -> AppImage/.deb
#   scripts/build_model_pack.sh пак модели whisper
#   scripts/release.sh          публикация в GitHub Releases

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

TARGETS=()
CLEAN=0
DRAFT=0

usage() { sed -n '2,29p' "$0"; }

for a in "$@"; do
  case "$a" in
    win-gpu|win-cpu|linux|model-pack|all|publish) TARGETS+=("$a") ;;
    --clean) CLEAN=1 ;;
    --draft) DRAFT=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $a" >&2; echo; usage >&2; exit 2 ;;
  esac
done

if [ "${#TARGETS[@]}" -eq 0 ]; then
  echo "ERROR: no target given." >&2
  echo
  usage >&2
  exit 2
fi

# Разворачиваем `all` и убираем повторы, сохраняя порядок.
EXPANDED=()
add_target() {
  local t
  if [ "${#EXPANDED[@]}" -gt 0 ]; then
    for t in "${EXPANDED[@]}"; do [ "$t" = "$1" ] && return 0; done
  fi
  EXPANDED+=("$1")
}
for t in "${TARGETS[@]}"; do
  case "$t" in
    all) add_target win-gpu; add_target win-cpu; add_target linux; add_target model-pack ;;
    *)   add_target "$t" ;;
  esac
done

# Сборка Linux требует venv с установленным проектом (build_linux.sh найдёт
# .venv/bin/python сам, но стоит проверить заранее, чтобы не падать посреди
# долгого `all` после Windows-сборки).
NEEDS_LINUX=0
for t in "${EXPANDED[@]}"; do [ "$t" = "linux" ] && NEEDS_LINUX=1; done
if [ "$NEEDS_LINUX" -eq 1 ] && [ ! -x "$ROOT_DIR/.venv/bin/python" ]; then
  echo "ERROR: Linux-сборка требует venv, но '$ROOT_DIR/.venv' не найден." >&2
  echo "Активация НЕ нужна — достаточно создать один раз:" >&2
  echo "  python3 -m venv .venv" >&2
  echo "  .venv/bin/pip install -e \".[dev]\" pyinstaller" >&2
  exit 2
fi

run() {
  echo ""
  echo "======================================================================"
  echo ">>> [build.sh] $*"
  echo "======================================================================"
  "$@"
}

for t in "${EXPANDED[@]}"; do
  case "$t" in
    win-gpu)
      args=(-Installer)
      [ "$CLEAN" -eq 1 ] && args=(--clean "${args[@]}")
      run bash scripts/sync_and_build.sh "${args[@]}"
      ;;
    win-cpu)
      args=(-Cpu -Installer)
      [ "$CLEAN" -eq 1 ] && args=(--clean "${args[@]}")
      run bash scripts/sync_and_build.sh "${args[@]}"
      ;;
    linux)
      run bash scripts/build_linux.sh
      ;;
    model-pack)
      # --clean сюда НЕ транслируется: build_model_pack переиспользует уже
      # скачанную модель, --clean заставил бы тянуть 1.6 ГБ заново.
      run bash scripts/build_model_pack.sh
      ;;
    publish)
      args=()
      [ "$DRAFT" -eq 1 ] && args+=(--draft)
      run bash scripts/release.sh "${args[@]}"
      ;;
  esac
done

echo ""
echo ">>> [build.sh] готово: ${EXPANDED[*]}"
echo ">>> Артефакты:"
echo ">>>   installer/  — Windows-инсталляторы, пак модели"
echo ">>>   dist/       — Linux AppImage/.deb"
