# -*- mode: python ; coding: utf-8 -*-
# Linux build spec: produces dist/mockingbird/ (onedir, GUI) and
# dist/mockingbird-cli/ — packaged into an AppImage and a .deb by
# scripts/build_linux.sh. Mirrors scripts/mockingbird.spec minus the
# Windows-specific parts:
#   - no icon (icons are irrelevant for ELF, the .desktop file carries SVG)
#   - no pyaudiowpatch (loopback uses PulseAudio monitors via sounddevice)
#   - no flat nvidia-DLL layout (the Linux loader uses RPATH/LD_LIBRARY_PATH)
#   - Qt xcb/wayland platform plugins must be collected explicitly

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


def _find_project_root(start: str) -> str:
    d = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(d, "src", "mockingbird", "main.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit(
                f"Could not locate project root (src/mockingbird/main.py). "
                f"Searched up from {os.path.abspath(start)!r}."
            )
        d = parent


_ROOT = _find_project_root(SPECPATH)
_ENTRY = os.path.join(_ROOT, "src", "mockingbird", "main.py")
_SRC = os.path.join(_ROOT, "src")
_VENDOR = os.path.join(_ROOT, "vendor")

_pyannote_modules = []
try:
    _pyannote_modules = (
        collect_submodules("pyannote")
        + collect_submodules("pyannote.audio")
        + collect_submodules("speechbrain")
    )
except Exception:
    pass

_vendor_datas = []
_vendor_hidden = []
_vendor_pathex = []
if not _pyannote_modules:
    _vendor_datas = [(os.path.join(_VENDOR, "pyannote"), "pyannote")]
    _vendor_hidden = ["pyannote"]
    _vendor_pathex = [_VENDOR]

datas = (
    collect_data_files("mockingbird")
    + [(os.path.join(_ROOT, "src", "mockingbird", "sound.mp3"), "mockingbird")]
    + [(os.path.join(_ROOT, "src", "mockingbird", "assets", "icons", "*.svg"),
        os.path.join("mockingbird", "assets", "icons"))]
    + collect_data_files("faster_whisper")
    + collect_data_files("ctranslate2")
    + collect_data_files("transformers")
    + collect_data_files("tokenizers")
    + _vendor_datas
)


def _collect_optional(name: str):
    try:
        return list(collect_dynamic_libs(name))
    except Exception:
        return []


binaries = (
    collect_dynamic_libs("ctranslate2")
    + collect_dynamic_libs("onnxruntime")
    + collect_dynamic_libs("sentencepiece")
    + _collect_optional("torch")
)

hiddenimports = (
    collect_submodules("mockingbird")
    + collect_submodules("faster_whisper")
    + collect_submodules("ctranslate2")
    + collect_submodules("onnxruntime")
    + collect_submodules("sounddevice")
    + collect_submodules("openai")
    + collect_submodules("transformers")
    + collect_submodules("tokenizers")
    + collect_submodules("sentencepiece")
    + collect_submodules("hydra")
    + collect_submodules("omegaconf")
    + _pyannote_modules
    + _vendor_hidden
    + [
        "PySide6.QtSvg",
        # Linux windowing: Qt 6.7+ splits xcb/wayland into plugins that
        # PyInstaller's PySide6 hooks may miss when building headless.
        "PySide6.QtXcbQpa",
    ]
)

a = Analysis(
    [_ENTRY],
    pathex=[_SRC] + _vendor_pathex,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["pyaudiowpatch"],
    noarchive=False,
)


pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="mockingbird",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
exe_cli = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="mockingbird-cli",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    exe_cli,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="mockingbird",
)
