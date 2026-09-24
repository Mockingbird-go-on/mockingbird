# -*- mode: python ; coding: utf-8 -*-
# Linux build spec: produces dist/mockingbird/ — packaged into an AppImage
# and a .deb by scripts/build_linux.sh. Mirrors scripts/mockingbird.spec
# minus the Windows-specific parts:
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


def _sounddevice_libs():
    """Locate the system libportaudio for the Linux bundle.

    sounddevice dlopens libportaudio at import time; without the .so inside
    the bundle the frozen app dies at startup («OSError: PortAudio library
    not found»). Returns [] (with a loud warning) when the system package
    is missing.
    """
    import ctypes.util
    import subprocess

    found = ctypes.util.find_library("portaudio")
    if not found:
        print(
            "WARNING: libportaudio not found on this system — the AppImage "
            "will crash at startup (fix: sudo apt install libportaudio2)"
        )
        return []
    resolved = []
    try:
        out = subprocess.run(
            ["/sbin/ldconfig", "-p"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if found in line:
                path = line.split("->")[-1].strip()
                if path:
                    resolved.append((path, "."))
                break
    except Exception:  # noqa: BLE001
        pass
    if not resolved:
        print(f"WARNING: portaudio soname {found!r} not resolvable via ldconfig")
    return resolved


_ROOT = _find_project_root(SPECPATH)
_ENTRY = os.path.join(_ROOT, "src", "mockingbird", "main.py")
_SRC = os.path.join(_ROOT, "src")

# CPU-only build switch. Set by scripts/build_linux.sh --cpu. On Linux the
# CUDA stack is linked from the system (RPATH/LD_LIBRARY_PATH), so there is no
# nested nvidia tree to drop; the flag mainly keeps PyInstaller from following
# any nvidia-* packages present in the build env.
_CPU_ONLY = os.environ.get("MOCKINGBIRD_CPU") == "1"

datas = (
    collect_data_files("mockingbird")
    + [(os.path.join(_ROOT, "src", "mockingbird", "sound.mp3"), "mockingbird")]
    + [(os.path.join(_ROOT, "src", "mockingbird", "assets", "icons", "*.svg"),
        os.path.join("mockingbird", "assets", "icons"))]
    + collect_data_files("faster_whisper")
    + collect_data_files("ctranslate2")
    + collect_data_files("tokenizers")
)


binaries = (
    collect_dynamic_libs("ctranslate2")
    + collect_dynamic_libs("onnxruntime")
    + collect_dynamic_libs("sentencepiece")
    + _sounddevice_libs()
)

hiddenimports = (
    collect_submodules("mockingbird")
    + collect_submodules("faster_whisper")
    + collect_submodules("ctranslate2")
    + collect_submodules("onnxruntime")
    + collect_submodules("sounddevice")
    + collect_submodules("openai")
    + collect_submodules("tokenizers")
    + collect_submodules("sentencepiece")
    + [
        "PySide6.QtSvg",
        "PySide6.QtMultimedia",
    ]
)

_excludes = [
    "pyaudiowpatch",
    # GigaAM leftovers — nothing imports them; keep the bundle lean.
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "speechbrain",
    "pyannote",
    "hydra",
    "omegaconf",
    "matplotlib",
    "tkinter",
    "IPython",
]
if _CPU_ONLY:
    _excludes += ["nvidia"]

a = Analysis(
    [_ENTRY],
    pathex=[_SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=_excludes,
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
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="mockingbird",
)
