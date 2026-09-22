# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# Find the project root by walking upward from the spec file until we hit the
# directory that contains src/mockingbird/main.py. This works no matter where
# the project folder was copied to on the build machine.
def _find_project_root(start: str) -> str:
    d = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(d, "src", "mockingbird", "main.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit(
                f"Could not locate project root. Make sure src/ and scripts/ "
                f"live under the same folder (e.g. C:\\projects\\mockingbird). "
                f"Searched up from {os.path.abspath(start)!r}."
            )
        d = parent


_ROOT = _find_project_root(SPECPATH)
_ENTRY = os.path.join(_ROOT, "src", "mockingbird", "main.py")
_SRC = os.path.join(_ROOT, "src")
_ICON = os.path.join(SPECPATH, "logo_mockingbird.ico")

# CPU-only build switch. Set by scripts/build_windows.ps1 -Cpu (env var, since
# PyInstaller does not forward custom CLI args to the spec). Drops the whole
# nvidia CUDA stack (~1.9 GB) from the bundle; ctranslate2 then runs on CPU.
_CPU_ONLY = os.environ.get("MOCKINGBIRD_CPU") == "1"

datas = (
    collect_data_files("mockingbird")
    + [(os.path.join(SPECPATH, "logo_mockingbird.ico"), "mockingbird")]
    + [(os.path.join(_ROOT, "src", "mockingbird", "sound.mp3"), "mockingbird")]
    + [(os.path.join(_ROOT, "src", "mockingbird", "assets", "icons", "*.svg"),
        os.path.join("mockingbird", "assets", "icons"))]
    + [(os.path.join(_ROOT, "src", "mockingbird", "assets", "models", "*.onnx"),
        os.path.join("mockingbird", "assets", "models"))]
    + collect_data_files("faster_whisper")
    + collect_data_files("ctranslate2")
    + collect_data_files("tokenizers")
)

def _flat_nvidia_libs(collected):
    """Find the installed nvidia-* wheels and return (src, dest) pairs that
    put every CUDA DLL FLAT into _internal/ctranslate2/, next to
    ctranslate2.dll. The Windows loader only searches the process dir and
    add_dll_directory() paths, so the nvidia/<lib>/bin/ wheel layout leaves
    cuDNN's sub-DLLs unresolvable -> ctranslate2 drops float16 -> whisper
    decodes on float32. No-op when the nvidia packages are absent (CPU
    builds)."""
    import glob
    try:
        import nvidia
    except ImportError:
        return []
    # "nvidia" is a namespace package (no __init__.py): __file__ is None,
    # the package root lives in the __path__ iterable.
    pkg_root = next(iter(nvidia.__path__), None)
    if not pkg_root:
        return []
    pairs = []
    # Skip DLLs already shipped FLAT next to ctranslate2.dll: duplicate dest
    # paths make COLLECT fail. Entries that land in a nested tree
    # (nvidia/<lib>/bin/...) do NOT count — the Windows loader cannot see
    # them there, so those DLLs still need a flat copy.
    # NOTE: PyInstaller's dest for binaries is a DIRECTORY: the final location
    # is always dest + basename(src). Passing a full file path here makes
    # PyInstaller materialize a DIRECTORY named "<name>.dll" containing the
    # file — invisible to the Windows loader.
    def _is_flat(src: str, dest: str) -> bool:
        norm = dest.replace("\\", "/").strip("/")
        return norm.lower() == "ctranslate2"

    already = {
        os.path.basename(src)
        for src, dest in collected
        if _is_flat(src, dest)
    }
    for dll in glob.glob(os.path.join(pkg_root, "*", "bin", "*.dll")):
        base = os.path.basename(dll)
        if base in already:
            continue
        already.add(base)
        pairs.append((dll, "ctranslate2"))
    return pairs


_base_binaries = (
    collect_dynamic_libs("ctranslate2")
    + collect_dynamic_libs("onnxruntime")
    + collect_dynamic_libs("sentencepiece")
)

# CUDA: cuDNN/cuBLAS/cuFFT/cuRAND sub-DLLs (cudnn_engines_*, cudnn_graph*,
# cublas*, ...) must sit NEXT to ctranslate2.dll in a directory the Windows
# loader searches (_internal/ctranslate2/). The nvidia/<lib>/bin/ wheel layout
# is invisible to the loader, so cuDNN fails to load and whisper silently falls
# back to float32. We therefore ship ONE flat copy and do NOT collect the
# nested nvidia tree (that used to duplicate every DLL, ~1.9 GB). Skipped
# entirely on CPU builds.
if _CPU_ONLY:
    binaries = _base_binaries
else:
    binaries = _base_binaries + _flat_nvidia_libs(_base_binaries)

hiddenimports = (
    collect_submodules("mockingbird")
    + collect_submodules("faster_whisper")
    + collect_submodules("ctranslate2")
    + collect_submodules("onnxruntime")
    + collect_submodules("sounddevice")
    + collect_submodules("pyaudiowpatch")
    + collect_submodules("openai")
    + collect_submodules("tokenizers")
    + collect_submodules("sentencepiece")
    + ["PySide6.QtSvg", "PySide6.QtMultimedia"]
)

_excludes = [
    # GigaAM leftovers in the build env (installed as user packages):
    # nothing in mockingbird imports them anymore, but PyInstaller still
    # follows them through transitive deps and ships ~2 GB of torch.
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
    # No CUDA runtime in a CPU build. The nvidia-* wheels may still be
    # installed in the build env; keep PyInstaller from following them.
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


def _version_file():
    """Write a temporary VSVersionInfo file for the current app version.

    The version string lives in ``mockingbird.__version__`` (single source of
    truth); the Windows exe must carry it so Inno Setup's
    ``GetVersionNumbersString`` produces ``Mockingbird-Setup-<ver>.exe``
    instead of a 0.0.0 default.
    """
    import sys

    sys.path.insert(0, _SRC)
    from mockingbird import __version__  # noqa: E402

    quad = ".".join((__version__.split(".") + ["0", "0", "0"])[:4])
    path = os.path.join(SPECPATH, "build", f"version_info_{__version__}.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            f"""# UTF-8
#
# Generated from mockingbird.__version__ by mockingbird.spec — do not edit.
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({", ".join(quad.split("."))}),
    prodvers=({", ".join(quad.split("."))}),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          '040904B0',
          [StringStruct('CompanyName', 'Mockingbird'),
          StringStruct('FileDescription', 'Mockingbird — live interview assistant'),
          StringStruct('FileVersion', '{__version__}'),
          StringStruct('InternalName', 'mockingbird'),
          StringStruct('OriginalFilename', 'mockingbird.exe'),
          StringStruct('ProductName', 'Mockingbird'),
          StringStruct('ProductVersion', '{__version__}')])
      ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
        )
    return path


_VERSION_FILE = _version_file()

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
    icon=_ICON,
    version=_VERSION_FILE,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="mockingbird",
)
