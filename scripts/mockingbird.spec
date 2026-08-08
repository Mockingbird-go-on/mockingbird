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
_VENDOR = os.path.join(_ROOT, "vendor")
_ICON = os.path.join(SPECPATH, "logo_mockingbird.ico")

datas = (
    collect_data_files("mockingbird")
    + [(os.path.join(SPECPATH, "logo_mockingbird.ico"), "mockingbird")]
    + [(os.path.join(_ROOT, "sound.mp3"), "mockingbird")]
    + collect_data_files("faster_whisper")
    + collect_data_files("ctranslate2")
    + collect_data_files("transformers")
    + collect_data_files("tokenizers")
)

def _collect_optional(name: str):
    """Collect dynamic libs for an optional backend package (CUDA-only deps
    are absent on CPU builds; never fail the whole freeze because of that)."""
    try:
        return list(collect_dynamic_libs(name))
    except Exception:
        return []


binaries = (
    collect_dynamic_libs("ctranslate2")
    + collect_dynamic_libs("onnxruntime")
    + collect_dynamic_libs("sentencepiece")
    # CUDA torch: torch/lib holds c10/torch_cuda/asmjit/fbgemm/OpenMP DLLs.
    # cuDNN/cuBLAS/etc. live in the nvidia-* wheels and are picked up by
    # PyInstaller's nvidia hooks; the guarded collections below are a fallback
    # for any lib the hooks miss on GPU builds (and are no-ops on CPU builds).
    + _collect_optional("torch")
    + _collect_optional("nvidia.cudnn")
    + _collect_optional("nvidia.cublas")
    + _collect_optional("nvidia.cufft")
    + _collect_optional("nvidia.curand")
    + _collect_optional("nvidia.cusolver")
    + _collect_optional("nvidia.cusparse")
    + _collect_optional("nvidia.cuda_runtime")
)
hiddenimports = (
    collect_submodules("mockingbird")
    + collect_submodules("faster_whisper")
    + collect_submodules("ctranslate2")
    + collect_submodules("onnxruntime")
    + collect_submodules("sounddevice")
    + collect_submodules("pyaudiowpatch")
    + collect_submodules("openai")
    + collect_submodules("transformers")
    + collect_submodules("tokenizers")
    + collect_submodules("sentencepiece")
    + collect_submodules("hydra")
    + collect_submodules("omegaconf")
    # The remote GigaAM modeling file imports `pyannote` in a code path this
    # app never uses; transformers' check_imports still requires the top-level
    # package to be importable. vendor/pyannote is a tiny stand-in for it.
    + ["pyannote"]
)

a = Analysis(
    [_ENTRY],
    pathex=[_SRC, _VENDOR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
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
    icon=_ICON,
)
# Console (CLI) variant for headless testing: same code, but with a console
# window so the interactive REPL (``mockingbird-cli`` / ``--cli``) works when
# launched from Explorer or as a scheduled task.
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
    icon=_ICON,
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
