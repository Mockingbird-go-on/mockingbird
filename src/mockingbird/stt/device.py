"""Compute-device selection shared by the STT engines.

Kept free of heavy imports so it can be unit-tested on the build machine
(no torch) and so a frozen .exe can decide the device before importing the
backend stack. The whisper engine supplies its own CUDA probe (ctranslate2).
"""
from __future__ import annotations

import glob
import logging
import os

log = logging.getLogger(__name__)

_VALID = {"auto", "cpu", "cuda"}

_cudnn_dll_dirs_added = False


def ensure_cudnn_dll_dirs() -> None:
    """Make cuDNN 9 loadable by ctranslate2 on Windows.

    The nvidia-* wheels ship their DLLs in ``nvidia/<lib>/bin/`` subfolders
    that the Windows loader never searches, so ``cudnn64_9.dll`` resolves but
    its sub-DLLs (cudnn_engines_*, cudnn_graph*, cudnn_ops*, ...) do not:
    cuDNN fails to load and ctranslate2 silently drops float16 from the
    supported compute types (whisper then decodes on float32). Register every
    ``nvidia/*/bin`` directory with ``os.add_dll_directory`` before the first
    ctranslate2 call. Idempotent; no-op on non-Windows or when the nvidia
    packages are absent (CPU builds).
    """
    global _cudnn_dll_dirs_added
    if _cudnn_dll_dirs_added or os.name != "nt":
        return
    _cudnn_dll_dirs_added = True
    try:
        import nvidia

        roots = list(nvidia.__path__ or [])
    except Exception:  # noqa: BLE001
        return
    added = 0
    for root in roots:
        for lib_dir in glob.glob(os.path.join(root, "*", "bin")):
            try:
                os.add_dll_directory(lib_dir)
                added += 1
            except OSError:
                continue
    if added:
        log.info("device: registered %d nvidia DLL directories for cuDNN", added)



def is_valid_device(device: str | None) -> bool:
    return (device or "auto").lower() in _VALID


def device_label(device: str | None) -> str:
    """Human-readable badge text for a resolved device: GPU/CPU/…."""
    if (device or "").lower() == "cuda":
        return "GPU"
    if (device or "").lower() == "cpu":
        return "CPU"
    return "…"


def resolve_device(preferred: str | None, cuda_available: bool = False) -> str:
    """Map a configured device string to a concrete backend device.

    * ``"auto"``  -> ``"cuda"`` when a CUDA device exists, else ``"cpu"``.
    * ``"cpu"``   -> ``"cpu"`` always.
    * ``"cuda"``  -> ``"cuda"`` when available, else ``"cpu"`` (graceful
      fallback so the app never crashes on a machine without a driver).
    Unknown values are treated as ``"auto"``.
    """
    choice = (preferred or "auto").lower()
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda" if cuda_available else "cpu"
    return "cuda" if cuda_available else "cpu"


def ctranslate2_cuda_available() -> bool:
    """CUDA probe for the faster-whisper backend (ctranslate2 is required)."""
    try:
        import ctranslate2

        ensure_cudnn_dll_dirs()
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False
