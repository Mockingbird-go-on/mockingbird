"""Compute-device selection shared by the STT engines.

Kept free of heavy imports so it can be unit-tested on the build machine
(no torch) and so a frozen .exe can decide the device before importing the
backend stack. Each engine supplies its own CUDA probe (torch for GigaAM,
ctranslate2 for faster-whisper).
"""
from __future__ import annotations

_VALID = {"auto", "cpu", "cuda"}


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


def torch_cuda_available() -> bool:
    """CUDA probe for the GigaAM backend (torch is already required)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def ctranslate2_cuda_available() -> bool:
    """CUDA probe for the faster-whisper backend (ctranslate2 is required)."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False
