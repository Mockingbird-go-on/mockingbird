"""Environment health checks for missing Linux dependencies.

On Windows everything is bundled (WASAPI/PortAudio, CUDA DLLs in the exe),
so every check here short-circuits to "ok" there. On Linux a few pieces
are commonly missing on minimal/headless installs, and each breaks a
specific feature *silently*:

Audio (``audio_health_issues``):
- ``libportaudio2`` — no mic capture at all (``import sounddevice`` raises);
- PulseAudio/PipeWire (``pactl``) — loopback mode («Динамик») has no monitors;
- zero input devices — nothing to record from (headless box, no ALSA).

Platform (``platform_health_issues``):
- XCB/X11 libraries — Qt GUI cannot start
  (``Could not load the Qt platform plugin "xcb"``);
- ``libfuse2`` — the AppImage itself refuses to launch;
- cuDNN/cuBLAS — NVIDIA driver present but CUDA userspace libraries are
  missing → whisper silently falls back to a very slow CPU decode.

``*_health_lines()`` return human-readable lines for the startup banner;
``*_health_issues()`` return structured issues (feature, reason, fix
command) for the UI badge. Every check is best-effort: a failing check
is logged and skipped, never raised.
"""
from __future__ import annotations

import ctypes.util
import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)


class AudioIssue:
    """One missing piece of the audio stack (Linux)."""

    def __init__(self, feature: str, reason: str, fix: str):
        self.feature = feature
        self.reason = reason
        self.fix = fix

    @property
    def title(self) -> str:
        return f"{self.feature}: {self.reason}"

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"AudioIssue({self.feature!r}, {self.reason!r})"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _check_portaudio() -> AudioIssue | None:
    """PortAudio shared library: without it sounddevice cannot even import."""
    if ctypes.util.find_library("portaudio") is not None:
        return None
    return AudioIssue(
        feature="Захват звука (микрофон)",
        reason="библиотека PortAudio не найдена",
        fix="sudo apt install libportaudio2",
    )


def _check_pulse() -> AudioIssue | None:
    """PulseAudio/PipeWire presence: required for loopback («Динамик»)."""
    if shutil.which("pactl") is not None:
        return None
    return AudioIssue(
        feature="Режим «Динамик» (loopback)",
        reason="PulseAudio/PipeWire не найдены (нет утилиты pactl)",
        fix="sudo apt install pipewire-pulseaudio"
        "  # или pulseaudio на старых системах",
    )


def _check_input_devices() -> AudioIssue | None:
    """At least one input device must exist for mic capture."""
    try:
        import sounddevice as sd
    except Exception:  # noqa: BLE001 - PortAudio missing: reported separately
        return None
    try:
        devices = sd.query_devices()
    except Exception:  # noqa: BLE001
        return None
    inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
    if inputs:
        return None
    return AudioIssue(
        feature="Захват звука (микрофон)",
        reason="не найдено ни одного устройства ввода (ALSA)",
        fix="подключите аудиоустройство / проверьте драйверы ALSA",
    )


_CHECKS = (_check_portaudio, _check_pulse, _check_input_devices)


def audio_health_issues() -> list[AudioIssue]:
    """All detected audio problems; empty list = everything is fine.

    On Windows (and any non-Linux platform) no checks run — the bundled
    WASAPI/PortAudio stack is always expected to work.
    """
    if not _is_linux():
        return []
    return _run_checks(_CHECKS)


def audio_health_lines() -> list[str]:
    """Banner lines (log file + diagnostics zip). Empty when healthy/Windows."""
    issues = audio_health_issues()
    if not _is_linux():
        return []
    if not issues:
        return ["audio: ok"]
    lines = ["audio: ⚠ problems detected (Linux):"]
    lines += [f"audio:   - {i.title} → {i.fix}" for i in issues]
    return lines


# --- Platform: GUI stack, AppImage, CUDA userspace -----------------------------


def _have_xcb_cursor() -> bool:
    """libxcb-cursor0: required by the Qt xcb plugin since Qt 6.5."""
    for cand in ("xcb_cursor", "xcb-cursor", "xcb_cursor0"):
        try:
            if ctypes.util.find_library(cand) is not None:
                return True
        except Exception:  # noqa: BLE001
            return True  # probe failed — do not false-positive
    return False


def _check_xcb() -> AudioIssue | None:
    """Qt XCB platform plugin dependencies (X11 client libraries).

    ``libxkbcommon-x11`` pulls the rest of the xcb family in; without it
    Qt dies at startup with «Could not load the Qt platform plugin "xcb"».
    ``libxcb-cursor0`` is needed separately since Qt 6.5. Skipped when a
    display is unavailable anyway (headless/ssh) — there the user knowingly
    runs offscreen or X-forwarded.
    """
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return None
    if not _have_xcb_cursor():
        return AudioIssue(
            feature="Графический интерфейс (Qt/X11)",
            reason="не найдена libxcb-cursor0 — окно не откроется (Qt ≥ 6.5)",
            fix="sudo apt install libxcb-cursor0",
        )
    if ctypes.util.find_library("X11") is not None and (
        ctypes.util.find_library("xkbcommon") is not None
    ):
        return None
    return AudioIssue(
        feature="Графический интерфейс (Qt/X11)",
        reason="не найдены библиотеки X11/xkbcommon — окно может не открыться",
        fix="sudo apt install libxkbcommon-x11-0 libxcb-cursor0 libgl1",
    )


def _check_fuse() -> AudioIssue | None:
    """libfuse2: only relevant when running as an AppImage."""
    appimage = os.environ.get("APPIMAGE") or (
        sys.argv[0].endswith(".AppImage") if sys.argv and sys.argv[0] else False
    )
    if not appimage:
        return None
    for cand in ("fuse2", "libfuse.so.2", "fuse"):
        if ctypes.util.find_library(cand) is not None:
            return None
    return AudioIssue(
        feature="Запуск AppImage",
        reason="не найдена libfuse2 — AppImage не запустится",
        fix="sudo apt install libfuse2",
    )


def _nvidia_driver_present() -> bool:
    try:
        if shutil.which("nvidia-smi") is None:
            return False
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def _check_cuda_userspace() -> AudioIssue | None:
    """NVIDIA driver OK but cuDNN/cuBLAS missing → silent slow CPU fallback."""
    if not _nvidia_driver_present():
        return None
    missing = [n for n in ("cudnn", "cublas", "cudart") if ctypes.util.find_library(n) is None]
    if not missing:
        return None
    return AudioIssue(
        feature="GPU-ускорение (CUDA)",
        reason="найден драйвер NVIDIA, но нет библиотек: "
        + ", ".join(missing)
        + " — распознавание будет медленным (CPU)",
        fix="sudo apt install libcudnn9 libcublas12   # версии — по вашему CUDA",
    )


_PLATFORM_CHECKS = (_check_xcb, _check_fuse, _check_cuda_userspace)


def platform_health_issues() -> list[AudioIssue]:
    """GUI/AppImage/CUDA problems; empty on non-Linux (bundled in the exe)."""
    if not _is_linux():
        return []
    return _run_checks(_PLATFORM_CHECKS)


def platform_health_lines() -> list[str]:
    issues = platform_health_issues()
    if not _is_linux():
        return []
    if not issues:
        return ["platform: ok"]
    lines = ["platform: ⚠ problems detected (Linux):"]
    lines += [f"platform:   - {i.title} → {i.fix}" for i in issues]
    return lines


def _run_checks(checks) -> list[AudioIssue]:
    issues: list[AudioIssue] = []
    for check in checks:
        try:
            issue = check()
        except Exception:  # noqa: BLE001 - health check must never crash
            log.debug("health check %s failed", check.__name__, exc_info=True)
            continue
        if issue is not None:
            issues.append(issue)
    return issues
