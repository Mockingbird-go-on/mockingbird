"""Windows capture-affinity: hide windows from screen capture (Zoom/browser/...).

On Windows 10 build 19041+ uses ``WDA_EXCLUDEFROMCAPTURE`` — the window is
visible on the physical monitor but completely absent from any screen-capture
(Zoom, Teams, OBS, Win+Shift+S). On older Windows / Linux — no-op (returns
False without crashing).
"""
from __future__ import annotations

import ctypes
import logging
import sys

log = logging.getLogger(__name__)

WDA_NONE = 0x0
WDA_MONITOR = 0x1
WDA_EXCLUDEFROMCAPTURE = 0x11  # Windows 10 version 2004 (build 19041) and newer

_MIN_BUILD = 19041

# Win32 function prototypes (must be set explicitly on 64-bit to avoid
# pointer truncation -> STATUS_STACK_BUFFER_OVERRUN).
_user32 = ctypes.windll.user32 if sys.platform == "win32" else None
_ntdll = ctypes.windll.ntdll if sys.platform == "win32" else None

if _user32 is not None:
    _user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _user32.SetWindowDisplayAffinity.restype = ctypes.c_int  # BOOL

if _ntdll is not None:
    _ntdll.RtlGetVersion.argtypes = [ctypes.c_void_p]
    _ntdll.RtlGetVersion.restype = ctypes.c_long  # NTSTATUS (LONG)


class _OSVERSIONINFOEXW(ctypes.Structure):
    """RTL_OSVERSIONINFOEXW — full structure (required for RtlGetVersion)."""

    _fields_ = [
        ("dwOSVersionInfoSize", ctypes.c_ulong),
        ("dwMajorVersion", ctypes.c_ulong),
        ("dwMinorVersion", ctypes.c_ulong),
        ("dwBuildNumber", ctypes.c_ulong),
        ("dwPlatformId", ctypes.c_ulong),
        ("szCSDVersion", ctypes.c_wchar * 128),
        ("wServicePackMajor", ctypes.c_ushort),
        ("wServicePackMinor", ctypes.c_ushort),
        ("wSuiteMask", ctypes.c_ushort),
        ("wProductType", ctypes.c_byte),
        ("wReserved", ctypes.c_byte),
    ]


def is_supported() -> bool:
    return sys.platform == "win32"


def set_exclude_from_capture(hwnd_int: int) -> bool:
    """Mark window as excluded from screen capture. Returns True on success."""
    if not is_supported():
        return False
    try:
        ok = _user32.SetWindowDisplayAffinity(int(hwnd_int), WDA_EXCLUDEFROMCAPTURE)
        if not ok:
            log.warning(
                "SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) failed — "
                "old Windows build (<19041)?"
            )
        return bool(ok)
    except Exception:
        log.exception("capture affinity failed")
        return False


def clear(hwnd_int: int) -> bool:
    """Restore normal capture behaviour (WDA_NONE)."""
    if not is_supported():
        return False
    try:
        return bool(_user32.SetWindowDisplayAffinity(int(hwnd_int), WDA_NONE))
    except Exception:
        log.exception("capture affinity clear failed")
        return False


def windows_build() -> int | None:
    """Return the Windows build number, or None (not win32 / lookup failed).

    Uses ``ntdll!RtlGetVersion`` which is NOT affected by manifest-based
    compatibility lies (unlike ``GetVersionEx`` / ``VerifyVersionInfo``).
    """
    if not is_supported():
        return None
    try:
        info = _OSVERSIONINFOEXW()
        info.dwOSVersionInfoSize = ctypes.sizeof(_OSVERSIONINFOEXW)
        if _ntdll.RtlGetVersion(ctypes.byref(info)) == 0:
            return int(info.dwBuildNumber)
    except Exception:
        log.exception("RtlGetVersion failed")
    return None


def is_capture_protection_available() -> bool:
    """True if WDA_EXCLUDEFROMCAPTURE will actually work on this system."""
    build = windows_build()
    return build is not None and build >= _MIN_BUILD
