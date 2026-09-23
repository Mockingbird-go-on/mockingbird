"""Global hotkey via WinAPI RegisterHotKey (no extra dependency, no admin).

Registers Ctrl+Alt+H and calls a callback on a dedicated listener thread.
On non-Windows platforms ``start()`` returns False (no-op).
"""
from __future__ import annotations

import ctypes
import logging
import sys
import threading

log = logging.getLogger(__name__)

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_NOREPEAT = 0x4000
VK_H = 0x48
VK_S = 0x53

PM_REMOVE = 0x0001


def is_supported() -> bool:
    return sys.platform == "win32"


class GlobalHotkey:
    """Register Ctrl+Alt+H; call callback on the dedicated listener thread.

    The callback is invoked from a non-Qt thread — callers must bridge to Qt
    via a signal or ``QMetaObject.invokeMethod``.
    """

    def __init__(self, callback, modifiers: int | None = None, vk: int | None = None,
                 hotkey_id: int = 1):
        self._callback = callback
        self._modifiers = modifiers if modifiers is not None else (MOD_CONTROL | MOD_ALT)
        self._vk = vk if vk is not None else VK_H
        self._hotkey_id = hotkey_id
        self._thread: threading.Thread | None = None
        self._stop = False
        self._registered = False

    def start(self) -> bool:
        if not is_supported() or self._thread is not None:
            return False
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="global-hotkey")
        self._thread.start()
        return True

    def _run(self):
        user32 = ctypes.windll.user32

        # Set argtypes/restype explicitly for 64-bit safety.
        user32.RegisterHotKey.argtypes = [
            ctypes.c_void_p,  # hwnd
            ctypes.c_int,     # id
            ctypes.c_uint,    # fsModifiers
            ctypes.c_uint,    # vk
        ]
        user32.RegisterHotKey.restype = ctypes.c_int  # BOOL

        user32.UnregisterHotKey.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.UnregisterHotKey.restype = ctypes.c_int

        class MSG(ctypes.Structure):
            _fields_ = [
                ("hwnd", ctypes.c_void_p),
                ("message", ctypes.c_uint),
                ("wParam", ctypes.c_ulonglong),
                ("lParam", ctypes.c_longlong),
                ("time", ctypes.c_ulong),
                ("pt_x", ctypes.c_long),
                ("pt_y", ctypes.c_long),
            ]

        user32.PeekMessageW.argtypes = [
            ctypes.c_void_p,  # lpMsg
            ctypes.c_void_p,  # hWnd
            ctypes.c_uint,    # wMsgFilterMin
            ctypes.c_uint,    # wMsgFilterMax
            ctypes.c_uint,    # wRemoveMsg
        ]
        user32.PeekMessageW.restype = ctypes.c_int  # BOOL

        if not user32.RegisterHotKey(
            None, self._hotkey_id, self._modifiers | MOD_NOREPEAT, self._vk
        ):
            log.warning(
                "RegisterHotKey(id=%d) failed — key already taken?", self._hotkey_id
            )
            return
        self._registered = True

        msg = MSG()
        while not self._stop:
            if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                if msg.message == 0x0312:  # WM_HOTKEY
                    try:
                        self._callback()
                    except Exception:
                        log.exception("hotkey callback failed")
            threading.Event().wait(0.05)

        if self._registered:
            user32.UnregisterHotKey(None, self._hotkey_id)
            self._registered = False

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
