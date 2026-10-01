"""Target picker for live test mode.

Windows: click-to-pick — a fullscreen crosshair overlay; clicking anywhere
resolves the root window under the cursor via Win32 (WindowFromPoint +
GetAncestor) and returns (hwnd, title, rect).

Other platforms / fallback: reuse ScreenGrabOverlay region selection; the
watcher then captures that desktop region instead of a specific window.
"""
from __future__ import annotations

import logging
import sys

from PySide6.QtCore import Qt, QRect, Signal

from mockingbird.i18n import t
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

log = logging.getLogger(__name__)


def pick_window_under_cursor() -> tuple[int, str, QRect] | None:
    """Win32: resolve the root window under the mouse cursor."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class POINT(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

    pos = QCursor.pos()
    pt = POINT(pos.x(), pos.y())
    hwnd = ctypes.windll.user32.WindowFromPoint(pt)
    if not hwnd:
        return None
    GA_ROOT = 2
    root = ctypes.windll.user32.GetAncestor(hwnd, GA_ROOT)
    if not root:
        root = hwnd
    title = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetWindowTextW(root, title, 256)
    rect = wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(root, ctypes.byref(rect)):
        return None
    return int(root), title.value or "(без заголовка)", QRect(
        rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    )


class WindowPickOverlay(QWidget):
    """Fullscreen click-to-pick overlay (crosshair + hint).

    Signals:
        picked(int, str, QRect)  — hwnd, title, window rect (Windows only)
        region_picked(QRect)     — desktop region chosen (Linux/fallback)
        cancelled()
    """

    picked = Signal(int, str, QRect)
    region_picked = Signal(QRect)
    cancelled = Signal()

    def __init__(self, allow_region_fallback: bool = True, parent=None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self._fallback = allow_region_fallback and sys.platform != "win32"
        self.setCursor(Qt.CursorShape.CrossCursor)
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry() if screen else QRect(0, 0, 1920, 1080)
        self.setGeometry(geo)
        self._hint = t("Кликните по окну с тестом (Esc — отмена)")

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 90))
        p.setPen(QPen(QColor(255, 255, 255, 220), 2))
        f = p.font()
        f.setPointSize(12)
        p.setFont(f)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._hint)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            self.cancelled.emit()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.close()
        if sys.platform == "win32":
            info = pick_window_under_cursor()
            if info:
                hwnd, title, rect = info
                self.picked.emit(hwnd, title, rect)
                return
            log.warning("test-mode: WindowFromPoint returned nothing")
        if self._fallback:
            # Non-Windows / failed pick: fall back to the rubber-band
            # region selector from the screenshot feature.
            from mockingbird.ui.screenshot import ScreenGrabOverlay

            ov = ScreenGrabOverlay(self)
            ov.finished.connect(
                lambda r: self.region_picked.emit(r) if r.isValid() else self.cancelled.emit()
            )
            ov.cancelled.connect(self.cancelled)
            ov.start()
        else:
            self.cancelled.emit()
