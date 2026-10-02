"""Target picker for live test mode.

Windows: click-to-pick — a fullscreen freeze-and-dim overlay (a real
screenshot of the desktop drawn slightly darkened, so every window stays
visible); clicking anywhere resolves the root window under the cursor via
Win32 (WindowFromPoint + GetAncestor) and returns (hwnd, title, rect).

Other platforms / fallback: reuse ScreenGrabOverlay region selection; the
watcher then captures that desktop region instead of a specific window.
"""
from __future__ import annotations

import logging
import sys

from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from mockingbird.i18n import t

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
    # Use the native cursor position directly: QCursor.pos() returns Qt
    # virtual-desktop coordinates which can mismatch physical pixels under
    # mixed-DPI monitors — GetCursorPos is always physical.
    import ctypes.wintypes as _wt

    native = _wt.POINT()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(native)):
        return None
    pt = POINT(native.x, native.y)
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
    """Fullscreen freeze-and-dim click-to-pick overlay.

    The desktop is captured ONCE and drawn with a light dim + the hint
    banner on top; the actual windows stay fully recognizable (the naive
    approach — a translucent window over the desktop — rendered as solid
    black on Windows because the window had no translucent background
    attribute, hiding every window from the user).

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
        self._shot: QPixmap | None = None
        # Multi-monitor: cover the screen the cursor is on (the user is
        # about to pick a window THERE); fall back to the primary screen.
        from PySide6.QtGui import QCursor as _QC

        scr = QApplication.screenAt(_QC.pos()) or QApplication.primaryScreen()
        geo = scr.availableGeometry() if scr else QRect(0, 0, 1920, 1080)
        self.setGeometry(geo)
        self._hint = t("Кликните по окну с тестом (Esc — отмена)")

    def start(self) -> None:
        scr = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        if scr is None:
            self.cancelled.emit()
            return
        self._shot = scr.grabWindow(0)
        self.showFullScreen()
        self.activateWindow()
        self.raise_()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        if self._shot is not None:
            p.drawPixmap(0, 0, self._shot)
            p.fillRect(self.rect(), QColor(0, 0, 0, 60))  # light dim
        else:
            p.fillRect(self.rect(), QColor(20, 20, 20))
        p.setPen(QPen(QColor(255, 255, 255, 240), 2))
        f = p.font()
        f.setPointSize(12)
        f.setBold(True)
        p.setFont(f)
        # Banner plate behind the text so it stays readable over any content
        from PySide6.QtCore import QRectF

        fm = p.fontMetrics()
        text_w = fm.horizontalAdvance(self._hint)
        banner = QRectF(
            self.width() / 2 - text_w / 2 - 16,
            self.height() / 2 - fm.height() / 2 - 8,
            text_w + 32,
            fm.height() + 16,
        )
        p.setBrush(QColor(0, 0, 0, 170))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(banner, 8, 8)
        p.setPen(QPen(QColor(255, 255, 255, 240)))
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
            # region selector from the screenshot feature. parent=None —
            # ``self`` is already closed and would hide the child overlay;
            # the attribute keeps the Qt object alive (GC guard).
            from mockingbird.ui.screenshot import ScreenGrabOverlay

            self._region_ov = ScreenGrabOverlay(None)
            self._region_ov.finished.connect(
                lambda r: self.region_picked.emit(r) if r.isValid() else self.cancelled.emit()
            )
            self._region_ov.cancelled.connect(self.cancelled)
            self._region_ov.start()
        else:
            self.cancelled.emit()
