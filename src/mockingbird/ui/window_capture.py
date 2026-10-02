"""Per-window capture for the live test mode.

``QScreen.grabWindow(hwnd)`` uses the legacy BitBlt path, which returns a
blank/white image for GPU-composited windows (Firefox, Chrome, Electron —
they render through DirectX and don't paint into the legacy surface). The
reliable way on Windows is ``PrintWindow`` with PW_RENDERFULLCONTENT, which
asks the window to render its full (GPU) content into our bitmap.

Capture priority (Windows):
1. PrintWindow(PW_RENDERFULLCONTENT) into a DIB
2. QScreen.grabWindow(hwnd) fallback (works for legacy windows)

Non-Windows: QScreen.grabWindow(0) + crop (region-based, caller decides).
"""
from __future__ import annotations

import logging
import sys

from mockingbird.i18n import t

log = logging.getLogger(__name__)

_PW_RENDERFULLCONTENT = 0x00000002


def _print_window_capture(hwnd: int, w: int, h: int):
    """Capture via PrintWindow into a QImage. Returns QImage or None."""
    import ctypes
    from ctypes import wintypes

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.UINT),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.USHORT),
            ("biBitCount", wintypes.USHORT),
            ("biCompression", wintypes.UINT),
            ("biSizeImage", wintypes.UINT),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.UINT),
            ("biClrImportant", wintypes.UINT),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.UINT * 3)]

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    hdc_window = user32.GetDC(hwnd)
    if not hdc_window:
        return None
    try:
        hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
        if not hdc_mem:
            return None
        try:
            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = w
            bmi.bmiHeader.biHeight = -h  # top-down
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = 0  # BI_RGB

            ppv = ctypes.c_void_p()
            bitmap = gdi32.CreateDIBSection(
                hdc_mem, ctypes.byref(bmi), 0, ctypes.byref(ppv), None, 0
            )
            if not bitmap or not ppv:
                log.warning("test-mode: CreateDIBSection failed")
                return None
            gdi32.SelectObject(hdc_mem, bitmap)
            try:
                # WM_PRINTCLIENT-style full-content render (incl. GPU)
                res = user32.PrintWindow(hwnd, hdc_mem, _PW_RENDERFULLCONTENT)
                if not res:
                    log.warning("test-mode: PrintWindow failed (res=%s)", res)
                    return None
                # Copy DIB bits into a QImage (ARGB32, top-down)
                from PySide6.QtGui import QImage

                ptr = ctypes.cast(
                    ppv, ctypes.POINTER(ctypes.c_ubyte * (w * h * 4))
                )
                # RGB32 (not ARGB32): PrintWindow does not guarantee a
                # meaningful alpha channel — stale/zero alpha in ARGB32
                # would leak black/transparent artefacts into the JPEG.
                image = QImage(
                    bytes(ptr.contents), w, h, w * 4,
                    QImage.Format.Format_RGB32,
                )
                if image.isNull():
                    return None
                # Check the capture is not blank (all-white/all-black/empty):
                # a GPU window that refused to render yields a uniform image.
                if _is_blank(image):
                    log.warning("test-mode: PrintWindow frame looks blank")
                    return None
                return image
            finally:
                gdi32.DeleteObject(bitmap)
        finally:
            gdi32.DeleteDC(hdc_mem)
    finally:
        user32.ReleaseDC(hwnd, hdc_window)


def _is_blank(image) -> bool:
    """True when the image is (near-)uniform — a failed GPU capture."""
    small = image.scaled(64, 64)
    first = small.pixelColor(2, 2)
    fr, fg, fb = first.red(), first.green(), first.blue()
    for y in range(0, 64, 4):
        for x in range(0, 64, 4):
            c = small.pixelColor(x, y)
            if abs(c.red() - fr) > 8 or abs(c.green() - fg) > 8 or abs(c.blue() - fb) > 8:
                return False
    return True


def capture_window(hwnd: int, max_dim: int = 1600, jpeg_quality: int = 80):
    """Capture window ``hwnd`` -> (QImage, jpeg_bytes, QPixmap).

    Raises RuntimeError when every method fails (window closed/minimized).
    """
    from PySide6.QtCore import QBuffer, QIODevice, Qt
    from PySide6.QtGui import QGuiApplication, QPixmap

    rect = None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        rc = wintypes.RECT()
        if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rc)):
            w = rc.right - rc.left
            h = rc.bottom - rc.top
            if w > 10 and h > 10:
                rect = (w, h)

    image = None
    if rect is not None and sys.platform == "win32":
        try:
            image = _print_window_capture(hwnd, rect[0], rect[1])
            if image is not None:
                log.info("test-mode: captured via PrintWindow (%dx%d)", rect[0], rect[1])
        except Exception as exc:  # noqa: BLE001
            log.warning("test-mode: PrintWindow error: %s", exc)
            image = None

    pix = None
    if image is not None:
        pix = QPixmap.fromImage(image)

    if pix is None or pix.isNull():
        # Fallback: legacy BitBlt via Qt (works for non-GPU windows).
        # Pick the screen the window lives on — grabWindow on the primary
        # screen mangles windows that live on secondary monitors.
        screen = QGuiApplication.primaryScreen()
        if rect is not None and sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            rc = wintypes.RECT()
            if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rc)):
                center_x = (rc.left + rc.right) // 2
                center_y = (rc.top + rc.bottom) // 2
                from PySide6.QtCore import QPoint

                scr = QGuiApplication.screenAt(QPoint(center_x, center_y))
                if scr is not None:
                    screen = scr
        if screen is None:
            raise RuntimeError(t("нет экрана"))
        pix = screen.grabWindow(int(hwnd))
        if pix.isNull() or pix.width() < 10:
            raise RuntimeError(t("пустой кадр (окно свёрнуто?)"))
        image = pix.toImage()
        if _is_blank(image):
            # BOTH methods produced a uniform frame — the capture is dead
            # (minimised window, secured content). Raise so the watcher's
            # capture-fail counter can auto-stop instead of spinning
            # forever on a constant hash.
            log.warning(
                "test-mode: both PrintWindow and BitBlt returned a blank frame"
            )
            raise RuntimeError(t("пустой кадр (окно свёрнуто?)"))
        log.info("test-mode: captured via BitBlt fallback")

    image = pix.toImage()
    if max(image.width(), image.height()) > max_dim:
        pix = pix.scaled(
            max_dim, max_dim,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    pix.save(buf, "JPEG", jpeg_quality)
    return image, bytes(buf.data()), pix
