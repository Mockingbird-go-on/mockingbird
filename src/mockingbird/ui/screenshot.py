"""Screenshot-to-answer: select a screen region, ask the LLM about it.

Flow: toolbar button / global hotkey → fullscreen selection overlay
(freeze-and-dim with a rubber-band rect) → JPEG (bounded, quality from the
config) → ScreenshotQuestionDialog (preview + question entry) → the answer
streams into the main «Ответ ИИ» pane through the regular LlmAnswer channel
and is recorded in the session history with the screenshot thumbnail.
"""
from __future__ import annotations

import base64
import logging

from PySide6.QtCore import QSize, Qt, QRect, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from mockingbird.i18n import t

log = logging.getLogger(__name__)

_MIN_REGION_PX = 10


def _screen_records() -> list[dict]:
    """Per-screen metadata: logical geometry + DPR, for the virtual desktop.

    Pure data collection (no grabbing) — unit-testable with stub screens.
    """
    from PySide6.QtGui import QGuiApplication

    out = []
    for scr in QGuiApplication.screens():
        out.append({"geometry": scr.geometry(), "dpr": scr.devicePixelRatio()})
    return out


def virtual_desktop_geometry(records: list[dict]):
    """Union rectangle of all screen records (logical coordinates)."""
    if not records:
        return QRect(0, 0, 0, 0)
    union = QRect(records[0]["geometry"])
    for rec in records[1:]:
        union = union.united(rec["geometry"])
    return union


def _clamp_to_desktop(rect: QRect, records: list[dict]) -> QRect:
    """Intersect the selection with the virtual desktop (a drag started on
    the overlay can never exceed it, but keep grab_screen_region robust for
    programmatic callers too)."""
    return rect.intersected(virtual_desktop_geometry(records))


def grab_screen_region(rect: QRect, max_dim: int = 1600, jpeg_quality: int = 80) -> tuple[bytes, QPixmap]:
    """Capture ``rect`` (global logical coordinates in the virtual desktop) as JPEG.

    The whole virtual desktop (ALL monitors, mixed DPI supported) is
    stitched per-screen and cropped. The crop scale is derived from the
    ACTUAL canvas/union size ratio (not a per-screen DPR), so it stays
    correct on any monitor layout (secondary left/above the primary →
    union origin != 0,0) and any DPI mix.
    The longest side is bounded by ``max_dim`` so a 4K screenshot does not
    blow up the request payload. Returns (jpeg_bytes, preview_pixmap).
    """
    records = _screen_records()
    if not records:
        raise RuntimeError("нет доступных экранов")
    rect = _clamp_to_desktop(rect, records)
    canvas = grab_virtual_desktop(records)
    return _crop_canvas(canvas, records, rect, max_dim, jpeg_quality)


def crop_device_rect(
    rect: QRect, union: QRect, canvas_w: int, canvas_h: int,
) -> tuple[int, int, int, int]:
    """Map a global-logical ``rect`` onto the stitched canvas device pixels.

    Pure math (no Qt widgets): the scale is ``canvas device px / union
    logical px`` — one uniform factor for the whole desktop, independent of
    monitor layout (union origin may be negative) and DPI mix. Returns
    ``(x, y, w, h)`` in canvas device pixels for ``QPixmap.copy``.
    """
    scale_x = canvas_w / max(union.width(), 1)
    scale_y = canvas_h / max(union.height(), 1)
    x = rect.x() - union.x()
    y = rect.y() - union.y()
    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        int(round(rect.width() * scale_x)),
        int(round(rect.height() * scale_y)),
    )


def _crop_canvas(
    canvas: QPixmap,
    records: list[dict],
    rect: QRect,
    max_dim: int,
    jpeg_quality: int,
) -> tuple[bytes, QPixmap]:
    """Crop ``rect`` (global logical coords) out of a stitched ``canvas``.

    The scale is ``canvas device px / union logical px`` — a single uniform
    factor for the whole stitched desktop. Using the ratio keeps the math
    correct regardless of how the canvas was produced (mixed DPR/layout).
    """
    union = virtual_desktop_geometry(records)
    x, y, w, h = crop_device_rect(rect, union, canvas.width(), canvas.height())
    pix = canvas.copy(x, y, w, h)
    if pix.isNull() or pix.width() < _MIN_REGION_PX or pix.height() < _MIN_REGION_PX:
        raise RuntimeError("выделена пустая область")
    scaled = pix
    if max(pix.width(), pix.height()) > max_dim:
        scaled = pix.scaled(
            max_dim, max_dim,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    from PySide6.QtCore import QBuffer, QIODevice

    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    scaled.save(buf, "JPEG", jpeg_quality)
    return bytes(buf.data()), scaled


def grab_virtual_desktop(records: list[dict] | None = None) -> QPixmap:
    """Stitch a freeze-frame of the ENTIRE virtual desktop (all monitors).

    The canvas uses the MAX per-screen DPR as its uniform scale: a shot
    from a lower-DPR monitor is upscaled (slightly soft) but geometry stays
    consistent, while hi-DPI monitors keep native sharpness. Mixed-DPI
    layouts are thereby covered with a single crop coordinate space.
    """
    if records is None:
        records = _screen_records()
    if not records:
        raise RuntimeError("нет доступных экранов")
    union = virtual_desktop_geometry(records)
    canvas_dpr = max((rec["dpr"] or 1.0) for rec in records)
    canvas = QPixmap(
        int(round(union.width() * canvas_dpr)),
        int(round(union.height() * canvas_dpr)),
    )
    canvas.setDevicePixelRatio(canvas_dpr)
    canvas.fill(QColor(0, 0, 0))
    from PySide6.QtGui import QGuiApplication

    screens = QGuiApplication.screens()
    painter = QPainter(canvas)
    try:
        for i, rec in enumerate(records):
            scr = screens[i] if i < len(screens) else None
            if scr is None:
                continue
            shot = scr.grabWindow(0)
            # The canvas carries devicePixelRatio(canvas_dpr), so QPainter
            # works in LOGICAL coordinates on it — do NOT pre-multiply the
            # target by canvas_dpr (that double-scaled every monitor and
            # pushed them off the canvas at >100% Windows scaling). The
            # higher-DPR canvas makes the painter upscale a lower-DPR shot
            # into the uniform grid automatically.
            painter.drawPixmap(
                rec["geometry"].x() - union.x(),
                rec["geometry"].y() - union.y(),
                rec["geometry"].width(),
                rec["geometry"].height(),
                shot,
            )
    finally:
        painter.end()
    return canvas


class ScreenGrabOverlay(QWidget):
    """Freeze-and-dim fullscreen overlay with a rubber-band selection.

    The whole virtual desktop is captured once and drawn darkened; the
    selected rectangle is revealed un-darkened. Esc cancels; releasing the
    mouse after a valid drag emits :attr:`finished`.
    """

    finished = Signal(QRect)
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(None,
                         Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._shot: QPixmap | None = None
        self._origin = None
        self._rect = QRect()

    def start(self) -> None:
        records = _screen_records()
        if not records:
            self.cancelled.emit()
            return
        try:
            self._shot = grab_virtual_desktop(records)
        except Exception as exc:  # noqa: BLE001
            log.warning("virtual desktop grab failed: %s", exc)
            self.cancelled.emit()
            return
        # The shot was stitched on a canvas tagged with canvas_dpr — strip
        # it: the overlay widget lives in LOGICAL desktop coordinates, and
        # paintEvent draws with explicit logical → device mapping below.
        self._shot.setDevicePixelRatio(1)
        # Cover the WHOLE virtual desktop (all monitors): NOT showFullScreen
        # — Qt fullscreen snaps a window to a single screen, which is why
        # multi-monitor setups used to see the dim overlay on one monitor.
        geo = virtual_desktop_geometry(records)
        self.setGeometry(geo)
        self.setMouseTracking(True)
        self.showNormal()
        self.activateWindow()
        self.raise_()

    # -- events ---------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._shot is None:
            return
        p = QPainter(self)
        # Widget coordinates == logical virtual-desktop coordinates; the
        # shot is DPR-1 device pixels of the stitched canvas. Scale it to
        # the widget size explicitly (mixed-DPR: uniform canvas scale).
        p.drawPixmap(0, 0, self.width(), self.height(), self._shot)
        dim = QColor(0, 0, 0, 130)
        p.fillRect(self.rect(), dim)
        if not self._rect.isNull() and self._rect.width() > 0:
            # Re-draw the selected region undimmed (source in shot device
            # pixels via the same uniform scale).
            scale_x = self._shot.width() / max(self.width(), 1)
            scale_y = self._shot.height() / max(self.height(), 1)
            src = QRect(
                int(self._rect.x() * scale_x), int(self._rect.y() * scale_y),
                int(self._rect.width() * scale_x), int(self._rect.height() * scale_y),
            )
            p.drawPixmap(self._rect, self._shot, src)
            pen = QPen(QColor(120, 180, 255, 220), 2)
            p.setPen(pen)
            p.drawRect(self._rect)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._origin = e.position().toPoint()
            self._rect = QRect(self._origin, self._origin)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._origin is not None:
            self._rect = QRect(self._origin, e.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() != Qt.MouseButton.LeftButton or self._origin is None:
            return
        self._origin = None
        r = self._rect.normalized()
        if r.width() < _MIN_REGION_PX or r.height() < _MIN_REGION_PX:
            self._toast_too_small()
            return
        # Emit in GLOBAL logical desktop coordinates: the overlay widget is
        # positioned at the union origin and mouse positions are widget-local,
        # so a secondary monitor left/above the primary would otherwise shift
        # the crop (locals != globals when union.x()/y() != 0).
        self.finished.emit(QRect(self.mapToGlobal(r.topLeft()), r.size()))
        self.close()

    def crop_selection(
        self, rect: QRect, max_dim: int = 1600, jpeg_quality: int = 80,
    ) -> tuple[bytes, QPixmap]:
        """Crop ``rect`` (global logical coords) out of the FROZEN frame.

        Captures exactly what the user saw and outlined — no second desktop
        grab (the screen could have changed, and capture-excluded windows
        would appear/disappear between the freeze and the crop). ``self._shot``
        is DPR-1 device pixels; the widget covers the union rectangle, so the
        offset is ``global - geometry().topLeft()`` and the scale is shot/union.
        """
        if self._shot is None:
            raise RuntimeError("нет замороженного кадра")
        scale_x = self._shot.width() / max(self.width(), 1)
        scale_y = self._shot.height() / max(self.height(), 1)
        ox, oy = self.geometry().x(), self.geometry().y()
        x = rect.x() - ox
        y = rect.y() - oy
        pix = self._shot.copy(
            int(round(x * scale_x)), int(round(y * scale_y)),
            int(round(rect.width() * scale_x)),
            int(round(rect.height() * scale_y)),
        )
        if pix.isNull() or pix.width() < _MIN_REGION_PX or pix.height() < _MIN_REGION_PX:
            raise RuntimeError("выделена пустая область")
        scaled = pix
        if max(pix.width(), pix.height()) > max_dim:
            scaled = pix.scaled(
                max_dim, max_dim,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        from PySide6.QtCore import QBuffer, QIODevice

        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        scaled.save(buf, "JPEG", jpeg_quality)
        return bytes(buf.data()), scaled

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()

    def _toast_too_small(self) -> None:
        # Minimal feedback: keep the overlay open for another try.
        self._rect = QRect()
        self.update()


class ScreenshotQuestionDialog(QWidget):
    """Compact card: screenshot preview + question entry.

    The answer itself is NOT rendered here — it streams into the main
    «Ответ ИИ» pane (single answer surface, per product decision). The
    dialog shows the vision-model status and a short progress note while
    the answer is being generated.
    """

    asked = Signal(str)  # question text
    rejected = Signal(str)  # human-readable failure reason

    def __init__(self, jpeg_bytes: bytes, preview: QPixmap, parent=None,
                 vision_ok: bool | None = None):
        super().__init__(None,
                         Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(480)

        from PySide6.QtWidgets import (
            QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget as QW,
        )

        from . import theme
        from .icons import CloseButton, icon as lucide_icon

        self._jpeg = jpeg_bytes
        self._busy = False
        self._drag_pos = None  # offset for frameless window dragging

        root = QW(self)
        root.setObjectName("mddCard")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(20, 16, 20, 14)
        lay.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(10)
        glyph = QLabel()
        try:
            glyph.setPixmap(
                lucide_icon("image", size=18, color=theme.current.status_idle).pixmap(18, 18)
            )
        except Exception:  # noqa: BLE001
            pass
        glyph.setFixedSize(18, 18)
        title = QLabel(t("Вопрос по скриншоту"))
        title.setObjectName("mddTitle")
        head.addWidget(glyph)
        head.addWidget(title, 1)
        close = CloseButton()
        close.clicked.connect(self.close)
        head.addWidget(close)
        lay.addLayout(head)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setPixmap(
            preview.scaled(
                440, 220,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._preview.setStyleSheet("border-radius: 2px; background: rgba(0,0,0,0.25);")
        lay.addWidget(self._preview)

        self._status = QLabel("")
        self._status.setObjectName("mddDetail")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText(t("Спросить по скриншоту… (Enter — отправить)"))
        self._edit.returnPressed.connect(self._on_ask)
        row.addWidget(self._edit, 1)
        self._ask_btn = QPushButton(t("Спросить"))
        self._ask_btn.setProperty("primary", True)
        self._ask_btn.clicked.connect(self._on_ask)
        row.addWidget(self._ask_btn)
        lay.addLayout(row)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        self._set_vision(vision_ok)

    # -- public ---------------------------------------------------------------

    def set_busy(self, busy: bool, note: str = "") -> None:
        self._busy = busy
        self._ask_btn.setEnabled(not busy)
        self._edit.setEnabled(not busy)
        if note:
            self._status.setText(note)

    def center_on(self, window: QWidget | None) -> None:
        if window is not None:
            geo = window.geometry()
            self.adjustSize()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )
        self.show()
        self.raise_()
        self.activateWindow()
        self._edit.setFocus()

    def jpeg_base64(self) -> str:
        return base64.b64encode(self._jpeg).decode("ascii")

    def jpeg_bytes(self) -> bytes:
        return self._jpeg

    # -- internals ------------------------------------------------------------

    def _set_vision(self, ok: bool | None) -> None:
        if ok is True:
            self._status.setText(t("✓ Модель поддерживает изображения"))
            self._status.setStyleSheet("color: #2ea043;")  # green — capability OK
        elif ok is False:
            from . import theme

            self._status.setText(t("✗ Текущая LLM-модель не поддерживает изображения — измените модель в настройках"))
            self._status.setStyleSheet(
                f"color: {getattr(theme.current, 'status_error', '#e5484d')};"
            )
        else:
            self._status.setText(t("Проверка поддержки изображений моделью…"))
            self._status.setStyleSheet("")

    # Default question used when the user sends the screenshot without
    # typing anything (the question entry is optional).
    # Lazy t(): a class-attribute would freeze the language at import time.
    DEFAULT_QUESTION_KEY = "Распознай текст на экране. Если там вопрос(ы) — ответь на них."

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            # Frameless window: drag the card by any point (except the
            # text inputs, which consume their own mouse events).
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._drag_pos is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None

    def _on_ask(self) -> None:
        if self._busy:
            return
        q = self._edit.text().strip() or t(self.DEFAULT_QUESTION_KEY)
        self.set_busy(True, t("Вопрос отправлен — ответ появится в «Ответ ИИ»"))
        self.asked.emit(q)
        # Close the card immediately so the streamed answer in the main
        # «Ответ ИИ» pane is visible (the user asked for this behaviour).
        self.close()
