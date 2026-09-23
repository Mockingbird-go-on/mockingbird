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

from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QWidget

log = logging.getLogger(__name__)

_MIN_REGION_PX = 10


def grab_screen_region(rect: QRect, max_dim: int = 1600, jpeg_quality: int = 80) -> tuple[bytes, QPixmap]:
    """Capture ``rect`` (device pixels) from the primary screen as JPEG bytes.

    The longest side is bounded by ``max_dim`` so a 4K screenshot does not
    blow up the request payload. Returns (jpeg_bytes, preview_pixmap).
    """
    screen = QApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("нет доступных экранов")
    # grabWindow takes a pixmap of the whole desktop; crop in device pixels.
    full = screen.grabWindow(0)
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    # Map logical to device pixels when the screen is scaled.
    dpr = screen.devicePixelRatio()
    pix = full.copy(
        int(x * dpr), int(y * dpr), int(w * dpr), int(h * dpr)
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
        screen = QApplication.primaryScreen()
        if screen is None:
            self.cancelled.emit()
            return
        self._shot = screen.grabWindow(0)
        geo = screen.geometry()
        self.setGeometry(geo)
        self.setMouseTracking(True)
        self.showFullScreen()
        self.activateWindow()
        self.raise_()

    # -- events ---------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._shot is None:
            return
        p = QPainter(self)
        p.drawPixmap(0, 0, self._shot)
        dim = QColor(0, 0, 0, 130)
        p.fillRect(self.rect(), dim)
        if not self._rect.isNull() and self._rect.width() > 0:
            # Re-draw the selected region undimmed.
            p.drawPixmap(self._rect, self._shot, self._rect)
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
        self.finished.emit(r)
        self.close()

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
        from .icons import icon as lucide_icon

        self._jpeg = jpeg_bytes
        self._busy = False

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
        title = QLabel("Вопрос по скриншоту")
        title.setObjectName("mddTitle")
        head.addWidget(glyph)
        head.addWidget(title, 1)
        close = QPushButton("×")
        close.setFixedSize(24, 24)
        close.setFlat(True)
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
        self._preview.setStyleSheet("border-radius: 8px; background: rgba(0,0,0,0.25);")
        lay.addWidget(self._preview)

        self._status = QLabel("")
        self._status.setObjectName("mddDetail")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("Спросить по скриншоту… (Enter — отправить)")
        self._edit.returnPressed.connect(self._on_ask)
        row.addWidget(self._edit, 1)
        self._ask_btn = QPushButton("Спросить")
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
            self._status.setText("✓ Модель поддерживает изображения")
        elif ok is False:
            self._status.setText("✗ Текущая LLM-модель не поддерживает изображения — "
                                 "измените модель в настройках")
        else:
            self._status.setText("Проверка поддержки изображений моделью…")

    def _on_ask(self) -> None:
        if self._busy:
            return
        q = self._edit.text().strip()
        if not q:
            return
        self.set_busy(True, "Вопрос отправлен — ответ появится в «Ответ ИИ»")
        self.asked.emit(q)
