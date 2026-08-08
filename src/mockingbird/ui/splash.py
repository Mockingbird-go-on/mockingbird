"""Animated circular loader splash screen — shown during app startup."""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from mockingbird.ui import theme


def _resolve_logo() -> QPixmap | None:
    """Locate ``logo_mockingbird.ico`` (dev + frozen) and return it clipped to a circle."""
    candidates: list[str] = []
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        candidates = [
            os.path.join(base, "mockingbird", "logo_mockingbird.ico"),
            os.path.join(base, "mockingbird", "assets", "logo_mockingbird.ico"),
        ]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.normpath(os.path.join(here, "..", "..", "scripts", "logo_mockingbird.ico")),
            os.path.normpath(os.path.join(here, "..", "assets", "logo_mockingbird.ico")),
        ]
    for path in candidates:
        if os.path.isfile(path):
            pix = QPixmap(path)
            if not pix.isNull():
                return pix
    return None


def _circular_pixmap(source: QPixmap, diameter: int) -> QPixmap:
    """Clip ``source`` to a smooth circle of ``diameter`` px, transparent outside."""
    out = QPixmap(diameter, diameter)
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    path = QPainterPath()
    path.addEllipse(0, 0, diameter, diameter)
    p.setClipPath(path)
    # Scale the source to fully cover the circle (KeepAspectRatioByExpanding).
    scaled = source.scaled(
        diameter, diameter, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    x = (scaled.width() - diameter) // 2
    y = (scaled.height() - diameter) // 2
    p.drawPixmap(0, 0, scaled, x, y, diameter, diameter)
    p.end()
    return out


class LoaderSplash(QWidget):
    """Frameless centered splash with a spinning circular loader + the app logo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(280, 280)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)
        # Pre-render the circular logo (cached for every paint).
        raw = _resolve_logo()
        if raw is not None and not raw.isNull():
            self._logo = _circular_pixmap(raw, 56)
        else:
            self._logo = None

    def show(self) -> None:
        # Center on screen
        from PySide6.QtWidgets import QApplication

        screen_obj = QApplication.primaryScreen()
        if screen_obj is None:
            screen = QRectF(0, 0, self.width(), self.height())
        else:
            g = screen_obj.geometry()
            screen = QRectF(g)
        self.move(
            int(screen.center().x() - self.width() / 2),
            int(screen.center().y() - self.height() / 2),
        )
        super().show()
        self._timer.start()

    def hide(self) -> None:
        self._timer.stop()
        super().hide()

    def _tick(self) -> None:
        self._angle = (self._angle + 8) % 360
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        t = theme.current

        # Background circle (card)
        rect = QRectF(20, 20, 240, 240)
        bg = QColor(t.bg)
        bg.setAlpha(230)
        painter.setBrush(bg)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 20, 20)

        # Spinner ring (background track) — sized so the logo fits inside.
        cx, cy, r = 140, 110, 42
        track_color = QColor(t.border)
        track_pen_width = 4
        painter.setPen(QPen(track_color, track_pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(
            int(cx - r),
            int(cy - r),
            int(r * 2),
            int(r * 2),
            0,
            360 * 16,
        )

        # Spinner arc (rotating)
        arc_color = QColor(t.accent)
        painter.setPen(QPen(arc_color, track_pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(
            int(cx - r),
            int(cy - r),
            int(r * 2),
            int(r * 2),
            int(self._angle * 16),
            100 * 16,  # ~100° arc
        )

        # Logo centred inside the ring (or a brand glyph if the icon is missing).
        if self._logo is not None:
            lw = self._logo.width()
            painter.drawPixmap(int(cx - lw / 2), int(cy - lw / 2), self._logo)
        else:
            painter.setPen(QColor(t.accent))
            font = painter.font()
            font.setPointSize(22)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                QRectF(cx - r, cy - r, r * 2, r * 2),
                Qt.AlignmentFlag.AlignCenter,
                "M",
            )

        # Text
        painter.setPen(QColor(t.text_secondary))
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(
            QRectF(20, 175, 240, 30),
            Qt.AlignmentFlag.AlignCenter,
            "Загрузка…",
        )
