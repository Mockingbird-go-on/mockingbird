"""Animated toggle switch (web-style) used in place of QCheckBox.

``ToggleSwitch`` is a drop-in replacement for :class:`QCheckBox`: same
``toggled``/``setChecked``/``blockSignals`` API, optional text label painted
next to the switch. The thumb slides with a ~150 ms ease-in-out animation;
colors follow the active palette in :mod:`mockingbird.ui.theme`.
"""
from __future__ import annotations

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
)
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton

from mockingbird.ui import theme

_TRACK_W = 36
_TRACK_H = 20
_THUMB_MARGIN = 2
_GAP = 8
_ANIM_MS = 150


class ToggleSwitch(QAbstractButton):
    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._text = text
        self._thumb = 0.0
        self._anim = QPropertyAnimation(self, b"thumbPos", self)
        self._anim.setDuration(_ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggled.connect(self._on_toggled)

    # --- geometry ---------------------------------------------------------

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self.font())
        text_w = fm.horizontalAdvance(self._text) if self._text else 0
        w = _TRACK_W + (_GAP + text_w if text_w else 0)
        return QSize(w, max(_TRACK_H, fm.height()))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    # --- animation property ----------------------------------------------

    def get_thumb_pos(self) -> float:
        return self._thumb

    def set_thumb_pos(self, value: float) -> None:
        self._thumb = value
        self.update()

    thumbPos = Property(float, get_thumb_pos, set_thumb_pos)

    def setChecked(self, checked: bool) -> None:  # noqa: FBT001 (Qt API)
        # Called directly (not via the toggled signal) when the owner syncs
        # state with blockSignals(True) — animate from here as well.
        super().setChecked(checked)
        self._animate_to(1.0 if checked else 0.0)

    def _on_toggled(self, checked: bool) -> None:
        self._animate_to(1.0 if checked else 0.0)

    def _animate_to(self, target: float) -> None:
        if abs(self._thumb - target) < 1e-6:
            return
        self._anim.stop()
        self._anim.setStartValue(self._thumb)
        self._anim.setEndValue(target)
        self._anim.start()

    # --- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        track = QRectF(0.0, (self.height() - _TRACK_H) / 2.0, _TRACK_W, _TRACK_H)
        radius = _TRACK_H / 2.0

        off = QColor("#3a4048")
        on = QColor(theme.current.accent)
        track_color = self._mix(off, on, self._thumb)
        if not self.isEnabled():
            track_color.setAlpha(120)
        painter.setPen(QPen(QColor(theme.current.border), 1))
        painter.setBrush(track_color)
        painter.drawRoundedRect(track, radius, radius)

        thumb_d = _TRACK_H - 2 * _THUMB_MARGIN
        x = _THUMB_MARGIN + self._thumb * (_TRACK_W - thumb_d - 2 * _THUMB_MARGIN)
        thumb_center = QPointF(
            track.left() + x + thumb_d / 2.0, track.center().y()
        )
        thumb_color = QColor(theme.LIGHT if theme.current.name == "dark" else theme.BG)
        if not self.isEnabled():
            thumb_color.setAlpha(160)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(thumb_color)
        painter.drawEllipse(thumb_center, thumb_d / 2.0, thumb_d / 2.0)

        if self._text:
            painter.setPen(
                QColor(theme.current.text if self.isEnabled() else theme.current.text_secondary)
            )
            text_x = _TRACK_W + _GAP
            painter.drawText(
                QRectF(text_x, 0.0, self.width() - text_x, self.height()),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self._text,
            )

    @staticmethod
    def _mix(a: QColor, b: QColor, t: float) -> QColor:
        t = max(0.0, min(1.0, t))
        return QColor(
            round(a.red() + (b.red() - a.red()) * t),
            round(a.green() + (b.green() - a.green()) * t),
            round(a.blue() + (b.blue() - a.blue()) * t),
        )
