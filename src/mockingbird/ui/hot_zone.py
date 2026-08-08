"""Top hot-zone + fade group for Simple Mode (minimal UI on hover).

The hot-zone is a thin invisible widget at the top of the central layout.
When the cursor enters it, the hidden chrome (toolbar/sidebar/etc.) is
revealed via a fade animation; when it leaves, a short delay hides it again.
"""
from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget

_FADE_MS = 200


class TopHotZone(QWidget):
    """Thin invisible strip that emits hover_enter/hover_leave signals."""

    hover_enter = Signal()
    hover_leave = Signal()

    def __init__(self, parent=None, height: int = 8):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def enterEvent(self, event):
        self.hover_enter.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hover_leave.emit()
        super().leaveEvent(event)


class _FadeGroup:
    """Manages opacity animations for a list of widgets.

    Each widget gets a QGraphicsOpacityEffect; ``fade_to`` animates all of
    them in parallel with an OutCubic easing curve. ``set_visible_immediate``
    bypasses animation for one-shot show/hide (init/teardown).
    """

    def __init__(self, widgets: list[QWidget]):
        self._widgets = widgets
        self._effects: dict[QWidget, QGraphicsOpacityEffect] = {}
        self._animations: list[QPropertyAnimation] = []
        for w in widgets:
            eff = QGraphicsOpacityEffect(w)
            eff.setOpacity(1.0)
            w.setGraphicsEffect(eff)
            self._effects[w] = eff

    def fade_to(self, target_opacity: float, on_finished=None) -> None:
        # Stop any in-flight animations to avoid overlap.
        for anim in self._animations:
            anim.stop()
        self._animations.clear()
        for w, eff in self._effects.items():
            anim = QPropertyAnimation(eff, b"opacity", w)
            anim.setDuration(_FADE_MS)
            anim.setStartValue(eff.opacity())
            anim.setEndValue(target_opacity)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            if on_finished is not None:
                anim.finished.connect(on_finished)
            anim.start(QPropertyAnimation.DeletionPolicy.KeepWhenStopped)
            self._animations.append(anim)

    def set_visible_immediate(self, visible: bool) -> None:
        """Instant show/hide (no animation) — for init or teardown."""
        for anim in self._animations:
            anim.stop()
        self._animations.clear()
        for w, eff in self._effects.items():
            w.setVisible(visible)
            eff.setOpacity(1.0 if visible else 0.0)
