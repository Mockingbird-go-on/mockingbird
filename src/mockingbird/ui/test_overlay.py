"""Compact always-on-top overlay showing live test answers."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from mockingbird.i18n import t

from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget,
)


class TestModeOverlay(QWidget):
    """Floating answer card: «1 → B  2 → D …», status line, controls.

    Signals:
        stop_requested()
        force_requested()   — «сейчас» button (manual re-send)
    """

    stop_requested = Signal()
    force_requested = Signal()

    def __init__(self, target_title: str = "", parent=None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setWindowTitle("Mockingbird — тест")
        self.setFixedSize(340, 420)
        self._drag_pos = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)

        header = QHBoxLayout()
        self._title = QLabel(target_title or t("Тест"))
        self._title.setStyleSheet("font-weight: bold;")
        header.addWidget(self._title, 1)
        btn_force = QPushButton(t("Сейчас"))
        btn_force.setFixedHeight(24)
        btn_force.clicked.connect(self.force_requested)
        header.addWidget(btn_force)
        btn_stop = QPushButton("×")
        btn_stop.setFixedSize(26, 24)
        btn_stop.clicked.connect(self.stop_requested)
        header.addWidget(btn_stop)
        root.addLayout(header)

        self._answers = QTextBrowser()
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._answers.setFont(mono)
        self._answers.setPlaceholderText(t("Ожидание стабильного кадра…"))
        root.addWidget(self._answers, 1)

        self._status = QLabel("запуск…")
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._status)

    # -- drag by header ---------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_pos = None

    # -- updates ------------------------------------------------------------
    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_answers(self, pairs: list[tuple[str, str]], changed: bool = True) -> None:
        if not pairs:
            self._answers.setPlainText(t("Точных ответов не найдено"))
            return
        lines = []
        for num, ans in pairs:
            lines.append(f"{num} → {ans}")
        self._answers.setPlainText("\n".join(lines))
        if changed:
            self.set_status(t("{0} ответов · обновлено").format(len(pairs)))
        else:
            self.set_status(t("{0} ответов · без изменений").format(len(pairs)))
