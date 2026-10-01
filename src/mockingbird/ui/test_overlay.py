"""Compact always-on-top overlay showing live test answers."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget,
)

from mockingbird.i18n import t

_ACCENT = "#4f9cf9"


class TestModeOverlay(QWidget):
    """Floating answer card: «1 → B  2 → D …», status line, controls.

    Signals:
        stop_requested()
        force_requested()   — re-analyse button (manual re-send)
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
        self._target = target_title or t("Тест")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)

        header = QHBoxLayout()
        self._title = QLabel(self._target)
        self._title.setStyleSheet("font-weight: bold;")
        header.addWidget(self._title, 1)

        btn_force = QPushButton()
        from mockingbird.ui.icons import icon as lucide_icon

        btn_force.setIcon(lucide_icon("rotate-cw"))
        btn_force.setIconSize(QSize(15, 15))
        btn_force.setFixedSize(26, 24)
        btn_force.setToolTip(t("Переанализировать текущий кадр"))
        btn_force.clicked.connect(self.force_requested)
        header.addWidget(btn_force)

        btn_stop = QPushButton()
        btn_stop.setIcon(lucide_icon("circle-x"))
        btn_stop.setIconSize(QSize(15, 15))
        btn_stop.setFixedSize(26, 24)
        btn_stop.setToolTip(t("Остановить наблюдение"))
        btn_stop.clicked.connect(self.stop_requested)
        header.addWidget(btn_stop)
        root.addLayout(header)

        self._answers = QTextBrowser()
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._answers.setFont(mono)
        self._answers.setPlaceholderText(t("Ожидание стабильного кадра…"))
        root.addWidget(self._answers, 1)

        # "Updated HH:MM:SS" — bright, bold: instantly tells the user the
        # answers below belong to THIS page version, not a stale one.
        self._updated_at = QLabel("")
        self._updated_at.setStyleSheet(
            f"color:{_ACCENT}; font-size: 11px; font-weight: bold;"
        )
        root.addWidget(self._updated_at)

        self._status = QLabel("запуск…")
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._status)

        # Liveness indicator: uptime in the header, refreshed every second —
        # a dead watcher loop is immediately visible (clock stops).
        from PySide6.QtCore import QTimer

        self._started_at = None
        self._alive = QTimer(self)
        self._alive.setInterval(1000)
        self._alive.timeout.connect(self._tick_alive)
        self._alive.start()

    def _tick_alive(self) -> None:
        from time import monotonic

        if self._started_at is None:
            return
        secs = int(monotonic() - self._started_at)
        self._title.setText(
            f"{self._target} · {secs//60:02d}:{secs%60:02d}"
        )

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

    def set_answers(self, pairs: list[tuple[str, str, str]], changed: bool = True) -> None:
        from time import localtime, strftime

        stamp = strftime("%H:%M:%S", localtime())
        if not pairs:
            self._answers.setPlainText(t("Точных ответов не найдено"))
            self._updated_at.setText("")
            self.set_status(t("тест не распознан на кадре") + f" · {stamp}")
            return
        lines = []
        for num, ans, note in pairs:
            line = f"<b>{num} → {ans}</b>"
            if note:
                line += f' <span style="color:#888;">— {note}</span>'
            lines.append(line)
        self._answers.setHtml("<br>".join(lines))
        if changed:
            self._updated_at.setText(
                t("обновлено в {time}", time=stamp)
                + f" · {t('{0} ответов').format(len(pairs))}"
            )
            self.set_status("")
        else:
            self._updated_at.setText(
                t("без изменений · {time}", time=stamp)
                + f" · {t('{0} ответов').format(len(pairs))}"
            )
            self.set_status("")
