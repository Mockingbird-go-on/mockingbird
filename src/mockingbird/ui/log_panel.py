"""Log viewer tab: tail of the application log with copy/clear controls."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mockingbird.ui import theme

_MAX_LINES = 5000


class LogPanel(QWidget):
    def __init__(self, log_file: str | None = None, parent=None):
        super().__init__(parent)
        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(_MAX_LINES)
        self._view.setFont(QFont("monospace"))
        self._view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        self._copy_btn = QPushButton("Copy all")
        self._copy_btn.clicked.connect(self._copy_all)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._view.clear)

        buttons = QHBoxLayout()
        self._file_label = None
        if log_file:
            self._file_label = QLabel(f"Log file: {log_file}")
            self._file_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            buttons.addWidget(self._file_label)
        buttons.addStretch(1)
        buttons.addWidget(self._copy_btn)
        buttons.addWidget(self._clear_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        layout.addLayout(buttons)
        self.update_theme()

    def update_theme(self) -> None:
        if self._file_label is not None:
            self._file_label.setStyleSheet(f"color:{theme.TEXT_SECONDARY};")

    def append_line(self, line: str) -> None:
        self._view.appendPlainText(line)
        bar = self._view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _copy_all(self) -> None:
        view = self._view
        view.selectAll()
        view.copy()
        cursor = view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        view.setTextCursor(cursor)
