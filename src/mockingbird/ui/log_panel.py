"""Log viewer tab: optional live tail of the application log.

When disabled (default) the panel shows a placeholder with a checkbox that
turns logging on. When enabled, the toolbar in the top-right corner carries
a checkbox ("Лог включён") that turns logging back off; the buffer content
is preserved across off/on cycles. The in-process logging.Handler in
``mockingbird.app.App._install_log_handler`` is only wired when the user
opts in. The log file on disk is still written by ``logging_setup.py``
regardless of the checkbox — that's a config concern, not a UI one.
"""
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
from mockingbird.ui.toggle import ToggleSwitch

_MAX_LINES = 5000


class LogPanel(QWidget):
    def __init__(self, log_file: str | None = None, parent=None):
        super().__init__(parent)
        self._enabled: bool = False

        # Live tail (created up front but hidden until the user opts in).
        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(_MAX_LINES)
        self._view.setFont(QFont("monospace"))
        self._view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._view.setVisible(False)

        self._copy_btn = QPushButton("Copy all")
        self._copy_btn.clicked.connect(self._copy_all)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._view.clear)

        self._toolbar = QHBoxLayout()
        self._file_label = None
        if log_file:
            self._file_label = QLabel(f"Log file: {log_file}")
            self._file_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._toolbar.addWidget(self._file_label)
        self._active_checkbox = ToggleSwitch("Лог включён")
        self._active_checkbox.setChecked(False)
        self._active_checkbox.toggled.connect(self._on_toggle)
        self._toolbar.addStretch(1)
        self._toolbar.addWidget(self._copy_btn)
        self._toolbar.addWidget(self._clear_btn)
        self._toolbar.addWidget(self._active_checkbox)
        self._toolbar_widget = QWidget()
        self._toolbar_widget.setLayout(self._toolbar)
        self._toolbar_widget.setVisible(False)

        # Placeholder: shown while logging is disabled.
        self._placeholder = QWidget()
        ph_layout = QVBoxLayout(self._placeholder)
        ph_layout.setContentsMargins(24, 24, 24, 24)
        ph_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_text = QLabel(
            "Вкладка лога отключена.\n\n"
            "Включите её, чтобы видеть технический журнал приложения в реальном "
            "времени (полезно для диагностики). Файл лога на диске пишется всегда."
        )
        ph_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_text.setWordWrap(True)
        self._enable_checkbox = ToggleSwitch("Включить лог")
        self._enable_checkbox.setChecked(False)
        self._enable_checkbox.toggled.connect(self._on_toggle)
        ph_layout.addStretch(1)
        ph_layout.addWidget(ph_text)
        ph_layout.addSpacing(8)
        ph_layout.addWidget(self._enable_checkbox, alignment=Qt.AlignmentFlag.AlignCenter)
        ph_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._placeholder)
        layout.addWidget(self._view)
        layout.addWidget(self._toolbar_widget)
        self.update_theme()

    def update_theme(self) -> None:
        if self._file_label is not None:
            self._file_label.setStyleSheet(f"color:{theme.TEXT_SECONDARY};")

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Programmatic enable/disable (used at startup for the default-on)."""
        self._on_toggle(enabled)

    def append_line(self, line: str) -> None:
        """Slot: receives lines from ``AppSignals.log_line``.

        The bridge (``logging`` Handler) is wired by ``MainWindow`` *after* the
        user opts in; this method is a no-op until then.
        """
        if not self._enabled:
            return
        self._view.appendPlainText(line)
        bar = self._view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_toggle(self, checked: bool) -> None:
        self._enabled = checked
        for box in (self._enable_checkbox, self._active_checkbox):
            box.blockSignals(True)
            box.setChecked(checked)
            box.blockSignals(False)
        self._placeholder.setVisible(not checked)
        self._view.setVisible(checked)
        self._toolbar_widget.setVisible(checked)
        # Notify the owner (MainWindow) so it can wire/unwire the bridge.
        owner = self.window()
        if owner is not None and hasattr(owner, "_on_log_panel_toggled"):
            owner._on_log_panel_toggled(checked)

    def _copy_all(self) -> None:
        view = self._view
        view.selectAll()
        view.copy()
        cursor = view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        view.setTextCursor(cursor)
