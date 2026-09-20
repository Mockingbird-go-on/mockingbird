"""Resume panel — tab for importing the user's PDF resume.

The resume is processed through the LLM pipeline (``ResumeLoader``) and saved
as the ``resume`` KB topic, which powers first-person answers in personal
interview mode.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mockingbird.ui import theme
from mockingbird.ui import icons


class _ResumeImportThread(QThread):
    """Background thread for PDF resume processing."""

    progress = Signal(str, float)  # message, percent (-1 = indeterminate)
    finished_ok = Signal(dict)  # result dict
    failed = Signal(str)  # error message

    def __init__(self, app, pdf_path: str):
        super().__init__()
        self._app = app
        self._pdf_path = pdf_path

    def run(self) -> None:
        try:
            result = self._app.import_resume(
                self._pdf_path,
                on_progress=lambda msg, pct: self.progress.emit(msg, pct),
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class _IconButton(QPushButton):
    """Compact icon-only button with a tooltip."""

    def __init__(self, icon, tooltip: str, parent=None):
        super().__init__(parent)
        self.setIcon(icon)
        self.setToolTip(tooltip)
        self.setFixedSize(34, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class ResumePanel(QWidget):
    """Tab panel: PDF resume import for personal interview mode."""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self._app = app
        self._thread: _ResumeImportThread | None = None
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        group = QGroupBox("Резюме")
        group_layout = QVBoxLayout(group)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color:{theme.TEXT_SECONDARY};padding:4px;")
        group_layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._btn_load = _IconButton(
            icons.icon("folder-open"),
            "Загрузить PDF-резюме",
        )
        self._btn_remove = _IconButton(
            icons.icon("trash-2", color=theme.current.status_error),
            "Удалить резюме",
        )
        self._btn_load.clicked.connect(self._on_load)
        self._btn_remove.clicked.connect(self._on_remove)
        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_remove)
        btn_row.addStretch(1)
        group_layout.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        group_layout.addWidget(self._progress)

        hint = QLabel(
            "Резюме используется для personal-вопросов («что ты делал?»). "
            "Поддерживается PDF с текстовым слоем; сканы не обрабатываются."
        )
        hint.setStyleSheet(f"color:{theme.TEXT_SECONDARY};font-size:11px;")
        group_layout.addWidget(hint)

        layout.addWidget(group)
        layout.addStretch(1)

    def refresh(self) -> None:
        """Refresh resume status."""
        try:
            from mockingbird.kb.resume_loader import ResumeLoader

            info = ResumeLoader.get_info()
            if info:
                self._status.setText(
                    f"✅ Резюме загружено: {info['blocks']} блоков\n"
                    f"Обновлено: {info['modified']}"
                )
            else:
                self._status.setText(
                    "⬜ Резюме не загружено. Personal-вопросы работают в constructive-режиме."
                )
        except Exception:
            self._status.setText("Статус резюме недоступен.")

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите PDF-резюме", "", "PDF (*.pdf)")
        if not path:
            return
        llm = getattr(self._app, "llm", None)
        if llm is None or not getattr(llm, "available", False):
            QMessageBox.warning(
                self, "LLM недоступен",
                "Для обработки резюме нужен настроенный LLM.\n"
                "Проверьте Настройки → LLM (API ключ и URL).",
            )
            return
        self._btn_load.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        self._thread = _ResumeImportThread(self._app, path)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_ok.connect(self._on_done)
        self._thread.failed.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, msg: str, pct: float) -> None:
        self._status.setText(msg)
        if pct >= 0:
            self._progress.setValue(int(pct * 100))
        else:
            self._progress.setRange(0, 0)  # indeterminate

    def _on_done(self, result: dict) -> None:
        self._progress.setVisible(False)
        self._btn_load.setEnabled(True)
        QMessageBox.information(
            self, "Резюме загружено",
            f"Обработано блоков: {result['blocks']}. Резюме готово к использованию.",
        )
        self.refresh()

    def _on_error(self, error: str) -> None:
        self._progress.setVisible(False)
        self._btn_load.setEnabled(True)
        QMessageBox.warning(self, "Ошибка загрузки резюме", f"Не удалось обработать PDF:\n\n{error}")

    def _on_remove(self) -> None:
        reply = QMessageBox.question(self, "Удалить резюме", "Удалить загруженное резюме?")
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from mockingbird.kb.resume_loader import ResumeLoader

                ResumeLoader.remove()
            except Exception:
                pass
            self.refresh()

    def update_theme(self) -> None:
        self._status.setStyleSheet(f"color:{theme.TEXT_SECONDARY};padding:4px;")
        self._btn_load.setIcon(icons.icon("folder-open"))
        self._btn_remove.setIcon(
            icons.icon("trash-2", color=theme.current.status_error)
        )
        self.refresh()
