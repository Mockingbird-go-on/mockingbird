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
from mockingbird.i18n import t


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

        group = QGroupBox(t("Резюме"))
        self._group = group
        # Opaque card background like the Interview/Log tabs: the window's
        # background image otherwise bleeds through and makes the hint
        # labels hard to read.
        group.setStyleSheet(
            f"QGroupBox {{ background-color: {theme.current.surface};"
            " border: 1px solid palette(mid); border-radius: 4px;"
            " margin-top: 12px; padding: 8px 6px 6px 6px; }}"
            "QGroupBox::title { subcontrol-origin: margin;"
            " left: 10px; padding: 0 4px; }"
        )
        group_layout = QVBoxLayout(group)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color:{theme.TEXT_SECONDARY};padding:4px;")
        # Native status icon in front of the text (replaces ✅/⬜ emoji).
        self._status_icon = QLabel("")
        self._status_icon.setFixedSize(18, 18)
        self._status_icon.setVisible(False)
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.addWidget(self._status_icon)
        status_row.addWidget(self._status, stretch=1)
        group_layout.addLayout(status_row)

        btn_row = QHBoxLayout()
        self._btn_load = _IconButton(
            icons.icon("folder-open"),
            t("Загрузить PDF-резюме"),
        )
        self._btn_remove = _IconButton(
            icons.icon("trash-2", color=theme.current.status_error),
            t("Удалить резюме"),
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
            t("Резюме используется для personal-вопросов («что ты делал?»). Поддерживается PDF с текстовым слоем; сканы не обрабатываются.")
        )
        self._hint = hint
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
                self._set_status(
                    t("Резюме загружено: {n} блоков\nОбновлено: {m}", n=info['blocks'], m=info['modified']),
                    ok=True,
                )
            else:
                self._set_status(
                    t("Резюме не загружено. Personal-вопросы работают в constructive-режиме."),
                    ok=False,
                )
        except Exception:
            self._set_status(t("Статус резюме недоступен."), ok=None)

    def _set_status(self, message: str, ok: bool | None) -> None:
        """Status text + native Lucide icon (ok=True check / False empty box)."""
        self._status.setText(message)
        self._status.setStyleSheet(
            f"color:{theme.current.text if ok else theme.TEXT_SECONDARY};padding:4px;"
        )
        if ok is None:
            self._status_icon.setVisible(False)
            return
        name, color = (
            ("circle-check", theme.current.status_running) if ok
            else ("circle-alert", theme.current.status_muted)
        )
        self._status_icon.setPixmap(icons.icon(name, size=18, color=color).pixmap(18, 18))
        self._status_icon.setVisible(True)

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, t("Выберите PDF-резюме"), "", "PDF (*.pdf)")
        if not path:
            return
        llm = getattr(self._app, "llm", None)
        if llm is None or not getattr(llm, "available", False):
            QMessageBox.warning(
                self, t("LLM недоступен"),
                t("Для обработки резюме нужен настроенный LLM.\nПроверьте Настройки → LLM (API ключ и URL)."),
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
            self,
            t("Резюме загружено"),
            t("Обработано блоков: {n}. Резюме готово к использованию.", n=result['blocks']),
        )
        self.refresh()

    def _on_error(self, error: str) -> None:
        self._progress.setVisible(False)
        self._btn_load.setEnabled(True)
        QMessageBox.warning(self, t("Ошибка загрузки резюме"), t("Не удалось обработать PDF:\n\n{error}", error=error))

    def _on_remove(self) -> None:
        reply = QMessageBox.question(self, t("Удалить резюме"), t("Удалить загруженное резюме?"))
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from mockingbird.kb.resume_loader import ResumeLoader

                ResumeLoader.remove()
            except Exception:
                pass
            self.refresh()

    def update_theme(self) -> None:
        # Group-box card + hint label were styled inline at build time —
        # re-apply with the current theme or the old colors linger.
        self._group.setStyleSheet(
            f"QGroupBox {{ background-color: {theme.current.surface};"
            " border: 1px solid palette(mid); border-radius: 4px;"
            " margin-top: 12px; padding: 8px 6px 6px 6px; }}"
            "QGroupBox::title { subcontrol-origin: margin;"
            " left: 10px; padding: 0 4px; }"
        )
        self._hint.setStyleSheet(f"color:{theme.TEXT_SECONDARY};font-size:11px;")
        self._status.setStyleSheet(f"color:{theme.TEXT_SECONDARY};padding:4px;")
        self._btn_load.setIcon(icons.icon("folder-open"))
        self._btn_remove.setIcon(
            icons.icon("trash-2", color=theme.current.status_error)
        )
        self.refresh()
