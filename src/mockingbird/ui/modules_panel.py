"""Modules panel — tab widget for KB module management and PDF resume import.

Displayed as the third tab in the main window ('Модули'), alongside
'Интервью' and 'Лог'.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from mockingbird.ui import theme


class _ResumeImportCancelled(Exception):
    """Raised inside the import pipeline when the user cancels the PDF import."""


class _ResumeImportThread(QThread):
    """Background thread for PDF resume processing (cooperative cancellation)."""

    progress = Signal(str, float)  # message, percent (-1 = indeterminate)
    finished_ok = Signal(dict)  # result dict
    failed = Signal(str)  # error message
    cancelled = Signal()  # user pressed Cancel

    def __init__(self, app, pdf_path: str):
        super().__init__()
        self._app = app
        self._pdf_path = pdf_path
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation; the import loop checks ``is_cancelled``."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def run(self) -> None:
        try:
            result = self._app.import_resume(
                self._pdf_path,
                on_progress=lambda msg, pct: self.progress.emit(msg, pct),
                cancel_check=self.is_cancelled,
            )
            if self.is_cancelled():
                self.cancelled.emit()
            else:
                self.finished_ok.emit(result)
        except _ResumeImportCancelled:
            self.cancelled.emit()
        except Exception as exc:
            if self.is_cancelled():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))


class ModulesPanel(QWidget):
    """Tab panel: KB module management + PDF resume import."""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self._app = app
        self._mgr = None
        self._resume_thread: _ResumeImportThread | None = None
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # === Section: KB Modules ===
        kb_group = QGroupBox("Модули базы знаний")
        kb_layout = QVBoxLayout(kb_group)

        # Splitter: list of modules on the left, details on the right.
        kb_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._module_list = QListWidget()
        self._module_list.setMinimumHeight(160)
        self._module_list.itemSelectionChanged.connect(self._on_module_selected)
        kb_splitter.addWidget(self._module_list)

        self._module_detail = QTextBrowser()
        self._module_detail.setOpenExternalLinks(False)
        self._module_detail.setPlaceholderText("Выберите модуль, чтобы увидеть описание.")
        self._module_detail.setMinimumWidth(280)
        kb_splitter.addWidget(self._module_detail)
        kb_splitter.setStretchFactor(0, 2)
        kb_splitter.setStretchFactor(1, 3)
        kb_splitter.setSizes([320, 480])
        kb_layout.addWidget(kb_splitter, stretch=1)

        kb_btn_row = QHBoxLayout()
        self._btn_import_kb = QPushButton("Импортировать ZIP…")
        self._btn_toggle_kb = QPushButton("Включить/выключить")
        self._btn_remove_kb = QPushButton("Удалить")
        self._btn_import_kb.clicked.connect(self._on_import_kb)
        self._btn_toggle_kb.clicked.connect(self._on_toggle_kb)
        self._btn_remove_kb.clicked.connect(self._on_remove_kb)
        kb_btn_row.addWidget(self._btn_import_kb)
        kb_btn_row.addWidget(self._btn_toggle_kb)
        kb_btn_row.addWidget(self._btn_remove_kb)
        kb_layout.addLayout(kb_btn_row)

        self._kb_status = QLabel("")
        self._kb_status.setStyleSheet(f"color:{theme.TEXT_SECONDARY};font-size:11px;")
        kb_layout.addWidget(self._kb_status)

        layout.addWidget(kb_group)

        # === Section: PDF Resume ===
        resume_group = QGroupBox("Резюме для personal-режима")
        resume_layout = QVBoxLayout(resume_group)

        self._resume_status = QLabel("")
        self._resume_status.setWordWrap(True)
        self._resume_status.setStyleSheet(f"color:{theme.TEXT_SECONDARY};padding:4px;")
        resume_layout.addWidget(self._resume_status)

        resume_btn_row = QHBoxLayout()
        self._btn_load_pdf = QPushButton("Загрузить PDF…")
        self._btn_remove_resume = QPushButton("Удалить резюме")
        self._btn_cancel_pdf = QPushButton("Отменить")
        self._btn_cancel_pdf.setVisible(False)
        self._btn_load_pdf.clicked.connect(self._on_load_pdf)
        self._btn_remove_resume.clicked.connect(self._on_remove_resume)
        self._btn_cancel_pdf.clicked.connect(self._on_cancel_pdf)
        resume_btn_row.addWidget(self._btn_load_pdf)
        resume_btn_row.addWidget(self._btn_cancel_pdf)
        resume_btn_row.addWidget(self._btn_remove_resume)
        resume_layout.addLayout(resume_btn_row)

        self._resume_progress = QProgressBar()
        self._resume_progress.setVisible(False)
        resume_layout.addWidget(self._resume_progress)

        hint = QLabel("Поддерживается PDF с текстовым слоем. Сканы не обрабатываются.")
        hint.setStyleSheet(f"color:{theme.TEXT_SECONDARY};font-size:11px;")
        resume_layout.addWidget(hint)

        layout.addWidget(resume_group)
        layout.addStretch(1)

    # -- KB module management --

    def refresh(self) -> None:
        """Refresh module list and resume status."""
        try:
            from mockingbird.kb.module_manager import ModuleManager

            self._mgr = ModuleManager()
        except Exception:
            self._mgr = None

        self._module_list.clear()
        self._module_manifests: dict[str, "ModuleManifest"] = {}
        if self._mgr is not None:
            modules = self._mgr.list_modules()
            for manifest, entry in modules:
                self._module_manifests[manifest.id] = manifest
                # Status dot: filled green when enabled, hollow grey when off.
                dot = "\u25cf" if entry.enabled else "\u25cb"  # ● / ○
                dot_color = theme.current.accent if entry.enabled else theme.TEXT_SECONDARY
                topic_word = "топик" if len(manifest.topics) == 1 else "топиков"
                spec = f" · {manifest.specialization}" if manifest.specialization else ""
                # Rich-text body. QListWidget renders item.text() through a
                # plain-text delegate, so we MUST NOT put HTML there (the user
                # would see raw ``<span …>``). Instead we install a QLabel
                # widget per item via setItemWidget() and keep item.text()
                # empty (the delegate then paints nothing behind the widget).
                from PySide6.QtWidgets import QLabel as _QLabel

                html = (
                    f"<span style='color:{dot_color};font-size:13px;'>{dot}</span>&nbsp;"
                    f"<b>{manifest.name}</b>"
                    f"<br/><span style='color:{theme.TEXT_SECONDARY};font-size:11px;'>"
                    f"v{manifest.version} · {len(manifest.topics)} {topic_word}{spec}</span>"
                )
                label = _QLabel(html)
                label.setWordWrap(True)
                label.setTextFormat(Qt.TextFormat.RichText)
                label.setStyleSheet("padding:6px 8px;")
                label.setToolTip(manifest.description or manifest.name)

                item = QListWidgetItem(self._module_list)
                # Leave item.text() empty so the default delegate does not draw
                # raw markup behind the label widget.
                item.setSizeHint(label.sizeHint())
                item.setToolTip(manifest.description or manifest.name)
                item.setData(Qt.ItemDataRole.UserRole, manifest.id)
                item.setData(Qt.ItemDataRole.UserRole + 1, entry.enabled)
                self._module_list.addItem(item)
                self._module_list.setItemWidget(item, label)
            if not modules:
                QListWidgetItem("Нет установленных модулей. Импортируйте ZIP-файл.", self._module_list)
            total = sum(
                len(m.topics) for m, e in modules if e.enabled
            )
            self._kb_status.setText(f"Активно: {total} топиков из {len([e for _, e in modules if e.enabled])} модуля(ей)")
        else:
            QListWidgetItem("Менеджер модулей недоступен.", self._module_list)

        self._refresh_resume_status()

    def _refresh_resume_status(self) -> None:
        try:
            from mockingbird.kb.resume_loader import ResumeLoader

            info = ResumeLoader.get_info()
            if info:
                self._resume_status.setText(
                    f"✅ Резюме загружено: {info['blocks']} блоков\n"
                    f"Обновлено: {info['modified']}"
                )
            else:
                self._resume_status.setText(
                    "⬜ Резюме не загружено. Personal-вопросы работают в constructive-режиме."
                )
        except Exception:
            self._resume_status.setText("Статус резюме недоступен.")

    def _on_import_kb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Импорт модуля KB", "", "KB Module (*.zip)")
        if not path or self._mgr is None:
            return
        manifest = self._mgr.install_zip(path)
        if manifest:
            spec = f" · {manifest.specialization}" if manifest.specialization else ""
            author = f"\nАвтор: {manifest.author}" if manifest.author else ""
            desc = f"\n\n{manifest.description}" if manifest.description else ""
            QMessageBox.information(
                self, "Модуль установлен",
                f"{manifest.name} v{manifest.version}{spec}\n"
                f"{len(manifest.topics)} топиков базы знаний.{author}{desc}",
            )
            self.refresh()
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось установить модуль. Проверьте формат ZIP.")

    def _on_module_selected(self) -> None:
        """Show the selected module's manifest details on the right pane."""
        items = self._module_list.selectedItems()
        if not items:
            return
        mod_id = items[0].data(Qt.ItemDataRole.UserRole)
        manifest = getattr(self, "_module_manifests", {}).get(mod_id)
        if manifest is None:
            self._module_detail.clear()
            return
        dot = "\u25cf" if items[0].data(Qt.ItemDataRole.UserRole + 1) else "\u25cb"
        status = "включён" if items[0].data(Qt.ItemDataRole.UserRole + 1) else "выключен"
        desc = manifest.description or "Описание отсутствует — добавьте поле description в manifest.yaml модуля."
        author = manifest.author or "—"
        spec = manifest.specialization or "—"
        html = (
            f"<h3>{manifest.name}</h3>"
            f"<p><b>Статус:</b> {dot} {status} · <b>Версия:</b> {manifest.version}</p>"
            f"<p><b>Специализация:</b> {spec}<br/>"
            f"<b>Автор:</b> {author}<br/>"
            f"<b>Топиков в модуле:</b> {len(manifest.topics)}<br/>"
            f"<b>Минимальная версия приложения:</b> {manifest.min_app_version}</p>"
            f"<hr/><p>{desc}</p>"
        )
        self._module_detail.setHtml(html)

    def _on_toggle_kb(self) -> None:
        item = self._module_list.currentItem()
        if item is None or self._mgr is None:
            return
        mod_id = item.data(Qt.ItemDataRole.UserRole)
        if not mod_id:
            return
        current = item.data(Qt.ItemDataRole.UserRole + 1)
        self._mgr.set_enabled(mod_id, not current)
        self.refresh()

    def _on_remove_kb(self) -> None:
        item = self._module_list.currentItem()
        if item is None or self._mgr is None:
            return
        mod_id = item.data(Qt.ItemDataRole.UserRole)
        if not mod_id:
            return
        reply = QMessageBox.question(self, "Удалить модуль", f"Удалить модуль «{mod_id}»?")
        if reply == QMessageBox.StandardButton.Yes:
            self._mgr.remove(mod_id)
            self.refresh()

    # -- PDF resume --

    def _on_load_pdf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите PDF резюме", "", "PDF (*.pdf)")
        if not path:
            return
        self._btn_load_pdf.setEnabled(False)
        self._btn_cancel_pdf.setVisible(True)
        self._resume_progress.setVisible(True)
        self._resume_progress.setRange(0, 100)

        self._resume_thread = _ResumeImportThread(self._app, path)
        self._resume_thread.progress.connect(self._on_resume_progress)
        self._resume_thread.finished_ok.connect(self._on_resume_done)
        self._resume_thread.failed.connect(self._on_resume_error)
        self._resume_thread.cancelled.connect(self._on_resume_cancelled)
        self._resume_thread.start()

    def _on_cancel_pdf(self) -> None:
        """Cooperative cancel: signal the thread, never terminate() it.

        ``QThread.terminate()`` kills the thread at an arbitrary instruction,
        risking half-written YAML or dangling HTTP/LLM connections. Instead we
        set the cancel event; the import loop polls it and raises
        ``_ResumeImportCancelled`` at a safe checkpoint.
        """
        if self._resume_thread is not None and self._resume_thread.isRunning():
            self._resume_thread.cancel()
            self._btn_cancel_pdf.setEnabled(False)
            self._resume_status.setText("Отменяю…")

    def _on_resume_cancelled(self) -> None:
        self._resume_progress.setVisible(False)
        self._btn_load_pdf.setEnabled(True)
        self._btn_cancel_pdf.setVisible(False)
        self._btn_cancel_pdf.setEnabled(True)
        self._resume_status.setText("Загрузка отменена.")

    def _on_resume_progress(self, msg: str, pct: float) -> None:
        self._resume_status.setText(msg)
        if pct >= 0:
            self._resume_progress.setValue(int(pct * 100))
        else:
            self._resume_progress.setRange(0, 0)  # indeterminate

    def _on_resume_done(self, result: dict) -> None:
        self._resume_progress.setVisible(False)
        self._btn_load_pdf.setEnabled(True)
        self._btn_cancel_pdf.setVisible(False)
        QMessageBox.information(
            self, "Резюме загружено",
            f"Обработано блоков: {result['blocks']}. Резюме готово к использованию.",
        )
        self.refresh()

    def _on_resume_error(self, error: str) -> None:
        self._resume_progress.setVisible(False)
        self._btn_load_pdf.setEnabled(True)
        self._btn_cancel_pdf.setVisible(False)
        QMessageBox.warning(self, "Ошибка загрузки резюме", f"Не удалось обработать PDF:\n\n{error}")

    def _on_remove_resume(self) -> None:
        reply = QMessageBox.question(self, "Удалить резюме", "Удалить загруженное резюме?")
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from mockingbird.kb.resume_loader import ResumeLoader
                ResumeLoader.remove()
            except Exception:
                pass
            self.refresh()

    def update_theme(self) -> None:
        self._kb_status.setStyleSheet(f"color:{theme.TEXT_SECONDARY};font-size:11px;")
        self._resume_status.setStyleSheet(f"color:{theme.TEXT_SECONDARY};padding:4px;")
        self.refresh()
