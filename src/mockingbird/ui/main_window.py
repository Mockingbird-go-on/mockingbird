"""Main application window: large terms view, controls, mic level, interview cockpit."""
from __future__ import annotations

import logging

from PySide6.QtCore import QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mockingbird.app import App
from mockingbird.ui.interview_panel import InterviewPanel
from mockingbird.ui.log_panel import LogPanel
from mockingbird.ui.modules_panel import ResumePanel
from mockingbird.ui.settings_dialog import SettingsDialog
from mockingbird.ui import theme
from mockingbird.ui.icons import icon as lucide_icon
from mockingbird.ui.widgets import (
    ActivityBar,
    BackgroundWidget,
    DeviceBadge,
    LogoBadge,
    SourceBadge,
    StatusPill,
)

log = logging.getLogger(__name__)


def _format_elapsed(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class MainWindow(QMainWindow):
    # Bridge: session-stop worker thread → GUI thread completion callback.
    # A plain Signal on the instance-level won't marshal; class-level Signals
    # with a bound-slot connection deliver via the owning (GUI) thread.
    _stop_done = Signal()

    def __init__(self, app: App):
        super().__init__()
        self._app = app
        self._sig = app.signals
        from mockingbird import __version__

        self.setWindowTitle(f"Mockingbird {__version__}")
        self.resize(app.config.window.width, app.config.window.height)

        self._settings = QSettings("Mockingbird", "Mockingbird")
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            if not self.restoreGeometry(geometry):
                self.resize(app.config.window.width, app.config.window.height)

        self._session_seconds = 0
        # Lazy download overlay — created on the first download progress
        # event (see _on_model_load_progress).
        self._model_dl = None
        # Vision capability of the current LLM: True/False after the probe,
        # None before it (dialog shows «проверка…»).
        self._vision_state: bool | None = None
        # The question of the screenshot stream in flight — used for the
        # history entry (the engine's pending query may belong to a voice
        # question that arrived meanwhile).
        self._pending_screenshot_question: str = ""
        self._session_timer = QTimer(self)
        self._session_timer.setInterval(1000)
        self._session_timer.timeout.connect(self._tick_session)
        self._log_label = QLabel(self._log_path_text())
        self._log_label.setStyleSheet(f"color:{theme.TEXT_SECONDARY};")
        self.statusBar().addPermanentWidget(self._log_label)

        self._bg = BackgroundWidget()
        central = self._bg
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        self._tabs = QTabWidget()
        self._interview = InterviewPanel(
            resolve=self._app.kb_matcher.resolve,
            answer_query=self._app.interview.answer_query,
            regenerate_callback=self._app.interview.regenerate_answer,
            concept_callback=self._app.interview.ask_concept,
            llm_primary=self._app.config.interview.llm_primary,
            llm_available=self._app.llm.available,
            llm_busy=lambda: bool(getattr(self._app.llm, "is_streaming", False)),
        )
        self._tabs.addTab(self._interview, "Интервью")
        self._modules_panel = ResumePanel(self._app)
        self._tabs.addTab(self._modules_panel, "Резюме")
        self._log_panel = LogPanel(log_file=self._app.config.storage.log_dir)
        self._tabs.addTab(self._log_panel, "Лог")
        self._toolbar = self._build_toolbar()
        layout.addWidget(self._toolbar)
        layout.addWidget(self._tabs, stretch=1)
        self.setCentralWidget(central)

        self._connect_signals()
        # Logging is on by default; the user can turn it off from the tab.
        # Must run AFTER _connect_signals: set_enabled notifies this window
        # (self.window()) — before the window is constructed the notification
        # would be lost and the log handler never installed.
        self._log_panel.set_enabled(True)
        self._set_running(False)
        self._sig.status.emit("idle", "")

    def closeEvent(self, event) -> None:
        self._session_timer.stop()
        self._settings.setValue("window/geometry", self.saveGeometry())
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Apply capture affinity once the window has a valid HWND.
        self._sync_capture_button()
        self._apply_capture_affinity()

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("toolbar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        self._start_btn = QPushButton()
        self._start_btn.setIcon(lucide_icon("play", color=theme.LIGHT))
        self._start_btn.setIconSize(QSize(18, 18))
        self._start_btn.setProperty("primary", True)
        self._start_btn.setToolTip("Старт")
        self._stop_btn = QPushButton()
        self._stop_btn.setIcon(lucide_icon("square", color=theme.current.status_error))
        self._stop_btn.setIconSize(QSize(18, 18))
        self._stop_btn.setToolTip("Стоп")
        self._mute_btn = QPushButton()
        self._mute_btn.setIcon(self._mute_icon(False))
        self._mute_btn.setIconSize(QSize(18, 18))
        self._mute_btn.setToolTip("Мьют")
        self._settings_btn = QPushButton()
        self._settings_btn.setIcon(lucide_icon("settings-2"))
        self._settings_btn.setIconSize(QSize(18, 18))
        self._settings_btn.setToolTip("Настройки")
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        self._mute_btn.clicked.connect(self._on_toggle_mute)
        self._settings_btn.clicked.connect(self._on_settings)
        self._status = StatusPill()
        self._device_badge = DeviceBadge()
        self._source_badge = SourceBadge()
        self._activity = ActivityBar()
        # Small Cancel affordance shown next to the loader while the model is
        # loading into memory (the download overlay covers the download phase,
        # but the in-memory phase used to have no way out).
        self._cancel_load_btn = QPushButton()
        self._cancel_load_btn.setIcon(
            lucide_icon("circle-x", color=theme.current.text_secondary)
        )
        self._cancel_load_btn.setIconSize(QSize(14, 14))
        self._cancel_load_btn.setFixedSize(20, 20)
        self._cancel_load_btn.setFlat(True)
        self._cancel_load_btn.setToolTip("Отменить загрузку модели")
        self._cancel_load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_load_btn.clicked.connect(self._on_cancel_model_load)
        self._cancel_load_btn.hide()
        # Screenshot-to-answer: toolbar button (global hotkey is Windows-only).
        self._shot_btn = QPushButton()
        self._shot_btn.setIcon(lucide_icon("camera"))
        self._shot_btn.setIconSize(QSize(18, 18))
        self._shot_btn.setToolTip("Скриншот-вопрос (Ctrl+Shift+S)\nВыделите область экрана и задайте вопрос")
        self._shot_btn.clicked.connect(self._on_screenshot)
        if not getattr(self._app.config, "screenshot", None) or not self._app.config.screenshot.enabled:
            self._shot_btn.hide()
        self._timer_label = QLabel("00:00")
        self._timer_label.setObjectName("sessionTimer")
        self._timer_label.setStyleSheet(
            f"color:{theme.TEXT_SECONDARY};font-weight:bold;"
        )
        self._logo = LogoBadge()
        layout.addWidget(self._start_btn)
        layout.addWidget(self._stop_btn)
        layout.addWidget(self._mute_btn)
        layout.addSpacing(12)
        layout.addWidget(self._activity)
        layout.addWidget(self._cancel_load_btn)
        layout.addWidget(self._timer_label)
        layout.addStretch(1)
        layout.addWidget(self._shot_btn)
        layout.addWidget(self._device_badge)
        layout.addWidget(self._status)
        layout.addWidget(self._settings_btn)
        layout.addWidget(self._logo)
        return bar

    # -- capture guard -----------------------------------------------------

    def _sync_capture_button(self) -> None:
        """No-op placeholder — capture toggle moved to Settings dialog."""
        pass

    def _on_toggle_capture(self) -> None:
        hidden = self._app.config.window.hide_from_capture
        self._app.config.window.hide_from_capture = not hidden
        self._app.save_settings()
        self._apply_capture_affinity()

    def _on_toggle_capture_via_signal(self) -> None:
        """Bridge: called from the global-hotkey thread via Qt signal."""
        self._on_toggle_capture()

    def _apply_capture_affinity(self) -> None:
        """Apply or remove capture exclusion on this window + visible dialogs."""
        from mockingbird.ui import capture_guard

        if not capture_guard.is_supported():
            return
        enabled = self._app.config.window.hide_from_capture
        hwnd = int(self.winId())
        if enabled:
            capture_guard.set_exclude_from_capture(hwnd)
        else:
            capture_guard.clear(hwnd)
        for w in QApplication.topLevelWidgets():
            if w is not self and w.isWindow() and w.isVisible():
                wh = int(w.winId())
                if enabled:
                    capture_guard.set_exclude_from_capture(wh)
                else:
                    capture_guard.clear(wh)

    # -- icon helpers --

    @staticmethod
    def _mute_icon(muted: bool):
        """Volume / volume-off Lucide icon; muted renders in the secondary colour."""
        if muted:
            return lucide_icon("volume-x", color=theme.current.text_secondary)
        return lucide_icon("volume-2")

    def _refresh_toolbar_icons(self) -> None:
        """Re-render toolbar icons in the colours of the active theme."""
        self._start_btn.setIcon(lucide_icon("play", color=theme.LIGHT))
        self._stop_btn.setIcon(
            lucide_icon("square", color=theme.current.status_error)
        )
        self._mute_btn.setIcon(self._mute_icon(self._app.muted))
        self._settings_btn.setIcon(lucide_icon("settings-2"))

    def _apply_theme(self, name: str) -> None:
        theme.apply_theme(QApplication.instance(), name)
        self._settings.setValue("ui/theme", name)
        self._bg.theme_changed()
        self._status.update_theme()
        self._device_badge.update_theme()
        self._source_badge.update_theme()
        self._logo.update_theme()
        self._activity.update_theme()
        self._timer_label.setStyleSheet(
            f"color:{theme.TEXT_SECONDARY};font-weight:bold;"
        )
        self._log_label.setStyleSheet(f"color:{theme.TEXT_SECONDARY};")
        self._interview.retheme()
        self._log_panel.update_theme()
        self._modules_panel.update_theme()
        self._refresh_toolbar_icons()

    def _connect_signals(self) -> None:
        self._sig.partial.connect(self._interview.on_partial)
        self._sig.question.connect(self._interview.on_question)
        self._sig.final.connect(self._interview.on_final)
        self._sig.answer.connect(self._interview.on_answer)
        self._sig.llm_answer.connect(self._interview.on_llm_answer)
        self._sig.context.connect(self._interview.on_context)
        self._sig.mic_level.connect(self._activity.set_level)
        self._sig.status.connect(self._on_status)
        self._sig.device.connect(self._device_badge.set_device)
        self._sig.source.connect(self._source_badge.set_source)
        self._sig.speech.connect(self._activity.flash_speech)
        self._sig.error.connect(self._on_error)
        self._sig.cuda_fallback.connect(self._on_cuda_fallback)
        # log_line is wired lazily — see _on_log_panel_toggled.
        self._sig.model_load.connect(self._on_model_load_progress)
        self._sig.model_load_failed.connect(self._on_model_load_failed)
        self._sig.model_load_cancelled.connect(self._on_model_load_cancelled)
        # Async session stop completion (marshalled from the stop worker).
        self._stop_done.connect(self._on_stop_done)
        # Screenshot-to-answer.
        self._sig.screenshot_answer_done.connect(self._on_screenshot_answer_done)
        self._sig.vision_probe_result.connect(self._on_vision_probe_result)
        self._sig.screenshot_request.connect(self._on_screenshot)

    def _on_vision_probe_result(self, ok) -> None:
        self._vision_state = bool(ok)
        dlg = getattr(self, "_shot_dlg", None)
        if dlg is not None and dlg.isVisible():
            dlg._set_vision(self._vision_state)

    def _on_start(self) -> None:
        try:
            self._app.start_session()
        except Exception as exc:  # noqa: BLE001
            log.exception("start failed")
            # Явное модальное уведомление: статусбар легко не заметить.
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Не удалось начать сессию",
                str(exc),
            )
            self._sig.error.emit(str(exc))
            return
        self._set_running(True)
        self._log_label.setText(self._log_path_text(session=self._app.session_id))

    def _on_stop(self) -> None:
        """Stop the session without freezing the GUI.

        ``stop_session`` can block up to ~8 s on the engine join, so it runs
        in a background thread; the buttons stay disabled ("stopping") until
        the completion signal fires on the GUI thread. The signal is wired
        once in ``_connect_signals`` — no per-click connect accumulation.
        """
        self._set_stopping(True)
        self._app.stop_session_async(on_done=self._stop_done.emit)

    def _on_stop_done(self) -> None:
        """GUI-thread slot: session teardown finished (success or failure)."""
        self._session_timer.stop()
        # Reset the elapsed counter so the next session starts from 00:00
        # rather than accumulating across sessions.
        self._session_seconds = 0
        self._timer_label.setText("00:00")
        self._set_running(False)
        self._log_label.setText(self._log_path_text())

    # -- model download overlay ------------------------------------------------

    def _on_model_load_progress(self, message: str, percent: float) -> None:
        """Route model_load progress to the download overlay.

        The dialog is created lazily on the FIRST download progress event
        (percent >= 0 or a "Downloading…" message). The in-memory phase
        ("Loading model into memory…", percent -1) does NOT spawn the overlay
        — the model is already on disk — but it DOES show the small Cancel
        cross next to the toolbar loader so the user can still abort.
        """
        if not message and percent >= 100.0:
            # Model ready: nothing is cancellable any more.
            self._set_cancel_load_visible(False)
            # Reset the toolbar loader (the direct model_load→set_loading
            # connect is gone; without this the spinner ran forever after
            # the model became ready).
            if self._app.session_id is not None:
                self._activity.set_live()
            else:
                self._activity.set_idle()
            if self._model_dl is not None and not self._model_dl._finalized:
                self._model_dl.done_ok()
            return
        downloading = percent >= 0 or "download" in message.lower() or "скачиван" in message.lower()
        if not downloading:
            # In-memory load: keep the cancel cross available.
            self._set_cancel_load_visible(True)
            if self._model_dl is not None and not self._model_dl._finalized:
                self._model_dl.done_ok()
            # No overlay is open for this phase — show the toolbar loader.
            if self._model_dl is None or not self._model_dl.isVisible():
                self._activity.set_loading(message, percent)
            return
        # Download phase: the dedicated overlay owns the progress display;
        # the toolbar loader is suppressed so the user is not ping-ponged
        # between two busy indicators.
        self._activity.set_idle()
        if self._model_dl is None:
            from mockingbird.ui.model_download_dialog import ModelDownloadDialog

            self._model_dl = ModelDownloadDialog(parent=None)
            self._model_dl.cancelled.connect(self._on_model_dl_cancel)
            self._model_dl.set_model_name(
                f"Модель: {self._app.config.whisper.model_size}"
            )
            self._model_dl.show_above(self)
        self._model_dl.set_progress(message, percent)

    def _set_cancel_load_visible(self, visible: bool) -> None:
        self._cancel_load_btn.setVisible(visible)
        if not visible:
            # Reset any "Отмена…" latched state so the next load starts clean.
            self._cancel_load_btn.setEnabled(True)
            self._cancel_load_btn.setToolTip("Отменить загрузку модели")

    def _on_cancel_model_load(self) -> None:
        """Small cross in the toolbar: cancel the in-flight model load."""
        self._cancel_load_btn.setEnabled(False)
        self._cancel_load_btn.setToolTip("Отмена…")
        self._app.cancel_model_download()

    def _on_model_load_cancelled(self) -> None:
        """User cancelled the load: reset the affordances without an error."""
        self._set_cancel_load_visible(False)
        if self._model_dl is not None:
            self._model_dl._cancelled_confirmed()  # idempotent, also when hidden
        self._activity.set_idle()

    def _on_model_load_failed(self, error: str) -> None:
        """All download attempts failed (or cancelled): close the overlay and
        offer retry / offline hint. Goes through the NotificationBus FIFO —
        never a bare QMessageBox (overlap chaos, 2026-09-26)."""
        if self._model_dl is not None:
            self._model_dl.done_failed(error)
        self._on_model_load_cancelled()

        low = error.lower()
        if "cancelled" in low:
            return  # user cancelled deliberately — no nagging
        if "certificate" in low:
            text = (
                "Не удалось скачать модель: соединение блокируется "
                "прокси-сервером или антивирусом (подмена SSL-сертификата).\n\n"
                "Варианты:\n"
                "• Попросить IT добавить huggingface.co в исключения SSL-инспекции\n"
                "• Перенести папку модели с другой машины в\n"
                f"  {self._app.config.whisper.model_dir or '~/.mockingbird/models'}"
            )
        else:
            text = f"Не удалось скачать модель распознавания:\n{error}\n\nПроверьте интернет-соединение."
        from mockingbird.ui.notify import bus as notify_bus

        notify_bus.error(
            "Загрузка модели",
            text,
            buttons=(("Повторить", self._app.retry_model_download), ("Закрыть", None)),
            dedup_key="model-load-failed",
        )

    def _on_model_dl_cancel(self) -> None:
        self._app.cancel_model_download()

    def _set_stopping(self, stopping: bool) -> None:
        """Intermediate state while the session is being torn down."""
        self._start_btn.setEnabled(not stopping)
        self._stop_btn.setEnabled(not stopping)
        self._mute_btn.setEnabled(not stopping)
        if stopping:
            self._status.set_state("loading", "остановка…")

    def _on_status(self, state: str, detail: str) -> None:
        self._status.set_state(state, detail)
        if state == "loading":
            self._activity.set_loading(detail or "Загрузка модели…", -1)
            # "stopping" is a teardown, not a model load — never offer Cancel.
            self._set_cancel_load_visible(detail != "stopping")
        else:
            # Any non-loading state means nothing is being loaded any more.
            self._set_cancel_load_visible(False)
            if state == "running":
                self._activity.set_live()
                # Start the session timer only when the model is actually ready.
                if not self._session_timer.isActive():
                    self._session_timer.start()
            elif state == "muted":
                self._activity.set_muted()
            elif state == "idle":
                self._activity.set_idle()
            elif state == "error":
                self._activity.set_error(detail or "Ошибка")

    def _tick_session(self) -> None:
        self._session_seconds += 1
        self._timer_label.setText(_format_elapsed(self._session_seconds))

    def _log_path_text(self, session: str | None = None) -> str:
        path = getattr(self._app.config.storage, "log_dir", "")
        prefix = f"Сессия #{session} · " if session else ""
        return f"{prefix}Лог: {path}"

    def _on_toggle_mute(self) -> None:
        self._app.toggle_mute()
        muted = self._app.muted
        self._mute_btn.setIcon(self._mute_icon(muted))
        self._mute_btn.setToolTip("Снять мьют" if muted else "Мьют")
        self._sig.status.emit("muted" if muted else "running", "")

    def _on_settings(self) -> None:
        dialog = SettingsDialog(self._app.config, self)
        if dialog.exec():
            prev_profile = self._app.config.profile_id
            dialog.apply()
            # The dialog swaps the global palette itself; re-apply through
            # MainWindow so every widget re-reads theme colours (labels with
            # baked stylesheets, toolbar icons, panels' update_theme).
            # NB: _theme_choice is set inside apply() — read it AFTER the call.
            theme_choice = getattr(dialog, "_theme_choice", None)
            if isinstance(theme_choice, str):
                self._apply_theme(theme_choice)
            if self._app.config.profile_id != prev_profile:
                try:
                    self._app.apply_profile(self._app.config.profile_id)
                except Exception:
                    log.exception("apply_profile failed")
            self._app.save_settings()
            self._apply_capture_affinity()
            if dialog.restart_required:
                self._prompt_restart()

    def _on_log_panel_toggled(self, enabled: bool) -> None:
        """Bridge: user toggled logging in the LogPanel.

        When enabling, attach a fresh ``QtLogHandler`` to the root logger and
        start forwarding ``log_line`` signals to the panel. When disabling,
        reverse both — the panel keeps zero overhead while hidden.
        """
        if enabled:
            try:
                self._app.install_log_handler()
            except Exception:
                log.exception("install_log_handler failed")
            try:
                self._sig.log_line.connect(self._log_panel.append_line)
            except (RuntimeError, TypeError):
                pass  # already connected
        else:
            try:
                self._sig.log_line.disconnect(self._log_panel.append_line)
            except (RuntimeError, TypeError):
                pass
            try:
                self._app.remove_log_handler()
            except Exception:
                log.exception("remove_log_handler failed")

    def _prompt_restart(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Перезапуск")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("Часть изменений вступит в силу после перезапуска приложения.")
        restart = box.addButton(
            "Перезапустить сейчас", QMessageBox.ButtonRole.AcceptRole
        )
        later = box.addButton("Позже", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(later)
        box.exec()
        if box.clickedButton() is restart:
            self._restart_app()

    def _restart_app(self) -> None:
        import os
        import subprocess
        import sys

        from PySide6.QtWidgets import QApplication

        if getattr(sys, "frozen", False):
            # Linux AppImage: sys.executable points INTO the mounted squashfs
            # (/tmp/.mount_*), which is unmounted on exit — relaunch the
            # original .AppImage path instead.
            appimage = os.environ.get("APPIMAGE")
            argv = [appimage] if appimage else [sys.executable]
        else:
            argv = [sys.executable, "-m", "mockingbird"]
        # Tear the session down BEFORE spawning the new process: the old
        # instance holds the SQLite DB and GPU memory, and a fresh copy
        # starting concurrently risks "database is locked" and VRAM conflicts.
        # shutdown() (aboutToQuit) performs the full synchronous teardown —
        # here we only kick it off asynchronously to avoid freezing the GUI
        # on an engine join for up to ~8 s.
        self._app.stop_session_async()
        # Flush QSettings so window geometry survives the restart.
        self._settings.sync()
        try:
            subprocess.Popen(argv, cwd=os.getcwd())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(None, "Ошибка", f"Не удалось перезапустить приложение: {exc}")
            return
        QApplication.quit()

    def _on_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)
        self._set_cancel_load_visible(False)
        self._activity.set_error(message)
        self._sig.status.emit("error", "")
        # If the session never actually started (engine load failed, capture
        # error), return the buttons to the idle state — otherwise Start stays
        # disabled forever with no live session behind it.
        if self._app.session_id is None:
            self._set_running(False)

    # -- screenshot-to-answer ---------------------------------------------------

    def request_screenshot(self) -> None:
        """Programmatic entry (global hotkey bridge / tests)."""
        self._on_screenshot()

    def _on_screenshot(self) -> None:
        cfg = getattr(self._app.config, "screenshot", None)
        if cfg is None or not cfg.enabled:
            return
        from mockingbird.ui.screenshot import ScreenGrabOverlay

        self._shot_overlay = ScreenGrabOverlay()
        self._shot_overlay.finished.connect(self._on_region_selected)
        self._shot_overlay.cancelled.connect(lambda: setattr(self, "_shot_overlay", None))
        self._shot_overlay.start()

    def _on_region_selected(self, rect) -> None:
        from PySide6.QtCore import QRect

        cfg = self._app.config.screenshot
        from mockingbird.ui.screenshot import ScreenshotQuestionDialog, grab_screen_region

        try:
            jpeg, preview = grab_screen_region(
                QRect(rect), max_dim=cfg.max_image_dim, jpeg_quality=cfg.jpeg_quality
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("screenshot grab failed: %s", exc)
            self.statusBar().showMessage(f"Скриншот не получен: {exc}", 5000)
            return
        # Close (and let Qt delete) any previous dialog before opening a new
        # one — the overwritten reference used to orphan a visible dialog
        # that could then be garbage-collected on screen.
        old = getattr(self, "_shot_dlg", None)
        if old is not None:
            old.close()
            old.deleteLater()
        self._shot_overlay = None
        dlg = ScreenshotQuestionDialog(jpeg, preview, vision_ok=self._vision_state)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        dlg.asked.connect(self._on_screenshot_question)
        self._shot_dlg = dlg
        dlg.center_on(self)
        # Vision probe lazily on first use (and again if it failed before).
        if self._vision_state is None or self._vision_state is False:
            self._app.check_vision_async()

    def _on_screenshot_question(self, question: str) -> None:
        dlg = getattr(self, "_shot_dlg", None)
        if dlg is None:
            return
        # Prepare the answer pane BEFORE the (queue-serialized) stream starts:
        # latches the pending query, resets any previous stream buffer, shows
        # the placeholder and starts the 15 s watchdog (a hung image stream
        # previously left the old answer on screen with no timeout at all).
        import uuid as _uuid

        stream_id = f"shot-{_uuid.uuid4().hex[:12]}"
        self._interview.begin_external_stream(question, stream_id)
        self._pending_screenshot_question = question
        self._app.answer_screenshot(dlg.jpeg_bytes(), question, stream_id=stream_id)

    def _on_screenshot_answer_done(self, shot_id: str) -> None:
        dlg = getattr(self, "_shot_dlg", None)
        if dlg is not None and dlg.isVisible():
            dlg.set_busy(False, "Ответ — в панели «Ответ ИИ»")
            dlg.close()
        # History entry: the query is the screenshot question (NOT the
        # engine's _pending_llm_query — a voice question may have arrived
        # meanwhile and would be mislabeled as a screenshot).
        query = getattr(self, "_pending_screenshot_question", "") or ""
        if query:
            self._interview._history.add_entry(f"📸 {query}", "screenshot")
            self._pending_screenshot_question = ""

    def _on_cuda_fallback(self, detail: str) -> None:
        """Configured CUDA turned out unusable; the engine already reloaded
        on CPU. Ask the user how to proceed (info-only — CPU already works)."""
        from mockingbird.build import is_cpu_build

        if is_cpu_build():
            # CPU bundle: the user deliberately chose it — "GPU not working"
            # is not a fallback here, no dialog.
            log.info("cuda_fallback suppressed: CPU build")
            return
        from PySide6.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("GPU недоступен")
        box.setText(
            "GPU (CUDA) настроен, но не работает:\n"
            f"{detail}\n\n"
            "Приложение уже переключилось на CPU (распознавание медленнее). "
            "Продолжить на CPU?"
        )
        stay = box.addButton("Работать на CPU", QMessageBox.ButtonRole.YesRole)
        restart = box.addButton("Перезапустить", QMessageBox.ButtonRole.NoRole)
        box.setDefaultButton(stay)
        box.exec()
        if box.clickedButton() is restart:
            self._restart_app()

    def _set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._mute_btn.setEnabled(running)
