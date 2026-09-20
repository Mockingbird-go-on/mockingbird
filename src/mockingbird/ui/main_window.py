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
        self.setWindowTitle("Mockingbird")
        self.resize(app.config.window.width, app.config.window.height)

        self._settings = QSettings("Mockingbird", "Mockingbird")
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            if not self.restoreGeometry(geometry):
                self.resize(app.config.window.width, app.config.window.height)

        self._session_seconds = 0
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
        layout.addWidget(self._timer_label)
        layout.addStretch(1)
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
        # log_line is wired lazily — see _on_log_panel_toggled.
        self._sig.model_load.connect(self._activity.set_loading)
        # Async session stop completion (marshalled from the stop worker).
        self._stop_done.connect(self._on_stop_done)

    def _on_start(self) -> None:
        try:
            self._app.start_session()
        except Exception as exc:  # noqa: BLE001
            log.exception("start failed")
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
        elif state == "running":
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
            argv = [sys.executable]
        else:
            argv = [sys.executable, "-m", "mockingbird"]
        # Tear the session down BEFORE spawning the new process: the old
        # instance holds the SQLite DB and GPU memory, and a fresh copy
        # starting concurrently risks "database is locked" and VRAM conflicts.
        try:
            self._app.stop_session()
        except Exception:  # noqa: BLE001
            log.exception("stop before restart failed")
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
        self._activity.set_error(message)
        self._sig.status.emit("error", "")
        # If the session never actually started (engine load failed, capture
        # error), return the buttons to the idle state — otherwise Start stays
        # disabled forever with no live session behind it.
        if self._app.session_id is None:
            self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._mute_btn.setEnabled(running)
