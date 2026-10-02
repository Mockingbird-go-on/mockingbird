"""Main application window: large terms view, controls, mic level, interview cockpit."""
from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QObject, QSettings, QSize, Qt, QTimer, Signal
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
from mockingbird.i18n import t
from mockingbird.ui.icons import icon as lucide_icon
from mockingbird.ui.widgets import (
    ActivityBar,
    BackgroundWidget,
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


def _window_pid(hwnd: int) -> int | None:
    """Owning process id of a Win32 window (None off-Windows/on failure)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(int(hwnd), ctypes.byref(pid))
        return int(pid.value) or None
    except Exception:  # noqa: BLE001
        return None


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
        # Last answered screenshot: (jpeg bytes, question). The regenerate
        # button must re-send the IMAGE too — routing it through the regular
        # text-only regenerate path would answer from the question text alone.
        self._last_screenshot: tuple[bytes, str] | None = None
        self._test_overlay = None
        self._picker = None
        self._last_test_text = ""
        self._test_llm_error_notified_at = 0.0
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
            regenerate_callback=self._on_regenerate_query,
            concept_callback=self._app.interview.ask_concept,
            llm_primary=self._app.config.interview.llm_primary,
            llm_available=lambda: bool(self._app.llm.available),
            llm_busy=lambda: bool(getattr(self._app.llm, "is_streaming", False)),
        )
        self._tabs.addTab(self._interview, t("Интервью"))
        self._modules_panel = ResumePanel(self._app)
        self._tabs.addTab(self._modules_panel, t("Резюме"))
        self._log_panel = LogPanel(log_file=self._app.config.storage.log_dir)
        self._tabs.addTab(self._log_panel, t("Лог"))
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
        self._start_btn.setToolTip(t("Старт"))
        self._stop_btn = QPushButton()
        self._stop_btn.setIcon(lucide_icon("square", color=theme.current.status_error))
        self._stop_btn.setIconSize(QSize(18, 18))
        self._stop_btn.setToolTip(t("Стоп"))
        self._mute_btn = QPushButton()
        self._mute_btn.setIcon(self._mute_icon(False))
        self._mute_btn.setIconSize(QSize(18, 18))
        self._mute_btn.setToolTip(t("Мьют"))
        self._settings_btn = QPushButton()
        self._settings_btn.setIcon(lucide_icon("settings-2"))
        self._settings_btn.setIconSize(QSize(18, 18))
        self._settings_btn.setToolTip(t("Настройки"))
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        self._mute_btn.clicked.connect(self._on_toggle_mute)
        self._settings_btn.clicked.connect(self._on_settings)
        self._status = StatusPill()
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
        self._cancel_load_btn.setToolTip(t("Отменить загрузку модели"))
        self._cancel_load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_load_btn.clicked.connect(self._on_cancel_model_load)
        self._cancel_load_btn.hide()
        # Screenshot-to-answer: toolbar button (global hotkey is Windows-only).
        self._shot_btn = QPushButton()
        self._shot_btn.setIcon(lucide_icon("camera"))
        self._shot_btn.setIconSize(QSize(18, 18))
        self._shot_btn.setToolTip(t("Скриншот-вопрос (Ctrl+Shift+S)\nВыделите область экрана и задайте вопрос"))
        self._shot_btn.clicked.connect(self._on_screenshot)
        if not getattr(self._app.config, "screenshot", None) or not self._app.config.screenshot.enabled:
            self._shot_btn.hide()
        # Sync the button state with the cached probe result (if any) so a
        # vision-capable / text-only model is reflected without waiting for
        # the next probe.
        self._refresh_screenshot_button()
        # Live test mode: watch a window/region and answer quiz questions.
        self._test_btn = QPushButton()
        self._test_btn.setIcon(lucide_icon("clipboard-check"))
        self._test_btn.setIconSize(QSize(18, 18))
        self._test_btn.setToolTip(t("Режим «Тест»: следить за окном и подсказывать ответы"))
        self._test_btn.setCheckable(True)
        self._test_btn.clicked.connect(self._on_test_mode_toggle)
        if not getattr(self._app.config, "test_mode", None) or not self._app.config.test_mode.enabled:
            self._test_btn.hide()
        self._test_overlay = None
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
        layout.addWidget(self._source_badge)
        layout.addWidget(self._status)
        layout.addWidget(self._shot_btn)
        layout.addWidget(self._test_btn)
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
        self._install_capture_filter()
        self._apply_capture_affinity_to(self)
        for w in QApplication.topLevelWidgets():
            if w is not self and w.isWindow() and w.isVisible():
                self._apply_capture_affinity_to(w)

    def _install_capture_filter(self) -> None:
        """Install an app-wide event filter syncing capture affinity.

        ``_apply_capture_affinity`` alone only touches windows that are
        visible AT THE MOMENT of a settings change. Dialogs opened later
        (Settings, model download overlay, ...) kept a stale affinity: if
        the privacy mode was ON when they were first shown and OFF now,
        they stayed invisible to OBS until restart. The filter applies
        the CURRENT config to every top-level window on each Show event —
        both set and clear.
        """
        if getattr(self, "_capture_filter", None) is not None:
            return
        from PySide6.QtCore import QEvent

        main = self

        class _CaptureAffinityFilter(QObject):
            def eventFilter(self, obj, ev):  # noqa: N802 — Qt naming
                if ev.type() == QEvent.Type.Show and obj.isWindow():
                    try:
                        main._apply_capture_affinity_to(obj)
                    except Exception:  # noqa: BLE001 — must never crash UI
                        log.exception("capture affinity filter failed")
                return False

        self._capture_filter = _CaptureAffinityFilter(self)
        QApplication.instance().installEventFilter(self._capture_filter)
        log.info("capture-affinity: event filter installed")

    def _apply_capture_affinity_to(self, widget) -> None:
        """Apply the CURRENT hide_from_capture setting to one window."""
        from mockingbird.ui import capture_guard

        if not capture_guard.is_supported():
            return
        enabled = self._app.config.window.hide_from_capture
        if not enabled and getattr(widget, "_mb_keep_capture_exclude", False):
            return  # region-capture overlay: exclusion is intentional
        try:
            hwnd = int(widget.winId())
        except Exception:  # noqa: BLE001 — native handle may not exist yet
            return
        if enabled:
            ok = capture_guard.set_exclude_from_capture(hwnd)
        else:
            ok = capture_guard.clear(hwnd)
        if not ok:
            log.warning(
                "capture-affinity: %s failed for %s (hwnd=%s)",
                "WDA_EXCLUDEFROMCAPTURE" if enabled else "WDA_NONE(clear)",
                type(widget).__name__,
                hwnd,
            )

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
        self._shot_btn.setIcon(lucide_icon("camera"))
        self._test_btn.setIcon(lucide_icon("clipboard-check"))
        # The model-load cancel cross uses text_secondary — repaint it too,
        # or it keeps the old theme's shade when visible.
        self._cancel_load_btn.setIcon(
            lucide_icon("circle-x", color=theme.current.text_secondary)
        )

    def _apply_theme(self, name: str) -> None:
        theme.apply_theme(QApplication.instance(), name)
        self._settings.setValue("ui/theme", name)
        self._bg.theme_changed()
        self._status.update_theme()
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
        # A voice question becomes the current query — the screenshot
        # regenerate association must not survive it (its regenerate click
        # would otherwise re-send a stale image answer).
        self._sig.question.connect(lambda *_: setattr(self, "_last_screenshot", None))
        self._sig.final.connect(self._interview.on_final)
        self._sig.answer.connect(self._interview.on_answer)
        self._sig.llm_answer.connect(self._interview.on_llm_answer)
        self._sig.screenshot_pending.connect(self._interview.on_screenshot_pending)
        self._sig.context.connect(self._interview.on_context)
        self._sig.mic_level.connect(self._activity.set_level)
        self._sig.status.connect(self._on_status)
        self._sig.device.connect(self._status.set_device)
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
        self._sig.test_mode_request.connect(
            lambda: self._on_test_mode_toggle(not self._test_btn.isChecked())
        )
        self._sig.test_answer.connect(self._on_test_answer)

    def _on_vision_probe_result(self, ok) -> None:
        self._vision_state = bool(ok)
        self._refresh_screenshot_button()
        self._refresh_test_button()
        dlg = getattr(self, "_shot_dlg", None)
        if dlg is not None and dlg.isVisible():
            dlg._set_vision(self._vision_state)

    def _refresh_screenshot_button(self) -> None:
        """Reflect the cached vision-probe result in the screenshot button.

        When the configured LLM does not support image inputs the button is
        disabled and the tooltip explains why; otherwise it stays active.
        The probe is launched from _on_start (and lazily on first use); until
        the result lands the button keeps its default state.
        """
        if not hasattr(self, "_shot_btn"):
            return
        if self._vision_state is False:
            self._shot_btn.setEnabled(False)
            self._shot_btn.setToolTip(
                t("Скриншот-вопрос недоступен: текущая LLM не поддерживает изображения.\nИзмените модель в «Настройки».")
            )
        else:
            self._shot_btn.setEnabled(True)
            self._shot_btn.setToolTip(
                t("Скриншот-вопрос (Ctrl+Shift+S)\nВыделите область экрана и задайте вопрос")
            )

    def _refresh_test_button(self) -> None:
        """The test mode needs the SAME vision capability as screenshots:
        grey the button out with an explaining tooltip when the model
        can't take images."""
        if not hasattr(self, "_test_btn"):
            return
        if self._vision_state is False:
            self._test_btn.setEnabled(False)
            self._test_btn.setToolTip(
                t("Режим «Тест» недоступен: текущая LLM не поддерживает изображения.\nИзмените модель в «Настройки».")
            )
        else:
            self._test_btn.setEnabled(True)
            self._test_btn.setToolTip(t("Режим «Тест»: следить за окном и подсказывать ответы"))

    def _on_start(self) -> None:
        try:
            self._app.start_session()
        except Exception as exc:  # noqa: BLE001
            log.exception("start failed")
            # Go through NotificationBus — same FIFO as model-load-failed,
            # vision probe, etc. A direct modal dialog would stack on top of
            # the bus and freeze the GUI (2026-09-26 incident).
            from mockingbird.ui.notify import bus as notify_bus

            notify_bus.error(
                t("Не удалось начать сессию"),
                str(exc),
            )
            self._sig.error.emit(str(exc))
            return
        self._set_running(True)
        self._log_label.setText(self._log_path_text(session=self._app.session_id))
        # Run the vision-capability probe in the background after the session
        # starts. The result lands on _on_vision_probe_result, which updates
        # the screenshot button (greyed-out + tooltip when the model is
        # text-only). The probe is cheap (1×1 PNG, ~5-token reply) and the
        # result is cached per (base_url, model) by LlmClient.probe_vision.
        if self._vision_state is None:
            self._app.check_vision_async()

    def _on_stop(self) -> None:
        """Stop the session without freezing the GUI.

        ``stop_session`` can block up to ~8 s on the engine join, so it runs
        in a background thread; the buttons stay disabled ("stopping") until
        the completion signal fires on the GUI thread. The signal is wired
        once in ``_connect_signals`` — no per-click connect accumulation.
        """
        self._set_stopping(True)
        try:
            self._app.stop_session_async(on_done=self._stop_done.emit)
        except Exception:  # noqa: BLE001 — never leave the UI stuck on "stopping"
            log.exception("stop_session_async failed synchronously")
            self._set_stopping(False)

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

            # Pass `self` as parent so the overlay is owned by the main
            # window — it stays on top of Mockingbird without going above
            # unrelated applications (no more stealing focus over a
            # browser the user switched to mid-download).
            self._model_dl = ModelDownloadDialog(parent=self)
            self._model_dl.cancelled.connect(self._on_model_dl_cancel)
            self._model_dl.set_model_name(
                t("Модель: {name}", name=self._app.config.whisper.model_size)
            )
            self._model_dl.show_above(self)
        self._model_dl.set_progress(message, percent)

    def _set_cancel_load_visible(self, visible: bool) -> None:
        self._cancel_load_btn.setVisible(visible)
        if not visible:
            # Reset any "Отмена…" latched state so the next load starts clean.
            self._cancel_load_btn.setEnabled(True)
            self._cancel_load_btn.setToolTip(t("Отменить загрузку модели"))

    def _on_cancel_model_load(self) -> None:
        """Small cross in the toolbar: cancel the in-flight model load."""
        self._cancel_load_btn.setEnabled(False)
        self._cancel_load_btn.setToolTip(t("Отмена…"))
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
            text = t(
                "Не удалось скачать модель: соединение блокируется "
                "прокси-сервером или антивирусом (подмена SSL-сертификата).\n\n"
                "Варианты:\n"
                "• Попросить IT добавить huggingface.co в исключения SSL-инспекции\n"
                "• Перенести папку модели с другой машины в\n"
                "  {model_dir}",
                model_dir=self._app.config.whisper.model_dir or "~/.mockingbird/models",
            )
        else:
            text = t(
                "Не удалось скачать модель распознавания:\n{error}\n\nПроверьте интернет-соединение.",
                error=error,
            )
        from mockingbird.ui.notify import bus as notify_bus

        notify_bus.error(
            t("Загрузка модели"),
            text,
            buttons=((t("Повторить"), self._app.retry_model_download), (t("Закрыть"), None)),
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
            self._status.set_state("loading", t("остановка…"))

    def _on_status(self, state: str, detail: str) -> None:
        self._status.set_state(state, detail)
        self._interview.set_session_active(state == "running")
        if state == "loading":
            self._activity.set_loading(detail or t("Загрузка модели…"), -1)
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
                self._activity.set_error(detail or t("Ошибка"))

    def _tick_session(self) -> None:
        self._session_seconds += 1
        self._timer_label.setText(_format_elapsed(self._session_seconds))

    def _log_path_text(self, session: str | None = None) -> str:
        path = getattr(self._app.config.storage, "log_dir", "")
        prefix = t("Сессия #{n} · ", n=session) if session else ""
        return t("{prefix}Лог: {path}", prefix=prefix, path=path)

    def _on_toggle_mute(self) -> None:
        self._app.toggle_mute()
        muted = self._app.muted
        self._mute_btn.setIcon(self._mute_icon(muted))
        self._mute_btn.setToolTip(t("Снять мьют") if muted else t("Мьют"))
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
            try:
                self._app.save_settings()
            except Exception:  # noqa: BLE001 — settings I/O failure must not kill the slot
                log.exception("save_settings failed")
            try:
                self._apply_capture_affinity()
            except Exception:  # noqa: BLE001
                log.exception("apply_capture_affinity failed")
            # LLM base_url/model may have changed in the dialog — the cached
            # vision capability is stale now. Reset and re-probe so the
            # screenshot button reflects the NEW model immediately.
            self._vision_state = None
            self._refresh_screenshot_button()
            try:
                self._app.check_vision_async()
            except Exception:  # noqa: BLE001
                log.exception("check_vision_async failed")
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
        box.setWindowTitle(t("Перезапуск"))
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(t("Часть изменений вступит в силу после перезапуска приложения."))
        restart = box.addButton(
            t("Перезапустить сейчас"), QMessageBox.ButtonRole.AcceptRole
        )
        later = box.addButton(t("Позже"), QMessageBox.ButtonRole.RejectRole)
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
            QMessageBox.critical(None, t("Ошибка"), t("Не удалось перезапустить приложение: {exc}", exc=exc))
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
            self.statusBar().showMessage(t("Скриншот не получен: {exc}", exc=exc), 5000)
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

    def _on_regenerate_query(self, query: str):
        """Regenerate dispatcher: screenshot answers re-send the IMAGE.

        The panel's regenerate button passes the last query; when that query
        was a screenshot question the regeneration must go through
        answer_screenshot (vision stream) — the engine's text-only
        regenerate_answer would answer from the question text alone, losing
        the image context entirely.
        """
        shot = self._last_screenshot
        if shot is not None and shot[1].strip() == (query or "").strip():
            import uuid as _uuid

            jpeg, question = shot
            stream_id = f"shot-{_uuid.uuid4().hex[:12]}"
            self._interview.begin_external_stream(question, stream_id)
            self._pending_screenshot_question = question
            try:
                self._app.answer_screenshot(jpeg, question, stream_id=stream_id)
            except Exception as exc:  # noqa: BLE001 — GUI slot must not throw
                log.exception("screenshot regenerate failed")
                from mockingbird.ui.notify import bus as notify_bus

                notify_bus.error(
                    t("Ошибка"),
                    t("Не удалось получить ответ по скриншоту: {exc}", exc=exc),
                )
            return None
        try:
            return self._app.interview.regenerate_answer(query)
        except Exception as exc:  # noqa: BLE001
            log.exception("regenerate_answer failed")
            from mockingbird.ui.notify import bus as notify_bus

            notify_bus.error(t("Ошибка"), t("Ошибка: {error}", error=exc))

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
        self._last_screenshot = (dlg.jpeg_bytes(), question)
        try:
            self._app.answer_screenshot(dlg.jpeg_bytes(), question, stream_id=stream_id)
        except Exception as exc:  # noqa: BLE001 — GUI slot must not throw
            log.exception("answer_screenshot failed")
            dlg.set_busy(False, t("Не удалось получить ответ по скриншоту: {exc}", exc=exc))

    def _on_screenshot_answer_done(self, shot_id: str) -> None:
        dlg = getattr(self, "_shot_dlg", None)
        if dlg is not None and dlg.isVisible():
            dlg.set_busy(False, t("Ответ — в панели «Ответ ИИ»"))
            dlg.close()
        # NOTE: no history entry here — `on_screenshot_pending` already adds
        # it the moment the question is asked (instant feedback). Adding a
        # second one here produced a duplicate entry with a different tag
        # ("screenshot" vs localized "Скриншот").
        self._pending_screenshot_question = ""

    # ------------------------------------------------------------------
    # Live test mode
    # ------------------------------------------------------------------
    def _on_test_mode_toggle(self, checked: bool) -> None:
        if checked:
            self._start_test_mode()
        else:
            self._stop_test_mode()

    def _start_test_mode(self) -> None:
        from mockingbird.ui.window_picker import WindowPickOverlay

        # A stale picker from a previous toggle must not linger fullscreen.
        old = getattr(self, "_picker", None)
        self._picker = None
        if old is not None:
            old.close()
            old.deleteLater()
        self._picker = WindowPickOverlay()
        self._picker.picked.connect(self._on_test_window_picked)
        self._picker.region_picked.connect(self._on_test_region_picked)
        self._picker.cancelled.connect(self._on_test_pick_cancelled)
        self._picker.start()

    def _stop_test_mode(self) -> None:
        self._dispose_picker()
        self._test_btn.setChecked(False)
        ov = self._test_overlay
        self._test_overlay = None
        self._app.stop_test_mode()
        if ov is not None:
            ov.close()
            ov.deleteLater()

    def _dispose_picker(self) -> None:
        """Close and schedule deletion of a live window picker (if any)."""
        p = getattr(self, "_picker", None)
        self._picker = None
        if p is not None:
            try:
                p.close()
                p.deleteLater()
            except RuntimeError:  # already deleted (C++ object gone)
                pass

    def _on_test_pick_cancelled(self) -> None:
        self._dispose_picker()
        self._test_btn.setChecked(False)

    def _on_test_capture_failed(self, msg: str) -> None:
        """Target window gone — the watcher already stopped itself."""
        ov = self._test_overlay
        if ov is not None:
            ov.set_status(msg)
        self._test_btn.setChecked(False)
        self._test_overlay = None
        self._app.stop_test_mode()
        if ov is not None:
            from PySide6.QtCore import QTimer

            watcher_ptr = ov

            def _close() -> None:
                watcher_ptr.close()
                watcher_ptr.deleteLater()

            QTimer.singleShot(3000, _close)  # let the user read the status

    def _on_test_window_picked(self, hwnd: int, title: str, rect) -> None:
        # Windows reuses hwnd values: remember the owning PID and verify it
        # on every capture — a closed target whose hwnd got reassigned to an
        # unrelated window must not leak into the LLM frames.
        self._test_hwnd_pid = _window_pid(hwnd)
        self._launch_test_watcher(
            title, lambda: self._capture_window(hwnd), is_region=False
        )

    def _on_test_region_picked(self, rect) -> None:
        self._launch_test_watcher(
            t("область экрана"), lambda: self._capture_region(rect), is_region=True
        )

    def _launch_test_watcher(self, title: str, capture_fn, is_region: bool = False) -> None:
        from mockingbird.ui.test_overlay import TestModeOverlay

        # Close a stale overlay from a previous watcher run — its «×»
        # button is still wired to _stop_test_mode and would kill the NEW
        # watcher.
        old = self._test_overlay
        self._test_overlay = None
        if old is not None:
            old.close()
            old.deleteLater()
        self._app.stop_test_mode()
        watcher = self._app.start_test_mode(capture_fn)
        self._test_llm_error_notified_at = 0.0  # fresh warning budget per run
        ov = TestModeOverlay(title)
        ov.stop_requested.connect(self._stop_test_mode)
        ov.force_requested.connect(self._app.force_test_frame)
        watcher.frame_skipped.connect(ov.set_status)
        watcher.frame_pending.connect(ov.set_status)
        watcher.capture_failed.connect(self._on_test_capture_failed)
        self._test_overlay = ov
        ov.show()
        ov.set_status(t("наблюдение за {title}…", title=title))
        # Liveness timer start
        import time as _time

        ov._started_at = _time.monotonic()
        # Region mode captures raw desktop pixels — the overlay would feed
        # itself into the frame there, so exclude it from capture (Windows).
        # Window mode (PrintWindow) renders the TARGET window's content only
        # — overlapping windows can't appear, no exclusion needed: the user
        # may want the overlay visible in screen sharing/streaming.
        if is_region:
            # Marker: the capture-affinity event filter must NOT clear the
            # exclusion on this overlay even when privacy mode is off —
            # region capture would feed the overlay into its own frames.
            ov._mb_keep_capture_exclude = True
            try:
                from mockingbird.ui.capture_guard import set_exclude_from_capture

                set_exclude_from_capture(int(ov.winId()))
            except Exception:  # noqa: BLE001
                pass

    def _capture_window(self, hwnd: int):
        from mockingbird.ui.window_capture import capture_window

        expected_pid = getattr(self, "_test_hwnd_pid", None)
        if expected_pid is not None:
            actual_pid = _window_pid(hwnd)
            if actual_pid is not None and actual_pid != expected_pid:
                raise RuntimeError(
                    t("окно закрыто или недоступно — наблюдение остановлено")
                )
        cfg = self._app.config.test_mode
        image, jpeg, pix = capture_window(
            hwnd, max_dim=cfg.max_image_dim, jpeg_quality=cfg.jpeg_quality
        )
        log.info(
            "test-mode: window %s captured %dx%d -> jpeg %dB",
            hwnd, image.width(), image.height(), len(jpeg),
        )
        return image, jpeg, pix

    def _capture_region(self, rect):
        from mockingbird.ui.screenshot import grab_screen_region

        log.info("test-mode: capture region %s", rect)
        cfg = self._app.config.test_mode
        jpeg, pix = grab_screen_region(
            rect, max_dim=cfg.max_image_dim, jpeg_quality=cfg.jpeg_quality
        )
        log.info("test-mode: region captured -> jpeg %dB", len(jpeg))
        return pix.toImage(), jpeg, pix

    def _on_test_answer(self, ok: bool, text: str) -> None:
        ov = self._test_overlay
        if ov is None:
            return
        if text == "\u23f3":  # busy marker: frame deferred, voice answer streaming
            ov.set_status(t("ждём — идёт голосовой ответ…"))
            return
        if not ok and text == t("LLM не настроен — задайте модель в настройках"):
            # Terminal outcome from the worker: the watcher would spin
            # silently — stop the mode cleanly and tell the user what to fix.
            ov.set_status(text)
            from mockingbird.ui.notify import bus as notify_bus_local

            notify_bus_local.warning(
                t("Режим «Тест»"), text,
            )
            self._stop_test_mode()
            return
        if not ok:
            ov.set_status(t("ошибка LLM, повтор через пару секунд…"))
            # Persistent LLM failure (bad model / no network / provider
            # down) must not pass silently — surface a notification, but
            # throttled: backoff retries would spam one per failure.
            from time import monotonic

            last = getattr(self, "_test_llm_error_notified_at", 0.0)
            if monotonic() - last > 60.0:
                self._test_llm_error_notified_at = monotonic()
                from mockingbird.ui.notify import bus as notify_bus

                notify_bus.warning(
                    t("Режим «Тест»: проблема с LLM"),
                    t(
                        "Не удалось получить ответ от модели: проверьте "
                        "подключение (сеть, API-ключ, модель) в «Настройки». "
                        "Наблюдение продолжится с повторами."
                    ),
                )
            return
        from mockingbird.vision.test_watcher import parse_test_answers

        pairs = parse_test_answers(text)
        changed = text != self._last_test_text
        self._last_test_text = text
        if not pairs:
            ov.set_status(t("тест не распознан на кадре"))
            return
        ov.set_answers(pairs, changed=changed)

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
        box.setWindowTitle(t("GPU недоступен"))
        box.setText(
            t(
                "GPU (CUDA) настроен, но не работает:\n"
                "{detail}\n\n"
                "Приложение уже переключилось на CPU (распознавание медленнее). "
                "Продолжить на CPU?",
                detail=detail,
            )
        )
        stay = box.addButton(t("Работать на CPU"), QMessageBox.ButtonRole.YesRole)
        restart = box.addButton(t("Перезапустить"), QMessageBox.ButtonRole.NoRole)
        box.setDefaultButton(stay)
        box.exec()
        if box.clickedButton() is restart:
            self._restart_app()

    def _set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._mute_btn.setEnabled(running)
