"""Main application window: large terms view, controls, mic level, interview cockpit."""
from __future__ import annotations

import logging

from PySide6.QtCore import QSettings, QSize, Qt, QTimer
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
from mockingbird.ui.modules_panel import ModulesPanel
from mockingbird.ui.settings_dialog import SettingsDialog
from mockingbird.ui import theme
from mockingbird.ui.widgets import (
    ActivityBar,
    BackgroundWidget,
    DeviceBadge,
    LogoBadge,
    SourceBadge,
    StatusDot,
    StatusPill,
)

log = logging.getLogger(__name__)


def _format_elapsed(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class MainWindow(QMainWindow):
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
            llm_primary=self._app.config.interview.llm_primary,
            llm_available=self._app.llm.available,
        )
        self._tabs.addTab(self._interview, "Интервью")
        self._modules_panel = ModulesPanel(self._app)
        self._tabs.addTab(self._modules_panel, "Модули")
        self._log_panel = LogPanel(log_file=self._app.config.storage.log_dir)
        self._tabs.addTab(self._log_panel, "Лог")
        # Hot-zone (reveal-on-hover for Simple Mode) above the toolbar.
        from mockingbird.ui.hot_zone import TopHotZone, _FadeGroup

        self._hotzone = TopHotZone(self)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(800)
        self._hide_timer.timeout.connect(self._hide_simple)
        self._hotzone.hover_enter.connect(self._reveal_simple)
        self._hotzone.hover_leave.connect(self._schedule_hide_simple)
        self._toolbar = self._build_toolbar()
        layout.addWidget(self._hotzone)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._tabs, stretch=1)
        self.setCentralWidget(central)

        # Fade groups for Simple Mode.
        self._fade_group = _FadeGroup([self._toolbar, self.statusBar()])
        # Minimal status dot (top-left of the central widget) — visible only
        # in Simple Mode.
        self._status_dot = StatusDot(self._bg)
        self._status_dot.move(12, 12)
        self._status_dot.setVisible(False)
        self._simple_mode = False

        self._connect_signals()
        self._set_running(False)
        self._sig.status.emit("idle", "")

    def closeEvent(self, event) -> None:
        self._session_timer.stop()
        if self._hide_timer.isActive():
            self._hide_timer.stop()
        self._settings.setValue("window/geometry", self.saveGeometry())
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Keep the status dot anchored to the top-left of the central widget.
        self._status_dot.move(12, 12)
        self._status_dot.raise_()

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
        self._start_btn.setIcon(self._make_play_icon())
        self._start_btn.setIconSize(QSize(18, 18))
        self._start_btn.setProperty("primary", True)
        self._start_btn.setToolTip("Старт")
        self._stop_btn = QPushButton()
        self._stop_btn.setIcon(self._make_stop_icon())
        self._stop_btn.setIconSize(QSize(18, 18))
        self._stop_btn.setToolTip("Стоп")
        self._mute_btn = QPushButton()
        self._mute_btn.setIcon(self._make_mute_icon(False))
        self._mute_btn.setIconSize(QSize(18, 18))
        self._mute_btn.setToolTip("Мьют")
        self._simple_btn = QPushButton()
        self._simple_btn.setIcon(self._make_simple_icon())
        self._simple_btn.setIconSize(QSize(18, 18))
        self._simple_btn.setCheckable(True)
        self._simple_btn.setChecked(self._app.config.window.simple_mode)
        self._simple_btn.setToolTip("Simple Mode — скрыть лишние элементы UI")
        self._simple_btn.clicked.connect(self._on_toggle_simple)
        self._settings_btn = QPushButton()
        self._settings_btn.setIcon(self._make_settings_icon())
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
        layout.addWidget(self._simple_btn)
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

    # -- Simple Mode -------------------------------------------------------

    def _on_toggle_simple(self) -> None:
        enabled = self._simple_btn.isChecked()
        self.set_simple_mode(enabled)
        self._app.save_settings()

    def set_simple_mode(self, enabled: bool) -> None:
        """Toggle the minimal UI: hide chrome, keep only the answer pane.

        Reveal-on-hover is driven by the top hot-zone; an 800 ms delay avoids
        flicker when the cursor grazes the zone edge.
        """
        self._simple_mode = enabled
        self._app.config.window.simple_mode = enabled
        self._simple_btn.setChecked(enabled)
        if enabled:
            # Stop the activity-bar animation (avoids wasted repaints of a
            # hidden widget) and let InterviewPanel hide its own chrome.
            self._activity.set_idle()
            self._interview.set_simple_mode(True)
            self._fade_group.fade_to(0.0, on_finished=lambda: self._fade_group.set_visible_immediate(False))
            self._status_dot.setVisible(True)
            self._status_dot.set_state("running" if self._app.session_id else "idle")
        else:
            self._hide_timer.stop()
            self._fade_group.set_visible_immediate(True)
            self._fade_group.fade_to(1.0)
            self._interview.set_simple_mode(False)
            self._status_dot.setVisible(False)
            if self._app.session_id is not None:
                self._activity.set_live()

    def _reveal_simple(self) -> None:
        """Cursor entered the hot-zone: show the chrome (if simple mode is on)."""
        if not self._simple_mode:
            return
        self._hide_timer.stop()
        self._fade_group.set_visible_immediate(True)
        self._fade_group.fade_to(1.0)
        self._interview.set_simple_mode(False)

    def _schedule_hide_simple(self) -> None:
        """Cursor left the hot-zone: hide the chrome after a short delay."""
        if not self._simple_mode:
            return
        self._hide_timer.start(800)

    def _hide_simple(self) -> None:
        """Timer expired: re-hide the chrome (still in simple mode)."""
        if not self._simple_mode:
            return
        self._activity.set_idle()
        self._interview.set_simple_mode(True)
        self._fade_group.fade_to(0.0, on_finished=lambda: self._fade_group.set_visible_immediate(False))

    # -- icon helpers --

    @staticmethod
    def _make_play_icon():
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap

        from mockingbird.ui import theme

        pix = QPixmap(18, 18)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.current.text))
        # Rounded triangle (play)
        path = QPainterPath()
        path.moveTo(13, 9)
        path.lineTo(5, 4)
        path.lineTo(5, 14)
        path.closeSubpath()
        p.drawPath(path)
        p.end()
        return QIcon(pix)

    @staticmethod
    def _make_stop_icon():
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

        from mockingbird.ui import theme

        pix = QPixmap(18, 18)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.current.text))
        p.drawRoundedRect(QRectF(4, 4, 10, 10), 2.0, 2.0)
        p.end()
        return QIcon(pix)

    @staticmethod
    def _make_settings_icon():
        """Vector gear icon painted in the current text colour."""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

        from mockingbird.ui import theme

        pix = QPixmap(18, 18)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(theme.current.text)
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        # Gear body: outer toothed ring + inner hub hole.
        path = QPainterPath()
        cx, cy, r_out, r_in, r_hub = 9.0, 9.0, 7.5, 5.0, 2.4
        import math

        teeth = 8
        pts: list[QPointF] = []
        for i in range(teeth * 2):
            angle = math.pi / teeth * i - math.pi / 2
            radius = r_out if i % 2 == 0 else r_in
            pts.append(QPointF(cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        path.moveTo(pts[0])
        for pt in pts[1:]:
            path.lineTo(pt)
        path.closeSubpath()
        # Hub hole (even-odd fill leaves it transparent).
        hole = QPainterPath()
        hole.addEllipse(QPointF(cx, cy), r_hub, r_hub)
        path = path.subtracted(hole)
        p.drawPath(path)
        p.end()
        return QIcon(pix)

    @staticmethod
    def _make_mute_icon(muted: bool):
        """Vector speaker icon; crossed-out when muted."""
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

        from mockingbird.ui import theme

        pix = QPixmap(18, 18)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(theme.current.text)
        muted_color = QColor(theme.TEXT_SECONDARY)
        p.setBrush(color if not muted else muted_color)
        p.setPen(Qt.PenStyle.NoPen)
        # Speaker body: small rectangle (horn throat) + trapezoid (cone).
        p.drawRoundedRect(QRectF(2.5, 6.5, 3.5, 5.0), 0.8, 0.8)
        path = QPainterPath()
        path.moveTo(5.5, 6.0)
        path.lineTo(10.0, 3.0)
        path.lineTo(10.0, 15.0)
        path.lineTo(5.5, 12.0)
        path.closeSubpath()
        p.drawPath(path)
        if not muted:
            # Sound waves (two arcs to the right of the speaker).
            p.setBrush(Qt.BrushStyle.NoBrush)
            wave_pen = QPen(color, 1.4)
            wave_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(wave_pen)
            p.drawArc(QRectF(11.5, 5.0, 6.0, 8.0), -45 * 16, 90 * 16)
            p.drawArc(QRectF(11.5, 3.0, 10.0, 12.0), -45 * 16, 90 * 16)
        else:
            # Diagonal strike-through line (muted).
            p.setBrush(Qt.BrushStyle.NoBrush)
            strike = QPen(muted_color, 1.6)
            strike.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(strike)
            p.drawLine(QPointF(3.0, 3.0), QPointF(15.0, 15.0))
        p.end()
        return QIcon(pix)

    @staticmethod
    def _make_simple_icon():
        """Four-point sparkle (minimalist focus/minimise-mode glyph)."""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap

        from mockingbird.ui import theme

        pix = QPixmap(18, 18)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(theme.current.accent)
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        # Four-point sparkle centred at (9, 9).
        cx, cy = 9.0, 9.0
        path = QPainterPath()
        # Main sparkle (large).
        path.moveTo(cx, cy - 6.0)
        path.cubicTo(cx + 1.6, cy - 2.0, cx + 2.0, cy - 1.6, cx + 6.0, cy)
        path.cubicTo(cx + 2.0, cy + 1.6, cx + 1.6, cy + 2.0, cx, cy + 6.0)
        path.cubicTo(cx - 1.6, cy + 2.0, cx - 2.0, cy + 1.6, cx - 6.0, cy)
        path.cubicTo(cx - 2.0, cy - 1.6, cx - 1.6, cy - 2.0, cx, cy - 6.0)
        path.closeSubpath()
        # Small sparkle (top-right).
        sx, sy = cx + 4.2, cy - 4.2
        path.moveTo(sx, sy - 2.4)
        path.cubicTo(sx + 0.6, sy - 0.8, sx + 0.8, sy - 0.6, sx + 2.4, sy)
        path.cubicTo(sx + 0.8, sy + 0.6, sx + 0.6, sy + 0.8, sx, sy + 2.4)
        path.cubicTo(sx - 0.6, sy + 0.8, sx - 0.8, sy + 0.6, sx - 2.4, sy)
        path.cubicTo(sx - 0.8, sy - 0.6, sx - 0.6, sy - 0.8, sx, sy - 2.4)
        path.closeSubpath()
        p.drawPath(path)
        p.end()
        return QIcon(pix)

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
        self._status_dot.update_theme()
        # Re-assert Simple Mode visibility after a theme switch (some widgets
        # may have been re-shown by their update_theme callbacks).
        if self._simple_mode:
            self.set_simple_mode(True)

    def _connect_signals(self) -> None:
        self._sig.partial.connect(self._interview.on_partial)
        self._sig.question.connect(self._interview.on_question)
        self._sig.final.connect(self._interview.on_final)
        self._sig.answer.connect(self._interview.on_answer)
        self._sig.predictions.connect(self._interview.on_predictions)
        self._sig.llm_answer.connect(self._interview.on_llm_answer)
        self._sig.context.connect(self._interview.on_context)
        self._sig.mic_level.connect(self._activity.set_level)
        self._sig.status.connect(self._on_status)
        self._sig.device.connect(self._device_badge.set_device)
        self._sig.source.connect(self._source_badge.set_source)
        self._sig.speech.connect(self._activity.flash_speech)
        self._sig.error.connect(self._on_error)
        self._sig.log_line.connect(self._log_panel.append_line)
        self._sig.model_load.connect(self._activity.set_loading)

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
        self._app.stop_session()
        self._session_timer.stop()
        # Reset the elapsed counter so the next session starts from 00:00
        # rather than accumulating across sessions.
        self._session_seconds = 0
        self._timer_label.setText("00:00")
        self._set_running(False)
        self._log_label.setText(self._log_path_text())

    def _on_status(self, state: str, detail: str) -> None:
        self._status.set_state(state, detail)
        self._status_dot.set_state(state)
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
        self._mute_btn.setIcon(self._make_mute_icon(muted))
        self._mute_btn.setToolTip("Снять мьют" if muted else "Мьют")
        self._sig.status.emit("muted" if muted else "running", "")

    def _on_settings(self) -> None:
        dialog = SettingsDialog(self._app.config, self)
        if dialog.exec():
            dialog.apply()
            self._app.save_settings()
            self._apply_capture_affinity()
            if dialog.restart_required:
                self._prompt_restart()

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

    @staticmethod
    def _restart_app() -> None:
        import os
        import subprocess
        import sys

        from PySide6.QtWidgets import QApplication

        if getattr(sys, "frozen", False):
            argv = [sys.executable]
        else:
            argv = [sys.executable, "-m", "mockingbird"]
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

    def _set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._mute_btn.setEnabled(running)
