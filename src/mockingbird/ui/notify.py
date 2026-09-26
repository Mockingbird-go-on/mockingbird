"""NotificationBus — единая очередь модальных уведомлений + toast-стек.

Проблема (2026-09-26): при старте приложения одновременно могли появляться
краш-репорт, системные предупреждения, оверлей загрузки модели и ошибки
скачивания — всё вразнобой, поверх друг друга («каша» из окон).

Решение:
- **Модальные уведомления** идут через строгую FIFO-очередь: в один момент
  времени показан ровно один QMessageBox-подобный диалог. Пока он открыт,
  новые запросы копятся (с dedup по ключу).
- **scope="startup"** — показываются до отрисовки главного окна (краш-репорт,
  системные варнинги); ``flush()`` вызывают вручную до ``window.show()``.
- **scope="runtime"** — обычные ошибки рантайма (фейл загрузки модели и т.п.).
- **info-уровень** — toast в правом верхнем углу, автоскрытие, очередь не
  блокирует (стек до ``_MAX_TOASTS``, старые вытесняются).
- Координация с оверлеем загрузки модели: пока открыт модальный диалог,
  ``modal_active`` == True — оверлей не дёргает ``raise_()`` (см.
  ModelDownloadDialog._sync_visibility_with_main).

GUI-only: все вызовы должны идти из Qt main thread.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from . import theme

log = logging.getLogger("mockingbird.notify")

# Severity → icon glyph (lucide names used by ui.icons).
_SEVERITY_ICON = {"info": "info", "warning": "alert-triangle", "error": "octagon-alert"}
_SEVERITY_COLOR = {"info": "#8a99a8", "warning": "#FFB020", "error": "#FF5148"}

_MAX_TOASTS = 4
_TOAST_MS = 6000


@dataclass
class Notification:
    title: str
    text: str
    severity: str = "info"  # info | warning | error
    scope: str = "runtime"  # startup | runtime
    buttons: tuple = ()  # (("Повторить", on_retry), ("Закрыть", None))
    dedup_key: str = ""
    # Internal: action callbacks keyed by button label.
    _actions: dict = field(default_factory=dict, repr=False)


class Toast(QWidget):
    """Corner toast (right-top). Auto-closes, click = dismiss."""

    def __init__(self, notif: Notification, bus: "NotificationBus"):
        super().__init__(None)
        self._bus = bus
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        color = _SEVERITY_COLOR.get(notif.severity, _SEVERITY_COLOR["info"])
        from PySide6.QtWidgets import QVBoxLayout

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        card = QWidget()
        card.setObjectName("toastCard")
        card.setStyleSheet(
            f"""
            #toastCard {{
                background: {theme.current().bg};
                border: 1px solid {color};
                border-left: 3px solid {color};
                border-radius: 6px;
            }}
            QLabel {{ color: {theme.current().fg}; }}
            """
        )
        box = QVBoxLayout(card)
        box.setContentsMargins(14, 10, 14, 10)
        box.setSpacing(2)
        title = QLabel(notif.title)
        titlef = title.font()
        titlef.setBold(True)
        title.setFont(titlef)
        body = QLabel(notif.text)
        body.setWordWrap(True)
        body.setMaximumWidth(320)
        box.addWidget(title)
        box.addWidget(body)
        outer.addWidget(card)
        self.adjustSize()
        QTimer.singleShot(_TOAST_MS, self.close)

    def mousePressEvent(self, _event) -> None:  # noqa: N802 — Qt naming
        self.close()


class NotificationBus:
    """Singleton FIFO queue for modal notifications + toasts."""

    def __init__(self) -> None:
        self._queue: list[Notification] = []
        self._modal_depth = 0
        self._toasts: list[Toast] = []
        self._startup_flushed = False
        self._overlay_gate: Callable[[], bool] | None = None

    # -- Public API --------------------------------------------------------

    def info(self, title: str, text: str, **kw) -> None:
        self.push(title, text, severity="info", **kw)

    def warning(self, title: str, text: str, **kw) -> None:
        self.push(title, text, severity="warning", **kw)

    def error(self, title: str, text: str, **kw) -> None:
        self.push(title, text, severity="error", **kw)

    def push(
        self,
        title: str,
        text: str,
        severity: str = "info",
        scope: str = "runtime",
        buttons: tuple = (),
        dedup_key: str = "",
    ) -> None:
        """Queue a notification.

        buttons: tuple of (label, callback-or-None). Callbacks run on the GUI
        thread after the dialog closes; first button gets AcceptRole, rest —
        RejectRole-like. Empty → single "OK".
        """
        notif = Notification(
            title=title,
            text=text,
            severity=severity,
            scope=scope,
            dedup_key=dedup_key or f"{severity}:{title}",
        )
        notif._actions = {label: cb for label, cb in buttons}
        if any(n.dedup_key == notif.dedup_key for n in self._queue):
            return  # already queued — do not spam
        if severity == "info" and not buttons:
            self._show_toast(notif)
            return
        if scope == "startup" and self._startup_flushed:
            # Late arrival after the startup phase — show as runtime anyway
            # (rare; better visible than silently queued forever).
            scope = "runtime"
            notif.scope = "runtime"
        self._queue.append(notif)
        self._pump()

    def flush_startup(self) -> None:
        """Show all queued startup notifications synchronously (modal exec).

        Call BEFORE window.show() so the user answers the crash-report /
        system warnings before any overlay or download starts.
        """
        # Hold _modal_open so push() calls made while flushing do not pump
        # inline (they would interleave with the startup dialogs).
        self._modal_depth += 1
        try:
            startup = [n for n in self._queue if n.scope == "startup"]
            for n in startup:
                self._queue.remove(n)
                self._exec_dialog(n)
            self._startup_flushed = True
        finally:
            self._modal_depth -= 1
        self._pump()

    @property
    def modal_active(self) -> bool:
        return self._modal_depth > 0

    def set_overlay_gate(self, gate: Callable[[], bool] | None) -> None:
        """Register a callback the download overlay uses to decide whether
        raise_() is allowed (False while a modal dialog is on screen)."""
        self._overlay_gate = gate

    # -- Internals ----------------------------------------------------------

    def _pump(self) -> None:
        if self._modal_depth or not self._queue:
            return
        notif = self._queue.pop(0)
        self._exec_dialog(notif)
        self._pump()

    def _exec_dialog(self, notif: Notification) -> None:
        from PySide6.QtWidgets import QMessageBox

        self._modal_depth += 1
        try:
            box = QMessageBox()
            box.setWindowTitle(notif.title)
            box.setText(notif.text)
            icon = {
                "info": QMessageBox.Icon.Information,
                "warning": QMessageBox.Icon.Warning,
                "error": QMessageBox.Icon.Critical,
            }.get(notif.severity, QMessageBox.Icon.Information)
            box.setIcon(icon)
            if notif._actions:
                for label, _cb in notif.buttons:
                    box.addButton(label, QMessageBox.ButtonRole.AcceptRole)
            else:
                box.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
            box.exec()
            clicked = box.clickedButton()
            cb = notif._actions.get(clicked.text()) if clicked else None
            if cb:
                try:
                    cb()
                except Exception:
                    log.exception("notification action failed: %s", notif.title)
        finally:
            self._modal_depth -= 1

    def _show_toast(self, notif: Notification) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        toast = Toast(notif, self)
        self._toasts.append(toast)
        while len(self._toasts) > _MAX_TOASTS:
            old = self._toasts.pop(0)
            old.close()
        # Stack toasts top-right; reposition existing ones.
        y = geo.top() + 12
        for t in reversed(self._toasts):
            if not t.isVisible():
                t.show()
            t.move(geo.right() - t.width() - 12, y)
            y += t.height() + 8

    def _forget_toast(self, toast: Toast) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
            self._relayout_toasts()

    def _relayout_toasts(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        y = geo.top() + 12
        for t in reversed(self._toasts):
            t.move(geo.right() - t.width() - 12, y)
            y += t.height() + 8


# Module-level singleton.
bus = NotificationBus()
