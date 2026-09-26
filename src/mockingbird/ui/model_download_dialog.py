"""Model download overlay: a small always-on-top progress window.

Shown while the whisper model is being downloaded (first launch, model
switch). Non-modal — the user can keep exploring the app; the model is
only needed when "Старт" is pressed. Provides a real percentage (from the
huggingface_hub per-file progress), speed/ETA, and a Cancel button.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .icons import icon as lucide_icon


class ModelDownloadDialog(QWidget):
    """Frameless overlay anchored above the main window."""

    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(380)
        self._t0 = time.monotonic()

        root = QWidget(self)
        root.setObjectName("mddCard")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(24, 20, 24, 18)
        lay.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(10)
        glyph = QLabel()
        try:
            pix = lucide_icon(
                "download", size=20, color=theme.current.status_idle
            ).pixmap(20, 20)
            glyph.setPixmap(pix)
        except Exception:  # noqa: BLE001 — icon must never break the dialog
            pass
        glyph.setFixedSize(20, 20)
        title = QLabel("Загрузка модели распознавания")
        title.setObjectName("mddTitle")
        head.addWidget(glyph)
        head.addWidget(title, 1)
        lay.addLayout(head)

        self._model_label = QLabel("")
        self._model_label.setWordWrap(True)
        lay.addWidget(self._model_label)

        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        lay.addWidget(self._bar)

        self._detail = QLabel("Подготовка…")
        self._detail.setObjectName("mddDetail")
        lay.addWidget(self._detail)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn = QPushButton("Отмена")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)
        lay.addLayout(btn_row)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

        # Track the main window's activation: the overlay is only visible
        # while the app itself is active. Switching to a browser hides it
        # (it has WindowStaysOnTopHint and would otherwise hover over
        # unrelated windows); coming back re-shows it mid-download.
        self._main_window: QWidget | None = None
        self._download_active = False
        # Once True (done_ok/done_failed/cancel confirmed) no late progress
        # event may resurrect the overlay.
        self._finalized = False
        self._active_timer = QTimer(self)
        self._active_timer.setInterval(300)
        self._active_timer.timeout.connect(self._sync_visibility_with_main)

    # -- public API ---------------------------------------------------------

    def set_model_name(self, name: str) -> None:
        self._model_label.setText(name)

    def set_progress(self, message: str, percent: float) -> None:
        """Update from the model_load(str, float) signal.

        percent < 0 means "indeterminate" (cache check / loading into
        memory): show a busy marquee text without moving the bar.
        """
        if self._finalized:
            # done_ok/done_failed already fired — late progress events
            # (the in-memory phase, warm-up etc.) must not resurrect the
            # overlay ("stuck at 99%" regression).
            return
        self._download_active = True
        if percent >= 0:
            self._bar.setRange(0, 100)
            self._bar.setValue(int(percent))
            elapsed = max(0.001, time.monotonic() - self._t0)
            self._detail.setText(f"{message}  ·  {percent:.0f}%  ·  {elapsed:.0f} с")
        else:
            self._bar.setRange(0, 0)
            self._detail.setText(message)
        # Late progress events must also restore visibility if the app
        # window is active (covers the case where the timer was stopped).
        self._sync_visibility_with_main()

    def done_ok(self) -> None:
        """Download finished (unpack / in-memory load may still run).

        Marks the overlay finalized: progress events can no longer re-show
        it, and it hides on the fade timer regardless of visibility."""
        self._finalized = True
        self._download_active = False
        self._active_timer.stop()
        self._hide_timer.stop()
        self.set_progress("Модель загружена", 100.0)
        self._hide_timer.start(1200)

    def done_failed(self, error: str) -> None:
        self._finalized = True
        self._download_active = False
        self._active_timer.stop()
        self.hide()

    def _cancelled_confirmed(self) -> None:
        """Download cancellation was confirmed by the engine."""
        self._finalized = True
        self._download_active = False
        self._active_timer.stop()

    # -- internals -----------------------------------------------------------

    def _on_cancel(self) -> None:
        self._cancel_btn.setEnabled(False)
        self._detail.setText("Отмена…")
        self.cancelled.emit()
        # The abort seam is best-effort: if the underlying transfer ignores
        # the cancel (e.g. hub version without the hook), model_load_cancelled
        # never fires and the dialog would hang on «Отмена…» forever.
        # Close it ourselves after a grace period (download continues in the
        # background — the dialog is non-modal by design).
        self._hide_timer.stop()
        QTimer.singleShot(3000, self._cancel_grace_elapsed)

    def _cancel_grace_elapsed(self) -> None:
        if self.isVisible() and not self._cancel_btn.isEnabled():
            self._download_active = False
            self._active_timer.stop()
            self.hide()

    def _sync_visibility_with_main(self) -> None:
        """Show the overlay only while the main window is the active window."""
        if not self._download_active:
            return
        # A modal NotificationBus dialog owns the screen — do not fight it
        # for stacking (was: raise_() on every timer tick over modal boxes).
        from .notify import bus as notify_bus

        if notify_bus.modal_active:
            self.hide()
            return
        main_active = QApplication.activeWindow() is not None
        if main_active:
            if not self.isVisible():
                self.show()
                self.raise_()
        else:
            self.hide()

    def show_above(self, window: QWidget) -> None:
        if window is not None:
            self._main_window = window
            geo = window.geometry()
            # Fully centre over the parent window (was: 90 px below the top,
            # which read as off-centre on tall screens).
            self.adjustSize()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )
        self.show()
        self.raise_()
        self._finalized = False
        self._download_active = True
        self._active_timer.start()
