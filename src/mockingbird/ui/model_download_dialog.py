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

    # -- public API ---------------------------------------------------------

    def set_model_name(self, name: str) -> None:
        self._model_label.setText(name)

    def set_progress(self, message: str, percent: float) -> None:
        """Update from the model_load(str, float) signal.

        percent < 0 means "indeterminate" (cache check / loading into
        memory): show a busy marquee text without moving the bar.
        """
        if percent >= 0:
            self._bar.setRange(0, 100)
            self._bar.setValue(int(percent))
            elapsed = max(0.001, time.monotonic() - self._t0)
            self._detail.setText(f"{message}  ·  {percent:.0f}%  ·  {elapsed:.0f} с")
        else:
            self._bar.setRange(0, 0)
            self._detail.setText(message)

    def done_ok(self) -> None:
        """Model loaded — fade the dialog out shortly."""
        self.set_progress("Модель загружена", 100.0)
        self._hide_timer.start(1200)

    def done_failed(self, error: str) -> None:
        self.hide()

    # -- internals -----------------------------------------------------------

    def _on_cancel(self) -> None:
        self._cancel_btn.setEnabled(False)
        self._detail.setText("Отмена…")
        self.cancelled.emit()

    def show_above(self, window: QWidget) -> None:
        if window is not None:
            geo = window.geometry()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.y() + 90,
            )
        self.show()
        self.raise_()
