"""Cross-thread event bus.

Workers emit from their own threads; Qt queued connections marshal delivery
to the GUI thread automatically, keeping the UI render path lightweight.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class AppSignals(QObject):
    mic_level = Signal(float)
    partial = Signal(object)
    final = Signal(object)
    term = Signal(object)
    question = Signal(object)
    answer = Signal(object)
    predictions = Signal(object)
    llm_answer = Signal(object)
    context = Signal(object)
    topics = Signal(object)
    status = Signal(str, str)
    device = Signal(str)
    error = Signal(str)
    log_line = Signal(str)
    model_load = Signal(str, float)
    # Live VAD speech state (True when speech starts, False when it ends).
    speech = Signal(bool)
    # Primary audio source kind: "system" (loopback) or "mic".
    source = Signal(str)
    # Bridge: global hotkey (Ctrl+Alt+H) → MainWindow toggle capture mode.
    toggle_capture_request = Signal()
    # Bridge: STT worker → GUI thread for SQLite writes (avoids cross-thread
    # access to the shared SQLiteStore connection from the decode worker).
    save_segment_request = Signal(object)
    # Bridge: audio callback thread → GUI thread to start the audio watchdog.
    # QTimer must be created on the GUI thread, so the callback signals it.
    _start_watchdog = Signal()
