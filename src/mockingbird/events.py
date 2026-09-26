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
    llm_answer = Signal(object)
    context = Signal(object)
    status = Signal(str, str)
    device = Signal(str)
    error = Signal(str)
    # Emitted from the STT worker when CUDA was configured but unusable and
    # the engine silently fell back to CPU. Payload: the probe detail string.
    cuda_fallback = Signal(str)
    log_line = Signal(str)
    model_load = Signal(str, float)
    # Emitted when the warm-start model load/download FAILED (payload: the
    # exception text). The UI shows a human-readable dialog with retry.
    model_load_failed = Signal(str)
    # Emitted when the user cancelled an in-flight model load (download OR
    # "Loading model into memory…"). The UI hides the cancel affordance and
    # returns to idle without showing an error.
    model_load_cancelled = Signal()
    # Live VAD speech state (True when speech starts, False when it ends).
    speech = Signal(bool)
    # Primary audio source kind: "system" (loopback) or "mic".
    source = Signal(str)
    # Bridge: global hotkey (Ctrl+Alt+H) → MainWindow toggle capture mode.
    toggle_capture_request = Signal()
    # Screenshot-to-answer: vision-capability check result (payload: True,
    # False or None while the probe is still running).
    vision_probe_result = Signal(object)
    # Bridge: Ctrl+Shift+S global hotkey → MainWindow opens the grab overlay.
    screenshot_request = Signal()
    # Screenshot answer finished (payload: shot_id) — UI refreshes history.
    screenshot_answer_done = Signal(str)
    # Bridge: STT worker → GUI thread for SQLite writes (avoids cross-thread
    # access to the shared SQLiteStore connection from the decode worker).
    save_segment_request = Signal(object)
    # Bridge: audio callback thread → GUI thread to start the audio watchdog.
    # QTimer must be created on the GUI thread, so the callback signals it.
    _start_watchdog = Signal()
    # Bridge: session-stop daemon thread → GUI thread to stop the watchdog
    # QTimer. Calling QTimer.stop() from a non-GUI thread is undefined
    # behaviour in Qt (sporadic access violations on Windows).
    _stop_watchdog = Signal()
    # Bridge: STT worker → GUI thread for the ready sound. QMediaPlayer must
    # be created and played on the GUI thread (object affinity); the engine
    # worker used to construct it directly, which crashes/never plays.
    _play_ready_sound = Signal()
