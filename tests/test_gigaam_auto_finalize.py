"""Tests for GigaAMEngine._run auto-finalize (F2).

Mirrors the whisper_engine._run auto-finalize pattern: when a speculative
decode is buffered and the audio/VAD pipeline stalls (no CMD_END / CMD_AUDIO
within 5s), the worker must call ``_finalize`` itself so the answer is not
lost. The test injects a stub queue with no commands and verifies the loop
exits via ``queue.Empty`` → ``_finalize`` call → loop continues.
"""
from __future__ import annotations

import queue as _queue
import threading
import time
from unittest.mock import MagicMock

import numpy as np

from mockingbird.config import GigaAMConfig
from mockingbird.stt import gigaam_engine as ge


def _make_engine():
    """Construct an engine with the model loading + decode mocked out."""
    cfg = GigaAMConfig(device="cpu")
    eng = ge.GigaAMEngine.__new__(ge.GigaAMEngine)
    eng._cfg = cfg
    eng._queue = _queue.Queue()
    eng._lock = threading.Lock()
    eng._sr = 16000
    eng._rolling = np.zeros(0, dtype=np.float32)
    eng._segment_id = "seg-test"
    eng._speculative = {"text": "test partial"}  # simulate buffered partial
    eng.on_ready = None
    eng.on_error = None
    return eng


def test_run_auto_finalizes_when_queue_stalls_with_speculative():
    """queue.Empty + speculative set → _finalize is called, loop continues."""
    eng = _make_engine()
    finalize_calls = []

    def fake_finalize(audio, segment_id):
        finalize_calls.append((audio.copy(), segment_id))

    eng._finalize = fake_finalize
    eng._load_model = MagicMock()  # bypass heavy model load

    # Run the worker in a thread; stop after it has had a chance to auto-finalize.
    def stop_after_delay():
        time.sleep(0.7)
        eng._queue.put((ge._CMD_STOP, None))

    threading.Thread(target=stop_after_delay, daemon=True).start()
    eng._run()

    # _finalize should have been called at least once from the auto-finalize path.
    assert finalize_calls, "auto-finalize never fired"
    assert finalize_calls[0][1] == "seg-test"


def test_run_does_not_finalize_on_empty_queue_without_speculative():
    """queue.Empty + no speculative → loop just continues, _finalize NOT called."""
    eng = _make_engine()
    eng._speculative = None  # no partial buffered
    finalize_calls = []
    eng._finalize = lambda audio, sid: finalize_calls.append((audio, sid))
    eng._load_model = MagicMock()

    def stop_after_delay():
        time.sleep(0.7)
        eng._queue.put((ge._CMD_STOP, None))

    threading.Thread(target=stop_after_delay, daemon=True).start()
    eng._run()

    assert finalize_calls == []


def test_run_processes_audio_command_normally():
    """Sanity: a CMD_AUDIO in the queue is consumed and processed."""
    eng = _make_engine()
    eng._speculative = None
    finalize_calls = []
    eng._finalize = lambda audio, sid: finalize_calls.append((audio, sid))
    eng._load_model = MagicMock()
    eng._maybe_decode = MagicMock()

    # Queue an audio chunk followed by STOP after a short delay.
    audio = np.zeros(1600, dtype=np.float32)  # 100 ms of silence
    eng._queue.put((ge._CMD_AUDIO, audio))

    def stop_after_delay():
        time.sleep(0.3)
        eng._queue.put((ge._CMD_STOP, None))

    threading.Thread(target=stop_after_delay, daemon=True).start()
    eng._run()

    # _maybe_decode was triggered by the CMD_AUDIO.
    assert eng._maybe_decode.called
    # No _finalize calls (queue stayed alive, no auto-finalize path).
    assert finalize_calls == []
