"""Two-stage endpointing: confident-question early finalize (2026-09-27).

The chunker may close a segment after ~250 ms of silence when the latest
partial decode is a finished confident question — well before the full
min_silence window. Narrative/hanging-tail utterances keep the full window.
"""
from __future__ import annotations

import numpy as np

from mockingbird.audio.chunker import (
    _EARLY_FINALIZE_S,
    _is_confident_question_close,
)
from mockingbird.audio.vad import SileroVAD, VadStateMachine


class _RecordingEngine:
    """Minimal engine double for the chunker."""

    last_partial_text = ""

    def __init__(self):
        self.fed = 0
        self.ended: list = []
        self.stop_hints = 0

    def start_segment(self):
        return "seg-1"

    def feed(self, audio):
        self.fed += len(audio)

    def end_segment(self, audio, segment_id=None):
        self.ended.append((audio, segment_id))

    def on_speech_stop(self):
        self.stop_hints += 1

    def on_speech_resume(self):
        pass


class _FakeVad:
    def __init__(self):
        self.cancelled = False

    def process(self, audio):
        return []

    def cancel_pending_end(self):
        self.cancelled = True


def test_confident_question_close_positive():
    assert _is_confident_question_close("что такое kubernetes?")
    assert _is_confident_question_close("расскажи про мониторинг.")
    assert _is_confident_question_close("какие инструменты вы использовали?")


def test_confident_question_close_negative():
    # No terminal punctuation — the full window must hold.
    assert not _is_confident_question_close("расскажи про kubernetes")
    # Hanging tail — mid-question pause, hold open.
    assert not _is_confident_question_close("расскажи про какие?")
    # Single word — too short to trust.
    assert not _is_confident_question_close("kubernetes?")
    # Statement without a question shape.
    assert not _is_confident_question_close("мы использовали docker в проде.")
    assert not _is_confident_question_close("")


def test_early_finalize_window_is_short():
    # The whole point: strictly below the 400 ms min_silence window.
    assert 0.0 < _EARLY_FINALIZE_S < 0.4


def test_vad_cancel_pending_end_resets_machine():
    m = VadStateMachine()
    m._triggered = True
    m._speech = np.ones(1000, dtype=np.float32)
    m._silence = 500
    m._stop_hint_fired = True
    m.cancel_pending_end()
    assert not m._triggered
    assert m._stop_hint_fired is False
    assert len(m._speech) == 0
    # The next consume must behave like a fresh machine (pre-roll path).
    events = m.consume(np.zeros(512, dtype=np.float32), 0.0)
    assert events == []


def test_silero_vad_exposes_cancel_pending_end():
    assert hasattr(SileroVAD, "cancel_pending_end")


def test_engine_end_segment_none_uses_rolling(monkeypatch):
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = object.__new__(WhisperEngine)
    import queue as _q

    eng._queue = _q.Queue()
    eng._pending_final_cmds = 0
    eng._lock = __import__("threading").Lock()
    eng._rolling = np.ones(16000, dtype=np.float32)
    eng.end_segment(None, "seg-9")
    cmd, payload = eng._queue.get_nowait()
    assert cmd == "end"
    audio, seg = payload
    assert seg == "seg-9"
    assert len(audio) == 16000
    assert eng._pending_final_cmds == 1


def test_early_finalize_closes_on_silence_with_confident_partial():
    from mockingbird.audio.chunker import SpeechChunker

    vad = _FakeVad()
    engine = _RecordingEngine()
    engine.last_partial_text = "что такое kubernetes?"
    chunker = SpeechChunker(vad, engine)
    # Drive the chunker as the audio loop would: start → stop-hint (silence)
    # → silence audio blocks until the early window expires.
    chunker.on_audio(np.zeros(512, dtype=np.float32), 0.0)
    chunker._segment_id = "seg-1"
    engine.stop_hints = 0
    chunker.on_audio(np.zeros(512, dtype=np.float32), 0.0)  # no events
    # Simulate the speech_stop event flow directly.
    import time as _time

    chunker._early_finalize_at = _time.monotonic() - 0.001
    for _ in range(3):
        chunker.on_audio(np.zeros(512, dtype=np.float32), 0.0)
    assert vad.cancelled is True
    assert len(engine.ended) == 1
    audio, seg = engine.ended[0]
    assert seg == "seg-1"


def test_early_finalize_skipped_on_narrative():
    from mockingbird.audio.chunker import SpeechChunker

    vad = _FakeVad()
    engine = _RecordingEngine()
    engine.last_partial_text = "мы использовали docker и kubernetes в проде"  # no punct
    chunker = SpeechChunker(vad, engine)
    chunker._segment_id = "seg-1"
    import time as _time

    chunker._early_finalize_at = _time.monotonic() - 0.001
    for _ in range(3):
        chunker.on_audio(np.zeros(512, dtype=np.float32), 0.0)
    assert len(engine.ended) == 0
    assert vad.cancelled is False


def test_trace_summary_uses_ms_for_subsecond():
    from mockingbird.trace import SegmentTrace

    tr = SegmentTrace("abcdefgh1234")
    tr._marks = {
        "speech_stop": 100.0,
        "speech_end": 100.25,
        "stt_final": 100.95,
    }
    s = tr.summary()
    assert "vad=250ms" in s
    assert "stt=700ms" in s


def test_trace_summary_seconds_for_long():
    from mockingbird.trace import SegmentTrace

    tr = SegmentTrace("abcdefgh1234")
    tr._marks = {"speech_start": 0.0, "stt_final": 2.5}
    assert "stt=2.50s" in tr.summary()
