"""SpeechChunker routes VAD events to the STT engine (no audio models needed)."""
from __future__ import annotations

import numpy as np

from mockingbird.audio.chunker import SpeechChunker


class _FakeVad:
    def __init__(self, events):
        self._events = events

    def process(self, audio):
        return self._events


class _StubEngine:
    def __init__(self):
        self.calls = []

    def start_segment(self):
        self.calls.append("start")
        return "seg1"

    def feed(self, audio):
        self.calls.append(("audio", len(audio)))

    def on_speech_stop(self):
        self.calls.append("speech_stop")

    def on_speech_resume(self):
        self.calls.append("speech_resume")

    def end_segment(self, audio, segment_id=None):
        self.calls.append(("end", segment_id))


def test_chunker_routes_stop_and_resume():
    engine = _StubEngine()
    events = [
        {"kind": "start"},
        {"kind": "audio", "audio": np.zeros(100, dtype=np.float32)},
        {"kind": "speech_stop"},
        {"kind": "audio", "audio": np.zeros(50, dtype=np.float32)},
        {"kind": "speech_resume"},
        {"kind": "end", "audio": np.zeros(200, dtype=np.float32)},
    ]
    chunker = SpeechChunker(_FakeVad(events), engine)
    chunker.on_audio(np.zeros(0, dtype=np.float32), 0.0)

    assert engine.calls == [
        "start",
        ("audio", 100),
        "speech_stop",
        ("audio", 50),
        "speech_resume",
        ("end", "seg1"),
    ]


def test_chunker_routes_without_stop_events():
    engine = _StubEngine()
    events = [
        {"kind": "start"},
        {"kind": "audio", "audio": np.zeros(100, dtype=np.float32)},
        {"kind": "end", "audio": np.zeros(100, dtype=np.float32)},
    ]
    chunker = SpeechChunker(_FakeVad(events), engine)
    chunker.on_audio(np.zeros(0, dtype=np.float32), 0.0)

    assert engine.calls == [
        "start",
        ("audio", 100),
        ("end", "seg1"),
    ]


def test_chunker_no_events_no_calls():
    engine = _StubEngine()
    chunker = SpeechChunker(_FakeVad([]), engine)
    chunker.on_audio(np.zeros(0, dtype=np.float32), 0.0)
    assert engine.calls == []
