"""Integration test for the whisper engine.

Disabled by default; enable with MOCKINGBIRD_TEST_WHISPER=1 to download the
tiny model plus a real speech sample and exercise the full pipeline.
"""
from __future__ import annotations

import io
import os
import time
import urllib.request
import wave

import pytest

from mockingbird.config import WhisperConfig
from mockingbird.stt.whisper_engine import WhisperEngine

pytestmark = pytest.mark.skipif(
    os.environ.get("MOCKINGBIRD_TEST_WHISPER") != "1",
    reason="set MOCKINGBIRD_TEST_WHISPER=1 to run the whisper integration test",
)

JFK_WAV_URL = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"


def _load_speech_wav() -> tuple[object, int]:
    """Fetch the standard JFK sample (16 kHz mono 16-bit)."""
    import numpy as np

    with urllib.request.urlopen(JFK_WAV_URL, timeout=60) as resp:
        data = resp.read()
    with wave.open(io.BytesIO(data), "rb") as w:
        frames = w.readframes(w.getnframes())
        sample_rate = w.getframerate()
    return (
        np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0,
        sample_rate,
    )


def _wait_until(predicate, timeout=300.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def test_engine_transcribes_real_speech():
    engine = WhisperEngine(WhisperConfig(model_size="tiny"), sample_rate=16000)
    ready = []
    errors = []
    finals = []
    engine.on_ready = lambda name: ready.append(name)
    engine.on_error = lambda msg: errors.append(msg)
    engine.on_final = lambda m: finals.append(m.text)

    audio, sample_rate = _load_speech_wav()
    assert sample_rate == 16000

    engine.start()
    assert _wait_until(lambda: ready), "model did not load in time"

    engine.start_segment()
    chunk = int(sample_rate * 0.5)
    for i in range(0, len(audio), chunk):
        engine.feed(audio[i : i + chunk])
    engine.flush()
    time.sleep(5)
    engine.stop(timeout=10)

    assert not errors
    assert finals, "no final transcript produced"
    assert any("americans" in text.lower() for text in finals)
