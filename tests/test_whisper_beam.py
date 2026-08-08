"""Beam-size wiring for whisper partial/final decodes (no model required)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mockingbird.config import WhisperConfig
from mockingbird.stt.whisper_engine import WhisperEngine


class _StubModel:
    """Minimal faster-whisper model fake: records transcribe kwargs."""

    def __init__(self):
        self.last_kwargs = None

    def transcribe(self, audio, **kwargs):
        self.last_kwargs = kwargs
        return iter([]), SimpleNamespace(duration=0.5, language="ru")


def _make_engine(**cfg_overrides):
    cfg = WhisperConfig(**cfg_overrides)
    engine = WhisperEngine(cfg, sample_rate=16000)
    engine._model = _StubModel()
    return engine


def test_partial_decode_uses_partial_beam_size():
    engine = _make_engine(beam_size=1, final_beam_size=5)
    sr = engine._sr
    engine._rolling = np.zeros(int(0.6 * sr), dtype=np.float32)
    engine._last_decode = -10.0
    engine._segment_id = "seg"
    engine._maybe_decode()
    assert engine._model.last_kwargs["beam_size"] == 1


def test_final_decode_uses_final_beam_size():
    engine = _make_engine(beam_size=1, final_beam_size=5)
    sr = engine._sr
    engine._finalize(np.zeros(int(0.6 * sr), dtype=np.float32), "seg")
    assert engine._model.last_kwargs["beam_size"] == 5


def test_final_decode_custom_final_beam_size():
    engine = _make_engine(final_beam_size=8)
    sr = engine._sr
    engine._finalize(np.zeros(int(0.6 * sr), dtype=np.float32), "seg")
    assert engine._model.last_kwargs["beam_size"] == 8


def test_transcribe_accepts_explicit_beam_size():
    engine = _make_engine()
    audio = np.zeros(int(0.5 * engine._sr), dtype=np.float32)
    engine._transcribe(audio, kind="decode", beam_size=3)
    assert engine._model.last_kwargs["beam_size"] == 3
