"""Finalize speculative reuse + GPU compute-type preference (no heavy imports)."""
from __future__ import annotations

import numpy as np

from mockingbird.config import WhisperConfig
from mockingbird.stt.whisper_engine import (
    WhisperEngine,
    _SPECULATIVE_REUSE_MAX_DELTA_S,
    select_compute_type,
)


def _engine(sample_rate: int = 16000) -> WhisperEngine:
    eng = WhisperEngine(WhisperConfig(model_size="tiny"), sample_rate=sample_rate)
    eng._model = object()  # bypass the None guard in _finalize
    return eng


# --- compute type ---

def test_pascal_prefers_float32_over_int8_float32():
    # Pascal (CC < 7.0): float16 absent, int8_float32 present but NOT auto —
    # A/B showed int8_float32 degrades WER; float32 is the quality fallback.
    assert (
        select_compute_type("cuda", "float16", ["int8_float32", "float32", "int8"])
        == "float32"
    )


def test_float16_still_preferred_on_modern_gpu():
    assert (
        select_compute_type("cuda", "int8", ["float16", "int8_float32", "float32"])
        == "float16"
    )


def test_int8_float16_preferred_over_int8_float32():
    assert (
        select_compute_type("cuda", "float16", ["int8_float16", "int8_float32", "float32"])
        == "int8_float16"
    )


def test_float32_fallback_without_any_int8_mixed():
    assert (
        select_compute_type("cuda", "float16", ["float32", "int8"]) == "float32"
    )


# --- speculative reuse in _finalize ---

def _capture(engine):
    finals = []
    engine.on_final = lambda msg: finals.append(msg)
    return finals


def _monkeypatch_decode(engine, returned=("привет мир", 0.9, 1.0)):
    calls = []

    def fake(audio, kind="final", beam_size=1):
        calls.append((kind, len(audio)))
        return returned

    engine._decode_cached = fake
    return calls


def test_finalize_reuses_speculative_when_no_new_audio():
    eng = _engine()
    finals = _capture(eng)
    calls = _monkeypatch_decode(eng)
    seg = "seg-1"
    eng._segment_id = seg
    dur = 5.0
    audio = np.zeros(int(16000 * dur), dtype=np.float32)
    eng._speculative = {
        "segment_id": seg,
        "text": "Что такое DNS?",
        "confidence": 0.85,
        "duration": dur,
    }
    eng._last_partial_text = "Что такое DNS?"

    eng._finalize(audio, seg)

    assert calls == [], f"expected no re-decode, got {calls}"
    assert len(finals) == 1
    assert finals[0].text == "Что такое DNS?"


def test_finalize_redecodes_when_buffer_grew_beyond_delta():
    eng = _engine()
    finals = _capture(eng)
    calls = _monkeypatch_decode(eng, returned=("новый текст вопроса", 0.8, 7.0))
    seg = "seg-1"
    eng._segment_id = seg
    dur = 5.0
    # buffer grew by more than _SPECULATIVE_REUSE_MAX_DELTA_S
    extra = (_SPECULATIVE_REUSE_MAX_DELTA_S + 1.0)
    audio = np.zeros(int(16000 * (dur + extra)), dtype=np.float32)
    eng._speculative = {
        "segment_id": seg,
        "text": "старый текст",
        "confidence": 0.8,
        "duration": dur,
    }
    eng._last_partial_text = "старый текст"

    eng._finalize(audio, seg)

    assert len(calls) == 1
    assert calls[0][0] == "final"
    assert len(finals) == 1


def test_finalize_redecodes_when_segment_id_differs():
    eng = _engine()
    finals = _capture(eng)
    calls = _monkeypatch_decode(eng)
    eng._segment_id = "seg-current"
    audio = np.zeros(int(16000 * 5.0), dtype=np.float32)
    eng._speculative = {
        "segment_id": "seg-old",
        "text": "старый текст",
        "confidence": 0.8,
        "duration": 5.0,
    }
    eng._last_partial_text = ""

    eng._finalize(audio, "seg-current")

    assert len(calls) == 1


def test_finalize_redecodes_when_speculative_empty():
    eng = _engine()
    calls = _monkeypatch_decode(eng)
    eng._segment_id = "seg-1"
    audio = np.zeros(int(16000 * 5.0), dtype=np.float32)
    eng._speculative = {"segment_id": "seg-1", "text": "", "duration": 5.0}
    eng._last_partial_text = ""

    eng._finalize(audio, "seg-1")

    assert len(calls) == 1
