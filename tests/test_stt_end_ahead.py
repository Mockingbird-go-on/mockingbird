"""End-ahead speculative partials + quality-first finalize (no real models)."""
from __future__ import annotations

import numpy as np
import pytest

from mockingbird import protocol
from mockingbird.config import GigaAMConfig, WhisperConfig
from mockingbird.stt.gigaam_engine import GigaAMEngine
from mockingbird.stt.whisper_engine import WhisperEngine


def _whisper_engine():
    engine = WhisperEngine(WhisperConfig(), sample_rate=16000)
    engine._model = object()
    return engine


def _gigaam_engine():
    engine = GigaAMEngine(GigaAMConfig(), sample_rate=16000)
    engine._model = object()
    return engine


@pytest.fixture(
    params=[
        pytest.param(_whisper_engine, id="whisper"),
        pytest.param(_gigaam_engine, id="gigaam"),
    ]
)
def engine(request):
    return request.param()


def _stub_transcribe(engine, text="Вопрос?", confidence=0.9):
    calls = []

    def fake(audio, kind="decode", beam_size=1):
        calls.append((kind, len(audio)))
        return text, confidence, len(audio) / 16000.0

    engine._transcribe = fake
    return calls


def _prime_rolling(engine, seconds=2.0):
    sid = engine.start_segment()
    engine._rolling = np.zeros(int(16000 * seconds), dtype=np.float32)
    return sid


def _collect_finals(engine):
    finals = []
    engine.on_final = finals.append
    return finals


def test_finalize_reuses_speculative_when_buffer_unchanged(engine, request):
    """Speculative-reuse (whisper only): the finalize buffer matches the
    speculative decode (only the VAD silence tail was added) → the
    speculative text is reused without a re-decode (GPU waste guard).
    GigaAM always re-decodes on finalize (no reuse path there)."""
    calls = _stub_transcribe(engine)
    sid = _prime_rolling(engine)
    finals = _collect_finals(engine)

    engine._handle_stop_hint()
    assert len(calls) == 1  # speculative decode of 2s (emitted as partial)

    engine._finalize(np.zeros((16000 * 2), dtype=np.float32), sid)
    if request.node.callspec.id == "whisper":
        # delta = 0s ≤ 3s → no second decode, speculative text reused
        assert len(calls) == 1
    else:
        # gigaam: no speculative-reuse, always re-decodes
        assert len(calls) == 2
    assert len(finals) == 1
    assert finals[0].text == "Вопрос?"
    assert finals[0].segment_id == sid
    assert len(engine._rolling) == 0


def test_finalize_after_audio_growth_redecodes(engine):
    """Audio grew >3s past the speculative duration → full re-decode (the
    speculative text would miss the newly spoken tail)."""
    calls = _stub_transcribe(engine)
    sid = _prime_rolling(engine, seconds=2.0)
    finals = _collect_finals(engine)

    engine._handle_stop_hint()
    assert len(calls) == 1

    # 2s speculative + 4s growth = 4s delta > _SPECULATIVE_REUSE_MAX_DELTA_S
    engine._finalize(np.zeros((16000 * 6), dtype=np.float32), sid)
    assert len(calls) == 2
    assert calls[1][1] == (16000 * 6)
    assert len(finals) == 1
    assert finals[0].segment_id == sid


def test_final_shorter_than_partial_uses_partial(engine):
    """Safety-net: the full decode dropped a word the last partial had
    (GigaAM losing «Agile» on a different audio cut) → partial wins.

    Forces a re-decode (audio grew >3s) so reconcile_final_with_partial runs
    on the fresh final text."""
    texts = iter(["в чем связь между Agile и", "в чем связь между и"])

    def fake(audio, kind="decode", beam_size=1):
        return next(texts), 0.9, len(audio) / 16000.0

    engine._transcribe = fake
    sid = _prime_rolling(engine)
    finals = _collect_finals(engine)

    engine._handle_stop_hint()  # partial: full text with Agile
    engine._finalize(np.zeros((16000 * 6), dtype=np.float32), sid)
    assert len(finals) == 1
    assert "Agile" in finals[0].text


def test_final_ok_keeps_final_not_partial(engine):
    """Final decode at least as complete as the partial → keep the final.
    Forces a re-decode (audio grew >3s)."""
    texts = iter(["в чем связь", "в чем связь между Agile и DevOps"])

    def fake(audio, kind="decode", beam_size=1):
        return next(texts), 0.9, len(audio) / 16000.0

    engine._transcribe = fake
    sid = _prime_rolling(engine)
    finals = _collect_finals(engine)

    engine._handle_stop_hint()
    engine._finalize(np.zeros((16000 * 6), dtype=np.float32), sid)
    assert finals[0].text == "в чем связь между Agile и DevOps"


def test_finalize_different_text_partial_not_applied(engine):
    """Different utterance (low token similarity) → the partial is NOT merged.
    Forces a re-decode (audio grew >3s)."""
    texts = iter(["совсем другой вопрос про сети", "в чем связь между и"])

    def fake(audio, kind="decode", beam_size=1):
        return next(texts), 0.9, len(audio) / 16000.0

    engine._transcribe = fake
    sid = _prime_rolling(engine)
    finals = _collect_finals(engine)

    engine._handle_stop_hint()
    engine._finalize(np.zeros((16000 * 6), dtype=np.float32), sid)
    assert finals[0].text == "в чем связь между и"


def test_disabled_falls_back_to_normal_final(engine):
    engine._end_ahead = False
    calls = _stub_transcribe(engine)
    sid = _prime_rolling(engine)
    finals = _collect_finals(engine)

    engine._handle_stop_hint()
    assert len(calls) == 0  # no speculative decode

    engine._finalize(np.zeros((16000 * 2), dtype=np.float32), sid)
    assert len(calls) == 1
    assert len(finals) == 1


def test_resume_discards_speculative(engine):
    calls = _stub_transcribe(engine)
    sid = _prime_rolling(engine)
    finals = _collect_finals(engine)

    engine._handle_stop_hint()
    assert len(calls) == 1
    engine._speculative = None  # what the worker does on _CMD_RESUME

    engine._finalize(np.zeros((16000 * 2), dtype=np.float32), sid)
    assert len(calls) == 2  # re-decode on final
    assert len(finals) == 1


def test_speculative_only_for_matching_segment(engine):
    calls = _stub_transcribe(engine)
    _prime_rolling(engine)
    finals = _collect_finals(engine)

    engine._handle_stop_hint()
    assert len(calls) == 1

    engine._finalize(np.zeros((16000 * 2), dtype=np.float32), "other-segment")
    assert len(calls) == 2
    assert len(finals) == 1
    # partial from a different segment must not leak into this final
    assert finals[0].text == "Вопрос?"


def test_short_audio_skips_speculative(engine):
    calls = _stub_transcribe(engine)
    sid = engine.start_segment()
    engine._rolling = np.zeros(int(16000 * 0.2), dtype=np.float32)

    engine._handle_stop_hint()
    assert len(calls) == 0
    assert engine._speculative is None

    engine._finalize(np.zeros(int(16000 * 0.3), dtype=np.float32), sid)
    assert len(calls) == 1


def test_speech_stop_enqueues_hint(engine):
    engine.on_speech_stop()
    engine.on_speech_resume()
    cmds = [c for c, _ in list(engine._queue.queue)]
    assert cmds == ["stop_hint", "resume"]


def test_whisper_speculative_uses_final_beam():
    engine = _whisper_engine()
    engine._model = object()
    seen = {}

    def fake(audio, kind="decode", beam_size=1):
        seen[kind] = beam_size
        return "ok", 0.9, 2.0

    engine._transcribe = fake
    _prime_rolling(engine)
    engine._handle_stop_hint()
    assert seen["speculative"] == engine._cfg.final_beam_size


def test_stop_hint_emits_full_text_partial(engine):
    calls = _stub_transcribe(engine)
    sid = _prime_rolling(engine)
    partials = []
    engine.on_partial = partials.append

    engine._handle_stop_hint()
    assert len(calls) == 1
    assert len(partials) == 1
    msg = partials[0]
    assert msg.type == protocol.MessageType.PARTIAL_TRANSCRIPT
    assert msg.text == "Вопрос?"
    assert msg.segment_id == sid
    assert msg.end == pytest.approx(2.0)
    assert engine._spec_partial_emitted is True


def test_no_tail_partial_after_speculative(engine):
    calls = _stub_transcribe(engine)
    _prime_rolling(engine)
    partials = []
    engine.on_partial = partials.append

    engine._handle_stop_hint()
    assert len(partials) == 1
    engine._last_decode = 0.0
    engine._maybe_decode()
    # the speculative partial already carries the full text: no tail decode
    assert len(calls) == 1
    assert len(partials) == 1


def test_disabled_emits_no_speculative_partial(engine):
    engine._end_ahead = False
    calls = _stub_transcribe(engine)
    _prime_rolling(engine)
    partials = []
    engine.on_partial = partials.append

    engine._handle_stop_hint()
    assert len(calls) == 0
    assert partials == []
    assert engine._spec_partial_emitted is False


def test_resume_reenables_partials(engine):
    calls = _stub_transcribe(engine)
    _prime_rolling(engine)
    engine._handle_stop_hint()
    assert engine._spec_partial_emitted is True
    # what the worker does on _CMD_RESUME: drop the hint result, re-enable decode
    engine._speculative = None
    engine._spec_partial_emitted = False
    engine._last_decode = 0.0
    engine._maybe_decode()
    assert len(calls) == 2
    assert engine._spec_partial_emitted is False


def test_start_segment_clears_partial_emitted(engine):
    _stub_transcribe(engine)
    _prime_rolling(engine)
    engine._handle_stop_hint()
    assert engine._spec_partial_emitted is True
    engine.start_segment()
    assert engine._spec_partial_emitted is False
    assert engine._last_partial_text == ""
