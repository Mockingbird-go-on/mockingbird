"""Tests for the 2026-09-25 latency optimizations (Tier A + B1 + B2).

A1: VAD min_silence_ms default 700→500 (faster end-of-speech detection).
A2: speculative reuse budget 3.0→4.0 s (hold-open 2.0 s + silence tail used
    to cross 3.0 and force a pointless full re-decode).
A5: accumulation window halves (0.2→0.1 s) when the pending final is already
    a confident question or a partial-based answer is streaming.
B1: chunker hold-open budget 2.5→2.0 s.
B2: speculative answers (opt-in GUI toggle): raw-utterance stream starts
    before Tier-3 rescue classifies; "not a question" cancels the stream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "mockingbird"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


# ── A1: min_silence_ms ──────────────────────────────────────────────

def test_vad_min_silence_default_500():
    from mockingbird.config import load_config

    cfg = load_config()
    assert cfg.vad.min_silence_ms == 500
    src = _read("audio/vad.py")
    assert "min_silence_ms: int = 500" in src
    assert "int(0.5 * 16000)" in src


# ── A2: speculative reuse budget ────────────────────────────────────

def test_speculative_reuse_budget_4s():
    from mockingbird.stt import whisper_engine as we

    assert we._SPECULATIVE_REUSE_MAX_DELTA_S == 4.0


def test_hold_open_budget_2s():
    from mockingbird.audio import chunker

    assert chunker._HOLD_OPEN_MAX_S == 2.0


# ── A5: fast accumulation window ────────────────────────────────────

def test_fast_accum_window_constant():
    from mockingbird.kb import interview_engine as ie

    assert ie._ACCUM_WINDOW_S == 0.2
    assert ie._ACCUM_FAST_WINDOW_S == 0.1
    assert ie._ACCUM_FAST_WINDOW_S < ie._ACCUM_WINDOW_S


class _FakeLlm:
    def __init__(self, streaming: bool = False):
        self.is_streaming = streaming


def _engine_with(ie, llm, provisional=""):
    eng = object.__new__(ie.InterviewEngine)
    eng._llm = llm
    eng._provisional_query = provisional
    return eng


def test_fast_flush_on_confident_question():
    from mockingbird.kb import interview_engine as ie

    eng = _engine_with(ie, _FakeLlm(streaming=False), provisional="")
    seg = type("Seg", (), {"text": "Расскажи, что такое Kubernetes?"})()
    assert eng._pending_fast_flush(seg) is True


def test_no_fast_flush_on_streamless_statement():
    from mockingbird.kb import interview_engine as ie

    eng = _engine_with(ie, _FakeLlm(streaming=False), provisional="")
    seg = type("Seg", (), {"text": "Мы использовали docker в прошлом проекте"})()
    assert eng._pending_fast_flush(seg) is False


def test_fast_flush_while_answer_streaming():
    from mockingbird.kb import interview_engine as ie

    eng = _engine_with(ie, _FakeLlm(streaming=True), provisional="")
    seg = type("Seg", (), {"text": "Мы использовали docker в прошлом проекте"})()
    assert eng._pending_fast_flush(seg) is True


def test_fast_flush_with_provisional_query():
    from mockingbird.kb import interview_engine as ie

    eng = _engine_with(ie, _FakeLlm(streaming=False), provisional="что такое iac")
    seg = type("Seg", (), {"text": "Мы использовали docker в прошлом проекте"})()
    assert eng._pending_fast_flush(seg) is True


# ── B2: speculative answers ─────────────────────────────────────────

def test_speculative_answers_default_off():
    from mockingbird.config import load_config

    cfg = load_config()
    assert cfg.interview.speculative_answers is False


def test_speculative_cancel_emits_cancelled_message():
    import threading

    from mockingbird.kb import interview_engine as ie
    from mockingbird.protocol import LlmAnswer

    eng = object.__new__(ie.InterviewEngine)
    eng._spec_cancel = threading.Event()
    eng._spec_query = "длинная реплика без маркеров вопроса"
    emitted: list[LlmAnswer] = []
    eng.on_llm_answer = emitted.append
    ie_threading = threading  # noqa: F841 — clarity
    eng._cancel_speculative("other")
    assert eng._spec_cancel is None
    assert eng._spec_query == ""
    assert len(emitted) == 1
    assert emitted[0].cancelled is True
    assert emitted[0].done is True


def test_cancel_event_stops_llm_stream():
    """LlmClient.answer_question_stream must stop yielding once the cancel
    event is set (B2 teardown path)."""
    import threading
    import types

    from mockingbird.llm.client import LlmClient
    from mockingbird.config import LlmConfig

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    cancel = threading.Event()

    def fake_stream(system, user, gen):
        for i in range(100):
            yield f"tok{i}"

    client._hedged_answer_stream = fake_stream  # type: ignore[method-assign]
    client._ensure = lambda: types.SimpleNamespace()  # type: ignore[method-assign]
    client._enter_stream = lambda: None  # type: ignore[method-assign]
    client._exit_stream = lambda: None  # type: ignore[method-assign]

    got: list[str] = []
    for delta in client.answer_question_stream("q", cancel_event=cancel):
        got.append(delta)
        if len(got) == 3:
            cancel.set()
    assert len(got) == 3  # stopped right after cancellation


# ── A4: harvest after fan-out ───────────────────────────────────────

def test_harvest_runs_after_interview_fanout():
    src = _read("app.py")
    harvest_pos = src.index("self._harvest_answer_terms(msg.text)")
    fanout_pos = src.index("self.interview.on_final(msg)")
    assert harvest_pos > fanout_pos, "harvest must run AFTER interview.on_final"


# ── Architectural pass (#1 clean-snapshot reuse, #2 keepalive, #4 batching) ──

def test_clean_snapshot_flag_exists_and_resets_on_resume():
    src = _read("stt/whisper_engine.py")
    assert "self._spec_dirty = False" in src
    # speech_resume marks the snapshot dirty
    assert 'elif cmd == _CMD_RESUME:' in src
    assert "self._spec_dirty = True" in src


def test_reuse_ignores_delta_when_snapshot_clean():
    """A clean snapshot (only silence appended) must reuse at ANY delta."""
    src = _read("stt/whisper_engine.py")
    assert "(not self._spec_dirty or delta_ok)" in src


def test_stop_hint_deferred_while_decoding():
    src = _read("stt/whisper_engine.py")
    assert "_stop_hint_pending" in src
    assert "_refire_pending_stop_hint" in src


def test_llm_client_keepalive_expiry():
    src = _read("llm/client.py")
    assert "_KEEPALIVE_EXPIRY_S" in src
    assert "http_client=self._http_client()" in src
    # Both primary and failover clients get it.
    assert src.count("http_client=self._http_client()") == 2


def test_llm_client_keepalive_value():
    from mockingbird.llm.client import LlmClient

    assert LlmClient._KEEPALIVE_EXPIRY_S > 60.0


def test_delta_coalescing_in_answer_worker():
    src = _read("kb/interview_engine.py")
    assert "_flush_emit" in src
    assert "len(buffer) == 1" in src  # first delta emitted immediately


def test_clean_snapshot_reuses_at_any_delta():
    """Clean snapshot + 10s of silence tail → still reuse (no re-decode)."""
    import numpy as np

    from tests.test_whisper_finalize_reuse import (
        _capture, _engine, _monkeypatch_decode,
    )

    eng = _engine()
    finals = _capture(eng)
    calls = _monkeypatch_decode(eng)
    seg = "seg-clean"
    eng._segment_id = seg
    eng._speculative = {
        "segment_id": seg,
        "text": "Что такое PVC?",
        "confidence": 0.9,
        "duration": 3.0,
    }
    eng._spec_dirty = False  # clean: only silence appended since
    eng._last_partial_text = "Что такое PVC?"

    audio = np.zeros(int(16000 * 13.0), dtype=np.float32)  # 10s delta!
    eng._finalize(audio, seg)

    assert calls == [], f"clean snapshot must reuse at any delta, got {calls}"
    assert len(finals) == 1
    assert finals[0].text == "Что такое PVC?"


def test_dirty_snapshot_redecodes_beyond_delta():
    import numpy as np

    from tests.test_whisper_finalize_reuse import (
        _capture, _engine, _monkeypatch_decode,
    )

    eng = _engine()
    finals = _capture(eng)
    calls = _monkeypatch_decode(eng, returned=("новый текст", 0.8, 12.0))
    seg = "seg-dirty"
    eng._segment_id = seg
    eng._speculative = {
        "segment_id": seg,
        "text": "старый текст",
        "confidence": 0.8,
        "duration": 3.0,
    }
    eng._spec_dirty = True  # speech resumed — budget applies
    eng._last_partial_text = "старый текст"

    eng._finalize(np.zeros(int(16000 * 13.0), dtype=np.float32), seg)
    assert len(calls) == 1


def test_stop_hint_deferred_until_decode_finishes():
    """Hint arriving mid-decode is re-fired after the decode completes."""
    from tests.test_stt_end_ahead import _collect_finals, _prime_rolling, _stub_transcribe

    from mockingbird.stt import whisper_engine as we

    eng = object.__new__(we.WhisperEngine)
    # minimal attrs used by the path under test
    eng._end_ahead = True
    eng._model = object()
    eng._decoding = True
    eng._stop_hint_pending = False
    eng._spec_dirty = False
    eng._speculative = None

    fired = []
    eng._handle_stop_hint = lambda: fired.append(True)
    # Hint while decoding → deferred, not fired
    we.WhisperEngine._handle_stop_hint(eng)
    assert fired == []
    assert eng._stop_hint_pending is True
    # Refire after the decode finished
    eng._decoding = False
    we.WhisperEngine._refire_pending_stop_hint(eng)
    assert fired == [True]
    assert eng._stop_hint_pending is False


def test_http_client_constructs_with_keepalive():
    """Regression: keepalive_expiry belongs to httpx.Limits, not Client —
    passing it to Client() raised TypeError on every connection check
    (onboarding 'Проверка...' hang, settings connect-test error)."""
    from mockingbird.config import LlmConfig
    from mockingbird.llm.client import LlmClient

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    http = client._http_client()
    assert http is not None
    assert http._transport._pool._keepalive_expiry == LlmClient._KEEPALIVE_EXPIRY_S


# ── Cache replay vs cooldown (2026-09-26 stuck-answer fix) ──────────

class _CacheFake:
    """Minimal answer-cache double."""
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def put(self, key, value):
        self.store[key] = value


def test_cached_answer_replayed_before_cooldown_gate():
    """A cached answer must be re-emitted even inside the cooldown window.

    Wild regression (2026-09-26 log): an early-started (partial-based)
    answer stream finished BEFORE the final transcript reached the panel
    (on_question), so the panel dropped the done-message and never painted
    it. The final-path _maybe_answer_llm call then hit the 7 s cooldown
    gate — which ran BEFORE the cache lookup — and returned early, leaving
    the pane on the watchdog's «Ответ ИИ задерживается» notice forever.
    """
    import time as _time

    from mockingbird.kb import interview_engine as ie

    eng = object.__new__(ie.InterviewEngine)
    eng._llm = _FakeLlm()
    eng._cfg = type("C", (), {
        "llm_primary": True, "answer_cache": True, "answer_cooldown_s": 7.0,
    })()
    eng._llm_answer_available = lambda: True
    eng._answer_cache = _CacheFake()
    eng._context_summary = lambda: ""
    eng._last_answer_ts = _time.monotonic() - 3.0  # inside cooldown
    eng._question_queue = None
    eng._trace = None

    view = type("V", (), {"topic": "general", "title": "A"})()
    query = "Что ты знаешь в Zabbix?"
    key = ie._query_key(query)
    eng._answer_cache.store[key] = "Кэшированный ответ."

    emitted = []
    eng.on_llm_answer = lambda m: emitted.append(m)
    # force=False is the final-transcript path after an equivalent wording
    # (no restart): the cooldown gate would return early without the fix.
    ie.InterviewEngine._maybe_answer_llm(eng, view, query, force=False)
    assert any(
        getattr(m, "done", False) and m.answer == "Кэшированный ответ."
        for m in emitted
    ), "cached answer must replay even while the cooldown gate is active"
