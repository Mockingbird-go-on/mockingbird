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
